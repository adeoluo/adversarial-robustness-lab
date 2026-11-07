"""HTTP service: classify, attack and certify CIFAR-10 images with a trained checkpoint.

A standard-library threaded server, like the other services in this portfolio.

| Route | Method | Body |
|---|---|---|
| ``/health`` | GET | liveness and version |
| ``/ready`` | GET | model loaded, architecture, parameters |
| ``/metrics`` | GET | Prometheus text format |
| ``/v1/predict`` | POST | ``{"index": i}`` or ``{"pixels": 3x32x32}`` |
| ``/v1/attack`` | POST | image plus ``label``, ``attack``, ``eps``, ``steps`` |
| ``/v1/certify`` | POST | image plus ``sigma``, ``n0``, ``n``, ``alpha`` |

``pixels`` is a nested list of floats in ``[0, 1]`` or integers in ``[0, 255]``; ``index``
selects a test image. Work per request is bounded (steps, queries, samples, body size).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch

from adversarial_robustness_lab import __version__
from adversarial_robustness_lab.attacks import ATTACK_NAMES, attack_norm, make_attack
from adversarial_robustness_lab.data import CLASSES, ImageSplit, load_cifar10, synthetic_split
from adversarial_robustness_lab.evaluation import check_threat_model
from adversarial_robustness_lab.models import count_parameters
from adversarial_robustness_lab.runtime import load_checkpoint, pick_device
from adversarial_robustness_lab.smoothing import ABSTAIN, Smooth

MAX_BODY = 1 << 20
MAX_STEPS = 200
MAX_QUERIES = 5000
MAX_SAMPLES = 100_000
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class BadRequestError(ValueError):
    """A client error, reported with status 400."""


@dataclass
class Metrics:
    """Counters and a latency histogram in Prometheus text format."""

    requests: dict[tuple[str, int], int] = field(default_factory=lambda: defaultdict(int))
    attacks: dict[tuple[str, bool], int] = field(default_factory=lambda: defaultdict(int))
    certificates: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    buckets: list[int] = field(default_factory=lambda: [0] * (len(BUCKETS) + 1))
    latency_sum: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, route: str, status: int, seconds: float) -> None:
        with self.lock:
            self.requests[(route, status)] += 1
            self.latency_sum += seconds
            for i, bound in enumerate(BUCKETS):
                if seconds <= bound:
                    self.buckets[i] += 1
            self.buckets[-1] += 1

    def render(self) -> str:
        with self.lock:
            lines = [
                "# HELP arl_requests_total HTTP requests by route and status.",
                "# TYPE arl_requests_total counter",
            ]
            for (route, status), n in sorted(self.requests.items()):
                lines.append(f'arl_requests_total{{route="{route}",status="{status}"}} {n}')
            lines += [
                "# HELP arl_attacks_total Attacks run, by attack and outcome.",
                "# TYPE arl_attacks_total counter",
            ]
            for (name, fooled), n in sorted(self.attacks.items()):
                lines.append(
                    f'arl_attacks_total{{attack="{name}",fooled="{str(fooled).lower()}"}} {n}'
                )
            lines += [
                "# HELP arl_certificates_total Smoothing certificates, by outcome.",
                "# TYPE arl_certificates_total counter",
            ]
            for outcome, n in sorted(self.certificates.items()):
                lines.append(f'arl_certificates_total{{outcome="{outcome}"}} {n}')
            lines += [
                "# HELP arl_request_seconds Request latency.",
                "# TYPE arl_request_seconds histogram",
            ]
            for bound, n in zip(BUCKETS, self.buckets, strict=False):
                lines.append(f'arl_request_seconds_bucket{{le="{bound}"}} {n}')
            lines.append(f'arl_request_seconds_bucket{{le="+Inf"}} {self.buckets[-1]}')
            lines.append(f"arl_request_seconds_sum {self.latency_sum:.6f}")
            lines.append(f"arl_request_seconds_count {self.buckets[-1]}")
            return "\n".join(lines) + "\n"


class App:
    """Request handling, independent of the HTTP transport (and unit tested directly)."""

    def __init__(
        self,
        model: torch.nn.Module,
        meta: dict[str, Any],
        split: ImageSplit | None,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.meta = meta
        self.split = split
        self.device = device
        self.metrics = Metrics()
        self.lock = threading.Lock()

    # Inputs -------------------------------------------------------------------------
    def image(self, body: dict[str, Any]) -> tuple[torch.Tensor, int | None]:
        if "index" in body:
            if self.split is None:
                raise BadRequestError("no test set loaded; send pixels")
            idx = body["index"]
            if not isinstance(idx, int) or not 0 <= idx < len(self.split):
                raise BadRequestError(f"index must be an integer in [0, {len(self.split)})")
            x, y = self.split.subset([idx]).tensors(self.device)
            return x, int(y[0])
        if "pixels" not in body:
            raise BadRequestError("send index or pixels")
        try:
            t = torch.tensor(body["pixels"], dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise BadRequestError("pixels must be a numeric 3x32x32 nested list") from exc
        if tuple(t.shape) != (3, 32, 32):
            raise BadRequestError(f"pixels must have shape 3x32x32, got {tuple(t.shape)}")
        if not bool(torch.isfinite(t).all()):
            raise BadRequestError("pixels must be finite")
        if float(t.max()) > 1.0:
            t = t / 255.0
        if float(t.min()) < 0.0 or float(t.max()) > 1.0:
            raise BadRequestError("pixels must be in [0, 1] or [0, 255]")
        return t.unsqueeze(0).to(self.device), None

    @staticmethod
    def _number(body: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
        value = body.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise BadRequestError(f"{key} must be a number")
        if not lo <= float(value) <= hi:
            raise BadRequestError(f"{key} must be in [{lo}, {hi}]")
        return float(value)

    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(self.model(x), 1)[0]

    # Routes -------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    def ready(self) -> dict[str, Any]:
        return {
            "ready": True,
            "arch": self.meta.get("arch"),
            "method": self.meta.get("config", {}).get("method"),
            "parameters": count_parameters(self.model),
            "test_images": 0 if self.split is None else len(self.split),
            "device": str(self.device),
        }

    def predict(self, body: dict[str, Any]) -> dict[str, Any]:
        x, label = self.image(body)
        with self.lock:
            p = self._probs(x)
        top = int(p.argmax())
        return {
            "prediction": top,
            "class": CLASSES[top],
            "confidence": float(p[top]),
            "probabilities": {c: round(float(v), 6) for c, v in zip(CLASSES, p, strict=True)},
            "label": label,
        }

    def attack(self, body: dict[str, Any]) -> dict[str, Any]:
        x, label = self.image(body)
        if label is None:
            if not isinstance(body.get("label"), int) or not 0 <= body["label"] < len(CLASSES):
                raise BadRequestError("label (0 to 9) is required with pixels")
            label = int(body["label"])
        name = body.get("attack", "pgd")
        if name not in ATTACK_NAMES:
            raise BadRequestError(f"attack must be one of {list(ATTACK_NAMES)}")
        eps = self._number(body, "eps", 8 / 255, 0.0, 1.0 if attack_norm(name) == "linf" else 10.0)
        steps = int(self._number(body, "steps", 20, 1, MAX_STEPS))
        queries = int(self._number(body, "queries", 1000, 1, MAX_QUERIES))
        attack = make_attack(name, eps, steps, 1, queries, int(body.get("seed", 0)))
        y = torch.tensor([label], device=self.device)
        with self.lock:
            x_adv = attack(self.model, x, y)
            check_threat_model(x, x_adv, eps, attack_norm(name))  # type: ignore[arg-type]
            p_clean, p_adv = self._probs(x), self._probs(x_adv)
        fooled = int(p_adv.argmax()) != label
        with self.metrics.lock:
            self.metrics.attacks[(name, fooled)] += 1
        delta = (x_adv - x).flatten()
        return {
            "attack": name,
            "eps": eps,
            "label": label,
            "clean": {"class": CLASSES[int(p_clean.argmax())], "confidence": float(p_clean.max())},
            "adversarial": {
                "class": CLASSES[int(p_adv.argmax())],
                "confidence": float(p_adv.max()),
            },
            "fooled": fooled,
            "linf": float(delta.abs().max()),
            "l2": float(delta.norm()),
        }

    def certify(self, body: dict[str, Any]) -> dict[str, Any]:
        x, label = self.image(body)
        default_sigma = float(self.meta.get("config", {}).get("sigma", 0.25))
        sigma = self._number(body, "sigma", default_sigma, 1e-3, 2.0)
        n0 = int(self._number(body, "n0", 100, 1, 10_000))
        n = int(self._number(body, "n", 1000, 1, MAX_SAMPLES))
        alpha = self._number(body, "alpha", 0.001, 1e-6, 0.5)
        smooth = Smooth(self.model, len(CLASSES), sigma, 500, int(body.get("seed", 0)))
        with self.lock:
            cert = smooth.certify(x[0], n0, n, alpha)
        outcome = "abstain" if cert.prediction == ABSTAIN else "certified"
        with self.metrics.lock:
            self.metrics.certificates[outcome] += 1
        return {
            "prediction": None if cert.prediction == ABSTAIN else CLASSES[cert.prediction],
            "abstain": cert.prediction == ABSTAIN,
            "radius": cert.radius,
            "p_lower": cert.p_lower,
            "sigma": sigma,
            "n0": n0,
            "n": n,
            "alpha": alpha,
            "label": None if label is None else CLASSES[label],
        }

    def handle(self, method: str, path: str, raw: bytes) -> tuple[int, str, bytes]:
        """Route a request; returns ``(status, content type, body)``."""
        get_routes = {"/health": self.health, "/ready": self.ready}
        post_routes = {
            "/v1/predict": self.predict,
            "/v1/attack": self.attack,
            "/v1/certify": self.certify,
        }
        try:
            if method == "GET" and path == "/metrics":
                return 200, "text/plain; version=0.0.4", self.metrics.render().encode()
            if method == "GET" and path in get_routes:
                return 200, "application/json", json.dumps(get_routes[path]()).encode()
            if path in post_routes:
                if method != "POST":
                    return 405, "application/json", b'{"error": "use POST"}'
                if len(raw) > MAX_BODY:
                    return 413, "application/json", b'{"error": "body too large"}'
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError as exc:
                    raise BadRequestError(f"invalid JSON: {exc.msg}") from exc
                if not isinstance(body, dict):
                    raise BadRequestError("body must be a JSON object")
                return 200, "application/json", json.dumps(post_routes[path](body)).encode()
            return 404, "application/json", b'{"error": "not found"}'
        except BadRequestError as exc:
            return 400, "application/json", json.dumps({"error": str(exc)}).encode()


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"adversarial-robustness-lab/{__version__}"

        def _respond(self, method: str) -> None:
            t0 = time.perf_counter()
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                status, ctype, body = 413, "application/json", b'{"error": "body too large"}'
            else:
                raw = self.rfile.read(length) if length else b""
                status, ctype, body = app.handle(method, self.path.split("?")[0], raw)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            app.metrics.observe(self.path.split("?")[0], status, time.perf_counter() - t0)

        def do_GET(self) -> None:
            self._respond("GET")

        def do_POST(self) -> None:
            self._respond("POST")

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def build_app(args: argparse.Namespace) -> App:
    device = pick_device(args.device)
    model, meta = load_checkpoint(args.checkpoint, device)
    split: ImageSplit | None
    if args.synthetic:
        split = synthetic_split(args.synthetic, 1)
    else:
        try:
            split = load_cifar10(args.data_dir).test if args.data_dir else load_cifar10().test
        except FileNotFoundError:
            split = None
    return App(model, meta, split, device)


def serve(args: argparse.Namespace) -> None:  # pragma: no cover - blocking loop
    app = build_app(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"serving {app.meta.get('arch')} on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
