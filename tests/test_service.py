from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch
from torch import nn

from adversarial_robustness_lab.data import ImageSplit
from adversarial_robustness_lab.models import SmallCNN
from adversarial_robustness_lab.runtime import save_checkpoint
from adversarial_robustness_lab.service import App, build_app, make_handler


@pytest.fixture()
def app(trained_cnn: nn.Module, test_split: ImageSplit) -> App:
    return App(
        trained_cnn,
        {"arch": "small_cnn", "config": {"method": "standard", "sigma": 0.25}},
        test_split,
        torch.device("cpu"),
    )


def call(app: App, method: str, path: str, body: object = None) -> tuple[int, dict[str, object]]:
    raw = b"" if body is None else json.dumps(body).encode()
    status, _, out = app.handle(method, path, raw)
    return status, json.loads(out)


def test_health_ready(app: App) -> None:
    assert call(app, "GET", "/health")[1]["status"] == "ok"
    status, ready = call(app, "GET", "/ready")
    assert status == 200 and ready["test_images"] == 64 and ready["method"] == "standard"


def test_predict_by_index_and_pixels(app: App, test_split: ImageSplit) -> None:
    status, out = call(app, "POST", "/v1/predict", {"index": 0})
    assert status == 200 and out["label"] == int(test_split.labels[0])
    assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)  # type: ignore[union-attr]
    pixels = test_split.images[0].tolist()
    status, out2 = call(app, "POST", "/v1/predict", {"pixels": pixels})
    assert status == 200 and out2["prediction"] == out["prediction"]
    status, out3 = call(
        app, "POST", "/v1/predict", {"pixels": (test_split.images[0] / 255.0).tolist()}
    )
    assert out3["prediction"] == out["prediction"]


def test_attack_route(app: App) -> None:
    status, out = call(
        app, "POST", "/v1/attack", {"index": 1, "attack": "pgd", "eps": 0.1, "steps": 5}
    )
    assert status == 200 and out["linf"] <= 0.1 + 1e-6  # type: ignore[operator]
    status, out = call(
        app, "POST", "/v1/attack", {"pixels": [[[0.5] * 32] * 32] * 3, "label": 2, "attack": "fgsm"}
    )
    assert status == 200 and "fooled" in out


def test_certify_route(app: App) -> None:
    status, out = call(app, "POST", "/v1/certify", {"index": 2, "n0": 10, "n": 200})
    assert status == 200 and out["sigma"] == 0.25 and out["radius"] >= 0  # type: ignore[operator]


@pytest.mark.parametrize(
    ("path", "body", "match"),
    [
        ("/v1/predict", {}, "index or pixels"),
        ("/v1/predict", {"index": 10_000}, "index"),
        ("/v1/predict", {"index": "3"}, "index"),
        ("/v1/predict", {"pixels": [[1, 2]]}, "shape"),
        ("/v1/predict", {"pixels": "abc"}, "numeric"),
        ("/v1/predict", {"pixels": [[[-5.0] * 32] * 32] * 3}, "in \\[0, 1\\]"),
        ("/v1/predict", {"pixels": [[[float("nan")] * 32] * 32] * 3}, "finite"),
        ("/v1/attack", {"index": 0, "attack": "deepfool"}, "attack must be"),
        ("/v1/attack", {"index": 0, "eps": 5}, "eps"),
        ("/v1/attack", {"index": 0, "steps": 10_000}, "steps"),
        ("/v1/attack", {"index": 0, "eps": "big"}, "number"),
        ("/v1/attack", {"pixels": [[[0.5] * 32] * 32] * 3}, "label"),
        ("/v1/certify", {"index": 0, "n": 10**9}, "n must"),
    ],
)
def test_bad_requests(app: App, path: str, body: object, match: str) -> None:
    status, out = call(app, "POST", path, body)
    assert status == 400
    import re

    assert re.search(match, str(out["error"]))


def test_protocol_errors(app: App) -> None:
    assert app.handle("POST", "/v1/predict", b"{bad")[0] == 400
    assert app.handle("POST", "/v1/predict", b"[1]")[0] == 400
    assert app.handle("GET", "/v1/predict", b"")[0] == 405
    assert app.handle("GET", "/nope", b"")[0] == 404
    assert app.handle("POST", "/v1/predict", b" " * (2 << 20))[0] == 413


def test_no_test_set(trained_cnn: nn.Module) -> None:
    bare = App(trained_cnn, {}, None, torch.device("cpu"))
    status, out = call(bare, "POST", "/v1/predict", {"index": 0})
    assert status == 400 and "no test set" in str(out["error"])


def test_metrics_render(app: App) -> None:
    call(app, "POST", "/v1/attack", {"index": 0, "steps": 2})
    call(app, "POST", "/v1/certify", {"index": 0, "n0": 5, "n": 20})
    app.metrics.observe("/v1/attack", 200, 0.02)
    status, ctype, body = app.handle("GET", "/metrics", b"")
    text = body.decode()
    assert status == 200 and ctype.startswith("text/plain")
    assert 'arl_requests_total{route="/v1/attack",status="200"} 1' in text
    assert "arl_attacks_total" in text and "arl_certificates_total" in text
    assert 'arl_request_seconds_bucket{le="+Inf"} 1' in text


def test_real_socket_roundtrip(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    save_checkpoint(ckpt, SmallCNN(), "small_cnn", {"config": {"method": "standard"}})
    args = argparse.Namespace(checkpoint=ckpt, device="cpu", synthetic=8, data_dir=None)
    app = build_app(args)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/ready") as r:
            assert json.loads(r.read())["test_images"] == 8
        req = urllib.request.Request(f"{base}/v1/predict", data=b'{"index": 1}', method="POST")
        with urllib.request.urlopen(req) as r:
            assert "class" in json.loads(r.read())
        bad = urllib.request.Request(f"{base}/v1/predict", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad)
        assert exc.value.code == 400
        with urllib.request.urlopen(f"{base}/metrics") as r:
            assert b'route="/ready"' in r.read()
    finally:
        server.shutdown()
        server.server_close()


def test_build_app_without_data(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    save_checkpoint(ckpt, SmallCNN(), "small_cnn", {})
    app = build_app(
        argparse.Namespace(checkpoint=ckpt, device="cpu", synthetic=0, data_dir=tmp_path)
    )
    assert app.split is None
