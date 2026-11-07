"""Robust accuracy evaluation with threat-model verification.

Every adversarial batch is checked before it is scored: each example must lie in the
pixel box and within ``eps`` of its source in the attack's norm (plus a float tolerance
of 1e-5). An attack that leaves the threat set raises ``ThreatModelError`` instead
of inflating the attack success rate.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from adversarial_robustness_lab.attacks.base import Attack
from adversarial_robustness_lab.attacks.geometry import Norm, flat_norm
from adversarial_robustness_lab.data import iterate_batches, to_float
from adversarial_robustness_lab.runtime import synchronize
from adversarial_robustness_lab.stats import wilson_interval

TOLERANCE = 1e-5


class ThreatModelError(AssertionError):
    """An attack returned an input outside its threat set."""


def check_threat_model(
    x: torch.Tensor, x_adv: torch.Tensor, eps: float, norm: Norm, tol: float = TOLERANCE
) -> None:
    """Raise unless every ``x_adv`` is in ``[0, 1]`` and within ``eps`` of ``x``."""
    x_adv = x_adv.detach()
    if x_adv.shape != x.shape:
        raise ThreatModelError(f"shape changed from {tuple(x.shape)} to {tuple(x_adv.shape)}")
    if not bool(torch.isfinite(x_adv).all()):
        raise ThreatModelError("non-finite values in adversarial examples")
    if float(x_adv.min()) < -tol or float(x_adv.max()) > 1 + tol:
        raise ThreatModelError("adversarial examples leave the pixel box [0, 1]")
    dist = float(flat_norm(x_adv - x, norm).max()) if x.shape[0] else 0.0
    if dist > eps + tol:
        raise ThreatModelError(f"{norm} distance {dist:.6g} exceeds eps {eps:.6g}")


@dataclass
class AttackResult:
    """Per-sample results of one attack over a dataset."""

    name: str
    eps: float
    norm: str
    clean_correct: np.ndarray
    robust_correct: np.ndarray
    seconds: float
    queries: int
    max_distance: float
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.clean_correct.size)

    @property
    def clean_accuracy(self) -> float:
        return float(self.clean_correct.mean()) if self.n else 0.0

    @property
    def robust_accuracy(self) -> float:
        return float(self.robust_correct.mean()) if self.n else 0.0

    def robust_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        return wilson_interval(int(self.robust_correct.sum()), self.n, confidence)

    def summary(self) -> dict[str, float | int | str]:
        lo, hi = self.robust_ci()
        return {
            "attack": self.name,
            "eps": self.eps,
            "norm": self.norm,
            "n": self.n,
            "clean_accuracy": self.clean_accuracy,
            "robust_accuracy": self.robust_accuracy,
            "robust_ci95_low": lo,
            "robust_ci95_high": hi,
            "seconds": self.seconds,
            "queries": self.queries,
            "max_distance": self.max_distance,
            **self.extra,
        }


def evaluate_attack(
    model: nn.Module,
    attack: Attack | Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    norm: Norm = "linf",
    batch_size: int = 250,
    name: str | None = None,
    attack_misclassified: bool = False,
    verify: bool = True,
    progress: Callable[[str], None] | None = None,
) -> AttackResult:
    """Run ``attack`` over ``(x, y)`` in batches and score the adversarial examples.

    ``x`` may be ``uint8`` or float; batches are converted on the fly. Points the model
    already misclassifies count as non-robust and, unless ``attack_misclassified``, are
    not attacked (they cannot become robust).
    """
    model.eval()
    device = next(model.parameters()).device
    clean_all: list[np.ndarray] = []
    robust_all: list[np.ndarray] = []
    max_dist = 0.0
    queries_before = getattr(attack, "queries", 0)
    synchronize(device)
    t0 = time.perf_counter()
    done = 0
    for xu, yb in iterate_batches(x, y, batch_size):
        xb = to_float(xu).to(device)
        yb = yb.to(device)
        with torch.no_grad():
            clean = model(xb).argmax(1) == yb
        robust = clean.clone()
        idx = (
            torch.arange(xb.shape[0], device=device)
            if attack_misclassified
            else clean.nonzero().flatten()
        )
        if idx.numel() > 0:
            x_adv = attack(model, xb[idx], yb[idx])
            if verify:
                check_threat_model(xb[idx], x_adv, eps, norm)
            with torch.no_grad():
                still = model(x_adv).argmax(1) == yb[idx]
            robust[idx] = clean[idx] & still
            max_dist = max(max_dist, float(flat_norm(x_adv - xb[idx], norm).max()))
        clean_all.append(clean.cpu().numpy())
        robust_all.append(robust.cpu().numpy())
        done += xb.shape[0]
        if progress is not None:
            acc = float(np.concatenate(robust_all).mean())
            progress(f"{done}/{x.shape[0]} robust so far {acc:.4f}")
    synchronize(device)
    seconds = time.perf_counter() - t0
    return AttackResult(
        name=str(name or getattr(attack, "name", type(attack).__name__)),
        eps=eps,
        norm=norm,
        clean_correct=np.concatenate(clean_all) if clean_all else np.zeros(0, bool),
        robust_correct=np.concatenate(robust_all) if robust_all else np.zeros(0, bool),
        seconds=seconds,
        queries=int(getattr(attack, "queries", 0) - queries_before),
        max_distance=max_dist,
    )


def clean_accuracy(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 250
) -> np.ndarray:
    """Per-sample correctness on clean inputs."""
    model.eval()
    device = next(model.parameters()).device
    out: list[np.ndarray] = []
    with torch.no_grad():
        for xu, yb in iterate_batches(x, y, batch_size):
            pred = model(to_float(xu).to(device)).argmax(1)
            out.append((pred == yb.to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, bool)
