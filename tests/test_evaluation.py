from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from adversarial_robustness_lab.attacks import PGD
from adversarial_robustness_lab.evaluation import (
    ThreatModelError,
    check_threat_model,
    clean_accuracy,
    evaluate_attack,
)


def test_check_threat_model_accepts_valid() -> None:
    x = torch.rand(4, 3, 8, 8) * 0.5 + 0.25
    check_threat_model(x, x + 0.03, 0.03, "linf")


@pytest.mark.parametrize(
    ("delta", "eps", "norm", "match"),
    [
        (0.05, 0.03, "linf", "exceeds"),
        (0.01, 0.1, "l2", "exceeds"),
        (float("nan"), 1.0, "linf", "non-finite"),
    ],
)
def test_check_threat_model_rejects(delta: float, eps: float, norm: str, match: str) -> None:
    x = torch.full((2, 3, 8, 8), 0.5)
    with pytest.raises(ThreatModelError, match=match):
        check_threat_model(x, x + delta, eps, norm)  # type: ignore[arg-type]


def test_check_threat_model_box_and_shape() -> None:
    x = torch.full((1, 3, 4, 4), 0.99)
    with pytest.raises(ThreatModelError, match="pixel box"):
        check_threat_model(x, x + 0.02, 0.05, "linf")
    with pytest.raises(ThreatModelError, match="shape"):
        check_threat_model(x, x[:, :2], 0.05, "linf")


class Cheater:
    name = "cheater"
    queries = 0

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def test_evaluate_attack_refuses_cheating(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    with pytest.raises(ThreatModelError):
        evaluate_attack(trained_cnn, Cheater(), x, y, 8 / 255)
    res = evaluate_attack(trained_cnn, Cheater(), x, y, 8 / 255, verify=False)
    assert res.robust_accuracy <= res.clean_accuracy


def test_evaluate_attack_results(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    messages: list[str] = []
    res = evaluate_attack(
        trained_cnn, PGD(16 / 255, steps=5), x, y, 16 / 255, batch_size=20, progress=messages.append
    )
    assert res.n == 64 and len(messages) == 4
    assert res.robust_accuracy <= res.clean_accuracy
    assert np.all(res.robust_correct <= res.clean_correct)
    assert res.max_distance <= 16 / 255 + 1e-6
    assert res.queries > 0
    lo, hi = res.robust_ci()
    assert lo <= res.robust_accuracy <= hi
    summary = res.summary()
    assert summary["attack"] == "pgd" and summary["n"] == 64


def test_evaluate_attack_on_uint8_and_misclassified(
    trained_cnn: nn.Module, test_split: object
) -> None:
    from adversarial_robustness_lab.data import ImageSplit

    assert isinstance(test_split, ImageSplit)
    xu, y = test_split.tensors(as_uint8=True)
    wrong = (y + 1) % 10
    res = evaluate_attack(
        trained_cnn, PGD(8 / 255, steps=2), xu, wrong, 8 / 255, attack_misclassified=True
    )
    assert res.clean_accuracy < 0.2 and res.robust_accuracy <= res.clean_accuracy
    per = clean_accuracy(trained_cnn, xu, y)
    assert per.dtype == bool and per.mean() > 0.9


def test_empty_results() -> None:
    from adversarial_robustness_lab.evaluation import AttackResult

    r = AttackResult("x", 0.1, "linf", np.zeros(0, bool), np.zeros(0, bool), 0.0, 0, 0.0)
    assert r.clean_accuracy == 0.0 and r.robust_accuracy == 0.0
