from __future__ import annotations

import pytest
import torch
from torch import nn

from adversarial_robustness_lab.attacks import (
    APGD,
    PGD,
    APGDTargeted,
    AutoAttackEnsemble,
    SquareAttack,
    build_attacks,
    predict,
)


def test_versions() -> None:
    std = build_attacks(8 / 255)
    assert [type(a) for a in std] == [APGD, APGDTargeted, SquareAttack]
    assert [a.name for a in build_attacks(8 / 255, version="fast")] == ["apgd-ce", "apgd-dlr"]
    assert all(isinstance(a, APGD) and a.restarts == 5 for a in build_attacks(0.1, version="rand"))
    assert len(build_attacks(0.5, "l2")) == 2
    with pytest.raises(ValueError):
        build_attacks(0.1, version="plus")
    with pytest.raises(ValueError):
        AutoAttackEnsemble([])


def test_ensemble_is_worst_case(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    eps = 12 / 255
    ens = AutoAttackEnsemble.from_version(
        eps, steps=10, n_target_classes=2, square_queries=100, batch_size=25
    )
    logs: list[str] = []
    res = ens.run(trained_cnn, x, y, progress=logs.append)
    assert len(logs) == 3
    counts = list(res.robust_after.values())
    assert counts == sorted(counts, reverse=True)
    assert res.robust_accuracy <= res.clean_accuracy
    pgd_acc = float(
        (predict(trained_cnn, PGD(eps, steps=10)(trained_cnn, x, y)) == y).float().mean()
    )
    assert res.robust_accuracy <= pgd_acc + 0.05
    # Every non-robust, clean-correct sample is misclassified at its stored example.
    broken = res.clean_correct & ~res.robust
    assert bool((predict(trained_cnn, res.x_adv[broken]) != y[broken]).all())
    assert float((res.x_adv - x).abs().max()) <= eps + 1e-6
    assert set(res.broken_by) <= {"", "clean", "apgd-ce", "apgd-t", "square"}
    assert set(res.seconds) == {"apgd-ce", "apgd-t", "square"}
