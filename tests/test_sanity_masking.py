from __future__ import annotations

import pytest
import torch
from torch import nn

from adversarial_robustness_lab.attacks import PGD, SquareAttack, predict
from adversarial_robustness_lab.masking import InputQuantization, bpda_view, quantize
from adversarial_robustness_lab.sanity import (
    RandomNoise,
    TransferAttack,
    run_sanity_checks,
    zero_gradient_fraction,
)


def test_quantize_levels_and_gradients() -> None:
    x = torch.linspace(0, 1, 101).requires_grad_(True)
    q = quantize(x, 2)
    assert torch.unique(q).numel() == 4
    torch.testing.assert_close(q * 3, torch.round(q * 3))
    q.sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) == 0.0
    x2 = torch.linspace(0, 1, 11).requires_grad_(True)
    quantize(x2, 2, straight_through=True).sum().backward()
    assert x2.grad is not None
    torch.testing.assert_close(x2.grad, torch.ones(11))


def test_input_quantization_validation() -> None:
    with pytest.raises(ValueError):
        InputQuantization(nn.Identity(), bits=0)


def test_sanity_checks_pass_on_honest_model(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    log: list[str] = []
    report = run_sanity_checks(
        trained_cnn,
        x,
        y,
        eps=16 / 255,
        steps=5,
        square_queries=100,
        surrogate=trained_cnn,
        tol=0.05,
        progress=log.append,
    )
    names = {c.name for c in report.checks}
    assert "transfer_weaker_than_white_box" in names and len(report.checks) == 8
    failed = [c.name for c in report.checks if not c.passed]
    assert failed == [], failed
    assert report.as_dict()["passed"] is True
    assert log


def test_sanity_checks_flag_masked_gradients(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    masked = InputQuantization(trained_cnn, bits=3).eval()
    zero, bad = zero_gradient_fraction(masked, x, y)
    assert zero == 1.0 and bad == 0.0
    report = run_sanity_checks(masked, x, y, eps=16 / 255, steps=5, square_queries=200, tol=0.01)
    failed = {c.name for c in report.checks if not c.passed}
    assert "gradients_informative" in failed
    assert not report.passed


def test_zero_gradient_fraction_does_not_depend_on_batch_size(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    masked = InputQuantization(trained_cnn, bits=3).eval()
    xm = torch.cat([x[: x.shape[0] // 2], x[: x.shape[0] // 2] * 0.5 + 0.25])
    whole = zero_gradient_fraction(trained_cnn, xm, y[: xm.shape[0]], batch_size=xm.shape[0])
    for bs in (1, 3, 7):
        assert zero_gradient_fraction(trained_cnn, xm, y[: xm.shape[0]], batch_size=bs) == whole
        assert zero_gradient_fraction(masked, xm, y[: xm.shape[0]], batch_size=bs) == (1.0, 0.0)
    assert zero_gradient_fraction(trained_cnn, x[:0], y[:0]) == (0.0, 0.0)


def test_bpda_recovers_attack_on_quantized_model(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    masked = InputQuantization(trained_cnn, bits=3).eval()
    eps = 24 / 255
    naive = PGD(eps, steps=10)(masked, x, y)
    adaptive = PGD(eps, steps=10)(bpda_view(masked), x, y)
    acc_naive = float((predict(masked, naive) == y).float().mean())
    acc_bpda = float((predict(masked, adaptive) == y).float().mean())
    assert acc_bpda < acc_naive


def test_random_noise_and_transfer_attacks(
    trained_cnn: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    noise = RandomNoise(0.03)(trained_cnn, x, y)
    assert float((noise - x).abs().max()) <= 0.03 + 1e-6
    t = TransferAttack(trained_cnn, PGD(0.03, steps=2))
    assert t.eps == 0.03 and t(nn.Identity(), x, y).shape == x.shape
    assert SquareAttack(0.03, 5).name == "square"
