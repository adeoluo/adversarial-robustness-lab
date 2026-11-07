"""A deliberately gradient-masking defense, and the adaptive attack that breaks it.

``InputQuantization`` rounds every pixel to ``2^bits`` levels before the classifier. The
rounding has zero derivative almost everywhere, so the input gradient of the defended
model is zero and gradient attacks stall, while the model is no more robust: an
attacker who knows the defense can still move pixels across rounding thresholds.
Athalye, Carlini and Wagner (2018) call this shattered gradients and break it with
BPDA (Backward Pass Differentiable Approximation): keep the rounding in the forward
pass and treat it as the identity in the backward pass (a straight-through estimator).

The defense exists to show that the sanity checks in
:mod:`adversarial_robustness_lab.sanity` catch masking and that the evaluation then
recovers the true robustness with BPDA; it is not a defense to deploy.
"""

from __future__ import annotations

import torch
from torch import nn


class _RoundStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: torch.Tensor, levels: float) -> torch.Tensor:
        return torch.round(x * levels) / levels

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad, None


def quantize(x: torch.Tensor, bits: int, straight_through: bool = False) -> torch.Tensor:
    """Round ``x`` in ``[0, 1]`` to ``2^bits`` levels; BPDA gradient if ``straight_through``."""
    levels = float(2**bits - 1)
    if straight_through:
        out: torch.Tensor = _RoundStraightThrough.apply(x, levels)
        return out
    return torch.round(x * levels) / levels


class InputQuantization(nn.Module):
    """``model(quantize(x))``; set ``bpda=True`` to expose straight-through gradients."""

    def __init__(self, model: nn.Module, bits: int = 3, bpda: bool = False) -> None:
        super().__init__()
        if not 1 <= bits <= 8:
            raise ValueError("bits must be between 1 and 8")
        self.model = model
        self.bits = bits
        self.bpda = bpda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(quantize(x, self.bits, self.bpda))


def bpda_view(defended: InputQuantization) -> InputQuantization:
    """The same defended model with BPDA gradients (shares the weights)."""
    return InputQuantization(defended.model, defended.bits, bpda=True)
