"""Common interface for evasion attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from adversarial_robustness_lab.attacks.geometry import Norm, check_norm
from adversarial_robustness_lab.attacks.losses import LossFn


class Attack(ABC):
    """An evasion attack on a classifier that maps images in ``[0, 1]`` to logits.

    ``perturb`` returns adversarial images in the threat set ``S(x)`` of the attack. Attacks
    never change the model's train or eval mode; the caller decides (evaluation uses eval
    mode, adversarial training generates its examples in the mode its paper prescribes).
    Gradient attacks count forward and backward passes in ``queries`` (forward passes per
    image, summed over the batch) so cost can be compared across attacks.
    """

    name: str = "attack"

    def __init__(self, eps: float, norm: str = "linf", seed: int = 0) -> None:
        if eps < 0:
            raise ValueError("eps must be non-negative")
        self.eps = float(eps)
        self.norm: Norm = check_norm(norm)
        self.seed = seed
        self.queries = 0

    def generator(self) -> torch.Generator:
        """A CPU generator seeded from ``seed`` (random draws are moved to the device)."""
        return torch.Generator().manual_seed(self.seed)

    @abstractmethod
    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return adversarial examples for ``(x, y)``."""

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.perturb(model, x, y)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(eps={self.eps:.6g}, norm={self.norm})"


def loss_and_grad(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: LossFn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample loss, logits and the gradient of the summed loss with respect to ``x``.

    Summing makes each sample's gradient independent of the batch size.
    """
    x = x.detach().requires_grad_(True)
    logits = model(x)
    loss = loss_fn(logits, y)
    (grad,) = torch.autograd.grad(loss.sum(), x)
    return loss.detach(), logits.detach(), grad.detach()


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Predicted labels for ``x`` in batches."""
    out = [model(x[i : i + batch_size]).argmax(dim=1) for i in range(0, x.shape[0], batch_size)]
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=x.device)
