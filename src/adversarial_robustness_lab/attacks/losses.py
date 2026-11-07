"""Objective functions maximized by the attacks, all returned per sample.

With logits ``z`` and true class ``y``:

* Cross-entropy: ``CE(z, y) = -z_y + log sum_j exp(z_j)``.
* Margin (Carlini and Wagner, 2017): ``max_{j != y} z_j - z_y``; positive exactly when the
  sample is misclassified. ``kappa`` caps it at a confidence level.
* DLR, difference of logits ratio (Croce and Hein, 2020):
  ``-(z_y - max_{j != y} z_j) / (z_pi1 - z_pi3 + 1e-12)`` where ``z_pi1 >= z_pi2 >= ...`` are
  the sorted logits. It is invariant to shifting and to positive rescaling of the logits,
  which is what makes it robust to the logit-scaling form of gradient masking that
  defeats cross-entropy.
* Targeted DLR: ``-(z_y - z_t) / (z_pi1 - (z_pi3 + z_pi4) / 2 + 1e-12)``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y, reduction="none")


def other_max(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Largest logit over the wrong classes."""
    mask = F.one_hot(y, logits.shape[1]).bool()
    return logits.masked_fill(mask, float("-inf")).amax(dim=1)


def margin(logits: torch.Tensor, y: torch.Tensor, kappa: float = float("inf")) -> torch.Tensor:
    """Carlini-Wagner margin ``max_{j != y} z_j - z_y``, clipped above at ``kappa``."""
    m = other_max(logits, y) - logits.gather(1, y[:, None]).squeeze(1)
    return m.clamp(max=kappa) if kappa != float("inf") else m


def dlr(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Untargeted difference of logits ratio (larger means closer to misclassified)."""
    if logits.shape[1] < 3:
        raise ValueError("DLR needs at least 3 classes")
    z_sorted = logits.sort(dim=1).values
    z_y = logits.gather(1, y[:, None]).squeeze(1)
    return -(z_y - other_max(logits, y)) / (z_sorted[:, -1] - z_sorted[:, -3] + 1e-12)


def dlr_targeted(logits: torch.Tensor, y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Targeted DLR towards class ``target`` (larger means closer to the target)."""
    if logits.shape[1] < 4:
        raise ValueError("targeted DLR needs at least 4 classes")
    z_sorted = logits.sort(dim=1).values
    z_y = logits.gather(1, y[:, None]).squeeze(1)
    z_t = logits.gather(1, target[:, None]).squeeze(1)
    return -(z_y - z_t) / (z_sorted[:, -1] - 0.5 * (z_sorted[:, -3] + z_sorted[:, -4]) + 1e-12)


LOSSES: dict[str, LossFn] = {
    "ce": cross_entropy,
    "margin": margin,
    "dlr": dlr,
}


def get_loss(name: str) -> LossFn:
    try:
        return LOSSES[name]
    except KeyError as exc:
        raise ValueError(f"unknown loss {name!r}; choose from {sorted(LOSSES)}") from exc
