"""Square attack, L-infinity (Andriushchenko et al., 2020): a score-based black-box attack.

Square uses only the model's output scores, no gradients, so it is unaffected by
gradient masking; that is why AutoAttack ends with it. Each query changes one random
square window of side ``s`` of the current perturbation to a vertex of the L-infinity
ball (``+eps`` or ``-eps`` per colour channel) and keeps the change if it lowers the
margin loss ``z_y - max_{j != y} z_j``. The attack stops for a sample as soon as the
margin is negative (misclassified).

* Initialization: vertical stripes, ``delta[c, :, w] = +-eps`` drawn per column and channel.
* Window size: ``s = round(sqrt(p * h * w))`` clipped to ``[1, h - 1]``, where the fraction
  ``p`` starts at ``p_init`` and is halved at query 10, 50, 200, 500, 1000, 2000, 4000,
  6000 and 8000 (rescaled to a budget of 10,000 queries).
* A window whose new values equal the current ones (after clipping to ``[0, 1]``) is
  redrawn, so every query changes the image.

Window positions are drawn per image (as in the authors' NumPy code); all randomness
comes from a seeded CPU generator. ``query_counts`` holds, after each call, the number of
model queries each sample used, including the initial one.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from adversarial_robustness_lab.attacks.base import Attack
from adversarial_robustness_lab.attacks.losses import margin


def p_selection(p_init: float, it: int, n_queries: int) -> float:
    """Fraction of pixels changed at query ``it`` (the schedule of the paper)."""
    i = int(it / n_queries * 10000)
    for bound, div in (
        (8000, 512),
        (6000, 256),
        (4000, 128),
        (2000, 64),
        (1000, 32),
        (500, 16),
        (200, 8),
        (50, 4),
        (10, 2),
    ):
        if i > bound:
            return p_init / div
    return p_init


def _signs(shape: tuple[int, ...], gen: torch.Generator, device: torch.device) -> torch.Tensor:
    return (torch.randint(0, 2, shape, generator=gen) * 2 - 1).to(
        device=device, dtype=torch.float32
    )


class SquareAttack(Attack):
    """Score-based Square attack under the L-infinity norm."""

    name = "square"

    def __init__(
        self,
        eps: float,
        n_queries: int = 5000,
        p_init: float = 0.8,
        seed: int = 0,
        max_redraws: int = 10,
    ) -> None:
        super().__init__(eps, "linf", seed)
        if n_queries < 1:
            raise ValueError("n_queries must be at least 1")
        if not 0 < p_init <= 1:
            raise ValueError("p_init must be in (0, 1]")
        self.n_queries = n_queries
        self.p_init = p_init
        self.max_redraws = max_redraws
        self.query_counts: torch.Tensor | None = None

    @torch.no_grad()
    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gen = self.generator()
        n, c, h, w = x.shape
        dev = x.device
        x_best = (x + self.eps * _signs((n, c, 1, w), gen, dev)).clamp(0.0, 1.0)
        # Positive while the sample is still classified correctly.
        margin_loss = -margin(model(x_best), y)
        queries = torch.ones(n, device=dev)
        rows = torch.arange(h, device=dev)
        cols = torch.arange(w, device=dev)
        for it in range(1, self.n_queries):
            todo = (margin_loss > 0).nonzero().flatten()
            if todo.numel() == 0:
                break
            m = todo.numel()
            xc, xb = x[todo], x_best[todo]
            delta = xb - xc
            p = p_selection(self.p_init, it, self.n_queries)
            s = min(max(round(math.sqrt(p * h * w)), 1), h - 1)
            vh = torch.randint(0, h - s, (m,), generator=gen).to(dev)
            vw = torch.randint(0, w - s, (m,), generator=gen).to(dev)
            in_h = (rows[None, :] >= vh[:, None]) & (rows[None, :] < vh[:, None] + s)
            in_w = (cols[None, :] >= vw[:, None]) & (cols[None, :] < vw[:, None] + s)
            window = (in_h[:, :, None] & in_w[:, None, :])[:, None]
            new_delta = torch.where(window, self.eps * _signs((m, c, 1, 1), gen, dev), delta)
            for _ in range(self.max_redraws):
                changed = ((xc + new_delta).clamp(0, 1) - xb).abs().mul(window).flatten(1)
                same = changed.amax(dim=1) < 1e-7
                if not bool(same.any()):
                    break
                redraw = torch.where(window, self.eps * _signs((m, c, 1, 1), gen, dev), new_delta)
                new_delta = torch.where(same.view(-1, 1, 1, 1), redraw, new_delta)
            x_new = (xc + new_delta).clamp(0.0, 1.0)
            new_margin = -margin(model(x_new), y[todo])
            queries[todo] += 1
            improved = new_margin < margin_loss[todo]
            idx = todo[improved]
            x_best[idx] = x_new[improved]
            margin_loss[idx] = new_margin[improved]
        self.queries += int(queries.sum())
        self.query_counts = queries
        return x_best
