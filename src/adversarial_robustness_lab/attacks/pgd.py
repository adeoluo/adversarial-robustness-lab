"""Gradient attacks with a fixed step: FGSM and projected gradient descent (PGD).

**FGSM** (Goodfellow et al., 2015) takes one step of size ``eps`` along the sign of the
input gradient of the loss: ``x' = clip_[0,1](x + eps * sign(grad_x L(f(x), y)))``.

**PGD** (Madry et al., 2018) iterates a normalized ascent step followed by projection
onto the threat set ``S(x)``::

    x^0     = x, or a uniform random point of S(x)
    x^{t+1} = Proj_S(x)( x^t + alpha * d(grad_x L(f(x^t), y)) )

with ``d(g) = sign(g)`` under L-infinity and ``d(g) = g / ||g||_2`` under L2 (the steepest
ascent direction of each norm). With ``restarts > 1`` the attack reruns from new random
starts and keeps, per sample, the first example that is misclassified (otherwise the one
with the highest final loss). ``loss="margin"`` gives the PGD variant with the
Carlini-Wagner margin loss often reported as CW-infinity.
"""

from __future__ import annotations

import torch
from torch import nn

from adversarial_robustness_lab.attacks.base import Attack, loss_and_grad
from adversarial_robustness_lab.attacks.geometry import (
    normalize_gradient,
    project_to_threat,
    random_start,
)
from adversarial_robustness_lab.attacks.losses import get_loss


class FGSM(Attack):
    """Fast gradient sign method (single step of size ``eps``, L-infinity only)."""

    name = "fgsm"

    def __init__(self, eps: float, loss: str = "ce", seed: int = 0) -> None:
        super().__init__(eps, "linf", seed)
        self.loss_fn = get_loss(loss)

    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, grad = loss_and_grad(model, x, y, self.loss_fn)
        self.queries += 2 * x.shape[0]
        return (x + self.eps * grad.sign()).clamp(0.0, 1.0).detach()


class PGD(Attack):
    """Projected gradient descent under the L-infinity or L2 norm, with random restarts."""

    name = "pgd"

    def __init__(
        self,
        eps: float,
        alpha: float | None = None,
        steps: int = 10,
        norm: str = "linf",
        random_init: bool = True,
        restarts: int = 1,
        loss: str = "ce",
        seed: int = 0,
    ) -> None:
        super().__init__(eps, norm, seed)
        if steps < 1 or restarts < 1:
            raise ValueError("steps and restarts must be at least 1")
        # Default step: 2.5 * eps / steps (Madry et al.), so the walk can cross the ball.
        self.alpha = float(alpha) if alpha is not None else 2.5 * self.eps / steps
        self.steps = steps
        self.random_init = random_init
        self.restarts = restarts
        self.loss_name = loss
        self.loss_fn = get_loss(loss)

    def _run(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, gen: torch.Generator
    ) -> torch.Tensor:
        x_adv = random_start(x, self.eps, self.norm, gen) if self.random_init else x.clone()
        for _ in range(self.steps):
            _, _, grad = loss_and_grad(model, x_adv, y, self.loss_fn)
            x_adv = x_adv + self.alpha * normalize_gradient(grad, self.norm)
            x_adv = project_to_threat(x, x_adv, self.eps, self.norm)
        self.queries += 2 * self.steps * x.shape[0]
        return x_adv.detach()

    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gen = self.generator()
        best = x.clone()
        best_loss = torch.full((x.shape[0],), float("-inf"), device=x.device)
        fooled = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for _ in range(self.restarts):
            x_adv = self._run(model, x, y, gen)
            with torch.no_grad():
                logits = model(x_adv)
                loss = self.loss_fn(logits, y)
                success = logits.argmax(dim=1) != y
            self.queries += x.shape[0]
            # Keep successes first, then the highest loss among the not yet fooled.
            take = (success & ~fooled) | (~fooled & ~success & (loss > best_loss))
            best[take] = x_adv[take]
            best_loss = torch.where(take, loss, best_loss)
            fooled |= success
            if bool(fooled.all()):
                break
        return best
