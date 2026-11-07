"""Carlini-Wagner L2 attack (Carlini and Wagner, 2017).

C&W looks for the smallest L2 perturbation that changes the prediction by solving::

    minimize_w  ||x'(w) - x||_2^2 + c * f(x'(w)),   x'(w) = (tanh(w) + 1) / 2
    f(x')     = max(z_y(x') - max_{j != y} z_j(x'), -kappa)

The tanh change of variables keeps ``x'`` inside the pixel box without clipping, so Adam
can optimize ``w`` freely. The constant ``c`` balances distance against misclassification;
a binary search over ``c`` (``binary_search_steps``) finds, per sample, the smallest ``c``
that still succeeds, which gives a near-minimal perturbation. The attack returns the
successful example with the smallest L2 distance found (or ``x`` itself when none was
found), so ``||x' - x||_2`` is an upper bound on each sample's minimal adversarial
distance (``last_distance``). With a finite budget ``eps``, examples farther than ``eps``
are discarded, so robust accuracy at ``eps`` counts a sample as robust when no example
within the budget was found.
"""

from __future__ import annotations

import torch
from torch import nn

from adversarial_robustness_lab.attacks.base import Attack
from adversarial_robustness_lab.attacks.losses import other_max


def _atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log((1 + x) / (1 - x))


class CarliniWagnerL2(Attack):
    """Minimum-norm C&W L2 attack with a per-sample binary search on ``c``."""

    name = "cw_l2"

    def __init__(
        self,
        eps: float = float("inf"),
        c: float = 1.0,
        kappa: float = 0.0,
        steps: int = 100,
        lr: float = 0.01,
        binary_search_steps: int = 5,
        abort_early: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(eps if eps != float("inf") else 0.0, "l2", seed)
        self.budget = eps
        if steps < 1 or binary_search_steps < 1:
            raise ValueError("steps and binary_search_steps must be at least 1")
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.last_distance: torch.Tensor | None = None

    def _f(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        real = logits.gather(1, y[:, None]).squeeze(1)
        return torch.clamp(real - other_max(logits, y), min=-self.kappa)

    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        device = x.device
        w0 = _atanh((2 * x - 1).clamp(-1 + 1e-6, 1 - 1e-6))
        const = torch.full((n,), float(self.c), device=device)
        lower = torch.zeros(n, device=device)
        upper = torch.full((n,), 1e10, device=device)
        best_dist = torch.full((n,), float("inf"), device=device)
        best_adv = x.clone()
        for _ in range(self.binary_search_steps):
            w = w0.clone().requires_grad_(True)
            opt = torch.optim.Adam([w], lr=self.lr)
            succeeded = torch.zeros(n, dtype=torch.bool, device=device)
            prev = float("inf")
            for step in range(self.steps):
                x_adv = (torch.tanh(w) + 1) / 2
                logits = model(x_adv)
                dist = (x_adv - x).pow(2).flatten(1).sum(1)
                f = self._f(logits, y)
                loss = (dist + const * f).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.queries += 2 * n
                with torch.no_grad():
                    wrong = logits.argmax(1) != y
                    l2 = dist.sqrt()
                    better = wrong & (l2 < best_dist)
                    best_dist = torch.where(better, l2, best_dist)
                    best_adv[better] = x_adv.detach()[better]
                    succeeded |= wrong
                if self.abort_early and step % max(1, self.steps // 10) == 0:
                    current = float(loss.detach())
                    if current > prev * 0.9999:
                        break
                    prev = current
            # Binary search: shrink c where the attack succeeded, grow it elsewhere.
            upper = torch.where(succeeded, torch.minimum(upper, const), upper)
            lower = torch.where(succeeded, lower, torch.maximum(lower, const))
            const = torch.where(upper < 1e9, (lower + upper) / 2, const * 10)
        if self.budget != float("inf"):
            # Under a budget, an example farther than eps is not an attack success.
            over = best_dist > self.budget
            best_adv[over] = x[over]
            best_dist = torch.where(over, torch.full_like(best_dist, float("inf")), best_dist)
        self.last_distance = best_dist
        return best_adv.detach()
