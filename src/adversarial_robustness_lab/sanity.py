"""Gradient-masking sanity checks (Athalye et al., 2018; Carlini et al., 2019).

A defense that hides or shatters gradients makes gradient attacks fail without making
the model robust. Each check below compares attacks whose ordering is known when
gradients are informative; a failed check means the white-box number cannot be trusted
and the evaluation needs an adaptive attack (for example BPDA, see
:mod:`adversarial_robustness_lab.masking`).

1. **Iterative beats single step**: PGD accuracy is at most FGSM accuracy.
2. **White box beats black box**: PGD accuracy is at most Square (score-based) accuracy.
3. **Unbounded attack succeeds**: with ``eps = 1`` (any image is reachable) PGD drives
   accuracy to about zero.
4. **Monotone in epsilon**: accuracy does not increase as the budget grows.
5. **More steps help the attacker**: PGD with 5x the steps is at least as strong.
6. **Gradients are informative**: few inputs have an all-zero or non-finite gradient.
7. **Random noise is weaker**: uniform noise in the ball hurts less than PGD.
8. **Transfer is weaker than white box** (with a surrogate model): PGD examples crafted
   on the surrogate hurt less than PGD on the model itself.

Each check allows a slack ``tol`` (default one percentage point) for sampling noise.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from adversarial_robustness_lab.attacks.base import Attack
from adversarial_robustness_lab.attacks.geometry import random_start
from adversarial_robustness_lab.attacks.pgd import FGSM, PGD
from adversarial_robustness_lab.attacks.square import SquareAttack
from adversarial_robustness_lab.evaluation import evaluate_attack


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    values: dict[str, float] = field(default_factory=dict)


@dataclass
class SanityReport:
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}


class RandomNoise(Attack):
    """Uniform noise in the L-infinity ball (a baseline, not an attack)."""

    name = "uniform_noise"

    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return random_start(x, self.eps, "linf", self.generator())


class TransferAttack(Attack):
    """Craft PGD examples on a surrogate model and apply them to the target."""

    name = "transfer_pgd"

    def __init__(self, surrogate: nn.Module, inner: Attack) -> None:
        super().__init__(inner.eps, inner.norm, inner.seed)
        self.surrogate = surrogate
        self.inner = inner

    def perturb(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.inner.perturb(self.surrogate, x, y)


def zero_gradient_fraction(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[float, float]:
    """Fractions of inputs whose loss gradient is all zero and non-finite."""
    xg = x.detach().clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(xg), y, reduction="sum")
    (grad,) = torch.autograd.grad(loss, xg)
    flat = grad.flatten(1)
    zero = (flat.abs().amax(1) == 0).float().mean()
    bad = (~torch.isfinite(flat).all(1)).float().mean()
    return float(zero), float(bad)


def run_sanity_checks(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8 / 255,
    steps: int = 20,
    square_queries: int = 1000,
    surrogate: nn.Module | None = None,
    tol: float = 0.01,
    batch_size: int = 250,
    progress: Callable[[str], None] | None = None,
) -> SanityReport:
    """Run every check on ``(x, y)`` (images as float or ``uint8``) and return the report."""
    model.eval()

    def acc(attack: Attack, e: float = eps) -> float:
        res = evaluate_attack(model, attack, x, y, e, "linf", batch_size, verify=True)
        if progress is not None:
            progress(f"{attack!r}: {res.robust_accuracy:.4f}")
        return res.robust_accuracy

    pgd = acc(PGD(eps, 2.5 * eps / steps, steps))
    fgsm = acc(FGSM(eps))
    checks = [
        Check(
            "iterative_beats_single_step",
            pgd <= fgsm + tol,
            f"PGD-{steps} {pgd:.4f} vs FGSM {fgsm:.4f}",
            {"pgd": pgd, "fgsm": fgsm},
        )
    ]
    square = acc(SquareAttack(eps, square_queries))
    checks.append(
        Check(
            "white_box_beats_black_box",
            pgd <= square + tol,
            f"PGD-{steps} {pgd:.4f} vs Square-{square_queries} {square:.4f}",
            {"pgd": pgd, "square": square},
        )
    )
    unbounded = acc(PGD(1.0, 0.1, 50), 1.0)
    checks.append(
        Check(
            "unbounded_attack_succeeds",
            unbounded <= tol,
            f"PGD-50 at eps=1: {unbounded:.4f}",
            {"unbounded": unbounded},
        )
    )
    grid = [eps / 4, eps / 2, eps, 2 * eps]
    curve = [acc(PGD(e, 2.5 * e / steps, steps), e) for e in grid]
    monotone = all(b <= a + tol for a, b in itertools.pairwise(curve))
    checks.append(
        Check(
            "monotone_in_eps",
            monotone,
            "accuracy at eps x (1/4, 1/2, 1, 2): " + ", ".join(f"{c:.4f}" for c in curve),
            {f"eps_{e:.5f}": c for e, c in zip(grid, curve, strict=True)},
        )
    )
    more = acc(PGD(eps, 2.5 * eps / (5 * steps), 5 * steps))
    checks.append(
        Check(
            "more_steps_help_attacker",
            more <= pgd + tol,
            f"PGD-{5 * steps} {more:.4f} vs PGD-{steps} {pgd:.4f}",
            {"pgd_more": more, "pgd": pgd},
        )
    )
    device = next(model.parameters()).device
    n = min(500, x.shape[0])
    xs = x[:n].to(device)
    xs = xs.float() / 255 if xs.dtype == torch.uint8 else xs
    zero, bad = zero_gradient_fraction(model, xs, y[:n].to(device))
    checks.append(
        Check(
            "gradients_informative",
            zero <= tol and bad == 0.0,
            f"all-zero gradients {zero:.4f}, non-finite {bad:.4f}",
            {"zero_gradient": zero, "nonfinite_gradient": bad},
        )
    )
    noise = acc(RandomNoise(eps))
    checks.append(
        Check(
            "random_noise_weaker",
            pgd <= noise + tol,
            f"uniform noise {noise:.4f} vs PGD-{steps} {pgd:.4f}",
            {"noise": noise, "pgd": pgd},
        )
    )
    if surrogate is not None:
        surrogate.eval()
        transfer = acc(TransferAttack(surrogate, PGD(eps, 2.5 * eps / steps, steps)))
        checks.append(
            Check(
                "transfer_weaker_than_white_box",
                pgd <= transfer + tol,
                f"transfer {transfer:.4f} vs white-box {pgd:.4f}",
                {"transfer": transfer, "pgd": pgd},
            )
        )
    return SanityReport(checks)
