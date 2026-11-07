"""An AutoAttack-style ensemble: run attacks in sequence on the points still robust.

AutoAttack (Croce and Hein, 2020) evaluates a model with a fixed, parameter-free
sequence of complementary attacks and reports the worst case per sample: a point counts
as robust only if every attack in the sequence fails on it. Running each attack only on
the points that survived the previous ones keeps the cost close to that of the
strongest single attack.

``standard`` here is APGD-CE, then APGD-T (targeted DLR over the top 9 wrong classes),
then Square with 5,000 queries: AutoAttack's standard version without FAB-T, which is
not implemented here (the comparison with torchattacks' AutoAttack, which includes it,
measures what that omission costs). ``fast`` is APGD-CE then untargeted APGD-DLR, the
pair the original paper first proposed. Square exists only for L-infinity, so L2 ensembles stop after APGD-T.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from adversarial_robustness_lab.attacks.apgd import APGD, APGDTargeted
from adversarial_robustness_lab.attacks.base import Attack, predict
from adversarial_robustness_lab.attacks.geometry import check_norm
from adversarial_robustness_lab.attacks.square import SquareAttack

VERSIONS: tuple[str, ...] = ("standard", "fast", "rand")


@dataclass
class EnsembleResult:
    """Per-sample outcome of an ensemble run."""

    clean_correct: torch.Tensor
    robust: torch.Tensor
    broken_by: list[str]
    x_adv: torch.Tensor
    robust_after: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def clean_accuracy(self) -> float:
        return float(self.clean_correct.float().mean())

    @property
    def robust_accuracy(self) -> float:
        return float(self.robust.float().mean())


def build_attacks(
    eps: float,
    norm: str = "linf",
    version: str = "standard",
    steps: int = 100,
    n_target_classes: int = 9,
    square_queries: int = 5000,
    seed: int = 0,
) -> list[Attack]:
    """The attack sequence of an ensemble version."""
    norm = check_norm(norm)
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    if version == "fast":
        return [
            APGD(eps, norm, steps, "ce", seed=seed),
            APGD(eps, norm, steps, "dlr", seed=seed),
        ]
    if version == "rand":
        # For randomized defenses AutoAttack averages gradients over draws; here we
        # rerun APGD-CE and APGD-DLR with several restarts instead.
        return [
            APGD(eps, norm, steps, "ce", restarts=5, seed=seed),
            APGD(eps, norm, steps, "dlr", restarts=5, seed=seed),
        ]
    attacks: list[Attack] = [
        APGD(eps, norm, steps, "ce", seed=seed),
        APGDTargeted(eps, norm, steps, n_target_classes, seed=seed),
    ]
    if norm == "linf":
        attacks.append(SquareAttack(eps, square_queries, 0.8, seed=seed))
    return attacks


class AutoAttackEnsemble:
    """Sequential worst-case evaluation with a list of attacks."""

    def __init__(self, attacks: list[Attack], batch_size: int = 250) -> None:
        if not attacks:
            raise ValueError("an ensemble needs at least one attack")
        self.attacks = attacks
        self.batch_size = batch_size

    @classmethod
    def from_version(
        cls, eps: float, norm: str = "linf", version: str = "standard", **kw: int
    ) -> AutoAttackEnsemble:
        batch_size = kw.pop("batch_size", 250)
        return cls(build_attacks(eps, norm, version, **kw), batch_size)

    def run(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        progress: Callable[[str], None] | None = None,
    ) -> EnsembleResult:
        clean = predict(model, x, self.batch_size) == y
        robust = clean.clone()
        broken_by = ["clean" if not bool(c) else "" for c in clean.tolist()]
        x_adv = x.clone()
        result = EnsembleResult(clean, robust, broken_by, x_adv)
        result.robust_after["clean"] = int(clean.sum())
        for attack in self.attacks:
            name = getattr(attack, "name", type(attack).__name__)
            t0 = time.perf_counter()
            todo = robust.nonzero().flatten()
            for start in range(0, todo.numel(), self.batch_size):
                idx = todo[start : start + self.batch_size]
                adv = attack.perturb(model, x[idx], y[idx])
                with torch.no_grad():
                    fooled = model(adv).argmax(1) != y[idx]
                broken = idx[fooled]
                x_adv[broken] = adv[fooled]
                robust[broken] = False
                for i in broken.tolist():
                    broken_by[i] = name
            result.seconds[name] = time.perf_counter() - t0
            result.robust_after[name] = int(robust.sum())
            if progress is not None:
                progress(f"{name}: robust {int(robust.sum())}/{x.shape[0]}")
        return result
