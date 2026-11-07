"""Evasion attacks: FGSM, PGD (L-infinity and L2), C&W L2, APGD, APGD-T, Square, ensembles."""

from __future__ import annotations

from adversarial_robustness_lab.attacks.apgd import APGD, APGDTargeted
from adversarial_robustness_lab.attacks.autoattack import (
    AutoAttackEnsemble,
    EnsembleResult,
    build_attacks,
)
from adversarial_robustness_lab.attacks.base import Attack, predict
from adversarial_robustness_lab.attacks.cw import CarliniWagnerL2
from adversarial_robustness_lab.attacks.pgd import FGSM, PGD
from adversarial_robustness_lab.attacks.square import SquareAttack

ATTACK_NAMES: tuple[str, ...] = (
    "fgsm",
    "pgd",
    "pgd-l2",
    "cw-linf",
    "cw-l2",
    "apgd-ce",
    "apgd-dlr",
    "apgd-t",
    "square",
)


def make_attack(
    name: str,
    eps: float,
    steps: int = 20,
    restarts: int = 1,
    queries: int = 5000,
    seed: int = 0,
) -> Attack:
    """Build an attack by its command-line name with the defaults used in the benchmarks."""
    if name == "fgsm":
        return FGSM(eps, seed=seed)
    if name == "pgd":
        return PGD(eps, 2.5 * eps / steps, steps, "linf", True, restarts, "ce", seed)
    if name == "pgd-l2":
        return PGD(eps, 2.5 * eps / steps, steps, "l2", True, restarts, "ce", seed)
    if name == "cw-linf":
        return PGD(eps, 2.5 * eps / steps, steps, "linf", True, restarts, "margin", seed)
    if name == "cw-l2":
        return CarliniWagnerL2(eps, steps=steps, seed=seed)
    if name == "apgd-ce":
        return APGD(eps, "linf", steps, "ce", restarts, seed=seed)
    if name == "apgd-dlr":
        return APGD(eps, "linf", steps, "dlr", restarts, seed=seed)
    if name == "apgd-t":
        return APGDTargeted(eps, "linf", steps, 9, restarts, seed=seed)
    if name == "square":
        return SquareAttack(eps, queries, seed=seed)
    raise ValueError(f"unknown attack {name!r}; choose from {ATTACK_NAMES}")


def attack_norm(name: str) -> str:
    """The norm an attack name works in."""
    return "l2" if name in ("pgd-l2", "cw-l2") else "linf"


__all__ = [
    "APGD",
    "ATTACK_NAMES",
    "FGSM",
    "PGD",
    "APGDTargeted",
    "Attack",
    "AutoAttackEnsemble",
    "CarliniWagnerL2",
    "EnsembleResult",
    "SquareAttack",
    "attack_norm",
    "build_attacks",
    "make_attack",
    "predict",
]
