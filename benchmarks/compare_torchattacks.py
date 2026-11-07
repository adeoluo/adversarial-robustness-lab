"""Attack strength and speed against torchattacks 3.5 on the same model and images.

    uv run python benchmarks/compare_torchattacks.py wrn16_2_pgd [--n-cheap 10000] [--n-costly 1000]

Each pair runs the same attack with the same budget and settings in both libraries on
the same test images; lower robust accuracy means a stronger attack. Cheap attacks
(FGSM, PGD, PGD-L2) use ``--n-cheap`` images, the costly ones (C&W, APGD, Square)
``--n-costly`` and the AutoAttack pair ``--n-autoattack``. torchattacks' AutoAttack
includes FAB-T, which this package does not implement. ``--only`` runs a subset of the
pairs and merges them into the existing results file. Results: ``benchmarks/results/compare/<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable

import torch
import torchattacks
from torch import nn

from adversarial_robustness_lab.attacks import (
    APGD,
    FGSM,
    PGD,
    AutoAttackEnsemble,
    CarliniWagnerL2,
    SquareAttack,
    build_attacks,
)
from adversarial_robustness_lab.attacks.geometry import flat_norm
from adversarial_robustness_lab.data import load_cifar10
from adversarial_robustness_lab.evaluation import evaluate_attack
from adversarial_robustness_lab.runtime import limit_memory, load_checkpoint, pick_device
from adversarial_robustness_lab.stats import wilson_interval
from common import RESULTS, checkpoint, hardware, write_json

EPS = 8 / 255
L2_EPS = 0.5
AttackFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


def ta(attack: torchattacks.attack.Attack, budget: float | None = None) -> AttackFn:
    def run(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        adv = attack(x, y).to(x.device)
        if budget is not None:
            # C&W returns its best example at any distance; outside the budget it does not count.
            over = flat_norm(adv - x, "l2") > budget + 1e-6
            adv[over] = x[over]
        return adv

    return run


def ensemble_fn(eps: float, targets: int, queries: int) -> AttackFn:
    def run(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ens = AutoAttackEnsemble(
            build_attacks(eps, "linf", "standard", 100, targets, queries), x.shape[0]
        )
        return ens.run(model, x, y).x_adv

    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--n-cheap", type=int, default=10_000)
    parser.add_argument("--n-costly", type=int, default=1000)
    parser.add_argument("--n-autoattack", type=int, default=500)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--memory-fraction", type=float, default=0.36)
    args = parser.parse_args(argv)
    device = pick_device()
    limit_memory(device, args.memory_fraction)
    model, meta = load_checkpoint(checkpoint(args.name), device)
    test = load_cifar10().test
    m = model
    pairs: list[tuple[str, str, float, int, AttackFn, AttackFn]] = [
        ("FGSM", "linf", EPS, args.n_cheap, FGSM(EPS), ta(torchattacks.FGSM(m, eps=EPS))),
        (
            "PGD-20",
            "linf",
            EPS,
            args.n_cheap,
            PGD(EPS, 2 / 255, 20),
            ta(torchattacks.PGD(m, eps=EPS, alpha=2 / 255, steps=20, random_start=True)),
        ),
        (
            "PGD-L2-20",
            "l2",
            L2_EPS,
            args.n_cheap,
            PGD(L2_EPS, 2.5 * L2_EPS / 20, 20, "l2"),
            ta(
                torchattacks.PGDL2(
                    m, eps=L2_EPS, alpha=2.5 * L2_EPS / 20, steps=20, random_start=True
                )
            ),
        ),
        (
            "C&W-L2 (c=1, 100 steps)",
            "l2",
            L2_EPS,
            args.n_costly,
            CarliniWagnerL2(
                L2_EPS, c=1.0, steps=100, lr=0.01, binary_search_steps=1, abort_early=False
            ),
            ta(torchattacks.CW(m, c=1.0, kappa=0, steps=100, lr=0.01), L2_EPS),
        ),
        (
            "APGD-CE-100",
            "linf",
            EPS,
            args.n_costly,
            APGD(EPS, "linf", 100, "ce"),
            ta(torchattacks.APGD(m, eps=EPS, steps=100, loss="ce", n_restarts=1, seed=0)),
        ),
        (
            "APGD-DLR-100",
            "linf",
            EPS,
            args.n_costly,
            APGD(EPS, "linf", 100, "dlr"),
            ta(torchattacks.APGD(m, eps=EPS, steps=100, loss="dlr", n_restarts=1, seed=0)),
        ),
        (
            "Square-1000",
            "linf",
            EPS,
            args.n_costly,
            SquareAttack(EPS, 1000, 0.8),
            ta(torchattacks.Square(m, eps=EPS, n_queries=1000, p_init=0.8, n_restarts=1, seed=0)),
        ),
        (
            "AutoAttack standard",
            "linf",
            EPS,
            args.n_autoattack,
            ensemble_fn(EPS, 9, 5000),
            ta(
                torchattacks.AutoAttack(
                    m, norm="Linf", eps=EPS, version="standard", n_classes=10, seed=0
                )
            ),
        ),
    ]
    rows: list[dict[str, object]] = []
    out = RESULTS / "compare" / f"{args.name}.json"
    for label, norm, eps, n, ours, ref in pairs:
        if args.only and label not in args.only:
            continue
        x, y = test.subset(range(n)).tensors("cpu", as_uint8=True)
        row: dict[str, object] = {"attack": label, "norm": norm, "eps": eps, "n": n}
        outcomes = {}
        for side, fn in (("ours", ours), ("torchattacks", ref)):
            res = evaluate_attack(m, fn, x, y, eps, norm, args.batch_size, name=label)  # type: ignore[arg-type]
            k = int(res.robust_correct.sum())
            row[side] = {
                "robust_accuracy": res.robust_accuracy,
                "ci95": list(wilson_interval(k, n)),
                "seconds": res.seconds,
                "clean_accuracy": res.clean_accuracy,
            }
            outcomes[side] = res.robust_correct
            print(
                f"[{time.strftime('%H:%M:%S')}] {label:<26} {side:<13} "
                f"{res.robust_accuracy:.4f} {res.seconds:7.1f}s",
                flush=True,
            )
        a, b = outcomes["ours"], outcomes["torchattacks"]
        row["only_ours_breaks"] = int((~a & b).sum())
        row["only_torchattacks_breaks"] = int((a & ~b).sum())
        row["both_robust"] = int((a & b).sum())
        rows.append(row)
        prev = [] if not out.exists() or args.only is None else json.loads(out.read_text())["rows"]
        merged = [r for r in prev if r["attack"] not in {q["attack"] for q in rows}] + rows
        write_json(
            out,
            {
                "name": args.name,
                "arch": meta["arch"],
                "method": meta["config"]["method"],
                "rows": merged,
                "hardware": hardware(),
                "device": str(device),
                "torchattacks": torchattacks.__version__,
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
