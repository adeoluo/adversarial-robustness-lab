"""Ablations and scaling curves on the first ``--n`` test images.

    uv run python benchmarks/ablations.py [--n 1000]

* PGD steps (1 to 100) and random restarts on the PGD-AT model.
* APGD-CE against PGD at the same number of iterations.
* Square attack: robust accuracy as a function of the query budget (one 5,000-query
  run; a sample counts as broken at budget q if Square fooled it within q queries).
* Robust accuracy against epsilon (PGD-20) for every WRN-16-2 model.
Results: ``benchmarks/results/ablations.json``. Each item above is a resumable part:
after ``--max-minutes`` the script exits with status 3 and a rerun continues.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable

import numpy as np
import torch

from adversarial_robustness_lab.attacks import APGD, PGD, SquareAttack
from adversarial_robustness_lab.data import load_cifar10
from adversarial_robustness_lab.evaluation import evaluate_attack
from adversarial_robustness_lab.runtime import limit_memory, load_checkpoint, pick_device
from common import INCOMPLETE, RESULTS, Parts, checkpoint, hardware, write_json

EPS = 8 / 255


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--model", default="wrn16_2_pgd")
    parser.add_argument(
        "--eps-models",
        nargs="+",
        default=["wrn16_2_standard", "wrn16_2_fgsm_rs", "wrn16_2_pgd", "wrn16_2_trades"],
    )
    parser.add_argument("--memory-fraction", type=float, default=0.36)
    parser.add_argument("--max-minutes", type=float, default=35.0)
    args = parser.parse_args(argv)
    config = {"n": args.n, "model": args.model, "eps_models": args.eps_models}
    parts = Parts("ablations", config, args.max_minutes)
    device = pick_device()
    limit_memory(device, args.memory_fraction)
    x, y = load_cifar10().test.subset(range(args.n)).tensors("cpu", as_uint8=True)
    model, _ = load_checkpoint(checkpoint(args.model), device)
    out: dict[str, object] = {"n": args.n, "model": args.model, "eps": EPS, "hardware": hardware()}

    def run(attack: object, eps: float = EPS, m: object = model) -> tuple[float, float]:
        res = evaluate_attack(m, attack, x, y, eps, "linf")  # type: ignore[arg-type]
        print(
            f"[{time.strftime('%H:%M:%S')}] {attack!r} "
            f"{res.robust_accuracy:.4f} {res.seconds:.0f}s",
            flush=True,
        )
        return res.robust_accuracy, res.seconds

    def part(key: str, fn: Callable[[], object]) -> bool:
        if key not in parts:
            if parts.out_of_time():
                return False
            parts.save(key, fn())
        return True

    def square_curve() -> dict[str, object]:
        xf, yd = x.to(device).float() / 255, y.to(device)
        square = SquareAttack(EPS, 5000)
        adv = square(model, xf, yd)
        with torch.no_grad():
            clean = (model(xf).argmax(1) == yd).cpu().numpy()
            fooled = (model(adv).argmax(1) != yd).cpu().numpy()
        q = (
            square.query_counts.cpu().numpy()
            if square.query_counts is not None
            else np.zeros(len(y))
        )
        return {
            "curve": {
                b: float((clean & ~(fooled & (q <= b))).mean())
                for b in (10, 30, 100, 300, 1000, 3000, 5000)
            },
            "median_queries_fooled": float(np.median(q[clean & fooled]))
            if (clean & fooled).any()
            else None,
        }

    def eps_curve(name: str) -> dict[int, float]:
        m, _ = load_checkpoint(checkpoint(name), device)
        grid = [1, 2, 4, 6, 8, 10, 12, 16]
        return {e: run(PGD(e / 255, 2.5 * e / 255 / 20, 20), e / 255, m)[0] for e in grid}

    steps = [
        (
            "pgd_steps",
            lambda: {s: run(PGD(EPS, 2.5 * EPS / s, s))[0] for s in (1, 2, 5, 10, 20, 50, 100)},
        ),
        ("pgd_restarts", lambda: {r: run(PGD(EPS, 2 / 255, 20, restarts=r))[0] for r in (1, 2, 5)}),
        (
            "pgd_vs_apgd",
            lambda: {
                it: {
                    "pgd": run(PGD(EPS, 2.5 * EPS / it, it))[0],
                    "apgd_ce": run(APGD(EPS, "linf", it, "ce"))[0],
                }
                for it in (10, 25, 50, 100)
            },
        ),
        ("square", square_curve),
    ]
    for name in args.eps_models:
        if checkpoint(name).exists():
            steps.append((f"eps_curve/{name}", lambda name=name: eps_curve(name)))
    for key, fn in steps:
        if not part(key, fn):
            print("time budget reached; rerun to continue", flush=True)
            return INCOMPLETE
    out["pgd_steps"] = parts["pgd_steps"]
    out["pgd_restarts"] = parts["pgd_restarts"]
    out["pgd_vs_apgd"] = parts["pgd_vs_apgd"]
    out["square_queries"] = parts["square"]["curve"]
    out["square_median_queries_fooled"] = parts["square"]["median_queries_fooled"]
    out["eps_curve"] = {
        k.split("/", 1)[1]: v for k, v in parts.parts.items() if k.startswith("eps_curve/")
    }
    write_json(RESULTS / "ablations.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
