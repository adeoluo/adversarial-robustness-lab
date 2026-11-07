"""Clean and robust accuracy of the model zoo on the CIFAR-10 test set (L-infinity, 8/255).

    uv run python benchmarks/robustness.py wrn16_2_pgd [--n 10000] [--ensemble lite|none]

For each model: clean accuracy, FGSM, PGD-20, PGD-20 with the C&W margin loss, and the
AutoAttack-style ensemble (APGD-CE 100 steps, then APGD-T over the top ``--targets``
wrong classes, then Square with ``--queries`` queries, each on the survivors of the
previous ones). Results go to ``benchmarks/results/robustness/<name>.json`` with
per-sample outcomes in a sibling ``.npz`` (for bootstrap intervals and overlaps).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
from common import RESULTS, checkpoint, hardware, write_json

from adversarial_robustness_lab.attacks import FGSM, PGD, AutoAttackEnsemble, build_attacks
from adversarial_robustness_lab.data import load_cifar10
from adversarial_robustness_lab.evaluation import check_threat_model, evaluate_attack
from adversarial_robustness_lab.models import count_parameters
from adversarial_robustness_lab.runtime import limit_memory, load_checkpoint, pick_device
from adversarial_robustness_lab.stats import wilson_interval

EPS = 8 / 255


def summarize(
    name: str, robust: np.ndarray, seconds: float, extra: dict[str, object] | None = None
) -> dict[str, object]:
    k, n = int(robust.sum()), int(robust.size)
    lo, hi = wilson_interval(k, n)
    return {
        "attack": name,
        "robust_accuracy": k / n,
        "ci95": [lo, hi],
        "n": n,
        "seconds": seconds,
        **(extra or {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--ensemble", choices=["lite", "none"], default="lite")
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--memory-fraction", type=float, default=0.36)
    args = parser.parse_args(argv)
    device = pick_device()
    limit_memory(device, args.memory_fraction)
    model, meta = load_checkpoint(checkpoint(args.name), device)
    test = load_cifar10().test.subset(range(args.n))
    x, y = test.tensors("cpu", as_uint8=True)
    rows: list[dict[str, object]] = []
    per_sample: dict[str, np.ndarray] = {}

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {args.name} {msg}", flush=True)

    single = {
        "fgsm": FGSM(EPS),
        "pgd20": PGD(EPS, 2 / 255, 20),
        "cw_pgd20": PGD(EPS, 2 / 255, 20, loss="margin"),
    }
    clean = None
    for key, attack in single.items():
        res = evaluate_attack(model, attack, x, y, EPS, "linf", args.batch_size)
        clean = res.clean_correct
        per_sample[key] = res.robust_correct
        rows.append(summarize(key, res.robust_correct, res.seconds, {"max_linf": res.max_distance}))
        log(f"{key}: {res.robust_accuracy:.4f} ({res.seconds:.0f}s)")
    assert clean is not None
    per_sample["clean"] = clean
    payload: dict[str, object] = {
        "name": args.name,
        "arch": meta["arch"],
        "method": meta["config"]["method"],
        "parameters": count_parameters(model),
        "epochs": len(meta["history"]),
        "train_seconds": meta["train_seconds"],
        "eps": EPS,
        "n": args.n,
        "clean_accuracy": float(clean.mean()),
        "clean_ci95": list(wilson_interval(int(clean.sum()), int(clean.size))),
        "attacks": rows,
        "hardware": hardware(),
        "device": str(device),
    }
    if args.ensemble == "lite":
        attacks = build_attacks(EPS, "linf", "standard", 100, args.targets, args.queries)
        ens = AutoAttackEnsemble(attacks, args.batch_size)
        xf = x.to(device).float().div_(255)
        res = ens.run(model, xf, y.to(device), progress=log)
        check_threat_model(xf, res.x_adv, EPS, "linf")
        robust = res.robust.cpu().numpy()
        per_sample["ensemble"] = robust
        payload["ensemble"] = {
            "attacks": [a.name for a in attacks],
            "targets": args.targets,
            "square_queries": args.queries,
            "robust_after": res.robust_after,
            "seconds": res.seconds,
            **summarize("ensemble", robust, sum(res.seconds.values())),
        }
        log(f"ensemble: {robust.mean():.4f}")
    write_json(RESULTS / "robustness" / f"{args.name}.json", payload)
    np.savez_compressed(RESULTS / "robustness" / f"{args.name}.npz", **per_sample)
    if device.type == "mps":
        torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
