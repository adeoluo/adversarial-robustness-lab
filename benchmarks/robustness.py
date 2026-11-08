"""Clean and robust accuracy of the model zoo on the CIFAR-10 test set (L-infinity, 8/255).

    uv run python benchmarks/robustness.py wrn16_2_pgd [--n 10000] [--max-minutes 35]

For each model: clean accuracy, FGSM, PGD-20, PGD-20 with the C&W margin loss, and the
AutoAttack-style ensemble (APGD-CE 100 steps, then APGD-T over the top ``--targets``
wrong classes, then Square with ``--queries`` queries, each on the survivors of the
previous ones). Results go to ``benchmarks/results/robustness/<name>.json`` with
per-sample outcomes in a sibling ``.npz`` (for bootstrap intervals and overlaps).

The run is resumable: per-sample progress is saved to ``$ARL_DATA_DIR/runs/<name>.robustness.npz``
(keyed by the settings and a SHA-256 prefix of the checkpoint, so new weights start afresh)
after every batch, and the script exits with status 3 once ``--max-minutes`` have passed
so that long evaluations run as a sequence of short steps. Rerun the same command to
continue; the summary is written when every stage has covered every image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from adversarial_robustness_lab.attacks import FGSM, PGD, build_attacks
from adversarial_robustness_lab.attacks.base import Attack
from adversarial_robustness_lab.data import load_cifar10, to_float
from adversarial_robustness_lab.evaluation import check_threat_model
from adversarial_robustness_lab.models import count_parameters
from adversarial_robustness_lab.runtime import limit_memory, load_checkpoint, pick_device
from adversarial_robustness_lab.stats import wilson_interval
from common import RESULTS, RUNS, checkpoint, fingerprint, hardware, write_json

EPS = 8 / 255
INCOMPLETE = 3


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


class State:
    """Per-sample progress of every stage, saved after each batch."""

    def __init__(self, path: str, n: int, stages: list[str], config: dict[str, object]) -> None:
        self.path = path
        self.arrays: dict[str, np.ndarray] = {}
        self.meta: dict[str, object] = {"config": config, "seconds": {}, "max_linf": {}}
        try:
            data = np.load(path)
            meta = json.loads(str(data["meta"]))
            if meta["config"] != config:
                raise ValueError("configuration changed")
            self.meta = meta
            self.arrays = {k: data[k] for k in data.files if k != "meta"}
        except (OSError, ValueError, KeyError):
            pass
        self.arrays.setdefault("clean", np.zeros(n, dtype=bool))
        self.arrays.setdefault("clean_done", np.zeros(n, dtype=bool))
        for s in stages:
            self.arrays.setdefault(s, np.zeros(n, dtype=bool))
            self.arrays.setdefault(f"{s}_done", np.zeros(n, dtype=bool))

    def save(self) -> None:
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(self.meta)), **self.arrays)
        os.replace(tmp, self.path)

    def add_seconds(self, stage: str, s: float) -> None:
        sec = self.meta["seconds"]
        assert isinstance(sec, dict)
        sec[stage] = sec.get(stage, 0.0) + s

    def note_linf(self, stage: str, d: float) -> None:
        m = self.meta["max_linf"]
        assert isinstance(m, dict)
        m[stage] = max(m.get(stage, 0.0), d)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--ensemble", choices=["lite", "none"], default="lite")
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--memory-fraction", type=float, default=0.36)
    parser.add_argument("--max-minutes", type=float, default=35.0)
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    device = pick_device()
    limit_memory(device, args.memory_fraction)
    model, meta = load_checkpoint(checkpoint(args.name), device)
    test = load_cifar10().test.subset(range(args.n))
    x, y = test.tensors("cpu", as_uint8=True)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {args.name} {msg}", flush=True)

    single: dict[str, Attack] = {
        "fgsm": FGSM(EPS),
        "pgd20": PGD(EPS, 2 / 255, 20),
        "cw_pgd20": PGD(EPS, 2 / 255, 20, loss="margin"),
    }
    ens_attacks = (
        build_attacks(EPS, "linf", "standard", 100, args.targets, args.queries)
        if args.ensemble == "lite"
        else []
    )
    ens_names = [a.name for a in ens_attacks]
    config: dict[str, object] = {
        k: getattr(args, k) for k in ("n", "ensemble", "targets", "queries")
    }
    config["checkpoint"] = fingerprint(args.name)
    RUNS.mkdir(parents=True, exist_ok=True)
    state = State(str(RUNS / f"{args.name}.robustness.npz"), args.n, [*single, *ens_names], config)
    arr = state.arrays

    def out_of_time() -> bool:
        return (time.perf_counter() - t_start) / 60 > args.max_minutes

    # clean predictions
    for start in range(0, args.n, args.batch_size):
        sl = slice(start, start + args.batch_size)
        if arr["clean_done"][sl].all():
            continue
        with torch.no_grad():
            pred = model(to_float(x[sl]).to(device)).argmax(1).cpu()
        arr["clean"][sl] = (pred == y[sl]).numpy()
        arr["clean_done"][sl] = True
    state.save()
    clean = arr["clean"]

    def run_stage(stage: str, attack: Attack, eligible: np.ndarray) -> bool:
        """Attack the eligible, not yet processed points; return True when complete."""
        done = arr[f"{stage}_done"]
        done |= ~eligible  # ineligible points are settled (not robust)
        todo = np.flatnonzero(~done)
        if todo.size == 0:
            return True
        log(f"{stage}: {todo.size} points to attack")
        for start in range(0, todo.size, args.batch_size):
            if out_of_time():
                state.save()
                return False
            idx = todo[start : start + args.batch_size]
            t0 = time.perf_counter()
            xb = to_float(x[idx]).to(device)
            yb = y[idx].to(device)
            adv = attack.perturb(model, xb, yb)
            check_threat_model(xb, adv, EPS, "linf")
            with torch.no_grad():
                still = (model(adv).argmax(1) == yb).cpu().numpy()
            if device.type == "mps":
                torch.mps.synchronize()
            arr[stage][idx] = still
            done[idx] = True
            state.add_seconds(stage, time.perf_counter() - t0)
            state.note_linf(stage, float((adv - xb).abs().amax()))
            state.save()
        return True

    complete = True
    for key, attack in single.items():
        if not run_stage(key, attack, clean):
            complete = False
            break
        log(f"{key}: {arr[key].mean():.4f}")
    if complete:
        survivors = clean.copy()
        for attack in ens_attacks:
            if not run_stage(attack.name, attack, survivors):
                complete = False
                break
            survivors = arr[attack.name] & survivors
            log(f"ensemble after {attack.name}: robust {int(survivors.sum())}/{args.n}")
    if not complete:
        log("time budget reached; rerun to continue")
        return INCOMPLETE

    seconds = state.meta["seconds"]
    max_linf = state.meta["max_linf"]
    assert isinstance(seconds, dict) and isinstance(max_linf, dict)
    rows = [
        summarize(k, arr[k], seconds.get(k, 0.0), {"max_linf": max_linf.get(k, 0.0)})
        for k in single
    ]
    per_sample = {"clean": clean, **{k: arr[k] for k in single}}
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
    if ens_attacks:
        survivors = clean.copy()
        robust_after = {"clean": int(clean.sum())}
        for name in ens_names:
            survivors = survivors & arr[name]
            robust_after[name] = int(survivors.sum())
        per_sample["ensemble"] = survivors
        ens_seconds = {k: seconds.get(k, 0.0) for k in ens_names}
        payload["ensemble"] = {
            "attacks": ens_names,
            "targets": args.targets,
            "square_queries": args.queries,
            "robust_after": robust_after,
            **summarize("ensemble", survivors, sum(ens_seconds.values())),
            "stage_seconds": ens_seconds,
        }
        log(f"ensemble: {survivors.mean():.4f}")
    write_json(RESULTS / "robustness" / f"{args.name}.json", payload)
    np.savez_compressed(RESULTS / "robustness" / f"{args.name}.npz", **per_sample)
    if device.type == "mps":
        torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
