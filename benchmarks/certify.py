"""Randomized smoothing certification on CIFAR-10 (Cohen et al., 2019 protocol).

    uv run python benchmarks/certify.py wrn16_2_noise025 [--skip 20] [--n 10000]

Certifies every ``--skip``-th test image (default every 20th, 500 images; Cohen et al.
certified all 10,000 with n = 100,000, beyond this machine's budget) with
``n0 = 100`` selection samples and up to ``--n`` estimation samples at ``alpha = 0.001``.
Estimation counts are recorded at nested sample sizes (100, 1,000, 10,000), so one run
also gives the certified accuracy as a function of the Monte Carlo budget. Results:
``benchmarks/results/certify/<name>.json``.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from adversarial_robustness_lab.data import load_cifar10
from adversarial_robustness_lab.runtime import limit_memory, load_checkpoint, pick_device
from adversarial_robustness_lab.smoothing import (
    ABSTAIN,
    Smooth,
    certified_accuracy,
    max_certifiable_radius,
)
from adversarial_robustness_lab.stats import clopper_pearson_lower, norm_ppf
from common import RESULTS, checkpoint, hardware, write_json

RADII = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--skip", type=int, default=20)
    parser.add_argument("--n0", type=int, default=100)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--memory-fraction", type=float, default=0.36)
    args = parser.parse_args(argv)
    device = pick_device()
    limit_memory(device, args.memory_fraction)
    model, meta = load_checkpoint(checkpoint(args.name), device)
    sigma = args.sigma if args.sigma is not None else float(meta["config"]["sigma"])
    smooth = Smooth(model, 10, sigma, args.batch_size, seed=0)
    test = load_cifar10().test
    budgets = [b for b in (100, 1000, 10_000, 100_000) if b <= args.n]
    if args.n not in budgets:
        budgets.append(args.n)
    indices = list(range(0, len(test), args.skip))
    labels, guesses, counts, seconds = [], [], {b: [] for b in budgets}, []
    t_start = time.perf_counter()
    for j, i in enumerate(indices):
        x, y = test.subset([i]).tensors(device)
        t0 = time.perf_counter()
        guess = int(np.argmax(smooth.sample_counts(x[0], args.n0)))
        total = np.zeros(10, dtype=np.int64)
        drawn = 0
        for b in budgets:
            total += smooth.sample_counts(x[0], b - drawn)
            drawn = b
            counts[b].append(int(total[guess]))
        if device.type == "mps":
            torch.mps.synchronize()
        seconds.append(time.perf_counter() - t0)
        labels.append(int(y[0]))
        guesses.append(guess)
        if j % 50 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] {args.name} {j}/{len(indices)} {seconds[-1]:.2f}s",
                flush=True,
            )
    lab = np.array(labels)
    per_budget = {}
    for b in budgets:
        p_lower = np.array([clopper_pearson_lower(k, b, args.alpha) for k in counts[b]])
        preds = np.where(p_lower >= 0.5, np.array(guesses), ABSTAIN)
        radii = np.where(
            p_lower >= 0.5,
            sigma * np.array([norm_ppf(p) if p >= 0.5 else 0.0 for p in p_lower]),
            0.0,
        )
        per_budget[str(b)] = {
            "certified_accuracy": {
                f"{r:.2f}": certified_accuracy(preds, radii, lab, r) for r in RADII
            },
            "abstain_rate": float((preds == ABSTAIN).mean()),
            "max_certifiable_radius": max_certifiable_radius(b, args.alpha, sigma),
            "mean_radius_correct": float(radii[preds == lab].mean())
            if (preds == lab).any()
            else 0.0,
        }
        print(b, per_budget[str(b)]["certified_accuracy"], flush=True)
    write_json(
        RESULTS / "certify" / f"{args.name}.json",
        {
            "name": args.name,
            "arch": meta["arch"],
            "train_sigma": meta["config"]["sigma"],
            "sigma": sigma,
            "n0": args.n0,
            "alpha": args.alpha,
            "images": len(indices),
            "skip": args.skip,
            "budgets": per_budget,
            "seconds_per_image_mean": float(np.mean(seconds)),
            "seconds_total": time.perf_counter() - t_start,
            "labels": labels,
            "guesses": guesses,
            "counts": {str(b): counts[b] for b in budgets},
            "hardware": hardware(),
            "device": str(device),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
