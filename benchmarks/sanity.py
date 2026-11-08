"""Gradient-masking sanity checks on the trained models and on a masking defense.

    uv run python benchmarks/sanity.py [--n 1000]

Runs every check of :mod:`adversarial_robustness_lab.sanity` on the PGD-AT and TRADES
models (with the standard model as the transfer surrogate) and on the PGD-AT model
wrapped in 3-bit input quantization, a defense known to shatter gradients. For the
quantized model it then reruns PGD with BPDA (straight-through gradients) to show the
robustness the checks flagged as suspect disappears. Results:
``benchmarks/results/sanity.json``. Each model is one resumable part: after
``--max-minutes`` the script exits with status 3 and a rerun continues.
"""

from __future__ import annotations

import argparse
import sys
import time

from adversarial_robustness_lab.attacks import PGD, SquareAttack
from adversarial_robustness_lab.data import load_cifar10
from adversarial_robustness_lab.evaluation import evaluate_attack
from adversarial_robustness_lab.masking import InputQuantization, bpda_view
from adversarial_robustness_lab.runtime import limit_memory, load_checkpoint, pick_device
from adversarial_robustness_lab.sanity import run_sanity_checks
from common import INCOMPLETE, RESULTS, Parts, checkpoint, fingerprint, hardware, write_json

EPS = 8 / 255


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument(
        "--models", nargs="+", default=["wrn16_2_pgd", "wrn16_2_trades", "wrn16_2_fgsm_rs"]
    )
    parser.add_argument("--surrogate", default="wrn16_2_standard")
    parser.add_argument("--memory-fraction", type=float, default=0.36)
    parser.add_argument("--max-minutes", type=float, default=35.0)
    args = parser.parse_args(argv)
    parts = Parts(
        "sanity",
        {
            "n": args.n,
            "models": args.models,
            "surrogate": args.surrogate,
            "checkpoints": fingerprint(*args.models, args.surrogate),
        },
        args.max_minutes,
    )
    device = pick_device()
    limit_memory(device, args.memory_fraction)
    x, y = load_cifar10().test.subset(range(args.n)).tensors(device)
    surrogate, _ = load_checkpoint(checkpoint(args.surrogate), device)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    results: dict[str, object] = {}
    for name in args.models:
        if not checkpoint(name).exists():
            continue
        if name not in parts:
            if parts.out_of_time():
                return INCOMPLETE
            model, _ = load_checkpoint(checkpoint(name), device)
            report = run_sanity_checks(model, x, y, EPS, 20, 1000, surrogate, progress=log)
            parts.save(name, report.as_dict())
            log(f"{name}: passed={report.passed}")
        results[name] = parts[name]
    quantized = f"{args.models[0]}+quantize3"
    if quantized in parts:
        results[quantized] = parts[quantized]
        return finish(args.n, results)
    if parts.out_of_time():
        return INCOMPLETE
    base, _ = load_checkpoint(checkpoint(args.models[0]), device)
    masked = InputQuantization(base, bits=3).eval()
    report = run_sanity_checks(masked, x, y, EPS, 20, 1000, surrogate, progress=log)
    pgd = evaluate_attack(masked, PGD(EPS, 2 / 255, 20), x, y, EPS)
    bpda = evaluate_attack(bpda_view(masked), PGD(EPS, 2 / 255, 20), x, y, EPS)
    # Score the BPDA examples on the deployed (non-BPDA) model: same forward pass.
    square = evaluate_attack(masked, SquareAttack(EPS, 1000), x, y, EPS)
    results[quantized] = parts.save(
        quantized,
        {
            **report.as_dict(),
            "pgd20": pgd.robust_accuracy,
            "pgd20_bpda": bpda.robust_accuracy,
            "square1000": square.robust_accuracy,
            "clean": pgd.clean_accuracy,
        },
    )
    log(
        f"quantized: PGD {pgd.robust_accuracy:.4f}, BPDA {bpda.robust_accuracy:.4f}, "
        f"Square {square.robust_accuracy:.4f}"
    )
    return finish(args.n, results)


def finish(n: int, results: dict[str, object]) -> int:
    write_json(
        RESULTS / "sanity.json", {"n": n, "eps": EPS, "results": results, "hardware": hardware()}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
