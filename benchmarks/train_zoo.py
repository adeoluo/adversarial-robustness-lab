"""Train the model zoo on the full CIFAR-10 training set (50,000 images).

    uv run python benchmarks/train_zoo.py rn18_standard [--resume]

Each run checkpoints after every epoch to ``$ARL_DATA_DIR/runs/<name>.pt`` and writes
its configuration and per-epoch history (loss, accuracy, seconds) to
``benchmarks/results/train/<name>.json``. Runs are sized for an 8 GB Apple M1 shared
with other work; see docs/EVALUATION_REPORT.md for the budget.
"""

from __future__ import annotations

import argparse
import sys

from adversarial_robustness_lab.data import load_cifar10
from adversarial_robustness_lab.runtime import limit_memory, pick_device
from adversarial_robustness_lab.training import TrainConfig, run_metadata, train
from common import RESULTS, checkpoint, hardware, write_json

EPS = 8 / 255
ZOO: dict[str, TrainConfig] = {
    "small_cnn_standard": TrainConfig(arch="small_cnn", method="standard", epochs=20, lr_max=0.05),
    "rn18_standard": TrainConfig(arch="preact_resnet18", method="standard", epochs=10),
    "wrn16_2_fgsm_rs": TrainConfig(
        arch="wrn_16_2", method="fgsm_rs", epochs=10, lr_max=0.2, eps=EPS
    ),
    "wrn16_2_pgd": TrainConfig(
        arch="wrn_16_2", method="pgd", epochs=10, eps=EPS, alpha=2 / 255, steps=5, lr_max=0.1
    ),
    "wrn16_2_trades": TrainConfig(
        arch="wrn_16_2", method="trades", epochs=10, eps=EPS, alpha=2 / 255, steps=5, beta=6.0
    ),
    "wrn16_2_standard": TrainConfig(arch="wrn_16_2", method="standard", epochs=10),
    "wrn16_2_noise025": TrainConfig(arch="wrn_16_2", method="noise", epochs=10, sigma=0.25),
    "wrn16_2_noise050": TrainConfig(arch="wrn_16_2", method="noise", epochs=10, sigma=0.50),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=sorted(ZOO))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=0.36)
    parser.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    parser.add_argument(
        "--stop-after", type=int, default=None, help="train at most this many epochs, then exit 3"
    )
    args = parser.parse_args(argv)
    cfg = ZOO[args.name]
    if args.epochs is not None:
        cfg.epochs = args.epochs
    device = pick_device()
    limit_memory(device, args.memory_fraction)
    data = load_cifar10()
    model, history = train(
        cfg,
        data.train,
        data.test,
        device,
        checkpoint(args.name),
        resume=args.resume,
        log=lambda m: print(m, flush=True),
        stop_after=args.stop_after,
    )
    payload = run_metadata(cfg, model, history)
    payload["name"] = args.name
    payload["device"] = str(device)
    payload["hardware"] = hardware()
    write_json(RESULTS / "train" / f"{args.name}.json", payload)
    return 0 if len(history) >= cfg.epochs else 3


if __name__ == "__main__":
    sys.exit(main())
