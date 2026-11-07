"""Command line: ``python -m adversarial_robustness_lab <command>``.

Commands: ``data``, ``info``, ``train``, ``evaluate``, ``autoattack``, ``attack``,
``certify``, ``sanity`` and ``serve``. Epsilons accept fractions such as ``8/255``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adversarial_robustness_lab import __version__
from adversarial_robustness_lab.attacks import ATTACK_NAMES, attack_norm, make_attack
from adversarial_robustness_lab.attacks.autoattack import VERSIONS, AutoAttackEnsemble
from adversarial_robustness_lab.data import (
    ARCHIVE_MD5,
    ARCHIVE_NAME,
    ARCHIVE_URL,
    CLASSES,
    DEFAULT_DATA_DIR,
    ImageSplit,
    load_cifar10,
    md5sum,
    synthetic_split,
)
from adversarial_robustness_lab.evaluation import evaluate_attack
from adversarial_robustness_lab.masking import InputQuantization, bpda_view
from adversarial_robustness_lab.models import ARCHITECTURES, count_parameters
from adversarial_robustness_lab.runtime import (
    limit_memory,
    load_checkpoint,
    pick_device,
    seed_everything,
)
from adversarial_robustness_lab.sanity import run_sanity_checks
from adversarial_robustness_lab.smoothing import ABSTAIN, Smooth, certified_accuracy
from adversarial_robustness_lab.training import METHODS, TrainConfig, train


def parse_eps(text: str) -> float:
    """Parse ``8/255``, ``0.5`` or ``2``-style budgets."""
    try:
        value = float(Fraction(text.strip()))
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"invalid epsilon {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("epsilon must be non-negative")
    return value


def data_dir(args: argparse.Namespace) -> Path:
    return Path(args.data_dir or os.environ.get("ARL_DATA_DIR") or DEFAULT_DATA_DIR)


def test_split(args: argparse.Namespace) -> ImageSplit:
    """The evaluation split: synthetic when ``--synthetic N``, else the CIFAR-10 test set."""
    if getattr(args, "synthetic", 0):
        return synthetic_split(args.synthetic, seed=1)
    split = load_cifar10(data_dir(args)).test
    start = getattr(args, "start", 0)
    n = getattr(args, "n", None) or len(split) - start
    return split.subset(range(start, min(len(split), start + n)))


def _emit(payload: Any, as_json: bool, out: Path | None = None) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    if as_json:
        print(json.dumps(payload, indent=2))


def cmd_data(args: argparse.Namespace) -> int:
    root = data_dir(args)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / ARCHIVE_NAME
    if not archive.exists():
        print(f"downloading {ARCHIVE_URL} -> {archive}")
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
    digest = md5sum(archive)
    if digest != ARCHIVE_MD5:
        print(f"MD5 mismatch: {digest} != {ARCHIVE_MD5}", file=sys.stderr)
        return 1
    data = load_cifar10(root)
    print(
        f"CIFAR-10 ready in {root}: {len(data.train)} train, {len(data.test)} test (MD5 {digest})"
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    device = pick_device(args.device)
    info = {
        "version": __version__,
        "torch": torch.__version__,
        "device": str(device),
        "architectures": {k: count_parameters(f(10)) for k, f in ARCHITECTURES.items()},
        "attacks": list(ATTACK_NAMES),
        "training_methods": list(METHODS),
        "ensembles": list(VERSIONS),
    }
    if args.checkpoint:
        _, meta = load_checkpoint(args.checkpoint)
        info["checkpoint"] = {
            "arch": meta["arch"],
            "method": meta.get("config", {}).get("method"),
            "epochs": len(meta.get("history", [])),
            "train_seconds": round(meta.get("train_seconds", 0.0), 1),
        }
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for key, value in info.items():
            print(f"{key:>18}: {value}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = TrainConfig(
        arch=args.arch,
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_max=args.lr,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        beta=args.beta,
        sigma=args.sigma,
        seed=args.seed,
        max_train=args.max_train,
        eval_clean=args.eval_clean,
        eval_robust=args.eval_robust,
    )
    device = pick_device(args.device)
    limit_memory(device, args.memory_fraction)
    if args.synthetic:
        train_set, test_set = synthetic_split(args.synthetic, 0), synthetic_split(256, 1)
    else:
        data = load_cifar10(data_dir(args))
        train_set, test_set = data.train, data.test
    train(
        cfg,
        train_set,
        test_set,
        device,
        args.out,
        resume=args.resume,
        log=lambda m: print(m, flush=True),
    )
    if args.out:
        print(f"saved {args.out}")
    return 0


def _load(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    device = pick_device(args.device)
    limit_memory(device, getattr(args, "memory_fraction", None))
    model, meta = load_checkpoint(args.checkpoint, device)
    if getattr(args, "quantize_bits", 0):
        model = InputQuantization(model, args.quantize_bits).eval()
        if getattr(args, "bpda", False):
            model = bpda_view(model).eval()
    return model, meta, device


def cmd_evaluate(args: argparse.Namespace) -> int:
    model, meta, _ = _load(args)
    split = test_split(args)
    x, y = split.tensors("cpu", as_uint8=True)
    rows = []
    for name in args.attacks:
        attack = make_attack(name, args.eps, args.steps, args.restarts, args.queries, args.seed)
        res = evaluate_attack(model, attack, x, y, args.eps, attack_norm(name), args.batch_size)  # type: ignore[arg-type]
        rows.append(res.summary())
        lo, hi = res.robust_ci()
        if not args.json:
            print(
                f"{name:>9}  eps {args.eps:.4f}  clean {res.clean_accuracy:.4f}  robust "
                f"{res.robust_accuracy:.4f} [{lo:.4f}, {hi:.4f}]  {res.seconds:.1f}s  n={res.n}"
            )
    _emit(
        {"checkpoint": str(args.checkpoint), "arch": meta["arch"], "results": rows},
        args.json,
        args.out,
    )
    return 0


def cmd_autoattack(args: argparse.Namespace) -> int:
    model, meta, device = _load(args)
    split = test_split(args)
    x, y = split.tensors(device)
    ensemble = AutoAttackEnsemble.from_version(
        args.eps,
        args.norm,
        args.version,
        steps=args.steps,
        n_target_classes=args.targets,
        square_queries=args.queries,
        batch_size=args.batch_size,
    )
    res = ensemble.run(model, x, y, progress=None if args.json else print)
    payload = {
        "checkpoint": str(args.checkpoint),
        "arch": meta["arch"],
        "version": args.version,
        "eps": args.eps,
        "norm": args.norm,
        "n": int(x.shape[0]),
        "clean_accuracy": res.clean_accuracy,
        "robust_accuracy": res.robust_accuracy,
        "robust_after": res.robust_after,
        "seconds": res.seconds,
    }
    if not args.json:
        print(f"clean {res.clean_accuracy:.4f}  robust {res.robust_accuracy:.4f}  (n={x.shape[0]})")
    _emit(payload, args.json, args.out)
    return 0


def cmd_attack(args: argparse.Namespace) -> int:
    model, _, device = _load(args)
    split = test_split(args)
    x, y = split.tensors(device)
    attack = make_attack(args.attack, args.eps, args.steps, 1, args.queries, args.seed)
    x_adv = attack(model, x, y)
    with torch.no_grad():
        p_clean = torch.softmax(model(x), 1)
        p_adv = torch.softmax(model(x_adv), 1)
    rows = []
    for i in range(x.shape[0]):
        delta = (x_adv[i] - x[i]).flatten()
        row = {
            "index": args.start + i,
            "label": CLASSES[int(y[i])] if not args.synthetic else int(y[i]),
            "clean": CLASSES[int(p_clean[i].argmax())],
            "clean_confidence": float(p_clean[i].max()),
            "adversarial": CLASSES[int(p_adv[i].argmax())],
            "adversarial_confidence": float(p_adv[i].max()),
            "linf": float(delta.abs().max()),
            "l2": float(delta.norm()),
            "fooled": bool(p_adv[i].argmax() != y[i]),
        }
        rows.append(row)
        if not args.json:
            mark = "FOOLED" if row["fooled"] else "held"
            print(
                f"#{row['index']:<5} {row['label']:>10} | clean {row['clean']:>10} "
                f"{row['clean_confidence']:.2f} | {args.attack} {row['adversarial']:>10} "
                f"{row['adversarial_confidence']:.2f} | Linf {row['linf'] * 255:.2f}/255 "
                f"L2 {row['l2']:.3f} | {mark}"
            )
    if not args.json:
        fooled = sum(r["fooled"] for r in rows)
        print(f"{fooled}/{len(rows)} fooled by {args.attack} at eps {args.eps:.4f}")
    _emit(rows, args.json, args.out)
    return 0


def cmd_certify(args: argparse.Namespace) -> int:
    model, meta, device = _load(args)
    split = test_split(args)
    sigma = args.sigma if args.sigma is not None else meta.get("config", {}).get("sigma", 0.25)
    smooth = Smooth(model, 10, sigma, args.batch_size, args.seed)
    preds, radii, labels = [], [], []
    for i in range(0, len(split), args.skip):
        x, y = split.subset([i]).tensors(device)
        cert = smooth.certify(x[0], args.n0, args.n_mc, args.alpha)
        preds.append(cert.prediction)
        radii.append(cert.radius)
        labels.append(int(y[0]))
        if not args.json:
            pred = "abstain" if cert.prediction == ABSTAIN else CLASSES[cert.prediction]
            ok = (
                "correct"
                if cert.prediction == int(y[0])
                else "wrong"
                if cert.prediction != ABSTAIN
                else ""
            )
            print(
                f"#{args.start + i:<5} {CLASSES[int(y[0])]:>10} -> {pred:>10}  radius "
                f"{cert.radius:.3f}  pA>={cert.p_lower:.4f}  {cert.seconds:.2f}s {ok}"
            )
    p, r, lab = np.array(preds), np.array(radii), np.array(labels)
    table = {f"{rad:.2f}": certified_accuracy(p, r, lab, rad) for rad in args.radii}
    if not args.json:
        print("certified accuracy: " + "  ".join(f"r={k}: {v:.3f}" for k, v in table.items()))
    _emit(
        {
            "sigma": sigma,
            "n0": args.n0,
            "n": args.n_mc,
            "alpha": args.alpha,
            "certified_accuracy": table,
            "predictions": preds,
            "radii": radii,
            "labels": labels,
        },
        args.json,
        args.out,
    )
    return 0


def cmd_sanity(args: argparse.Namespace) -> int:
    model, _, device = _load(args)
    split = test_split(args)
    x, y = split.tensors(device)
    surrogate = None
    if args.surrogate:
        surrogate, _ = load_checkpoint(args.surrogate, device)
    report = run_sanity_checks(
        model, x, y, args.eps, args.steps, args.queries, surrogate, batch_size=args.batch_size
    )
    if not args.json:
        for c in report.checks:
            print(f"{'PASS' if c.passed else 'FAIL'}  {c.name:<32} {c.detail}")
        verdict = "no sign of gradient masking" if report.passed else "possible gradient masking"
        print(
            f"{sum(c.passed for c in report.checks)}/{len(report.checks)} checks passed: {verdict}"
        )
    _emit(report.as_dict(), args.json, args.out)
    return 0 if report.passed or not args.strict else 2


def cmd_serve(args: argparse.Namespace) -> int:
    from adversarial_robustness_lab.service import serve

    serve(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adversarial-robustness-lab", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, checkpoint: bool = True) -> None:
        if checkpoint:
            p.add_argument("--checkpoint", "-c", type=Path, required=True)
        p.add_argument("--data-dir", type=Path, default=None)
        p.add_argument("--device", default="auto")
        p.add_argument("--memory-fraction", type=float, default=None)
        p.add_argument(
            "--synthetic", type=int, default=0, help="use N synthetic images instead of CIFAR-10"
        )
        p.add_argument("--n", type=int, default=None, help="number of test images (default all)")
        p.add_argument("--start", type=int, default=0, help="first test index")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--json", action="store_true")
        p.add_argument("--out", type=Path, default=None, help="also write the JSON result here")

    p = sub.add_parser("data", help="download and verify CIFAR-10, build the cache")
    p.add_argument("--data-dir", type=Path, default=None)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("info", help="versions, device, architectures and attacks")
    p.add_argument("--checkpoint", "-c", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("train", help="train a model")
    common(p, checkpoint=False)
    p.add_argument("--arch", choices=sorted(ARCHITECTURES), default="preact_resnet18")
    p.add_argument("--method", choices=METHODS, default="standard")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--eps", type=parse_eps, default=8 / 255)
    p.add_argument("--alpha", type=parse_eps, default=2 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--beta", type=float, default=6.0)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--eval-clean", type=int, default=1000)
    p.add_argument("--eval-robust", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="robust accuracy under one or more attacks")
    common(p)
    p.add_argument("--attacks", nargs="+", choices=ATTACK_NAMES, default=["pgd"])
    p.add_argument("--eps", type=parse_eps, default=8 / 255)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--queries", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument(
        "--quantize-bits", type=int, default=0, help="wrap the model in input quantization"
    )
    p.add_argument("--bpda", action="store_true", help="attack the quantized model with BPDA")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("autoattack", help="AutoAttack-style ensemble (worst case per image)")
    common(p)
    p.add_argument("--version", choices=VERSIONS, default="standard")
    p.add_argument("--eps", type=parse_eps, default=8 / 255)
    p.add_argument("--norm", choices=["linf", "l2"], default="linf")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--targets", type=int, default=9)
    p.add_argument("--queries", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=250)
    p.set_defaults(func=cmd_autoattack)

    p = sub.add_parser("attack", help="attack a few test images and show the predictions")
    common(p)
    p.add_argument("--attack", choices=ATTACK_NAMES, default="pgd")
    p.add_argument("--eps", type=parse_eps, default=8 / 255)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--quantize-bits", type=int, default=0)
    p.add_argument("--bpda", action="store_true")
    p.set_defaults(func=cmd_attack, n=8)

    p = sub.add_parser("certify", help="randomized smoothing certificates (Cohen et al.)")
    common(p)
    p.add_argument(
        "--sigma", type=float, default=None, help="default: the sigma the model was trained with"
    )
    p.add_argument("--n0", type=int, default=100)
    p.add_argument("--n-samples", dest="n_mc", type=int, default=10_000)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--skip", type=int, default=1, help="certify every skip-th image")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--radii", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.set_defaults(func=cmd_certify)

    p = sub.add_parser("sanity", help="gradient-masking sanity checks")
    common(p)
    p.add_argument("--eps", type=parse_eps, default=8 / 255)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--surrogate", type=Path, default=None, help="checkpoint for the transfer check")
    p.add_argument("--quantize-bits", type=int, default=0)
    p.add_argument("--bpda", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 2 when a check fails")
    p.set_defaults(func=cmd_sanity)

    p = sub.add_parser("serve", help="HTTP service: classify, attack and certify")
    p.add_argument("--checkpoint", "-c", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--synthetic", type=int, default=0)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed_everything(getattr(args, "seed", 0))
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
