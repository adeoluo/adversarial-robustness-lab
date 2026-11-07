"""Training: standard, PGD adversarial training, TRADES, fast FGSM training and noise.

Each method minimizes an empirical risk over the training set, differing in the input
the loss is evaluated on:

* ``standard``: ``E[CE(f(x), y)]``.
* ``pgd`` (PGD-AT, Madry et al., 2018): the saddle-point problem
  ``min_theta E[max_{x' in S(x)} CE(f(x'), y)]``, with the inner maximum approximated by
  PGD with a random start. Examples are generated in train mode, as in the reference
  code of Rice et al. (2020).
* ``trades`` (Zhang et al., 2019): ``E[CE(f(x), y) + beta * max_{x' in S(x)} KL(f(x) || f(x'))]``
  trading natural accuracy against the boundary error. The inner maximization starts
  at ``x + 0.001 * N(0, I)`` and runs in eval mode, following the authors' code.
* ``fgsm_rs`` (Wong et al., 2020): one FGSM step of size ``1.25 * eps`` from a uniform
  random start, about two forward-backward passes per batch instead of ``steps + 1``.
* ``noise`` (Cohen et al., 2019): Gaussian data augmentation ``x + sigma * N(0, I)``, the
  base classifier for randomized smoothing.

The learning rate follows a one-cycle schedule (linear warm-up to ``lr_max`` over the
first 40% of steps, then linear decay to zero) or a piecewise schedule (divide by 10 at
50% and 75%). Each epoch is checkpointed so long runs can resume.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from adversarial_robustness_lab.attacks.geometry import project_to_threat
from adversarial_robustness_lab.attacks.pgd import PGD
from adversarial_robustness_lab.data import ImageSplit, augment, iterate_batches, to_float
from adversarial_robustness_lab.models import build_model, count_parameters
from adversarial_robustness_lab.runtime import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    synchronize,
)

Method = Literal["standard", "pgd", "trades", "fgsm_rs", "noise"]
METHODS: tuple[str, ...] = ("standard", "pgd", "trades", "fgsm_rs", "noise")


@dataclass
class TrainConfig:
    """Hyperparameters of one training run (defaults follow the cited papers)."""

    arch: str = "preact_resnet18"
    method: str = "standard"
    epochs: int = 10
    batch_size: int = 128
    lr_max: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    schedule: str = "onecycle"
    eps: float = 8 / 255
    alpha: float = 2 / 255
    steps: int = 10
    beta: float = 6.0
    sigma: float = 0.25
    seed: int = 0
    augment: bool = True
    eval_every: int = 1
    eval_clean: int = 1000
    eval_robust: int = 500
    max_train: int | None = None

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.schedule not in ("onecycle", "piecewise", "constant"):
            raise ValueError("schedule must be onecycle, piecewise or constant")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")


@dataclass
class EpochLog:
    epoch: int
    lr: float
    train_loss: float
    train_acc: float
    seconds: float
    clean_acc: float | None = None
    pgd_acc: float | None = None
    images_per_second: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


def learning_rate(cfg: TrainConfig, progress: float) -> float:
    """Learning rate at ``progress`` in ``[0, 1]`` of the whole run."""
    if cfg.schedule == "constant":
        return cfg.lr_max
    if cfg.schedule == "piecewise":
        if progress < 0.5:
            return cfg.lr_max
        return cfg.lr_max * (0.1 if progress < 0.75 else 0.01)
    peak = 0.4
    if progress <= peak:
        return cfg.lr_max * progress / peak
    return cfg.lr_max * max(0.0, (1.0 - progress) / (1.0 - peak))


def trades_adversary(
    model: nn.Module,
    x: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Inner maximization of TRADES: sign ascent on ``KL(f(x) || f(x'))`` in the L-inf ball."""
    with torch.no_grad():
        p_nat = F.softmax(model(x), dim=1)
    noise = torch.randn(x.shape, generator=generator).to(x.device)
    x_adv = (x + 0.001 * noise).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_nat, reduction="sum")
        (grad,) = torch.autograd.grad(kl, x_adv)
        x_adv = project_to_threat(x, x_adv.detach() + alpha * grad.sign(), eps, "linf")
    return x_adv.detach()


def training_loss(
    cfg: TrainConfig,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss for one batch under ``cfg.method`` and the logits it was computed from."""
    if cfg.method == "standard":
        logits = model(x)
        return F.cross_entropy(logits, y), logits
    if cfg.method == "noise":
        noise = torch.randn(x.shape, generator=generator).to(x.device)
        logits = model(x + cfg.sigma * noise)
        return F.cross_entropy(logits, y), logits
    if cfg.method == "pgd":
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator))
        attack = PGD(cfg.eps, cfg.alpha, cfg.steps, "linf", True, 1, "ce", seed)
        x_adv = attack._run(model, x, y, attack.generator())
        logits = model(x_adv)
        return F.cross_entropy(logits, y), logits
    if cfg.method == "fgsm_rs":
        delta = (2 * torch.rand(x.shape, generator=generator).to(x.device) - 1) * cfg.eps
        x0 = (x + delta).clamp(0, 1).requires_grad_(True)
        (grad,) = torch.autograd.grad(F.cross_entropy(model(x0), y), x0)
        x_adv = project_to_threat(x, x0.detach() + 1.25 * cfg.eps * grad.sign(), cfg.eps, "linf")
        logits = model(x_adv)
        return F.cross_entropy(logits, y), logits
    # TRADES
    model.eval()
    x_adv = trades_adversary(model, x, cfg.eps, cfg.alpha, cfg.steps, generator)
    model.train()
    logits = model(x)
    robust = F.kl_div(
        F.log_softmax(model(x_adv), dim=1), F.softmax(logits, dim=1), reduction="batchmean"
    )
    return F.cross_entropy(logits, y) + cfg.beta * robust, logits


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 200) -> float:
    model.eval()
    correct = 0
    for xb, yb in iterate_batches(x, y, batch_size):
        correct += int((model(to_float(xb)).argmax(1) == yb).sum())
    return correct / max(1, x.shape[0])


def robust_accuracy_pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 10,
    batch_size: int = 125,
) -> float:
    """PGD-``steps`` robust accuracy used for monitoring during training."""
    model.eval()
    attack = PGD(eps, 2 / 255, steps, "linf", True, 1, "ce", 0)
    correct = 0
    for xu, yb in iterate_batches(x, y, batch_size):
        xb = to_float(xu)
        x_adv = attack.perturb(model, xb, yb)
        with torch.no_grad():
            correct += int((model(x_adv).argmax(1) == yb).sum())
    return correct / max(1, x.shape[0])


def train(
    cfg: TrainConfig,
    train_split: ImageSplit,
    test_split: ImageSplit,
    device: torch.device,
    out: Path | str | None = None,
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[nn.Module, list[EpochLog]]:
    """Train ``cfg.arch`` with ``cfg.method``; checkpoint to ``out`` after every epoch."""
    cfg.validate()
    gen = seed_everything(cfg.seed)
    model = build_model(cfg.arch).to(device)
    history: list[EpochLog] = []
    start_epoch = 0
    out_path = Path(out) if out is not None else None
    if resume and out_path is not None and out_path.exists():
        loaded, meta = load_checkpoint(out_path, device)
        model.load_state_dict(loaded.state_dict())
        history = [EpochLog(**h) for h in meta.get("history", [])]
        start_epoch = len(history)
        gen.manual_seed(cfg.seed + start_epoch)
        log(f"resumed {out_path} at epoch {start_epoch}")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr_max,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    if resume and out_path is not None:
        opt_path = out_path.with_suffix(".opt.pt")
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))
    train_set = train_split if cfg.max_train is None else train_split.subset(range(cfg.max_train))
    x_train, y_train = train_set.tensors(device, as_uint8=True)
    x_test, y_test = test_split.tensors(device, as_uint8=True)
    steps_per_epoch = -(-x_train.shape[0] // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        synchronize(device)
        t0 = time.perf_counter()
        loss_sum, correct, seen = 0.0, 0, 0
        lr = cfg.lr_max
        for i, (xb, yb) in enumerate(iterate_batches(x_train, y_train, cfg.batch_size, True, gen)):
            lr = learning_rate(cfg, (epoch * steps_per_epoch + i + 1) / total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            xb = to_float(xb)
            xb = augment(xb, generator=gen) if cfg.augment else xb
            loss, logits = training_loss(cfg, model, xb, yb, gen)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * xb.shape[0]
            correct += int((logits.detach().argmax(1) == yb).sum())
            seen += xb.shape[0]
        synchronize(device)
        seconds = time.perf_counter() - t0
        entry = EpochLog(epoch + 1, lr, loss_sum / seen, correct / seen, seconds)
        entry.images_per_second = seen / seconds
        if (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs:
            n_c = min(cfg.eval_clean, x_test.shape[0])
            entry.clean_acc = accuracy(model, x_test[:n_c], y_test[:n_c])
            if cfg.method in ("pgd", "trades", "fgsm_rs") and cfg.eval_robust > 0:
                n_r = min(cfg.eval_robust, x_test.shape[0])
                entry.pgd_acc = robust_accuracy_pgd(model, x_test[:n_r], y_test[:n_r], cfg.eps)
        history.append(entry)
        log(
            f"epoch {entry.epoch}/{cfg.epochs} lr {lr:.4f} loss {entry.train_loss:.4f} "
            f"train {entry.train_acc:.4f} clean {entry.clean_acc} pgd {entry.pgd_acc} "
            f"{seconds:.0f}s ({entry.images_per_second:.0f} img/s)"
        )
        if out_path is not None:
            meta = run_metadata(cfg, model, history)
            save_checkpoint(out_path, model, cfg.arch, meta)
            torch.save(optimizer.state_dict(), out_path.with_suffix(".opt.pt"))
    model.eval()
    return model, history


def run_metadata(cfg: TrainConfig, model: nn.Module, history: list[EpochLog]) -> dict[str, Any]:
    return {
        "config": asdict(cfg),
        "parameters": count_parameters(model),
        "history": [asdict(h) for h in history],
        "train_seconds": sum(h.seconds for h in history),
        "torch": torch.__version__,
    }


def write_history(path: Path | str, cfg: TrainConfig, model: nn.Module, history: list[EpochLog]) -> None:
    Path(path).write_text(json.dumps(run_metadata(cfg, model, history), indent=2) + "\n")
