from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from adversarial_robustness_lab.data import ImageSplit
from adversarial_robustness_lab.models import SmallCNN
from adversarial_robustness_lab.runtime import (
    limit_memory,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    seed_everything,
    synchronize,
)
from adversarial_robustness_lab.training import (
    METHODS,
    TrainConfig,
    learning_rate,
    trades_adversary,
    train,
    write_history,
)

CPU = torch.device("cpu")


def test_onecycle_schedule() -> None:
    cfg = TrainConfig(lr_max=0.2)
    assert learning_rate(cfg, 0.0) == 0.0
    assert learning_rate(cfg, 0.4) == pytest.approx(0.2)
    assert learning_rate(cfg, 0.2) == pytest.approx(0.1)
    assert learning_rate(cfg, 0.7) == pytest.approx(0.1)
    assert learning_rate(cfg, 1.0) == 0.0


def test_piecewise_and_constant_schedules() -> None:
    cfg = TrainConfig(lr_max=0.1, schedule="piecewise")
    assert [learning_rate(cfg, p) for p in (0.1, 0.6, 0.9)] == pytest.approx([0.1, 0.01, 0.001])
    assert learning_rate(TrainConfig(lr_max=0.3, schedule="constant"), 0.9) == 0.3


@pytest.mark.parametrize(
    "cfg",
    [TrainConfig(method="boost"), TrainConfig(schedule="cosine"), TrainConfig(epochs=0)],
)
def test_config_validation(cfg: TrainConfig) -> None:
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize("method", METHODS)
def test_every_method_trains(
    method: str, train_split: ImageSplit, test_split: ImageSplit, tmp_path: Path
) -> None:
    cfg = TrainConfig(
        arch="small_cnn",
        method=method,
        epochs=1,
        batch_size=64,
        lr_max=0.05,
        steps=2,
        max_train=128,
        eval_clean=32,
        eval_robust=16,
        seed=1,
    )
    logs: list[str] = []
    model, history = train(cfg, train_split, test_split, CPU, tmp_path / "m.pt", log=logs.append)
    assert len(history) == 1 and history[0].clean_acc is not None
    assert (history[0].pgd_acc is not None) == (method in ("pgd", "trades", "fgsm_rs"))
    assert history[0].images_per_second > 0
    assert logs and (tmp_path / "m.pt").exists()
    loaded, meta = load_checkpoint(tmp_path / "m.pt")
    assert meta["config"]["method"] == method and meta["arch"] == "small_cnn"
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        torch.testing.assert_close(a.cpu(), b)


def test_resume_continues_from_checkpoint(
    train_split: ImageSplit, test_split: ImageSplit, tmp_path: Path
) -> None:
    out = tmp_path / "r.pt"
    cfg = TrainConfig(arch="small_cnn", epochs=1, batch_size=64, max_train=128, eval_clean=16)
    train(cfg, train_split, test_split, CPU, out)
    cfg2 = TrainConfig(arch="small_cnn", epochs=2, batch_size=64, max_train=128, eval_clean=16)
    logs: list[str] = []
    _, history = train(cfg2, train_split, test_split, CPU, out, resume=True, log=logs.append)
    assert [h.epoch for h in history] == [1, 2]
    assert logs[0].startswith("resumed")


def test_training_reduces_loss(train_split: ImageSplit, test_split: ImageSplit) -> None:
    cfg = TrainConfig(
        arch="small_cnn",
        epochs=3,
        batch_size=64,
        lr_max=0.05,
        max_train=256,
        eval_clean=64,
        augment=False,
    )
    _, history = train(cfg, train_split, test_split, CPU)
    assert history[-1].train_loss < history[0].train_loss


def test_trades_adversary_stays_in_ball(
    trained_cnn: torch.nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, _ = batch
    x_adv = trades_adversary(trained_cnn, x, 8 / 255, 2 / 255, 3, torch.Generator().manual_seed(0))
    assert float((x_adv - x).abs().max()) <= 8 / 255 + 1e-6
    assert float(x_adv.min()) >= 0 and float(x_adv.max()) <= 1


def test_write_history(tmp_path: Path) -> None:
    cfg = TrainConfig(arch="small_cnn")
    write_history(tmp_path / "h.json", cfg, SmallCNN(), [])
    payload = json.loads((tmp_path / "h.json").read_text())
    assert payload["parameters"] == 591_274 and payload["history"] == []


def test_runtime_helpers(tmp_path: Path) -> None:
    assert pick_device("cpu").type == "cpu"
    assert pick_device().type in ("cpu", "mps", "cuda")
    limit_memory(CPU, 0.5)
    synchronize(CPU)
    g = seed_everything(3)
    assert isinstance(g, torch.Generator)
    model = SmallCNN()
    save_checkpoint(tmp_path / "a" / "b.pt", model, "small_cnn", {"k": 1})
    loaded, meta = load_checkpoint(tmp_path / "a" / "b.pt")
    assert meta == {"k": 1, "arch": "small_cnn"} and not loaded.training


def test_checkpoint_refuses_pickled_code(tmp_path: Path) -> None:
    import pickle

    class Evil:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            import os

            return (os.getcwd, ())  # inert stand-in for a malicious call

    path = tmp_path / "evil.pt"
    torch.save({"arch": "small_cnn", "state_dict": {}, "meta": Evil()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)
