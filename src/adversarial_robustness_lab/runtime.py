"""Device selection, seeding, memory limits and checkpoints."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from adversarial_robustness_lab.models import build_model


def pick_device(name: str = "auto") -> torch.device:
    """``auto`` uses ``$ARL_DEVICE`` if set, else prefers CUDA, then Apple MPS, then CPU."""
    if name == "auto":
        name = os.environ.get("ARL_DEVICE", "auto")
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on the test machines
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - CI runs on Linux
        return torch.device("mps")
    return torch.device("cpu")


def limit_memory(device: torch.device, fraction: float | None) -> None:
    """Cap the MPS allocator at ``fraction`` of the recommended working set."""
    if fraction is None or device.type != "mps":
        return
    torch.mps.set_per_process_memory_fraction(fraction)  # pragma: no cover - MPS only


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and torch; return a CPU generator for data order and noise."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def synchronize(device: torch.device) -> None:
    """Wait for queued kernels so wall-clock timings are honest."""
    if device.type == "mps":  # pragma: no cover - MPS only
        torch.mps.synchronize()
    elif device.type == "cuda":  # pragma: no cover
        torch.cuda.synchronize()


def save_checkpoint(path: Path | str, model: nn.Module, arch: str, meta: dict[str, Any]) -> None:
    """Save weights plus JSON-serializable metadata (loadable with ``weights_only=True``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"arch": arch, "state_dict": state, "meta": json.dumps(meta)}, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: Path | str, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a checkpoint written by :func:`save_checkpoint`, in eval mode.

    ``weights_only=True`` refuses arbitrary pickled objects, so a tampered checkpoint
    cannot run code on load.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = build_model(str(blob["arch"]))
    model.load_state_dict(blob["state_dict"])
    meta = json.loads(blob["meta"])
    meta["arch"] = blob["arch"]
    return model.to(device).eval(), meta
