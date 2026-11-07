"""Shared paths and helpers for the full-scale benchmark scripts."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from adversarial_robustness_lab.data import DEFAULT_DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
DATA_DIR = Path(os.environ.get("ARL_DATA_DIR", DEFAULT_DATA_DIR))
RUNS = DATA_DIR / "runs"


def checkpoint(name: str) -> Path:
    return RUNS / f"{name}.pt"


def hardware() -> dict[str, Any]:
    chip = platform.processor()
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    return {
        "chip": chip,
        "cpus": os.cpu_count(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
