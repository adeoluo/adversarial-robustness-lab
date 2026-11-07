"""Shared paths and helpers for the full-scale benchmark scripts."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import time
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


INCOMPLETE = 3


class Parts:
    """Resumable multi-part benchmark: finished parts are kept in ``$ARL_DATA_DIR/runs``.

    A script computes each part only if it is not saved yet and calls ``out_of_time``
    between parts; when the budget is spent it exits with ``INCOMPLETE`` and the next
    run with the same ``config`` continues from the saved parts.
    """

    def __init__(self, key: str, config: dict[str, Any], max_minutes: float) -> None:
        self.path = RUNS / f"{key}.parts.json"
        self.max_minutes = max_minutes
        self.t0 = time.perf_counter()
        saved = read_json(self.path) if self.path.exists() else {}
        self.config = json.loads(json.dumps(config))
        same = saved.get("config") == self.config
        self.parts: dict[str, Any] = saved.get("parts", {}) if same else {}

    def __contains__(self, part: str) -> bool:
        return part in self.parts

    def __getitem__(self, part: str) -> Any:
        return self.parts[part]

    def save(self, part: str, value: Any) -> Any:
        self.parts[part] = json.loads(json.dumps(value))
        write_json(self.path, {"config": self.config, "parts": self.parts})
        return self.parts[part]

    def out_of_time(self) -> bool:
        return (time.perf_counter() - self.t0) / 60 > self.max_minutes
