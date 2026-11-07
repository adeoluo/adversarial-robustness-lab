"""Entrypoint: `python -m adversarial_robustness_lab`."""

from __future__ import annotations

from adversarial_robustness_lab import __version__
from adversarial_robustness_lab.configuration import settings


def main() -> None:
    header = f"Adversarial Robustness Lab v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()
