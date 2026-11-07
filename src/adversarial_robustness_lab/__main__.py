"""Entrypoint: ``python -m adversarial_robustness_lab``."""

from __future__ import annotations

import sys

from adversarial_robustness_lab.cli import main

if __name__ == "__main__":
    sys.exit(main())
