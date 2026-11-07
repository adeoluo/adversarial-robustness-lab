#!/usr/bin/env bash
# Train the zoo one run at a time (each run resumes from its last epoch checkpoint).
#   scripts/train_queue.sh wrn16_2_standard wrn16_2_pgd ...
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS="${ARL_DATA_DIR:-../.bench-data/adversarial-robustness-lab}/runs"
mkdir -p "$RUNS"
for name in "$@"; do
  echo "$(date '+%F %T') start $name"
  uv run --no-sync python benchmarks/train_zoo.py "$name" --resume >> "$RUNS/$name.log" 2>&1
  echo "$(date '+%F %T') done $name (exit $?)"
done
