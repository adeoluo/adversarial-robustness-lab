#!/usr/bin/env bash
# Run the full-scale evaluations one at a time after training (see docs/EVALUATION_REPORT.md).
#   scripts/eval_queue.sh [wait-for-pid]
set -uo pipefail
cd "$(dirname "$0")/../benchmarks"
LOGS="${ARL_DATA_DIR:-../../.bench-data/adversarial-robustness-lab}/runs"
if [[ $# -gt 0 ]]; then
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi
step() {
  echo "$(date '+%F %T') start $*"
  uv run --no-sync python "$@" >> "$LOGS/eval.log" 2>&1
  echo "$(date '+%F %T') done $* (exit $?)"
}
step robustness.py wrn16_2_standard
step robustness.py small_cnn_standard
step robustness.py rn18_standard
step robustness.py wrn16_2_fgsm_rs
step robustness.py wrn16_2_pgd
step robustness.py wrn16_2_trades
step certify.py wrn16_2_noise025
step certify.py wrn16_2_noise050
step sanity.py
step ablations.py
step compare_torchattacks.py wrn16_2_standard
step compare_torchattacks.py wrn16_2_pgd
