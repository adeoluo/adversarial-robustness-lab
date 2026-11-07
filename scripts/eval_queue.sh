#!/usr/bin/env bash
# Run the full-scale training and evaluation steps one at a time (see docs/EVALUATION_REPORT.md).
#   scripts/eval_queue.sh [wait-for-pid]
# Every step is bounded (about 35 minutes); resumable steps exit with status 3 when their
# budget runs out and are rerun until they finish. Other failures (for example an MPS
# out-of-memory error while the GPU is shared) are retried up to three times after a pause.
set -uo pipefail
cd "$(dirname "$0")/../benchmarks"
LOGS="${ARL_DATA_DIR:-../../.bench-data/adversarial-robustness-lab}/runs"
if [[ $# -gt 0 ]]; then
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi
step() {
  local code=3 failures=0
  while [[ $code -ne 0 ]]; do
    echo "$(date '+%F %T') start $*"
    uv run --no-sync python "$@" >> "$LOGS/eval.log" 2>&1
    code=$?
    echo "$(date '+%F %T') done $* (exit $code)"
    if [[ $code -ne 0 && $code -ne 3 ]]; then
      failures=$((failures + 1))
      [[ $failures -ge 3 ]] && return
      sleep 180
    fi
  done
}
step robustness.py wrn16_2_standard
step robustness.py wrn16_2_pgd
step robustness.py wrn16_2_trades
step certify.py wrn16_2_noise025
step certify.py wrn16_2_noise050
for pair in "FGSM" "PGD-20" "PGD-L2-20" "C&W-L2 (c=1, 100 steps)" "APGD-CE-100" "APGD-DLR-100" "Square-1000" "AutoAttack standard"; do
  step compare_torchattacks.py wrn16_2_pgd --only "$pair"
done
step sanity.py
step ablations.py
step robustness.py wrn16_2_fgsm_rs
step robustness.py small_cnn_standard
step train_zoo.py rn18_standard --resume --stop-after 4
step robustness.py rn18_standard
for pair in "FGSM" "PGD-20" "PGD-L2-20" "C&W-L2 (c=1, 100 steps)" "APGD-CE-100" "Square-1000"; do
  step compare_torchattacks.py wrn16_2_standard --only "$pair"
done
