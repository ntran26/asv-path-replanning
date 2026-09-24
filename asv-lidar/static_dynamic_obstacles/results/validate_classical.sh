#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for C in colregs_vo los_dwa; do
  echo "== $(date '+%H:%M:%S') validating $C"
  python tools/diagnostics/tune_classical.py --controller $C --validate --processes 10 \
    > results/classical_tuning/${C}_validate.log 2>&1
  echo "== $(date '+%H:%M:%S') $C exit $?"
done
echo "== VALIDATION DONE"
