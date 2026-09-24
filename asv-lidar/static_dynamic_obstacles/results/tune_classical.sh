#!/usr/bin/env bash
# C3a: tune both comparators on the development set while the campaign is paused.
cd "$(dirname "$0")/.."
for C in colregs_vo los_dwa; do
  echo "== $(date '+%H:%M:%S') tuning $C"
  python tools/diagnostics/tune_classical.py --controller $C --per-class 10 --processes 10 \
    > results/classical_tuning/$C.log 2>&1
  echo "== $(date '+%H:%M:%S') $C exit $?"
done
echo "== TUNING DONE"
