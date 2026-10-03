#!/usr/bin/env bash
# The user (2026-10-03): run the crossing trace on the development set after G4,
# and find near-impossible episodes so they can be left out of (or kept to a very
# small share of) the test and development sets.
#   1. oracle on the development sets (crossings: full library)
#   2. SAC crossing trace on the development crossings, then its report
#   3. oracle on test set v3 (screening)
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
P=5
note "== feasibility run: oracle (development sets), crossing trace, oracle (test set v3); $P processes"
python tools/diagnostics/feasibility/oracle_sets.py dev --processes $P > results/feasibility/oracle_dev.log 2>&1
note "   oracle development sets exit $? (results/feasibility/oracle_summary.txt)"
python tools/diagnostics/crossing/crossing_trace.py run --processes $P > results/crossing_trace/run.log 2>&1 \
  && python tools/diagnostics/crossing/crossing_trace.py report >> results/crossing_trace/run.log 2>&1
note "   crossing trace exit $? (results/crossing_trace/summary.txt)"
python tools/diagnostics/feasibility/oracle_sets.py test --processes $P > results/feasibility/oracle_test.log 2>&1
note "   oracle test set v3 exit $? (results/feasibility/oracle_summary.txt)"
note "== feasibility run done"
