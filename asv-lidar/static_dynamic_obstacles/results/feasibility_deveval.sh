#!/usr/bin/env bash
# Decision (2026-10-04): near-impossible episodes leave the development set too.
# After the test set v4 build has finished, grade the formulation-v4 development
# set (frozen-like 120, field development set 150, the 4.1 extension 60) with the
# seeds the training evaluation gives it (oracle_sets.py deveval).
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
note "== development-set oracle (evaluation seeds) queued: waiting for the test set v4 build"
until grep -qE "test set 4.0: |Traceback|SystemExit|slots still open" results/test_set_v4_build.log results/test_set_v4_build.err 2>/dev/null; do sleep 120; done
note "   test set v4 build: $(grep -E 'test set 4.0: |slots still open' results/test_set_v4_build.log | tail -1 | cut -c1-140)"
python tools/diagnostics/feasibility/oracle_sets.py deveval --processes 6 > results/feasibility/oracle_deveval.log 2>&1
note "   development-set oracle exit $? (results/feasibility/oracle_summary.txt)"
note "== development-set oracle done"
