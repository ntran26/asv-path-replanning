#!/usr/bin/env bash
# Decision (2026-10-04): evaluate the kept SAC 3 M policy on test set v4 once it is
# built.  The 933 episodes v4 shares with v3 (same id, scenario digest and seed)
# reuse the v3 rows; the 67 replacements are run (safety off).
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
note "== SAC on test set v4 queued: waiting for the build"
until grep -qE "test set 4.0: |Traceback|slots still open" results/test_set_v4_build.log results/test_set_v4_build.err 2>/dev/null; do sleep 60; done
if ! grep -q "test set 4.0: " results/test_set_v4_build.log; then note "   test set v4 build FAILED: SAC test not run"; exit 1; fi
python tools/tiers/test_set.py --version 4 --model runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip \
  --tag sacs0_bl3 --safety off --processes 3 > results/test_set_v4_sacs0_bl3.log 2>&1
note "   SAC on test set v4 exit $? (results/test_set/v4/sacs0_bl3/summary.txt)"
note "== SAC on test set v4 done"
