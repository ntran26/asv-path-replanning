#!/usr/bin/env bash
# The frozen suite (Tier B, suite 3.1) on seed 0 of every learner, run once the
# four seed-0 trainings have finished (your call, 2026-09-23).
#
#   bash results/frozen_seed0.sh
#
# Tier B only -- the default frozen suite. Tier A is the extended set and is not
# run here; add --tiers a,b to a single call when it is wanted.
# PPO seed 0 is included: its earlier result was scored against suite 3.0 and is
# set aside as results/frozen_suite/ppos0_bl2_suite30_superseded.
cd "$(dirname "$0")/.."
LOG=results/frozen_seed0.log
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

python src/baseline_config.py --check >> "$LOG" 2>&1 || { note "BASELINE CHECK FAILED"; exit 1; }
for A in ppo recurrent_ppo sac tqc; do
  RUN=runs/${A}_formulation_seed0_bl2
  TAG=${A}s0_bl2
  if [ ! -f "$RUN/best_model.zip" ]; then note "== $A seed 0: no best_model.zip, skipped"; continue; fi
  if [ -f "results/frozen_suite/$TAG/summary.txt" ]; then note "== $TAG: already evaluated"; continue; fi
  note "== $TAG: frozen suite (Tier B, both supervisor modes)"
  python tools/tiers/frozen_suite.py --model "$RUN/best_model.zip" --tag "$TAG" \
    --supervisor both > "results/frozen_suite/$TAG.log" 2>&1
  note "   exit $?"
done
note "== FROZEN SUITE DONE"
