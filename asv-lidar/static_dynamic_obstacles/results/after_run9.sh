#!/usr/bin/env bash
# Queued analysis for run 9 (F84).
cd "$(dirname "$0")/.."
until grep -qE "done in|Traceback" runs/ppo_formulation_seed0_v9.log; do sleep 120; done
if grep -q Traceback runs/ppo_formulation_seed0_v9.log; then echo "RUN9 FAILED"; exit 1; fi
echo "== run 9 done"
for m in off on; do
  python tools/tiers/tier1_replay.py --model runs/ppo_formulation_seed0_v9/final_model.zip --tag run9_supervisor_$m --supervisor $m > results/tiers/tier1_run9_supervisor_$m.log 2>&1
  echo "== tier1 run9 $m exit $?"
done
python tools/diagnostics/crossing_diagnosis.py --model runs/ppo_formulation_seed0_v9/final_model.zip --tag run9 > results/crossing_diagnosis_run9.log 2>&1
echo "== crossing diagnosis run9 exit $?"
echo "== ANALYSIS DONE"
