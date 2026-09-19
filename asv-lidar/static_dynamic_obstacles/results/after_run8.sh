#!/usr/bin/env bash
# Queued analysis for run 8 (F82).
cd "$(dirname "$0")/.."
until grep -qE "done in|Traceback" runs/ppo_formulation_seed0_v8.log; do sleep 120; done
if grep -q Traceback runs/ppo_formulation_seed0_v8.log; then echo "RUN8 FAILED"; exit 1; fi
echo "== run 8 done"
for m in off on; do
  python tools/tiers/tier1_replay.py --model runs/ppo_formulation_seed0_v8/final_model.zip --tag run8_supervisor_$m --supervisor $m > results/tiers/tier1_run8_supervisor_$m.log 2>&1
  echo "== tier1 run8 $m exit $?"
done
python tools/diagnostics/crossing_diagnosis.py --model runs/ppo_formulation_seed0_v8/final_model.zip --tag run8 > results/crossing_diagnosis_run8.log 2>&1
echo "== crossing diagnosis run8 exit $?"
echo "== ANALYSIS DONE"
