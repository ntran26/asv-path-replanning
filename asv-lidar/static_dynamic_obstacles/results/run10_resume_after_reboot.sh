#!/usr/bin/env bash
# Use ONLY if run 10's suspended processes are gone (reboot or sign-out) -- otherwise
# resume them in place:  powershell -ExecutionPolicy Bypass -File tools\pause_run.ps1 -Tag v10 -Action resume
# Continues run 10 from its latest checkpoint (1.25 M steps at the pause), then analyses it.
cd "$(dirname "$0")/.."
python src/train_formulation.py --resume runs/ppo_formulation_seed0_v10 >> runs/ppo_formulation_seed0_v10.log 2>&1
echo "== run 10 exit $?"
grep -q Traceback runs/ppo_formulation_seed0_v10.log && { echo "RUN10 FAILED"; exit 1; }
for m in off on; do
  python tools/tiers/tier1_replay.py --model runs/ppo_formulation_seed0_v10/final_model.zip --tag run10_supervisor_$m --supervisor $m > results/tiers/tier1_run10_supervisor_$m.log 2>&1
done
python tools/diagnostics/crossing_diagnosis.py --model runs/ppo_formulation_seed0_v10/final_model.zip --tag run10 > results/crossing_diagnosis_run10.log 2>&1
python tools/diagnostics/standon_speed.py --model runs/ppo_formulation_seed0_v10/final_model.zip --tag run10 > results/standon_speed_run10.log 2>&1
echo "== RUN10 ANALYSIS DONE"
