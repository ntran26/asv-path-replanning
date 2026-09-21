#!/usr/bin/env bash
# Run 11 seed 1, continued from its latest checkpoint after the suspended
# processes did not survive an overnight pause (F90).
cd "$(dirname "$0")/.."
python src/train_formulation.py --resume runs/ppo_formulation_seed1_v11 >> runs/ppo_formulation_seed1_v11.log 2>&1
echo "== run 11 seed 1 exit $?"
if grep -q Traceback runs/ppo_formulation_seed1_v11.log; then echo "RUN11S1 FAILED"; exit 1; fi
M=runs/ppo_formulation_seed1_v11/final_model.zip
for m in off on; do
  python tools/tiers/tier1_replay.py --model $M --tag run11s1_supervisor_$m --supervisor $m > results/tiers/tier1_run11s1_supervisor_$m.log 2>&1
  echo "== tier1 run11s1 $m exit $?"
done
python tools/diagnostics/crossing_diagnosis.py --model $M --tag run11s1 > results/crossing_diagnosis_run11s1.log 2>&1; echo "== crossing exit $?"
python tools/diagnostics/standon_speed.py --model $M --tag run11s1 > results/standon_speed_run11s1.log 2>&1; echo "== standon exit $?"
python tools/diagnostics/swerve_timing.py --model $M --tag run11s1 > results/swerve_timing_run11s1.log 2>&1; echo "== swerve exit $?"
echo "== RUN11S1 ANALYSIS DONE"
