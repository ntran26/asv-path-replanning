#!/usr/bin/env bash
# Run 11 (F88): run 10's setup (60/40, A29) + v_port weighted by the latched peak risk.
cd "$(dirname "$0")/.."
python -c "import sys; sys.path.insert(0,'src'); import constants as c; assert c.CROSSING_PORT_SHARE_TRAINING == 0.60 and c.V_HOLD_GROWS and c.V_PORT_LATCHED_RHO; print('== config ok')" || exit 1
python tools/scale_audit.py --episodes 240 --workers 10 --out results/scale_audit_f88.json > results/scale_audit_f88.log 2>&1
echo "== audit exit $?"
python src/train_formulation.py --algo ppo --timesteps 2000000 --seed 0 --num-envs 10 --eval-freq 200000 --eval-per-class 20 --tag v11 --train-supervisor off --low-speed-start-frac 0.15 --eval-supervisor both > runs/ppo_formulation_seed0_v11.log 2>&1
echo "== run 11 exit $?"
if grep -q Traceback runs/ppo_formulation_seed0_v11.log; then echo "RUN11 FAILED"; exit 1; fi
M=runs/ppo_formulation_seed0_v11/final_model.zip
for m in off on; do
  python tools/tiers/tier1_replay.py --model $M --tag run11_supervisor_$m --supervisor $m > results/tiers/tier1_run11_supervisor_$m.log 2>&1
  echo "== tier1 run11 $m exit $?"
done
python tools/diagnostics/crossing_diagnosis.py --model $M --tag run11 > results/crossing_diagnosis_run11.log 2>&1; echo "== crossing exit $?"
python tools/diagnostics/standon_speed.py --model $M --tag run11 > results/standon_speed_run11.log 2>&1; echo "== standon exit $?"
python tools/diagnostics/swerve_timing.py --model $M --tag run11 > results/swerve_timing_run11.log 2>&1; echo "== swerve exit $?"
echo "== RUN11 ANALYSIS DONE"
