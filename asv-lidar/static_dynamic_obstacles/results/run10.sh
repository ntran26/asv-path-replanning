#!/usr/bin/env bash
# Run 10 (F84, F85): run 8's setup (60/40 port share) + A29's growing v_hold; then its analysis.
cd "$(dirname "$0")/.."
python -c "import sys; sys.path.insert(0,'src'); import constants as c; assert c.CROSSING_PORT_SHARE_TRAINING == 0.60 and c.V_HOLD_GROWS; print('== config ok: share', c.CROSSING_PORT_SHARE_TRAINING, 'v_hold grows', c.V_HOLD_GROWS)" || exit 1
python src/train_formulation.py --algo ppo --timesteps 2000000 --seed 0 --num-envs 10 --eval-freq 200000 --eval-per-class 20 --tag v10 --train-supervisor off --low-speed-start-frac 0.15 --eval-supervisor both > runs/ppo_formulation_seed0_v10.log 2>&1
echo "== run 10 exit $?"
if grep -q Traceback runs/ppo_formulation_seed0_v10.log; then echo "RUN10 FAILED"; exit 1; fi
for m in off on; do
  python tools/tiers/tier1_replay.py --model runs/ppo_formulation_seed0_v10/final_model.zip --tag run10_supervisor_$m --supervisor $m > results/tiers/tier1_run10_supervisor_$m.log 2>&1
  echo "== tier1 run10 $m exit $?"
done
python tools/diagnostics/crossing_diagnosis.py --model runs/ppo_formulation_seed0_v10/final_model.zip --tag run10 > results/crossing_diagnosis_run10.log 2>&1
echo "== crossing diagnosis run10 exit $?"
python tools/diagnostics/standon_speed.py --model runs/ppo_formulation_seed0_v10/final_model.zip --tag run10 > results/standon_speed_run10.log 2>&1
echo "== standon run10 exit $?"
echo "== RUN10 ANALYSIS DONE"
