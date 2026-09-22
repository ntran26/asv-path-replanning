#!/usr/bin/env bash
# Run 12 (F91, A31): run 11 + no held-heading charge where the compliant turn has
# no room + stage 3 drawing crossings as often as head-ons.  Two seeds, because
# the seed spread is 0.05 on the headline and 0.20 in crossings (F90).
cd "$(dirname "$0")/.."
python -c "import sys; sys.path.insert(0,'src'); import constants as c; assert c.V_PORT_HEADING_NEEDS_ADMISSIBLE and c.CURRICULUM_STAGES[3].get('weights') and c.V_PORT_LATCHED_RHO and c.V_HOLD_GROWS and c.CROSSING_PORT_SHARE_TRAINING==0.60; print('== config ok')" || exit 1
python tools/scale_audit.py --episodes 240 --workers 10 --out results/scale_audit_f91.json > results/scale_audit_f91.log 2>&1
echo "== audit exit $?"
for S in 0 1; do
  python src/train_formulation.py --algo ppo --timesteps 2000000 --seed $S --num-envs 10 --eval-freq 200000 --eval-per-class 20 --tag v12 --train-supervisor off --low-speed-start-frac 0.15 --eval-supervisor both > runs/ppo_formulation_seed${S}_v12.log 2>&1
  echo "== run 12 seed $S exit $?"
  if grep -q Traceback runs/ppo_formulation_seed${S}_v12.log; then echo "RUN12 SEED $S FAILED"; continue; fi
  M=runs/ppo_formulation_seed${S}_v12/final_model.zip
  T=run12s${S}
  for m in off on; do
    python tools/tiers/tier1_replay.py --model $M --tag ${T}_supervisor_$m --supervisor $m > results/tiers/tier1_${T}_supervisor_$m.log 2>&1
  done
  python tools/diagnostics/crossing_diagnosis.py --model $M --tag $T > results/crossing_diagnosis_${T}.log 2>&1
  python tools/diagnostics/standon_speed.py --model $M --tag $T > results/standon_speed_${T}.log 2>&1
  python tools/diagnostics/swerve_timing.py --model $M --tag $T > results/swerve_timing_${T}.log 2>&1
  echo "== ANALYSIS DONE seed $S"
done
python tools/diagnostics/channel_headon.py --models runs/ppo_formulation_seed0_v8/final_model.zip=run8 runs/ppo_formulation_seed0_v11/final_model.zip=run11s0 runs/ppo_formulation_seed0_v12/final_model.zip=run12s0 runs/ppo_formulation_seed1_v12/final_model.zip=run12s1 > results/channel_headon_run12.log 2>&1
echo "== channel head-on exit $?"
echo "== RUN12 ALL DONE"
