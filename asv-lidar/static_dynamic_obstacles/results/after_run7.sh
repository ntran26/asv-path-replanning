#!/usr/bin/env bash
# Queued analysis for run 7 (F77), then the TD3/RecurrentPPO throughput gate.
cd "$(dirname "$0")/.."
until grep -qE "done in|Traceback" runs/ppo_formulation_seed0_v7.log; do sleep 120; done
if grep -q Traceback runs/ppo_formulation_seed0_v7.log; then echo "RUN7 FAILED"; exit 1; fi
echo "== run 7 done"
for m in off on; do
  python tools/tiers/tier1_replay.py --model runs/ppo_formulation_seed0_v7/final_model.zip --tag run7_supervisor_$m --supervisor $m > results/tiers/tier1_run7_supervisor_$m.log 2>&1
  echo "== tier1 run7 $m exit $?"
done
python tools/tiers/tier1_replay.py --model runs/ppo_formulation_seed0_v6/final_model.zip --tag run6_on_basin_devset --supervisor off > results/tiers/tier1_run6_on_basin_devset.log 2>&1
echo "== tier1 run6 on the basin dev set exit $?"
python tools/diagnostics/crossing_diagnosis.py --model runs/ppo_formulation_seed0_v7/final_model.zip --tag run7 > results/crossing_diagnosis_run7.log 2>&1
echo "== crossing diagnosis run7 exit $?"
echo "== ANALYSIS DONE"
for spec in "td3 10" "td3 2" "tqc 10" "tqc 2" "recurrent_ppo 0"; do
  set -- $spec
  timeout 1500 python src/train_formulation.py --algo $1 --smoke --timesteps 6000 --num-envs 10 --gradient-steps $2 --torch-threads 4 --eval-freq 100000 --eval-per-class 1 --tag gate_g$2 --train-supervisor off --eval-supervisor off > runs/${1}_gate_g$2_smoke.log 2>&1
  echo "== gate $1 gradient_steps=$2 exit $? $(grep -E 'fps' runs/${1}_gate_g$2_smoke.log | tail -1)"
done
echo "== GATE DONE"
