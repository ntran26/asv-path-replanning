#!/usr/bin/env bash
# Baseline campaign on the frozen formulation baseline-v1 (F93): every learner
# trains on configs/baseline_v1.json and differs in the learner only.
#
#   ALGOS="td3 sac" SEEDS="0 1" bash results/baseline_campaign.sh
#
# Defaults: all five learners, the seeds in the config.  PPO seeds 0 and 1 are
# run 11 (verified identical to baseline-v1), so they are skipped unless
# RERUN_PPO_01=1.  Budget and gradient-step ratio are A26's call -- the config
# records 2 M steps and 1.0 gradient steps per transition (F80: one seed of all
# five ~135 h at 1.0).  Not launched automatically.
cd "$(dirname "$0")/.."
CONFIG=configs/baseline_v1.json
ALGOS=${ALGOS:-"ppo recurrent_ppo td3 sac tqc"}
SEEDS=${SEEDS:-$(python -c "import json; print(' '.join(map(str, json.load(open('$CONFIG'))['campaign']['seeds'])))")}

python src/baseline_config.py --check || { echo "BASELINE CHECK FAILED"; exit 1; }
mkdir -p results/tiers
for A in $ALGOS; do
  for S in $SEEDS; do
    if [ "$A" = ppo ] && [ "$S" -le 1 ] && [ "${RERUN_PPO_01:-0}" != 1 ]; then
      echo "== ppo seed $S: run 11 (runs/ppo_formulation_seed${S}_v11)"; continue
    fi
    RUN=runs/${A}_formulation_seed${S}_bl1
    python src/train_formulation.py --config $CONFIG --algo $A --seed $S --tag bl1 > $RUN.log 2>&1
    echo "== $A seed $S exit $?"
    if grep -qx "Traceback.*" $RUN.log || grep -q "^Traceback" $RUN.log; then echo "$A SEED $S FAILED"; continue; fi
    for m in off on; do
      python tools/tiers/tier1_replay.py --model $RUN/final_model.zip --tag ${A}s${S}_bl1_supervisor_$m \
        --supervisor $m > results/tiers/tier1_${A}s${S}_bl1_supervisor_$m.log 2>&1
    done
    echo "== TIER1 DONE $A seed $S"
  done
done
echo "== CAMPAIGN DONE"
