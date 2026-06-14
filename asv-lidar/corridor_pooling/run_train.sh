#!/usr/bin/env bash
set -euo pipefail
python train_test_asv.py \
  --mode train \
  --algo sac \
  --timesteps "${1:-1000000}" \
  --num-envs "${NUM_ENVS:-8}" \
  --seed "${SEED:-675973}" \
  --eval-freq "${EVAL_FREQ:-50000}" \
  --save-freq "${SAVE_FREQ:-100000}" \
  --model-path sac_corridor_pooling.zip
