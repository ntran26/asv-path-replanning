#!/usr/bin/env bash
# Fine-tune one baseline-v2 run with field layouts mixed in, then test its best
# model on the Paper 2 deployment-layout set and the frozen suite (F106).
#
#   bash results/finetune_field.sh sac 0                                  # v1: 2 M -> 3 M
#   bash results/finetune_field.sh sac 0 configs/finetune_field_v2.json \
#        runs/sac_formulation_seed0_bl2_ftfield1                          # v2: from v1's best
set -u
cd "$(dirname "$0")/.."
A=$1; S=$2
SPEC=${3:-configs/finetune_field_v1.json}
BASE=runs/${A}_formulation_seed${S}_bl2
FROM=${4:-$BASE}
TAG=$(python -c "import json; print(json.load(open('$SPEC'))['tag'])")
RUN=${BASE}_${TAG}
ID=${A}s${S}_bl2_${TAG}
LOG=results/finetune_field.log
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
note "== $ID: fine-tuning ($SPEC, from $FROM)"
python src/finetune_field.py --from $FROM --spec $SPEC > $RUN.log 2>&1
rc=$?; note "   fine-tune exit $rc"
[ $rc -eq 0 ] && [ -f $RUN/best_model.zip ] || { note "$ID FAILED (see $RUN.log)"; exit 1; }
python tools/tiers/paper2_suite.py --model $RUN/best_model.zip --tag $ID > results/paper2_set_$ID.log 2>&1
note "   Paper 2 set exit $?"
python tools/tiers/frozen_suite.py --model $RUN/best_model.zip --tag $ID --supervisor both > results/frozen_suite/$ID.log 2>&1
note "   frozen suite exit $?"
note "== $ID done"
