#!/usr/bin/env bash
# Train one learner x seed on the frozen formulation baseline-v2, on demand, then
# evaluate it -- what the campaign did for each run, one run at a time (your call,
# 2026-09-27: the campaign mechanism is gone).
#
#   bash results/train_seed.sh sac 1
#
# * A finished run (final_model.zip) is not retrained; its evaluations still run
#   if they are missing.
# * A run with a checkpoint resumes from it (off-policy runs reload their replay
#   buffer from PhD/asv_replay_buffers/).
# * A run that died before its first checkpoint is set aside as
#   <run>_incomplete_<date> and restarted.
# * Then Tier 1 (development set, a diagnostic, supervisor off and on) and the
#   frozen suite (headline + robustness set, supervisor off and on) on the run's
#   best development-set checkpoint -- each skipped if already done.
#
# Events go to results/train_seed.log; training output to runs/<run>.log.
set -u
cd "$(dirname "$0")/.."
if [ $# -ne 2 ]; then echo "usage: bash results/train_seed.sh <ppo|recurrent_ppo|sac|tqc> <seed>"; exit 2; fi
A=$1; S=$2
CONFIG=configs/baseline_v2.json
LOG=results/train_seed.log
read -r TAG CKPT PLANNED <<< "$(python -c "
import json; c = json.load(open('$CONFIG'))['campaign']
print(c['tag'], c['checkpoint'], int('$A' in c['algos'] and $S in c['seeds']))")"
RUN=runs/${A}_formulation_seed${S}_${TAG}
ID=${A}s${S}_${TAG}
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

python src/baseline_config.py --check >> "$LOG" 2>&1 || { note "BASELINE CHECK FAILED ($A seed $S)"; exit 1; }
[ "$PLANNED" = 1 ] || note "   note: $A seed $S is outside the planned learners x seeds in $CONFIG"

if [ -f "$RUN/final_model.zip" ]; then
  note "== $A seed $S: already trained"
else
  if [ -d "$RUN" ] && ls "$RUN"/${A}_*_steps.zip > /dev/null 2>&1; then
    note "== $A seed $S: resuming"
    python src/train_formulation.py --config $CONFIG --algo $A --seed $S --tag $TAG \
      --resume "$RUN" >> "$RUN.log" 2>&1
  else
    if [ -d "$RUN" ]; then
      mv "$RUN" "${RUN}_incomplete_$(date +%Y%m%d_%H%M%S)"
      note "   $A seed $S: no checkpoint, set aside and restarted"
    fi
    note "== $A seed $S: training"
    python src/train_formulation.py --config $CONFIG --algo $A --seed $S --tag $TAG > "$RUN.log" 2>&1
  fi
  rc=$?
  note "== $A seed $S exit $rc"
  if [ $rc -ne 0 ] || [ ! -f "$RUN/final_model.zip" ]; then note "$A SEED $S FAILED (see $RUN.log)"; exit 1; fi
fi

mkdir -p results/tiers
for m in off on; do
  out=results/tiers/tier1_${ID}_supervisor_$m
  [ -f "$out.log" ] && grep -q "^Tier 1 (" "$out.log" 2>/dev/null && continue
  python tools/tiers/tier1_replay.py --model "$RUN/$CKPT" --tag ${ID}_supervisor_$m \
    --supervisor $m > "$out.log" 2>&1
  note "   tier 1 $ID supervisor $m exit $?"
done

out=results/frozen_suite/$ID
if [ -f "$out/summary.txt" ] && grep -q "^Frozen suite" "$out/summary.txt"; then
  note "   frozen suite $ID: already evaluated"
else
  mkdir -p results/frozen_suite
  python tools/tiers/frozen_suite.py --model "$RUN/$CKPT" --tag $ID --supervisor both > "$out.log" 2>&1
  note "   frozen suite $ID exit $?"
fi
note "== $A seed $S done"
