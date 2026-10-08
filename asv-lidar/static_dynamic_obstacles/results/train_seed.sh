#!/usr/bin/env bash
# Train one learner x seed on a frozen formulation (default baseline-v3, the paper's
# formulation), on demand, then evaluate it -- what the campaign did for each run,
# one run at a time (decision, 2026-09-27: the campaign mechanism is gone).  It trains
# to the config's budget (2.5 M for baseline-v3); the extension to 3.0 M is run by
# the per-run wrapper (results/rppo_bl3_run.sh is the pattern).
#
#   bash results/train_seed.sh sac 1 configs/baseline_v3.json
#   bash results/train_seed.sh sac 0 configs/baseline_v2.json   # an earlier frozen config
#
# * A finished run (final_model.zip) is not retrained; its evaluations still run
#   if they are missing.
# * A run with a checkpoint resumes from it (off-policy runs reload their replay
#   buffer from PhD/asv_replay_buffers/).
# * A run that died before its first checkpoint is set aside as
#   <run>_incomplete_<date> and restarted.
# * Then Tier 1 (development set, a diagnostic) and the frozen suite (headline +
#   robustness set) on the run's best development-set checkpoint -- each skipped
#   if already done.
# * baseline-v3 (or any config other than baseline-v2) also runs the Paper 2
#   deployment-layout set, the coupled layouts its stages 6-7 train for.
# * Safety layer (decision, 2026-10-07): every evaluation here runs with the safety
#   layer off, the development evaluations during training included
#   (--eval-safety-override off; reporting only, selection is safety off anyway).
#   Safety-on evaluations run separately, on the test set, when requested.
#
# Events go to results/train_seed.log; training output to runs/<run>.log.
set -u
cd "$(dirname "$0")/.."
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "usage: bash results/train_seed.sh <ppo|recurrent_ppo|sac|tqc> <seed> [config, default configs/baseline_v3.json]"; exit 2; fi
A=$1; S=$2
CONFIG=${3:-configs/baseline_v3.json}          # the paper's formulation (was baseline-v2 until 2026-10-08)
LOG=results/train_seed.log
read -r TAG CKPT PLANNED <<< "$(python -c "
import json; c = json.load(open('$CONFIG'))['campaign']
print(c['tag'], c['checkpoint'], int('$A' in c['algos'] and $S in c['seeds']))")"
RUN=runs/${A}_formulation_seed${S}_${TAG}
ID=${A}s${S}_${TAG}
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

python src/baseline_config.py --check --config $CONFIG >> "$LOG" 2>&1 || { note "BASELINE CHECK FAILED ($A seed $S, $CONFIG)"; exit 1; }
[ "$PLANNED" = 1 ] || note "   note: $A seed $S is outside the planned learners x seeds in $CONFIG"

if [ -f "$RUN/final_model.zip" ]; then
  note "== $A seed $S: already trained"
else
  if [ -d "$RUN" ] && ls "$RUN"/${A}_*_steps.zip > /dev/null 2>&1; then
    note "== $A seed $S: resuming"
    python src/train_formulation.py --config $CONFIG --algo $A --seed $S --tag $TAG \
      --eval-safety-override off --resume "$RUN" >> "$RUN.log" 2>&1
  else
    if [ -d "$RUN" ]; then
      mv "$RUN" "${RUN}_incomplete_$(date +%Y%m%d_%H%M%S)"
      note "   $A seed $S: no checkpoint, set aside and restarted"
    fi
    note "== $A seed $S: training"
    python src/train_formulation.py --config $CONFIG --algo $A --seed $S --tag $TAG \
      --eval-safety-override off > "$RUN.log" 2>&1
  fi
  rc=$?
  note "== $A seed $S exit $rc"
  if [ $rc -ne 0 ] || [ ! -f "$RUN/final_model.zip" ]; then note "$A SEED $S FAILED (see $RUN.log)"; exit 1; fi
fi

mkdir -p results/tiers
for m in off; do
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
  python tools/tiers/frozen_suite.py --model "$RUN/$CKPT" --tag $ID --supervisor off > "$out.log" 2>&1
  note "   frozen suite $ID exit $?"
fi
if [ "$CONFIG" != configs/baseline_v2.json ]; then
  out=results/paper2_set/$ID
  if [ -f "$out/summary.txt" ]; then
    note "   Paper 2 set $ID: already evaluated"
  else
    mkdir -p results/paper2_set
    python tools/tiers/paper2_suite.py --model "$RUN/$CKPT" --tag $ID --safety off > results/paper2_set_$ID.log 2>&1
    note "   Paper 2 set $ID exit $?"
  fi
fi
note "== $A seed $S done"
