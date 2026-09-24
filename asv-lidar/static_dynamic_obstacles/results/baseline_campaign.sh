#!/usr/bin/env bash
# Baseline campaign on the frozen formulation baseline-v2 (F96, A26): four
# learners x three seeds, 2 M steps each, off-policy at 1.0 gradient step per
# transition, each seed represented by its best development-set checkpoint.
#
#   bash results/baseline_campaign.sh          # start, or continue after a reboot
#   touch runs/CAMPAIGN_STOP                   # stop cleanly after the current run
#
# Safe to re-run at any time: a finished run (final_model.zip) is skipped, a run
# with a checkpoint continues from it (--resume), and a run that died before its
# first checkpoint is set aside and restarted.  **Every run is retrained on
# baseline-v2**: run 11 and the `_bl1` runs were trained on baseline-v1, whose
# being-overtaken draws differ (F96).  Order: seed 0 of all five learners first
# (a first look at each, ~6 days), then seeds 1-2 seed by seed, so the campaign
# can move to a cluster at any point with every learner partly done.
cd "$(dirname "$0")/.."
CONFIG=configs/baseline_v2.json
LOG=results/baseline_campaign.log
STOP=runs/CAMPAIGN_STOP
CKPT=$(python -c "import json; print(json.load(open('$CONFIG'))['campaign']['checkpoint'])")
TAG=$(python -c "import json; print(json.load(open('$CONFIG'))['campaign']['tag'])")
mkdir -p results/tiers
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

python src/baseline_config.py --check >> "$LOG" 2>&1 || { note "BASELINE CHECK FAILED"; exit 1; }
note "== campaign start ($(python -c "import json; print(json.load(open('$CONFIG'))['formulation_digest'])"))"

frozen() {  # $1 run dir, $2 tag -- the evaluation the paper reports (Tier A + Tier B)
  local out=results/frozen_suite/$2
  [ -f "$out/summary.txt" ] && grep -q "^Frozen suite" "$out/summary.txt" 2>/dev/null && return 0
  python tools/tiers/frozen_suite.py --model "$1/$CKPT" --tag $2 --supervisor both     > "$out.log" 2>&1
  note "   frozen suite $2 exit $?"
}

tier1() {   # $1 run dir, $2 tag -- development-set diagnostic, not reported
  for m in off on; do
    local out=results/tiers/tier1_${2}_supervisor_$m
    [ -f "$out.log" ] && grep -q "^Tier 1 (" "$out.log" 2>/dev/null && continue
    python tools/tiers/tier1_replay.py --model "$1/$CKPT" --tag ${2}_supervisor_$m \
      --supervisor $m > "$out.log" 2>&1
    note "   tier 1 $2 supervisor $m exit $?"
  done
}

# JOBS may be overridden for a partial run, e.g.
#   JOBS="ppo:0 recurrent_ppo:0 sac:0 tqc:0" bash results/baseline_campaign.sh
JOBS=${JOBS:-"ppo:0 recurrent_ppo:0 sac:0 tqc:0
      ppo:1 recurrent_ppo:1 sac:1 tqc:1
      ppo:2 recurrent_ppo:2 sac:2 tqc:2"}
for job in $JOBS; do
  A=${job%%:*}; S=${job##*:}
  RUN=runs/${A}_formulation_seed${S}_${TAG}
  if [ -f "$STOP" ]; then note "== stop file found, stopping before $A seed $S"; exit 0; fi
  if [ ! -f "$RUN/final_model.zip" ]; then
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
    if [ $rc -ne 0 ] || [ ! -f "$RUN/final_model.zip" ]; then note "$A SEED $S FAILED (see $RUN.log)"; continue; fi
  fi
  tier1 "$RUN" ${A}s${S}_${TAG}
  frozen "$RUN" ${A}s${S}_${TAG}
done
note "== CAMPAIGN DONE"
