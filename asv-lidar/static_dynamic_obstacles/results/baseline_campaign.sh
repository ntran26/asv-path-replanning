#!/usr/bin/env bash
# Baseline campaign on the frozen formulation baseline-v1 (F93, A26): five
# learners x five seeds, 2 M steps each, off-policy at 1.0 gradient step per
# transition, each seed represented by its best development-set checkpoint.
#
#   bash results/baseline_campaign.sh          # start, or continue after a reboot
#   touch runs/CAMPAIGN_STOP                   # stop cleanly after the current run
#
# Safe to re-run at any time: a finished run (final_model.zip) is skipped, a run
# with a checkpoint continues from it (--resume), and a run that died before its
# first checkpoint is set aside and restarted.  PPO seeds 0-1 are run 11
# (verified identical to baseline-v1).  Order (your call, 2026-09-22): seed 0
# of RecurrentPPO, TD3, SAC and TQC first -- a first look at every learner,
# ~5.5 days -- then PPO seeds 2-4 and RecurrentPPO 1-4, then the off-policy
# seeds 1-4 seed by seed, so the campaign can move to a cluster at any point
# with every learner partly done.
cd "$(dirname "$0")/.."
CONFIG=configs/baseline_v1.json
LOG=results/baseline_campaign.log
STOP=runs/CAMPAIGN_STOP
CKPT=$(python -c "import json; print(json.load(open('$CONFIG'))['campaign']['checkpoint'])")
mkdir -p results/tiers
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

python src/baseline_config.py --check >> "$LOG" 2>&1 || { note "BASELINE CHECK FAILED"; exit 1; }
note "== campaign start ($(python -c "import json; print(json.load(open('$CONFIG'))['formulation_digest'])"))"

tier1() {   # $1 run dir, $2 tag
  for m in off on; do
    local out=results/tiers/tier1_${2}_supervisor_$m
    [ -f "$out.log" ] && grep -q "^Tier 1 (" "$out.log" 2>/dev/null && continue
    python tools/tiers/tier1_replay.py --model "$1/$CKPT" --tag ${2}_supervisor_$m \
      --supervisor $m > "$out.log" 2>&1
    note "   tier 1 $2 supervisor $m exit $?"
  done
}

JOBS="recurrent_ppo:0 td3:0 sac:0 tqc:0 run11
      ppo:2 ppo:3 ppo:4 recurrent_ppo:1 recurrent_ppo:2 recurrent_ppo:3 recurrent_ppo:4
      td3:1 sac:1 tqc:1 td3:2 sac:2 tqc:2 td3:3 sac:3 tqc:3 td3:4 sac:4 tqc:4"
for job in $JOBS; do
  if [ "$job" = run11 ]; then
    # PPO seeds 0-1: run 11, evaluated on the same checkpoint rule.
    tier1 runs/ppo_formulation_seed0_v11 ppos0_bl1
    tier1 runs/ppo_formulation_seed1_v11 ppos1_bl1
    continue
  fi
  A=${job%%:*}; S=${job##*:}
  RUN=runs/${A}_formulation_seed${S}_bl1
  if [ -f "$STOP" ]; then note "== stop file found, stopping before $A seed $S"; exit 0; fi
  if [ ! -f "$RUN/final_model.zip" ]; then
    if [ -d "$RUN" ] && ls "$RUN"/${A}_*_steps.zip > /dev/null 2>&1; then
      note "== $A seed $S: resuming"
      python src/train_formulation.py --config $CONFIG --algo $A --seed $S --tag bl1 \
        --resume "$RUN" >> "$RUN.log" 2>&1
    else
      if [ -d "$RUN" ]; then
        mv "$RUN" "${RUN}_incomplete_$(date +%Y%m%d_%H%M%S)"
        note "   $A seed $S: no checkpoint, set aside and restarted"
      fi
      note "== $A seed $S: training"
      python src/train_formulation.py --config $CONFIG --algo $A --seed $S --tag bl1 > "$RUN.log" 2>&1
    fi
    rc=$?
    note "== $A seed $S exit $rc"
    if [ $rc -ne 0 ] || [ ! -f "$RUN/final_model.zip" ]; then note "$A SEED $S FAILED (see $RUN.log)"; continue; fi
  fi
  tier1 "$RUN" ${A}s${S}_bl1
done
note "== CAMPAIGN DONE"
