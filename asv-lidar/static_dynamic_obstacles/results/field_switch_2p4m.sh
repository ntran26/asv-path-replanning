#!/usr/bin/env bash
# One-off (your call, 2026-09-28): read v1's 2.4 M field evaluation.  If its field
# goal rate beats the best so far (0.54), let v1 run to 3 M and then test it;
# otherwise stop v1 and start the stronger fine-tune (configs/finetune_field_v2.json)
# from v1's best checkpoint -- which runs its own tests at the end.
# Replaces v1's wrapper, which was detached so it could not resume a changed script.
cd "$(dirname "$0")/.."
LOG=results/finetune_field.log
V1=runs/sac_formulation_seed0_bl2_ftfield1
LEARNER=17672                    # v1's learner process
BEST=0.54                        # best field goal so far (2.1 M and 2.2 M)
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
alive() { tasklist //FI "PID eq $1" //NH 2>/dev/null | grep -q " $1 "; }

note "== switch watcher armed: v1's field goal at 2.4 M must exceed $BEST"
until grep -q "\[EVAL\] t=2,400,000 supervisor off" $V1.log; do
  alive $LEARNER || { note "   v1's learner exited before 2.4 M -- watcher stops"; exit 1; }
  sleep 60
done
line=$(grep "\[EVAL\] t=2,400,000 supervisor off" $V1.log | tail -1)
field=$(echo "$line" | sed -n 's/.*field goal \([0-9.]*\).*/\1/p')
note "   v1 at 2.4 M: $line"
if python -c "import sys; sys.exit(0 if float('$field') > $BEST else 1)"; then
  note "== v1 improved (field $field > $BEST): v1 continues to 3 M"
  while alive $LEARNER; do sleep 60; done
  ID=sacs0_bl2_ftfield1
  [ -f $V1/final_model.zip ] || { note "$ID did not finish (see $V1.log)"; exit 1; }
  note "   v1 finished; testing its best model"
  python tools/tiers/paper2_suite.py --model $V1/best_model.zip --tag $ID > results/paper2_set_$ID.log 2>&1
  note "   Paper 2 set exit $?"
  python tools/tiers/frozen_suite.py --model $V1/best_model.zip --tag $ID --supervisor both > results/frozen_suite/$ID.log 2>&1
  note "   frozen suite exit $?"
  note "== $ID done"
else
  note "== v1 did not improve (field $field <= $BEST): stopping v1, starting v2 from its best checkpoint"
  taskkill //PID $LEARNER //T //F > /dev/null 2>&1
  sleep 20
  exec bash results/finetune_field.sh sac 0 configs/finetune_field_v2.json $V1
fi
