#!/usr/bin/env bash
# RecurrentPPO baseline-v3 seed 0 (decision, 2026-10-07: v3 stays the baseline after the v4.3
# full run; the next learner of the baseline set).  Same protocol as SAC bl3 and PPO bl3:
# 2.5 M steps on the frozen v3 config, extended to 3.0 M (stage 7 continues), then
# train_seed.sh's evaluations (Tier 1, frozen suite, Paper 2 set) on the best checkpoint,
# then test set v4 (report only), as for SAC.  No hand-back starts, no safety layer in training.
#
# CPU use (decision, 2026-10-07): the on-policy learners (PPO, RecurrentPPO) run unpinned on all
# 12 logical CPUs with PyTorch's default threads, as the September RecurrentPPO runs did
# (70-100 steps/s); pinning stays for SAC and TQC only.  Pinned to the two P-cores, this run
# managed 13 steps/s (10 threads) and 21-25 steps/s (4 threads): its LSTM update is one large
# batched computation that uses every core, while the workers sit idle during it.
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
RUN=runs/recurrent_ppo_formulation_seed0_bl3
ID=recurrent_ppos0_bl3
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

python src/baseline_config.py --check --config configs/baseline_v3.json >> "$LOG" 2>&1 || { note "BASELINE CHECK FAILED ($ID)"; exit 1; }
if [ ! -f "$RUN/final_model_2500000.zip" ] && [ ! -f "$RUN/final_model.zip" ]; then
  note "== $ID: training 0 -> 2.5 M (baseline-v3, unpinned)"
  python src/train_formulation.py --config configs/baseline_v3.json --algo recurrent_ppo --seed 0 --tag bl3 \
    > "$RUN.log" 2>&1
  rc=$?
  note "   $ID 2.5 M exit $rc"
  [ $rc -eq 0 ] && [ -f "$RUN/final_model.zip" ] || { note "$ID FAILED (see $RUN.log)"; exit 1; }
fi
if [ ! -f "$RUN/final_model_2500000.zip" ]; then
  cp "$RUN/final_model.zip" "$RUN/final_model_2500000.zip"
  cp "$RUN/final_vecnormalize.pkl" "$RUN/final_vecnormalize_2500000.pkl"
  note "== $ID: 2.5 M -> 3.0 M"
  python src/train_formulation.py --config configs/baseline_v3.json --algo recurrent_ppo --seed 0 --tag bl3 \
    --resume "$RUN" --extend-to 3000000 >> "$RUN.log" 2>&1
  rc=$?
  note "   $ID 3.0 M exit $rc"
  [ $rc -eq 0 ] || { note "$ID FAILED at the extension (see $RUN.log)"; exit 1; }
fi
bash results/train_seed.sh recurrent_ppo 0 configs/baseline_v3.json || { note "$ID evaluations FAILED"; exit 1; }
python tools/tiers/test_set.py --version 4 --model $RUN/best_model.zip --tag $ID --safety off --processes 4 \
  > results/test_set_v4_$ID.log 2>&1
note "   test set v4, $ID exit $? (results/test_set/v4/$ID/summary.txt)"
note "== $ID run done"
