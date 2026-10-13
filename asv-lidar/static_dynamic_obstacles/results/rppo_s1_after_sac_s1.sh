#!/usr/bin/env bash
# One-off (request, 2026-10-13): once SAC seed 1's run is done (training, evaluations and
# test set v4), train RecurrentPPO seed 1 with train_asv.py -- the script's first full
# use -- then its evaluations (train_seed.sh, which skips the finished training) and
# test set v4, as results/run_bl3_3m.sh does.  Never edited while it runs; removed after.
#
#   bash results/rppo_s1_after_sac_s1.sh
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
A=recurrent_ppo; S=1
ID=${A}s${S}_bl3
RUN=runs/${A}_formulation_seed${S}_bl3
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
busy() {   # training or evaluation processes still running
  powershell -NoProfile -Command "
    @(Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\" | Where-Object { \$_.CommandLine -match 'train_formulation.py|train_asv.py|test_set.py|frozen_suite.py|paper2_suite.py|tier1_replay.py' }).Count" | tr -d '\r'
}

note "== $ID armed: train_asv.py starts once sacs1_bl3 has finished"
until grep -q "== sacs1_bl3 run done" "$LOG"; do
  if grep -q "sacs1_bl3.*FAILED" "$LOG"; then note "   $ID not started: sacs1_bl3 failed"; exit 1; fi
  sleep 300
done
until [ "$(busy)" = "0" ]; do sleep 60; done

python train_asv.py --algo $A --seed $S > results/${A}_s${S}_bl3_run.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$RUN/final_model.zip" ]; then note "$ID training via train_asv.py FAILED ($rc; see $RUN.log)"; exit 1; fi
bash results/train_seed.sh $A $S configs/baseline_v3.json || { note "$ID evaluations FAILED"; exit 1; }
python tools/tiers/test_set.py --version 4 --model $RUN/best_model.zip --tag $ID --safety off --processes 4 \
  > results/test_set_v4_$ID.log 2>&1
note "   test set v4, $ID exit $? (results/test_set/v4/$ID/summary.txt)"
note "== $ID run done"
