#!/usr/bin/env bash
# Hand-back starts (decision, 2026-10-01; planning/archive/HANDBACK_STARTS_PLAN.md): after
# the SAC fix-1 fine-tune and its evaluations, collect the states where safety
# layer v2 hands the helm back (dev set + near-deployment layouts, kept 3 M SAC
# policy), then train PPO baseline-v3 seed 0 with 25 % of stage-6/7 episodes
# starting from them (2.5 M, then extended to 3.0 M as SAC was), then the frozen
# suite and the Paper 2 set, safety off and v2 on.
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
POOL=results/handback_starts/pool_v1.pkl
RUN=runs/ppo_formulation_seed0_bl3_hb
ID=ppos0_bl3_hb
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
busy() {   # any fix-1 stage still running?
  powershell -NoProfile -Command "
    @(Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\" |
      Where-Object { \$_.CommandLine -match 'finetune_field|frozen_suite|paper2_suite|fix1' }).Count"
}
pin() {   # learner on the performance cores, workers on the efficiency cores
  powershell -NoProfile -Command "
    \$procs = Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\";
    \$l = \$procs | Where-Object { \$_.CommandLine -match 'train_formulation' } | Select-Object -First 1;
    if (\$l) { \$p = Get-Process -Id \$l.ProcessId; \$p.ProcessorAffinity = [IntPtr]0x00F; \$p.PriorityClass = 'AboveNormal';
      \$procs | Where-Object { \$_.ParentProcessId -eq \$l.ProcessId } | ForEach-Object { (Get-Process -Id \$_.ProcessId).ProcessorAffinity = [IntPtr]0xFF0 };
      'pinned learner ' + \$l.ProcessId } else { 'no learner found' }"
}

note "== $ID watcher armed: waiting for sacs0_bl3_fix1 (training + evaluations) to finish"
until grep -qE "== sacs0_bl3_fix1 done|sacs0_bl3_fix1 FAILED" "$LOG" && [ "$(busy | tr -d '\r')" = "0" ]; do sleep 300; done
note "   fix-1 finished; collecting hand-back states"

if [ ! -f "$POOL" ]; then
  python tools/diagnostics/harvest_handback_starts.py --out "$POOL" --processes 8 > results/handback_starts_harvest.log 2>&1
  rc=$?
  note "   hand-back collection exit $rc ($(python -c "import json;m=json.load(open('${POOL%.pkl}.json'));print(m['records'],'records from',m['episodes'],'episodes')" 2>/dev/null))"
  [ $rc -eq 0 ] && [ -f "$POOL" ] || { note "$ID FAILED at collection (see results/handback_starts_harvest.log)"; exit 1; }
fi

if [ ! -f "$RUN/final_model_2500000.zip" ] && [ ! -f "$RUN/final_model.zip" ]; then
  note "== $ID: training 0 -> 2.5 M (hand-back starts, 25 % from stage 6)"
  python src/train_formulation.py --config configs/baseline_v3.json --algo ppo --seed 0 --tag bl3_hb \
    --start-pool "$POOL" > "$RUN.log" 2>&1 &
  PY=$!; sleep 150; note "   $(pin)"; wait $PY; rc=$?
  note "   $ID 2.5 M exit $rc"
  [ $rc -eq 0 ] && [ -f "$RUN/final_model.zip" ] || { note "$ID FAILED (see $RUN.log)"; exit 1; }
  cp "$RUN/final_model.zip" "$RUN/final_model_2500000.zip"
  cp "$RUN/final_vecnormalize.pkl" "$RUN/final_vecnormalize_2500000.pkl"
fi

note "== $ID: 2.5 M -> 3.0 M"
python src/train_formulation.py --config configs/baseline_v3.json --algo ppo --seed 0 --tag bl3_hb \
  --resume "$RUN" --extend-to 3000000 >> "$RUN.log" 2>&1 &
PY=$!; sleep 150; note "   $(pin)"; wait $PY; rc=$?
note "   $ID 3.0 M exit $rc"
[ $rc -eq 0 ] && [ -f "$RUN/best_model.zip" ] || { note "$ID FAILED at the extension (see $RUN.log)"; exit 1; }

mkdir -p results/frozen_suite
python tools/tiers/frozen_suite.py --model $RUN/best_model.zip --tag $ID --safety both --safety-version 2 \
  > results/frozen_suite/$ID.log 2>&1
note "   frozen suite $ID (safety off / v2) exit $?"
python tools/tiers/paper2_suite.py --model $RUN/best_model.zip --tag $ID --safety both --safety-version 2 \
  > results/paper2_set_$ID.log 2>&1
note "   Paper 2 set $ID (safety off / v2) exit $?"
note "== $ID done"
