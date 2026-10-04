#!/usr/bin/env bash
# PPO baseline-v3 seed 0, filter-free (decision, 2026-10-02): the development
# policy for safety layer v3 and the planned PPO v3 baseline.  Same protocol as
# SAC bl3: 2.5 M on the frozen v3 config, extended to 3.0 M (stage 7 continues),
# then train_seed.sh's evaluations (tier 1, frozen suite, Paper 2 set).  Nothing
# else changes -- no hand-back starts, no safety layer in training.
#
# It first waits for the held hand-back PPO run (ppos0_bl3_hb, started by mistake
# on 2 Oct 02:50) to be stopped, then removes that run's folder.
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
RUN=runs/ppo_formulation_seed0_bl3
ID=ppos0_bl3
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
held() {
  powershell -NoProfile -Command "
    @(Get-CimInstance Win32_Process | Where-Object { \$_.Name -ne 'powershell.exe' -and \$_.CommandLine -match 'handback_run|bl3_hb' }).Count" | tr -d '\r'
}
pin() {   # learner on the performance cores, workers on the efficiency cores
  powershell -NoProfile -Command "
    \$procs = Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\";
    \$l = \$procs | Where-Object { \$_.CommandLine -match 'train_formulation' } | Select-Object -First 1;
    if (\$l) { \$p = Get-Process -Id \$l.ProcessId; \$p.ProcessorAffinity = [IntPtr]0x00F; \$p.PriorityClass = 'AboveNormal';
      \$procs | Where-Object { \$_.ParentProcessId -eq \$l.ProcessId } | ForEach-Object { (Get-Process -Id \$_.ProcessId).ProcessorAffinity = [IntPtr]0xFF0 };
      'pinned learner ' + \$l.ProcessId } else { 'no learner found' }"
}

if [ "$(held)" != "0" ]; then
  note "== $ID armed: waiting for the held hand-back run (ppos0_bl3_hb) to be stopped"
  until [ "$(held)" = "0" ]; do sleep 60; done
fi
if [ -d runs/ppo_formulation_seed0_bl3_hb ]; then
  rm -rf runs/ppo_formulation_seed0_bl3_hb runs/ppo_formulation_seed0_bl3_hb.log runs/tensorboard/ppo_formulation_seed0_bl3_hb_*
  note "   removed the held hand-back run's folder (ppos0_bl3_hb)"
fi

if [ ! -f "$RUN/final_model_2500000.zip" ] && [ ! -f "$RUN/final_model.zip" ]; then
  note "== $ID: training 0 -> 2.5 M (filter-free, baseline-v3)"
  python src/train_formulation.py --config configs/baseline_v3.json --algo ppo --seed 0 --tag bl3 > "$RUN.log" 2>&1 &
  PY=$!; sleep 150; note "   $(pin)"; wait $PY; rc=$?
  note "   $ID 2.5 M exit $rc"
  [ $rc -eq 0 ] && [ -f "$RUN/final_model.zip" ] || { note "$ID FAILED (see $RUN.log)"; exit 1; }
fi
if [ ! -f "$RUN/final_model_2500000.zip" ]; then
  cp "$RUN/final_model.zip" "$RUN/final_model_2500000.zip"
  cp "$RUN/final_vecnormalize.pkl" "$RUN/final_vecnormalize_2500000.pkl"
  note "== $ID: 2.5 M -> 3.0 M"
  python src/train_formulation.py --config configs/baseline_v3.json --algo ppo --seed 0 --tag bl3 \
    --resume "$RUN" --extend-to 3000000 >> "$RUN.log" 2>&1 &
  PY=$!; sleep 150; note "   $(pin)"; wait $PY; rc=$?
  note "   $ID 3.0 M exit $rc"
  [ $rc -eq 0 ] || { note "$ID FAILED at the extension (see $RUN.log)"; exit 1; }
fi
exec bash results/train_seed.sh ppo 0 configs/baseline_v3.json
