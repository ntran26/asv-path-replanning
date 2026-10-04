#!/usr/bin/env bash
# Gate G4 (decision, 2026-10-03; planning/BASELINE_V4_PLAN.md): after the PPO v3
# run (ppos0_bl3) finishes training and its evaluations, run the two pilot arms
# from its 1.5 M checkpoint -- A: draft v4 at gamma 0.951; B: draft v4 at gamma
# 0.98 -- then evaluate both against that run's own 2.0 M checkpoint (control).
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
pin() {   # learner on the performance cores, workers on the efficiency cores
  powershell -NoProfile -Command "
    \$procs = Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\";
    \$l = \$procs | Where-Object { \$_.CommandLine -match 'pilot.py train' } | Select-Object -First 1;
    if (\$l) { \$p = Get-Process -Id \$l.ProcessId; \$p.ProcessorAffinity = [IntPtr]0x00F; \$p.PriorityClass = 'AboveNormal';
      \$procs | Where-Object { \$_.ParentProcessId -eq \$l.ProcessId } | ForEach-Object { (Get-Process -Id \$_.ProcessId).ProcessorAffinity = [IntPtr]0xFF0 };
      'pinned pilot learner ' + \$l.ProcessId } else { 'no pilot learner found' }"
}
START_LINE=$(wc -l < "$LOG")
note "== G4 pilot watcher armed: waiting for ppos0_bl3 (training + evaluations) to finish"
until tail -n +"$((START_LINE + 1))" "$LOG" | grep -qE "== ppo seed 0 done|PPOS0_BL3 FAILED|ppo SEED 0 FAILED|ppos0_bl3 FAILED"; do sleep 300; done
note "   ppos0_bl3 finished; G4 pilots start"

for spec in "A 0.9509900498999999" "B 0.98"; do
  set -- $spec
  if [ -f "runs/pilot_v4_$1/final_model.zip" ]; then note "   pilot $1 already trained"; continue; fi
  note "== G4 pilot $1: v4 from 1.5 M to 2.0 M, gamma $2"
  python tools/diagnostics/v4_gates/pilot.py train "$1" --gamma "$2" > "runs/pilot_v4_$1.log" 2>&1 &
  PY=$!; sleep 150; note "   $(pin)"; wait $PY; rc=$?
  note "   pilot $1 exit $rc"
  [ $rc -eq 0 ] && [ -f "runs/pilot_v4_$1/final_model.zip" ] || { note "G4 PILOT $1 FAILED (see runs/pilot_v4_$1.log)"; exit 1; }
done
python tools/diagnostics/v4_gates/pilot.py eval A B --processes 3 > results/v4_gates_g4.log 2>&1
note "   G4 evaluation exit $? (results/v4_gates/g4_summary.txt)"
note "== G4 done"
