#!/usr/bin/env bash
# One-off (your call, 2026-09-30): continue baseline-v3 SAC seed 0 from its
# 2.5 M checkpoint to 3.0 M steps (stage 7 continues), then run the evaluations
# train_seed.sh would have run at 2.5 M (tier 1, frozen suite, Paper 2 set) on
# the run's best development-set checkpoint.  train_seed.sh's wrapper was
# detached from the running learner so it would not evaluate at 2.5 M first.
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
RUN=runs/sac_formulation_seed0_bl3
LEARNER=1708                          # the 0 -> 2.5 M learner
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
alive() { tasklist //FI "PID eq $1" //NH 2>/dev/null | grep -q " $1 "; }
pin() {   # learner on the performance cores, workers on the efficiency cores (F: 29 Sep slowdown)
  powershell -NoProfile -Command "
    \$procs = Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\";
    \$l = \$procs | Where-Object { \$_.CommandLine -match 'train_formulation' } | Select-Object -First 1;
    if (\$l) { \$p = Get-Process -Id \$l.ProcessId; \$p.ProcessorAffinity = [IntPtr]0x00F; \$p.PriorityClass = 'AboveNormal';
      \$procs | Where-Object { \$_.ParentProcessId -eq \$l.ProcessId } | ForEach-Object { (Get-Process -Id \$_.ProcessId).ProcessorAffinity = [IntPtr]0xFF0 };
      'pinned learner ' + \$l.ProcessId } else { 'no learner found' }"
}

note "== extend watcher armed: waiting for the 2.5 M run (PID $LEARNER) to finish, then 2.5 M -> 3.0 M"
while alive $LEARNER; do sleep 60; done
if [ ! -f "$RUN/final_model.zip" ]; then
  note "   2.5 M run did not finish cleanly (no final model) -- see $RUN.log; stopping"
  exit 1
fi
[ -f "$RUN/sac_2500000_steps.zip" ] || note "   no 2.5 M checkpoint: resuming from the latest one ($(ls $RUN | grep -o 'sac_[0-9]*_steps.zip' | sort -t_ -k2 -n | tail -1))"
cp "$RUN/final_model.zip" "$RUN/final_model_2500000.zip"
cp "$RUN/final_vecnormalize.pkl" "$RUN/final_vecnormalize_2500000.pkl"
note "   2.5 M finished; final model kept as final_model_2500000.zip; resuming to 3.0 M"
python src/train_formulation.py --config configs/baseline_v3.json --algo sac --seed 0 --tag bl3 \
  --resume "$RUN" --extend-to 3000000 >> "$RUN.log" 2>&1 &
PY=$!
sleep 120
note "   $(pin)"
wait $PY
rc=$?
note "== sac seed 0 (bl3) 2.5 M -> 3.0 M exit $rc"
if [ $rc -ne 0 ]; then note "EXTENSION FAILED (see $RUN.log)"; exit 1; fi
exec bash results/train_seed.sh sac 0 configs/baseline_v3.json
