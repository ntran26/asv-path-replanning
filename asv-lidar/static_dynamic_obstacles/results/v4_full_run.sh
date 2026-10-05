#!/usr/bin/env bash
# The full SAC baseline-v4.3 run (decision, 2026-10-05; planning/BASELINE_V4_PLAN.md section 5e):
# SAC seed 0 from scratch, 3 M steps, on configs/baseline_v4.json (fix 1 off, hard-state
# starts on), through results/train_seed.sh (training, then the development set, the frozen
# suite and the Paper 2 set on its best checkpoint), then test set v4 on the same model.
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
pin() {   # learner on the performance cores, workers on the efficiency cores
  powershell -NoProfile -Command "
    \$procs = Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\";
    \$l = \$procs | Where-Object { \$_.CommandLine -match 'train_formulation.py --config configs/baseline_v4' } | Select-Object -First 1;
    if (\$l) { \$p = Get-Process -Id \$l.ProcessId; \$p.ProcessorAffinity = [IntPtr]0x00F; \$p.PriorityClass = 'AboveNormal';
      \$procs | Where-Object { \$_.ParentProcessId -eq \$l.ProcessId } | ForEach-Object { (Get-Process -Id \$_.ProcessId).ProcessorAffinity = [IntPtr]0xFF0 };
      'pinned learner ' + \$l.ProcessId } else { 'no learner found' }"
}
RUN=runs/sac_formulation_seed0_bl4
note "== SAC baseline-v4.3 full run: seed 0, 3 M steps from scratch (configs/baseline_v4.json)"
bash results/train_seed.sh sac 0 configs/baseline_v4.json &
TS=$!
# Pin once the learner and its workers exist (development-set build first), and again after a resume.
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 300
  out=$(pin); echo "$out" | grep -q "pinned" && { note "   $out"; break; }
done
wait $TS
rc=$?
note "   train_seed.sh exit $rc"
[ $rc -eq 0 ] && [ -f $RUN/best_model.zip ] || { note "SAC V4.3 FULL RUN FAILED (see $RUN.log)"; exit 1; }
python tools/tiers/test_set.py --version 4 --model $RUN/best_model.zip --tag sacs0_bl4 --safety off --processes 4 \
  > results/test_set_v4_sacs0_bl4.log 2>&1
note "   test set v4, SAC v4.3 exit $? (results/test_set/v4/sacs0_bl4/summary.txt)"
note "== SAC baseline-v4.3 full run done"
