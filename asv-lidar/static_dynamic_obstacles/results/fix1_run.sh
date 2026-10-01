#!/usr/bin/env bash
# Fix 1 (your call, 2026-10-01; configs/finetune_v3_fix1.json): static obstacles in
# the compliant-turn admissibility test, from 2.5 M to 3.0 M, then the frozen
# suite and the Paper 2 set on its best model, for comparison with the kept 3 M
# policy (runs/sac_formulation_seed0_bl3/kept_best_3M, sacs0_bl3).
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
RUN=runs/sac_formulation_seed0_bl3_fix1
ID=sacs0_bl3_fix1
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
pin() {
  powershell -NoProfile -Command "
    \$procs = Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\";
    \$l = \$procs | Where-Object { \$_.CommandLine -match 'finetune_field' } | Select-Object -First 1;
    if (\$l) { \$p = Get-Process -Id \$l.ProcessId; \$p.ProcessorAffinity = [IntPtr]0x00F; \$p.PriorityClass = 'AboveNormal';
      \$procs | Where-Object { \$_.ParentProcessId -eq \$l.ProcessId } | ForEach-Object { (Get-Process -Id \$_.ProcessId).ProcessorAffinity = [IntPtr]0xFF0 };
      'pinned learner ' + \$l.ProcessId } else { 'no learner found' }"
}
note "== $ID: fix 1 from 2.5 M to 3.0 M (configs/finetune_v3_fix1.json)"
python src/finetune_field.py --from runs/sac_formulation_seed0_bl3 --spec configs/finetune_v3_fix1.json > $RUN.log 2>&1 &
PY=$!
sleep 150
note "   $(pin)"
wait $PY
rc=$?
note "   fix-1 fine-tune exit $rc"
[ $rc -eq 0 ] && [ -f $RUN/best_model.zip ] || { note "$ID FAILED (see $RUN.log)"; exit 1; }
mkdir -p results/frozen_suite results/paper2_set
python tools/tiers/frozen_suite.py --model $RUN/best_model.zip --tag $ID --supervisor both > results/frozen_suite/$ID.log 2>&1
note "   frozen suite $ID exit $?"
python tools/tiers/paper2_suite.py --model $RUN/best_model.zip --tag $ID > results/paper2_set_$ID.log 2>&1
note "   Paper 2 set $ID exit $?"
note "== $ID done"
