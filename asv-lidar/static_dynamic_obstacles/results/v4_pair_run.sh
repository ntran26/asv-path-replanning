#!/usr/bin/env bash
# The v4.3 SAC pair (decision, 2026-10-04; planning/BASELINE_V4_PLAN.md section 5e):
# SAC baseline-v3 seed 0 continued from its final 3.0 M model and replay buffer by
# +0.5 M steps, one arm at a time --
#   v4.3 arm: v4's stage 7 + hard-state starts (configs/finetune_v4_pair_v43.json)
#   v3 arm:   v3's stage 7, the control       (configs/finetune_v4_pair_v3.json)
# -- then the paired comparison on formulation v4.2's development sets
# (tools/diagnostics/v4_gates/pair_compare.py) and, for the record, test set v4 on
# each arm's selected model.  Waits for the hard-state pool first.
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
pin() {   # learner on the performance cores, workers on the efficiency cores
  powershell -NoProfile -Command "
    \$procs = Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\";
    \$l = \$procs | Where-Object { \$_.CommandLine -match 'finetune_field' } | Select-Object -First 1;
    if (\$l) { \$p = Get-Process -Id \$l.ProcessId; \$p.ProcessorAffinity = [IntPtr]0x00F; \$p.PriorityClass = 'AboveNormal';
      \$procs | Where-Object { \$_.ParentProcessId -eq \$l.ProcessId } | ForEach-Object { (Get-Process -Id \$_.ProcessId).ProcessorAffinity = [IntPtr]0xFF0 };
      'pinned learner ' + \$l.ProcessId } else { 'no learner found' }"
}
note "== v4.3 SAC pair queued: waiting for the hard-state pool"
until grep -qE "kept in|Traceback" results/hard_starts/harvest.log results/hard_starts/harvest.err 2>/dev/null; do sleep 120; done
if [ ! -f results/hard_starts/pool_v4.pkl ]; then note "   hard-state pool missing: v4.3 SAC pair NOT started"; exit 1; fi
note "   pool: $(grep 'kept in' results/hard_starts/harvest.log | cut -c1-100)"

for arm in v43 v3; do
  RUN=runs/sac_formulation_seed0_bl3_${arm}pair
  note "== v4.3 SAC pair, arm $arm: 3.0 M -> 3.5 M (configs/finetune_v4_pair_${arm}.json)"
  python src/finetune_field.py --from runs/sac_formulation_seed0_bl3 --spec configs/finetune_v4_pair_${arm}.json > $RUN.log 2>&1 &
  PY=$!
  sleep 240
  note "   $(pin)"
  wait $PY
  rc=$?
  note "   arm $arm exit $rc"
  [ $rc -eq 0 ] && [ -f $RUN/best_model.zip ] || { note "V4.3 SAC PAIR ARM $arm FAILED (see $RUN.log)"; exit 1; }
done

python tools/diagnostics/v4_gates/pair_compare.py > results/v4_gates/pair_compare.log 2>&1
note "   pair comparison exit $? (results/v4_gates/pair_summary.txt)"
for arm in v43 v3; do
  RUN=runs/sac_formulation_seed0_bl3_${arm}pair
  python tools/tiers/test_set.py --version 4 --model $RUN/best_model.zip --tag sacs0_bl3_${arm}pair \
    --safety off --processes 4 > results/test_set_v4_sacs0_bl3_${arm}pair.log 2>&1
  note "   test set v4, arm $arm exit $? (results/test_set/v4/sacs0_bl3_${arm}pair/summary.txt)"
done
note "== v4.3 SAC pair done"
