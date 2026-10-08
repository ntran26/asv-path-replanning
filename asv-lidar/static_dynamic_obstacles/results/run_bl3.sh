#!/usr/bin/env bash
# One baseline-v3 paper run, start to finish (2026-10-08):
#
#   bash results/run_bl3.sh <ppo|recurrent_ppo|sac|tqc> <seed> [--after-evaluations]
#
# 1. train 0 -> 2.5 M on configs/baseline_v3.json (resuming from the last checkpoint if
#    the run folder has one), development evaluations with the safety layer off;
# 2. extend 2.5 -> 3.0 M with stage 7 continuing;
# 3. train_seed.sh's evaluations on the best checkpoint (Tier 1, frozen suite, reference
#    layouts), safety layer off;
# 4. test set v4 (report only), safety layer off, with the paper-table metrics.
#
# CPU: PIN=1 pins the learner to the P-cores (0x00F, AboveNormal) and its workers to the
# E-cores (0xFF0); PIN=0 leaves everything on all 12 CPUs. THREADS=n sets PyTorch's thread
# count (default: PyTorch's own). Defaults follow the 2026-10-07 decision (pin SAC/TQC,
# not PPO/RecurrentPPO); results/bench_cpu.sh measures which is faster and passes the winner
# (decision, 2026-10-08: use the faster option). Both are recorded in the run's config.json. --after-evaluations waits until no test-set, frozen-suite,
# reference-layout or Tier 1 evaluation is running, so the run does not share the CPU.
cd "$(dirname "$0")/.."
A=$1; S=$2; WAIT=${3:-}
[ -n "$A" ] && [ -n "$S" ] || { echo "usage: bash results/run_bl3.sh <ppo|recurrent_ppo|sac|tqc> <seed> [--after-evaluations]"; exit 2; }
CONFIG=configs/baseline_v3.json
LOG=results/train_seed.log
RUN=runs/${A}_formulation_seed${S}_bl3
ID=${A}s${S}_bl3
case "$A" in sac|tqc) PIN=${PIN:-1} ;; *) PIN=${PIN:-0} ;; esac
THREADS=${THREADS:-}
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

busy() {   # number of evaluation processes running
  powershell -NoProfile -Command "
    @(Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\" | Where-Object { \$_.CommandLine -match 'test_set.py|frozen_suite.py|paper2_suite.py|tier1_replay.py' }).Count" | tr -d '\r'
}
pin() {   # SAC/TQC: learner on the performance cores, workers on the efficiency cores
  powershell -NoProfile -Command "
    \$procs = Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\";
    \$l = \$procs | Where-Object { \$_.CommandLine -match 'train_formulation.py --config configs/baseline_v3.json --algo $A --seed $S ' } | Select-Object -First 1;
    if (\$l) { \$p = Get-Process -Id \$l.ProcessId; \$p.ProcessorAffinity = [IntPtr]0x00F; \$p.PriorityClass = 'AboveNormal';
      \$procs | Where-Object { \$_.ParentProcessId -eq \$l.ProcessId } | ForEach-Object { (Get-Process -Id \$_.ProcessId).ProcessorAffinity = [IntPtr]0xFF0 };
      'pinned learner ' + \$l.ProcessId } else { 'no learner found' }"
}
pin_when_up() {   # the workers exist once the development set is built
  [ "$PIN" = 1 ] || return 0
  for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 120
    out=$(pin); echo "$out" | grep -q "pinned" && { note "   $out"; return; }
  done
  note "   learner not found for pinning"
}
train() {   # $@: extra trainer arguments; output appended to the run log
  python src/train_formulation.py --config $CONFIG --algo $A --seed $S --tag bl3 \
    --eval-safety-override off ${THREADS:+--torch-threads $THREADS} "$@" >> "$RUN.log" 2>&1 &
  PY=$!; pin_when_up; wait $PY
}

if [ "$WAIT" = "--after-evaluations" ]; then
  if [ "$(busy)" != "0" ]; then
    note "== $ID armed: waiting for the running evaluations to finish"
    until [ "$(busy)" = "0" ]; do sleep 60; done
  fi
fi
python src/baseline_config.py --check --config $CONFIG >> "$LOG" 2>&1 || { note "BASELINE CHECK FAILED ($ID)"; exit 1; }

if [ ! -f "$RUN/final_model_2500000.zip" ] && [ ! -f "$RUN/final_model.zip" ]; then
  if [ -d "$RUN" ] && ls "$RUN"/${A}_*_steps.zip > /dev/null 2>&1; then
    note "== $ID: resuming the 0 -> 2.5 M phase (baseline-v3)"
    train --resume "$RUN"
  else
    [ -d "$RUN" ] && { mv "$RUN" "${RUN}_incomplete_$(date +%Y%m%d_%H%M%S)"; note "   $ID: no checkpoint, set aside"; }
    note "== $ID: training 0 -> 2.5 M (baseline-v3; PIN=$PIN, THREADS=${THREADS:-default})"
    : > "$RUN.log"
    train
  fi
  rc=$?
  note "   $ID 2.5 M exit $rc"
  [ $rc -eq 0 ] && [ -f "$RUN/final_model.zip" ] || { note "$ID FAILED (see $RUN.log)"; exit 1; }
fi
if [ ! -f "$RUN/final_model_2500000.zip" ]; then
  cp "$RUN/final_model.zip" "$RUN/final_model_2500000.zip"
  cp "$RUN/final_vecnormalize.pkl" "$RUN/final_vecnormalize_2500000.pkl"
  note "== $ID: 2.5 M -> 3.0 M"
  train --resume "$RUN" --extend-to 3000000
  rc=$?
  note "   $ID 3.0 M exit $rc"
  [ $rc -eq 0 ] || { note "$ID FAILED at the extension (see $RUN.log)"; exit 1; }
elif ! grep -q "done in" <(tail -n 3 "$RUN.log" 2>/dev/null) && ls "$RUN"/${A}_*_steps.zip > /dev/null 2>&1 \
     && [ "$(python -c "import json; print(json.load(open('$RUN/config.json'))['timesteps'])")" != 3000000 ]; then
  note "== $ID: resuming the 2.5 -> 3.0 M extension"
  train --resume "$RUN" --extend-to 3000000
fi
bash results/train_seed.sh $A $S $CONFIG || { note "$ID evaluations FAILED"; exit 1; }
python tools/tiers/test_set.py --version 4 --model $RUN/best_model.zip --tag $ID --safety off --processes 4 \
  > results/test_set_v4_$ID.log 2>&1
note "   test set v4, $ID exit $? (results/test_set/v4/$ID/summary.txt)"
note "== $ID run done"
