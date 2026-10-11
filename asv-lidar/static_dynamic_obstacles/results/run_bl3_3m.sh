#!/usr/bin/env bash
# One baseline-v3 paper run for seeds 1 and 2, start to finish, trained straight to
# 3.0 M (decision, 2026-10-10):
#
#   bash results/run_bl3_3m.sh <ppo|recurrent_ppo|sac|tqc> <seed> [--after-evaluations]
#
# Seed 0 of every learner was trained to 2.5 M, stopped and extended to 3.0 M
# (results/run_bl3.sh). Seeds 1-2 train in one run to 3.0 M with the same curriculum
# timing: `--train-to 3000000` keeps the stage boundaries on the frozen 2.5 M schedule
# (0.16, 0.36, 0.64, 1.0, 1.5, 2.0 M) and holds stage 7 to the end, as the extension
# did. The evaluation points match seed 0 as well: every 200,000 steps, plus 2.5 M for
# SAC and TQC (where seed 0's first phase ended and was evaluated; PPO and
# RecurrentPPO kept no evaluation there). The 2.5 M checkpoint is kept as
# final_model_2500000.zip, as for seed 0.
#
# Then train_seed.sh's evaluations on the best checkpoint (Tier 1, frozen suite,
# reference layouts) and test set v4 (report only), safety layer off.
#
# CPU: PIN=1 pins the learner to the P-cores (0x00F, AboveNormal) and its workers to the
# E-cores (0xFF0); PIN=0 leaves everything on all 12 CPUs. THREADS=n sets PyTorch's thread
# count. Defaults are the measured winners: PPO/RecurrentPPO unpinned (2026-10-07), TQC
# pinned with 4 threads (results/bench_cpu_tqc.txt); SAC is benchmarked by
# results/bench_cpu.sh on its first seed-1 run, which then starts this script.
# --after-evaluations waits until no test-set, frozen-suite, reference-layout or Tier 1
# evaluation is running.
cd "$(dirname "$0")/.."
A=$1; S=$2; WAIT=${3:-}
[ -n "$A" ] && [ -n "$S" ] || { echo "usage: bash results/run_bl3_3m.sh <ppo|recurrent_ppo|sac|tqc> <seed> [--after-evaluations]"; exit 2; }
CONFIG=configs/baseline_v3.json
LOG=results/train_seed.log
RUN=runs/${A}_formulation_seed${S}_bl3
ID=${A}s${S}_bl3
case "$A" in
  sac) PIN=${PIN:-1}; EXTRA="--extra-evals 2500000" ;;
  tqc) PIN=${PIN:-1}; THREADS=${THREADS-4}; EXTRA="--extra-evals 2500000" ;;
  *)   PIN=${PIN:-0}; EXTRA="" ;;
esac
THREADS=${THREADS:-}
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

busy() {   # number of evaluation processes running
  powershell -NoProfile -Command "
    @(Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\" | Where-Object { \$_.CommandLine -match 'test_set.py|frozen_suite.py|paper2_suite.py|tier1_replay.py' }).Count" | tr -d '\r'
}
pin() {   # learner on the performance cores, workers on the efficiency cores
  powershell -NoProfile -Command "
    \$procs = Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\";
    \$l = \$procs | Where-Object { \$_.CommandLine -match 'train_formulation.py --config configs/baseline_v3.json --algo $A --seed $S ' } | Select-Object -First 1;
    if (\$l) { \$p = Get-Process -Id \$l.ProcessId; \$p.ProcessorAffinity = [IntPtr]0x00F; \$p.PriorityClass = 'AboveNormal';
      \$procs | Where-Object { \$_.ParentProcessId -eq \$l.ProcessId } | ForEach-Object { (Get-Process -Id \$_.ProcessId).ProcessorAffinity = [IntPtr]0xFF0 };
      'pinned learner ' + \$l.ProcessId } else { 'no learner found' }"
}
pin_when_up() {   # the workers exist once the validation set is built
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

if [ ! -f "$RUN/final_model.zip" ]; then
  if [ -d "$RUN" ] && ls "$RUN"/${A}_*_steps.zip > /dev/null 2>&1; then
    note "== $ID: resuming the straight run to 3.0 M (baseline-v3)"
    train --resume "$RUN"
  else
    [ -d "$RUN" ] && { mv "$RUN" "${RUN}_incomplete_$(date +%Y%m%d_%H%M%S)"; note "   $ID: no checkpoint, set aside"; }
    note "== $ID: training 0 -> 3.0 M in one run (baseline-v3, stages on the 2.5 M schedule; PIN=$PIN, THREADS=${THREADS:-default})"
    : > "$RUN.log"
    train --train-to 3000000 $EXTRA
  fi
  rc=$?
  note "   $ID 3.0 M exit $rc"
  [ $rc -eq 0 ] && [ -f "$RUN/final_model.zip" ] || { note "$ID FAILED (see $RUN.log)"; exit 1; }
fi
if [ ! -f "$RUN/final_model_2500000.zip" ] && [ -f "$RUN/${A}_2500000_steps.zip" ]; then
  cp "$RUN/${A}_2500000_steps.zip" "$RUN/final_model_2500000.zip"
  cp "$RUN/${A}_vecnormalize_2500000_steps.pkl" "$RUN/final_vecnormalize_2500000.pkl" 2>/dev/null
fi
bash results/train_seed.sh $A $S $CONFIG || { note "$ID evaluations FAILED"; exit 1; }
python tools/tiers/test_set.py --version 4 --model $RUN/best_model.zip --tag $ID --safety off --processes 4 \
  > results/test_set_v4_$ID.log 2>&1
note "   test set v4, $ID exit $? (results/test_set/v4/$ID/summary.txt)"
note "== $ID run done"
