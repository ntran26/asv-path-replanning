#!/usr/bin/env bash
# Pick the faster CPU setting for a learner, then start its baseline-v3 run (decision,
# 2026-10-08: "pinning or unpinning, use the faster option").
#
#   bash results/bench_cpu.sh <algo> <seed>
#
# Waits until no evaluation is running, then trains three 4,096-step smoke runs on
# configs/baseline_v3.json -- pinned with PyTorch's default threads, pinned with 4 threads,
# unpinned with default threads -- and measures training speed after the first 1,500
# steps (learning has started and pinning is in place). Speed only: the formulation, seed
# and hyperparameters are the same in every variant. Writes results/bench_cpu_<algo>.txt
# and starts results/run_bl3.sh with the fastest setting.
cd "$(dirname "$0")/.."
A=$1; S=$2
LOG=results/train_seed.log
OUT=results/bench_cpu_${A}.txt
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
busy() {
  powershell -NoProfile -Command "
    @(Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\" | Where-Object { \$_.CommandLine -match 'test_set.py|frozen_suite.py|paper2_suite.py|tier1_replay.py|train_formulation.py' }).Count" | tr -d '\r'
}
pin() {
  powershell -NoProfile -Command "
    \$procs = Get-CimInstance Win32_Process -Filter \"Name='python3.10.exe'\";
    \$l = \$procs | Where-Object { \$_.CommandLine -match 'train_formulation.py --config configs/baseline_v3.json --algo $A --seed $S --smoke' } | Select-Object -First 1;
    if (\$l) { \$w = @(\$procs | Where-Object { \$_.ParentProcessId -eq \$l.ProcessId });
      if (\$w.Count -lt 10) { 'workers not up' } else {
      \$p = Get-Process -Id \$l.ProcessId; \$p.ProcessorAffinity = [IntPtr]0x00F; \$p.PriorityClass = 'AboveNormal';
      \$w | ForEach-Object { (Get-Process -Id \$_.ProcessId).ProcessorAffinity = [IntPtr]0xFF0 };
      'pinned learner ' + \$l.ProcessId } } else { 'no learner found' }"
}
rate() {   # steps per second from the logged (time_elapsed, total_timesteps) pairs after 1,500 steps
  python - "$1" <<'EOF'
import re, sys
t = s = None; pts = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.search(r"\|\s+time_elapsed\s+\|\s+(\d+)", line)
    if m: t = int(m.group(1))
    m = re.search(r"\|\s+total_timesteps\s+\|\s+(\d+)", line)
    if m and t is not None:
        s = int(m.group(1)); pts.append((t, s))
pts = [p for p in pts if p[1] >= 1500]
print(f"{(pts[-1][1] - pts[0][1]) / max(pts[-1][0] - pts[0][0], 1e-9):.2f}" if len(pts) >= 2 else "nan")
EOF
}

if [ "$(busy)" != "0" ]; then
  note "== ${A}s${S}_bl3 CPU benchmark armed: waiting for the running evaluations and training to finish"
  until [ "$(busy)" = "0" ]; do sleep 60; done
fi
note "== ${A}s${S}_bl3 CPU benchmark: pinned / pinned with 4 threads / unpinned (4,096-step smoke runs)"
echo "CPU benchmark for $A seed $S, $(date '+%Y-%m-%d %H:%M'), steps/s after 1,500 steps" > "$OUT"
best=""; best_rate=0
for v in "1:" "1:4" "0:"; do
  P=${v%%:*}; T=${v#*:}
  L=results/bench_cpu_${A}_pin${P}_threads${T:-default}.log
  rm -rf runs/${A}_formulation_seed${S}_smoke
  python src/train_formulation.py --config configs/baseline_v3.json --algo $A --seed $S --smoke \
    ${T:+--torch-threads $T} > "$L" 2>&1 &
  PY=$!
  if [ "$P" = 1 ]; then
    for i in $(seq 1 60); do sleep 10; out=$(pin); echo "$out" | grep -q pinned && break; done
  fi
  wait $PY
  r=$(rate "$L")
  echo "PIN=$P THREADS=${T:-default}: $r steps/s" | tee -a "$OUT"
  if python -c "import sys; sys.exit(0 if '$r' != 'nan' and float('$r') > float('$best_rate') else 1)"; then
    best="$v"; best_rate=$r
  fi
done
rm -rf runs/${A}_formulation_seed${S}_smoke
[ -n "$best" ] || { note "${A}s${S}_bl3 CPU benchmark FAILED (see $OUT)"; exit 1; }
P=${best%%:*}; T=${best#*:}
echo "fastest: PIN=$P THREADS=${T:-default} ($best_rate steps/s)" | tee -a "$OUT"
note "   ${A}s${S}_bl3 CPU benchmark: fastest PIN=$P THREADS=${T:-default} ($best_rate steps/s; $OUT)"
PIN=$P THREADS=$T exec bash results/run_bl3.sh $A $S
