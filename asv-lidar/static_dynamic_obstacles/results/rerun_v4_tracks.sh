#!/usr/bin/env bash
# Test set v4 again for the paper table, now saving each episode's true track for the
# offline per-rule COLREGs checks (2026-10-08): the three seed-0 learners and the two
# classical comparators, safety layer off, then each run's compliance.csv
# (tools/diagnostics/colregs_compliance.py) and results/baseline_v3/paper_table.md.
# Evaluation is deterministic, so the per-episode rows repeat the earlier _metrics runs;
# only the track files are new. Then the TQC seed-0 baseline-v3 run starts, CPU benchmark
# first (results/bench_cpu.sh), logged to results/tqc_bl3_run.log.
cd "$(dirname "$0")/.."
LOG=results/train_seed.log
OUT=results/rerun_v4_tracks.log
P=10
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
ev() {   # folder tag, then test_set.py arguments
  local tag=$1; shift
  python tools/tiers/test_set.py --version 4 --safety off --processes $P "$@" \
    > results/test_set_v4_${tag}_tracks.log 2>&1
  local rc=$?
  note "   test set v4 with tracks, $tag exit $rc"
  [ $rc -eq 0 ] && [ -f results/test_set/v4/$tag/trajectories_off.npz ]
}
note "== test set v4 with tracks: SAC, PPO, RecurrentPPO, LOS-DWA, COLREGs-VO ($P processes)"
ok=1
ev sacs0_bl3_metrics --model runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip --tag sacs0_bl3_metrics || ok=0
ev ppos0_bl3_metrics --model runs/ppo_formulation_seed0_bl3/best_model.zip --tag ppos0_bl3_metrics || ok=0
ev recurrent_ppos0_bl3_metrics --model runs/recurrent_ppo_formulation_seed0_bl3/best_model.zip \
  --tag recurrent_ppos0_bl3_metrics || ok=0
ev los_dwa --policy los_dwa || ok=0
ev colregs_vo --policy colregs_vo || ok=0
python tools/diagnostics/colregs_compliance.py sacs0_bl3_metrics ppos0_bl3_metrics recurrent_ppos0_bl3_metrics \
  los_dwa colregs_vo >> "$OUT" 2>&1 || ok=0
python tools/diagnostics/paper_table.py > /dev/null 2>> "$OUT" || ok=0
note "   test set v4 with tracks: ALL DONE (ok=$ok); results/baseline_v3/paper_table.md"
exec bash results/bench_cpu.sh tqc 0 > results/tqc_bl3_run.log 2>&1
