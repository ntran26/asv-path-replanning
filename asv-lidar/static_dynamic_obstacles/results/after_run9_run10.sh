#!/usr/bin/env bash
# A29 (F85): after run 9's analysis, switch the growing v_hold on, audit, launch run 10, analyse it.
cd "$(dirname "$0")/.."
F9="C:/Users/hntran/AppData/Local/Temp/claude/C--Users-hntran-OneDrive---University-of-Tasmania-Documents-PhD-asv-path-replanning-asv-lidar/d009229f-c16f-485d-a74a-0d08e9ebc0f5/tasks/bqmecqwqh.output"
until grep -qxE "== ANALYSIS DONE|RUN9 FAILED" "$F9"; do sleep 120; done
grep -qx "RUN9 FAILED" "$F9" && { echo "RUN9 FAILED -- run 10 not started"; exit 1; }
# Run 9's stand-on diagnosis, for the run 8 comparison -- before the switch, so
# run 9 is scored with the reward it trained on.
python tools/diagnostics/standon_speed.py --model runs/ppo_formulation_seed0_v9/final_model.zip --tag run9 > results/standon_speed_run9.log 2>&1
echo "== standon run9 exit $?"
echo "== RUN9 ANALYSIS COMPLETE"
python - <<'PY'
p = "src/constants.py"
s = open(p, encoding="utf-8", newline="").read()
old = "V_HOLD_GROWS = False"
assert s.count(old) == 1, "switch not found"
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, "V_HOLD_GROWS = True"))
print("== V_HOLD_GROWS on")
PY
python -m pytest tests/test_reward.py -q -p no:cacheprovider 2>&1 | tail -1
python tools/scale_audit.py --episodes 240 --workers 10 --out results/scale_audit_a29.json > results/scale_audit_a29.log 2>&1
echo "== audit exit $?"
python src/train_formulation.py --algo ppo --timesteps 2000000 --seed 0 --num-envs 10 --eval-freq 200000 --eval-per-class 20 --tag v10 --train-supervisor off --low-speed-start-frac 0.15 --eval-supervisor both > runs/ppo_formulation_seed0_v10.log 2>&1
echo "== run 10 exit $?"
grep -q Traceback runs/ppo_formulation_seed0_v10.log && { echo "RUN10 FAILED"; exit 1; }
for m in off on; do
  python tools/tiers/tier1_replay.py --model runs/ppo_formulation_seed0_v10/final_model.zip --tag run10_supervisor_$m --supervisor $m > results/tiers/tier1_run10_supervisor_$m.log 2>&1
  echo "== tier1 run10 $m exit $?"
done
python tools/diagnostics/crossing_diagnosis.py --model runs/ppo_formulation_seed0_v10/final_model.zip --tag run10 > results/crossing_diagnosis_run10.log 2>&1
echo "== crossing diagnosis run10 exit $?"
python tools/diagnostics/standon_speed.py --model runs/ppo_formulation_seed0_v10/final_model.zip --tag run10 > results/standon_speed_run10.log 2>&1
echo "== standon run10 exit $?"
echo "== RUN10 ANALYSIS DONE"
