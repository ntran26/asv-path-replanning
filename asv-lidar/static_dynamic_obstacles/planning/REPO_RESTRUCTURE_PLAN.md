# Repository restructure: main code in src (plan, held until after calibration)

Date: 2026-11-20. Status: decided; partly done. The bridge stays where it is until after the two
calibration sessions (`FIELD_CALIBRATION_PLAN.md`), so the bridge verified on 2026-11-14 is not
moved before the basin uses it. Done on 2026-11-20, as it does not touch the bridge:
`train_asv.py` and `test_asv.py` into `src/` (with `src/setups.py`), and the 8 unused `src/`
modules with the tests and tools that import them into `archive/` (`archive/INDEX.md`).
Still to do: `tools/` by purpose, the one-off diagnostics, `field_deployment/` into
`src/field_deployment/`, the July logs into `data/field/`.

## 1. Decisions (2026-11-20)

- **Goal:** the project holds the main code to run in `src/`, as the static-obstacle project
  (`asv-lidar/static_obstacles/src/`) did; testing code in its own folder; the field deployment
  code in `src/` as well.
- **Scope:** the Paper 3 project only: `asv-lidar/static_dynamic_obstacles/` with
  `asv-lidar/field_deployment/` folded into it. The other folders of the repository
  (`static_obstacles`, `asv-grid`, `radar`, `bluefin_modelling`, `asv-lidar/archive`, `Results`,
  `models`) stay as they are.
- **Code the current results do not need** moves to `static_dynamic_obstacles/archive/`, tracked
  and runnable from there; nothing is deleted.
- **Timing:** after the calibration sessions; then verified as in section 4 before any commit.

## 2. Findings from the import graph (2026-11-20)

A static scan of every tracked `.py` file (imports, `importlib.import_module` and `__import__`
with literal names) from the entry points `train_asv.py`, `test_asv.py`,
`src/train_formulation.py`, `src/baseline_config.py`, `src/play.py`, `tools/tiers/test_set.py`,
`tools/tiers/run_test.py` and the bridge (`udp_live_p3.py`, `p3_bridge.py`,
`fake_vessel_replay.py`) reaches 116 files:

- **Almost all of `src/` is in use.** The deployed layer (`safety_v21`, E1) inherits through
  every earlier version, `safety_v2` to `safety_v20`, so none of them can move. Not reached:
  `compare.py`, `rollout.py`, `safety_fast_geometry.py`, `safety_reachability.py`, `ship_v2.py`,
  `train.py`, `train_ppo_baseline.py`, `train_sac_baseline.py`.
- **Tests:** 92 of the 101 test files import only reached code. Nine import code outside it and
  move with that code: `test_ported_harness.py` (`rollout`, `train`), the four
  `test_safety_reachability*.py`, `test_safety_fast_geometry.py`, `test_handback_starts.py`,
  `test_safety_estop.py` and `test_safety_frozen_queue.py` (the last three import diagnostics
  tools; those tools stay if the safety evaluation tools keep them).
- **Field deployment:** the Paper 3 bridge reaches the July bridge (`udp_live_rl.py`: `PoseSync`,
  `apply_emergency_stop`), `log_parser.py`, `test_run.py` and the July helpers `asv_lidar.py`,
  `lidar_pooling.py`, `ship_model.py`, `images.py`, `log_viewer.py`, `policy_render.py`. Three of
  these share a name with a simulator module (`asv_lidar`, `lidar_pooling`, and `ship_model`
  beside `ship`), and `p3_bridge` relies on putting the simulator's modules first on the path,
  so the field code goes into a subfolder, `src/field_deployment/`, not flat into `src/`. Not
  reached: `rl_env.py`, `rl_log_viewer.py`, `udp_listener.py`, `udp_send_receive_template.py`.
- **`bluefin/` stays where it is.** `src/ship.py` finds it at `../bluefin` (a frozen file), and
  the bridge's stopping-table fingerprint hashes `bluefin/*.py` and `*.json` by their path from
  the project root; moving it would change a frozen file and invalidate
  `p3_stopping_profiles.pkl`.
- **`tools/diagnostics/`** (123 files) is mostly one-off analysis. The scripts the planning
  records cite for reported results are kept as tools (section 3); the rest go to the archive.

## 3. Target layout

```
static_dynamic_obstacles/
  README.md              what to run, and this layout
  requirements.txt, pytest.ini
  src/                   simulator, training, policy, safety layer chain, comparators (as now,
                         without the 8 files of section 2); train_asv.py, test_asv.py and
                         setups.py (moved and added 2026-11-20, ahead of the rest)
    field_deployment/    udp_live_p3, p3_bridge, fake_vessel_replay, udp_live_rl, log_parser,
                         test_run and the July helpers; p3_stopping_profiles.pkl
  bluefin/               vessel-model identification (unchanged)
  configs/               unchanged
  tests/                 the 92 unit-test files of the main code
  tools/
    evaluation/          tools/tiers (test set, validation set, Paper 2 suite, frozen suite)
    safety/              the E1 test-set runner and reports (v21_iteration, v21_report,
                         suite_eval, suite_status, run_frozen_queue, prepare_testset_v4,
                         prepare_validation_set, estop_protocol, trace_case, trace_snapshot and
                         what they import)
    paper/               paper_snapshots, render_episode and the cited diagnostics
                         (paper_figures, paper_paths, paper_table, colregs_compliance,
                         artifact_results, compute_budget, learner_summary, comparators_config,
                         tune_classical, a18_a19_verify)
    field/               p3_parity_replay, p3_shadow_replay, p3_timing, trial_analysis/
    runs/                run_status, scale_audit, watch/pause/resume scripts
  data/field/            the July 2026 logs and videos (now field_deployment/2026-07-0*)
  archive/               everything else, with its tests, and an index of where each file came from
  results/, runs/, planning/, planning_records/   unchanged
```

The exact file list is fixed at the start of the work by re-running the import scan on the
tree as it then stands, with the safety and paper tool entry points added.

## 4. Procedure and verification

1. Re-run the import scan; write `archive/INDEX.md` (old path, new path, why archived).
2. Move with `git mv` only; then fix the path set-up of the moved scripts (`sys.path` lines,
   `HERE`/`SIM_ROOT` in the bridge, `test_asv.py`'s `tools/tiers` path). No change to the
   behaviour of `src/` or `configs/` (baseline-v3, digest 03a9c0c22f53ee0b).
3. Check, before any commit:
   - the full test suite (about 1,800 tests; the known `test_safety_v3` dead-ahead failure
     stays the only one);
   - bridge parity, 32/32 runs and 1,497 decisions, with the stopping table installed;
   - the July shadow replay, in process and over UDP loopback, equal to the 2026-11-14 run;
   - `test_asv.py` on a handful of test set v4 episodes, equal to `sacs0_bl3_metrics`;
   - one short training start (`train_asv.py`, a few thousand steps) to confirm the imports.
4. Update the paths in the live documents: `README_P3_BRIDGE.md`, `UDP_BENCH_HANDOVER.md`,
   `README_TRIAL_ANALYSIS.md`, `00_CURRENT_STATE.md`, both field plans, `PAPER3_CLAIMS_AND_TABLES.md`
   and the attachment lists in `PAPER/PAPER3_PROMPTS.md` and
   `MASTER_ARTIFACT/MASTER_ARTIFACT_PROMPTS.md`. Records in `planning_records/` and `results/`
   keep their historical paths; `archive/INDEX.md` maps them.
5. Commit on request; the personal laptop pulls and re-runs parity (`--modes shadow` at least)
   before the next basin session that uses the bridge.
