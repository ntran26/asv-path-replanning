# Archive (Paper 3 project)

Code that the current results, the training and evaluation entry points and the field bridge
do not use, moved here with `git mv` (history follows) under its original relative path.
Nothing is deleted. Plan: `planning/REPO_RESTRUCTURE_PLAN.md`.

## 2026-11-20: unused `src/` modules, with the tests and tools that use them

Found by a static import scan of every tracked `.py` file from the entry points
(`src/train_asv.py`, `src/test_asv.py`, `src/setups.py`, `src/field_test_cases.py`,
`src/train_formulation.py`, `src/baseline_config.py`, `src/play.py`, `tools/tiers/test_set.py`,
`tools/tiers/run_test.py`, `tools/tiers/field_suite.py`) and the field bridge and its bench
(`field_deployment/udp_live_p3.py`, `p3_bridge.py`, `fake_vessel_replay.py`,
`p3_parity_replay.py`, `p3_shadow_replay.py`, `p3_timing.py`, `trial_analysis/`). Each file
moved with every file that imports it, so nothing left in place breaks. The field bridge reaches
none of them.

| Original path | Now | What it is |
| --- | --- | --- |
| `src/compare.py` | `archive/src/compare.py` | Paper 2 comparison tables |
| `src/rollout.py` | `archive/src/rollout.py` | Paper 2 episode harness (ported) |
| `src/train.py` | `archive/src/train.py` | Paper 2 training front end (ported); `src/train_formulation.py` replaced it |
| `src/train_ppo_baseline.py`, `src/train_sac_baseline.py` | `archive/src/` | Paper 2 baseline trainers |
| `src/ship_v2.py` | `archive/src/ship_v2.py` | the v2 vessel model; `src/ship.py` (v3) replaced it |
| `src/safety_fast_geometry.py` | `archive/src/safety_fast_geometry.py` | tiled point-clearance arithmetic, never used by a reported layer |
| `src/safety_reachability.py` | `archive/src/safety_reachability.py` | conditional reachability tube (stage 1 study) |
| `tests/test_safety_fast_geometry.py`, `tests/test_safety_reachability*.py` (4) | `archive/tests/` | their tests |
| `tests/test_ported_harness.py` (the `rollout`/`train` tests) | `archive/tests/test_ported_harness_rollout_train.py` | split: the metrics, module-import and Paper 2 isolation checks stay in `tests/test_ported_harness.py` |
| `tools/tiers/compare_tier1.py` | `archive/tools/tiers/compare_tier1.py` | Tier 1 classical comparison (imports `compare`) |
| `tools/diagnostics/safety/reachability_containment.py`, `requirements-reachability.txt` | `archive/tools/diagnostics/safety/` | the reachability study's saved-data runner and its `mpmath` pin |

The two tools find the project folder one level further up and put `archive/src` on the path
beside `src`; nothing else in them changed.

Records in `results/` and `planning_records/` keep the old paths (some safety-run manifests
list `src/*.py` hashes at the time of the run); this table maps them.

## Running archived code

```bash
python -m pytest -c archive/pytest.ini --rootdir archive archive/tests   # the archived tests (128), from the project folder
python archive/tools/tiers/compare_tier1.py
python archive/tools/diagnostics/safety/reachability_containment.py
```

Archived modules import the project's own (`src/`); `archive/pytest.ini` puts both on the path.
To bring a file back into use, `git mv` it to its original path.
