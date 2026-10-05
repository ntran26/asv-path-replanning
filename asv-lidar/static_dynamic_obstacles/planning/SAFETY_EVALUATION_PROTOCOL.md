# Main safety evaluation: test set v3

Decision (2026-10-03): **test set v3 is the main evaluation set** for the safety layer. (Test set v4, 2026-10-04, is v3 with the 67 near-impossible episodes replaced; see `planning/BASELINE_V4_PLAN.md`.) It supersedes the earlier v2/DV3 challenge cohorts as the primary benchmark. Their results remain development history; they are not v3 results.

## Frozen benchmark and controllers

- Exactly 1,000 scenarios: 664 frozen-suite-derived and 336 field-layout scenarios.
- Canonical files: `results/test_set/v3/definition.csv`, `definition.json` and `set_v3.0.pkl`.
- Definition digest: `b8212ecf2a4f2529b6dec3a17dfaf178dabe0cf6590c606dd0d3427afdabb4bb`.
- Primary selection: `results/safety_dev/testset_v3_main/selection.json`; all cases are retained in definition order, without outcome-based selection.
- Policy: SAC baseline 3, kept best at 3M timesteps, `runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip`.
- Main comparison: fresh safety OFF versus frozen safety filter V16. V17 did not improve the last probe and is not the primary candidate. Test-set version and safety-filter version are independent.

Pair each result by test ID, episode seed and scenario digest. Do not pair by row position or origin ID. Preserve source, encounter cell, target variant, leg and panel count for subgroup reporting. Every original case stays in the denominator; report timeout and each collision type separately.

The primary metrics are completed goals, rescued SAC failures, and lost SAC successes. More goals alone does not satisfy the requirement if the filter still loses policy-success cases.

## Existing baseline and exposure

The saved `results/test_set/v3/sacs0_bl3/episodes.csv` reports 853 goals, 71 obstacle contacts, 66 target contacts and 10 boundary contacts. It contains all 1,000 IDs, but combines 755 reused v2 results with 245 new runs. It has no frozen checkpoint/config/source archive. Treat it as historical context, not a fresh matched baseline for the current filter.

V3 shares 755 exact geometry/seed pairs with v2 and introduces 245 new or changed cases. Because v2 was explicitly used for safety development, v3 is not wholly unseen. Report that overlap; keep candidate code frozen during evaluation. Making v3 the main evaluation set does not authorize tuning thresholds on its outcomes. Use the separately designated development sets for changes.

## Entry point and artifacts

From the main project directory:

```powershell
python -B tools/diagnostics/safety/prepare_testset_v3.py
python -B tools/diagnostics/safety/safety_candidate_iteration.py --tag v16_primary_paired --modes off,v16
```

Preparation validates the existing cache and freezes all 1,000 identities without generating scenes or running episodes. The second command is the **full paired evaluation: 2,000 new episode runs**, one process with one Torch/native thread. It has not been launched merely by changing the benchmark. Use an explicit `--cases TS3:...` subset for a quick check and label it as a subset, not the full result. Do not alter the frozen selection to remove difficult cases.

TS3 results go to `results/safety_dev/testset_v3_main/runs/<tag>/`, with reserved attempts, exact source/settings/checkpoint hashes, individual results and decision traces. Existing tags cannot be overwritten or silently retried. The report validator supports fresh same-run OFF comparisons and keeps TS3 separate from TS2, even where the textual IDs overlap.

Older cohort commands must specify their original `--selection` explicitly. TS2/DV3 selections retain the original `results/safety_dev/v10_iterations/` output location. The fixed V9 pilot script and its old selection are unchanged.

At most two evaluation processes may run. Preserve the currently active training jobs, including `pilot_v4_A`; do not use the legacy test-set runner's default three-worker setting. Constants, checkpoints, Paper 2 and older results remain unchanged.

Methods and citations remain in `SAFETY_LAYER_V16_PLAN.md` and the implementation files. This protocol changes the evaluation target, not the controller method.

## Completed small primary check (2026-10-03)

The metadata-selected18-case OFF/V16/V19 comparison is complete (54 runs): SAC14/18 goals, V16 and V19 each15/18. Both filters preserve all14 SAC successes and rescue the same one failure. V19 issues exactly the same commands as V16 in every case; it is not promoted. Three collisions remain. This18/1000-case subset does not replace the complete benchmark, and no primary result was used for tuning. See `results/safety_dev/testset_v3_main/quick_v19_report/report.md` and the linked command audit. No new evaluation remains queued.
