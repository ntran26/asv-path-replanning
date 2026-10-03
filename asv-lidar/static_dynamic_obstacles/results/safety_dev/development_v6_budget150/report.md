# Development safety campaign

Snapshot: 2026-10-02T16:14:13Z.

**Budget: 149/150 new attempts consumed; 1 remain.** 144 durable completed policy results, 0 recorded policy-run errors, 5 completed verification test simulations, 0 unresolved/in-flight reservations. The previous quick76/100 campaign is excluded.

Each tag is a separate frozen source/settings version. Only completed tags enter the outcome tables. Active or failed tags still consume budget. Episode JSON files are authoritative; no missing outcome is inferred. Runtimes sum completed episode seconds, excluding setup and failed/in-flight execution.

| Tag | Status | Attempts | Completed / planned | Errors | Completed seconds |
|---|---|---:|---:|---:|---:|
| broad_v6_50 | complete | 100 | 100 / 100 | 0 | 1030.31 |
| calibrated_boundary_regression_probe | aborted before execution | 0 | 0 / 1 | 0 | 0.00 |
| provisional_evidence_v2 | complete | 4 | 4 / 4 | 0 | 39.61 |
| provisional_targets_v1 | complete | 8 | 8 / 8 | 0 | 50.90 |
| search_calibrated_v1 | complete | 4 | 4 / 4 | 0 | 66.29 |
| search_evidence_history_v3 | complete | 6 | 6 / 6 | 0 | 119.49 |
| search_evidence_v2 | complete | 6 | 6 / 6 | 0 | 102.85 |
| trajectory_search_v1 | complete | 8 | 8 / 8 | 0 | 110.84 |
| v6_calibrated_hulls_pilot | complete | 8 | 8 / 8 | 0 | 38.43 |

`calibrated_boundary_regression_probe` was [aborted before execution](runs/calibrated_boundary_regression_probe/preflight_aborted.json): Frozen reference source changed: src/scenario.py. Zero attempts reserved and zero episodes started; this is not a policy-run error or a pending evaluation.

## Verification simulations charged separately

These completed test simulations consume budget but are not SAC/DV3 policy evaluations. Passed and failed assertions both complete their consumed slots; no goal or collision outcome is inferred. Recorded source hashes identify the test files at execution, without requiring later working copies to remain unchanged.

| Attempt | Test | Status | Test source SHA-256 |
|---:|---|---|---|
| 39 | `tests/test_safety_v2.py::test_filter_steers_off_an_obstacle_dead_ahead_and_leaves_open_water_alone` | passed | `8baedeeb199c278b810b7b5768733a635dea16fd399af25c2c6785cf20ff50b1` |
| 40 | `tests/test_safety_v2.py::test_filter_is_off_with_the_safety_layer_off` | passed | `8baedeeb199c278b810b7b5768733a635dea16fd399af25c2c6785cf20ff50b1` |
| 41 | `tests/test_safety_v3.py::test_filter_steers_off_an_obstacle_dead_ahead_and_leaves_open_water_alone` | failed | `0368e3f40ae6285c6124f933668ac71a0ec45206cd30e03ea2e11233e8d8bf10` |
| 42 | `tests/test_safety_v3.py::test_filter_is_off_with_the_safety_layer_off` | passed | `0368e3f40ae6285c6124f933668ac71a0ec45206cd30e03ea2e11233e8d8bf10` |
| 43 | `tests/test_safety_v3.py::test_v3_filter_is_selected_and_hands_back_only_with_steerage` | passed | `0368e3f40ae6285c6124f933668ac71a0ec45206cd30e03ea2e11233e8d8bf10` |

Unmodified SafetyFilterV3 collided in existing dead-ahead constant-command test. V6 is not constructed by this test; no legacy source or thresholds changed to fix it.

[Verification records and accounting explanation](verification_runs.json).

## Completed outcomes by frozen tag

| Tag | Mode | n | Goals | Obstacles | Boundaries | Targets | Timeouts | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| broad_v6_50 | v4 | 50 | 23 | 9 | 7 | 11 | 0 | 283.46 |
| broad_v6_50 | v6 | 50 | 33 | 9 | 3 | 5 | 0 | 746.85 |
| provisional_evidence_v2 | v6 | 4 | 3 | 0 | 0 | 1 | 0 | 39.61 |
| provisional_targets_v1 | v4 | 4 | 2 | 0 | 0 | 2 | 0 | 23.08 |
| provisional_targets_v1 | v6 | 4 | 2 | 1 | 0 | 1 | 0 | 27.82 |
| search_calibrated_v1 | v6 | 4 | 3 | 0 | 1 | 0 | 0 | 66.29 |
| search_evidence_history_v3 | v6 | 6 | 3 | 1 | 1 | 1 | 0 | 119.49 |
| search_evidence_v2 | v6 | 6 | 4 | 0 | 1 | 1 | 0 | 102.85 |
| trajectory_search_v1 | v4 | 4 | 2 | 0 | 2 | 0 | 0 | 45.37 |
| trajectory_search_v1 | v6 | 4 | 3 | 0 | 1 | 0 | 0 | 65.47 |
| v6_calibrated_hulls_pilot | v6 | 8 | 1 | 4 | 1 | 2 | 0 | 38.43 |

## Fresh v6 versus v4 pairs

Only matching case/seed/geometry records within the same tag are paired. Goal gains/losses compare goal against any non-goal outcome. Historical v4 outcomes are excluded.

| Tag | Fresh paired cases | V6 gained goals | V6 lost goals |
|---|---:|---:|---:|
| broad_v6_50 | 50 | 13 | 3 |
| provisional_evidence_v2 | 0 | 0 | 0 |
| provisional_targets_v1 | 4 | 1 | 1 |
| search_calibrated_v1 | 0 | 0 | 0 |
| search_evidence_history_v3 | 0 | 0 | 0 |
| search_evidence_v2 | 0 | 0 | 0 |
| trajectory_search_v1 | 4 | 1 | 0 |
| v6_calibrated_hulls_pilot | 0 | 0 | 0 |

## Repeated identities and provenance

14 scenario/mode identities recur across experiment tags. These planned reruns are listed with seed, scenario digest, tag, attempt and manifest hash in `summary.json`; they are not additional independent scenarios. Results from different source versions are not pooled.

- broad_v6_50: [manifest](runs/broad_v6_50/manifest.json), [source archive](runs/broad_v6_50/evaluated_sources.zip); manifest `5f86f112a75955df`.
- calibrated_boundary_regression_probe: [manifest](runs/calibrated_boundary_regression_probe/manifest.json), [source archive](runs/calibrated_boundary_regression_probe/evaluated_sources.zip); manifest `68d7c0c05607bab7`.
- provisional_evidence_v2: [manifest](runs/provisional_evidence_v2/manifest.json), [source archive](runs/provisional_evidence_v2/evaluated_sources.zip); manifest `626c7e996289574b`.
- provisional_targets_v1: [manifest](runs/provisional_targets_v1/manifest.json), [source archive](runs/provisional_targets_v1/evaluated_sources.zip); manifest `5e07d276a958c5c1`.
- search_calibrated_v1: [manifest](runs/search_calibrated_v1/manifest.json), [source archive](runs/search_calibrated_v1/evaluated_sources.zip); manifest `06ac1e671f1ce92b`.
- search_evidence_history_v3: [manifest](runs/search_evidence_history_v3/manifest.json), [source archive](runs/search_evidence_history_v3/evaluated_sources.zip); manifest `a72988841ac1c0f1`.
- search_evidence_v2: [manifest](runs/search_evidence_v2/manifest.json), [source archive](runs/search_evidence_v2/evaluated_sources.zip); manifest `ca3bc92bbd0d071b`.
- trajectory_search_v1: [manifest](runs/trajectory_search_v1/manifest.json), [source archive](runs/trajectory_search_v1/evaluated_sources.zip); manifest `7ac02399b0754dd5`.
- v6_calibrated_hulls_pilot: [manifest](runs/v6_calibrated_hulls_pilot/manifest.json), [source archive](runs/v6_calibrated_hulls_pilot/evaluated_sources.zip); manifest `52b5daf29e0e786d`.

## Historical reference, separately qualified

Historical selected v4 recorded **123/150 goals** on the original DV3 set. This is not a fresh within-tag comparator. The [reference audit](reference_audit.json) confirms the recorded policy/config, 150 seeds/scenario digests and v2–v4 effective constants, but historical `env.py` and `safety_v4.py` source hashes differ. Exact historical all-source behavioral equivalence is unproven; wrapper differences are reported separately.

These are selected development cases used to design and compare candidate methods. No held-out success estimate, 95–100% performance claim, confidence interval, or statistical significance follows from these pilots.

## Completed analysis and integration notes

- [Deterministic full-suite upper bound](finite_suite_bound.md)
- [Native v6 integration and source-drift audit](integration_audit.json)
- [blind zone oracle replay](offline/blind_zone_oracle_replay/REPORT.md)
- [broad v6 50 failure classification](offline/broad_v6_50_failure_classification/REPORT.md)
