# Saved safety-iteration report

All reported iteration records passed completion, identity, trace-counter and archived-source checks; no explicit filter-error fallback was recorded. These finite deliberately selected development episodes do not establish 100% safety, preservation of every SAC success, a population success rate, or real-world performance. No episodes were run to create this report.

Fresh prior-pilot references were executed separately on the same canonical scene and seed with matching checkpoint, config and effective constants. Historical saved-branch references are labeled separately and are not fresh evaluations. Source differences are listed below; identity checks alone do not establish numerical equivalence of changed source or native-thread settings.

| Iteration | Mode | N | Goal | Target | Obstacle | Boundary | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|
| v18_v19_development12 | v16 | 12 | 5 | 3 | 0 | 4 | 0 |
| v18_v19_development12 | v18 | 12 | 5 | 3 | 0 | 4 | 0 |
| v18_v19_development12 | v19 | 12 | 5 | 3 | 1 | 3 | 0 |

| Iteration / mode | Reference | Evidence | Matched | Gained goals | Lost goals | Net |
|---|---|---|---:|---:|---:|---:|
| v18_v19_development12 / v16 | off | fresh_prior_pilot | 8 | 1 | 2 | -1 |
| v18_v19_development12 / v16 | off | historical_saved_branch | 4 | 0 | 4 | -4 |
| v18_v19_development12 / v16 | v8 | fresh_prior_pilot | 8 | 2 | 0 | +2 |
| v18_v19_development12 / v16 | v8 | historical_saved_branch | 4 | 0 | 0 | +0 |
| v18_v19_development12 / v16 | v9 | fresh_prior_pilot | 8 | 1 | 1 | +0 |
| v18_v19_development12 / v16 | v16 | fresh_reference_iteration | 12 | 0 | 0 | +0 |
| v18_v19_development12 / v18 | off | fresh_prior_pilot | 8 | 1 | 2 | -1 |
| v18_v19_development12 / v18 | off | historical_saved_branch | 4 | 0 | 4 | -4 |
| v18_v19_development12 / v18 | v8 | fresh_prior_pilot | 8 | 2 | 0 | +2 |
| v18_v19_development12 / v18 | v8 | historical_saved_branch | 4 | 0 | 0 | +0 |
| v18_v19_development12 / v18 | v9 | fresh_prior_pilot | 8 | 1 | 1 | +0 |
| v18_v19_development12 / v18 | v16 | fresh_same_iteration | 12 | 0 | 0 | +0 |
| v18_v19_development12 / v19 | off | fresh_prior_pilot | 8 | 1 | 2 | -1 |
| v18_v19_development12 / v19 | off | historical_saved_branch | 4 | 0 | 4 | -4 |
| v18_v19_development12 / v19 | v8 | fresh_prior_pilot | 8 | 2 | 0 | +2 |
| v18_v19_development12 / v19 | v8 | historical_saved_branch | 4 | 0 | 0 | +0 |
| v18_v19_development12 / v19 | v9 | fresh_prior_pilot | 8 | 1 | 1 | +0 |
| v18_v19_development12 / v19 | v16 | fresh_same_iteration | 12 | 0 | 0 | +0 |

A gained goal versus off is a rescued policy failure; a lost goal versus off is a broken policy success. Collision→timeout is not counted as a rescued goal. Exact per-case transitions, source labels, and dataset/stratum denominators are in [paired.csv](paired.csv) and [summary.json](summary.json). Repeated cases across tags are intentional separate controller experiments, not independent replications or extra coverage; they are never pooled into a success-rate estimate.

| Iteration / mode | Selection stratum | SAC evidence | Matched | Preserved / SAC goals | Rescued / SAC failures | Broken SAC goals | Unresolved failures |
|---|---|---|---:|---:|---:|---:|---:|
| v18_v19_development12 / v16 | expired_consequences | fresh_prior_pilot | 2 | 1/2 | 0/0 | 1 | 0 |
| v18_v19_development12 / v16 | other_broken_boundary | fresh_prior_pilot | 1 | 0/1 | 0/0 | 1 | 0 |
| v18_v19_development12 / v16 | other_broken_target | fresh_prior_pilot | 1 | 1/1 | 0/0 | 0 | 0 |
| v18_v19_development12 / v16 | remaining_known_sac_regression | historical_saved_branch | 4 | 0/4 | 0/0 | 4 | 0 |
| v18_v19_development12 / v16 | rescued_searched_escape | fresh_prior_pilot | 1 | 0/0 | 1/1 | 0 | 0 |
| v18_v19_development12 / v16 | success_never_fire_field_layout | fresh_prior_pilot | 1 | 1/1 | 0/0 | 0 | 0 |
| v18_v19_development12 / v16 | success_never_fire_frozen | fresh_prior_pilot | 1 | 1/1 | 0/0 | 0 | 0 |
| v18_v19_development12 / v16 | unrescued_optimistic_no_fire | fresh_prior_pilot | 1 | 0/0 | 0/1 | 0 | 1 |
| v18_v19_development12 / v18 | expired_consequences | fresh_prior_pilot | 2 | 1/2 | 0/0 | 1 | 0 |
| v18_v19_development12 / v18 | other_broken_boundary | fresh_prior_pilot | 1 | 0/1 | 0/0 | 1 | 0 |
| v18_v19_development12 / v18 | other_broken_target | fresh_prior_pilot | 1 | 1/1 | 0/0 | 0 | 0 |
| v18_v19_development12 / v18 | remaining_known_sac_regression | historical_saved_branch | 4 | 0/4 | 0/0 | 4 | 0 |
| v18_v19_development12 / v18 | rescued_searched_escape | fresh_prior_pilot | 1 | 0/0 | 1/1 | 0 | 0 |
| v18_v19_development12 / v18 | success_never_fire_field_layout | fresh_prior_pilot | 1 | 1/1 | 0/0 | 0 | 0 |
| v18_v19_development12 / v18 | success_never_fire_frozen | fresh_prior_pilot | 1 | 1/1 | 0/0 | 0 | 0 |
| v18_v19_development12 / v18 | unrescued_optimistic_no_fire | fresh_prior_pilot | 1 | 0/0 | 0/1 | 0 | 1 |
| v18_v19_development12 / v19 | expired_consequences | fresh_prior_pilot | 2 | 1/2 | 0/0 | 1 | 0 |
| v18_v19_development12 / v19 | other_broken_boundary | fresh_prior_pilot | 1 | 0/1 | 0/0 | 1 | 0 |
| v18_v19_development12 / v19 | other_broken_target | fresh_prior_pilot | 1 | 1/1 | 0/0 | 0 | 0 |
| v18_v19_development12 / v19 | remaining_known_sac_regression | historical_saved_branch | 4 | 0/4 | 0/0 | 4 | 0 |
| v18_v19_development12 / v19 | rescued_searched_escape | fresh_prior_pilot | 1 | 0/0 | 1/1 | 0 | 0 |
| v18_v19_development12 / v19 | success_never_fire_field_layout | fresh_prior_pilot | 1 | 1/1 | 0/0 | 0 | 0 |
| v18_v19_development12 / v19 | success_never_fire_frozen | fresh_prior_pilot | 1 | 1/1 | 0/0 | 0 | 0 |
| v18_v19_development12 / v19 | unrescued_optimistic_no_fire | fresh_prior_pilot | 1 | 0/0 | 0/1 | 0 | 1 |

Named broken-SAC-success inventory (the comparison evidence is identified for every case):

- v18_v19_development12 / v16: DV3:DV3-CRS-CV-04 → collision:boundary (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v18: DV3:DV3-CRS-CV-04 → collision:boundary (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v19: DV3:DV3-CRS-CV-04 → collision:obstacle (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v16: TS2:BAS-HO-NC-059 → collision:target (expired_consequences; fresh_prior_pilot).
- v18_v19_development12 / v18: TS2:BAS-HO-NC-059 → collision:target (expired_consequences; fresh_prior_pilot).
- v18_v19_development12 / v19: TS2:BAS-HO-NC-059 → collision:target (expired_consequences; fresh_prior_pilot).
- v18_v19_development12 / v16: TS2:BAS-NU-CV-070 → collision:boundary (other_broken_boundary; fresh_prior_pilot).
- v18_v19_development12 / v18: TS2:BAS-NU-CV-070 → collision:boundary (other_broken_boundary; fresh_prior_pilot).
- v18_v19_development12 / v19: TS2:BAS-NU-CV-070 → collision:boundary (other_broken_boundary; fresh_prior_pilot).
- v18_v19_development12 / v16: TS2:P2-L2-CRS-FIX-19 → collision:boundary (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v18: TS2:P2-L2-CRS-FIX-19 → collision:boundary (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v19: TS2:P2-L2-CRS-FIX-19 → collision:boundary (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v16: TS2:P2-L3-BO-FIX-09 → collision:boundary (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v18: TS2:P2-L3-BO-FIX-09 → collision:boundary (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v19: TS2:P2-L3-BO-FIX-09 → collision:boundary (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v16: TS2:P2-L3-CRP-FIX-03 → collision:target (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v18: TS2:P2-L3-CRP-FIX-03 → collision:target (remaining_known_sac_regression; historical_saved_branch).
- v18_v19_development12 / v19: TS2:P2-L3-CRP-FIX-03 → collision:target (remaining_known_sac_regression; historical_saved_branch).

Across the included tags there are 36 completed iteration records covering 12 distinct canonical scenarios. Reference-pilot records are comparison evidence and are not added to the new iteration count.

The separate [baseline source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/baseline_source_audit.json>) records the rationale and checks for changed baseline-related sources. This report preserves that audit and the differing hashes; it does not erase the difference.

The [native-dispatch source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/native_dispatch_source_audit.json>) records the later environment-selector integration. The evaluated archives remain authoritative; this audit is linked because an environment source hash differs across the compared runs.

The [V15 integration source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/v15_integration_source_audit.json>) addresses recorded source differences across these runs. Its hash is preserved in provenance; the evaluated source archives remain authoritative.

The [V6 diagnostic-only AST audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/v15_v6_diagnostic_ast_audit.json>) addresses recorded source differences across these runs. Its hash is preserved in provenance; the evaluated source archives remain authoritative.

## v18_v19_development12

36/36 completed, 36 reserved attempts, 12 distinct scenarios. Timed episodes sum to 940.01 s; sum of component runner durations 944.15 s. Early collisions shorten episodes, so elapsed time does not isolate controller computation speed.

| Mode | Decisions | Changed-action steps | Requested-brake steps |
|---|---:|---:|---:|
| v16 | 581 | 79 | 35 |
| v18 | 581 | 79 | 35 |
| v19 | 575 | 79 | 35 |

Mechanism events are descriptive checks or handbacks, not proof that an accident was prevented. Requested braking does not imply a particular signed RPM; the trace records the executed command. Missing diagnostics are null/unrecorded, while explicitly recorded false flags contribute zero. Persistence hypothesis-decision counts include repeated predictions of the same vessel; they are not counts of distinct vessels.

- v16 prefix-request diagnostics: 581 recorded decisions; checked=165; request reasons={'adequate_parent_clearance': 20, 'ordinary_parent_below_trigger_margin': 47, 'missing_ordinary_selection': 4, 'currently_failing_parent': 37, 'missing_paired_check': 57}; skip reasons={'adequate_parent_clearance': 20, 'missing_ordinary_selection': 4, 'missing_paired_check': 57}.
- v16 motion-axis geometry: 581 recorded decisions; replacement decisions=50, ID-events=50, episodes with replacements=7. ID-events sum repeated observations across decisions and episodes; they are not distinct vessels.
- v18 prefix-request diagnostics: 581 recorded decisions; checked=165; request reasons={'adequate_parent_clearance': 20, 'ordinary_parent_below_trigger_margin': 47, 'missing_ordinary_selection': 4, 'currently_failing_parent': 37, 'missing_paired_check': 57}; skip reasons={'adequate_parent_clearance': 20, 'missing_ordinary_selection': 4, 'missing_paired_check': 57}.
- v18 motion-axis geometry: 581 recorded decisions; replacement decisions=50, ID-events=50, episodes with replacements=7. ID-events sum repeated observations across decisions and episodes; they are not distinct vessels.
- v19 prefix-request diagnostics: 575 recorded decisions; checked=159; request reasons={'adequate_parent_clearance': 20, 'ordinary_parent_below_trigger_margin': 47, 'missing_ordinary_selection': 4, 'currently_failing_parent': 37, 'missing_paired_check': 51}; skip reasons={'adequate_parent_clearance': 20, 'missing_ordinary_selection': 4, 'missing_paired_check': 51}.
- v19 motion-axis geometry: 575 recorded decisions; replacement decisions=50, ID-events=50, episodes with replacements=7. ID-events sum repeated observations across decisions and episodes; they are not distinct vessels.

This cohort combines disjoint completed slices after checking identical classes, constructor options, checkpoint/config, effective constants, runtime thread settings and every common archived source hash. Optional snapshot-helper inclusion and snapshot logging may differ. Any explicitly allowed inactive-version addition is audited below. Component identity is retained in each CSV row:
- v18_v19_probe_a: 18 records / 6 scenarios; snapshots=False; optional source difference=[].
- v18_v19_probe_b: 18 records / 6 scenarios; snapshots=False; optional source difference=[].

- v16 vs off (fresh_prior_pilot): gains TS2:P2-L1-CRP-VAR-12; losses TS2:BAS-HO-NC-059, TS2:BAS-NU-CV-070.
- v16 vs off (historical_saved_branch): gains none; losses DV3:DV3-CRS-CV-04, TS2:P2-L2-CRS-FIX-19, TS2:P2-L3-BO-FIX-09, TS2:P2-L3-CRP-FIX-03.
- v16 vs v8 (fresh_prior_pilot): gains DV3:DV3-BO-CV-04, TS2:CH-HO-CV-073; losses none.
- v16 vs v8 (historical_saved_branch): gains none; losses none.
- v18 vs off (fresh_prior_pilot): gains TS2:P2-L1-CRP-VAR-12; losses TS2:BAS-HO-NC-059, TS2:BAS-NU-CV-070.
- v18 vs off (historical_saved_branch): gains none; losses DV3:DV3-CRS-CV-04, TS2:P2-L2-CRS-FIX-19, TS2:P2-L3-BO-FIX-09, TS2:P2-L3-CRP-FIX-03.
- v18 vs v8 (fresh_prior_pilot): gains DV3:DV3-BO-CV-04, TS2:CH-HO-CV-073; losses none.
- v18 vs v8 (historical_saved_branch): gains none; losses none.
- v19 vs off (fresh_prior_pilot): gains TS2:P2-L1-CRP-VAR-12; losses TS2:BAS-HO-NC-059, TS2:BAS-NU-CV-070.
- v19 vs off (historical_saved_branch): gains none; losses DV3:DV3-CRS-CV-04, TS2:P2-L2-CRS-FIX-19, TS2:P2-L3-BO-FIX-09, TS2:P2-L3-CRP-FIX-03.
- v19 vs v8 (fresh_prior_pilot): gains DV3:DV3-BO-CV-04, TS2:CH-HO-CV-073; losses none.
- v19 vs v8 (historical_saved_branch): gains none; losses none.

Archived source differences relative to the prior fresh pilot:
- src/dev_set_v4.py
- src/env.py
- src/field_training.py
- src/safety_consistent_tracks.py
- src/safety_fast_geometry.py
- src/safety_heading_memory.py
- src/safety_motion_axis.py
- src/safety_policy_prefix.py
- src/safety_source_points.py
- src/safety_track_fallback.py
- src/safety_track_persistence.py
- src/safety_v10.py
- src/safety_v11.py
- src/safety_v12.py
- src/safety_v13.py
- src/safety_v14.py
- src/safety_v15.py
- src/safety_v16.py
- src/safety_v17.py
- src/safety_v18.py
- src/safety_v19.py
- src/safety_v6.py
- src/safety_yaw_observer.py
- src/scenario.py
- tools/diagnostics/safety/safety_candidate_iteration.py
- tools/diagnostics/safety/v9_paired_pilot.py

Additional fresh reference v18_v19_probe_a[v16]: 6 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Additional fresh reference v18_v19_probe_b[v16]: 6 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Checkpoint SHA-256: `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`. The frozen manifest records filter classes/options; [provenance.json](provenance.json) hashes every read manifest, token, result, completed trace and source archive. Result JSON and traces are authoritative; episodes.csv is not used as evidence.
