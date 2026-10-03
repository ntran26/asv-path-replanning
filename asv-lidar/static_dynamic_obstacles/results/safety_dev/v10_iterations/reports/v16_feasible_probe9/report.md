# Saved safety-iteration report

All reported iteration records passed completion, identity, trace-counter and archived-source checks; no explicit filter-error fallback was recorded. These finite, deliberately selected development episodes do not establish 100% safety, preservation of every SAC success, a population success rate, or real-world performance. No episodes were run to create this report.

Fresh prior-pilot references were executed separately on the same canonical scene and seed with matching checkpoint, config and effective constants. Historical saved-branch references are labeled separately and are not fresh evaluations. Source differences are listed below; identity checks alone do not establish numerical equivalence of changed source or native-thread settings.

| Iteration | Mode | N | Goal | Target | Obstacle | Boundary | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|
| v16_feasible_probe9 | v16 | 9 | 3 | 2 | 2 | 2 | 0 |

| Iteration / mode | Reference | Evidence | Matched | Gained goals | Lost goals | Net |
|---|---|---|---:|---:|---:|---:|
| v16_feasible_probe9 / v16 | off | fresh_prior_pilot | 9 | 0 | 2 | -2 |
| v16_feasible_probe9 / v16 | v8 | fresh_prior_pilot | 9 | 3 | 3 | +0 |
| v16_feasible_probe9 / v16 | v9 | fresh_prior_pilot | 9 | 1 | 3 | -2 |
| v16_feasible_probe9 / v16 | v16_default | fresh_reference_iteration | 9 | 0 | 3 | -3 |

A gained goal versus off is a rescued policy failure; a lost goal versus off is a broken policy success. Collision→timeout is not counted as a rescued goal. Exact per-case transitions, source labels, and dataset/stratum denominators are in [paired.csv](paired.csv) and [summary.json](summary.json). Repeated cases across tags are intentional separate controller experiments, not independent replications or extra coverage; they are never pooled into a success-rate estimate.

| Iteration / mode | Selection stratum | SAC evidence | Matched | Preserved / SAC goals | Rescued / SAC failures | Broken SAC goals | Unresolved failures |
|---|---|---|---:|---:|---:|---:|---:|
| v16_feasible_probe9 / v16 | expired_consequences | fresh_prior_pilot | 5 | 2/3 | 0/2 | 1 | 2 |
| v16_feasible_probe9 / v16 | other_broken_boundary | fresh_prior_pilot | 1 | 0/1 | 0/0 | 1 | 0 |
| v16_feasible_probe9 / v16 | other_broken_target | fresh_prior_pilot | 1 | 1/1 | 0/0 | 0 | 0 |
| v16_feasible_probe9 / v16 | rescued_searched_escape | fresh_prior_pilot | 1 | 0/0 | 0/1 | 0 | 1 |
| v16_feasible_probe9 / v16 | unrescued_other_active | fresh_prior_pilot | 1 | 0/0 | 0/1 | 0 | 1 |

Named broken-SAC-success inventory (the comparison evidence is identified for every case):

- v16_feasible_probe9 / v16: TS2:BAS-HO-NC-059 → collision:target (expired_consequences; fresh_prior_pilot).
- v16_feasible_probe9 / v16: TS2:BAS-NU-CV-070 → collision:boundary (other_broken_boundary; fresh_prior_pilot).

Across the included tags there are 9 completed iteration records covering 9 distinct canonical scenarios. Reference-pilot records are comparison evidence and are not added to the new iteration count.

The separate [baseline source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/baseline_source_audit.json>) records the rationale and checks for changed baseline-related sources. This report preserves that audit and the differing hashes; it does not erase the difference.

The [native-dispatch source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/native_dispatch_source_audit.json>) records the later environment-selector integration. The evaluated archives remain authoritative; this audit is linked because an environment source hash differs across the compared runs.

The [V15 integration source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/v15_integration_source_audit.json>) addresses recorded source differences across these runs. Its hash is preserved in provenance; the evaluated source archives remain authoritative.

The [V6 diagnostic-only AST audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/v15_v6_diagnostic_ast_audit.json>) addresses recorded source differences across these runs. Its hash is preserved in provenance; the evaluated source archives remain authoritative.

## v16_feasible_probe9

9/9 completed, 9 reserved attempts, 9 distinct scenarios. Timed episodes sum to 193.30 s; runner wall time 193.94 s. Early collisions shorten episodes, so elapsed time does not isolate controller computation speed.

| Mode | Decisions | Changed-action steps | Requested-brake steps |
|---|---:|---:|---:|
| v16 | 397 | 41 | 25 |

Mechanism events are descriptive checks or handbacks, not proof that an accident was prevented. Requested braking does not imply a particular signed RPM; the trace records the executed command. Missing diagnostics are null/unrecorded, while explicitly recorded false flags contribute zero. Persistence hypothesis-decision counts include repeated predictions of the same vessel; they are not counts of distinct vessels.

- v16 prefix-request diagnostics: 397 recorded decisions; checked=99; request reasons={'ordinary_parent_below_trigger_margin': 19, 'adequate_parent_clearance': 6, 'currently_failing_parent': 27, 'missing_paired_check': 44, 'missing_ordinary_selection': 3}; skip reasons={'adequate_parent_clearance': 6, 'missing_paired_check': 44, 'missing_ordinary_selection': 3}.
- v16 motion-axis geometry: 397 recorded decisions; replacement decisions=59, ID-events=59, episodes with replacements=7. ID-events sum repeated observations across decisions and episodes; they are not distinct vessels.

- v16 vs off (fresh_prior_pilot): gains none; losses TS2:BAS-HO-NC-059, TS2:BAS-NU-CV-070.
- v16 vs v8 (fresh_prior_pilot): gains DV3:DV3-BO-CV-04, TS2:BAS-CR-RE-072, TS2:CH-HO-CV-073; losses DV3:DV3-HO-CV-03, DV3:DV3-HO-VS-01, TS2:P2-L1-CRP-VAR-12.

Archived source differences relative to the prior fresh pilot:
- src/dev_set_v4.py
- src/env.py
- src/field_training.py
- src/safety_consistent_tracks.py
- src/safety_fast_geometry.py
- src/safety_motion_axis.py
- src/safety_policy_prefix.py
- src/safety_track_fallback.py
- src/safety_track_persistence.py
- src/safety_v10.py
- src/safety_v11.py
- src/safety_v12.py
- src/safety_v13.py
- src/safety_v14.py
- src/safety_v15.py
- src/safety_v16.py
- src/safety_v6.py
- src/scenario.py
- tools/diagnostics/safety/safety_candidate_iteration.py
- tools/diagnostics/safety/trace_snapshot.py

Additional fresh reference motion_axis_probe5: 4 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Additional fresh reference motion_axis_remaining27: 5 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Checkpoint SHA-256: `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`. The frozen manifest records filter classes/options; [provenance.json](provenance.json) hashes every read manifest, token, result, completed trace and source archive. Result JSON and traces are authoritative; episodes.csv is not used as evidence.
