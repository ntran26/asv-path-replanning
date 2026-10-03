# Saved safety-iteration report

All reported iteration records passed completion, identity, trace-counter and archived-source checks; no explicit filter-error fallback was recorded. These finite, deliberately selected development episodes do not establish 100% safety, preservation of every SAC success, a population success rate, or real-world performance. No episodes were run to create this report.

Fresh prior-pilot references were executed separately on the same canonical scene and seed with matching checkpoint, config and effective constants. Historical saved-branch references are labeled separately and are not fresh evaluations. Source differences are listed below; identity checks alone do not establish numerical equivalence of changed source or native-thread settings.

| Iteration | Mode | N | Goal | Target | Obstacle | Boundary | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|
| conditional_prefix32 | v15 | 32 | 23 | 6 | 2 | 1 | 0 |

| Iteration / mode | Reference | Evidence | Matched | Gained goals | Lost goals | Net |
|---|---|---|---:|---:|---:|---:|
| conditional_prefix32 / v15 | off | fresh_prior_pilot | 32 | 11 | 4 | +7 |
| conditional_prefix32 / v15 | v8 | fresh_prior_pilot | 32 | 8 | 1 | +7 |
| conditional_prefix32 / v15 | v9 | fresh_prior_pilot | 32 | 9 | 2 | +7 |
| conditional_prefix32 / v15 | v14 | fresh_reference_iteration | 32 | 3 | 2 | +1 |
| conditional_prefix32 / v15 | v14_prefix | fresh_reference_iteration | 32 | 3 | 1 | +2 |
| conditional_prefix32 / v15 | v10 | fresh_reference_iteration | 32 | 6 | 1 | +5 |

A gained goal versus off is a rescued policy failure; a lost goal versus off is a broken policy success. Collision→timeout is not counted as a rescued goal. Exact per-case transitions, source labels, and dataset/stratum denominators are in [paired.csv](paired.csv) and [summary.json](summary.json). Repeated cases across tags are intentional separate controller experiments, not independent replications or extra coverage; they are never pooled into a success-rate estimate.

Across the included tags there are 32 completed iteration records covering 32 distinct canonical scenarios. Reference-pilot records are comparison evidence and are not added to the new iteration count.

The separate [baseline source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/baseline_source_audit.json>) records the rationale and checks for changed baseline-related sources. This report preserves that audit and the differing hashes; it does not erase the difference.

The [native-dispatch source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/native_dispatch_source_audit.json>) records the later environment-selector integration. The evaluated archives remain authoritative; this audit is linked because an environment source hash differs across the compared runs.

The [V15 integration source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/v15_integration_source_audit.json>) addresses recorded source differences across these runs. Its hash is preserved in provenance; the evaluated source archives remain authoritative.

The [V6 diagnostic-only AST audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/v15_v6_diagnostic_ast_audit.json>) addresses recorded source differences across these runs. Its hash is preserved in provenance; the evaluated source archives remain authoritative.

## conditional_prefix32

32/32 completed, 32 reserved attempts, 32 distinct scenarios. Timed episodes sum to 586.61 s; runner wall time 591.18 s. Early collisions shorten episodes, so elapsed time does not isolate controller computation speed.

| Mode | Decisions | Changed-action steps | Requested-brake steps |
|---|---:|---:|---:|
| v15 | 1682 | 168 | 68 |

Mechanism events are descriptive checks or handbacks, not proof that an accident was prevented. Requested braking does not imply a particular signed RPM; the trace records the executed command. Missing diagnostics are null/unrecorded, while explicitly recorded false flags contribute zero. Persistence hypothesis-decision counts include repeated predictions of the same vessel; they are not counts of distinct vessels.

- v15 prefix-request diagnostics: 1682 recorded decisions; checked=344; request reasons={'ordinary_parent_below_trigger_margin': 96, 'adequate_parent_clearance': 39, 'currently_failing_parent': 90, 'missing_paired_check': 108, 'missing_ordinary_selection': 11}; skip reasons={'adequate_parent_clearance': 39, 'missing_paired_check': 108, 'missing_ordinary_selection': 11}.
- v15 motion-axis geometry: 0 recorded decisions; replacement decisions=None, ID-events=None, episodes with replacements=None. ID-events sum repeated observations across decisions and episodes; they are not distinct vessels.

- v15 vs off (fresh_prior_pilot): gains DV3:DV3-CRP-CV-04, DV3:DV3-CRP-CV-17, DV3:DV3-HO-CV-03, TS2:CH-CR-CV-073, TS2:CH-HO-RE-060, TS2:P2-L1-CRP-VAR-12, TS2:P2-L1-CRP-VAR-14, TS2:P2-L1-OT-VAR-20, TS2:P2-L2-CRP-VAR-02, TS2:P2-L2-HO-FIX-05, TS2:P2-L2-HO-FIX-19; losses DV3:DV3-BO-CV-04, TS2:BAS-HO-NC-059, TS2:BAS-NU-CV-070, TS2:CH-HO-CV-073.
- v15 vs v8 (fresh_prior_pilot): gains TS2:BAS-BO-CV-063, TS2:BAS-CR-RE-072, TS2:CH-CR-CV-007, TS2:CH-CR-CV-031, TS2:CH-CR-RE-038, TS2:P2-L1-CRP-VAR-14, TS2:P2-L2-CRP-VAR-16, TS2:P2-L2-HO-FIX-19; losses DV3:DV3-HO-VS-01.
- v15 vs v10 (fresh_reference_iteration): gains TS2:BAS-BO-CV-063, TS2:CH-CR-CV-007, TS2:CH-CR-CV-031, TS2:CH-CR-RE-038, TS2:P2-L1-CRP-VAR-14, TS2:P2-L2-CRP-VAR-16; losses DV3:DV3-HO-VS-01.

Archived source differences relative to the prior fresh pilot:
- src/dev_set_v4.py
- src/env.py
- src/field_training.py
- src/safety_consistent_tracks.py
- src/safety_fast_geometry.py
- src/safety_policy_prefix.py
- src/safety_track_fallback.py
- src/safety_track_persistence.py
- src/safety_v10.py
- src/safety_v11.py
- src/safety_v12.py
- src/safety_v13.py
- src/safety_v14.py
- src/safety_v15.py
- src/safety_v6.py
- src/scenario.py
- tools/diagnostics/safety/safety_candidate_iteration.py
- tools/diagnostics/safety/trace_snapshot.py

Additional fresh reference persistent_preference_probe4: 4 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Additional fresh reference persistent_preference_remaining28: 28 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Additional fresh reference persistent_prefix32: 32 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Additional fresh reference conditional_policy_32: 32 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Checkpoint SHA-256: `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`. The frozen manifest records filter classes/options; [provenance.json](provenance.json) hashes every read manifest, token, result, completed trace and source archive. Result JSON and traces are authoritative; episodes.csv is not used as evidence.
