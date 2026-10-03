# Saved safety-iteration report

All reported iteration records passed completion, identity, trace-counter and archived-source checks; no explicit filter-error fallback was recorded. These finite, deliberately selected development episodes do not establish 100% safety, preservation of every SAC success, a population success rate, or real-world performance. No episodes were run to create this report.

Fresh prior-pilot references were executed separately on the same canonical scene and seed with matching checkpoint, config and effective constants. Historical saved-branch references are labeled separately and are not fresh evaluations. Source differences are listed below; identity checks alone do not establish numerical equivalence of changed source or native-thread settings.

| Iteration | Mode | N | Goal | Target | Obstacle | Boundary | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|
| persistent_preference32 | v14 | 32 | 22 | 5 | 2 | 3 | 0 |

| Iteration / mode | Reference | Evidence | Matched | Gained goals | Lost goals | Net |
|---|---|---|---:|---:|---:|---:|
| persistent_preference32 / v14 | off | fresh_prior_pilot | 32 | 11 | 5 | +6 |
| persistent_preference32 / v14 | v8 | fresh_prior_pilot | 32 | 7 | 1 | +6 |
| persistent_preference32 / v14 | v9 | fresh_prior_pilot | 32 | 10 | 4 | +6 |
| persistent_preference32 / v14 | v10 | fresh_reference_iteration | 32 | 6 | 2 | +4 |
| persistent_preference32 / v14 | v11 | fresh_reference_iteration | 32 | 5 | 1 | +4 |

A gained goal versus off is a rescued policy failure; a lost goal versus off is a broken policy success. Collision→timeout is not counted as a rescued goal. Exact per-case transitions, source labels, and dataset/stratum denominators are in [paired.csv](paired.csv) and [summary.json](summary.json). Repeated cases across tags are intentional separate controller experiments, not independent replications or extra coverage; they are never pooled into a success-rate estimate.

Across the included tags there are 32 completed iteration records covering 32 distinct canonical scenarios. Reference-pilot records are comparison evidence and are not added to the new iteration count.

The separate [baseline source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/baseline_source_audit.json>) records the rationale and checks for changed baseline-related sources. This report preserves that audit and the differing hashes; it does not erase the difference.

The [native-dispatch source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/native_dispatch_source_audit.json>) records the later environment-selector integration. The evaluated archives remain authoritative; this audit is linked because an environment source hash differs across the compared runs.

## persistent_preference32

32/32 completed, 32 reserved attempts, 32 distinct scenarios. Timed episodes sum to 667.13 s; sum of component runner durations 669.36 s. Early collisions shorten episodes, so elapsed time does not isolate controller computation speed.

| Mode | Decisions | Changed-action steps | Requested-brake steps |
|---|---:|---:|---:|
| v14 | 1869 | 368 | 122 |

Mechanism events are descriptive checks or handbacks, not proof that an accident was prevented. Requested braking does not imply a particular signed RPM; the trace records the executed command. Missing diagnostics are null/unrecorded, while explicitly recorded false flags contribute zero. Persistence hypothesis-decision counts include repeated predictions of the same vessel; they are not counts of distinct vessels.

This cohort combines disjoint completed slices after checking identical classes, constructor options, checkpoint/config, effective constants, runtime thread settings and every common archived source hash. Optional snapshot-helper inclusion and snapshot logging may differ. Any explicitly allowed inactive-version addition is audited below. Component identity is retained in each CSV row:
- persistent_preference_probe4: 4 records / 4 scenarios; snapshots=True; optional source difference=[].
- persistent_preference_remaining28: 28 records / 28 scenarios; snapshots=False; optional source difference=['tools/diagnostics/safety/trace_snapshot.py'].

- v14 vs off (fresh_prior_pilot): gains DV3:DV3-CRP-CV-04, DV3:DV3-CRP-CV-17, DV3:DV3-HO-CV-03, DV3:DV3-HO-VS-01, TS2:CH-CR-CV-073, TS2:CH-HO-RE-060, TS2:P2-L1-CRP-VAR-12, TS2:P2-L1-CRP-VAR-14, TS2:P2-L1-CRS-VAR-08, TS2:P2-L1-OT-VAR-20, TS2:P2-L2-HO-FIX-05; losses DV3:DV3-BO-CV-04, TS2:BAS-HO-NC-059, TS2:BAS-NU-CV-070, TS2:CH-HO-CV-073, TS2:P2-L2-CRP-VAR-16.
- v14 vs v8 (fresh_prior_pilot): gains TS2:BAS-BO-CV-063, TS2:BAS-CR-RE-072, TS2:CH-CR-CV-007, TS2:CH-CR-CV-031, TS2:CH-CR-RE-038, TS2:P2-L1-CRP-VAR-14, TS2:P2-L1-CRS-VAR-08; losses TS2:P2-L2-CRP-VAR-02.
- v14 vs v10 (fresh_reference_iteration): gains TS2:BAS-BO-CV-063, TS2:CH-CR-CV-007, TS2:CH-CR-CV-031, TS2:CH-CR-RE-038, TS2:P2-L1-CRP-VAR-14, TS2:P2-L1-CRS-VAR-08; losses TS2:P2-L2-CRP-VAR-02, TS2:P2-L2-HO-FIX-19.

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
- src/scenario.py
- tools/diagnostics/safety/safety_candidate_iteration.py
- tools/diagnostics/safety/trace_snapshot.py

Additional fresh reference conditional_policy_32: 32 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Additional fresh reference persistence_three: 3 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Additional fresh reference persistence_remaining29: 29 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Checkpoint SHA-256: `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`. The frozen manifest records filter classes/options; [provenance.json](provenance.json) hashes every read manifest, token, result, completed trace and source archive. Result JSON and traces are authoritative; episodes.csv is not used as evidence.
