# Saved safety-iteration report

All reported iteration records passed completion, identity, trace-counter and archived-source checks; no explicit filter-error fallback was recorded. These finite, deliberately selected development episodes do not establish 100% safety, preservation of every SAC success, a population success rate, or real-world performance. No episodes were run to create this report.

Fresh prior-pilot references were executed separately on the same canonical scene and seed with matching checkpoint, config and effective constants. Historical saved-branch references are labeled separately and are not fresh evaluations. Source differences are listed below; identity checks alone do not establish numerical equivalence of changed source or native-thread settings.

| Iteration | Mode | N | Goal | Target | Obstacle | Boundary | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_thread_check | v8 | 2 | 1 | 1 | 0 | 0 | 0 |
| missing_target_diagnostics | v8 | 3 | 0 | 3 | 0 | 0 | 0 |

| Iteration / mode | Reference | Evidence | Matched | Gained goals | Lost goals | Net |
|---|---|---|---:|---:|---:|---:|
| baseline_thread_check / v8 | off | fresh_prior_pilot | 2 | 1 | 0 | +1 |
| baseline_thread_check / v8 | v8 | fresh_prior_pilot | 2 | 0 | 0 | +0 |
| baseline_thread_check / v8 | v9 | fresh_prior_pilot | 2 | 1 | 0 | +1 |
| missing_target_diagnostics / v8 | off | fresh_prior_pilot | 3 | 0 | 0 | +0 |
| missing_target_diagnostics / v8 | v8 | fresh_prior_pilot | 3 | 0 | 0 | +0 |
| missing_target_diagnostics / v8 | v9 | fresh_prior_pilot | 3 | 0 | 0 | +0 |

A gained goal versus off is a rescued policy failure; a lost goal versus off is a broken policy success. Collision→timeout is not counted as a rescued goal. Exact per-case transitions, source labels, and dataset/stratum denominators are in [paired.csv](paired.csv) and [summary.json](summary.json). Repeated cases across tags are intentional separate controller experiments, not independent replications or extra coverage; they are never pooled into a success-rate estimate.

Across the included tags there are 5 completed iteration records covering 4 distinct canonical scenarios. Reference-pilot records are comparison evidence and are not added to the new iteration count.

The separate [baseline source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/baseline_source_audit.json>) records the rationale and checks for changed baseline-related sources. This report preserves that audit and the differing hashes; it does not erase the difference.

## baseline_thread_check

2/2 completed, 2 reserved attempts, 2 distinct scenarios. Timed episodes sum to 138.38 s; runner wall time 138.60 s. Early collisions shorten episodes, so elapsed time does not isolate controller computation speed.

| Mode | Decisions | Changed-action steps | Requested-brake steps |
|---|---:|---:|---:|
| v8 | 156 | 32 | 9 |

Mechanism events are descriptive checks or handbacks, not proof that an accident was prevented. Requested braking does not imply a particular signed RPM; the trace records the executed command. Missing diagnostics are null/unrecorded, while explicitly recorded false flags contribute zero. Persistence hypothesis-decision counts include repeated predictions of the same vessel; they are not counts of distinct vessels.

- v8 vs off (fresh_prior_pilot): gains TS2:CH-CR-CV-073; losses none.
- v8 vs v8 (fresh_prior_pilot): gains none; losses none.

Archived source differences relative to the prior fresh pilot:
- src/field_training.py
- tools/diagnostics/safety/safety_iteration.py

Checkpoint SHA-256: `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`. The frozen manifest records filter classes/options; [provenance.json](provenance.json) hashes every read manifest, token, result, completed trace and source archive. Result JSON and traces are authoritative; episodes.csv is not used as evidence.

## missing_target_diagnostics

3/3 completed, 3 reserved attempts, 3 distinct scenarios. Timed episodes sum to 23.96 s; runner wall time 24.31 s. Early collisions shorten episodes, so elapsed time does not isolate controller computation speed.

| Mode | Decisions | Changed-action steps | Requested-brake steps |
|---|---:|---:|---:|
| v8 | 56 | 0 | 0 |

Mechanism events are descriptive checks or handbacks, not proof that an accident was prevented. Requested braking does not imply a particular signed RPM; the trace records the executed command. Missing diagnostics are null/unrecorded, while explicitly recorded false flags contribute zero. Persistence hypothesis-decision counts include repeated predictions of the same vessel; they are not counts of distinct vessels.

- v8 vs off (fresh_prior_pilot): gains none; losses none.
- v8 vs v8 (fresh_prior_pilot): gains none; losses none.

Archived source differences relative to the prior fresh pilot:
- src/field_training.py
- tools/diagnostics/safety/safety_iteration.py
- tools/diagnostics/safety/trace_snapshot.py

Checkpoint SHA-256: `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`. The frozen manifest records filter classes/options; [provenance.json](provenance.json) hashes every read manifest, token, result, completed trace and source archive. Result JSON and traces are authoritative; episodes.csv is not used as evidence.
