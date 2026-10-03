# Saved safety-iteration report

All reported iteration records passed completion, identity, trace-counter and archived-source checks; no explicit filter-error fallback was recorded. These finite, deliberately selected development episodes do not establish 100% safety, preservation of every SAC success, a population success rate, or real-world performance. No episodes were run to create this report.

Fresh prior-pilot references were executed separately on the same canonical scene and seed with matching checkpoint, config and effective constants. Historical saved-branch references are labeled separately and are not fresh evaluations. Source differences are listed below; identity checks alone do not establish numerical equivalence of changed source or native-thread settings.

| Iteration | Mode | N | Goal | Target | Obstacle | Boundary | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|
| persistence_three | v11 | 3 | 2 | 1 | 0 | 0 | 0 |

| Iteration / mode | Reference | Evidence | Matched | Gained goals | Lost goals | Net |
|---|---|---|---:|---:|---:|---:|
| persistence_three / v11 | off | fresh_prior_pilot | 3 | 2 | 0 | +2 |
| persistence_three / v11 | v8 | fresh_prior_pilot | 3 | 2 | 0 | +2 |
| persistence_three / v11 | v9 | fresh_prior_pilot | 3 | 2 | 0 | +2 |

A gained goal versus off is a rescued policy failure; a lost goal versus off is a broken policy success. Collision→timeout is not counted as a rescued goal. Exact per-case transitions, source labels, and dataset/stratum denominators are in [paired.csv](paired.csv) and [summary.json](summary.json). Repeated cases across tags are intentional separate controller experiments, not independent replications or extra coverage; they are never pooled into a success-rate estimate.

Across the included tags there are 3 completed iteration records covering 3 distinct canonical scenarios. Reference-pilot records are comparison evidence and are not added to the new iteration count.

The separate [baseline source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/baseline_source_audit.json>) records the rationale and checks for changed baseline-related sources. This report preserves that audit and the differing hashes; it does not erase the difference.

## persistence_three

3/3 completed, 3 reserved attempts, 3 distinct scenarios. Timed episodes sum to 149.48 s; runner wall time 149.88 s. Early collisions shorten episodes, so elapsed time does not isolate controller computation speed.

| Mode | Decisions | Changed-action steps | Requested-brake steps |
|---|---:|---:|---:|
| v11 | 195 | 49 | 23 |

Mechanism events are descriptive checks or handbacks, not proof that an accident was prevented. Requested braking does not imply a particular signed RPM; the trace records the executed command. Missing diagnostics are null/unrecorded, while explicitly recorded false flags contribute zero. Persistence hypothesis-decision counts include repeated predictions of the same vessel; they are not counts of distinct vessels.

- v11 vs off (fresh_prior_pilot): gains TS2:P2-L1-CRP-VAR-14, TS2:P2-L1-CRS-VAR-08; losses none.
- v11 vs v8 (fresh_prior_pilot): gains TS2:P2-L1-CRP-VAR-14, TS2:P2-L1-CRS-VAR-08; losses none.

Archived source differences relative to the prior fresh pilot:
- src/field_training.py
- src/safety_fast_geometry.py
- src/safety_policy_prefix.py
- src/safety_track_persistence.py
- src/safety_v10.py
- src/safety_v11.py
- tools/diagnostics/safety/safety_v11_iteration.py
- tools/diagnostics/safety/trace_snapshot.py

Checkpoint SHA-256: `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`. The frozen manifest records filter classes/options; [provenance.json](provenance.json) hashes every read manifest, token, result, completed trace and source archive. Result JSON and traces are authoritative; episodes.csv is not used as evidence.
