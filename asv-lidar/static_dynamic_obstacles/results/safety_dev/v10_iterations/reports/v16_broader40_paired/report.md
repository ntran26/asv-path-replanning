# Saved safety-iteration report

All reported iteration records passed completion, identity, trace-counter and archived-source checks; no explicit filter-error fallback was recorded. These finite, deliberately selected development episodes do not establish 100% safety, preservation of every SAC success, a population success rate, or real-world performance. No episodes were run to create this report.

Fresh prior-pilot references were executed separately on the same canonical scene and seed with matching checkpoint, config and effective constants. Historical saved-branch references are labeled separately and are not fresh evaluations. Source differences are listed below; identity checks alone do not establish numerical equivalence of changed source or native-thread settings.

| Iteration | Mode | N | Goal | Target | Obstacle | Boundary | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|
| v16_broader40_paired | off | 40 | 21 | 8 | 10 | 1 | 0 |
| v16_broader40_paired | v16 | 40 | 25 | 5 | 4 | 6 | 0 |

| Iteration / mode | Reference | Evidence | Matched | Gained goals | Lost goals | Net |
|---|---|---|---:|---:|---:|---:|
| v16_broader40_paired / off | off | historical_saved_branch | 40 | 0 | 0 | +0 |
| v16_broader40_paired / off | v8 | historical_saved_branch | 40 | 11 | 11 | +0 |
| v16_broader40_paired / v16 | off | fresh_same_iteration | 40 | 8 | 4 | +4 |
| v16_broader40_paired / v16 | v8 | historical_saved_branch | 40 | 7 | 3 | +4 |

A gained goal versus off is a rescued policy failure; a lost goal versus off is a broken policy success. Collision→timeout is not counted as a rescued goal. Exact per-case transitions, source labels, and dataset/stratum denominators are in [paired.csv](paired.csv) and [summary.json](summary.json). Repeated cases across tags are intentional separate controller experiments, not independent replications or extra coverage; they are never pooled into a success-rate estimate.

| Iteration / mode | Selection stratum | SAC evidence | Matched | Preserved / SAC goals | Rescued / SAC failures | Broken SAC goals | Unresolved failures |
|---|---|---|---:|---:|---:|---:|---:|
| v16_broader40_paired / v16 | both_fail_active_dv3 | fresh_same_iteration | 1 | 0/0 | 0/1 | 0 | 1 |
| v16_broader40_paired / v16 | both_fail_active_ts2 | fresh_same_iteration | 3 | 0/0 | 0/3 | 0 | 3 |
| v16_broader40_paired / v16 | both_fail_no_fire_dv3 | fresh_same_iteration | 1 | 0/0 | 0/1 | 0 | 1 |
| v16_broader40_paired / v16 | both_fail_no_fire_ts2 | fresh_same_iteration | 3 | 0/0 | 0/3 | 0 | 3 |
| v16_broader40_paired / v16 | both_goal_active_dv3 | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v16_broader40_paired / v16 | both_goal_active_ts2 | fresh_same_iteration | 4 | 4/4 | 0/0 | 0 | 0 |
| v16_broader40_paired / v16 | both_goal_no_fire_dv3 | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v16_broader40_paired / v16 | both_goal_no_fire_ts2 | fresh_same_iteration | 4 | 4/4 | 0/0 | 0 | 0 |
| v16_broader40_paired / v16 | remaining_known_sac_regression | fresh_same_iteration | 11 | 7/11 | 0/0 | 4 | 0 |
| v16_broader40_paired / v16 | rescue_control | fresh_same_iteration | 11 | 0/0 | 8/11 | 0 | 3 |

Named broken-SAC-success inventory (the comparison evidence is identified for every case):

- v16_broader40_paired / v16: DV3:DV3-CRS-CV-04 → collision:boundary (remaining_known_sac_regression; fresh_same_iteration).
- v16_broader40_paired / v16: TS2:P2-L2-CRS-FIX-19 → collision:boundary (remaining_known_sac_regression; fresh_same_iteration).
- v16_broader40_paired / v16: TS2:P2-L3-BO-FIX-09 → collision:boundary (remaining_known_sac_regression; fresh_same_iteration).
- v16_broader40_paired / v16: TS2:P2-L3-CRP-FIX-03 → collision:target (remaining_known_sac_regression; fresh_same_iteration).

Across the included tags there are 80 completed iteration records covering 40 distinct canonical scenarios. Reference-pilot records are comparison evidence and are not added to the new iteration count.

The separate [baseline source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/baseline_source_audit.json>) records the rationale and checks for changed baseline-related sources. This report preserves that audit and the differing hashes; it does not erase the difference.

## v16_broader40_paired

80/80 completed, 80 reserved attempts, 40 distinct scenarios. Timed episodes sum to 909.46 s; runner wall time 915.22 s. Early collisions shorten episodes, so elapsed time does not isolate controller computation speed.

| Mode | Decisions | Changed-action steps | Requested-brake steps |
|---|---:|---:|---:|
| off | 1541 | 0 | 0 |
| v16 | 2119 | 242 | 90 |

Mechanism events are descriptive checks or handbacks, not proof that an accident was prevented. Requested braking does not imply a particular signed RPM; the trace records the executed command. Missing diagnostics are null/unrecorded, while explicitly recorded false flags contribute zero. Persistence hypothesis-decision counts include repeated predictions of the same vessel; they are not counts of distinct vessels.

- off prefix-request diagnostics: 0 recorded decisions; checked=None; request reasons=None; skip reasons=None.
- off motion-axis geometry: 0 recorded decisions; replacement decisions=None, ID-events=None, episodes with replacements=None. ID-events sum repeated observations across decisions and episodes; they are not distinct vessels.
- v16 prefix-request diagnostics: 2119 recorded decisions; checked=456; request reasons={'adequate_parent_clearance': 60, 'ordinary_parent_below_trigger_margin': 151, 'missing_ordinary_selection': 14, 'currently_failing_parent': 92, 'missing_paired_check': 139}; skip reasons={'adequate_parent_clearance': 60, 'missing_ordinary_selection': 14, 'missing_paired_check': 139}.
- v16 motion-axis geometry: 2119 recorded decisions; replacement decisions=54, ID-events=54, episodes with replacements=9. ID-events sum repeated observations across decisions and episodes; they are not distinct vessels.

- off vs off (historical_saved_branch): gains none; losses none.
- off vs v8 (historical_saved_branch): gains DV3:DV3-CRP-CV-16, DV3:DV3-CRS-CV-04, DV3:DV3-CRS-CV-12, TS2:BAS-CR-CV-045, TS2:BAS-CR-RE-070, TS2:BAS-CR-RE-090, TS2:BAS-OT-CV-095, TS2:CH-OT-RE-064, TS2:P2-L2-CRS-FIX-19, TS2:P2-L3-BO-FIX-09, TS2:P2-L3-CRP-FIX-03; losses DV3:DV3-CRP-CV-12, DV3:DV3-CRS-CV-01, DV3:DV3-CRS-CV-10, TS2:BAS-BO-CV-055, TS2:BAS-CR-RE-022, TS2:BAS-OT-CV-087, TS2:P2-L1-CRP-FIX-09, TS2:P2-L1-CRP-FIX-15, TS2:P2-L2-CRS-VAR-14, TS2:P2-L3-CRP-FIX-17, TS2:P2-L3-CRP-FIX-19.
- v16 vs off (fresh_same_iteration): gains DV3:DV3-CRP-CV-12, DV3:DV3-CRS-CV-01, DV3:DV3-CRS-CV-10, TS2:BAS-BO-CV-055, TS2:BAS-OT-CV-087, TS2:P2-L1-CRP-FIX-09, TS2:P2-L1-CRP-FIX-15, TS2:P2-L3-CRP-FIX-19; losses DV3:DV3-CRS-CV-04, TS2:P2-L2-CRS-FIX-19, TS2:P2-L3-BO-FIX-09, TS2:P2-L3-CRP-FIX-03.
- v16 vs v8 (historical_saved_branch): gains DV3:DV3-CRP-CV-16, DV3:DV3-CRS-CV-12, TS2:BAS-CR-CV-045, TS2:BAS-CR-RE-070, TS2:BAS-CR-RE-090, TS2:BAS-OT-CV-095, TS2:CH-OT-RE-064; losses TS2:BAS-CR-RE-022, TS2:P2-L2-CRS-VAR-14, TS2:P2-L3-CRP-FIX-17.

Archived source differences relative to the prior fresh pilot:
- None among compared archive inventories, or no pilot-case overlap.

Checkpoint SHA-256: `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`. The frozen manifest records filter classes/options; [provenance.json](provenance.json) hashes every read manifest, token, result, completed trace and source archive. Result JSON and traces are authoritative; episodes.csv is not used as evidence.
