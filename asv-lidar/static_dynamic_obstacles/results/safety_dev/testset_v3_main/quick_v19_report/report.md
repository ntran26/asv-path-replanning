# Saved safety-iteration report

All reported iteration records passed completion, identity, trace-counter and archived-source checks; no explicit filter-error fallback was recorded. These finite primary test set v3 simulation episodes do not establish 100% safety, preservation of every SAC success, a population success rate, or real-world performance. No episodes were run to create this report.

Fresh prior-pilot references were executed separately on the same canonical scene and seed with matching checkpoint, config and effective constants. Historical saved-branch references are labeled separately and are not fresh evaluations. Source differences are listed below; identity checks alone do not establish numerical equivalence of changed source or native-thread settings.

Primary benchmark: **test set v3**, containing 1000 frozen scenarios. TS3 identities remain separate from test set v2; unavailable fresh references are left unreported.

- v19_primary_quick18: **subset**, 18/1000 scenarios per evaluated mode.

| Iteration | Mode | N | Goal | Target | Obstacle | Boundary | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|
| v19_primary_quick18 | off | 18 | 14 | 1 | 1 | 2 | 0 |
| v19_primary_quick18 | v16 | 18 | 15 | 1 | 0 | 2 | 0 |
| v19_primary_quick18 | v19 | 18 | 15 | 1 | 0 | 2 | 0 |

| Iteration / mode | Reference | Evidence | Matched | Gained goals | Lost goals | Net |
|---|---|---|---:|---:|---:|---:|
| v19_primary_quick18 / off | v16 | fresh_same_iteration | 18 | 0 | 1 | -1 |
| v19_primary_quick18 / v16 | off | fresh_same_iteration | 18 | 1 | 0 | +1 |
| v19_primary_quick18 / v16 | v16 | fresh_reference_iteration | 18 | 0 | 0 | +0 |
| v19_primary_quick18 / v19 | off | fresh_same_iteration | 18 | 1 | 0 | +1 |
| v19_primary_quick18 / v19 | v16 | fresh_same_iteration | 18 | 0 | 0 | +0 |

A gained goal versus off is a rescued policy failure; a lost goal versus off is a broken policy success. Collision→timeout is not counted as a rescued goal. Exact per-case transitions, source labels, and dataset/stratum denominators are in [paired.csv](paired.csv) and [summary.json](summary.json). Repeated cases across tags are intentional separate controller experiments, not independent replications or extra coverage; they are never pooled into a success-rate estimate.

| Iteration / mode | Selection stratum | SAC evidence | Matched | Preserved / SAC goals | Rescued / SAC failures | Broken SAC goals | Unresolved failures |
|---|---|---|---:|---:|---:|---:|---:|
| v19_primary_quick18 / v16 | frozen|basin-being_overtaken|CV | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | frozen|basin-being_overtaken|RE | fresh_same_iteration | 1 | 0/0 | 0/1 | 0 | 1 |
| v19_primary_quick18 / v16 | frozen|basin-crossing|RE | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | frozen|basin-head_on|RE | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | frozen|basin-null|CV | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | frozen|channel-crossing|CV | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | frozen|channel-head_on|NC | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | frozen|channel-overtaking|CV | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | frozen|channel-overtaking|RE | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | paper2|FS-BO|FIX | fresh_same_iteration | 1 | 0/0 | 0/1 | 0 | 1 |
| v19_primary_quick18 / v16 | paper2|FS-BO|VAR | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | paper2|FS-CRP|VAR | fresh_same_iteration | 1 | 0/0 | 1/1 | 0 | 0 |
| v19_primary_quick18 / v16 | paper2|FS-HO|FIX | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | paper2|FS-HO|VAR | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | paper2|FS-NT| | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | paper2|FS-OT|FIX | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | paper2|FS-OT|VAR | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v16 | paper2|L2-CRS|FIX | fresh_same_iteration | 1 | 0/0 | 0/1 | 0 | 1 |
| v19_primary_quick18 / v19 | frozen|basin-being_overtaken|CV | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | frozen|basin-being_overtaken|RE | fresh_same_iteration | 1 | 0/0 | 0/1 | 0 | 1 |
| v19_primary_quick18 / v19 | frozen|basin-crossing|RE | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | frozen|basin-head_on|RE | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | frozen|basin-null|CV | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | frozen|channel-crossing|CV | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | frozen|channel-head_on|NC | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | frozen|channel-overtaking|CV | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | frozen|channel-overtaking|RE | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | paper2|FS-BO|FIX | fresh_same_iteration | 1 | 0/0 | 0/1 | 0 | 1 |
| v19_primary_quick18 / v19 | paper2|FS-BO|VAR | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | paper2|FS-CRP|VAR | fresh_same_iteration | 1 | 0/0 | 1/1 | 0 | 0 |
| v19_primary_quick18 / v19 | paper2|FS-HO|FIX | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | paper2|FS-HO|VAR | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | paper2|FS-NT| | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | paper2|FS-OT|FIX | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | paper2|FS-OT|VAR | fresh_same_iteration | 1 | 1/1 | 0/0 | 0 | 0 |
| v19_primary_quick18 / v19 | paper2|L2-CRS|FIX | fresh_same_iteration | 1 | 0/0 | 0/1 | 0 | 1 |

Named broken-SAC-success inventory (the comparison evidence is identified for every case):

- None among the matched cases.

Across the included tags there are 54 completed iteration records covering 18 distinct canonical scenarios. Reference-pilot records are comparison evidence and are not added to the new iteration count.

The separate [baseline source audit](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/results/safety_dev/v10_iterations/baseline_source_audit.json>) records the rationale and checks for changed baseline-related sources. This report preserves that audit and the differing hashes; it does not erase the difference.

## v19_primary_quick18

54/54 completed, 54 reserved attempts, 18 distinct scenarios. Timed episodes sum to 862.39 s; sum of component runner durations 871.95 s. Early collisions shorten episodes, so elapsed time does not isolate controller computation speed.

| Mode | Decisions | Changed-action steps | Requested-brake steps |
|---|---:|---:|---:|
| off | 817 | 0 | 0 |
| v16 | 974 | 76 | 22 |
| v19 | 974 | 76 | 22 |

Mechanism events are descriptive checks or handbacks, not proof that an accident was prevented. Requested braking does not imply a particular signed RPM; the trace records the executed command. Missing diagnostics are null/unrecorded, while explicitly recorded false flags contribute zero. Persistence hypothesis-decision counts include repeated predictions of the same vessel; they are not counts of distinct vessels.

- off prefix-request diagnostics: 0 recorded decisions; checked=None; request reasons=None; skip reasons=None.
- off motion-axis geometry: 0 recorded decisions; replacement decisions=None, ID-events=None, episodes with replacements=None. ID-events sum repeated observations across decisions and episodes; they are not distinct vessels.
- v16 prefix-request diagnostics: 974 recorded decisions; checked=130; request reasons={'adequate_parent_clearance': 17, 'ordinary_parent_below_trigger_margin': 47, 'currently_failing_parent': 23, 'missing_paired_check': 39, 'missing_ordinary_selection': 4}; skip reasons={'adequate_parent_clearance': 17, 'missing_paired_check': 39, 'missing_ordinary_selection': 4}.
- v16 motion-axis geometry: 974 recorded decisions; replacement decisions=49, ID-events=49, episodes with replacements=6. ID-events sum repeated observations across decisions and episodes; they are not distinct vessels.
- v19 prefix-request diagnostics: 974 recorded decisions; checked=130; request reasons={'adequate_parent_clearance': 17, 'ordinary_parent_below_trigger_margin': 47, 'currently_failing_parent': 23, 'missing_paired_check': 39, 'missing_ordinary_selection': 4}; skip reasons={'adequate_parent_clearance': 17, 'missing_paired_check': 39, 'missing_ordinary_selection': 4}.
- v19 motion-axis geometry: 974 recorded decisions; replacement decisions=49, ID-events=49, episodes with replacements=6. ID-events sum repeated observations across decisions and episodes; they are not distinct vessels.

This cohort combines disjoint completed slices after checking identical classes, constructor options, checkpoint/config, effective constants, runtime thread settings and every common archived source hash. Optional snapshot-helper inclusion and snapshot logging may differ. Any explicitly allowed inactive-version addition is audited below. Component identity is retained in each CSV row:
- v19_quick_a: 27 records / 9 scenarios; snapshots=False; optional source difference=[].
- v19_quick_b: 27 records / 9 scenarios; snapshots=False; optional source difference=[].

- v16 vs off (fresh_same_iteration): gains TS3:FS-CRP-VAR-023; losses none.
- v19 vs off (fresh_same_iteration): gains TS3:FS-CRP-VAR-023; losses none.

Archived source differences relative to the prior fresh pilot:
- None among compared archive inventories, or no pilot-case overlap.

Additional fresh reference v19_quick_a[v16]: 9 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Additional fresh reference v19_quick_b[v16]: 9 matching scenario identities; its archived class/options and source differences are preserved in summary.json.

Checkpoint SHA-256: `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`. The frozen manifest records filter classes/options; [provenance.json](provenance.json) hashes every read manifest, token, result, completed trace and source archive. Result JSON and traces are authoritative; episodes.csv is not used as evidence.
