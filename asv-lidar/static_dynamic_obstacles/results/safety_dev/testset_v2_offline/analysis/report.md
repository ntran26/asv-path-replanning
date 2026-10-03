# Saved SAC test-set v2 analysis

SAC baseline 3 (3M checkpoint): **872/1000 goals (87.2%)**, with 61 target, 55 obstacle and 12 boundary collisions; 0 timeouts. All saved rows have safety off and zero intervention counters.

This report reads saved data only. No simulator reset, policy inference or new episode was performed. The user has authorized development on this set; any resulting tuning must be described as development, not an untouched test of the tuned filter.

## Provenance and limits

The definition, cached scenario digests, seeds and 1,000 unique result IDs agree. Manifest SHA256: `6df8076223414a88ee37a130c2261c7e26b45efba621f34fe024d4065c2cea8c`. The recorded checkpoint path is `runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip`. These results do not archive checkpoint, source or runtime hashes, so the path alone cannot establish exact current-runtime equivalence.

The set combines simulated frozen-suite scenarios and simulated deployment layouts. It is not a real-world vessel trial. Variant twins were collapsed, then geometrically near-duplicate cases removed within cells: 2,330 source runs became 1,103 underlying scenarios, then 1,000. These are mixed, deliberately selected cases; no population weighting, confidence interval or causal effect is inferred.

Version 2 replaced 15 L1 being-overtaken cases that began in obstacle contact, using a corrected start and a 0.30 m panel-clearance requirement for regenerated cases. The 985 shared v1 results agree numerically within 1e-12 after normalizing the blank/null class. The evaluation script reuses matching v1 rows; its recorded runtime must not be interpreted as timing for 1,000 fresh simulations.

## Outcome groups

Every rate uses the stated row count, including all collision types and timeouts in its denominator.

| Group | N | Goals | Target | Obstacle | Boundary | Timeout | Goal rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall: all | 1000 | 872 | 61 | 55 | 12 | 0 | 87.2% |
| source: frozen | 761 | 709 | 33 | 14 | 5 | 0 | 93.2% |
| source: paper2 | 239 | 163 | 28 | 41 | 7 | 0 | 68.2% |
| scenario_class: being_overtaken | 140 | 134 | 2 | 2 | 2 | 0 | 95.7% |
| scenario_class: crossing | 301 | 225 | 46 | 28 | 2 | 0 | 74.8% |
| scenario_class: head_on | 243 | 214 | 7 | 17 | 5 | 0 | 88.1% |
| scenario_class: no_target | 3 | 3 | 0 | 0 | 0 | 0 | 100.0% |
| scenario_class: null | 75 | 72 | 0 | 0 | 3 | 0 | 96.0% |
| scenario_class: overtaking | 238 | 224 | 6 | 8 | 0 | 0 | 94.1% |

The simulated layout subset has 76 of the 128 failures despite containing 239 of 1000 cases. Crossing cases account for 76 failures and 46 of the 61 target contacts. L2 head-on has 0/15 goals: 12 obstacle contacts and 3 boundary contacts. This is a geometry/encounter concentration worth investigating, not evidence that a target-directed turn caused each collision. See groups.csv for all cells and variants.

Paper2 FIX has 88/121 goals; VAR has 72/115. These are different retained scenarios, not matched twins; the difference cannot be attributed to speed variation alone. Nominal width also does not measure the available gap between panels.

## Available features and leakage

`scenario_features.csv` contains all 1,000 cases, with nominal pose/path geometry, width, target spawn, speed, DCPA/TCPA, crossing angle and obstacle-count metadata. Scenario truth is suitable for offline stratification; it is not automatically available to an onboard filter. Own spawn is the scenario's nominal start; `own_start_s_override` records an explicit reset-path offset when present.

Fixed deployment panels are cached in `flags.fixed_obstacles`, not `built.obstacles`. Frozen-suite obstacle placements are generated at reset and are absent from these saved rows/cache; `requested_obstacle_count` is not a reconstruction of their actual positions. Matching uses fixed panel count when present, otherwise the requested count. Thus these controls match recorded geometry, not exact obstacle layouts.

Saved episode summaries include steps, mean/max speed, minimum target range, RMS cross-track error and COLREG/encounter totals. They are retrospective and depend on episode duration and termination. Do not use them, final outcomes or case IDs as online trigger features. Median summaries with finite sample counts are in summary.json; their differences are descriptive, not predictive validation.

There are no saved action sequences, intermediate poses, LiDAR rays, track histories or safety candidate rollouts in this dataset directory. Therefore this report cannot identify the first harmful intervention, replay an unfiltered policy's future behavior, or prove an override unnecessary.

## Future regression controls (not executed)

For each failure, up to two successful controls are retrieved within its exact cell and variant using Euclidean distance on variable, fully observed static coordinates standardized by the same cell's population SD. 113 failures have controls; 15 have none; 151 unique successes appear in 226 pairs. L2-HO has no same-cell successful case. Distance is a retrieval aid, not a propensity score or a causal match; outcomes choose the two groups but never enter distance calculations. Controls can recur across failures. Scales and coordinates are frozen in summary.json, and ties resolve by test ID. The local methodological precedent is [the test-set feature construction](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/tools/tiers/test_set.py:93>) and [standardized geometric distance](<C:/Users/hntran/OneDrive - University of Tasmania/Documents/PhD/asv-path-replanning/asv-lidar/static_dynamic_obstacles/tools/tiers/test_set.py:155>); this report adapts them for success-control retrieval with exact cell/variant eligibility, rather than claiming a published causal matching method.

`failures.csv` inventories all failures; `matched_controls.csv` lists every pair and unmatched reason; `successful_controls.csv` preserves the unique successful cases, seeds and scenario digests. This selected control list does not cover all 872 successes and must not replace all-set regression accounting.

## Observable hypotheses to test before changing intervention rules

1. Distinguish imminent collision under the currently issued policy action from failure to find an 8 s backup. Candidate signals are short-horizon hull clearance and predicted contact time; the saved episode aggregates cannot establish an appropriate threshold.
2. Check whether existing policy steering is already reducing risk: recent measured yaw/lateral motion, relative bearing/closing-rate trends, and local side clearance. A successful action may look unsafe when held open loop; a genuine near-term collision still needs intervention.
3. Separate target-threat evidence from perception uncertainty: track hits/misses, fresh motion evidence, fit validity and innovation. Preserve static panel and boundary checks when target evidence is weak; neither missing detections nor increasing centroid range proves safety.
4. Log whether recovery continuation wins solely through its preference bonus despite an admissible policy candidate. That event is observable inside the filter and directly tests unnecessary override logic without using case labels or future success.

These are untested engineering hypotheses, not implemented methods or causal findings. This report uses only counts, rates, medians and an explicitly defined descriptive distance; it makes no paper-derived safety guarantee or statistical-significance claim.
