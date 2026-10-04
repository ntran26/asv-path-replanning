# Safety layer v4: prediction audit and onboard estimation corrections

Development continuation, 2026-10-02. Read `SAFETY_LAYER_NOTES.md` first.

**Selected v4 preset:** model-based ego observer + free-space memory clearing.
The other three experimental switches are disabled. This gives **123/150 goals,
9 obstacle, 7 boundary and 11 target contacts**, with no timeouts, on the matched
development set. It improves overall completion and collision count, but still
has more wall contacts than the policy alone or best v2 (two each). It is a
development candidate, not a validated deployment guarantee.

## Scope and baseline

The working project is Paper 3's LiDAR-based ASV path-following and collision
avoidance environment, with static panels, a tracked target, and a mapped basin
or channel boundary. Decisions run at 2 Hz. The policy observes the six-branch,
70-value `a25-v3-context` interface; the plant is the identified Bluefin model
with bootstrap parameter variation. Physical collision checks use the true
inflated hull and substeps, while a deployment filter must use onboard estimates.

This work wraps the unchanged baseline-3 SAC checkpoint at
`runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip`. It neither retrains
that policy nor changes its reward/observation formulation. The PPO training
process and its files are left alone. All development episodes come from
`formulation_v3.field_development_set()`, retaining reset seed `900120 + index`.
The frozen and Paper 2 evaluation sets are not used for tuning. `constants.py`
and `asv-lidar/static_obstacles/` remain untouched; no commits are made.

Historical references (150 matched development cases):

| Reference | Goal | Obstacle | Boundary | Target | Timeout |
|---|---:|---:|---:|---:|---:|
| Policy alone | 111 | 26 | 2 | 11 | 0 |
| Best v2, iteration 2 | 115 | 17 | 2 | 13 | 3 |
| v3 current | 108 | 18 | 14 | 10 | 0 |

## Diagnostics and what they measure

`tools/diagnostics/safety/prediction_audit.py` records the filter's actual
post-smoothing snapshot and pre-command actuator model, then replays the
**actual subsequent rudder and signed RPM commands** through the predictor.
Position, wrapped heading, and signed map-clearance errors are measured at
1, 2, 4, and 8 seconds. Comparing a hypothetical backup with the adaptive
policy's different actions would conflate control changes with model error.

Each origin is replayed twice: from the perceived initial ego state, and from
the true initial ego state using the same estimated actuator history. This
second replay still includes model, actuator, and reverse-thrust uncertainty.
Collision-truncated final decisions are excluded from fixed-time errors;
their final ten filter modes remain in the contact trace. Clearance minima
are sampled at decision boundaries. Overlapping windows are descriptive
samples, not independent trials for confidence intervals.

`oracle.py` is a diagnostics-only adapter: it supplies true current pose,
body velocities, target states and complete convex static polygons, bypassing
the ego EMA. The policy still receives ordinary observations. Predictor
parameters, target constant-velocity extrapolation, inflated hull dimensions,
safety gaps and terminal rules remain unchanged. Exact polygon clearance
includes long edges and containment, unlike sampling only obstacle vertices.

`integration_probe.py` compares the nominal predictor and nominal plant with
identical initial state and executed controls. In these probes, the 0.125 s
predictor differs from the 0.05 s plant by at most 0.001897 m and 0.02154 degrees
over eight seconds. This rules out integration resolution as the primary
mechanism in these probes, not every possible low-speed condition.

## Findings before changing thresholds

The two initial cases `DV3-OT-CV-16` and `DV3-BO-CV-04` reproduce their v3 wall
contacts. Full oracle v3 rescues OT16 but BO04 still hits a wall. V3 spends all
last ten OT16 decisions in `no escape`; BO04 has seven `no escape`, two nominal,
and one `last certificate`. Results: `results/safety_dev/wall_prediction_*`.

1. **Weak braking is not conservative for the whole trajectory.** In OT16,
   real full astern quickly removes surge while yaw persists. Yaw damping and
   rudder authority depend on forward flow, so the weak/delayed prediction can
   retain forward speed and predict a very different heading. On the sampled
   OT16 true-ego eight-second windows, no-brake mean errors are 0.083 m and
   1.03 degrees; windows containing braking average 1.79 m and 60.95 degrees.
   This motivated testing both a weak/delayed and a strong/immediate reverse
   prediction. The full-set ablation did not select that additional check.
2. **Smoothing can lag a crash stop.** In an illustrative reconstruction using
   true surge and the existing EMA weight, surge remains 0.331 m/s just after
   the plant reaches zero. This is not the actual recorded noisy filter state.
   In OT16's sampled origin 24, perceived replay has a large heading error even
   with no subsequent braking, while true-ego replay is much closer. Log the
   actual smoothed values before changing the observer.
3. **The backup library is restrictive.** V3 tests turns for exactly two
   seconds followed by a centered rudder. In the nominal probe this ends at
   21.93 degrees; adding two seconds of opposite rudder ends at -0.00384 degrees.
   Failing the original library does not prove all possible escapes fail.
4. **Nominal plans are discarded.** V3 checks a policy action plus a backup,
   then `_pass()` clears the plan. It only retains backups after intervention.
5. **Hull inflation is not an extra predictor error.** The simulator's contact
   geometry and predictor use the same `HULL_MARGIN`. The additional safety
   gaps are separate. Signed clearance errors go both ways, so the evidence
   does not support globally shrinking them.

Other source findings: stale-pose scans can contaminate long-lived static
memory, but default development pose-staleness probability is zero. Wall-only
engagement uses current clearance below 1.5 m; this is a candidate mechanism
to investigate from `idle` traces, not a threshold changed in this iteration.

## Initial candidate and retained ablation switches

`src/safety_v4.py`, selected explicitly with runtime `SAFETY_VERSION = 4`,
retains v3's perception, predictor, thresholds and selection rule, and adds:

- `COUNTERSTEER_RECOVERY`: two additional recovery templates, using the
  existing two-second turn duration for the turn and opposite turn.
- `RETAIN_NOMINAL_PLAN`: preserve the exact checked backup on nominal pass
  and handback. A continuation drops exactly the command already executed;
  expiration and the existing unchecked-plan limit remain enforced.

Both switches support paired ablations. With both disabled, selection matches
v3 in helper regressions. The initial two-case v4 test rescues BO04; OT16 changes
from boundary contact to target contact, which is **not** a successful rescue.
Full-set results, rather than these selected cases, decide whether to retain it.

`src/safety_prediction.py` supplies an experimental two-response brake
evaluator: the existing 0.25-efficiency, 0.75-second-delay response and the
simulator's nominal 0.5-efficiency, zero-delay response. Only sequences with
astern commands need a second rollout. First contact is the earlier of the
two and clearance the smaller; both must pass. These two responses do not
bound every unmeasured real reverse response and provide no certification.

## Reproduction

Run from `asv-lidar/static_dynamic_obstacles`, at most two evaluation processes:

```powershell
Remove-Item Env:V2_SET,Env:V3_SET,Env:V4_SET -ErrorAction SilentlyContinue
python -B tools/diagnostics/safety/prediction_audit.py --modes v3,oracle --cases DV3-OT-CV-16,DV3-BO-CV-04 --tag new_prediction_audit
python -B tools/diagnostics/safety/paired_eval.py v4 new_v4_run
python -B tools/diagnostics/safety/compare.py results/safety_dev/dev_new_v4_run.csv --mode v4
python -B -m pytest -q -p no:cacheprovider tests/test_safety_v4.py tests/test_safety_prediction.py tests/test_safety_perception.py tests/test_safety_observer.py tests/test_safety_v4_observer.py tests/test_safety_oracle.py tests/test_safety_prediction_audit.py
```

The new runner uses the selected defaults. The older `dev_eval.py` accepts only
off/v2/v3 and now rejects unsupported names before loading a model, preventing
an accidental version-1 run labelled v4. To select v4 in another runner, set
`cfg.SAFETY_VERSION = 4` at runtime and construct an environment with
`emergency_stop=True`; leave `constants.py` alone. Set any ablation switches
before the filter is constructed, and reset between configurations.

The selective development builder preserves authoritative generator seeds,
kwargs and full-set indices; cached scenarios avoid repeating expensive
space-time feasibility searches. Outputs and source/checkpoint provenance
are under `results/safety_dev/`. Comparisons retain every episode's outcome,
including gained/lost goals against policy alone, best v2 and current v3.

## Development results

The first full-set candidate (countersteering + nominal-plan retention) finishes
with 108 goals, 23 obstacle, 10 boundary and 9 target contacts. Against policy
alone it gains 15 goals but loses 18; against best v2 it gains 13 and loses 20;
against v3 it gains 8 and loses 8. **Rejected as the selected preset.** These
switches remain available to reproduce the experiment. Results and exact pairs:
`results/safety_dev/dev_v4_initial{,_paired,_comparison}.csv`.

The all-14-wall audit reproduces every original v3 wall contact. **All 14 final
decisions are `no escape`**, not idle, recovery or last-certificate commands.
The preceding ten-step windows contain 104 no-escape decisions, 27 last-certificate
decisions, 7 other recovery decisions and 2 nominal decisions (140 in total;
successive decisions within a case are not independent trials).

| Variant on the original 14 v3 wall cases | Goal | Boundary | Obstacle |
|---|---:|---:|---:|
| v3 | 0 | 14 | 0 |
| True ego + target state, remembered scans retained | 8 | 4 | 2 |
| True state + exact static geometry | 11 | 2 | 1 |
| V3 + free-space memory clearing only | 5 | 9 | 0 |
| V3 + two-response braking check only | 4 | 10 | 0 |
| V3 + model-based ego observer only | 9 | 3 | 2 |

These are selected failure cases, not overall success rates. Full versus
state-only oracle isolates static perception broadly (including ghosts,
pose-lifting error and incomplete surface coverage), not memory alone.

At eight seconds on the v3 wall-case trajectories, without future braking,
position-error median/p95 falls from 0.259/0.710 m with perceived initialization
to 0.042/0.182 m with true ego initialization. With braking, even true-ego replay
has 0.961/2.212 m median/p95 position error. This supports treating observer lag
and reverse prediction as distinct problems. These are descriptive quantiles
on failure-selected, overlapping windows (`all_wall_prediction_errors.csv`).

## Additional evidence-led mechanisms

`FREE_SPACE_MEMORY` installs `SafetyPerception` for v4 only. The original
90-second memory is retained; old points are permanently removed only when
fresh raw finite-return rays pass through them and no nearby return supports
them. It reuses the tracker's existing pass/explain tolerances. Max-range,
occluded, aft-masked and dead-zone observations do not certify clearing. Stale
poses neither clear memory nor insert fresh returns in an old coordinate frame.
The shared tracker and classical perception are unchanged.

`MODEL_EGO_OBSERVER` replaces a stationary EMA prior with a one-decision
identified-model prior, corrected by the existing 0.3 measurement weight.
`env.step()` supplies the issued rudder and **signed RPM after** bridge limiting
and brake-to-zero conversion, before physics. Only known commands and onboard
estimates enter the observer. The immediate nominal reverse law is a prior,
not a change to the safety predictor's two-response check. Held measurements
are not assimilated repeatedly. Oracle diagnostics bypass this prior too.

Offline replay on the 14 state-only-oracle traces (938 decisions), without new
policy episodes or gain tuning:

| Estimation error | Stationary EMA | Model-based prior |
|---|---:|---:|
| Mean absolute surge error | 0.0537 m/s | 0.0169 m/s |
| Mean absolute sway error | 0.0281 m/s | 0.0152 m/s |
| Mean absolute yaw error | 2.665 deg/s | 0.849 deg/s |
| 95th-percentile yaw error | 8.360 deg/s | 2.427 deg/s |

Raw measured yaw has slightly lower MAE (0.803 deg/s) on these traces, so the
observer is not claimed to dominate raw measurements in every component.
Reproduction: `tools/diagnostics/safety/observer_replay.py`; retained outputs
`results/safety_dev/observer_replay*.csv`.

The v4 intervention flag also counts astern when its transport vector equals
the policy's floor-throttle vector; these commands have different signed RPM.
This corrects counting, not action selection.

## Full-set ablations

Countersteering and nominal-plan retention are disabled in all rows below
except the initial candidate. No safety gaps, engagement thresholds, horizons,
observer gains or held-out scenarios were tuned.

| Candidate | Goal | Obstacle | Boundary | Target | Timeout |
|---|---:|---:|---:|---:|---:|
| Initial countersteer + retention | 108 | 23 | 10 | 9 | 0 |
| Free-space memory + two-response braking | 118 | 13 | 10 | 9 | 0 |
| Observer + memory + two-response braking | 118 | 11 | 10 | 11 | 0 |
| Observer only | 110 | 16 | 12 | 12 | 0 |
| **Observer + memory (selected)** | **123** | **9** | **7** | **11** | **0** |

Memory + braking gains/loses 18/11 goals against policy alone, 15/12 against
best v2 and 15/5 against current v3. Adding the observer gives 16/9, 13/10 and
20/10 respectively. Both candidates have 32 collisions, equal to best v2's
collision count; their three extra goals replace its three timeouts.

Equal totals hide different failure cases. Adding the observer exchanges nine
gained goals for nine lost and changes 26 outcomes. Memory + braking retains
six original v3 wall contacts and introduces four; observer + memory + braking
retains two and introduces eight. Only CRS-CV-04 and CRS-CV-19 are wall contacts
in both. See `combined_v4_case_comparison.csv`,
`combined_v4_boundary_regressions.csv` and `combined_v4_changes_from_observer.csv`.
These comparisons motivated the final observer-only and observer + memory
ablations. No combined step traces were available when deriving these
outcome-level interaction findings.

The selected observer + memory preset gains/loses **19/7** goals against policy
alone, **16/8** against best v2 and **24/9** against v3. It has 27 collisions,
compared with 39, 32 and 42 respectively. Observer alone's 110 goals shows that
improved state estimates are not sufficient by themselves. Adding the second
braking response to observer + memory reduces goals from 123 to 118 and raises
walls from seven to ten, so it stays disabled despite useful selected-case
rescues. No margin or threshold adjustment was needed for the selected preset.
Adding it changes exactly five outcomes, all from goal to collision: OT-CV-11,
OT-CV-15 and BO-CV-19 become boundary contacts; HO-CV-03 and CRS-CV-17 become
obstacle contacts. It rescues no cases in this paired comparison.

Selected settings in `src/safety_v4.py`:

```python
COUNTERSTEER_RECOVERY = False
RETAIN_NOMINAL_PLAN = False
DUAL_BRAKE_PREDICTION = False
FREE_SPACE_MEMORY = True
MODEL_EGO_OBSERVER = True
```

Authoritative selected results are
`results/safety_dev/dev_v4_observer_memory.csv`; its adjacent JSON records all
effective switches, source/checkpoint hashes and every case's seed and digest.
The `_paired.csv` and `_comparison.csv` files preserve all gained/lost episodes.
All four estimation/braking runs share identical ordered cases, seeds,
scenario digests and production-source hashes. The initial countersteering
run's older metadata lacks per-case digests; its cache was independently
verified byte-identical to the imported cache used by subsequent runs.

No global safety default or held-out evaluation configuration has been changed;
the two held-out runner CLIs merely accept explicit `--safety-version 4` now.

## Final contact audit and validation

With the selected source defaults and no runtime ablation overrides, a new
nine-case audit reproduces all seven remaining wall contacts plus the OT-CV-16
and BO-CV-04 successes exactly. Results: `v4_selected_contacts_*` and
`selected_contact_review.json` under `results/safety_dev/`. Six final contacts
occur in `no escape`; CRS-CV-11 occurs while following the `last certificate`.
The seven last-ten-step windows contain 40 no-escape, 17 other recovery and
13 last-certificate decisions. None are idle or nominal passes.

Of v3's original 14 wall cases, 12 now succeed, NT-CV-12 becomes an obstacle
contact and CRS-CV-04 remains a wall contact. Six other cases become boundaries.
Three of those replace v3 goals (CRP-VS-04, CRS-CV-11 and CRS-CV-19); the others
replace its obstacle/target contacts. Relative to policy alone, all seven walls
are new boundary outcomes, but only CRP-VS-04 and CRS-CV-04 replace goals. The
other five replace obstacle contacts. The full outcome matrix is
`final_fullset_comparison.csv`; aggregate counts are in `final_fullset_summary.csv`.

On these nine selected cases (552 decisions), filter-call median/p95/max is
0.077/0.122/0.299 s. Full environment-step p95/p99/max is 0.445/0.523/0.581 s.
These are wall-clock measurements on this host with PPO active and regression
checks partly concurrent. The filter-call timer excludes the issued-command
observer callback; this is not a real-time bound or a hardware benchmark.

Final focused validation: **65 new tests and four existing regressions pass**.
One existing v3 dead-ahead test fails at its no-collision assertion. The failure
was reproduced identically using the original HEAD `env.py` executed in memory
(SHA-256 `0bec1cf93f76360262a71d3dadcf20a186381faa34b5e10cb7fec9d6b558e9e3`),
without changing project files. That test is left intact and was deselected
from the final 69-pass run. The two existing safety suites use their established
scenario fixtures; no held-out suite was used for candidate selection or tuning.
New coverage includes exact v2/v3/v4 dispatch, issued signed RPM and command
limiting, model priors, stale frames, supported/occluded LiDAR memory, brake
response combinations, oracle geometry, scenario-cache provenance and replay
timing. A new replay test explicitly sets its positive RPM floor so it does not
depend on which curriculum a preceding test selected.

## Remaining work

Keep the 123-goal preset as the development reference. It has seven wall
contacts and eleven target contacts, so the safety problem is not solved.
Inspect the first action divergence in the newly introduced wall cases,
especially CRS-CV-11's unchecked continuation, before changing plan fallback
or thresholds. Use the original matched seeds and the per-case matrix to count
both rescues and regressions. Reverse-thrust identification and target-motion
uncertainty remain model limitations. Multiple candidates were selected on this
same development set; the improvement is not a held-out generalization result.
