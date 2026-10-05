# Safety v6: development-only model and geometry corrections

The objective is higher success without exchanging obstacle contacts for walls
or target ships. The selected v4 reference has 123 goals and 27 collisions in
the original 150 DV3 cases. Reaching at least 95% requires **143/150 goals**, or
20 net rescues with no losses; 100% requires all 27 remaining failures rescued.
These are targets, not a performance promise. A finite sampled predictor with
uncertain perception has no collision-free guarantee.

Use the same SAC baseline-3 checkpoint at 3M timesteps and original reset seeds.
No held-out, quick-campaign or simulated field-evaluation outcomes inform this
revision. Paper 2 is read-only, `constants.py` remains unchanged, and ongoing
PPO training must be left alone. The root coordinator owns episode evaluation
and its budget; the audit and implementation below ran no policy episodes.

## Current experimental candidate and completed broad comparison

The candidate evaluated in `broad_v6_50` combines **provisional safety tracks
with motion evidence** and **bounded trajectory search**. Its frozen overrides
are `PROVISIONAL_TRACKS=True`, `PROVISIONAL_MOTION_EVIDENCE=True` and
`TRAJECTORY_SEARCH=True`. Calibrated braking, fitted-hull perception and track
history are disabled. Inherited selected-v4 observer and free-space memory stay
enabled. This remains an opt-in experimental candidate after the broader
development check, not the existing runtime default or a 95–100% solution.

The six-case combined pilot `search_evidence_v2` achieved **4/6 goals**, versus
**2/6** in the union of fresh matching v4 pilot runs: CRP-CV-12 and CRP-VS-04
were rescued, with no lost goals among these six cases. CRP-CV-02 still hit the
target and CRS-CV-04 still hit the boundary. The reference is a union across
earlier tags, not six fresh v4 rows within the combined tag; the broad run will
provide a fresh comparison within one frozen tag. Selected pilot outcomes do
not establish performance on untested cases.

The completed broad comparison uses the fixed
[50-case selection](../../../results/safety_dev/development_v6_budget150/broad_selection50_v2.json):
all 27 historical selected-v4 failures and 23 original successes spanning all
11 encounter/model strata. Both v4 and v6 ran each case, for 100 new attempts.
The subset is deliberately enriched with failures; its goal fraction must not
be presented as full-DV3 or population success. Selection was frozen before
broad candidate outcomes. The completed results are:

| Mode | Goals | Obstacle contacts | Boundary contacts | Target contacts | Total contacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh v4 | 23/50 | 9 | 7 | 11 | 27 |
| Experimental v6 | 33/50 | 9 | 3 | 5 | 17 |

There are **13 gained goals and three lost goals**, for ten net additional
goals. The losses are HO-VS-01, CRP-CV-16 and CRS-VS-03. Total contacts fall by
ten; aggregate obstacle contacts stay at nine, while boundary/target contacts
fall by four/six. These are improvements on this fixed selected subset, with
material regressions. They do not establish full-set performance or justify
discarding failed cases. Native opt-in integration and the final source audit
are documented below; the variant remains experimental.
See the [campaign report](../../../results/safety_dev/development_v6_budget150/report.md)
and [frozen broad manifest](../../../results/safety_dev/development_v6_budget150/runs/broad_v6_50/manifest.json)
for status, exact settings and archived source provenance.

Budget accounting includes **five legacy verification simulations**. The
expanded pytest invocation reported 208 passed and one failed; its existing
v3 dead-ahead simulation test collided. That test does not construct v6, and no
legacy source/threshold was changed to make it pass. Four passed verification
simulations and this failed one consume attempts 39–43, separately from policy
results; no SAC outcome is inferred. The subsequent six-case history pilot
brings pre-broad usage to **49/150**. The completed 100-run broad comparison
brings usage to **149/150**, leaving one reserve. A prepared calibrated-braking
probe on HO-VS-01 was blocked before an episode or attempt-150 token: the source
guard detected an externally changed `src/scenario.py`. The final integration
audit below explains that drift; no probe outcome or extra consumed attempt
is inferred. The ledger, rather than
this schedule, is authoritative. The original selection file is preserved;
its v2 accounting revision changes no case IDs. See
[verification records](../../../results/safety_dev/development_v6_budget150/verification_runs.json).

Historical v4's 123/150 remains context. Its recorded `env.py` and
`safety_v4.py` bytes could not be recovered from inspected archives/Git, so
their differences from current sources cannot be proved merely cosmetic.
Observer, perception and v3 hashes do match. Use fresh paired current v4
results, as documented in the
[source audit](../../../results/safety_dev/development_v6_budget150/historical_v4_source_audit.json).

## Deterministic full-DV3 target bound

All 50 compared cases belong to the original 150-case DV3 set. This frozen v6
candidate achieved 33 goals and 17 contacts on those 50. Even granting success
on **all 100 untested cases**, it could achieve at most
**(33 + 100) / 150 = 133/150 = 88.67%** on the fixed full suite under the same
candidate, scenario and seed protocol. This is a deterministic finite-suite
upper bound, not an estimated overall success rate or a confidence interval;
selection bias does not invalidate this counting bound.

At least 95% requires 143 goals because goal counts are integers. Therefore at
least **43/50 selected goals** would be necessary even if all untested cases
succeeded. The measured 33/50 rules out 95% for this frozen candidate on that
full suite; it does not bound what a future changed policy/controller could
achieve. The true full-set result can be lower than 133 because the remaining
100 cases were not evaluated under this candidate.

All original cases remain in the denominator, including the initially
unobservable BO panels and cases that may be dynamically unrecoverable at
reset. Their diagnostic limitations do not authorize excluding them to improve
the claimed benchmark percentage. Further progress needs additional rescues
and regression prevention, or a separately declared revised benchmark with
its own justification. The source-hashed derivation is in
[finite_suite_bound.json](../../../results/safety_dev/development_v6_budget150/finite_suite_bound.json)
and [the report note](../../../results/safety_dev/development_v6_budget150/finite_suite_bound.md).

## Native opt-in integration and final provenance

The environment now selects `SafetyFilterV6` directly when the runtime setting
is `SAFETY_VERSION = 6`. The v6 defaults enable the three evaluated mechanisms:
`PROVISIONAL_TRACKS`, `PROVISIONAL_MOTION_EVIDENCE` and `TRAJECTORY_SEARCH`.
Calibrated braking, fitted-hull perception and track history remain disabled.
Existing default selection and version-4 behavior are unchanged; v6 remains an
explicit opt-in experimental variant, with the gains and regressions above.

The [integration audit](../../../results/safety_dev/development_v6_budget150/integration_audit.json)
compares the archived broad-run sources with the integrated files by AST. It
verifies that `env.py` changed only to add native v6 dispatch, and v6 changed
only its docstring and those three defaults. This establishes structural
correspondence to the evaluated runtime overrides; no new policy episode was
run after integration, and the audit is not another performance measurement.

The unexpected external `src/scenario.py` change occurred after the broad-run
process had imported the module for its first episode. The process retained
that loaded module and the runner does not reload it. The later probe's
preflight correctly rejected the changed disk source before reserving attempt
150 or starting any episode. The external edit was preserved, not restored or
bypassed, and is not approved as a replacement frozen baseline. The final
report marks that probe **aborted before execution**, with zero additional
attempts, outcomes or policy-run errors. Total budget use remains 149/150.

## Original DV3 failure taxonomy

Machine-readable case-level evidence, reset seeds and source hashes are in
[development_failure_taxonomy.json](../../../results/safety_dev/development_failure_taxonomy.json).
The selected source is `dev_v4_observer_memory.csv`, checked against the existing
full-set comparison. Counts below describe outcomes, not proven causes.

| Failure type | Cases | With changed actions | With no changed actions |
| --- | ---: | ---: | ---: |
| Target contact | 11 | 9 | 2 |
| Obstacle contact | 9 | 7 | 2 |
| Boundary contact | 7 | 7 | 0 |
| Total | 27 | 23 | 4 |

Seven failures replace policy-only goals: four target contacts, two boundaries
and one obstacle. Twenty were already policy failures. Ten of the eleven
target contacts are constant-speed cases; only CRS-VS-04 varies target speed.
Thus a future acceleration predictor alone cannot explain most target failures.
Four failures are no-target NT cases (three obstacles, one boundary), which
cannot be caused by exclusion around a dynamic target. Target geometry and
masking therefore need separate validation from motion and backup selection.

Detailed selected-v4 traces exist here for all seven boundary failures, plus
two successful cases. Six boundary contacts end in `no escape`; CRS-CV-11 ends
in `last certificate`. CRP-VS-02 and CRS-CV-19 enter `no escape` before their
first changed action (steps 14 before 18, and 15 before 17 respectively). The
other five do not show that ordering. This distinguishes early absence of a
sampled escape from failure after an intervention; it does not prove physical
inevitability. CRS-CV-11 executes 22 `last certificate` decisions spread over
its episode, despite each consecutive unverified period being capped. That
fallback remains a structural limit, not a valid safety certificate.

The four existing zero-override oracle comparisons are especially informative:
true ego plus true target state and exact static geometry rescue CRP-CV-02 and
CRP-CV-12, but BO-CV-10 and BO-VS-05 still contact obstacles without overrides.
The oracle bundles several changes, so it does not isolate a tracker fault.
At the start of this audit, no complete selected-v4 target/obstacle decision
traces were available; do not assign the masking or centre defect without those
measurements. Fourteen of the 27 failures were never goals under any of the
seven earlier reference/ablation columns in `final_fullset_comparison.csv`.

## Source-grounded perception defects

`classical/common.py:143` copies `Track.position` as the target centre while
using `last_fit_heading_deg` when available. The active tracker measurement is
`centroid`: `tracking.py:733` explicitly distinguishes a visible-return centroid
from a completed hull centre. `target_gap` then places a full LOA-by-breadth
rectangle at that centroid and advances it with constant velocity. An estimated
surface-return centroid is not a hull centre. The tracker already computes a
separate `last_fit_centre` on each matched reliable fit, but the safety snapshot
ignores it. Changing the global Kalman measurement is inappropriate: existing
tracking comments document velocity/association jumps when a fit appears or
disappears. A separate geometric view avoids changing that estimator, although
its predicted occupancy can still jump between fit and fallback.

`classical/common.py:156` excludes every current point within 1.8 m of a track;
line 163 also excludes remembered points within 2.0 m. Neither checks whether
the target hull explains the point. A deterministic geometry-only example
uses a north-facing target at `(0, 0)` and a separate static point at `(1.6, 0)`.
The radial rule removes the point. An own hull centred on that point has static
clearance -0.40 m while its inflated target-hull SAT gap is +0.80 m. Thus the
point can disappear without its occupancy being represented by the target.
This proves the masking defect, not its frequency in the 27 failures.

The existing free-space-memory change is retained. A finite ray passing beyond
a remembered point, without a supporting return nearby, can clear old target
ghosts. That mechanism does not repair radius-based suppression of a currently
observed independent panel.

## Optional implementation: fitted target geometry and footprint masking

`src/safety_hull_perception.py` adds
`HullSafetyPerception(memory_frames=..., fitted_track_centre=True,
hull_return_mask=True)`, a `SafetyPerception` subclass. The v6 wrapper can choose
it without changing common perception, tracking, v4, v5 or the policy.

The two constructor switches are independently ablatable:

- `fitted_track_centre` uses the existing fit centre and axis only for safety's
  `TrackView`. Track position/state, velocity and encounter contexts are left
  unchanged. A current fit needs a fresh pose, zero missed track updates and a
  finite two-coordinate centre and heading. A previously valid fit may coast
  using measured track velocity for at most `TRACK_MAX_MISSES` decisions, with
  its axis held. It then falls back to the original track view. Held pose frames
  cannot refresh the fit age. This is bounded reuse, not covariance propagation.
- `hull_return_mask` removes only returns inside a valid oriented fitted hull
  footprint padded by the existing `MOTION_EXPLAIN_M` tolerance. Physical known
  LOA/breadth are used; safety inflation is not added to this ownership mask.
  When no fit is reliable, unassigned returns stay as static evidence instead
  of using the old broad disc. That conservative fallback can create temporary
  target ghosts and must be checked for lost successes or stalls.

Fresh target-explained returns are excluded before memory ingestion. Masking
old points affects the current snapshot only; it never permanently erases an
occluded static point from memory. The inherited finite-ray clearing and normal
expiry remain the only permanent removal mechanisms. Stale pose frames neither
ingest fresh returns nor clear memory. The inherited routine receives a view
with no tracks to avoid its radial masking; this is local to the adapter and
does not mutate the environment. Centre-only mode preserves the old radial
mask's original anchors, and both switches off delegates exactly to v4's
`SafetyPerception`. `last_hull_stats` reports current/coasted fits, corrected
centres and excluded points.

The existing fitter cites [Zhang, Xu, Dong and Dolan (2017), *Efficient L-Shape
Fitting for Vehicle Detection Using Laser Scanners*](https://publications.ri.cmu.edu/efficient-l-shape-fitting-for-vehicle-detection-using-laser-scanners),
IEEE IV, pages 54-59, DOI
[10.1109/IVS.2017.7995698](https://doi.org/10.1109/IVS.2017.7995698).
The paper supports extracting oriented rectangles from laser returns. Known
vessel-size completion, reuse of the existing fit, the age limit and the return
mask are engineering choices here; the paper does not prove their safety.
For the broader distinction between kinematic state, spatial extent and
measurement association, see [Granström, Baum and Reuter, *Extended Object
Tracking: Introduction, Overview and Applications*](https://arxiv.org/abs/1604.00970).
This adapter does not implement their random-matrix or random-finite-set filters.

Validation: `python -B -m pytest -q tests/test_safety_hull_perception.py
tests/test_safety_perception.py` passed **34 tests** (23 new, 11 inherited) in
1.00 s. Tests cover removal of actual hull-face returns, retention of a nearby
independent panel that the legacy disc erased, orientation, geometry-only
views, bounded coasting/expiry, stale and invalid fits, independent switches,
no permanent hull-based memory deletion, inherited ray clearing and an
environment wrapper that rejects simulator-truth access. These tests preceded
the unsuccessful combined geometry/brake pilot recorded below.

## Development validation order

Keep centre-only, mask-only and their combination separate from brake-model
calibration. On selected development cases, log the first action divergence,
valid-fit availability, centre offset, points suppressed by the old disc but
retained by the footprint, and per-constraint margins before declaring a cause.
Record whether policy rejection is from target, static or boundary clearance;
`no escape` alone is insufficient. Compare unchanged-case successes as well as
failure rescues. Existing oracle results supply diagnostic motivation only;
production filtering must never receive target truth or exact hidden polygons.

Even if a failure-focused pilot improves, run the full original DV3 development
comparison before selecting a preset and preserve each gained/lost case. Check
123 reference successes as well as 27 failures. No threshold change, general
backup expansion or relaxation of the hard checks is bundled into this adapter.
Root schedules and counts all evaluations; no new full/held-out run is implied
by this plan.

The remaining model branch is offline brake-response identification from other
development measured-state/issued-command logs, with true ego used only to
score errors. The six isolated pulses in CRP-VS-04/CRS-CV-04 mean carrying delay
age across consecutive commands cannot explain those onsets. A better fit is
not automatically a better controller: the previous dual-brake envelope gave
118 goals versus selected v4's 123. Brake-calibration results and any integration
decision will be documented separately by the root coordinator.

## Root integration and initial development pilot

The new v6 wrapper keeps `CALIBRATED_BRAKING=False` and
`FITTED_HULL_PERCEPTION=False` by default. Its two geometry sub-switches map to
the adapter constructor, so centre and return ownership can be isolated. Root's
combined v6/perception/runner check passed 67 tests before the pilot. The latest
user-authorized development campaign has its own **150-attempt budget**; it is
separate from the completed earlier 100-cap campaign.

The independent offline brake calibration used 53 measured development pulses.
The effective acceleration at the observed -24 RPM command is
`a24 = 0.465206163762 m/s^2`. On the held-out *development cases* CRP-VS-04 and
CRS-CV-04, surge mean absolute error was 0.02433 m/s versus the old model's
0.20891 m/s. These are prediction-error figures, not goal/safety improvements,
and are not held-out benchmark results. See
`results/safety_dev/brake_calibration/case_validation/`. Related identification
method: [Ljung (2002), *Prediction Error Estimation Methods*](https://doi.org/10.1007/BF01211648).
Sparse isolated pulses do not independently identify all delay/strength effects
or justify extrapolation to unobserved reverse RPM values.

Root's first combined pilot uses eight original DV3 cases: CRP-CV-02,
CRP-CV-12, BO-CV-10, BO-VS-05, CRP-VS-04, CRS-CV-04, NT-CV-01 and HO-CV-01.
Both calibrated braking and fitted-hull perception are enabled by frozen
runtime overrides, not changes to defaults. The manifest and exact evaluated
source archive are under
`results/safety_dev/development_v6_budget150/runs/v6_calibrated_hulls_pilot/`.
This pilot combines two changes, so an outcome difference cannot identify
which change caused it. It completed with **1/8 goals**, versus **2/8** in the
historical selected-v4 records: NT-CV-01 remains a goal, while HO-CV-01 becomes
an obstacle contact. CRP-CV-02/12 still contact the target without any changed
actions; BO-CV-10/VS-05 still contact obstacles at decisions 1/2. CRP-VS-04
ends in obstacle contact and CRS-CV-04 in boundary contact. This is a failed
combined pilot, not evidence that either correction improves success. Eight
attempts were consumed, leaving 142 before the next comparison. Source and
manifest archives preserve that evaluated revision.

## Safety-only provisional track hypotheses

The first pilot exposed a publication gap independently of fitted hull centres.
`env._perceive()` publishes only `tracker.dynamic_tracks()` to ordinary
perception. Raw `tracker.tracks` also contains confirmed but non-dynamic tracks.
The dynamic label requires repeated finite-ray motion evidence and can be absent
when a moving target is partially occluded or its visible surface changes.
CRP-CV-02's last eight decisions contain no published tracks; CRP-CV-12 first
publishes one at decision 18, when the existing bank already reports no escape.
Those recorded absences are not proof that every raw cluster is the target.
Raw-track traces are therefore required for the next pilot; offline reconstructed
raycasts are supporting diagnostics, not an exact replay of tracker state.

`src/safety_provisional_tracks.py` adds
`ProvisionalTrackPerception(base_perception)`. It first obtains the unchanged
base snapshot, then appends safety-only `TrackView` hypotheses from onboard raw
tracks. It keeps the exact static-points array and does not mask any return for
the added tracks. Global KF state, velocity, `is_dynamic`, policy observations
and encounter contexts are untouched. Added hypotheses have no COLREG context.
The adapter can wrap either existing v4 perception or the optional hull adapter.

New admission requires a fresh pose, zero missed detections, at least the
existing `TRACK_MIN_HITS`, speed strictly above `TRACK_FIT_PRIOR_SPEED`, a finite
current fitted centre/axis, and at least `TRACK_FIT_MIN_POINTS` finite current
cluster points. Their extents in the fitted axes must fit physical LOA/breadth
plus `TRACK_FIT_FULL_EXTENT_TOL_M`. These existing thresholds are reused, not
tuned against pilot outcomes. A hypothesis may coast from its last accepted fit
for at most `TRACK_MAX_MISSES` decisions. A previously published dynamic ID is
retained after demotion while its raw track remains alive, using the last centre
offset with raw KF position rather than double-integrating its predicted motion.
Held poses never admit or refresh a hypothesis; bounded coasting still expires.
`last_provisional_stats` reports admitted/retained/coasted IDs and rejection counts.

This separates safety occupancy from encounter classification, inspired by the
kinematic/extent/association distinction in
[Granström, Baum and Reuter, *Extended Object Tracking*](https://arxiv.org/abs/1604.00970)
and the existing [Zhang et al. hull-fit reference](https://doi.org/10.1109/IVS.2017.7995698).
It does not implement a probabilistic multi-hypothesis tracker, nor inherit a
safety guarantee. A visible static edge can fit the same dimensions; centroid
motion can mimic vessel velocity. Retaining static evidence limits unsupported
erasure but may add false constraints, interventions or deadlock. Those losses
must be counted alongside any rescued targets.

Focused validation: `python -B -m pytest -q tests/test_safety_provisional_tracks.py`
passed **28 tests**. They cover early admission, oversized clusters in rotated
axes, invalid/missing fits, static evidence preservation, no shared-state
mutation, stale/missed coasting and expiry, retained demoted IDs, no double
prediction, no duplicate published tracks, and rejection of simulator-truth
access. No policy episode was run by this implementation task.

The next comparison completed eight attempts: provisional-only v6 and fresh
matched v4 on CRP-CV-02, CRP-CV-12, NT-CV-01 and HO-CV-01. Only the provisional
flag was enabled; calibrated braking, hull perception and other optional methods
remained off. Both modes achieved **2/4 goals**: v6 rescued CRP-CV-12, lost
HO-CV-01 to an obstacle, preserved NT-CV-01, and still hit the target in
CRP-CV-02. This is one gained goal and one lost goal, not a success improvement.
The records are in `results/safety_dev/development_v6_budget150/runs/provisional_targets_v1/`.
Completion brought campaign usage to 16 attempts; the campaign journal remains
authoritative as later pilots proceed. The shared 150-attempt cap includes all
development probes and both comparison modes.

The authoritative raw-track traces isolate the HO-CV-01 regression. Both modes
execute identical actions through decision 18. At decision 19, provisional ID42
represents the right static panel, whose visible-return centroid slides as own
ship passes it. Its estimated velocity is `(-0.068, +0.436)` m/s, and its fitted
centre `(6.671, 9.626)` extends past the actual panel. The entire prior motion
evidence history has zero appear/vacate violations. Adding this hypothesis
changes accepted policy motion into braking/hold-back, followed by obstacle
contact at decision 28. A second false hypothesis also has zero motion evidence.
Preserved static evidence does not prevent a false moving constraint from
causing a harmful intervention.

A narrow admission ablation is now implemented as constructor keyword
`require_motion_evidence=False`. Enabling it requires at least the existing
`MOTION_MIN_POINTS` violations for a new provisional ID, without the full
publication fraction and consecutive-frame requirement. Already admitted IDs
may refresh valid current fits without repeated motion evidence; published-ID
retention is independent of this gate. Stale or missed detections cannot admit
an ID using old evidence. Diagnostics record the exact admission frame and
appear/vacate/compared counts. The original 28 tests plus nine gate regressions
passed: **37 tests**. Default False preserves the first provisional pilot.

This gate rejects the two inspected HO false hypotheses; CRP-CV-12's first
changed action at decision 12 has five violations, and its decision 10 has
three. The four-case `provisional_evidence_v2` pilot then recorded **3/4 goals**:
CRP-CV-12 was rescued, NT-CV-01/HO-CV-01 remained goals, and CRP-CV-02 still
hit the target. These results improve that selected pilot, not the full set.
Primary related work is [Yoon, Tang and Barfoot (2018), *Mapless Online Detection
of Dynamic Objects in 3D Lidar*, Sec. III-C](https://arxiv.org/abs/1809.06972):
their free-space checks distinguish actual motion from newly visible surfaces
and viewpoint occlusion. Here the existing 2D tracker's finite-ray
evidence is reused with an added admission condition; their 3D detection
pipeline and continuous-time deskewing are not implemented, and its empirical results are not claimed.

CRP-CV-02 has a different remaining failure. Accurate provisional target fits
appear at decisions 3/4, but published tracking then uses the original centroid
view. Measured x-velocity decays from 0.283 m/s at decision 5 to -0.054 m/s at
decision 8 while the target continues at 0.351 m/s. That ID is subsequently
lost. The reacquired cluster has fitted headings 174/178 degrees at decisions
13/14 versus the target's 89.5 degrees; the extent guard correctly rejects those
unreliable fits. A valid fitted target returns at decision 17, when no sampled
escape remains. Simply loosening admission does not resolve velocity/axis
corruption and association loss during partial visibility. A future safety-only
motion-prior/uncertainty mechanism needs its own evidence and regression checks.

## Bounded cross-entropy trajectory search

`src/safety_trajectory_search.py` augments v4's finite primitive bank without
changing its hard clearance evaluator. The optimizer follows the sampling and
elite-refitting idea in [Zheng, Yang, Wu, Pan and Cheng (2022), *Safe Learning-based
Gradient-free Model Predictive Control Based on Cross-entropy Method*](https://arxiv.org/abs/2102.12124),
DOI [10.1016/j.engappai.2022.104731](https://doi.org/10.1016/j.engappai.2022.104731).
Only the optimizer idea is adapted: this implementation has no incremental
Gaussian-process model, CLF/CBF construction or inherited probabilistic guarantee.

Search runs when the original policy-bank margin is below the unchanged 0.15 m
trigger. First it fixes the policy action for the existing 1 s commit period
and searches its continuation. If that fails, an escape search is allowed only
when the best original plan or stored continuation also has margin below that
trigger. The policy stage is seeded from its original recovery plans and any
stored continuation; the escape stage takes the strongest 16 original plans
and any continuation. Modified warm seeds are checked again, never assumed safe.

Each search call evaluates its supplied seeds once, then **3 rounds of 64 new
samples = 192 sampled plans**, refitting from eight elites each round. The
8 s horizon is sampled in two-decision, 1 s control blocks. A decision invoking
both stages can therefore evaluate 384 new samples plus both seed sets; 192
is a per-search budget, not a per-decision maximum. A private NumPy generator
starts at seed 0 for each filter instance and does not consume the policy,
scenario or global random stream. Astern samples are available only when the
existing traffic condition permits them; the NaN marker retains signed-astern
command semantics and is not averaged into an invalid throttle.

A plan is accepted only if the complete unchanged evaluator finds it hard-safe
and its minimum margin meets the trigger. Among accepted plans, first-command
deviation from the policy is minimized, with clearance as a tie breaker. The
exact checked sequence is copied into stored recovery state; the first command
is issued and its remaining sequence is rechecked on following decisions. When
search fails, v4's original choice/fallback remains available. An accepted
fixed-policy search may issue the same immediate action while changing later
recovery behavior; search acceptance counts are not avoided-intervention counts.
Finite sampling and a finite horizon still do not establish recursive feasibility.

The fresh paired four-case `trajectory_search_v1` pilot gave v6 **3/4 goals**
versus v4 **2/4**: CRP-VS-04 was rescued, NT-CV-01 and HO-CV-01 stayed goals,
and CRS-CV-04 still hit the boundary. Adding calibrated braking in the separate
four-case `search_calibrated_v1` pilot again gave 3/4 with the same outcomes.
That supplied no outcome evidence for enabling calibration in the broad preset.
The combined search/evidence pilot and its reference qualification are recorded
at the top of this plan.

## Rejected optional track-history pilot

`src/safety_track_history.py` tested additional short-lived kinematic hypotheses
from previously admitted fitted tracks with appearance-backed motion evidence.
It appended prior measured views rather than replacing current views or removing
static returns, with at most the existing three-decision age window. This was an
engineering uncertainty approximation related to
[Wabersich and Zeilinger (2021), *A predictive safety filter for learning-based
control of constrained nonlinear dynamical systems*](https://arxiv.org/abs/1812.05506)
and [Granström, Baum and Reuter's extended-object tracking overview](https://arxiv.org/abs/1604.00970),
not their certified uncertainty set or probabilistic tracking algorithm.

The completed `search_evidence_history_v3` pilot gave **3/6 goals**, versus the
non-history combination's 4/6 on the same selected cases. CRP-CV-02 still hit
the target, and CRP-CV-12 regressed from goal to obstacle contact. The other four
case outcomes stayed the same. History is rejected for the broad comparison;
`TRACK_HISTORY=False` is frozen explicitly. No extra pilot is implied.

For inherited selected-v4 methods and their accurately limited attribution
(fixed-gain predict/correct observer, finite-ray scan-memory clearing), see
[the v5 related-prior-work section](SAFETY_LAYER_V5_PLAN.md#inherited-v4-methods-related-prior-work)
and [shared bibliography](../../SAFETY_LAYER_REFERENCES.bib). Those references were
added as related prior work, not claimed as the original implementation's cause.

## Initially invisible BO panels

Offline geometry inspection of the cached original BO-CV-10 and BO-VS-05 scenes
finds the third panel entirely absent from finite LiDAR returns at reset. The
sensor-to-near-face distances are approximately 0.381 and 0.665 m, inside the
existing 1 m deadzone; first intersections there become no-return beams rather
than observations of a farther surface. The inflated initial hull gaps are
approximately 0.218 and 0.461 m. Other panels are visible. The pilot's contacts
at decisions 1 and 2 are therefore not repaired by admitting raw tracks: no raw
cluster can recover this missing panel. Scan memory is empty at reset as well.

The scenario construction checks panel-centre spacing, and its feasibility
screen is holonomic; neither establishes initial sensor observability or
dynamic recoverability. The completed [offline actuator/geometry audit](../../../results/safety_dev/development_v6_budget150/offline/blind_zone_oracle_replay/REPORT.md)
reproduces recorded motion and finds all 10,426 sampled first commands contact
BO-CV-10 by 0.5 s. BO-VS-05 admits an immediate full-astern stop, but delaying
until the second decision contacts the panel. This is a finite numerical sample,
not a continuous-control proof; all cases remain in the benchmark denominator.

## Evaluation cost and final integration checks

The matched broad run used 283.46 summed episode seconds for V4 and 746.85
for V6, approximately 2.63 times as much evaluation wall time. The total was
1,030.31 seconds (17.2 minutes), excluding startup; both modes included the same
trace instrumentation. This is not a measured real-time control latency bound.

After native V6 integration, 147 focused controller/geometry/calibration/dispatch
tests passed. The updated source guard passed 28 synthetic tests, report accounting
17, and failure classification 11. The earlier unchanged V3 dead-ahead assertion
failure remains recorded separately; no benchmark outcome was relabelled or
legacy threshold changed to hide it. The runner accepts only the audited native
env dispatch hash transition; the unrelated scenario.py edit remains blocked.
