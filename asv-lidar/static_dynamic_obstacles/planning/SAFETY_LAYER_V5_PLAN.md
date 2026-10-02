# Safety v5: cited methods and expanded evaluation

Continuation of `SAFETY_LAYER_V4_PLAN.md`, 2026-10-02. The user requested further
safety improvements, a citation for every implemented method, and evaluation
over all development, frozen and field sets. Frozen/field evaluation is now
explicitly authorized. Candidate selection still uses development results;
the selected implementation will be frozen before inspecting held-out results.

**Current execution status:** the full sweep was cancelled. The quick campaign
is complete: four development probes plus 72 comparison episodes = **76 actual
new runs**. On 24 matched scenarios, v4 and experimental v5 both achieved
19 goals and five collisions; v5 has no goal gains/losses against v4. This
result does not support promoting v5. See
[Completed quick comparison](#completed-quick-comparison) for outcomes,
activity counts, timing and provenance. Keep the evaluated code unchanged;
older schedules and presets below are historical and must not be resumed.

## Preserved reference and constraints

The selected v4 remains unchanged: model-based ego observer and free-space
memory clearing, with its other ablations off. It achieved 123/150 goals,
9 obstacle, 7 boundary and 11 target contacts on the DV3 development cases.
The same SAC baseline-3 3M checkpoint is used throughout. No retraining,
`constants.py` edits, writes to `asv-lidar/static_obstacles/`, or commits.
PPO training continues; evaluations use at most two single-threaded workers.

## Inherited v4 methods: related prior work

The following references were identified retrospectively to document related
architectures. They do not establish that the earlier v4 implementation was
motivated by these papers, and their guarantees are not claimed here.

**Ego observer.** [Luenberger (1971), Section II.B, equation (2.6)](https://doi.org/10.1109/TAC.1971.1099826)
describes a linear continuous-time observer combining model dynamics with
measurement-error correction. The architectural connection in
`src/safety_observer.py` is prediction from issued commands followed by a fresh
measurement correction. Our implementation advances the identified nonlinear
hull/actuator model and corrects `[u, v, r]` as
`estimate = prior + EGO_SMOOTHING * (measurement - prior)`. It reuses a fixed
scalar gain and skips correction for a stale frame. It has no covariance
propagation, Kalman gain calculation, pole-placement design or established
observer-convergence guarantee. The reference documents the model-plus-error
architecture, not this particular nonlinear update or its gain choice.

**Scan-memory clearing.** [Hornung et al. (2013), Sections 4.4.1 and 5.1](https://www.arminhornung.de/Research/pub/hornung13auro.pdf)
describe free-space evidence along a measured range ray and preservation of
occupied endpoints within a scan update. In `src/safety_perception.py`, a fresh
finite LiDAR return beyond a remembered point provides clearing evidence only
when no nearby current return supports that point. Contradicted points are
removed from stored batches before new points are ingested. No-return rays,
occlusions, aft/dead-zone regions and stale pose frames provide no clearing
evidence; normal memory expiry still applies. This is direct 2D point-memory
maintenance with existing geometric tolerances. It implements neither
OctoMap's occupancy grid/octree nor its probabilistic log-odds update, and adds
no OctoMap dependency. The relationship is the ray-based free-space evidence
and priority given to current obstacle returns.

## Implemented methods and their sources

### 1. One-decision commitment

`ONE_DECISION_COMMIT` holds the proposed action for the actual 0.5-second
decision interval, then predicts a backup. V4 instead requires it to be held
for one second, although the filter runs again after half a second.
Inspiration: [Bastani's model-predictive shielding](https://arxiv.org/abs/1905.10691),
Section III, Algorithm 1, which checks recoverability after one learned-policy
action. Our sampled backups and finite terminal run-out do not supply that
paper's invariant backup set. This changes commitment, not the eight-second
prediction horizon or collision margins. Implemented in `safety_v5.plan_for`.

### 2. Constraint-slack recovery

`SOFT_RECOVERY` changes only the branch where every hard-constrained plan
fails. It ranks complete predicted sequences, including the stored
continuation, by integrated nonnegative clearance deficits, then policy-action
distance. An unverified plan gets no automatic preference and is not retained
as a certificate. Inspiration: [Wabersich and Zeilinger's predictive barrier
recovery](https://arxiv.org/abs/2105.10241), Section III, equation (9). This is a
finite-bank adaptation without their terminal barrier or stability proof.
Implemented in `src/safety_recovery.py` and the v5 infeasibility branch.

The deficit uses each static, boundary and tracked-target constraint in metres,
integrated over prediction time. The existing terminal deficit is weighted by
one control interval. A signed outside-map extension prevents a trajectory
from reducing its violation cost merely by passing entirely through a wall;
the historical hard acceptance checks remain unchanged. This geometric
correction supports the recovery cost and is not a new safety certificate.

### 3. Feedback course-holding backups

`FEEDBACK_BACKUPS` adds four course targets when the original bank does not
provide a roomy policy continuation, or during recovery. Rudder is recomputed
from predicted heading and yaw at every decision. Inspiration:
[Eriksen et al.'s maritime BC-MPC](https://doi.org/10.1002/rob.21900), Section 3.1,
and [Chen et al.'s feedback backup rollouts](https://arxiv.org/abs/2104.11332),
Section III. The four samples (nearest-edge parallel, current course and
current course +/-30 degrees) are engineering choices; they are not copied
parameters from those papers. `FEEDBACK_ONLY_INFEASIBLE` is a separate
conservative integration ablation: enable this additional bank only when no
original sequence or stored continuation is hard-safe. This preserves all
feasible v4 choices and changes only the infeasible branch; the gate itself is
an engineering choice, not a guarantee from either paper. Existing course-controller gains, cruise
propulsion, dynamics and clearance checks are reused. There is no LOS pull
back onto a path that may cross obstacles. Implemented in
`src/safety_feedback.py`; generated control sequences replay identically
through the existing model. The papers' invariance claims do not transfer.

## Development protocol

Each method has its own switch and regression tests. First compare isolated
methods on the same 150 DV3 cases and reset seeds `900120 + case index`. Preserve
every gain and loss versus v4 and policy alone. Investigate combinations only
when isolated results support them. No future true state, static geometry or
target trajectory enters production filtering.

The first one-decision probe uses the existing v4 runner with a runtime-only
`V2_SET=COMMIT_S=0.5`; v5's local implementation leaves the v2/v3/v4 source
defaults unchanged. Subsequent trials use `V5_SET` and record all effective
module constants and source/checkpoint/scenario hashes.

## Expanded evaluation inventory

The requested scope includes distinct sets rather than relabeling the 150
episodes as the full development corpus:

| Group | Definition | Cases per mode |
|---|---|---:|
| Development | DV3 field development | 150 |
| Development | Legacy formulation development | 120 |
| Development | Tier-1 head-on width sweep | 100 |
| Frozen | Tier B headline | 800 |
| Frozen | Tier R robustness | 900 |
| Frozen | Additional Tier A, realized cases | 35 |
| Field | Deployment/Paper-2-derived suite, revision 2.0 | 630 |
| Field | Distinct FV validation generator | 155 |
| **Total** | **Per controller mode** | **2,890** |

Exact realized manifests, not nominal definitions, determine final counts.
The Paper-2-derived suite is run from this project's tools; the original Paper
2 folder stays read-only. Historical outcomes may only be reused if checkpoint,
configuration, scenarios, reset seeds and relevant implementation match.
Otherwise rerun the policy-only reference. New outputs use unique tags under
`results/safety_dev/`, with per-case flushing and guarded resume.

## Results and selection

The one-decision-only probe finished with **111 goals, 18 obstacle, 9 boundary
and 12 target contacts** (`dev_v5_one_decision_probe.csv`). It is rejected as a
standalone preset: reducing the conservative commitment did not improve the
closed-loop result. The soft-recovery ablation finished with **121 goals, 12 obstacle, 7 boundary,
9 target contacts and 1 timeout** (`dev_v5_soft_only.csv`). Relative to v4 it
rescues three cases (CRS-CV-02, CRS-CV-08, CRS-CV-11), but loses five goals
(HO-CV-03, OT-CV-11, OT-CV-18, OT-VS-03, BO-CV-19), so it is not selected.
The broad feedback-only ablation finished at **121 goals, 13 obstacle,
6 boundary and 10 target contacts** (`dev_v5_feedback_only.csv`). It is not
selected. The stricter infeasibility-only feedback ablation finished with **123 goals,
10 obstacle, 6 boundary and 11 target contacts**
(`dev_v5_feedback_infeasible.csv`), at 5.89 seconds/episode. It has zero goals
gained or lost versus v4; only CRS-CV-04 changes from boundary to obstacle
contact. This is not a net collision reduction. It is the frozen **experimental
v5 benchmark preset**; v4 remains the recommended development reference.

An additional four-case oracle diagnostic targets collisions with zero counted
v4 overrides: CRP-CV-02, CRP-CV-12, BO-CV-10 and BO-VS-05. V4 has two target and
two obstacle contacts; true state plus exact geometry rescues both target cases
but still hits the two obstacles (`dev_v5_zero_override_oracle.csv`). Zero
overrides alone does not distinguish nominal acceptance from `no escape`, so
this result does not establish a particular tracker or geometry fault.

Before any held-out policy episode, freeze v5 as:
`ONE_DECISION_COMMIT=False`, `SOFT_RECOVERY=False`,
`FEEDBACK_BACKUPS=True`, `FEEDBACK_ONLY_INFEASIBLE=True`.
No held-out outcome has been used to alter a method. No claim that v5 improves
on v4 is supported by the 150-case result. The expanded runner is
`tools/diagnostics/safety/suite_eval.py`; it writes to
`results/safety_dev/suites/<tag>`, flushes every episode, and accepts guarded
`--resume`. Independent review checked original seeds/builders, checkpoint
setup, scenario-cache invalidation, durable journal recovery and serial
execution. Historical result manifests lack enough checkpoint/configuration
provenance for exact policy-only result reuse, so fresh references are planned.

Machine-readable bibliography: `SAFETY_LAYER_REFERENCES.bib`.

## Soft-recovery diagnosis

`v5_soft_regression_tradeoffs_*` reproduces all five lost cases. Among 159
soft decisions, only one selected a nonzero next-decision deficit while a
zero-deficit alternative existed, in OT-VS-03 (the timeout). The four new
collision cases had no such decision. Therefore next-decision prioritization
is not implemented as a proposed remedy for those collisions. The diagnostic
records every candidate score and delegates the original choice unchanged.

## Validation before expanded evaluation

`python -B -m pytest -q tests -k safety
--deselect=tests/test_safety_v3.py::test_filter_steers_off_an_obstacle_dead_ahead_and_leaves_open_water_alone`
passed 139 tests (552 nonselected tests). The excluded v3 failure was already
reproduced with the original HEAD environment in the v4 work; it is not a new
v5 failure or silently removed. Existing edited files retain LF line endings;
`git diff --check` passes for the tracked edits.

After adding the queue, shared-read status, full-scope reporting and plotting
checks, the same safety selection passed **182 tests**, with 552 deselected,
in 268.70 seconds. The only warning was the installed TensorBoard package's
NumPy `bool8` deprecation. Reporting now requires explicit
`--require-full-benchmark` for the final report: all eight components and all
three modes must supply exactly 8,670 valid committed records. Field labels
state that these are simulations. Collision-to-goal and collision-to-timeout
transitions are reported separately, and pooled rates are episode-weighted.

Expanded comparison modes are `off,v4,v5` (8,670 planned episodes). The optional
comparison-scope question received no answer before selection, so the stated
three-controller recommendation is used. Run at most two serial, one-thread
components at a time. Code and effective settings are frozen before inspecting
held-out outcomes; a benchmark failure is recorded and does not trigger tuning.

## Expanded benchmark execution (cancelled by user)

The user stopped this long sweep and requested a quick diagnostic instead,
with **100 new episode runs total** as an explicit cap. Original execution
notes below are historical, not instructions to resume. An interim immutable
snapshot at UTC 2026-10-02 11:50:51 records 3,141 completed rows. The matched
v4/off comparison includes 1,441 cases: 1,327 versus 1,331 goals, 16 versus 26
obstacle contacts, 26 versus 13 boundary contacts, and 72 versus 71 target
contacts. V4 has 41 goal gains and 45 losses on that partial slice. It does
not establish an aggregate improvement. See `v4_cancelled_full_sweep_*`.

Code/defaults were frozen before the first held-out policy episode. The first
component is `results/safety_dev/suites/v5_frozen_r`, 900 scenarios x three
modes; all 900 digests match the historical frozen manifest. The exact
checkpoint SHA256 is
`993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`.
`results/safety_dev/v5_benchmark_freeze.json` records the already-frozen
selection and points to the authoritative pre-episode manifest.

Planned component scheduling: robustness R then development/validation/A in
one lane; headline B then deployment field in the other. At most two serial
evaluation processes run concurrently; initial scenario preflight occupies one
lane until its cache is complete. All outcomes at this stage are partial.

Preflight completed all 2,890 cases: 1,700/1,700 historical frozen B/R digests
and 630/630 deployment-field digests match. Named Tier A realizes 35 of 38
definitions; `A-BO-N`, `A-NU-I`, `A-NU-N` are the declared generation shortfall.
The two active lanes are frozen R (`v5_frozen_r`) and frozen B
(`v5_frozen_b`), with B then deployment (`v5_field_deployment`) queued serially.
R is followed by `--suites dev,field_validation,frozen_a --tag v5_dev_fv_a`.

Infrastructure incident: the headline-B process stopped when Windows denied
opening `progress.json` for an update. Its JSONL journal already held 230 unique
completed cases while progress still showed 229. The identical command with
`--resume` validated the unchanged manifest, removed only its proven-stale
lock, reconstructed the CSV, and continued at case 231. No completed episode
was lost, duplicated, or excluded; no controller or runner source was changed.
Monitoring now treats complete JSONL records as authoritative and uses shared
read access to avoid blocking a writer.

Further integrity checks identified one missing interior journal record,
`off/frozen_b/B-06-040`, while its CSV goal result remains. The CSV evidence is
preserved in that run's `reconciliation_missing_off_B-06-040.json`. The reason
for the gap is unproven; the evaluator has no intentional interior-deletion
path. Reconciliation must replay every missing seeded case via unchanged
`--resume`, then validate all identities before reporting full results.

The new, separate `run_frozen_queue.py` handles only process launching and
bookkeeping recovery. It leaves all frozen source/settings untouched, logs
exact argv and source hashes, and verifies journal completeness after exit 0.
Policy exceptions, rejected manifests and invalid journals stop execution.
The already-loaded lane A wrapper predates this guard; its provenance sidecar
records that fact, so all final journals will be independently reconciled.

## Quick revision: sideslip-compensated course rescue

Existing DV3 wall traces show lateral velocity even when heading is nearly
parallel to the boundary. Add one predicted course backup using
`beta = atan2(sway, surge)` and `desired_heading = edge_course - beta`.
The kinematic inspiration is Fossen, Pettersen and Galeazzi (2015),
[Line-of-Sight Path Following for Dubins Paths with Adaptive Sideslip
Compensation of Drift Forces](https://doi.org/10.1109/TCST.2014.2338354),
Section II.B, Eq. (14) and the course/heading relationship. This implementation
does not use the paper's adaptive LOS law or inherit its stability proof;
large low-speed sideslip lies outside its small/constant-sideslip assumptions.

`src/safety_course_backup.py` reuses the existing feedback physics, controller
gains, cruise command, delays, commitment and horizon. It reads predicted
body velocities only. At exactly zero velocity, beta is zero. No threshold
tuning is introduced. `SIDESLIP_RESCUE` runs only when all old candidates,
stored continuation and existing feedback candidates are hard-infeasible.
Only a new bank containing a complete hard-safe plan is admitted. If it fails,
discard it entirely so it cannot alter the previous fallback ranking.

This change was motivated from existing development traces, before testing
the new fixed sample. The quick protocol is 24 canonical frozen/field cases
x v4/revised-v5 = 48 new runs, with a campaign-wide hard cap of 100 attempted
episodes including retries. Sample families/features and omissions are recorded
before evaluation. This is a coverage-oriented diagnostic, not a fresh held-out
population estimate or evidence that 24 cases reproduce all 8,670 outcomes.
Preserve the complete planned inventory, observed rows, hashes and missing
statuses. The original 77-file frozen baseline is archived separately.

Targeted validation after the change: **56 tests passed in 12.03 seconds**
(41 v5 selection, 8 existing feedback, 7 new sideslip/replay tests).
New regressions check
opposite sway signs, zero velocity, exact model replay, suppression by an
existing safe plan, complete-plan admission and unchanged rejected fallback,
including both feedback-bank appends with a stored continuation.

## Verified-policy priority during recovery

The original development traces contain at least nine occasions in two lost
cases where the policy action was hard-feasible and remained in the existing
clearance-qualified candidate pool, but a different stored-plan command won
because `W_CONTINUE=0.25` made its action-deviation score negative. In
CRS-CV-04 step 13, policy clearance is 0.85 m, policy distance is zero and the
continuation's biased distance is -0.0376. Low speed blocks full handback.
Evidence: `results/safety_dev/development_safe_policy_override_audit.json`.
This establishes an unnecessary deviation under the filter's own test; it
does not by itself establish that removing it prevents the later collision.

`PREFER_CERTIFIED_POLICY` lets the unchanged policy action win before that
bonus when it survives the SAME hard checks and clearance floor. It applies
only in recovery with the policy as reference. Recovery mode, handback gates,
thresholds and the checked backup plan are retained. Policy-ineligible states
keep their previous selection behavior. The inspiration is the first-action
minimum-deviation objective in Wabersich and Zeilinger,
[A predictive safety filter for learning-based control of constrained nonlinear
dynamical systems](https://arxiv.org/abs/1812.05506), Section 4.1, Eq. (5a).
Our finite, approximate predictor inherits no formal guarantee from the paper.

The first revised proposal added fresh policy-only references: the SAME 24
preselected canonical cases x off/v4/v5 = 72 runs, preceded by a two-episode
priority-only development probe. That initial 74-run proposal was superseded
by the additional two development probes and the candidate below: **four
development attempts plus 72 comparison episodes = 76 planned new runs**.
All attempts and retries share the existing 100-run cap. The old full queues
have exited with 3,347 committed records; no full-sweep resumes are authorized.

The priority-only development preset enabled `PREFER_CERTIFIED_POLICY` while
one-decision commitment, soft recovery, feedback-bank expansion and sideslip
rescue were disabled. The observer and free-space memory remained those of
selected v4. Its two probes still ended in boundary contacts; priority alone
has not demonstrated a rescue. The existing 56 v5/feedback/course regressions
passed again in 3.04 seconds after those defaults were selected. The final
quick candidate also enables the policy-only feedback check described next,
so its results cannot be attributed to priority alone.

## Policy-only feedback preservation

Existing DV3 traces show that the original short bank can reject the policy's
proposed action while admitting a different action. Failure of those sampled
backups does not establish that the policy's action has no recoverable
continuation. This motivates a narrow additional check before the initial
override, without expanding the competing action pool or shortening the
existing one-second commitment.

`POLICY_FEEDBACK_PRESERVATION` applies only in nominal mode, when the original
policy bank is below `TRIGGER_MARGIN_M` and at least one original sequence or
stored continuation is hard-safe. It reuses `safety_feedback.feedback_bank`
to evaluate only the exact policy action followed by the four existing
course-holding feedback backups. Acceptance requires one complete trajectory
to pass all original hard checks and the unchanged 0.15 m trigger margin.
On acceptance, the exact policy action passes and nominal mode is retained;
with selected v4's retention switch off, no stored plan remains. If the extra
bank is unsafe or below margin, it is discarded and the original alternative,
recovery and fallback selection is unchanged. Already-accepted nominal
commands, recovery decisions and wholly infeasible original banks skip this
extra check. No future policy commands, true state or hidden geometry are used.

The recoverability-before-override idea is inspired by
[Bastani, Section III, Algorithm 1](https://arxiv.org/abs/1905.10691).
Forward integration under a feedback backup is related to
[Chen et al., Section III](https://arxiv.org/abs/2104.11332); the four course
samples and physics are the existing BC-MPC-inspired helper described above.
These references motivate the architecture, not this gate or its sample
choices. This implementation has a finite horizon and approximate prediction,
without an invariant terminal set or uncertainty guarantee. It does not
inherit either paper's formal safety guarantee and does not simulate SAC's
future closed-loop decisions. The separate verified-policy priority remains
related to [Wabersich and Zeilinger's minimum-deviation objective](https://arxiv.org/abs/1812.05506).

The evaluated quick preset is `PREFER_CERTIFIED_POLICY=True` and
`POLICY_FEEDBACK_PRESERVATION=True`; `ONE_DECISION_COMMIT`, `SOFT_RECOVERY`,
`FEEDBACK_BACKUPS` and `SIDESLIP_RESCUE` are false.
`FEEDBACK_ONLY_INFEASIBLE=True` remains recorded but has no effect while
`FEEDBACK_BACKUPS=False`. Selected v4's observer and free-space memory remain
enabled, with countersteer, nominal-plan retention and dual braking disabled.
The exact effective settings and source hashes are frozen in
`results/safety_dev/quick_v5_budget100/runs/policy_feedback_v1/metadata.json`.

Four development attempts preceded the comparison:

| Probe | CRP-VS-04 | CRS-CV-04 | Goals rescued |
| --- | --- | --- | ---: |
| Priority only, `dev_priority_v1` | Boundary contact | Boundary contact | 0 |
| Priority + policy feedback, `dev_policy_feedback_v1` | Obstacle contact | Boundary contact | 0 |

The new branch was evaluated three times in each second-probe episode and
preserved the policy once in CRP-VS-04, zero times in CRS-CV-04. That one
preservation did not prevent collision. The four runs are repeated tests of
two development cases and do not add four independent comparison scenarios.

Fourteen focused tests in `tests/test_safety_policy_feedback.py` pass. They
cover exact-policy preservation, unchanged commitment, safe-continuation
eligibility, skipped gates, unchanged original selection after rejected
backups, the existing trigger boundary and optional retention of only the
newly checked trajectory. No formal safety conclusion follows from these
unit regressions.

The fixed quick comparison **COMPLETED** under tag `policy_feedback_v1`
(root exec session 97844 exited 0 and is closed), using sample manifest v3 and
one serial, single-thread worker. It ran 24 selected canonical simulated
frozen/field scenarios x off/v4/v5 = 72 episodes; together with the four
development probes, **76 actual new runs** consumed the shared 100-attempt
budget. Final results are below. Do not tune this candidate from the fixed
subset's outcomes. Count 24 paired scenarios per comparison, not 72 independent
pairs. Step counts such as `policy_feedback_evaluated_steps`,
`policy_feedback_preserved_steps` and `continuation_override_prevented_steps`
measure controller activity, not distinct episodes or collision rescues;
`safety_v2_steps` counts issued-action changes, not every checked decision.

For the cancelled full sweep, the final
[v4 stopped report](../results/safety_dev/v4_stopped_full_sweep_report.md)
supersedes the interim counts while preserving that historical snapshot.
It contains 3,347 records and 1,647 matched B/R pairs: v4/off goals 1,525/1,521,
collisions 122/126, and goal gains/losses 50/46. Contacts are obstacle 18/31,
boundary 26/13 and target 78/82. The missing off/B-06-040 record stays excluded.
The complete original map has 3,347 recorded and 5,323 unrecorded identities
out of 8,670; neither those sequentially stopped pairs nor the quick sample
supports a population-level performance claim.

## Completed quick comparison

The 72 comparison episodes and four development probes completed within the
user's 100-new-run cap. There are **76 actual attempts**, 72 comparison results
and 24 matched scenarios per controller comparison. The two earlier probes
repeat the same two development cases and are reported separately above.

| Mode | Goals / 24 | Obstacle | Boundary | Target | All collisions | Timeout | Changed-action steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Policy only | 18 | 4 | 0 | 2 | 6 | 0 | 0 |
| V4 | 19 | 2 | 1 | 2 | 5 | 0 | 152 |
| Experimental v5 | 19 | 1 | 1 | 3 | 5 | 0 | 121 |

V5 gains/loses **0/0 goals versus v4**, and **3/2 versus policy alone**. Both
filters reach 79.17% success on this small sample, versus 75.00% for the policy.
V5 changes only one terminal outcome versus v4: FV-HO-12 changes from obstacle
to target contact. That is a contact-type trade, not a collision reduction.
V5's 31 fewer changed-action steps show reduced deviation on these trajectories;
they do not establish greater safety. **No evidence here supports v5
outperforming v4; keep v4 as the reference and v5 experimental.** No promotion,
source revision or tuning on these held-out outcomes follows from this result.

| V5 activity counter | Decisions | Episodes with activity |
| --- | ---: | ---: |
| Continuation-bonus overrides prevented | 39 | 9 |
| Verified policy preserved in recovery | 132 | 10 |
| Policy-feedback check evaluated | 42 | 15 |
| Policy-feedback check accepted | 6 | 3 |

These counters overlap and must not be summed as distinct interventions. The
132 policy-preserved decisions include cases where the old selector could
already choose the policy; the narrower 39-step counter identifies prevention
of the continuation-bonus override. The six feedback acceptances are not
necessarily six avoided interventions because the old selector could still
select the same command under a different recovery state. None of these
counts is an episode rescue or evidence of a lower collision rate.

Summed episode time was **408.86 seconds (about 6.8 minutes)**, excluding
setup/reporting time. Mean episode times were 2.53 s off, 7.13 s v4 and 7.37 s
v5; these are observed runtime figures, not a hard real-time guarantee.
**All 99 focused tests passed**: 81 filter and 18 runner tests, including the
14 policy-feedback regressions described above.

Authoritative outputs are in
`results/safety_dev/quick_v5_budget100/runs/policy_feedback_v1/`:
[report](../results/safety_dev/quick_v5_budget100/runs/policy_feedback_v1/report.md),
`complete_24cccc8f.json`, per-attempt episode JSON files, and `metadata.json`.
The 79 evaluated source files were checked against that metadata and archived
in `evaluated_sources.zip` with `evaluated_sources_manifest.json`; the checkpoint
is identified by its unchanged SHA-256 rather than duplicated in the archive.
The run uses the preselected sample manifest v3 and fresh off/v4 references.

This is a coverage-oriented simulated diagnostic, not a population estimate or
new real-world trial. The original cancelled 8,670-record completion map is
unchanged at **3,347 committed and 5,323 unrecorded**; the new-revision quick
campaign is separate and must not be added to that inventory as if it completed
the cancelled sweep. The sampled finite-horizon filter still has no formal
collision-free guarantee.

## Next hypothesis: offline brake-response calibration on development logs

Development trace inspection, not the quick held-out outcomes, motivates
checking the brake-response model at onset. In
`results/safety_dev/v4_selected_contacts_steps.csv`, the two development losses
CRP-VS-04 and CRS-CV-04 contain six isolated negative-RPM brake pulses: steps
5/8/17 and 11/39/42 respectively. None is part of consecutive brake decisions.
A fix that only carries braking-delay age between consecutive decisions cannot
explain or repair a mismatch on those isolated onsets. This observation is
from code/log inspection, recorded with its source SHA-256 in
[development_brake_pulse_audit.json](../results/safety_dev/development_brake_pulse_audit.json);
it is not a new calibration experiment.

A future investigation should use other original DV3 measured-state and
issued-command logs to calibrate the identified brake response offline,
checking surge/sway/yaw response and trajectory error on separate development
traces. True ego may score prediction error in diagnostics; it must not enter
production filtering. Fit no parameter against held-out outcomes and keep
this completed candidate frozen. The earlier observer+memory+dual-brake
envelope achieved **118/150 goals versus 123/150** for selected v4, so a more
conservative envelope is not evidence of a better closed-loop controller.
Any future model correction requires development-only validation and explicit
budget accounting. This is a proposed next investigation; no additional
calibration, source edit or episode run was performed for this report.
