# V7: preserve feasible SAC actions, diagnose failures in test set v2

Date: 2026-10-03. Status: experimental implementation and saved-data analysis;
**zero new episode runs**. The current instruction permits saved-data
analysis and code tests only. Do not use the one unused V6 budget slot or start
a new evaluation. V7 has no measured success rate.

## Scope and evidence

The 1,000-case test set v2 is designated for safety-layer
development. These cases, including their frozen-suite and simulated field
members, are now development evidence for this work; they cannot subsequently
serve as untouched safety-layer validation. This does not authorize tuning on
additional frozen/field cases outside this set, modifying Paper 2 files, editing
`constants.py`, retraining SAC, or interrupting PPO.

The policy is SAC baseline 3, the kept 3M checkpoint. Canonical saved rows:
`results/test_set/v2/sacs0_bl3/episodes.csv`; definition digest starts
`6df8076223414a88`. Exactly 1,000 identities are present, all safety off:

| Outcome | Count |
| --- | ---: |
| Goal | 872 |
| Target contact | 61 |
| Obstacle contact | 55 |
| Boundary contact | 12 |
| Timeout | 0 |

Reaching 95% on this fixed inventory requires at least 78 net rescued goals;
each newly lost SAC success increases the number of rescues required.

Crossings account for 76 of 128 failures and 46 of 61 target contacts.
The simulated deployment layouts account for 76 failures among 239 episodes;
the frozen source contributes 52 among 761. All 15 L2 head-on cases fail:
12 obstacle contacts and three boundaries. The repaired L1 being-overtaken
cell succeeds in all 15 cases, so the previously impossible starts must not
be carried into this failure inventory.

Saved historical V4/OFF results have 735 matching scenario/seed pairs within
v2, all from the frozen source. They record 684 OFF goals and 680 V4 goals:
19 rescues and 23 lost OFF successes. The lost successes end in 16 target,
four obstacle and three boundary contacts. These are a partial historical
comparison, not V6/V7 results or an estimate for all 1,000 cases. The offline
join report documents checkpoint/settings provenance and its limits.

The v2 files contain episode summaries, not trajectories. Whole-episode
minimum target range, mean speed, RMS cross-track error and final outcome are
retrospective statistics. A classifier fed those fields at runtime would leak
future information. Scenario names, layout IDs and collision labels likewise
cannot be online triggers. No validated
success/failure classifier is therefore fitted or claimed from these files.

## Implemented control change: recheck SAC within the proposed backup

`src/safety_v7.py` subclasses V6 without changing V6's implementation. When
V6 proposes an override and retains a recovery plan:

1. Preserve the current pre-command actuator history and perception snapshot.
2. Copy the proposed backup and replace its **first 0.5-second command** with
   the current SAC command. Preserve every subsequent stored command. Pad a
   shortened stored tail to the original eight-second horizon using V3's
   existing centred-rudder continuation convention.
3. Re-evaluate the entire modified trajectory with the existing predictor,
   obstacle/target/boundary checks, terminal check and 0.15 m trigger margin.
   If enabled, the inherited dual braking envelope remains required.
4. Only if the complete check passes, return SAC's action, retain that exact
   checked backup, and reset recovery counters. Restore the pre-command
   actuator history and advance it once with the action actually returned.
5. If it fails, retain V6's proposed action, braking request, plan and history.
   Without a stored plan there is no extra policy pass. Command-rate-limited
   mode skips repair because the inherited predictor does not model that limit.

This targets a concrete unnecessary-intervention mechanism: V6's chosen
backup may still work after SAC acts once, even when the earlier action bank,
one-second commitment assumption or continuation preference rejected SAC.
It is not the failed blanket one-decision-commit ablation (V5 plan records
111/150 versus V4's 123/150). It does not invent future SAC observations,
lower a clearance threshold, or claim to predict eventual episode success.

**Method source:** Wabersich and Zeilinger (2021), *A predictive safety filter
for learning-based control of constrained nonlinear dynamical systems*,
Section 4.1, Eq. (5a), minimizes deviation of the first action subject to a
feasible backup: <https://arxiv.org/abs/1812.05506v4>.
The single-plan substitution here is an engineering adaptation, not their
optimization algorithm, uncertainty treatment or formal guarantee. V6's
sampled search and provisional-track methods retain their citations in the
V6 plan and source modules.

## Implemented diagnostic: action-conditioned hazard evidence

`src/safety_risk_monitor.py` takes only the existing onboard snapshot,
current policy command, issued-command history and predictor callback. V7
logs its result in `last["risk_monitor"]`; it has **no control authority**.

For remembered static returns, each known boundary edge and each estimated
target ID, it records current hull clearance, the predicted minimum over the
existing one-second commitment, first constraint violation, measured clearance
trend, and consecutive fresh threatening frames. A violation within the next
decision is labelled urgent; two consecutive fresh threatening frames are
labelled persistent. These are explicit, unvalidated diagnostic choices,
not learned probabilities. A one-second warning may be too late for this hull.

Missing/stale observations break persistence. A fresh pose does not imply a
fresh return for every remembered static point. The nearest static return
can change, and target association can change. A missing recommendation is
never treated as proof that the policy will succeed.

**Method sources:** Hsu, Hu and Fisac (2024), *The Safety Filter: A Unified
View of Safety-Critical Control in Autonomous Systems*, motivates separating
the monitoring and intervention components:
<https://arxiv.org/abs/2309.05837>. Predicted constraint checking follows the
Wabersich and Zeilinger source above. Persistence and urgency definitions are
diagnostic hypotheses made here; neither paper supplies these thresholds.

## Integration and validation

Experimental opt-in: set `constants.SAFETY_VERSION = 7` at runtime and use
the existing emergency-stop-enabled environment. `constants.py` and its
default version remain unchanged. Native environment dispatch adds only a
V7 branch before V6; older filter branches retain their behavior.

96 focused tests passed. Synthetic tests check complete-horizon replacement, preserved backup tails,
margin/violation rejection, braking and delayed-command bookkeeping,
diagnostic-only recommendations, and native dispatch stopped before physics.
No environment episode, policy rollout, training job or old evaluation queue
was started. Final test counts and source hashes are recorded with the offline
report under `results/safety_dev/testset_v2_offline/`.

Another task concurrently added the training start-clearance redraw and helper
to `src/env.py`. The final audit preserves and isolates that change from the
V7 dispatch branch. Eleven dispatch tests were repeated successfully against
the current file, stopping before physics; no training behavior was exercised.

## What remains unproven

No saved v2 decision traces means no measured trigger precision, warning lead
time, false-intervention rate or V7 goal improvement. The historical losses
identify whole episodes harmed by V4, not the exact first harmful override.
The model can still certify an inaccurate trajectory, and a finite backup
search can still miss a feasible action. V7 inherits V6's other fallback
limitations; this work does not certify collision avoidance.
Repeatedly finding a feasible backup after another policy action can also
postpone executing the escape; there is no invariant terminal set or proof
of progress to goal. Preserving one action is not a demonstrated episode rescue.

If later authorized, collect SAC-only decision traces first on failed cases
and nearby successful controls, including the 23 historical lost successes.
Log observations/actions before intervention and preserve all saved records.
Compare matched outcomes and intervention timing, then branch from identical
pre-intervention state and random-generator state to assess whether an override
actually helps. Fit any trigger only on causal, currently available features,
split by underlying scenario rather than frame, and retain independent cases
for validation. Report rescued failures and newly broken successes separately.
No such future runs are authorized or queued by this plan.
