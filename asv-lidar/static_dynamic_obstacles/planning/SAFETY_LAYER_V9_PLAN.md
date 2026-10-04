# V9 follow-up to the V8 trigger-counterfactual plan

2026-10-03. **Fresh pilot completed: V9 ties V8 overall, with mixed regressions.**
A later decision to run new episodes superseded the earlier no-new-runs
restriction for this comparison. The separate fixed pilot completed 96 fresh
runs; previous campaigns and their ledgers remain untouched.

## Fresh paired pilot: measured result

See `results/safety_dev/v9_paired_pilot/report.md`, `paired.csv`, `summary.json`
and the independent `verification.json`. Selection was fixed before running:
32 scenarios (27 test-set-v2, five DV3) x SAC alone/V8/default V9. These are
deliberately enriched development cases, not a representative success-rate
estimate. The same kept SAC baseline-3 3M checkpoint, cached scenes and reset
seeds were used. One process/one Torch thread completed all 96 runs in 28 minutes.
There were no retries, omitted attempts, threshold changes or source drift.

| Controller | Goals / 32 | Rescued SAC failures | Lost SAC successes |
| --- | ---: | ---: | ---: |
| SAC alone | 16 | - | - |
| V8 | 16 | 10 | 10 |
| V9 | 16 | 5 | 5 |

V9 gains six goals over V8 and loses six others. On the selected test-set-v2
cases V8/V9 score 12/27 versus 15/27; on selected DV3 cases they score 4/5
versus 1/5. All six successful controls remain successful. These slices are
purposively selected, so neither their rates nor the overall 50% can replace
V8's earlier full-set 912/1000 result.

The first command difference in 11 of the 12 goal switches is rejection of a
`last certificate`: five gains and six losses. The other gain is
`TS2:P2-L2-HO-FIX-19`, where `policy margin dominates` first occurs at decision
14: both plans pass, with 0.01481 m proposed clearance versus 0.04012 m for the
SAC prefix. That case changes from V8 boundary contact to V9 goal, while SAC
alone hits an obstacle. Subsequent decisions differ too; this does not isolate
the independent effect of either guard.

V9 suppresses 38 proposals (34 failed current checks, four margin preferences)
and changes 106 actions versus V8's 393. Reduced intervention has not increased
net success. Three known positive-margin, never-intervened target collisions
still occur. A failed finite backup check is not a reliable classifier of the
closed-loop policy's eventual outcome; this pilot alone cannot separate model
error from a restrictive backup family and later replanning.

All 64 fresh SAC/V8 outcomes reproduce the historical selections. The audit
checks all 96 attempt/result/trace records, 32 triples, 4,208 decision records,
89 archived/current source files, cached scenes, configuration and checkpoint.
The report's 17 synthetic validation tests passed; no additional episodes were
used for reporting or verification. Controller code and constants are unchanged.

**Decision:** keep V9 experimental; this pilot does not justify replacing V8.
Next evidence should isolate the two guards in a fixed matched ablation and
address never-triggered target hazards separately. No extra run is queued.

## Verified starting point

`tools/diagnostics/safety/v8_followup_audit.py` reconstructs the 1,150 existing
cases from base and nohold CSVs. It checks unique cases, seeds, main outcomes,
contiguous steps, fire counts and first-fire branch coverage. A firing case
without a branch is rejected, not assigned a fabricated result. Of 549 cases
in the nohold subset, 526 have a first-fire branch and 23 never fire; the other
601 cases never fired under base V7 and retain their recorded SAC outcome.

| Set | SAC | V7 | V8 | V8 rescues / losses |
| --- | ---: | ---: | ---: | ---: |
| Test set v2 (1000) | 872 | 904 | 912 | 57 / 17 |
| DV3 (150) | 111 | 128 | 127 | 20 / 4 |

V8's 88 test-set failures are 31 target, 27 obstacle and 30 boundary contacts.
They comprise 71 unrescued SAC failures and 17 newly broken successes. The
full case inventories and hashes are under
`results/safety_dev/v8_followup_offline/audit/`.

The earlier phrase "V8 never overrides without a certified escape" is false:
V8 removes hold-back, but still inherits `last certificate`. That branch
executes a stored continuation after its current check fails. At the first
fire it has zero rescues/three losses on test-set v2, but four rescues/one loss
on DV3. These counterexamples prohibit claiming that removing it will improve
total success. V9 tests that stricter design explicitly and remains separate
from measured V8.

Do not add a uniform 0.15 m intervention floor: 22 of the 57 V8 test-set rescues
began below that margin. A larger margin is not supported by this analysis.

## Baseline fix: isolate V8's nohold switch

V6 now obtains the hold-back gain through `_hold_back_gain_s()`, whose default
still returns the existing V2 constant. V8 overrides only that method to
return infinity. The old temporary global assignment could affect another
filter's nested/concurrent call even though it restored the setting afterward.
This refactor preserves serial nohold behavior without global mutation.
No safety threshold or V2–V7 default was changed. This is an implementation
isolation fix, not a new collision-avoidance method or a new performance result.

## Experimental V9 control changes

### Currently passing backup required for an override

After V8/V7 proposes a changed action, V9 constructs the actual returned
first command, preserving full-astern NaN semantics, followed by the stored
backup. Short tails use the existing centred-rudder continuation padding to
the full eight-second horizon. It rechecks this sequence from the current
snapshot and **pre-command** actuator history, including static, boundary,
target and existing terminal constraints. The optional inherited dual brake
envelope remains required when enabled.

An override passes only with first-violation time +infinity and nonnegative
minimum clearance after the existing hull margins and gaps. A negative margin
does not become acceptable because it is less negative than SAC's. If the
proposal fails, V9 returns SAC, clears recovery and restores/advances the
command history once. That fallback explicitly does **not** certify SAC safe.
The same applies when no backup exists or the unmodelled command-rate limiter
is active. This may lose the old unchecked fallback's empirical rescues.

### Compare SAC and the override with the same backup tail

The same batched check includes an otherwise identical sequence whose first
command is SAC's action. If both sequences pass and SAC's minimum clearance
is at least the override's, V9 preserves SAC and retains that exact checked
backup. If only the override passes, or it has more clearance, it remains
eligible. This primarily adds hard-safe comparisons below V7's 0.15 m prefix
repair threshold; it does not impose a higher threshold on existing rescues.

The comparison uses raw paired first-violation times and clearances, not
`policy_margin` (rounded, best of a different recovery bank) or `best_margin`
(excluding brakes). Old shadow CSVs omit the repair first-violation result and
the full plans, so comparing their margins alone cannot validate this gate.

`REQUIRE_CURRENT_PLAN` and `PREFER_POLICY_MARGIN` are independent switches,
both true by default in V9. Constructor overrides are also supported. Disable
both with zero target-turn rate to retain the parent's action/plan behavior;
single-switch runs can later isolate each mechanism. These ablations are
synthetic-tested, not episode-evaluated.

**Method inspiration for both changes:** Wabersich and Zeilinger (2021),
*A predictive safety filter for learning-based control of constrained nonlinear
dynamical systems*, Section 4.1, Eq. (5a): minimal first-action deviation
subject to feasible backup constraints, <https://arxiv.org/abs/1812.05506v4>.
Current rechecking and the same-tail preference are an engineering adaptation made here.
The paper's uncertainty treatment, recursive feasibility
proof and terminal invariant set are not implemented. Repeated feasible deferral can still prevent
progress; improved predicted margin does not prove improved episode outcome.

## Optional target-motion ensemble (disabled by default)

`src/safety_target_prediction.py` copies estimated target position, velocity
and hull heading into an immutable per-decision object. It includes CV and
sampled constant-speed port/starboard turns at an explicitly supplied rate.
Both the velocity direction and hull orientation rotate. An optional measured
turn rate can be supplied by a caller; this module does not estimate history.

The minimum SAT hull gap across these hypotheses supplements, never relaxes,
the existing checks. V9 caches one envelope per decision, shared by primitive
plans, CEM, continuations, V7 prefix repair and the final paired check. Static,
boundary, braking and terminal checks stay active. The separate risk monitor
remains CV-only shadow diagnostics and cannot bypass a check.

Default `safety_v9.TARGET_TURN_RATE_DEG_S = 0.0` preserves CV. A positive runtime
value or `SafetyFilterV9(target_turn_rate_deg_s=...)` enables the prototype.
No positive rate has been fitted or selected on episode outcomes. It is a
finite sample of motions, **not** a bound on every target manoeuvre, an IMM
tracker or an implementation of Rule 17. Acceleration, turn onset, reactive
responses, tracking errors and unobserved vessels remain uncovered.

**Method sources:**

- Johansen, Cristofaro and Perez (2016), *Ship Collision Avoidance Using
  Scenario-Based Model Predictive Control*, Sections 3.3–3.4, discusses
  uncertainty represented by multiple target-motion scenarios, including
  unexpected manoeuvres. [Author manuscript](https://torarnj.folk.ntnu.no/colregs_cams.pdf),
  DOI [10.1016/j.ifacol.2016.10.315](https://doi.org/10.1016/j.ifacol.2016.10.315).
- Li and Jilkov (2003), *Survey of maneuvering target tracking. Part I:
  Dynamic models*, IEEE TAES 39(4), 1333–1364, describes constant-turn model
  families. DOI [10.1109/TAES.2003.1261132](https://doi.org/10.1109/TAES.2003.1261132).
  The compass-coordinate closed form here was derived and checked synthetically,
  not copied as a calibrated vessel model from this paper.

## Integration, tests and provenance

Runtime `SAFETY_VERSION = 9` selects the experimental filter. The default version
and `constants.py` remain unchanged. Frozen/Paper2 suite CLIs accept version 9;
the counterfactual runner now selects actual V8/V9 classes instead of silently
treating every non-V4 string as V7. Unsupported versions fail explicitly and
future config files record the requested versions. Existing results are untouched.

134 focused pytest tests passed, plus three supplemental synthetic data checks.
Tests cover V8 isolation, inherited behavior, rejected stored
plans, hard-safe sub-trigger choices, paired-margin ties, signed braking,
actuator history, target kinematics/geometry and native dispatch halted before
physics. The legacy `tests/test_safety_v8.py` performs an environment episode
and was deliberately not run under the no-new-runs restriction. Source hashes,
test commands and counts are in the follow-up report's verification artifacts.

The frozen-source and simulated field members of test set v2 remain development
data for this safety layer by designation. No additional held-out
set was tuned. Paper 2's project directory, constants, checkpoints, PPO work and
earlier evaluation ledgers were not modified by this task.

## Limits and next evidence needed

V9 now has the targeted pilot result above, but no full-set success rate. The
older saved counterfactual CSVs lack target
states, LiDAR geometry, full candidate commands and serialized environment
copies; branch outcomes alone cannot re-score target uncertainty or perform
the exact new paired checks. The fresh pilot measures lost rescues as well as
preserved policy successes. Separate effects of the current-plan rule, margin
preference and target ensemble still require their own matched evaluation.

The pilot is complete; no additional evaluation is queued. The existing first-fire
oracle is only a choice between two recorded outcomes per case; it does not
bound a different trigger time or an improved rescue controller.
