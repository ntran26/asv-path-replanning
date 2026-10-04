# Safety layer — debugging notes (2026-10-02)

**Complete standalone history:** [SAFETY_LAYER_COMPLETE_SUMMARY.md](SAFETY_LAYER_COMPLETE_SUMMARY.md) consolidates V1-V19, citations, results, diagnosed failures, rejected approaches and the requirements for a defensible safety guarantee. Read it first; this file retains the chronological history.

This is a self-contained brief for continuing the safety-layer work in a new session.

Project root: `asv-lidar/static_dynamic_obstacles`. Run every command from that folder.

## Latest continuation: reachability containment prototype (2026-10-03)

Read [SAFETY_REACHABILITY_STAGE1_PLAN.md](SAFETY_REACHABILITY_STAGE1_PLAN.md) and [the saved-data results](../results/safety_dev/reachability_stage1/README.md). The new standalone `src/safety_reachability.py` models a conditional interval extension of the numerical plant; it is not a numbered controller or an online certificate. SAC/V16, constants, environment and checkpoint remain unchanged. There were zero new episodes and 101 focused tests passed.

BAS-NU decision 11: initial and 0.5-second recorded states are contained under the declared bounds, but both tested interval expressions exceed a usefulness guard at 1 second of the requested 8 seconds. Exact algebra reduces the 0.5-second x/y interval widths from 1.083/1.149 m to 0.802/0.811 m; this remains too broad for close clearance. BAS-HO records an 81.884644-degree target turn over 0.5 seconds, contradicting a global 8 degrees/s envelope. Gaussian sensor/parameter noise has no declared finite support; the diagnostic's three-sigma boxes are conditional hypotheses, not guaranteed limits. Target corridor resets and the mismatch between boundary predicates remain explicit model obligations.

Both source versions, assumptions, widths and hashes are preserved under `results/safety_dev/reachability_stage1/`. No mission-safety or success-rate improvement is claimed. The next step needs useful dependence-preserving enclosures and full hybrid target/geometry semantics, followed by a verified feedback/terminal continuation. Do not promote this prototype or shrink its bounds to make saved trajectories pass.

## Latest continuation: V18/V19 and disk cleanup (2026-10-03)

Start with `results/safety_dev/v18_development/report.md` and the V18/V19 plans. All90 new runs are complete:36 targeted development runs (12 scenarios x V16/V18/V19) and54 primary-v3 subset runs (18 scenarios x OFF/V16/V19). No retries or evaluations remain queued.

Neither new candidate improves the measured goal count over V16; neither is promoted. Development results are5/12 for all three filters, retaining the same six broken SAC successes. V18 activates but changes no commands; V19 changes one boundary failure to an obstacle failure. On the separately frozen18-case primary subset, fresh SAC reaches14 goals and V16/V19 each reach15, rescuing the same case and preserving all14 SAC successes. This is18/1000 v3 scenarios, not the full benchmark or proof of100% safety. The subset was selected from metadata before outcomes; no primary tuning occurred.

585 completed trace files were losslessly NTFS-compressed, recovering974,323,274 allocated bytes (0.907GiB), with every content hash preserved. Paths, source archives, results and scene caches remain intact; no files were deleted. Final verification is `results/safety_dev/v18_development/final_verification.json`. Native versions18/19 are selectable; defaults, constants, checkpoint and Paper2 remain unchanged. Other jobs were not signalled or stopped.

Two further causal physical-model-bank audits (one-step and continuous training objectives) still failed broader held-forward prediction checks. Neither is integrated; stop that family without further weight/window tuning. Citations and evidence are in the plans and `v10_iterations/audits/bootstrap_*bank_cohort/`. The earlier399-run campaign below is a separate completed batch.

## Primary evaluation target: test set v3 (user update, 2026-10-03)

**Test set v3 is now the main evaluation set.** Start with `planning/SAFETY_EVALUATION_PROTOCOL.md` and `results/safety_dev/testset_v3_main/README.md`. The canonical frozen set has 1,000 cases (664 frozen-derived, 336 field-layout); 755 geometry/seed pairs are shared with v2 and 245 are new or changed. It is not wholly unseen because v2 was used for development. Earlier v2/DV3 results below remain historical development evidence, not v3 results.

The generic safety runner now defaults to the complete v3 selection and writes TS3 results under `results/safety_dev/testset_v3_main/runs/<tag>/`. Explicit historical selections keep their old output root. IDs are namespaced `TS3:` and checked against the original cached geometry and episode seeds; no scene regeneration occurs. A `--cases` quick subset is explicitly labelled as a subset, not the full benchmark.

Primary comparison: the unchanged SAC baseline 3 kept best at 3M policy, safety OFF versus frozen V16. The saved v3 SAC result is 853/1,000 goals (71 obstacle, 66 target, 10 boundary contacts), but 755 rows were reused and no source/config/checkpoint archive was saved. Treat it as historical; use fresh matched OFF/V16 for current-runtime comparison. V17 was not promoted. Do not tune on the primary evaluation outcomes without a new explicit change of scope.

The benchmark switch prepares the suite and runner; it does not itself launch a new evaluation. Full paired evaluation means 2,000 episode runs. Keep at most two evaluation processes and protect the active `pilot_v4_A` training. No old queue, constants, checkpoint, Paper 2 folder or completed result ledger is changed.

## Previous completed work: V16, V17 and paired challenge (2026-10-03)

Start with `results/safety_dev/v10_iterations/report.md`, `planning/SAFETY_LAYER_V16_PLAN.md` and `planning/SAFETY_LAYER_V17_PLAN.md`. All **399 new episode runs are complete**, covering 72 distinct scenarios across 19 experiment tags, with no retries or queued evaluations. `final_verification.json` and `campaign_status.json` record completion and unchanged constants/checkpoint hashes. The earlier 96-run V9 pilot and all older ledgers remain separate.

**The objective is not achieved.** Default V16 reaches 25/32 goals versus SAC/V9 16, rescuing 11 SAC failures and preserving 14/16 SAC successes. All six successful controls remain goals. BAS-HO-NC-059 (target) and BAS-NU-CV-070 (boundary) are the two broken SAC successes. There are four target, two obstacle and one boundary contacts. See `reports/motion_axis32` in the campaign folder for the verified disjoint 5+27 cohort.

On the separate, previously frozen 40-case challenge, fresh SAC reaches 21 goals and V16 reaches 25. V16 rescues 8/19 failures, preserves 17/21 SAC successes and all 10 successful controls. The four lost SAC successes are DV3-CRS-CV-04, P2-L2-CRS-FIX-19 and P2-L3-BO-FIX-09 (boundary), and P2-L3-CRP-FIX-03 (target). Boundary contacts increase 1 to 6. Fresh SAC reproduces all 40 historical outcomes. See `reports/v16_broader40_paired`. These enriched development cohorts do not estimate full-suite success rates; keep their denominators separate.

V16 adds motion-axis known-hull completion for eligible fresh base tracks. V15 adds conditional current-policy-prefix search using the exact parent's selection floor. Earlier versions improve motion admission/persistence and partial-hull estimates. Citations, tests and limitations are in implementations and plans. Runtime truth, scenario IDs and future policy actions are not used. Native versions through 17 are selectable; existing defaults remain unchanged. Native integration occurred after evaluated source freezes, with separate V16/V17 source/newline audits. The diagnostic runner explicitly installed and verified the requested class during evaluation.

Two final shortcuts did not improve the result. V16 with `prefer_any_feasible_policy=True` reaches 3/9 versus default V16 6/9, with zero gains and three losses. V17's fresh measured-yaw option reaches 6/9: it rescues DV3-HO-VS-01 but loses P2-L2-HO-FIX-19, leaving the same two broken SAC successes. Keep V16 as the stronger measured candidate; neither option is promoted. Verified reports are under `reports/v16_feasible_probe9` and `reports/v17_fresh_yaw_probe9`.

Key remaining mechanisms: BAS-NU has own-vessel prediction error against static geometry despite a successful recorded SAC continuation. BAS-HO combines a lower-margin passing policy backup, sharply changing target behaviour on the recorded V16 branch, and delayed persistent-track refresh; future target states on the SAC-only branch were not recorded. Audits: `bas_nu_v15_gate`, `bas070_initial_state_decomposition`, `bas_ho_v16_first_divergence`. A causal past-measurement yaw-response gain fit improved one-step errors but worsened continuous forecasts on most tested V15 cases; it is rejected for production. See `audits/causal_yaw_response_cohort`. Do not turn that diagnostic fit into a released controller or claim the remaining failures are unavoidable.

Tests passed: 184 focused V16/core/report checks, 23 new yaw-observer tests, 28 final native-dispatch checks and 76 reporting checks. Counts overlap. The only warning was a third-party NumPy deprecation. No other training jobs were signalled or stopped.

The latest user request authorizes continuing development evaluations. Earlier saved-only restrictions and numerical caps below are historical and superseded for this continuation; do not ask for the same authorization again. Keep at most two evaluation processes; preserve other jobs, constants, checkpoints, Paper 2 and all old ledgers. Test-set-v2 and DV3 are development data by explicit user instruction; other held-out sets remain unsuitable for tuning.

Runner: `tools/diagnostics/safety/safety_candidate_iteration.py`; saved-data validator: `report_safety_iteration.py`. Every tag reserves attempts and stores source/settings/checkpoint/scene hashes, individual results, decision traces and a completion token. Do not overwrite tags, silently retry failures, restart the old 8,670-record queue, or count repeated versions as extra scenario coverage.

## Historical result: fresh V9 pilot (2026-10-03)

New episodes were authorized to compare V9. The separate campaign
`results/safety_dev/v9_paired_pilot/` is **complete: 96 fresh runs, no retries**,
32 fixed scenarios x SAC alone/V8/default V9, one process and one Torch thread,
28 minutes. No older queue or ledger was restarted. Start with its `report.md`
and `planning/SAFETY_LAYER_V9_PLAN.md`.

**V9 did not improve the net goal count in this pilot:** all three controllers
reach 16/32. V9 gains six goals over V8 and loses six. Relative to SAC, V8 rescues
10 failures and loses 10 successes; V9 rescues five and loses five. The cohort
is enriched for known failures, rescues and broken successes, so 50% is not a
new full-test-set success estimate. Selected TS2: V8 12/27, V9 15/27; selected
DV3: V8 4/5, V9 1/5. All six successful controls remain successful.

Eleven goal switches first diverge when V9 rejects `last certificate` (five
gains, six losses). One further gain first diverges at `policy margin dominates`
(`P2-L2-HO-FIX-19`). Later decisions also differ; independent guard ablations
have not been measured. V9 made 106 changed actions versus V8's 393, but lower
intervention did not increase net success. Three known never-triggered target
collisions persist. Keep V9 experimental; this does not justify replacing V8.

`paired.csv` records the first different executed command and paired margins;
`traces/` contains 4,208 decision records. `verification.json` independently
confirms all96 unique attempts/results/traces, all32 matched triples, all64
historical SAC/V8 outcomes reproduced, and matching source/cache/model hashes.
The exact evaluated sources are archived. Reporting tests: 17 passed. No
controller thresholds, constants, checkpoint, Paper 2 project or other jobs
were changed. No extra episodes or ablations are queued.

## Prior offline follow-up: V8 audit and experimental V9 (2026-10-03)

The V8 plan was read and the saved counterfactual results were audited; no new
episodes were run at that stage. Its then-current restriction permitted saved-data
analysis and code tests only; the later pilot above was separately authorized.
Do not restart the completed phases or any old
evaluation queue. Start with `results/safety_dev/v8_followup_offline/report.md`
and `planning/SAFETY_LAYER_V9_PLAN.md`.

Confirmed saved outcomes: V8 912/1000 test-set-v2 goals, 57 rescues and 17 lost
SAC successes; DV3 127/150, 20 rescues and four losses. V8's unchecked stored-plan
fallback remains: `last certificate` first fires produce 0 rescues/3 losses in
test set v2 but 4 rescues/1 loss in DV3. Removing it has no established net gain.
Three never-intervened target collisions still show positive predicted margins
at their last decision; snapshots/tracks needed to diagnose them were not saved.

V8 now disables hold-back using an instance method rather than temporarily
mutating a shared constant; serial behavior is preserved. Experimental V9
requires an override's complete proposed backup to pass its current check and
prefers SAC when the same backup tail after SAC's action also passes with at
least as much clearance. These guards have separate switches. Optional CV plus
turning-target hypotheses are implemented but disabled by default (turn rate 0).
Native runtime `SAFETY_VERSION = 9` and suite CLIs select the candidate; the
counterfactual runner now instantiates the requested class and records versions.

V9 initially had no episode-level result; the fresh pilot above now supplies one.
Current model feasibility is not a formal
safety guarantee, and rejecting an unsafe override can return an unchecked
policy action. Detailed method citations, synthetic tests, source hashes and
remaining case inventories are in the linked plan/report. The old V8 episode
test was not run. Constants, SAC, Paper 2's project and live PPO work are untouched.

## NEXT DIRECTION (user, 2026-10-03): a trigger that does not fire when the policy would succeed

**The problem, as stated:** how to trigger the safety layer correctly, so that it does
not fire when the policy would solve the case without it.

> **Status (2026-10-03, 07:00): this plan is being run in the main development workstream**, as
> decided. Results and next steps are in `planning/SAFETY_LAYER_V8_PLAN.md`.
> - Phase 1 done: 429 episodes.
> - Headline: v7 fires in 61 % of episodes the policy alone solves, but breaks only 13 overall,
>   against 79 rescues (oracle +79, v7 +66).
> - A SAC-critic gate does not help.
> - The breaks are mostly target-model mismatch (reactive or non-compliant targets) and
>   wall-adjacent crossings.
>
> Do not duplicate it. The four perception wrappers' `__getattr__` were guarded for deepcopy
> (behaviour unchanged).
>
> **Update 10:30:** closed loop on all 1,000 test-set-v2 episodes:
>
> | | Test set v2 | DV3 |
> |---|---|---|
> | SAC alone | 0.872 | – |
> | SAC + v7 | 0.904 | 128/150 |
> | **SAC + v8** | **0.912** | 127/150 |
>
> **v8** (`src/safety_v8.py`, `SAFETY_VERSION = 8`) is v7 without the hold-back fallback: with no
> certified escape, the policy's action stands. Hold-back was the only first-fire reason with more
> breaks than rescues.

**Historical run budget (superseded by the current authorization above):** this plan needed new episode runs. The earlier instruction (saved-data analysis and
code tests only) still stands until lifted, so **confirm the run budget
before starting any episodes.**

### Why the current trigger fires wrongly

The evidence:
- Saved v4 against safety off, on the 735 frozen test-set-v2 pairs: **19 rescues, 23 broken
  successes**.
- Development set: v4 gains 19 and loses 7.

Every version (v2 to v7) asks: *if the policy takes this action and the filter then runs one of
its own backup manoeuvres, is the hull clear for 8 s?* That test cannot know what the policy will
do next:
- **It is open-loop.** It assumes the policy's future is the filter's backup. The real policy keeps
  reacting, and often escapes in ways the finite backup library cannot represent.
- **It is model-based and conservative.** Inflated hull, gaps, an 8 s horizon and hand-set margins
  make it call states unsafe that the policy can handle.
- So a fire means "the filter found no way out", not "the policy will fail". V7's re-check of the
  stored backup after the policy's action is a step towards fixing this, but it still checks only
  the filter's own plan, not the policy's continuation.

**The criterion to aim for:** fire only when (a) the policy, left alone, would fail, **and** (b) the
intervention does better. Both are counterfactual questions about the policy, so the trigger must
be **policy-aware**.

### Verified building blocks (checked 2026-10-03, one L2 head-on episode, SAC 3 M)

- **Environment cloning works.** `copy.deepcopy(env)` mid-episode takes about 4 ms, and the clone's
  next step is identical: same pose, same reward. So counterfactual branches from the exact same
  state, RNG included, are cheap.
- **SAC's critic sees trouble coming.** Policy alone on L2 head-on: Q(s, π(s)) falls steadily from
  −2.8 (step 0) to −4.3 (step 8) and −7.6 (step 16); obstacle contact at step 19. That is roughly
  5 s of warning, from one episode, so separability is not yet measured. The values are in
  VecNormalize-scaled reward units. Code:
  ```python
  o, _ = model.policy.obs_to_tensor(obs)
  with torch.no_grad():
      a = model.policy.actor(o, deterministic=True)
      q1, q2 = model.critic(o, a)
  q = float(torch.min(q1, q2))
  ```
  The critic ships with the policy, so a critic gate is deployable. Its horizon is only about 10 s
  (γ 0.951), it can be miscalibrated in unfamiliar states, and it is a **gate, never a certificate**.
  Related: learned safety critics switching to a recovery policy (Thananjeyan et al., 2021,
  *Recovery RL*); the monitor/intervention split of Hsu, Hu and Fisac (2024), already cited in the
  V7 plan.

### Protocol

1. **Shadow runs: measure the current trigger.**
   - Run SAC alone (safety off for control), with v4 and v7 computing their decision every step
     **without acting** (shadow mode).
   - Run on:
     - the 150 development episodes;
     - test-set v2's 113 failures and their 226 matched successful controls
       (`results/safety_dev/testset_v2_offline/analysis/matched_controls.csv`).
   - Log every step, using onboard information only:
     - the would-fire flag and reason;
     - the filter's margins: policy margin, best backup margin, time to first predicted
       violation under the policy action;
     - the V7 risk-monitor fields;
     - SAC's Q(s, π(s)), Q of the filter's alternative, and action standard deviation;
     - the step index.
2. **Counterfactual labels.**
   - At each episode's first would-fire step, and a sample of later ones (for example every 4th),
     `deepcopy` the env (and filter state) and run two continuations to the end:
     - (i) policy alone;
     - (ii) the filter acting from that step.
   - Label each fire:
     - **necessary** if (i) fails, otherwise **unnecessary**;
     - **helpful** if (ii) succeeds where (i) fails, **harmful** if the reverse.
3. **Trigger metrics.**
   - Precision (necessary / all fires) and recall (failures preceded by a fire).
   - Helpful and harmful counts.
   - This is the measurement the V7 plan says is missing.
4. **Fit a policy-aware gate.**
   - Fire only when the filter says unsafe **and** a gate on onboard features (the critic Q and its
     trend, the filter margins, time to violation, risk-monitor persistence) predicts that the policy
     alone will fail.
   - Start simple: thresholds or a logistic model.
   - **Split by underlying scenario**, not by frame.
   - Fit on half, validate on the other half; choose the threshold that maximises rescues minus
     broken successes on the fitting half.
5. **Closed-loop check.**
   - v7 plus the gate on the development set and test set v2, against safety off.
   - Report rescues and newly broken successes **separately**, and the intervention rate.
   - The development set and test set v2 are development evidence here (user's designation), so
     say so.
6. **Optional upper bound.** Use the cloned env as an oracle trigger: fire only when the
   policy-alone branch fails within T s. It is not deployable, but it shows how much a perfect
   trigger could gain, and so whether chasing the gate is worth it.

**Outputs:**
- data: `results/safety_dev/trigger_counterfactual/`;
- scripts: `tools/diagnostics/safety/`;
- notes: a new `planning/SAFETY_LAYER_V8_PLAN.md`, or a section in the V7 plan.

**CPU:**
- The PPO baseline-v3 run (`runs/ppo_formulation_seed0_bl3`) is training, and the baseline-v4 G4
  pilots (`results/v4_pilot_run.sh`) start after it.
- Use **at most 2 processes**, and do not stop either.
- C: has about 7 GB free. Keep logs compact: per-step CSV, no rollout dumps.

**Estimated cost:** about a normal evaluation (about 490 episodes) plus the branches, which are short.
Roughly 2–4 h on 2 processes.

## Latest: test-set-v2 analysis and experimental V7 (2026-10-03)

The 1,000 test-set-v2 cases are now designated for safety-layer
development, superseding the earlier held-out restriction for those cases only.
The historical budget answer for this completed phase was **no new runs: saved-data analysis and code tests
only**. Zero episodes were run in this continuation. Do not consume the old
remaining V6 slot, restart an evaluation, or count v2 as untouched validation
for the safety layer developed from it.

Start with `results/safety_dev/testset_v2_offline/report.md` and
`planning/SAFETY_LAYER_V7_PLAN.md`. Saved SAC has 872/1000 goals, 61 target,
55 obstacle and 12 boundary contacts. Crossings contribute 76 failures; L2
head-on fails 15/15. The offline inventory includes all 128 failures and 151
nearby successful controls. There are no saved v2 decision traces, so episode
summaries cannot train an online trigger without future-information leakage.

A scene/seed/provenance-checked join to stopped historical OFF/V4 journals
finds 735 pairs within v2: OFF 684 goals, V4 680, with 19 rescues and 23 lost
goals. These archived outcomes strengthen the need to preserve SAC successes;
they do not identify the exact first harmful override or measure V7.

`src/safety_v7.py`, selected with runtime `SAFETY_VERSION = 7`, first lets V6
propose an action. For an override with a backup, it substitutes SAC's action
for the first decision and rechecks the full eight-second plan, preserving
SAC only when the unchanged checks and 0.15 m margin pass. It restores the
pre-command actuator history and retains the exact checked replacement plan.
Failed repairs leave V6's proposal intact. `src/safety_risk_monitor.py` adds
action-conditioned per-hazard diagnostics only; its recommendations never
bypass a safety check. Both modules and the plan cite their method sources.

V7 is experimental and has **no measured success rate**. The project's environment
change adds only V7 selection; constants/defaults, Paper 2, SAC and live PPO
work were not changed by this task. Another task concurrently added training
start-clearance redraw logic to `env.py`; that change is preserved and isolated
in the new source audit. 96 focused tests passed, and all 11 native dispatch
checks were repeated after the concurrent edit, without entering physics.
Existing V6 results and ledgers remain historical records.

<!-- V6_CONTINUATION_START -->
## Current continuation completed: experimental V6, target not reached

**Up to 150 new development runs** were authorized, separate from the old
76/100 quick campaign. **149/150 are consumed: 144 completed SAC evaluations
and five legacy simulator-test runs. No evaluation remains active.** Do not
resume the old 8,670 queues or tune on frozen/field outcomes.

Fresh paired results on 50 original DV3 cases:

| Controller | Goals | Obstacle | Boundary | Target |
| --- | ---: | ---: | ---: | ---: |
| V4 | 23 | 9 | 7 | 11 |
| V6 candidate | 33 | 9 | 3 | 5 |

V6 gained 13 goals and lost three: HO-VS-01 (boundary), CRP-CV-16 and CRS-VS-03
(obstacle). The selection deliberately includes all 27 historical V4 failures
plus 23 successful controls; its 66% success is not a population estimate.
Nevertheless, 17 known failures mean this fixed candidate cannot reach 95% on
the original 150 cases: even 100/100 untested successes would give only
133/150 (88.67%). No difficult or initially unobservable case was excluded.
Do not promote it as the established best full-suite controller on this evidence.

The candidate is available through runtime `constants.SAFETY_VERSION = 6`
with `ASVLidarEnv(..., emergency_stop=True)`. Its source defaults enable
`PROVISIONAL_TRACKS`, `PROVISIONAL_MOTION_EVIDENCE`, and `TRAJECTORY_SEARCH`.
Calibration, fitted-hull perception and track history remain disabled. Existing
version selection/defaults and the SAC baseline 3 3M checkpoint are unchanged;
`constants.py` was not edited. The ordinary environment has an explicit V6
branch. An AST audit verifies that integration changed only that new branch
and the three V6 default switches/docstring, matching the evaluated overrides.

Results, exact source ZIPs, manifests, traces and accounting are under
`results/safety_dev/development_v6_budget150/`. Start with `report.md`,
`finite_suite_bound.md`, `integration_audit.json`, and
`offline/broad_v6_50_failure_classification/REPORT.md`. The completed broad tag
is `broad_v6_50`. The V6 plan documents every method, primary citations, pilots,
regressions and limitations. Figures include rescues and a lost-goal example.
The broad comparison took 283.5 summed episode seconds for V4 and 746.9 for V6
with tracing; this is evaluation wall time, not a real-time control-latency test.

The reserved final braking-calibration probe **did not run**. Its source guard
rejected an external `src/scenario.py` edit before reserving an attempt. That
file changed at 15:58:25 UTC, after the broad worker had imported its sources
and completed its first case at 15:46:51; the worker does not reload modules.
The external start-position-override changes were preserved. See the probe's
`preflight_aborted.json` and the integration audit. Do not silently restore the
other work or whitelist it into the old benchmark; a future run needs an
explicitly audited source baseline. One authorized attempt remains unused.

Validation after native integration: 147 focused filter, geometry, calibration
and mocked-dispatch tests passed. Additional campaign/report/classifier tests
are recorded in their artifacts. An expanded earlier test command also ran
five existing V2/V3 L1 simulations, conservatively charged at attempts 39-43:
four passed, while unchanged V3's dead-ahead collision assertion failed. They
are verification runs, not SAC goal outcomes, and were not rerun or tuned away.

Of the 17 remaining V6 failures, 10 end in `no escape`, three in an unchecked
stored continuation, three in a margin-qualified check, and one in a check
below the trigger margin. All five final target contacts have a snapshot track
at contact; timing/kinematic errors matter, not just presence. The three new
regressions follow extended low-margin or infeasible recovery. Offline truth
replay separately audits the two initially invisible BO panels; it is never
used by the production filter. Numerical command sampling is not a proof over
all continuous controls or grounds for changing benchmark geometry.

A next-iteration hypothesis is saved in
`offline/broad_v6_50_failure_classification/SEARCH_MARGIN_OPPORTUNITIES.md`:
13 decisions across six failed cases (including all three regressions) have no
safe original plan but a nonnegative sampled-search clearance below 0.15 m.
The search rejects these while the inherited fallback proceeds. The maximizing
plan and its first-contact result were not retained, so this is an opportunity
to instrument and validate, not evidence that a lower margin would rescue them.
No acceptance threshold was changed in this iteration.

Historical V4's 123/150 remains contextual: its exact old env/V4 sources were
not found, despite matching seeds, scene digests, checkpoint/config and other
physics/filter dependencies. Current comparisons therefore used fresh V4.
<!-- V6_CONTINUATION_END -->

## Previous completed direction: stop the full sweep; quick v5 test (2026-10-02)

The full 8,670-run evaluation was cancelled in favour of a v4 report, and
subsequent work was capped at **100 new episode runs total**, with the
emphasis on a quick test. **Do not resume the old full queues.** The current
completed work is 24 fixed canonical simulated frozen/field scenarios x off,
v4 and revised v5 = 72 comparison episodes, plus four original-development
probes = **76 actual new runs total**, below the shared 100-attempt cap.
Selection used scenario families/features, not observed outcomes. The quick
comparison **COMPLETED** under tag `policy_feedback_v1`; root exec session
97844 exited 0 and is closed. V5 and v4 both reached **19/24 goals with five
collisions**; this does not support promoting v5 over v4. Keep the evaluated
candidate experimental and unchanged; do not tune it on held-out outcomes.

The original 77 frozen sources were verified and archived before revision in
`results/safety_dev/quick_v5_budget100/frozen_77_sources.zip` with
`frozen_baseline.json`. The evaluated candidate enables
`PREFER_CERTIFIED_POLICY=True` and `POLICY_FEEDBACK_PRESERVATION=True`.
The first gives a policy action already in the existing hard-safe,
clearance-qualified recovery pool priority over the continuation bonus.
The second, before an initial nominal-mode override, checks four feedback
backups of that exact policy action when some original alternative or stored
continuation is hard-safe. It preserves the policy only if a complete backup
passes the unchanged hard checks and trigger margin. Failed extra checks do
not change the original selector. These are sampled model checks, not formal
safety certificates or predictions of the policy's future actions.

`FEEDBACK_BACKUPS=False`, `SIDESLIP_RESCUE=False`, `ONE_DECISION_COMMIT=False`
and `SOFT_RECOVERY=False`. `FEEDBACK_ONLY_INFEASIBLE=True` is retained but
inactive while the general feedback-bank switch is off. The observer and
free-space memory are unchanged from selected v4; its other ablations remain
off. The optional sideslip method is implemented and unit-tested but inactive.
Citations and limitations are in the v5 plan. `constants.py` is untouched.

Final stopped-run evidence is in
[the v4 report](../results/safety_dev/v4_stopped_full_sweep_report.md) and
`results/safety_dev/v4_stopped_full_sweep_*`. The journals contain **3,347
rows**, with **1,647 matched frozen cases**: v4/off goals 1,525/1,521 and total
collisions 122/126; obstacle contacts 18/31, boundary 26/13, target 78/82.
V4 gains 50 goals and loses 46. The partial sequential subset does not
establish population-level improvement. The missing off/B-06-040 journal row
is excluded from paired counts; it must not be fabricated or replayed outside
the budget. The original complete DV3 result remains 123 versus 111 goals and
27 versus 39 collisions, reported separately. Earlier
`v4_cancelled_full_sweep_*` interim files are preserved unchanged.

Both old queues are stopped: exec sessions 38068/10110 exited and PIDs 28312,
27248 and 7328 are absent, verified at UTC 13:16:47. Windows process control
denied termination, so temporary read-only handles on the progress files
stopped workers AFTER journal commits. Final counts are R=1,756 and B=1,591.
This was administrative cancellation, not a policy error. One R-wrapper retry
exited 0xc0000142 before evaluation. The handles and pending elevation launcher
were released/stopped; manual intervention is no longer needed. PPO was not
targeted. See `results/safety_dev/cancelled/stop_confirmed.json`.

The fixed sample is
`results/safety_dev/quick_v5_budget100/sample_manifest_v3.json`; its family
coverage and geometry features sit beside it. Drafts v1/v2 remain preserved;
v3 retains v2's cases and adds fresh policy-only references. The full original
completion map `original_completion_map_20261002T132245Z_116a97b9.csv` marks
3,347 committed and 5,323 unrecorded identities out of 8,670. The smaller
comparison cannot establish performance over that full inventory.

Four development attempts are already complete, on CRP-VS-04 and CRS-CV-04
with their original reset seeds. Priority-only (`dev_priority_v1`) produced
two boundary contacts. Adding policy-feedback preservation
(`dev_policy_feedback_v1`) produced one obstacle and one boundary contact.
Neither probe rescued a goal. They are development evidence, not extra
independent frozen/field cases, and must count toward the same attempt cap.

Completed quick results (24 matched scenarios per mode):

| Mode | Goals | Obstacle | Boundary | Target | All collisions | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Policy only | 18 | 4 | 0 | 2 | 6 | 0 |
| V4 | 19 | 2 | 1 | 2 | 5 | 0 |
| Experimental v5 | 19 | 1 | 1 | 3 | 5 | 0 |

V5 gains/loses zero goals against v4 and 3/2 against policy alone. Its only
v4 outcome change is obstacle-to-target contact in FV-HO-12, not a collision
rescue. It changes actions on 121 steps versus v4's 152. Priority prevents
39 continuation-bonus overrides across nine episodes; the broader
verified-policy-preserved counter is 132 steps across ten episodes. The
policy-feedback branch checks 42 times across 15 episodes and accepts six
times across three episodes. Acceptance is not necessarily an avoided
intervention, and neither decision count proves a collision rescue.

The single-thread serial comparison used 408.86 seconds of summed episode
time (about 6.8 minutes; not end-to-end wall time). Mean episode times were
2.53 s off, 7.13 s v4 and 7.37 s v5. **All 99 focused tests passed**: 81 filter
and 18 runner tests, including the 14 policy-feedback regressions. There are
24 matched scenarios per comparison, not 72 independent pairs. The four
additional development episodes repeat two cases and do not enlarge the
comparison sample. No further episodes are needed to complete this task.

See the [quick report](../results/safety_dev/quick_v5_budget100/runs/policy_feedback_v1/report.md),
`complete_24cccc8f.json`, and the verified 79-file `evaluated_sources.zip` with
`evaluated_sources_manifest.json` in that run directory. The revised quick
campaign is separate from the original 8,670-record inventory, which remains
3,347 committed and 5,323 unrecorded. V4 remains the reference; reduced action
changes alone do not establish that v5 is safer.

Next development-only hypothesis: brake-response model error at onset, not
continued-braking age. The [brake-pulse audit](../results/safety_dev/development_brake_pulse_audit.json),
with source SHA-256 for `v4_selected_contacts_steps.csv`, finds six isolated negative-RPM pulses in the two development losses: CRP-VS-04
steps 5/8/17 and CRS-CV-04 steps 11/39/42. Carrying delay age across consecutive
brake decisions cannot fix an onset mismatch in these isolated commands.
A future investigation should calibrate predictions offline from other DV3
measured-state/issued-command logs, using true ego only to score error, then
validate on separate development traces. Do not use held-out outcomes to fit
it or add truth to production. The earlier dual-brake envelope achieved 118
versus selected v4's 123 goals, so correcting the model must not be assumed
to improve the controller. No such calibration or extra evaluation has run.

## Previous continuation: v5 experiments and expanded evaluation (cancelled)

**Start here.** See [SAFETY_LAYER_V5_PLAN.md](SAFETY_LAYER_V5_PLAN.md) and
[SAFETY_LAYER_REFERENCES.bib](SAFETY_LAYER_REFERENCES.bib). Every implemented
method needs a citation, and the complete
**development, frozen and field** evaluation is authorized. No tuning on held-out outcomes.

V4 remains the recommended reference. New v5 methods are independently
switchable: one-decision commitment (111/150 goals), constraint-slack recovery
(121/150), broad feedback backups (121/150), and restricted feedback backups
(123/150). The restricted preset preserves every v4 goal; only CRS-CV-04 changes
from boundary to obstacle contact. Thus no net improvement over v4 is claimed.

The experimental v5 preset is now **frozen**: `ONE_DECISION_COMMIT=False`,
`SOFT_RECOVERY=False`, `FEEDBACK_BACKUPS=True`,
`FEEDBACK_ONLY_INFEASIBLE=True`. Source switches live in `src/safety_v5.py`;
select version 5 at runtime with safety enabled. `constants.py` is unchanged.
All methods and inherited v4 architecture have precise citations in the plan.

Expanded inventory: 370 development, 1,735 frozen (including 35 realizable Tier A
cases), and 785 field cases = **2,890 per mode / 8,670 for off,v4,v5**. Outputs
are under `results/safety_dev/suites/`. The first frozen-R benchmark has started
in `v5_frozen_r`; its manifest froze code/settings before its first episode.
`results/safety_dev/v5_benchmark_freeze.json` records the selection and hashes.
Frozen headline B is also running in `v5_frozen_b`; its serial shell queue
then starts `v5_field_deployment`. After frozen R completes, the other lane
will run `v5_dev_fv_a` (`--suites dev,field_validation,frozen_a`). Inspect each
`progress.json` and live lock for current state; do not start duplicate workers.

`tools/diagnostics/safety/suite_eval.py` runs one serial worker / one Torch
thread and records scenario/reset-seed digests, source/checkpoint hashes and a
durable per-episode journal. Resume only with the identical command plus
`--resume`; never overwrite outputs. An active `running.lock` prevents duplicate
writers. **At most two workers**, including any heavy scenario-build process;
leave ongoing PPO training alone. The all-suite inventory builder (`inventory_v5`, build-only) completed all
2,890 cases. All 1,700 frozen B/R and 630 deployment-field digests match the
historical manifests. The three unrealized Tier A definitions are A-BO-N,
A-NU-I and A-NU-N, explicitly recorded. The inventory ran no policy episodes;
its old metadata need not be resumed after the source freeze. Caches are ready.

First component command (project root):

```powershell
python -B tools/diagnostics/safety/suite_eval.py --suites frozen_r --modes off,v4,v5 --tag v5_frozen_r --reference-frozen results/frozen_suite/sacs0_bl3/manifest.json
```

Run remaining components with separate `v5_<component>` tags, maximum two
active processes; add the matching historical reference for frozen B/R and
field deployment. `suite_compare.py RUN_DIR [RUN_DIR ...] --tag TAG` creates
component/group totals and paired gains/losses against off and v4; incomplete
runs require explicit `--allow-partial`. No candidate code/settings changes
are allowed after inspecting these held-out outcomes.

Validation: 139 safety tests passed, excluding the one known original v3
failure described below. After freezing defaults, 45 version/dispatch tests
passed again. The five soft regressions were reproduced with
`recovery_tradeoffs.py`: 159 decisions, only one avoidable immediate violation
(in the timeout), so that hypothesis does not explain the four new collisions.

### Current benchmark recovery and monitoring

Use `tools/diagnostics/safety/suite_status.py` with the four `v5_*` run
folders for authoritative journal counts. It uses Windows read/write/delete
sharing and ignores an unfinished final line. The build-only inventory is not
part of the 8,670-episode comparison.

Two transient `progress.json` write denials stopped processes; identical
manifest-guarded resumes continued them. A separate interior JSONL gap was
found for `off / frozen_b / B-06-040`, although its CSV result is present. Its
original CSV row (goal) is preserved in
`v5_frozen_b/reconciliation_missing_off_B-06-040.json`. Re-run missing cases
through the unchanged evaluator's `--resume`; never invent journal records
from the CSV. Final reports require full, unique manifest coverage.

`run_frozen_queue.py --lane A` serially resumes R then dev/FV/A;
`--lane B` serially resumes B then deployment. It retries only the specific
progress-file PermissionError, stops on policy/validation errors, and the
latest version independently checks journal completeness even after exit 0.
Start a lane only after its existing worker has exited. Its maximum is five
attempts per component. It never changes controller/evaluator code or settings.

The currently running lane A launcher was imported before the new completion
guard. Its original loaded code applies to both queued components. Read
`results/safety_dev/queue_session38068_source_provenance.json` for exact launcher
hashes and the later log-header caveat; all 77 frozen evaluator/controller source
hashes still match. Lane B currently runs the original serial shell queue.
Therefore independently reconcile **all four journals after both lanes finish**,
even if a launcher reports successful exit. The latest Python wrapper can do
this using unchanged commands; complete components need no policy reruns.

## Previous continuation: selected v4 (2026-10-02)

**Start here.** The original brief below is retained as history. Current code,
evidence and reproduction details are in [SAFETY_LAYER_V4_PLAN.md](SAFETY_LAYER_V4_PLAN.md).
The working checkpoint remains the unchanged SAC baseline 3 at 3M timesteps.

The four suggested diagnostic steps have been completed. All 14 original v3
wall cases were reproduced and classified; their final decisions were all
`no escape`. Full true-state/geometry oracle rescued 11, while true state with
remembered scans retained rescued eight. Actual-command replay identified
state-estimation lag and large heading errors during braking. Old scan memory
also caused failures. Margins and thresholds were left unchanged.

The selected `src/safety_v4.py` enables a model-based ego observer plus
free-space clearing of remembered LiDAR points. It uses onboard estimates and
issued commands only. Countersteering, nominal-plan retention and dual braking
remain available as **disabled ablations**. The observer alone achieved 110
goals; adding dual braking to the selected combination lost five successes
without rescuing any. Choose the measured combination, not individual fixes.

| Matched development set, 150 episodes | Goal | Obstacle | Boundary | Target | Timeout |
|---|---:|---:|---:|---:|---:|
| Policy alone | 111 | 26 | 2 | 11 | 0 |
| Best v2 iteration 2 | 115 | 17 | 2 | 13 | 3 |
| Previous v3 default | 108 | 18 | 14 | 10 | 0 |
| **Selected v4** | **123** | **9** | **7** | **11** | **0** |

V4 gains/loses 19/7 goals against policy alone, 16/8 against best v2 and 24/9
against v3. It rescues 12 of v3's 14 original wall cases but introduces other
wall contacts. This is a development improvement, not a collision-free or
held-out result. A final nine-case replay reproduces the selected defaults:
six remaining wall contacts end in `no escape`, one in `last certificate`.

New production files: `src/safety_v4.py`, `src/safety_observer.py`,
`src/safety_perception.py`, `src/safety_prediction.py`. The last is an optional
braking ablation. `src/env.py` adds v4 dispatch and a pre-physics callback with
the actual issued rudder and signed RPM. V2, v3 and shared prediction code are
unchanged. Set `cfg.SAFETY_VERSION = 4` at runtime with safety enabled; do not
edit `constants.py`. Set module switches before constructing/resetting a filter.

From the project root, with a fresh result tag:

```powershell
Remove-Item Env:V2_SET,Env:V3_SET,Env:V4_SET -ErrorAction SilentlyContinue
python -B tools/diagnostics/safety/paired_eval.py v4 new_v4_run
python -B tools/diagnostics/safety/compare.py results/safety_dev/dev_new_v4_run.csv --mode v4
```

Use `prediction_audit.py` for executed-command errors, oracle comparisons and
last-contact traces; `observer_replay.py` and `integration_probe.py` isolate
observer and numerical effects. The old `dev_eval.py` now rejects v4 explicitly
instead of silently selecting the wrong filter. The held-out runner CLIs accept
version 4, but those suites were not used for tuning.

Authoritative result: `results/safety_dev/dev_v4_observer_memory.csv` and its
JSON provenance, `_paired.csv` and `_comparison.csv`. All full experiments are
collected in `final_fullset_summary.csv` and `final_fullset_comparison.csv`.
Remaining-wall traces: `v4_selected_contacts_*`; compact review:
`selected_contact_review.json`. Cache, checkpoint, scenario digests and seeds
were checked across runs; all new full variants use identical paired scenarios.

Validation: 65 new tests and four existing tests pass. The existing v3
`test_filter_steers_off_an_obstacle_dead_ahead_and_leaves_open_water_alone`
still fails; the identical failure was verified with the original HEAD
`env.py` in memory. It was not hidden by changing the test or old filters.

Next investigation: first action divergence in the newly introduced wall
cases, particularly CRS-CV-11's last-certificate continuation. Preserve the
123-goal reference and compare rescues and regressions episode by episode.
All original constraints below still apply, including the active PPO run,
Paper 2 read-only, no constants edits, no held-out tuning and no commits.

## Archived initial brief

The following sections describe the state before the v4 continuation above.

## The problem in one paragraph

A SAC policy (and soon a PPO policy) is trained **without** any safety mechanism. At deployment a
runtime **predictive safety filter** wraps it. Every 0.5 s the filter predicts the next 8 s with the
identified ship model, and asks whether the policy's action still leaves an escape manoeuvre clear of:
- static LiDAR returns,
- the map boundary,
- the tracked target, assumed to hold constant velocity.

If no escape remains, it substitutes the nearest action that has one. On the 150-episode development
set it cuts obstacle contacts, but trades them for **wall contacts** and stalls. **No version clearly
beats the policy alone on goals.**

## Key files

| What | File |
|---|---|
| Filter v2 (predictive filter, braking model, run-out check, ego smoothing) | `src/safety_v2.py` |
| Filter v3 (v2 plus a committed backup plan and a recovery mode; current work) | `src/safety_v3.py` |
| Shared prediction helpers (Perception snapshot, Actuators, rollout, clearances, LOS) | `src/classical/common.py` |
| Where the filter is called (top of `step()`; `SAFETY_VERSION` 2 or 3 selects it; braking via `_v2_brake`) | `src/env.py` |
| Tests | `tests/test_safety_v2.py`, `tests/test_safety_v3.py` |
| v2 plan, iteration log, reverse-thrust caveat | `planning/SAFETY_LAYER_V2_PLAN.md` |
| v3 plan, literature basis, evaluation matrix, development log | `planning/SAFETY_LAYER_V3_PLAN.md` |
| Development-set evaluation (150 episodes, kept 3 M SAC policy) | `tools/diagnostics/safety/dev_eval.py` |
| Step-by-step trace of one dev case | `tools/diagnostics/safety/trace_case.py` |
| Trace of policy holding straight at the L1 obstacle | `tools/diagnostics/safety/trace_dead_ahead.py` |
| Past dev results, per episode | `results/safety_v2_dev_<tag>.csv` |
| Kept policy used for development | `runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip` |

Running the diagnostics:
- `python tools/diagnostics/safety/dev_eval.py off,v2,v3 <tag>` writes `results/safety_v2_dev_<tag>.csv`.
- To override constants for one run, set `V2_SET="NAME=value,..."` (for `safety_v2`) or
  `V3_SET="NAME=value"` (for `safety_v3`).
- `V2_CASES="DV3-...,DV3-..."` runs a subset of cases.
- `python tools/diagnostics/safety/trace_case.py DV3-OT-CV-16` prints the last 45 steps of one case:
  position, heading, speed, policy action, the filter's mode and its reason ("why"), and margins.

Evaluation tools:
- `tools/tiers/frozen_suite.py` and `tools/tiers/paper2_suite.py` take `--safety both --safety-version {1,2,3}`.
- These are the held-out test sets. **Do not tune on them.**

## Current results (dev set, 150 episodes, kept SAC policy)

| Version | Goals | Obstacle | Boundary | Target | Timeout | Interventions / episode |
|---|---|---|---|---|---|---|
| Safety off | 111 | 26 | 2 | 11 | 0 | – |
| v2 iteration 2 (best so far) | **115** | 17 | 2 | 13 | 3 | 6.0 |
| v2 current default (iteration 6, no boundary run-out) | 109 | 18 | 14 | 9 | 0 | 11.4 |
| v3 current default | 108 | 18 | 14 | 10 | 0 | 13.2 |

The full logs are in the two plan files.

## What is known

1. **Fixed.** The dead-ahead cliff. Holding straight at an obstacle, the margin dropped from "fine"
   to "no escape" in one step. Two fixes:
   - smoothing the measured yaw rate (its noise, 1°/s, swung the margin by ±0.4 m);
   - a terminal run-out check, because a fixed horizon had rewarded slowing.
2. **Fixed.** Braking traps. With no astern motion, a hull braked to a stop beside a wall cannot get
   out. Braking is now allowed only against traffic.
3. **Fixed (v3).** Stale stored plans. Plans now expire at the end of their checked horizon, and are
   followed unchecked for at most 2 s.
4. **Fixed (v3).** Rejoining the path was the wrong recovery reference, because the L1–L3 paths run
   through the panels. The recovery now stays nearest to the policy's action.
5. **OPEN — wall contacts (14 against 2).** They appeared with the iteration-3 changes and have not
   gone away. Already ruled out:
   - ego smoothing;
   - the boundary run-out;
   - braking rules;
   - turn-over-slowing preference;
   - room slack;
   - a longer horizon (16 s made it worse: 98 and 82 goals).

   In traced cases (DV3-OT-CV-16, DV3-BO-CV-04), after interventions the hull points at a wall.
   Every candidate then fails, the filter has no escape, and the policy acts. The policy *nearly*
   gets out of states the filter calls hopeless. **Working hypothesis: the prediction is
   pessimistic.** Candidate causes:
   - inflated hull margin and gaps;
   - identified model against simulator dynamics;
   - remembered LiDAR points;
   - estimated against true pose.
6. **Structural limits.**
   - The basin is 10 m wide; the hull's steady turning circle is about 10.5 m at any rpm, and there
     is no astern motion. No invariant terminal set exists, so the filter cannot be called certified.
   - Reverse thrust is not identified; the filter uses a pessimistic model.
   - The policy never trained with overrides, so heavy intervention puts it in unfamiliar states.

## Suggested next steps

1. **Measure prediction error.** Log the rollout the filter predicts for the executed action, and
   compare it with the realised trajectory over 1–8 s, on dev episodes. Report position and heading
   error quantiles. If the predicted clearance is systematically lower than the realised one, the
   filter is pessimistic: set the gaps and margins from these quantiles.
2. **Oracle mode.** Give the filter the true pose, the true static geometry and the true target
   state. If the wall contacts disappear, perception or estimation is the cause; if they stay, the
   cause is the logic or the model.
3. **Classify every boundary contact.** Using the "why" field, record the mode in the last 10 steps
   before each contact: no escape, last certificate, recovery, or nominal.
4. Only then change thresholds. Compare each change paired against safety off and v2 iteration 2
   (gained and lost episodes), not on totals alone.

## Where to write new files

| What | Where |
|---|---|
| Code | `src/` (e.g. `src/safety_v3.py`, or a new `src/safety_v4.py` selected by `SAFETY_VERSION = 4` in the `env.step` hook) |
| Diagnostics and scripts | `tools/diagnostics/safety/` |
| Results (CSV, figures) | `results/safety_dev/` (new folder; older results stay in `results/` as `safety_v2_dev_*.csv`) |
| Notes and decisions | append to `planning/SAFETY_LAYER_V3_PLAN.md`, or a new `planning/SAFETY_LAYER_V4_PLAN.md` |
| Tests | `tests/test_safety_*.py` |

## Constraints

- Never write to the Paper 2 folder `asv-lidar/static_obstacles/` (read-only).
- Do not edit `src/constants.py`: the baseline digests are read from it. Put switches in module
  constants or set them at run time.
- Never tune on the frozen suite or the Paper 2 set. Use the dev set (`formulation_v3.field_development_set`).
- Preserve line endings. Most planning markdown is LF; `PROJECT_STATE.md`, `CLAIM_LEDGER.md` and
  `RESULT_TABLES.md` are CRLF.
- **A PPO baseline-v3 run is training** (`runs/ppo_formulation_seed0_bl3`, started 2 Oct 12:24 via
  `results/ppo_bl3_run.sh`, detached). Heavy parallel evaluations slow it by about 30 %. Use at most
  1–2 evaluation processes, and do not stop it.
- Commit only when asked.
