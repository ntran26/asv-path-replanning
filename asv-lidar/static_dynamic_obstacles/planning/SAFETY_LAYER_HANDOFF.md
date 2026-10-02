# Safety layer — debugging handoff (2026-10-02)

This is a self-contained brief for continuing the safety-layer work in a new session.

Project root: `asv-lidar/static_dynamic_obstacles`. Run every command from that folder.

## Latest user direction: stop the full sweep; quick v5 test (2026-10-02)

The user cancelled the full 8,670-run evaluation, requested a v4 report, and
explicitly capped subsequent work at **100 new episode runs total**, then
emphasized a quick test. **Do not resume the old full queues.** The current
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
[SAFETY_LAYER_REFERENCES.bib](SAFETY_LAYER_REFERENCES.bib). The user requested
citations for implemented methods and explicitly authorized the complete
**development, frozen and field** evaluation. No tuning on held-out outcomes.

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

## Files to point the new chat to

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
