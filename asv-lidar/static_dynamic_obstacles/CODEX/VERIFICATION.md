# Verification record — 2026-09-17

All implementation changes and new run artifacts are inside `CODEX`. The
original experiment was already modified before this work and its PPO job was
left running. `tools/check_source_isolation.py` verified all 98 source files
against the creation manifest with no differences.

## Regression tests

The complete suite is run by `python -B tools/run_tests.py`. It retains the
command, exit code, elapsed time, log and JUnit XML under `results/tests/`.
Full-suite result: **460 passed**, one existing TensorBoard/NumPy deprecation
warning, in 1,601.89 seconds; process exit code 0. This collection preceded
the final goal-tolerance and reference-planner refinements, which have separate
follow-up checks below. The run used one process at below-normal priority while
the original training job continued.

After the final source freeze, the focused integration suite passed **36 tests**
in 141.65 seconds (one existing TensorBoard/NumPy warning). Its log and JUnit
report are `results/tests/final_integration.log` and
`results/tests/final_integration.xml`. This run covers the final goal guard,
perception/metrics, controller information boundary, actuator history, oriented
hull separation, curriculum, replay and checkpoint compatibility.

Focused checks completed during implementation:

- Nine perception/metric/goal regressions pass, including shared estimated pose,
  pure observation reads, configurable slot count, reproducible recovery
  starts, invalid recovery rejection, collision-type attribution and path
  recovery before goal completion. After tightening the goal tolerance, all
  57 environment/perception tests passed (224.61 seconds).
- Observation/context/reward integration retest: 48 passed. New tests cover
  recurring risk while clearing, safe separation before release, ineffective
  slowdown, perceived maneuver deltas, truth diagnostics and context masking.
- Eleven training regressions pass, including explicit realized strata,
  mastery gates, checkpoint compatibility, output isolation and independent
  physical heading diagnostics.
- Two independent actuator-history tests pass through repeated rudder
  reversals, with and without command limiting, against `ShipModel`, including
  the servo angle and the full delay buffer.
- Seven oriented-hull geometry checks pass, including crossing and parallel
  clearances, contact, overlap, rotations, broadcasting and hull inflation.
- Two stand-on regressions cover maintaining cruise when holding is predicted
  safe and retaining avoidance when it is unsafe. The earlier nominal-noise
  trajectory fails the strengthened active-encounter checks: 7.20% maximum
  speed error and 86.7% cruise commands. The corrected focused run has 0.176%
  maximum speed error, 100% cruise commands and zero stand-on overrides.
  Both noise modes pass; evidence is in `results/behaviors_standon_speed.json`
  and `results/behavior_additional_checks_before_speed_fix_active.json`.

These focused counts overlap the full suite and must not be added to its total.
The actuator and oriented-hull geometry files and the two stand-on regressions
were added after full-suite collection and are covered by the final integration
run. The retained full-suite JUnit report includes the earlier behavior tests.

## PPO execution and replay

| Run | Result |
| --- | --- |
| `runs/ppo_codex_seed0_integration_smoke` | 1,024 training steps completed; model and reward normalizer saved; model reloaded; eight paired, capped evaluation episodes completed. |
| `runs/ppo_codex_seed0_resume_check_smoke` | 128 steps resumed from that CODEX model and normalizer; save/reload and eight paired, capped evaluations completed. |

Each smoke evaluation covers four scene strata with the emergency supervisor
off and on, capped at 32 decision steps per episode. These runs validate
execution and persistence. They do **not** demonstrate a trained PPO policy
that can complete navigation or comply with the encounter rules.

An independent check generated and replayed 40 development cases across all
five curriculum levels. Geometry hashes, JSON round trips and initial poses
matched. Supervisor off/on resets gave byte-identical observations. Every
dynamic encounter class was covered in dynamic and combined scenes at levels
4 and 5. Evaluation observation normalization matches training (`norm_obs=False`).

## Reference-controller navigation

The main suite passed **20/20 cases** in 853.15 seconds: ten fixed scenes with
perception noise off and at nominal settings. Every case reached the goal
without collision or supervisor intervention. Maximum final cross-track error
was 0.304 m; minimum sampled static, target and boundary hull clearances were
0.263, 0.523 and 1.648 m respectively. The planner reported 101 prediction
fallbacks in 1,361 steps and zero directional-rule relaxations.

The complete report is `results/behaviors.json`, with trajectories in
`results/behaviors.png`. Its recorded source and harness hashes match the
final files. The independent nominal-noise seed 913 run also passed **6/6 cases**
in 331.24 seconds, with matching source and harness hashes:
`results/behaviors_seed913_final.json`.

| Seed 913 case | Final path error (m) | Minimum static hull clearance (m) | Minimum target hull clearance (m) |
| --- | ---: | ---: | ---: |
| Static | 0.186 | 0.168 | — |
| Head-on | 0.156 | — | 0.990 |
| Crossing from starboard | 0.135 | — | 0.490 |
| Crossing from port | 0.131 | — | 0.722 |
| Combined | 0.178 | 0.350 | 1.394 |
| Being overtaken | 0.077 | — | 1.862 |

All six reached the goal without collision, supervisor intervention or
directional-rule relaxation. On 67 of 426 decision steps the planner used its
permitted-candidate fallback because no candidate met every forecast margin.
Actual measured clearance contracts still passed. This matters: the finite
candidate planner is a tested comparator, not a guaranteed safety controller,
and its offline validation does not demonstrate meeting a 2 Hz real-time deadline.

The added seed 913 being-overtaken case maintained cruise commands throughout
all 33 active encounter steps. Its maximum measured speed error was 0.0962%,
with no stand-on override or prediction fallback.

See `results/BEHAVIOR_PROTOCOL.md` for fixed
scene definitions and the independent trajectory contracts. Intermediate
`behaviors_*debug*.json` and predictor artifacts preserve failed development
attempts; they are not the final result.

Both seeds used for the final scenario runs were also used during debugging.
They are development regressions, not a held-out test of generalization. The
second run independently repeats the checks against the frozen source.

The controller is a deterministic predictive LOS comparator, separate from
PPO. It uses the nominal identified dynamics for planning but never reads the
simulator's true obstacle/target state or live plant state to select actions.
The evaluator alone uses truth for hull clearances and trajectory checks.

## Limits of the evidence

This work retains the project's declared narrow-channel encounter assumptions,
including giving way in both crossing directions. It does not implement the
general open-water Rule 15/17 role table. Passing development scenes is not
legal certification, a held-out generalization result, or a field safety case.

Full PPO retraining and subsequent independent evaluation remain necessary for
a learned-policy performance claim. Existing 56-value checkpoints cannot be
used with the revised 70-value observation schema. Excluded difficult generator
cases, randomized plant parameters, multiple ships and degraded perception
beyond the nominal test mode need separate evaluation.

The original copied source hashes are in `BASELINE_MANIFEST.json`; each new
training run saves its own source provenance and observation schema. The
package versions used are listed in `requirements-tested.txt`. No package
installation was needed for these checks.
