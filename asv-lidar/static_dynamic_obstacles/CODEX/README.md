# CODEX navigation experiment

This folder is a self-contained working copy of the ASV simulator, controller,
tests and training entry point. All implementation changes and retained outputs
belong here. The parent implementation and its running training jobs are not
used as output locations. `BASELINE_MANIFEST.json` records the copied source
hashes and original Git revision, including the fact that the source was dirty.

## Run

Use the existing Python 3.10 environment, from this folder:

```powershell
python -B -m pytest -q -p no:cacheprovider
python -B tools/validate_behaviors.py --noise both --output results/behaviors.json
python -B src/train_formulation.py --smoke --num-envs 1 --device cpu
```

The behavior command runs the reference controller through empty-path,
recovery, static-obstacle, head-on, both crossing directions, overtaking,
being-overtaken, nonconflicting-target and combined scenes. It writes measured
trajectories and individual pass/fail checks, and exits unsuccessfully if any
contract fails. These are development scenarios, not held-out performance data.

The smoke command performs 1,024 PPO training steps, saves and reloads the
model, and checks paired evaluation with the emergency supervisor off/on.
Its evaluation episodes are capped at 32 steps. Passing this check establishes
that training works mechanically; it does not establish learned navigation.

Start a new training experiment with:

```powershell
python -B src/train_formulation.py --timesteps 2000000 --num-envs 1 --device cpu --tag experiment01
```

The requested training budget is rounded to complete PPO rollouts and does not
guarantee convergence. Outputs are
restricted to `CODEX/runs/`; an existing run directory is never overwritten.
Increase worker count only when sufficient compute is available. Full training
evaluates complete episodes and can take substantially longer than the smoke.

## What changed

| Area | Behavior |
| --- | --- |
| Perception | One pose and ego estimate feeds the boundary scan, tracking, path errors, CPA and observation per decision. Consecutive stale frames retain the last received estimate. Repeated observation reads are pure. |
| Observation | A sixth branch exposes the encounter latch, allowed action, maneuver progress and previous executed action. Default size increases from 56 to 70. |
| Encounter release | Renewed risk restores the latched obligation; release requires sustained opening and adequate actual estimated separation. |
| Reward | Slowing receives Rule 8 credit only when the slowdown escape is predicted to clear. Maneuver deltas use the same perceived state as the policy. Physical collision and clearance remain geometric truth. |
| Goal | Along-path completion requires cross-track error at most 0.60 m, replacing the former 1.60 m tolerance that ended avoidance recovery too early. The 0.50 m goal-point radius is unchanged. |
| Scene coverage | Final sampling weights are 20% empty, 25% static only, 40% dynamic only and 15% combined. Recovery starts train recentering. |
| Curriculum | Empty-path mastery precedes static avoidance, then head-on/null targets, then all target classes and combined scenes. Previously learned scenes remain in the mixture. Two consecutive measured passes are required to advance. |
| Evaluation | Exact channel, obstacles, seeds and recovery state are saved for replay. Supervisor off/on runs use the same cases. Boundary, static and target collisions have separate metrics. |
| Selection | Checkpoints are ranked by collisions, inadequate clearance, weakest scene completion, trajectory turn diagnostic, COLREG reward proxy, completion and tracking. Reward alone does not select a model. |
| Reference | A candidate-trajectory LOS controller provides a functioning comparator using perceived pose, mapped boundary/path, gated LiDAR and tracks. It does not access obstacle or target truth for control. |
| Stand-on behavior | During an active being-overtaken encounter, the reference preserves its path and cruise command when a feasible prediction exists. It exposes an override when holding is unsafe. Acceptance checks actual speed as well as path keeping. |

Old 56-value policies are incompatible and need retraining. The new trainer
checks the saved observation schema before loading a checkpoint. The inherited
alternative training scripts are retained for import compatibility; use
`src/train_formulation.py` for the revised curriculum.

## COLREG scope

The inherited encounter model assumes a small own ship giving way to a
channel-constrained target in both crossing directions. Its preferred port
alteration for a target approaching from port is a research convention under
that assumption. It is **not** the ordinary open-water Rule 15/17 role table.
For ordinary power-driven crossing encounters, the vessel with the other on
starboard gives way; a stand-on vessel taking discretionary action under Rule
17 has additional restrictions. Rule 9 depends on channel and vessel
circumstances. See the international provisions in the
[USCG navigation rules](https://www.navcen.uscg.gov/navigation-rules-amalgamated).

The behavior tests check concrete trajectory properties such as starboard
head-on alteration, passing astern, and completing a separated overtaking
pass. Reward violation fractions and these scenario contracts are not proof of
general legal compliance. Ordinary open-water roles, Rule 19 visibility,
signaling, multiple targets and field deployment are outside this validation.

Normal training excludes failed scripted crossing-escape cases and cases
already below the configured CPA floor. This is a deliberately easier learning
distribution; retained scenes are not thereby proved feasible. Separate stress
evaluation must report the excluded distribution.

See `OBSERVATION_SPEC.md`, `results/BEHAVIOR_PROTOCOL.md` and
`VERIFICATION.md` for the exact interface and measured validation status.

The recorded trajectories are in `results/behaviors.png`; add `--plot` to
the behavior command to regenerate that overview. The reference controller
is an offline comparator for straight, convex channels. Its timing has not
been qualified for the vessel's 2 Hz control loop.

`requirements-tested.txt` records the existing package versions used for the
checks. `python -B tools/run_tests.py` retains the full regression log and JUnit
report under `results/tests/`. `python -B tools/check_source_isolation.py`
compares the original sources with the creation snapshot.
