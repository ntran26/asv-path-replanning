# Behaviour development contracts

`python tools/validate_behaviors.py --noise both --plot` runs a single process
over ten fixed geometries, with and without nominal perception noise. The
result is `results/behaviors.json` and a trajectory overview. Any failing case
makes the command exit with status 1. The command does not select or omit cases
based on success.
In the overview, dashed black lines are reference paths, solid lines are the
own-ship trajectories, dotted red lines are target trajectories, and grey
polygons are static panels.

The controller is a **scripted reference comparator**, not PPO. Passing these
contracts demonstrates feasible end-to-end behaviours using the simulator,
perception stack, and this reference controller. It does not demonstrate that
a trained neural policy learned them, generalisation to randomly generated
encounters, robustness to the plant bootstrap, or real-vessel safety.

## Inputs and prediction

The reference reads cached estimated pose and ego velocity, known path and
corridor polygon, gated LiDAR ranges, and estimated dynamic tracks. It never
reads target truth, obstacle polygons, true own-ship state, or the live plant
state when selecting actions. Unit tests expose only these onboard inputs and
reject truth access. The candidate predictor uses the identified model with
0.05 s integration, estimated initial state, and command-derived actuator delay;
acceptance executes commands in the actual simulator plant. The predictor never
copies the simulator's hidden plant state. It is intended for straight, convex
channels. A short scan memory retains surfaces through the LiDAR dead zone.

Candidate manoeuvres include an offset followed by return after 6 or 10 s,
plus a held offset. The predictor retains its own command FIFO and estimated
servo state across actions. It checks oriented inflated rectangles and requires
0.4 m predicted target hull separation and 0.10 m predicted boundary separation;
the 2.6 m centre-distance preference is soft. Latched directional obligations
constrain the candidates, retaining neutral/coasting responses. If no permitted
candidate is feasible, the comparator reports a fallback; directional relaxation
is allowed only when the encounter context explicitly marks an emergency and a
physically feasible alternative exists. These events are counted in the report.
For an engaged or clearing being-overtaken encounter, a feasible path-holding
cruise candidate takes precedence over soft costs. If no such candidate is
feasible, avoidance remains available and `stand_on_override` records the
exception. The safe being-overtaken fixture requires no such exception.
Overtaking candidates can use the existing propulsion authority up to 1.5 times
cruise RPM. Other cases use at most cruise RPM. Model-based prediction and these
development contracts do not establish a formal safety guarantee.
The comparator is intended for offline validation; its runtime has not been
qualified for the vessel's 2 Hz control deadline.

The harness alone accesses simulation truth to compute trajectory-based
checks and exact hull-to-hull clearance at decision instants. The environment
also checks collisions on physics substeps. Minimum recorded clearance can
therefore overestimate the continuous-time minimum; no-collision comes from
the finer physics checks.

## Fixed scene contracts

All scenes require reaching the existing environment goal, no boundary/static/
target collision, final cross-track error below 0.65 m, at least 0.10 m
static/target hull clearance and 0.05 m boundary clearance. Additional checks:

- Empty, null, and being overtaken: maximum cross-track error below 0.65 m.
- Being overtaken: during perceived engaged/clearing intervals, actual speed
  remains within 5% of cruise, every propulsion command requests cruise, and
  no stand-on override occurs. At least one active interval must be observed.
- Recovery: begin 1.5 m off the reference and regain it.
- Static: a panel directly blocks the reference, requiring a genuine detour.
- Head-on: first committed offset to starboard and actual starboard passage.
- Crossing from either side: the own ship crosses the target's track at least
  0.75 m astern of the target. Slowing without a large turn is allowed.
- Overtaking: actually pass the slower vessel on its port side, rather than
  merely follow it to the goal.
- Combined: an on-path panel followed by a reciprocal vessel in the same run.

The passing-direction rules intentionally match the existing **restricted
narrow-channel research assumptions**. In particular the own ship gives way
in crossings from either side. This is not a general implementation or legal
certification of the international COLREGs. The 0.75 m side/astern tests verify
passing geometry, not a universal safe distance. The report separately records
target centre distance, exact hull clearances, and boundary clearance; a
collision-free trajectory need not satisfy all configured ship-domain margins.

Both modes disable the emergency supervisor and plant randomisation. Nominal
mode enables the configured pose and ego measurement noise; other degradation
settings retain their configured nominal values. Seed 812 is fixed by default;
use `--seed` for further trials. The final propulsion stage permits zero-thrust
coasting, with no reverse command available to the controller.

## Recorded validation

The final `behaviors.json` contains 20 passing fixed-scene runs (ten scenes in
each noise mode), and `behaviors_seed913_final.json` contains six passing
independent nominal-noise repeats. The main suite recorded minimum static,
target, and boundary hull clearances of 0.263, 0.523, and 1.648 m respectively,
with maximum final cross-track error 0.304 m. The independent suite includes
the previously failing starboard crossing, now with 0.490 m target clearance.
Both reports record zero directional relaxations and no emergency supervisor.
They do contain candidate-feasibility fallbacks: 101 of 1,361 main-suite steps
and 67 of 426 independent steps. These are successful empirical trials, not a
proof that a feasible predicted manoeuvre always exists.

`behavior_additional_checks.json` measures speed during the being-overtaken
scene and passing side in the combined scene from the saved main trajectories.
The combined runs pass to starboard by 2.391 m without noise and 2.420 m with
nominal noise. During the perceived active being-overtaken interval, maximum
actual speed departure from cruise is effectively zero without noise and
0.176% with nominal noise (mean 0.067%). The independent nominal repeat has
maximum departure 0.096%. Every active-encounter command requests cruise and
all three trials have zero stand-on overrides. Speed may change after the
encounter has cleared; the additional report separately records whole-episode
speed measurements. Reproduce these posthoc measurements with
`python tools/report_behavior_checks.py`; they do not count as additional
independent trials.

The earlier successful path-only reports are preserved as
`behaviors_before_speed_fix.json` and `behaviors_seed913_before_speed_fix.json`.
The stronger stand-on checks exposed discretionary slowing in the earlier
controller: the archived main nominal trajectory had 7.20% maximum active
speed error and only 86.7% cruise commands. The final controller gives a
physically feasible stand-on hold precedence over soft trajectory costs.
Both final reports fingerprint the source files and acceptance harness; these
fingerprints matched the files when validation completed.
