# CONSTANTS AND SCALES — Paper 3

**Revision 2.7** — F24 decided: **`CRUISE_RPM = 6`, `U_REF` = 0.558 m/s**. The
virtual corridor sweep is **restored**. Nominal pose and ego noise and hull
randomisation (1.0) are on for training, and the emergency stop has its own
reward treatment (`R_ESTOP`). New values in §14; tables quoting 1.116 m/s
below are updated where they define a current value.

**Revision 2.6** — **everything runs at 2 Hz** to match the vessel
(`UPDATE_RATE = 0.5`). Every step count is a duration through `steps_for`,
every dense reward weight is scaled by `REWARD_DT_SCALE = 5` so per-second
rates and 02a §8.1's episode integrals are unchanged, and the training
discounts are converted by `discount()`. The corridor is **fixed at 10 m**,
which retires the Study 1 sweep (§2). The static/dynamic split is a
**free-space consistency test** (§7.0). Pose staleness is native to the
environment (§13.3). Tables carry the 2 Hz values; prose quoting 10 Hz step
counts describes revision 2.5 and says so.

**Revision 2.5** — the identified hull from 05 part 1 (`bluefin/`) replaces the
calibrated v2 model: `U_REF` is now **derived from the plant** (1.116 m/s) and
`THRUST_CAL` is gone. New constants for the emergency stop and deployment
timing are in §13. Sections below that quote 1.14 m/s or `THRUST_CAL` describe
revision 2.4 and are marked where it matters.

**Revision 2.4** — tracks `02b_DECISIONS_AND_TASK_ORDER.md` through **T4**.
The speed calibration is **measured** (T1), the cross-track sign is flipped to
textbook (T2/C4), `r_path` exists (T3), the width thresholds are **computed**
rather than listed (C1), the terminal payoffs are decided, and §12 is the
reward.

Three things moved in T4: **F21 is resolved** — the sensor-resolution floor
binds, `d_abeam = 1.25 m` (§8.1). **F22** — `N_ref` had to be derived from the
measured cruise speed or `R-9` inverts (§12.2). **F23** — `02a §5.2`'s `d_safe`
breaches `02a §2`'s own invariant (§12.3).

Mirror of `src/constants.py`, which is the single source of truth. Every
unresolved value appears there with a `TODO` marker and **nowhere else** — no
consumer buries a magic number in a function body.

| Marker | Owner |
|---|---|
| `TODO(02)` | `planning/02_REWARD_AND_COLREGS.md` |
| `TODO(03)` | `planning/03_ENVIRONMENT_AND_TARGETS.md` |
| `TODO(04)` | `planning/04_SCENARIOS_AND_EVALUATION.md` |
| `TODO(05)` | `planning/05_VESSEL_MODEL_AND_SIM2REAL.md` |
| `TODO(decision)` | needs a call no open item currently covers |

**One `TODO(decision)` remains**, and it is `D_SAFE` (§12.3) — 02a's own
value breaches 02a's own invariant, and the resolution changes where the
boundary penalty begins. Everything else is `TODO(05)` — values that need a
measurement, not a call — plus three `TODO(03)`/`TODO(04)` items owned by later
tasks.

Closed by 02b: the terminal payoffs, `TCPA_CLIP`, `DCPA_CLIP_DOMAINS`,
`BEING_OVERTAKEN_SPEED_MARGIN`, `TARGET_SPEED_RANGE`, `TRACK_GATE_DIST`, the CRI
block, `DOMAIN_RADIUS_DCPA`, `USE_RECURRENCE`, `CLUSTER_*`,
`TARGET_COMPLIANT_SPAWN_PROB`, `NO_TARGET_EPISODE_PROB`.

---

## 1. Vessel reference lengths

| Symbol | Value | Note |
|---|---|---|
| `LOA` | 1.725 m | from `ship.VESSEL_LENGTH`; collision hull and LiDAR mount |
| `LBP` | 1.57 m | **ship-domain and CRI scaling only** |
| `BREADTH` | 0.50 m | **the unit for channel width** |

These are genuinely different numbers and conflating them would silently
mis-scale the ship domain by 10%. `ship.py` has no `LBP`; it is defined here.

## 2. Workspace and the Study 1 width sweep

| Symbol | Value | Status |
|---|---|---|
| `UPDATE_RATE` | **0.5 s** | **2 Hz, the vessel's decision rate** (revision 2.6; was 0.1) |
| `PHYSICS_DT` | 0.1 s | collision and target motion sub-step inside a decision |
| `MAX_EPISODE_STEPS` | 180 | 90 s (04a §4.1) |
| `MAP_WIDTH` / `MAP_HEIGHT` | 10.0 / 25.0 m | **O4 resolved** |
| `CORRIDOR_WIDTHS_M` | **10 m only** | **fixed at the basin** (revision 2.6); was the Study 1 sweep, 10–3.5 m |
| `PREDICTED_THRESHOLDS_M` | 6.80 / 6.30 / 4.95 / 3.80 m | per-class, 02a §2.2, at the floored domain — **all below 10 m** |
| `HEAD_ON_WALL_CLEARANCE` | 0.65 m | `TODO(05)` |

**Revision 2.6 fixed the corridor at 10 m; revision 2.7 restored the sweep below.** What 2.6 recorded: The generator,
the virtual-boundary machinery and every width parameter stay in the code, but
the curriculum, Tier B's strata, Around the Clock and Study 1 are all configured
at 10 m. Consequences recorded in `PROJECT_STATE.md` §3.14: every class's
compliant alteration fits a 10 m channel, so no width alone makes Rule 8(e) the
governing rule; no bend fits (F25); Tier B realises one width stratum of three;
28 of 34 Tier A cases no longer differ in geometry from their 10 m siblings; and
a crossing target has no water outside the corridor to start from. The sweep
text below describes revision 2.5.

**O4 resolved (03 §5): simulation matches the basin.** Maximum corridor width
10 m = 20 breadths, so every simulated width is physically reproducible — a
meaningful strengthening of the field-validation argument. The unconfined
reference case comes instead from the open-water "Around the Clock" variant,
which is unbounded by construction.

The sweep, in metres and breadths:

| 10 m | 8 m | 7 m | 6 m | 5 m | 4 m | 3.5 m |
|---|---|---|---|---|---|---|
| 20 B | 16 B | 14 B | 12 B | 10 B | 8 B | 7 B |

02a §2.2 predicts **four** per-class transitions, and 02b C1 makes them a
computed function of the domain rather than four literals that silently go
stale. `constants.predicted_thresholds(d_abeam, c_wall, breadth)` implements
02a §2.2's derivations directly and reproduces its figures to 1 cm:

| Class | Derivation | Computed | 02a |
|---|---|---|---|
| Crossing | `2(d_req + c_wall + B/2)` | 6.51 m | 6.52 |
| Head-on, centreline target | `2·d_req + 2·c_wall` | 6.01 m | 6.02 |
| Overtaking | `1.15(d_req + 2·c_wall + B)` | 4.78 m | 4.78 |
| Head-on, compliant target | `d_req + 2·c_wall` | 3.66 m | 3.66 |

Re-deriving after 05 is one call.

**F18 stands and 02b C1 confirms it: do not add 6.25 m.** Crossing and
centreline head-on share the (6, 7) bracket and cannot be separated by the
current sweep — but both thresholds derive from `d_abeam` and `c_wall`, and both
are `TODO(05)`. Adding a level to separate two numbers that will move is
premature, and it would bake a level into a suite that 04 §4.5 requires frozen
and hashed. The sweep holds at seven levels;
`test_crossing_and_centreline_head_on_still_share_a_bracket` keeps the collision
documented so it is revisited rather than forgotten.

## 3. Actuation and the speed calibration

> **Superseded in revision 2.5.** `U_REF` is `ship.steady_speed(CRUISE_RPM)` on the
> identified hull — 1.116 m/s — and `THRUST_CAL` no longer exists. The two methods
> (hull fit, log speed over ground) agree to 2 %. The calibration history below is
> kept because it records how the 0.55 m/s placeholder was found out.

| Symbol | Value | Status |
|---|---|---|
| `CRUISE_RPM` | **6.0** | F24 decided, 0.55 m/s (rev 2.7; was 12.0) |
| `RPM_STAGE` | 1 → (±3, 9, 15) | curriculum entry; **stage 4 is the endpoint** |
| `REVERSE_AVAILABLE` | False | `TODO(03)` — capability unverified |
| **`U_REF`** | **0.558 m/s** (1.116 at 12 rpm-units) | **derived from the identified plant** at `CRUISE_RPM`; the 1.14 m/s log median corroborates the plant at 12 |
| ~~`ship.THRUST_CAL`~~ | removed | the identified `T12` reproduces cruise without calibration |

**Propulsion widening is resolved** (03 §6, 02 §4.4). Rule 8(e) speed reduction
is the designated fallback whenever a compliant course alteration would push the
vessel into the boundary. Staged through the curriculum as in Paper 2, but the
final stage must expose stage 4 — it is the only one reaching 0 RPM.

### 3.1 `U_REF` is now measured, and all three prior figures were wrong

02b T1 mined the retained Paper 2 field logs, which carry commanded RPM
alongside the localiser pose in their `#ACTION` records. Speed over ground
differentiated from the pose track over the last 60% of each run, so the
acceleration ramp is excluded:

**Median 1.14 m/s at 12 RPM across 18 logs**, spread 0.56–1.25, and remarkably
consistent trial to trial. Froude 0.29 — a displacement hull.

| Source | Value | Fr | Verdict |
|---|---|---|---|
| **Measured, this analysis** | **1.14 m/s** | 0.29 | — |
| 02b C2 "Paper 2 field" | 0.55 m/s | 0.14 | ~2× too low |
| 02a §1 `U_ref` | 0.80 m/s | 0.20 | assumption |
| Simulator, pre-calibration | 1.77 m/s | 0.45 | ~1.55× too high |

02b C2's *direction* was right — the simulator was too fast and the thrust map
needed calibrating — but by 1.55×, not 3.2×.

> **Chain of custody, stated plainly.** The 0.55 m/s figure 02b treats as "a
> measurement" was not one. It entered as an unsourced placeholder in the first
> `constants.py`, carrying a comment claiming it was measured from the field
> trials. It was not measured; it was invented, and then adopted downstream as
> ground truth. The 1.14 m/s above *is* measured, from the logs, and the script
> is reproducible.

**One constant, consumed by everything.** `U_REF` feeds the `ego` normaliser,
both target-speed features, `SPEED_SCALE`, `TARGET_SPEED_RANGE`,
`BEING_OVERTAKEN_SPEED_MARGIN`, `TRACK_GATE_DIST` and every reward speed gate.
The earlier saturating-`ego` bug was two constants disagreeing about the same
physical quantity; one constant makes that impossible.

`ship.THRUST_CAL = 0.3751` scales the thrust map so steady surge at `CRUISE_RPM`
equals `U_REF`. Solved by bisection. The resulting envelope:

| RPM | 6 | 9 | 12 | 15 | 18 | 24 |
|---|---|---|---|---|---|---|
| u (m/s) | 0.49 | 0.83 | **1.14** | 1.42 | 1.69 | 2.16 |
| Fr | 0.12 | 0.21 | 0.29 | 0.36 | 0.43 | 0.55 |

`TODO(05)`: this is a *calibration*, not an identification. 05 replaces the
thrust map; then only this one number changes.

## 4. Raw LiDAR (RPLidar C1)

| Symbol | Value | Status |
|---|---|---|
| `LIDAR_BEAMS` | 720 | **confirmed from field logs** |
| `LIDAR_BEAM_RES_DEG` | 0.5° | **confirmed** |
| `LIDAR_RANGE` | 16.0 m | |
| `LIDAR_MIN_RANGE` | 1.0 m | **confirmed** — Paper 2 had no dead zone |
| `LIDAR_DROPOUT_P` | 0.0 | `TODO(05)`, Study 2 axis |
| `LIDAR_NO_RETURN_GRAZING_DEG` | 0.0 | `TODO(05)` |
| `LIDAR_AFT_MASK_HALF_DEG` | 0.0 | `TODO(05)` |

Evidence, from all 30 logs under `field_deployment/` (5597 pooled scans):

* 720 bins in every scan of every log; values in decimetres spanning 10–153,
  i.e. **1.0–15.3 m**, matching the C1's stated 1–16 m.
* A mean of **506** bins carry a return (5th–95th pct 424–588), but **96.5% of
  the empty bins fall in contiguous runs longer than 3 bins**. They are
  no-return / out-of-range arcs, not angular under-sampling. The resolution
  genuinely is 0.5°, so the simulator keeps 720 beams and models the no-return
  process separately (agreed at review; downsample later if the model is
  overloaded).
* **No aft self-occlusion arc is detectable.** No bin is zero in more than 98%
  of scans in any log, and the peak zero-rate bearing wanders between logs
  (108°–359°), so it tracks the scene rather than the mount. Settling it needs a
  static-spin recording — vessel stationary in a known surround — which is ten
  minutes at the next basin session.

  **This one gates the being-overtaken class.** Train the tracker to see astern
  when the real mount cannot, and Rule 17 behaviour fails in the field for
  reasons that have nothing to do with the policy (01 §2.3).
* **Motion distortion is not assessable** from these logs: one wall-clock stamp
  per revolution, no per-beam times. Needs a raw bag with `time_increment`.

The 1 m dead zone is a sim-to-real gap 01 does not mention. With the sensor at
the bow of a 1.57 m hull, a target alongside inside 1 m is invisible to the real
sensor and was fully visible in Paper 2's simulator.

## 5. Sector pooling

| Symbol | Value |
|---|---|
| `POOL_SWATH_HALF_DEG` | 135° |
| `LIDAR_SECTORS` | 27 |
| `FEASIBILITY_SAFE_WIDTH` | 0.80 m = `VESSEL_WIDTH + 2 × HULL_MARGIN` |

Allocation 15 + 8 + 4 = 27, covering 540 of 720 beams. Full table in
`OBSERVATION_SPEC.md` §1. Nothing here is unresolved.

The 11.25° sectors hold 22.5 beams at 0.5° and therefore **alternate 23/22**.
01 §2.2 writes "22–23", which is consistent, but the allocation cannot be
constant across that band and the tests assert the alternation.

## 6. Boundary branch

| Symbol | Value | Status |
|---|---|---|
| `BOUNDARY_BEARINGS_DEG` | −90, −60, −30, 0, +30, +60, +90 | |
| `BOUNDARY_MAX_RANGE` | 16.0 m | same normaliser as `c_t`, deliberately |
| `BOUNDARY_GATE_MARGIN` | 0.30 m | `TODO(05)` |
| `BOUNDARY_POSE_NOISE_XY` | 0.0 m | `TODO(05)`, Study 2 axis |
| `BOUNDARY_POSE_NOISE_HEADING_DEG` | 0.0° | `TODO(05)` |
| `BOUNDARY_POSE_NOISE_WALK` | 0.0 m/step | `TODO(05)` |

> **The pose-noise magnitudes are all 0.0, so the sim-to-real gap 01 §3.3 warns
> about is currently wide open.** The hook is on the execution path and tested,
> but a headline training run must not start until 05 supplies the rf2o drift
> figures. This remains the single most consequential outstanding number here.

**O5 resolved: software gating, not a physical barrier.** The facility walls
carry the fixed geometric features — recessed doorways, protruding benches —
that are the only along-track constraint available to the scan-to-map
localisation in 05, and a barrier would occlude them. So the pipeline must
**localise on the full scan including the walls, and gate only afterwards for
the tracker**. The walls are a liability for target tracking and an asset for
localisation, and `env._perceive` is ordered to treat them as both.

Gating is mandatory rather than preferable: during trials, operators standing on
the deck sit at scan height and move.

## 7. Tracking — the N1 headline

| Symbol | Value | Status |
|---|---|---|
| `CLUSTER_EPS` | 0.35 m | approved 02b §2 |
| `CLUSTER_MIN_POINTS` | 4 | approved 02b §2 |
| `TRACK_GATE_DIST` | 1.40 m | derived, `max(2.5·U_REF·Δt, 0.30)` at Δt = 0.5 s |
| `TRACK_MAX_MISSES` / `TRACK_MIN_HITS` | 3 / 2 updates | 1.5 s of coasting; published 0.5 s after first sight |
| `KF_PROCESS_NOISE_ACCEL` | 0.10 m/s² | `TODO(05)` |
| `KF_MEAS_NOISE_POS` | 0.05 m | `TODO(05)` |
| `MOTION_CLASSIFIER` | `"free_space"` | F31's fix (§7.0); `"speed"` is 01's original, kept as the ablation |
| `MOTION_WINDOW_S` | 2.0 s = 4 updates | 03a §6.3's `T_w` |
| `MOTION_PASS_TOL_M` | 0.25 m | `max(0.25, 5·√2·σ_p)` — `TODO(05)`: σ_p |
| `MOTION_EXPLAIN_M` | 0.30 m | "a return was near it" |
| `MOTION_MIN_POINTS` / `MOTION_MIN_FRAC` | 3 / 0.10 | violations in one update that count as motion |
| `DYNAMIC_PROMOTE_STEPS` / `_DEMOTE_STEPS` | 2 / 4 updates | 03a §6.3's 0.8 s / 2.0 s, asymmetric |
| `DYNAMIC_SPEED_ON` / `_OFF` | 0.15 / 0.08 m/s | speed classifier only; `TODO(05)` |

### 7.0 F31 — static and dynamic by free space, not centroid speed

01 classified a track by the Kalman speed of its cluster **centroid**. The
centroid is the mean of whatever face the sensor sees, and that face changes as
the own ship passes a panel, so it slides: 0.17–0.24 m/s with pose noise off,
p90 0.5–0.6 m/s, overlapping the slowest target (0.39 m/s). No threshold
separates them.

The free-space test asks what motion physically is, over a 2 s window:

* **appear** — a return now where a ray 2 s ago returned from beyond it, with no
  return then within `MOTION_EXPLAIN_M`;
* **vacate** — a return of the track's 2 s ago that a ray now returns from beyond,
  with no return now within `MOTION_EXPLAIN_M`.

A static solid can do neither from any viewpoint: a ray through a point on its
boundary must already have hit it, and a newly revealed face was occluded, not
empty. Only the beam toward the point certifies free space, and only if it
*returned* — the C1's 1 m dead zone reports nothing for a surface it is
touching, so an empty beam certifies nothing. And the certificate has to survive
a pose error of `MOTION_PASS_TOL_M` in **every** direction: every beam within
`atan(tol/ρ)` of the point must clear it as well. Checked radially alone, 3 cm of
pose noise produced phantoms on panel faces seen edge-on, where a centimetre
sideways lets a ray graze past the corner and return from the wall. Two fixes rode along: returns are
lifted from the **sensor**, 0.86 m ahead of the vessel origin (they were lifted
from the origin, so static objects appeared to move whenever the vessel turned),
and a stale-pose frame never reaches the tracker. Measured results are in
`PROJECT_STATE.md` §3.14.

**`CLUSTER_MIN_POINTS` was raised from 3 to 4.** Suspension lines run diagonally
across the basin and descend toward their anchors, so near the pool edges they
cross the scan plane; a taut rope returns on one or two beams (03 §4a). Four
points clears a rope while still admitting a genuine small obstacle — asserted
in `tests/test_tracking.py::test_min_points_rejects_a_taut_suspension_line`.

**The static/dynamic threshold is a property of localisation quality, not of
obstacle behaviour.** Field obstacles are suspended panels, confirmed from video
to hang stably, so apparent motion of a static object comes almost entirely from
ego-pose error — which affects every object in the scan identically. Set it from
measured pose noise (05 §4) and retighten as registration improves, and bias it
toward **under**-detection: promoting a static panel to a target ship is a false
positive with COLREGs consequences.

`tests/test_tracking.py::test_pose_drift_creates_false_velocity_on_a_static_object`
demonstrates the mechanism under the speed classifier: 0.2 m/s of apparent drift
misclassifies a fixed object. `tests/test_motion_classifier.py` shows the
free-space test ignoring a close pass that fools the speed classifier.

### 7.1 Study 2 degradation axes

Exposed as environment config so the sweep in 04 can drive them (01 §4.1). All
nominal-zero, so the tracker is exact unless a sweep asks otherwise.

| Axis | Constant | Injected at |
|---|---|---|
| Pose drift | `BOUNDARY_POSE_NOISE_*` | estimated pose → boundary raycast **and** tracker ego-motion compensation |
| Detection dropout | `DETECTION_DROPOUT_P` | tracker input |
| Occlusion duration | scenario geometry | measured as `Tracker.max_coast` |
| Velocity noise | `TRACK_VELOCITY_NOISE` | tracker velocity estimate |

Plus a fifth that 01 does not list but 05 §6 and 03 §7 do:

| `EGO_SPEED_NOISE`, `EGO_YAW_RATE_NOISE_DPS` | `ego` observation branch |
|---|---|

**An IMU is confirmed** (05 §4.7, Revision 2.2). That changes the character of
this gap rather than closing it:

* `r` comes from the gyro directly, so the residual is the sensor noise floor
  rather than pose-differentiation error. `r_dead` in 02a §6.2 is now set from
  the measured floor rather than guessed, and the yaw-rate-not-rudder criterion
  02 §4.2 depends on becomes **directly measurable in the field** instead of
  inferred.
* `u` and `v` are largely rescued by the accelerometer, but are still fused
  rather than measured, so a residual remains.

Scan-to-map supplies drift-free absolute pose at 10 Hz; the IMU fills in between.
Both magnitudes still come from 05. Log raw gyro and accelerometer at 100 Hz+,
time-synced to the LiDAR — 05 §4.7 flags the sync as the detail that will bite,
because a constant offset appears in the fit as actuator lag.

## 8. Ship domain — RESOLVED (provisional), with a floor that already bites

| Direction | Multiple | Metres |
|---|---|---|
| Ahead | 2.00 · Lpp | 3.14 |
| Astern | 1.00 · Lpp | 1.57 |
| Abeam (each side) | **0.796 · Lpp** (floored) | **1.25** |

Lateral footprint 2.50 m, about 25% of the widest channel. The abeam extent is
set by the sensor floor rather than by an Lpp multiple — see §8.1.

Chun et al.'s 3·Lpp fore/aft and 1·Lpp abeam gives 4.71 m fore-aft and 3.14 m
across at LBP = 1.57 m — nearly a fifth of the 25 m basin lengthwise, and enough
lateral footprint that two vessels could not pass without mutual domain
intrusion at any swept width. Every episode would score a violation and the
metric would carry no signal.

> **The principle matters more than the numbers.** These are a provisional
> *input*. The final values are an *output* of 05: derive them from measured
> manoeuvring performance — advance and tactical diameter from the turning
> circles, stopping distance from the stop test — so the domain is "sized to
> this vessel's demonstrated ability to avoid", which is the argument Thyri &
> Breivik make for confined water. Do not defend them as a scaled copy of
> someone else's domain. Szłapczyński & Szłapczyńska (2017) is the reference for
> justifying the compression.

`DOMAIN_RADIUS_DCPA` is undefined for an asymmetric domain. Convention adopted:
the **lateral semi-axis**, because DCPA is a closest-approach distance and
closest approach in a channel is overwhelmingly a beam-on passing geometry.

**Resolved (02b §2), and the resolution is that observation and reward normalise
differently on purpose.** This constant scales the observation feature only; the
reward uses the directional `d_dom(β)` at the target's actual bearing (02a §5.3)
and gates `ρ_t` on the constant `d_req`. Documented rather than unified, because
the two serve different jobs.

### 8.1 F21 — the sensor-resolution floor binds, and it is applied

02b §3.1 sets a hard floor: `d_abeam ≥ LIDAR_MIN_RANGE + B/2 = 1.25 m`. Below
it the ship domain sits inside the sensor's blind zone, and because `R-1`
evaluates domain intrusion on ground-truth geometry, the agent would be
penalised for intrusions it cannot perceive — `r_dom` stops being a shaping
signal and becomes unlearnable.

`0.75 · Lpp = 1.18 m` does not clear that floor. §3.1 says the provisional value
"sits 0.18 m outside the sensor blind zone", which compares it to
`LIDAR_MIN_RANGE` alone (1.0 m) — then defines the floor as 1.25 m. The two
halves of the section disagree.

**Resolved in favour of the floor**, because §3.1 says what to do in exactly
this case: "If the measured manoeuvring performance implies a smaller one, the
domain is floored at 1.25 m and the paper states why." Applying it executes
02b's decision rather than overriding 02a's — 02b is the later document and
declares itself a companion that amends 02a §1.

`d_abeam = 1.25 m` (0.796 · Lpp), `d_req = 2.50 m`, and all four Study 1
thresholds move:

| | at 1.18 m | **in force** |
|---|---|---|
| Crossing | 6.51 m | **6.80 m** |
| Head-on, centreline | 6.01 m | **6.30 m** |
| Overtaking | 4.78 m | **4.94 m** |
| Head-on, compliant | 3.66 m | **3.80 m** |

02b C1's bracket collision survives: 6.80 and 6.30 still share the (6, 7)
bracket, so the sweep stays at seven levels.
`RewardConfig` now **raises** on a domain below the floor, which is the
assertion 02b T4 step 4 asks for; `constants.check_domain()` still returns
rather than raising, so 05 can enumerate the problems with a candidate domain
without catching an exception per candidate.

**A sentence for the paper.** The domain's abeam extent is not a manoeuvring
figure at all — it is set by what the sensor can resolve. A domain smaller than
the sensor's blind zone is not a domain, it is a blind spot, and sizing it that
way would mean scoring intrusions the vessel cannot detect in the basin either.

## 8b. The width thresholds are computed, not listed

## 9. Collision Risk Index

| Symbol | Value | Status |
|---|---|---|
| `CRI_DCPA_SCALE` | 4.0 m | `TODO(decision)` |
| `CRI_TCPA_SCALE_BEFORE` / `_AFTER` | 20.0 / 6.0 s | `TODO(decision)` |
| `CRI_ED_SCALE` | 5.0 m | `TODO(decision)` |
| `CRI_BOW_CROSSING_GAIN` / `_HALF_DEG` | 1.3 / 45° | `TODO(decision)` |
| `DCPA_CLIP_DOMAINS` | 10.0 | `TODO(decision)` |
| `TCPA_CLIP` | 60.0 s | `TODO(decision)` |

**The decay rates could not be re-derived in ship lengths, and the reason should
be on the record.** Waltz & Okhrin scale their decay to 2 NM = 3704 m for a
320 m KVLCC2, i.e. **11.6 Lpp**. Scaled to LBP = 1.57 m that is **18.2 m —
larger than the 16 m sensor horizon.** A faithful re-derivation in ship lengths
produces a risk index that never decays within anything the vessel can perceive,
so every target would read as maximum risk at all times.

The constants above are therefore anchored to the **sensor horizon** rather than
to ship lengths. That is a different choice from the one 01 §5.2 specifies and
it needs sign-off. The underlying reason is that it is the *channel* that is
small, not the sensing: a 320 m ship with 2 NM of radar sees 11.6 hull lengths
ahead, the Bluefin with 16 m of LiDAR sees 10.2 — close enough that the mismatch
is easy to miss.

`CR_ED`, the plain Euclidean-distance term, is **not optional in a channel**.
Two vessels 2 m apart on near-parallel courses have a CPA far away in time, so a
pure CPA risk reads almost nothing — and in a corridor, near-parallel geometry is
the normal case rather than the exception.

## 10. Encounter classifier — five classes

| Symbol | Value | Status |
|---|---|---|
| `HEAD_ON_BEARING_HALF_DEG` | **10.0°** | **resolved** (01 §5.3) — source is 5.0° |
| `HEAD_ON_CT_HALF_DEG` | **10.0°** | kept symmetric — see below |
| `CROSSING_STBD_MAX_DEG` | 112.5° | source table |
| `CROSSING_PORT_MIN_DEG` | 247.5° | source table |
| `OVERTAKING_CT_HALF_DEG` | 67.5° | source table |
| `BEING_OVERTAKEN_BEARING_MIN/MAX_DEG` | 112.5° / 247.5° | **new class** |
| `BEING_OVERTAKEN_SPEED_MARGIN` | 0.10 m/s | `TODO(decision)` |
| `ENCOUNTER_HOLD_STEPS` | 8 steps | |
| `ENCOUNTER_BEARING_HYSTERESIS_DEG` | 3.0° | |

Three modifications to Waltz & Okhrin Table 1 (01 §5.3):

1. **Port and starboard crossing collapse into one class** under Rule 9(b). The
   own ship gives way either way, so the side is not a different obligation. The
   geometry is still computed and exposed as `encounter.crossing_side()` for
   02's passing-side reward term — it is simply not observed.
2. **The head-on band is widened from ±5° to ±10°** — resolved in 01 §5.3
   Revision 2.2. The source value is tight enough that a small heading error
   flips the classification; ±10° is within common practice and gives the
   hysteresis room to work. 01 resolves "the head-on band" without separating
   bearing from heading; kept symmetric here, so courses within 10° of
   reciprocal count as head-on, which is the reading matching the stated
   rationale (a small *heading* error).
3. **"Being overtaken" is added**, mirroring the overtaking condition with
   U_TS > U_OS. Only Rule 17(a)(i) passive course-keeping is in scope; active
   release under 17(a)(ii) is future work (S5).

**Rule 13 precedence.** Overtaking and being-overtaken are tested *before* the
crossing rules, matching Rule 13(a)'s "notwithstanding anything contained in the
Rules of Part B, Sections I and II". The CT bands happen to be disjoint, so this
changes no outcome today — but the precedence is structural rather than
accidental, which matters when 02 builds the Rule 9 vs Rules 13–16 table.

The speed margin decides when a marginally faster overtaker triggers Rule 17
stand-on. Too small and the class flickers on speed-estimation noise, which is
the noisiest quantity the tracker produces; too large and genuine overtakings
are missed. 0.10 m/s is 18% of cruise.

## 11. Every unresolved constant

Current as of T4. Values are what is **in force**, not the historical
placeholder.

| # | Symbol | In force | Marker |
|---|---|---|---|
| 1 | `HEAD_ON_WALL_CLEARANCE` (`c_wall`) | 0.65 m | `TODO(05)` |
| 2 | `OVERTAKING_PRUDENCE` | 1.15 | `TODO(05)` |
| 3 | `REVERSE_AVAILABLE` | False | `TODO(03)` |
| 4 | `LIDAR_DROPOUT_P` | 0.0 | `TODO(05)` |
| 5 | `LIDAR_NO_RETURN_GRAZING_DEG` | 0.0 | `TODO(05)` |
| 6 | `LIDAR_AFT_MASK_HALF_DEG` | 0.0 | `TODO(05)` |
| 7 | `BOUNDARY_GATE_MARGIN` | 0.30 m | `TODO(05)` |
| 8–10 | `BOUNDARY_POSE_NOISE_XY` / `_HEADING_DEG` / `_WALK` | 0.0 | `TODO(05)` |
| 11–12 | `KF_PROCESS_NOISE_ACCEL` / `KF_MEAS_NOISE_POS` | 0.10 / 0.05 | `TODO(05)` |
| 13–14 | `DYNAMIC_SPEED_ON` / `_OFF` | 0.15 / 0.08 m/s | `TODO(05)` |
| 15 | `DETECTION_DROPOUT_P` | 0.0 | `TODO(05)` |
| 16 | `TRACK_VELOCITY_NOISE` | 0.0 | `TODO(05)` |
| 17–18 | `EGO_SPEED_NOISE` / `EGO_YAW_RATE_NOISE_DPS` | 0.0 | `TODO(05)` |
| 19–20 | `DOMAIN_FORE` / `_AFT` | 3.14 / 1.57 m | `TODO(05)` |
| 21 | `DOMAIN_LATERAL` | **1.25 m, floored** (§8.1) | `TODO(05)` |
| 22 | `TARGET_COMPLIANT_SPAWN_PROB` | 0.5, report realised | `TODO(04)` |
| 23 | `USE_RECURRENCE` | False | `TODO(04)` |
| 24 | `KAPPA_N` (throttle rate scale) | 0.30 | `TODO(05)` |
| 25 | `R_REF` (full-severity yaw) | 0.20 rad/s | `TODO(05)` |
| 26 | `R_DEAD` (gyro noise floor) | 0.02 rad/s | `TODO(05)` |
| 27 | `DU_HOLD` | 0.10 m/s | `TODO(05)` |
| **28** | **`D_SAFE`** | **0.35 m** (§12.3) | **`TODO(decision)`** |

**Closed since Revision 2.2**, all by 02b or T4: the three terminal payoffs,
`TCPA_CLIP`, `DCPA_CLIP_DOMAINS`, `BEING_OVERTAKEN_SPEED_MARGIN`,
`TARGET_SPEED_RANGE`, `TRACK_GATE_DIST`, the whole CRI block,
`DOMAIN_RADIUS_DCPA`, `CLUSTER_EPS`, `CLUSTER_MIN_POINTS`,
`NO_TARGET_EPISODE_PROB`, `U_CRUISE`/`U_MAX_SURGE` (T1, measured),
`PREDICTED_THRESHOLDS_M` (C1, now computed), and `DOMAIN_LATERAL`'s
`TODO(decision)` half (F21).

`KAPPA_DELTA` is **derived** rather than unresolved:
`MAX_RUD_RATE_DPS · Δt / MAX_RUD_ANGLE = 0.05`. It moves when 05 delivers the
calibrated actuator model, but it moves as a consequence rather than as a call.

---

## 12. Reward (`02a` Rev 2.2, with `02b` C1–C4) — T4

Eight dense terms plus terminals. **Every dense term is normalised to `[-1, 0]`
before weighting** (`r_prog` to `[-1, +1]`), so the weight *is* the maximum
per-step contribution and the `02a §7` hierarchy holds by construction rather
than being discovered empirically. That is the direct fix for the Paper 2
failure, where a path term out-scaled the avoidance term through a factor nobody
had computed.

| Term | Weight | Range | Notes |
|---|---|---|---|
| Terminal collision | — | one-shot | −300 (`R-7`) |
| Terminal goal | — | one-shot | +100 |
| Terminal timeout | — | one-shot | **0**, requires `truncated=True` |
| `r_bnd` boundary | 15.0 (3.00) | `[-1,0]` | hard constraint, above the COLREGs group |
| `r_dom` target domain | 12.5 (2.50) | `[-1,0]` | ground truth, centre-to-centre |
| `r_obs` static obstacle | 11.0 (2.20) | `[-1,0]` | **shifted** exponential, zero past 2 m |
| `r_col` COLREGs group | 9.0 (1.80) | `[-1,0]` | clipped to unit range *before* the weight |
| `r_pf` path following | 3.0 (0.60) | `[-1,0]` | width-normalised on `W_local` |
| `r_prog` progress | 1.5 (0.30) | `[-1,+1]` | telescoping arclength (`R-9`) |
| `r_smooth` smoothness | 0.5 (0.10) | `[-1,0]` | `κ_δ` derived from the actuator |
| `r_exist` existence | 0.25 (0.05) | `−1` | suspended by `R-5` only |

**Revision 2.6:** weights are 02a's per-0.1 s values (in brackets) times
`REWARD_DT_SCALE = 5`, so each term keeps its per-second rate at the 2 Hz step.
The ordering is a common factor and unchanged; every episode integral in 02a
§8.1 is unchanged.

COLREGs sub-weights: `v_port` 0.55, `v_bow` 0.55, `v_side` 0.40, `v_hold` 0.45,
`v_r8` 0.50. Pre-clip maxima 1.45 / 1.60 / 1.50 / 0.45 by class, so two
concurrent severe violations saturate the group and one does not.

### 12.1 The validators, and what each one is for

They fail at **construction**, not at step 10,000. A reward whose hierarchy is
silently violated does not crash — it trains for a week and produces a policy
nobody can explain.

| Assertion | Guards against |
|---|---|
| `d_abeam ≥ 1.25` | a domain inside the sensor blind zone; `r_dom` unlearnable (02b §3.1) |
| `d_safe < c_wall − B/2` | the compliant narrow-channel manoeuvre triggering the boundary penalty |
| `w_bnd > w_dom > w_obs > w_col > w_pf > w_prog > w_smooth > w_exist` | the `02a §7` hierarchy |
| `\|r_collision\| > w_col · MAX_ENCOUNTER_STEPS` | a compliant collision scoring better than a non-compliant near-miss (`02 §5`); 300 > 9.0 × 20 |
| `t_act < t_engage` | the obligation becoming urgent before the encounter engages |
| `kappa_eng > 1` | engagement firing at the compliant separation itself |
| `kappa_rel > kappa_eng` | an encounter clearing at the range it engages; state-machine chatter |
| `d_cut > d_oa` | an obstacle term with no cut-off |
| `kappa_delta` set | 05 not having delivered the actuator rate limit |

### 12.2 F22 — `N_ref` is derived, not written down

`02b C3` replaces `02a §5.5`'s progress form with
`clip(N_ref · Δs / L_path, −1, +1)`, so the episode integral stops depending on
the unresolved cruise speed. The clip binds when `u > L_path / (N_ref · Δt)`.

**At C3's literal `N_ref = 250` over a 20 m path that is 0.80 m/s** — 02a's
*assumed* cruise, not the 1.14 m/s T1 measured. Every step at cruise would clip,
and slowing down would then *increase* the integral (175 → 250). That is `R-9`
running backwards: the term that exists to remove the creep exploit would have
introduced it, and invisibly, because the sum still telescopes — just to the
wrong thing.

```
N_REF_PROG = L_REF_PATH / (U_REF · Δt) = 20.0 / (0.558 × 0.5) = 71.7     # 35.8 at 1.116 m/s
```

which binds the clip at exactly cruise and restores 02a §5.5's stated intent:
telescoping exact for `u ≤ U_ref`, speeding gains nothing. **`w_prog · Σr_prog`
is +52.6, not the +75 in the `02a §8.1` table** — the only row it moves. All
three orderings survive with a 44-point margin.

**This puts a constraint on 03's generator**: the clip binds at
`U_REF · L_path / L_REF`, so a path much shorter than the 20 m design point
brings F22 back. Today's generator produces 20.00–20.42 m. Pinned by
`test_3d_the_generator_must_keep_path_length_near_the_design_point`.

### 12.3 F23 — `02a`'s `d_safe` breaches `02a`'s invariant

`02a §2` asserts `d_safe < c_wall − B/2`. With `c_wall = 0.65` and `B = 0.50`
the ceiling is **0.40 m**, and `02a §5.2`'s stated `d_safe = 0.50 m` does not
clear it — in the same sentence that says 0.50 was chosen *because of* this
invariant.

`D_SAFE = 0.35 m`, the largest 5 cm value that clears with margin.
`TODO(decision)`, and the last one in the tree. `d_safe` is the free parameter
of the pair: `c_wall` drives all four Study 1 thresholds and is a `TODO(05)`
measurement, so moving it would move published predictions, whereas `d_safe`
only sets where the boundary penalty begins.

**A second part, narrower.** `02a §8.1`'s "loitering to timeout ≈ −86" assumes
its own 300-step design point; `MAX_EPISODE_STEPS` is 700, a Paper 2 carry-over
never reconciled with it. Undiscounted, a vessel stopped dead pays
`w_pf + w_exist = 0.65` per step and reaches −455 against a collision's −300,
inverting the `02 §5` ordering. It holds under the discount the agent actually
uses (−65 at the headline SAC `γ = 0.99`) and is **marginal for the PPO
comparator at `γ = 0.999`**, which reaches −327. Pinned with the numbers rather
than fixed, because the fix is the step limit and that is a training decision.

**Revision 2.6.** At 2 Hz with the weights ×5 and the discounts converted
(`discount(0.99)` = 0.951, `discount(0.999)` = 0.995), a stopped vessel pays 3.25
per step over 180 steps: −585 undiscounted, −66 at SAC's discount, −386 at
PPO's — the 10 Hz numbers, by construction.

### 12.4 Speed references

| | Value | When |
|---|---|---|
| `U_REF` | 0.558 m/s | nominal, identified plant at `CRUISE_RPM = 6` |
| `U_ref_eff`, `R-2` | 0.456 m/s | give-way, compliant alteration inadmissible |
| `U_ref_eff`, `R-5` | `max(u_TS, 0.20)` | overtaking, port pass does not fit |

`R-5` also zeroes the existence cost. It is the **one** place that happens, and
it is gated on a geometric predicate rather than on CRI, which keeps it
consistent with `R-6` and avoids the degenerate-policy risk `02 §4.4` warns
about. Without it the narrow overtaking case is not a test of COLREGs reasoning
but a test of whether the agent tolerates an unwinnable reward — and it would
resolve it by overtaking anyway.

---

## 13. Vessel plant, emergency stop, deployment timing — revision 2.5

### 13.1 Plant (`ship.py`, from `bluefin/`)

| Symbol | Value | Status |
|---|---|---|
| `U_REF` | 0.558 m/s | derived: `steady_speed(CRUISE_RPM = 6)` on the identified hull |
| `U_REF_LOG_MEDIAN` | 1.14 m/s | measured, 02b T1 — corroboration only |
| `SUB_DT` | 0.05 s | required by `bluefin/REPORT.md` §9 |
| `COMMAND_RATE_PCT_S` | 50 %/s | the optional rudder limiter's rate — a true rate limit, identical in bridge and `env.step` |
| `RUDDER_COMMAND_LIMIT` | **False** | off in bridge and simulator: v3 predicts the raw-command July runs at least as well (holdout heading 9.8° vs 15.1° at 10 s); S1-B measures the servo |
| `MAX_RUD_RATE_DPS` | 20 °/s | derived from the above; `KAPPA_DELTA` = 0.25 per 2 Hz step |
| `REVERSE_THRUST_EFFICIENCY` | 0.5 | **`TODO(05)`** — basin S1-C2; T9 at 2 Hz needs ≥ 0.27, or ≥ 0.55 if thrust shares the rudder's 0.73 s delay |
| `VESSEL_RANDOMISATION_SCALE` | **1.0** | on for training (§14) |

Identified parameters live in `bluefin/ship_model_v3.py` and are not duplicated
here: a copy would drift out of step with part 2's refit.

### 13.2 Emergency stop (`emergency_stop.py`)

| Symbol | Value | Status |
|---|---|---|
| `EMERGENCY_STOP_ENABLED` | True | mechanism on |
| `ESTOP_TRIGGER` | `"supervisor"` | **back on in revision 2.6** — F31 fixed, zero phantom tracks in 24,024 target-free frames |
| `ESTOP_STOP_SPEED` | 0.05 m/s | **`TODO(05)`** — the field speed estimate's noise floor |
| `ESTOP_MIN_HOLD_S` | 2.0 s | design |
| `ESTOP_MAX_HOLD_S` | 10.0 s | design — a target stopped dead ahead must not pin the latch |
| `ESTOP_MAX_BRAKE_S` | 8.0 s | design — never command full astern indefinitely |
| `REVERSE_AVAILABLE` | False | unchanged: it describes the **action space**, which stays forward-only |

`S2 = rpm / 24 · 100` forward, extended linearly through zero, so full astern
`S2 = −100` is −24 rpm-units.

### 13.3 Deployment timing — native since revision 2.6

| Symbol | Value | Status |
|---|---|---|
| `DEPLOYED_DECISION_DT` | 0.5 s | measured, `dt_cmd` median over 1,085 July frames; asserted equal to `UPDATE_RATE` at import |
| `POSE_STALE_PROB` | **0.0** | the bridge waits for each frame's pose line: 0 of 1,085 July frames stale on replay |
| `MEASURED_POSE_STALE_PROB` | 0.408 | the old bridge, 443 of 1,085 frames; for robustness studies |
| `SPAWN_INSET_M` | 1.41 m | F35: inflated half-length + `D_SAFE` + 0.05 |

The `DeploymentTiming` wrapper is gone. A stale frame keeps its fresh LiDAR
branch, repeats the previous frame's boundary, ego and path branches — in the
bridge all three come from the latched pose line — and is not fed to the
tracker.

## 14. Revision 2.7 — decisions and training settings

| Symbol | Value | Status |
|---|---|---|
| `CRUISE_RPM` | **6.0** | F24 decided: `U_REF` = 0.558 m/s, Fr 0.142 |
| `RPM_STAGES` | 12-rpm table × `CRUISE_RPM`/12 | stage 4: 0–12 rpm-units |
| `N_REF_PROG` | 71.7 steps | `L_REF_PATH / (U_REF · Δt)` |
| `TRACK_GATE_DIST` | 0.70 m | `2.5 · U_REF · Δt` |
| `BOUNDARY_POSE_NOISE_XY` / `_HEADING_DEG` | **0.03 m / 0.2°** | nominal, `TODO(05)`: S1-A |
| `EGO_SPEED_NOISE` / `EGO_YAW_RATE_NOISE_DPS` | **0.05 m/s / 1.0 °/s** | nominal, `TODO(05)` |
| `VESSEL_RANDOMISATION_SCALE` | **1.0** | A1–A3 12/12 at 1.0, 10/12 at 1.5 |
| `R_ESTOP` | **−20** | one-off per stop; speed gate suspended while latched; `TODO(02)` |
| corridor widths | 10–3.5 m sweep, variation 1.0–1.8, bends | restored (revision 2.6's fixed 10 m reversed) |

### 14.1 After the formulation run (revision 2.8)

| Symbol | Value | Status |
|---|---|---|
| `GOAL_END_INSET_M` | **0.37 m** | F49, derived: `0.5·LOA + 0.15 − GOAL_ALONG_DIST + steady_speed(2·CRUISE_RPM)·Δt + 0.05`; being-overtaken path end stops this far short of the corridor end |
| `PF_OVERSPEED_TOL` | **0.20** | F50, A16 **decided** (option 1): `r_pf`'s speed gate is flat to 1.2 · `U_REF` |
| `PF_OVERSPEED_SPAN` | **0.50** | F50, A16 **decided**: then falls to 0 by 1.7 · `U_REF` (0.95 m/s) |
| PPO `target_kl` | **0.03** | F51: run 1's KL reached 0.25 with half of each batch clipped |
| `BEING_OVERTAKEN_DCPA_FLOOR` | **1.0 m** | F53, A15 **decided** (option 1): 80 % of being-overtaken draws uniform on [1.0, 2.0] m |
| `BEING_OVERTAKEN_BELOW_FLOOR_FRAC` | **0.20** | F53, A15: uniform on [0, 1.0) m, labelled `dcpa_below_floor` (Rule 17(b)) |
| crossing turn sense | **+1 from starboard, −1 from port** | F53, A17 **decided** (option 1): `compliant_turn_sense(cls, crossing_side)` |
| `ESTOP_CLEAR_DCPA_M` | **1.76 m** | F56, A18 **decided** (option 1), derived: `0.5·LOA + 0.5·B + 2·0.15 + D_SAFE`. The supervisor stop, R-2's slowdown carve-out and the Rule 8 speed credit apply only when the target's DCPA with the own ship stationary reaches this |
| classification heading | **path tangent at the own ship** | F56, A19 **decided** (option 1): class, crossing side and true class use it; CPA products and the Rule 8 accumulator keep the instantaneous heading; open water falls back to the heading |
| stop test (`dcpa_if_stopped`) | **closest approach along the latch's braking path, then stopped** | F67, A23 **decided** (option 1): `stopping.dcpa_over_stop`, nominal hull, `STOP_TEST_DT_S` 0.05; against `ESTOP_CLEAR_DCPA_M` 1.76 m. From cruise: 1.05 s, 0.32 m |
| `POLICY_SLOWDOWN_RPM` / `SLOWDOWN_TEST_MAX_S` | **0 / 20 s** | F68: the agent's own slowdown (a coast) for R-2 and the Rule 8 credit (`slowdown_clears`); the supervisor keeps the latch profile (`stop_clears`) |
| training supervisor | **switch** (`--train-supervisor`, default on) | F68: off trains without the latch, so `R_ESTOP` and the gate suspension drop out |
| low-speed starts | **`low_speed_start_frac`**, env default 0; `LOW_SPEED_START_ZERO_SHARE` 0.5 | F68: from rest, or uniform on (0, 0.5 `U_NOM`] |
| `R2_SLOWDOWN_TEST` | **"stop"** (A24, decided) | F70/F72: R-2's and the Rule 8 credit's slowing test — "stop" (A23, `stop_clears`, the default) or "coast" (F68, `slowdown_clears`); `train_formulation.py --r2-slowdown-test` |
| A29 (F85) | `V_HOLD_GROWS` (off for run 9, on from run 10), `V_HOLD_EXCESS_SPAN` **0.10 m/s**, `V_HOLD_CAP` **3.0** | your call, 2026-09-19 |
| A27 (F81) | `V_PORT_HEADING_DEAD_DEG` **5.0** (TODO(05)); `CROSSING_PORT_SHARE_TRAINING` **0.60** (training namespace only); stage 3 classes gain crossing | your call, 2026-09-19 |
| geometry | **basin default** (`DEFAULT_GEOMETRY_MODE`); `BASIN_START_Y` 2.0, `BASIN_GOAL_Y` 22.0, `BASIN_X_RANGE` (2.5, 7.5) m, `BASIN_NAV_INSET_M` 0.40, `BASIN_H_SIDE_CLIP` (0.60, 5.00); `p_basin` 1 / 1 / 0.85 / 0.75 / 0.75 by stage, channel only for `CHANNEL_CLASSES` | F74 (06, your calls) |
| feasibility | A* grid 0.25 m, walls 0.40 m, panels 0.45 m, route ≤ 2.25 × leg, 20 redraws then thin | F74 (Paper 2's filter) |
| off-policy learners | TD3 / SAC / TQC: lr 3e-4, buffer 1 M, batch 256, tau 0.005, learning starts 10 k; TD3 policy delay 2, target noise 0.2 (clip 0.5), exploration 0.1; TQC 2 critics × 25 quantiles, top 2 dropped per critic | F76, F77 |
| suite | 3.0: Tier A 38 (35 realised), Tier B 48 × 20; `TIER_B_EPISODES_PER_CELL` 20; curriculum `CURRICULUM_STAGE_FRACTIONS` | F75 |
| observation | **70 values, 6 branches** (`OBSERVATION_SCHEMA_VERSION` "a25-v3-context") | F72: adds `context` (12 per slot + 2 previous-action values); cross-track error scaled by the local channel half-width, not 25 m |
| `STOP_TEST_USES_HULL_FIT` | **False** | F66–F67: the C15 hull-fitted close-range view (within `STOP_TEST_FIT_RANGE_M` 4 m) is built but off — it doubles stops in Tier 1 |
| `TRACK_MEASUREMENT` | **"centroid"** | F64: `hull_fit` built but not adopted — better within 4 m, worse course at 6–9 m |
| crossing escape check | **accept only if coasting (RPM 0) or a 60° compliant alteration, after 1.5 s at cruise, clears the target hull and keeps the own hull in the corridor** | F63, A22 **decided** (option 1): `CROSSING_ESCAPE_DELAY_S` 1.5, `_TURN_DEG` 60, `_STOP_RPM` 0, `_TAIL_S` 6, steering at 2 Hz on the nominal hull |
| `CROSSING_UNESCAPABLE_FRAC` | **0.20** | F63, A22: drawn once per sample, labelled `scenario.crossing_escapable = False` |
| `CONFINED_CT_HALF_DEG` | **10°** | F61, A21 **decided** (option 1): overtaking, being-overtaken and null targets are drawn within ±10° of the channel direction (classifier bands unchanged) |
| confined track check | **hull inside the corridor, every 0.5 s to CPA** (null: 15 s) | F61, A21: the generator rejects draws whose target would breach the channel before CPA |
| `clamp_to_corridor` | **nudge, not teleport** | F61, A21: heading to the channel tangent, moved inward by the breach + 0.1 m |
| being-overtaken DCPA floor | **`contact_free_dcpa(ct, k)` + 0.35 m** | F61, A21: hull clearance per draw (0.65 m parallel, ~1.0–1.2 m at 10°) plus `BEING_OVERTAKEN_FLOOR_MARGIN = D_SAFE`; replaces the 1.0 m centre floor; 20 % below, labelled |
| `CORRIDOR_BENDS` | **False** | F59, your call (2026-09-16): no bends in any curriculum stage; `CORRIDOR_BEND_FRACTION` 0.40 → 0; Tier A's two bend cases withdrawn (34 → 32) |
| `STRAIGHT_REFERENCE_PATH` | **True** | F59: the path's Rule 9(a) offset is constant, `offset_frac · ½ · min(W)`, so a varying-width corridor no longer bends the path; `r_path` ≡ 0 |
| `N_SWITCH_STEPS` | **retired** (value kept, 2 steps) | F58, A20 **decided** (option 1): an engaged encounter keeps the class, crossing side and turn sense latched at engagement until it clears; a class switch no longer re-engages |
