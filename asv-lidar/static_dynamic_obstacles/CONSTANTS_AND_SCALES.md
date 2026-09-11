# CONSTANTS AND SCALES — Paper 3

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
| `UPDATE_RATE` | 0.1 s | 10 Hz, matches the field control loop |
| `MAX_EPISODE_STEPS` | 700 | 70 s cap — **verify against the longest corridor** (03 §8) |
| `MAP_WIDTH` / `MAP_HEIGHT` | 10.0 / 25.0 m | **O4 resolved** |
| `CORRIDOR_WIDTHS_M` | 10, 8, **7**, 6, 5, 4, 3.5 m | Study 1 sweep |
| `PREDICTED_THRESHOLDS_M` | 6.52 / 6.02 / 4.78 / 3.66 m | per-class, 02a §2.2 |
| `HEAD_ON_WALL_CLEARANCE` | 0.65 m | `TODO(05)` |

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

| Symbol | Value | Status |
|---|---|---|
| `CRUISE_RPM` | 12.0 | Paper 2 |
| `RPM_STAGE` | 1 → (±3, 9, 15) | curriculum entry; **stage 4 is the endpoint** |
| `REVERSE_AVAILABLE` | False | `TODO(03)` — capability unverified |
| **`U_REF`** | **1.14 m/s** | **measured** (T1) |
| `ship.THRUST_CAL` | 0.3751 | calibration, `TODO(05)` |

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
| `CLUSTER_EPS` | 0.35 m | `TODO(decision)` |
| `CLUSTER_MIN_POINTS` | 4 | `TODO(decision)` |
| `TRACK_GATE_DIST` | 0.80 m | `TODO(decision)` |
| `TRACK_MAX_MISSES` / `TRACK_MIN_HITS` | 5 / 3 steps | |
| `KF_PROCESS_NOISE_ACCEL` | 0.10 m/s² | `TODO(05)` |
| `KF_MEAS_NOISE_POS` | 0.05 m | `TODO(05)` |
| `DYNAMIC_SPEED_ON` / `_OFF` | 0.15 / 0.08 m/s | `TODO(05)` |
| `DYNAMIC_HOLD_STEPS` | 5 steps | |

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
demonstrates the mechanism: 0.02 m/step of drift is enough to misclassify a
fixed object as dynamic at the current threshold.

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
| `r_bnd` boundary | 3.00 | `[-1,0]` | hard constraint, above the COLREGs group |
| `r_dom` target domain | 2.50 | `[-1,0]` | ground truth, centre-to-centre |
| `r_obs` static obstacle | 2.20 | `[-1,0]` | **shifted** exponential, zero past 2 m |
| `r_col` COLREGs group | 1.80 | `[-1,0]` | clipped to unit range *before* the weight |
| `r_pf` path following | 0.60 | `[-1,0]` | width-normalised on `W_local` |
| `r_prog` progress | 0.30 | `[-1,+1]` | telescoping arclength (`R-9`) |
| `r_smooth` smoothness | 0.10 | `[-1,0]` | `κ_δ` derived from the actuator |
| `r_exist` existence | 0.05 | `−1` | suspended by `R-5` only |

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
| `\|r_collision\| > w_col · 100` | a compliant collision scoring better than a non-compliant near-miss (`02 §5`) |
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
N_REF_PROG = L_REF_PATH / (U_REF · Δt) = 20.0 / (1.14 × 0.1) = 175.4
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

### 12.4 Speed references

| | Value | When |
|---|---|---|
| `U_REF` | 1.14 m/s | nominal, measured (T1) |
| `U_ref_eff`, `R-2` | 0.456 m/s | give-way, compliant alteration inadmissible |
| `U_ref_eff`, `R-5` | `max(u_TS, 0.20)` | overtaking, port pass does not fit |

`R-5` also zeroes the existence cost. It is the **one** place that happens, and
it is gated on a geometric predicate rather than on CRI, which keeps it
consistent with `R-6` and avoids the degenerate-policy risk `02 §4.4` warns
about. Without it the narrow overtaking case is not a test of COLREGs reasoning
but a test of whether the agent tolerates an unwinnable reward — and it would
resolve it by overtaking anyway.
