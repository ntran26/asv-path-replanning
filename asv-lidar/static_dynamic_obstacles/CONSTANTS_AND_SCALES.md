# CONSTANTS AND SCALES — Paper 3

**Revision 2** — two-vessel repositioning. One dynamic target, five encounter
classes, ship domain resolved to 2.0 / 1.0 / 0.75 × Lpp, O4/O5/O6 closed.

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

**42 constants are unresolved.** §11 lists them all with the placeholder in
force. Placeholders are chosen to make the code run, not to look finished.

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
| `CORRIDOR_WIDTHS_M` | 10, 8, 6, 5, 4, 3.5 m | Study 1 sweep |
| `HEAD_ON_WALL_CLEARANCE` | 0.65 m | `TODO(05)` |

**O4 resolved (03 §5): simulation matches the basin.** Maximum corridor width
10 m = 20 breadths, so every simulated width is physically reproducible — a
meaningful strengthening of the field-validation argument. The unconfined
reference case comes instead from the open-water "Around the Clock" variant,
which is unbounded by construction.

The sweep in breadths, and whether a compliant head-on fits:

| Width | Breadths | Compliant head-on fits? |
|---|---|---|
| 10 m | 20 B | Yes, comfortably |
| 8 m | 16 B | Yes |
| 6 m | 12 B | Yes |
| 5 m | 10 B | Marginal |
| 4 m | 8 B | Tight |
| 3.5 m | 7 B | **No** — below threshold |

Minimum width for a compliant port-to-port head-on:
`2 × DOMAIN_LATERAL + 2 × HEAD_ON_WALL_CLEARANCE` = 2.36 + 1.30 = **3.66 m
(7.3 B)**, bracketed between the 4.0 m and 3.5 m levels.
`tests/test_cpa_cri.py::test_the_width_sweep_brackets_the_head_on_threshold`
checks this arithmetic, and will fail if the ship domain moves without the
sweep moving with it — which it will, once 05 lands.

## 3. Actuation

| Symbol | Value | Status |
|---|---|---|
| `CRUISE_RPM` | 12.0 | Paper 2 |
| `RPM_STAGE` | 1 → (±3, 9, 15) | **`TODO(02)`** |
| `U_CRUISE` | 0.55 m/s | `TODO(05)` |

**`RPM_STAGE` is a live decision, not a carry-over.** Rule 8(e) permits
slackening speed or stopping, and in a narrow channel speed reduction is
frequently the *only* admissible action when there is no room for a course
alteration. Paper 2's stage-1 authority is ±3 RPM around cruise. If the agent
cannot meaningfully slow down, a lawful manoeuvre has been removed from its
repertoire and a reviewer may notice (02 §4.4).

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

**There is no IMU on the platform.** u, v and r are differentiated from a noisy
pose, so the `ego` branch carries field error that simulation did not model at
all — a sim-to-real gap in the *observation*, not just in the dynamics.

## 8. Ship domain — RESOLVED (provisional)

| Direction | Multiple | Metres |
|---|---|---|
| Ahead | 2.00 · Lpp | 3.14 |
| Astern | 1.00 · Lpp | 1.57 |
| Abeam (each side) | 0.75 · Lpp | 1.18 |

Lateral footprint 2.36 m, about 24% of the widest channel.

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
closest approach in a channel is overwhelmingly a beam-on passing geometry. The
alternative is `sqrt(fore × lateral)`. `TODO(decision)`.

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
| `HEAD_ON_BEARING_HALF_DEG` | 8.0° | `TODO(decision)` — source is 5.0° |
| `HEAD_ON_CT_HALF_DEG` | 8.0° | `TODO(decision)` |
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
2. **The head-on band is widened** from ±5° to ±8°, the midpoint of the ±6–10°
   of common practice. 01 §5.3 marks the value and its justification `[TBC]`.
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

| # | Symbol | Placeholder | Marker |
|---|---|---|---|
| 1 | `HEAD_ON_WALL_CLEARANCE` | 0.65 m | `TODO(05)` |
| 2 | `RPM_STAGE` | 1 (±3 RPM) | `TODO(02)` |
| 3 | `U_CRUISE` | 0.55 m/s | `TODO(05)` |
| 4 | `LIDAR_DROPOUT_P` | 0.0 | `TODO(05)` |
| 5 | `LIDAR_NO_RETURN_GRAZING_DEG` | 0.0 | `TODO(05)` |
| 6 | `LIDAR_AFT_MASK_HALF_DEG` | 0.0 | `TODO(05)` |
| 7 | `BOUNDARY_GATE_MARGIN` | 0.30 m | `TODO(05)` |
| 8–10 | `BOUNDARY_POSE_NOISE_XY` / `_HEADING_DEG` / `_WALK` | 0.0 | `TODO(05)` |
| 11 | `CLUSTER_EPS` | 0.35 m | `TODO(decision)` |
| 12 | `CLUSTER_MIN_POINTS` | 4 | `TODO(decision)` |
| 13 | `TRACK_GATE_DIST` | 0.80 m | `TODO(decision)` |
| 14–15 | `KF_PROCESS_NOISE_ACCEL` / `KF_MEAS_NOISE_POS` | 0.10 / 0.05 | `TODO(05)` |
| 16–17 | `DYNAMIC_SPEED_ON` / `_OFF` | 0.15 / 0.08 m/s | `TODO(05)` |
| 18 | `DETECTION_DROPOUT_P` | 0.0 | `TODO(05)` |
| 19 | `TRACK_VELOCITY_NOISE` | 0.0 | `TODO(05)` |
| 20–21 | `EGO_SPEED_NOISE` / `EGO_YAW_RATE_NOISE_DPS` | 0.0 | `TODO(05)` |
| 22–24 | `DOMAIN_FORE` / `_AFT` / `_LATERAL` | 2.0 / 1.0 / 0.75 Lpp | `TODO(05)` |
| 25 | `DOMAIN_RADIUS_DCPA` | lateral semi-axis | `TODO(decision)` |
| 26 | `CRI_DCPA_SCALE` | 4.0 m | `TODO(decision)` |
| 27–28 | `CRI_TCPA_SCALE_BEFORE` / `_AFTER` | 20.0 / 6.0 s | `TODO(decision)` |
| 29 | `CRI_ED_SCALE` | 5.0 m | `TODO(decision)` |
| 30–31 | `CRI_BOW_CROSSING_GAIN` / `_HALF_DEG` | 1.3 / 45° | `TODO(decision)` |
| 32–33 | `HEAD_ON_BEARING_HALF_DEG` / `HEAD_ON_CT_HALF_DEG` | 8.0° | `TODO(decision)` |
| 34 | `BEING_OVERTAKEN_SPEED_MARGIN` | 0.10 m/s | `TODO(decision)` |
| 35 | `TCPA_CLIP` | 60.0 s | `TODO(decision)` |
| 36 | `DCPA_CLIP_DOMAINS` | 10.0 | `TODO(decision)` |
| 37 | `USE_RECURRENCE` | False | `TODO(04)` |
| 38–40 | `R_COLLISION` / `R_TIMEOUT` / `R_GOAL` | −1000 / −1000 / +50 | `TODO(02)` |
| 41 | `TARGET_SPEED_RANGE` | (0.30, 0.80) m/s | `TODO(03)` |
| 42 | `NO_TARGET_EPISODE_PROB` | 0.25 | `TODO(04)` |

The three terminal payoffs are structural, not shaping — they exist only so the
environment can be stepped before 02 lands. **02 §4.1 already states collision
−200 and goal +100 with timeout via value bootstrapping**, so the values in
force are Paper 2's and are known to be wrong; they are left marked rather than
silently changed, because the whole reward is 02's to design.
