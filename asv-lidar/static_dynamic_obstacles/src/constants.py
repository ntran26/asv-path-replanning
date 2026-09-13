"""Single source of truth for every Paper 3 constant.

**Revision 2** — repositioned to two-vessel encounters. One dynamic target,
five encounter classes, observation reduced to 56 dims.  Supersedes the
three-slot version.

Rules for this file (KICKOFF_01_PERCEPTION.md §5):

* Every unresolved value appears **here**, with a `TODO(...)` marker, and
  nowhere else.  No consumer may bury a magic number in a function body.
* A placeholder must make the code run.  It must not make the code look
  finished.  Anything marked TODO is a value that has not been decided, not a
  value that has been decided and left untidy.
* `TODO(05)` is owned by `planning/05_VESSEL_MODEL_AND_SIM2REAL.md`.
  `TODO(02)` is owned by `planning/02_REWARD_AND_COLREGS.md`.
  `TODO(03)`, `TODO(04)` likewise.
  `TODO(decision)` needs a call that no open item currently covers.

Vessel hydrodynamics and the collision hull live in `ship.py`, which is a
verbatim Paper 2 carry-over and is 05's territory.  This file holds the task,
the sensor, the perception stack and the observation scales.
"""

from __future__ import annotations

import numpy as np

from ship import (MAX_RUD_ANGLE, MAX_RUD_RATE_DPS,  # noqa: F401
                  VESSEL_LENGTH, VESSEL_WIDTH)

# ===========================================================================
# 1. Vessel reference lengths
# ===========================================================================
# `ship.VESSEL_LENGTH` (1.725 m) is the **LOA**, and it is what the collision
# hull and the LiDAR mount offset are built from.  Do not use it for ship-domain
# or CRI scaling: the literature those come from is written in Lpp.
LOA = float(VESSEL_LENGTH)               # 1.725 m, overall
LBP = 1.57                               # length between perpendiculars
BREADTH = float(VESSEL_WIDTH)            # 0.50 m — the unit for channel width

# ===========================================================================
# 2. Simulation and workspace
# ===========================================================================
UPDATE_RATE = 0.1                        # control period [s] -> 10 Hz
# 900 steps / 90 s (04a §4.1, re-verified in 03a §4.5).  Was 700.
#
# Rule 8(e) speed reduction is a *designated compliant behaviour* under 02 §4.4:
# a correct narrow-channel give-way may involve slowing to 0.2 m/s for 20 s or
# more.  A horizon tight enough to turn compliant slowing into a timeout puts
# the horizon in direct conflict with the reward design -- it would penalise the
# behaviour the paper is trying to elicit.
#
# Fixed rather than width-conditional, so timeout rates stay comparable across
# the Study 1 sweep.
MAX_EPISODE_STEPS = 900                  # 90 s episode cap
RENDER_FPS = 10
RENDER_SCALE = 25                        # pixels per metre

# O4 RESOLVED (03 §5): simulation matches the basin, so every simulated width is
# physically reproducible.  Maximum corridor width 10 m = 20 breadths.
MAP_WIDTH = 10.0
MAP_HEIGHT = 25.0

# Study 1 — channel-width sweep, parameterised in **breadths** so the sweep and
# the precedence thresholds are scale-explicit (03 §4).
# 02a §11.3 adds 7.0 m (14 B): the six original levels bracket all four
# predicted transitions, but the crossing threshold (6.52 m) and the
# centreline head-on threshold (6.02 m) land in adjacent brackets and could not
# be separated.  7 m splits them, and it is the level carrying N2's headline
# ordering result.
CORRIDOR_WIDTHS_M = (10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.5)

def predicted_thresholds(d_abeam=None, c_wall=None, breadth=None) -> dict:
    """Per-class width transitions in metres, derived exactly as 02a §2.2 does.

    **Computed, not a literal list** (02b C1).  Every input is `TODO(05)` -- they
    all move when the turning circle is identified -- so re-deriving is one call,
    and the sweep levels can be re-chosen against fresh numbers in one sitting
    rather than by hand-editing four figures that silently go stale.

    With `d_req = 2 * d_abeam` (two identical vessels abeam) and `c_wall` the
    clearance each side, 02a §2.2's derivations are:

      crossing            centreline path, `W/2 - (c_wall + B/2) >= d_req`
      head-on, centreline TS holds the middle, so the OS produces the WHOLE
                          separation alone: `2*d_req + 2*c_wall`
      head-on, compliant  TS keeps its own side, so `d_req + 2*c_wall`
      overtaking          TS at `W/2 - c_wall`, OS `d_req` to port of it, OS
                          needs `c_wall` to the port wall, hulls counted:
                          `d_req + 2*c_wall + B`, times a prudence factor

    Reproduces 02a's 6.52 / 6.02 / 4.78 / 3.66 at the provisional domain.
    """
    d_abeam = DOMAIN_LATERAL if d_abeam is None else float(d_abeam)
    c_wall = HEAD_ON_WALL_CLEARANCE if c_wall is None else float(c_wall)
    b = BREADTH if breadth is None else float(breadth)
    d_req = 2.0 * d_abeam
    return {
        "crossing": 2.0 * (d_req + c_wall + 0.5 * b),
        "head_on_centreline_target": 2.0 * d_req + 2.0 * c_wall,
        "overtaking": OVERTAKING_PRUDENCE * (d_req + 2.0 * c_wall + b),
        "head_on_compliant_target": d_req + 2.0 * c_wall,
    }


# (The snapshot `PREDICTED_THRESHOLDS_M` is built in §8, once the domain the
# function reads is defined.)


def widths_in_breadths(widths=CORRIDOR_WIDTHS_M) -> tuple:
    """Channel widths expressed in ship breadths: (20, 16, 12, 10, 8, 7)."""
    return tuple(round(w / BREADTH, 2) for w in widths)


# Minimum width admitting a compliant port-to-port head-on: two non-overlapping
# ship domains abeam (2 x 2 x DOMAIN_ABEAM) plus wall clearance each side.
# 03 §5 puts this at ~3.66 m (7.3 B) and brackets the transition between the
# 4.0 m and 3.5 m sweep levels.
# TODO(05): recompute once the ship domain is derived from the turning-circle
# data — the threshold moves with the domain.
HEAD_ON_WALL_CLEARANCE = 0.65            # m each side, TODO(05)

# 02a §2.2 applies a prudence factor to the geometric overtaking width: a pass
# that only just fits is not one a prudent mariner attempts.
OVERTAKING_PRUDENCE = 1.15               # TODO(05): moves with the domain

# Reference path.  03 owns the corridor generator (variable width, bends,
# off-centre paths).  Until it lands, the boundary branch is an affine function
# of cross-track error and must not be ablated (01 §3.3).
PATH_MODE = "straight"                   # "straight" | "curve" | "mixed"
CURVE_PROB = 0.0
LOOKAHEAD_FRACTION = 0.25

# Curvature below this reads as straight.  `ReferencePath.points` is float32
# (Paper 2's choice), and differencing closely-spaced float32 vertices leaves
# ~2e-5 1/m of noise on a nominally straight path -- a 49 km turn radius.
# Without a deadband that noise would leak into `r_path` and make 02a `R-8`'s
# `r - r_path` non-zero everywhere for no physical reason.
# 1e-4 1/m is a 10 km radius: unambiguously straight, and three orders below
# any bend that fits in a 25 m basin.
CURVATURE_EPS = 1e-4                     # 1/m

VERTICAL_PATH_PROB = 0.70
START_Y = 2.0
GOAL_Y_MARGIN = 3.0
START_X_MARGIN_FRAC = 0.25
START_X_MARGIN_MIN = 2.0

# Goal acceptance region.
GOAL_RADIUS = 0.5
GOAL_ALONG_DIST = 1.25
GOAL_CTE_RADIUS = 1.60

# ===========================================================================
# 3. Actuation
# ===========================================================================
# action = [rudder, throttle], both in [-1, 1].
CRUISE_RPM = 12.0
FIXED_RPM = False

# Propulsion authority WIDENS -- resolved (03 §6, 02 §4.4).  Rule 8(e) speed
# reduction is the designated fallback whenever a compliant course alteration
# would push the vessel into the boundary, so the agent must be able to slow
# substantially and ideally stop.
#
# Staged through the curriculum as in Paper 2, but **stage 4 must be exposed by
# the final stage** -- it is the only one that reaches 0 RPM.  Stage 1 remains
# the curriculum entry point.
RPM_STAGE = 1                            # curriculum entry; stage 4 is the endpoint
RPM_STAGES = {
    1: (3.0, 9.0, 15.0),
    2: (4.0, 8.0, 16.0),
    3: (6.0, 6.0, 18.0),
    4: (12.0, 0.0, 24.0),
}
RPM_DELTA, RPM_FLOOR, RPM_CEIL = RPM_STAGES[RPM_STAGE]

# 02a §10.5: with reverse the vessel can "take all way off"; without it, only
# slacken.  Default False deliberately -- do not flip it on a datasheet, because
# 05 must identify the reverse regime or the simulator extrapolates into an
# unmodelled envelope.
REVERSE_AVAILABLE = False                # TODO(03): capability unverified

# ---------------------------------------------------------------------------
# U_REF -- THE reference speed.  One constant, consumed by everything.
# ---------------------------------------------------------------------------
# **Measured** (02b T1): mined from the 18 usable Paper 2 field logs, which
# carry commanded RPM alongside the localiser pose in their `#ACTION` records.
# Speed over ground differentiated from the pose track over the last 60% of each
# run, so the acceleration ramp is excluded.  Median 1.14 m/s at 12 RPM, spread
# 0.56-1.25, remarkably consistent across trials.
#
#   Froude at 1.14 m/s, LBP 1.57 m  ->  Fr = 0.29, a displacement hull.
#
# This supersedes three earlier figures, all of them wrong:
#   0.55 m/s  -- an unsourced placeholder this file previously carried and
#                labelled "measured from the field trials".  It was not
#                measured.  02b C2 then adopted it as ground truth.
#   0.80 m/s  -- 02a §1's assumption
#   1.77 m/s  -- what the *simulator* produced, i.e. Fr 0.45, semi-planing
#
# 02b C2's direction was right -- the simulator is too fast and the thrust map
# needs calibrating -- but by ~1.55x, not ~3.2x.  `ship.THRUST_CAL` carries the
# correction so that steady surge at CRUISE_RPM equals this value.
# TODO(05): confirm against a dedicated straight-line run; the field logs are
# short avoidance manoeuvres in a 25 m basin and never hold a true steady state.
U_REF = 1.14                             # m/s at CRUISE_RPM, measured

# Retained for readability at call sites.  Same number, by construction.
U_CRUISE = U_REF

# `U_nom` in 03a/04a.  **One symbol, and the whole timing structure hangs off
# it** -- TCPA ranges, spawn geometry, the episode horizon, classification
# latency, the Rule 8(a) early-action metric and every width threshold that
# carries an exposure term.
#
# **F24 -- 03a §1.1 decides 0.55 m/s and attributes it to "the field
# measurement". It is not a measurement.** It is the same unsourced placeholder
# F20 recorded: it entered as a number in the first `constants.py` and has now
# been cited as authoritative in a third document. T1 mined the retained logs
# and found a median of **1.14 m/s at 12 RPM across 18 logs** (Fr 0.29).
#
# Left at the measured value, and every 03a/04a quantity derived from it rather
# than written down, so the decision is one constant either way.  See
# `froude()` and `full_scale()` below for the two columns the choice produces.
U_NOM = U_REF

GRAVITY = 9.81


def froude(speed: float = None, length: float = None) -> float:
    """`Fr = U / sqrt(g * Lpp)` -- the scale-invariant speed (03a §1.1)."""
    u = U_NOM if speed is None else float(speed)
    lbp = LBP if length is None else float(length)
    return float(u / np.sqrt(GRAVITY * lbp))


def full_scale(lam: float = 50.0, speed: float = None) -> dict:
    """Froude-scaled full-scale equivalents at geometric scale `lam`.

    03a §1.1 wants this table in the paper, and it is the strongest single
    answer to "this is a 1.7 m model boat": at lam = 50 it becomes a 78.5 m
    vessel in a 175-500 m fairway, which is the scale at which Rule 9 is
    actually argued about.

    Speeds scale as `sqrt(lam)`, times as `sqrt(lam)`, lengths as `lam`.
    """
    u = U_NOM if speed is None else float(speed)
    root = float(np.sqrt(lam))
    return {
        "lambda": float(lam),
        "Lpp_m": LBP * lam,
        "speed_mps": u * root,
        "speed_kn": u * root * 1.94384,
        "froude": froude(u),
        "corridor_wide_m": 10.0 * lam,
        "corridor_narrow_m": 3.5 * lam,
        "horizon_s": MAX_EPISODE_STEPS * UPDATE_RATE * root,
    }


# ===========================================================================
# 4. Raw LiDAR (RPLidar C1)
# ===========================================================================
# Confirmed against all 30 logs in `field_deployment/` (5597 scans): 720 bins
# per revolution in every scan, values in decimetres spanning 10..153
# (1.0..15.3 m).  See PORTING_MANIFEST.md F6.
LIDAR_BEAMS = 720
LIDAR_SWATH = 360.0
LIDAR_BEAM_RES_DEG = LIDAR_SWATH / LIDAR_BEAMS      # 0.5 deg, exactly
LIDAR_RANGE = 16.0                       # max range [m]

# The C1 does not return anything closer than 1 m -- confirmed across all logs,
# where the smallest non-zero value is 10 dm.  Paper 2 reported ranges down to 0.
# With the sensor at the bow of a 1.57 m hull, a target alongside inside 1 m is
# invisible to the real sensor and was fully visible in Paper 2's simulator.
LIDAR_MIN_RANGE = 1.0

# Field logs show a mean of 506 of the 720 bins carrying a return, but 96.5% of
# the empty bins lie in contiguous runs longer than 3 bins -- they are
# no-return/out-of-range arcs, not angular under-sampling.  So the simulator
# keeps all 720 beams at 0.5 deg and models the no-return process separately.
LIDAR_DROPOUT_P = 0.0                    # TODO(05): isolated per-beam dropout
LIDAR_NO_RETURN_GRAZING_DEG = 0.0        # TODO(05): incidence angle below which
                                         #   a surface stops returning

# Aft self-occlusion.  01 §2.3 item 2 assumes a blind or degraded arc exists.
# It is NOT detectable in the existing logs: no bin is zero in more than 98% of
# scans, and the peak zero-rate bearing wanders between logs (108..359 deg), so
# it tracks the scene rather than the mount.  Settling it needs a static-spin
# recording with the vessel stationary in a known surround.
# Half-width of the masked arc centred on dead astern; 0.0 = no mask.
# This one gates the **being-overtaken** class: if the tracker is trained to see
# astern and the real mount cannot, that class fails in the field for reasons
# unrelated to the policy (01 §2.3).
LIDAR_AFT_MASK_HALF_DEG = 0.0            # TODO(05): needs a static-spin log

# ===========================================================================
# 5. Sector pooling  (01 §2.2)
# ===========================================================================
# `c_t` is forward-biased and carries **static obstacles only**.  Borders are
# gated out (§3) and the dynamic target goes through the target branch (§5).
# The aft 90 deg is reserved for the tracker.
POOL_SWATH_HALF_DEG = 135.0              # pooled span is +/-135 deg

# Non-uniform allocation, from outboard port to outboard starboard.
POOL_BANDS = (
    (-135.0, -90.0, 22.5),               # port outer:  2 sectors, 45 beams each
    (-90.0, -45.0, 11.25),               # port mid:    4 sectors, 22-23 beams
    (-45.0, 45.0, 6.0),                  # bow:        15 sectors, 12 beams
    (45.0, 90.0, 11.25),                 # stbd mid:    4 sectors, 22-23 beams
    (90.0, 135.0, 22.5),                 # stbd outer:  2 sectors, 45 beams
)
LIDAR_SECTORS = 27

# Safety-adjusted width used by Algorithm 1 (feasibility pooling).  Matches the
# inflated collision hull, exactly as in Paper 2.
FEASIBILITY_SAFE_WIDTH_MARGIN = 0.15     # = ship.HULL_MARGIN


def sector_edges() -> np.ndarray:
    """Sector boundaries in degrees, ascending, length LIDAR_SECTORS + 1.

    Built from POOL_BANDS so the allocation has exactly one definition.
    """
    edges = [POOL_BANDS[0][0]]
    for lo, hi, width in POOL_BANDS:
        n = int(round((hi - lo) / width))
        edges.extend(lo + width * (k + 1) for k in range(n))
    return np.asarray(edges, dtype=np.float64)


# Structural invariants.  Cheap, and they catch a mis-edited band table at
# import time rather than in a training run.
_EDGES = sector_edges()
assert len(_EDGES) == LIDAR_SECTORS + 1, f"{len(_EDGES) - 1} sectors, expected {LIDAR_SECTORS}"
assert _EDGES[0] == -POOL_SWATH_HALF_DEG and _EDGES[-1] == POOL_SWATH_HALF_DEG
assert np.all(np.diff(_EDGES) > 0.0), "sector edges must be strictly increasing"

# ===========================================================================
# 6. Boundary branch  (01 §3)
# ===========================================================================
# Virtual range scan ray-cast against the known channel polygon from the
# *estimated* pose, then normalised to closeness identically to c_t.
#
# This is an architectural argument, not a workaround (01 §3.1): in a real
# narrow channel the navigable limit is usually a charted depth contour, a
# buoyed line or a regulatory limit -- none of which a LiDAR can see.  The basin
# reproduces that exactly, because the sensor sits above the pool edge and
# registers the facility walls 1-2 m beyond it instead.
BOUNDARY_BEARINGS_DEG = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
BOUNDARY_RAYS = len(BOUNDARY_BEARINGS_DEG)
BOUNDARY_MAX_RANGE = LIDAR_RANGE         # same normaliser as c_t, deliberately

# Field-side gating margin (01 §3.4).  O5 RESOLVED: software gating, not a
# physical barrier -- the facility walls carry the fixed geometric features
# (recessed doorways, protruding benches) that are the only along-track
# constraint available to scan-to-map localisation in 05, and a barrier would
# occlude them.  So: localise on the FULL scan including the walls, then apply
# this gate afterwards for the tracker only.  The walls are a liability for
# tracking and an asset for localisation, and the pipeline treats them as both.
#
# Gating is mandatory rather than preferable: during trials, operators standing
# on the deck sit at scan height and move.
BOUNDARY_GATE_MARGIN = 0.30              # m, TODO(05) for the localisation input

# Pose noise injected into the boundary raycast so training does not see a
# noiseless map (01 §3.3).  Also a Study 2 sweep axis (04 §6).
# TODO(05): all three are 0.0, i.e. the sim-to-real gap 01 §3.3 warns about is
# currently WIDE OPEN.  This must not reach a headline training run at 0.0.
BOUNDARY_POSE_NOISE_XY = 0.0             # m, 1-sigma,        TODO(05)
BOUNDARY_POSE_NOISE_HEADING_DEG = 0.0    # deg, 1-sigma,      TODO(05)
BOUNDARY_POSE_NOISE_WALK = 0.0           # m/step random walk, TODO(05)

# ===========================================================================
# 7. Target tracking pipeline  (01 §4)  -- the headline contribution N1
# ===========================================================================
# Clustering of gated returns.
CLUSTER_EPS = 0.35                       # m, approved 02b §2
# Suspension lines run diagonally across the basin and descend toward their
# anchors, so near the pool edges they cross the scan plane.  A taut rope
# returns on one or two beams.  The minimum-points threshold must reject them
# without rejecting genuine small obstacles (01 §8, 03 §4a).
CLUSTER_MIN_POINTS = 4                   # >= 3 to clear a rope; approved 02b §2

# Track association.  Nearest-neighbour is sufficient at one target (01 §4).
# Tied to the maximum plausible inter-frame displacement, so it rescales with
# the speed calibration instead of drifting out of step with it (02b §2).
TRACK_GATE_DIST = max(2.5 * U_REF * UPDATE_RATE, 0.30)   # m
TRACK_MAX_MISSES = 5                     # steps before a track is dropped
TRACK_MIN_HITS = 3                       # steps before a track is published

# Constant-velocity Kalman filter.
KF_PROCESS_NOISE_ACCEL = 0.10            # m/s^2, TODO(05)
KF_MEAS_NOISE_POS = 0.05                 # m,     TODO(05)
KF_INIT_VEL_VAR = 0.50                   # (m/s)^2

# Static vs dynamic split, with hysteresis so a track cannot chatter.
#
# **This threshold is set by localisation quality, not by obstacle behaviour**
# (01 §4 step 6, 03 §4a).  Field obstacles are suspended panels, confirmed from
# video to hang stably, so apparent motion of a static object comes almost
# entirely from ego-pose error -- which affects every object in the scan
# identically.  Set from measured pose noise (05 §4) and retighten as
# registration improves.
#
# Bias toward UNDER-detection: promoting a static panel to a target ship is a
# false positive with COLREGs consequences.
DYNAMIC_SPEED_ON = 0.15                  # m/s, static -> dynamic, TODO(05)
DYNAMIC_SPEED_OFF = 0.08                 # m/s, dynamic -> static, TODO(05)
DYNAMIC_HOLD_STEPS = 5                   # steps a classification must persist

# --- Study 2 degradation axes (01 §4.1, 04 §6) -----------------------------
# Exposed as environment config so the sweep in 04 can drive them.  Every one
# is nominal-zero here; Study 2 sweeps each independently, then jointly.
DETECTION_DROPOUT_P = 0.0                # per-track per-step miss, TODO(05)
TRACK_VELOCITY_NOISE = 0.0               # m/s 1-sigma on the estimate, TODO(05)

# Ego velocity error.  **IMU CONFIRMED (05 §4.7)** -- one will be added, logging
# raw gyro and accelerometer at 100 Hz+, time-synced to the LiDAR.  That changes
# the character of this gap rather than closing it:
#   r  -- now measured directly by the gyro, so the residual is the sensor noise
#         floor rather than pose-differentiation error.  Much smaller, and the
#         yaw-rate criterion 02 §4.2 relies on becomes directly measurable in the
#         field instead of inferred.
#   u,v -- "largely rescued" by the accelerometer, but still fused rather than
#         measured, so a residual remains.
# Scan-to-map supplies drift-free absolute pose at 10 Hz; the IMU fills in
# between.  Both magnitudes still come from 05.
EGO_SPEED_NOISE = 0.0                    # m/s 1-sigma on u and v, TODO(05)
EGO_YAW_RATE_NOISE_DPS = 0.0             # deg/s 1-sigma on r, TODO(05): gyro noise floor

# ===========================================================================
# 8. Ship domain  (01 §5.2)  -- RESOLVED, provisional
# ===========================================================================
# Chun et al.'s 3*Lpp fore/aft and 1*Lpp abeam gives 4.71 m fore-aft at
# LBP = 1.57 m, leaving almost no room in a 10 m channel and none at all in the
# 3.5 m sweep level.  01 §5.2 resolves it to a compressed asymmetric domain:
# HARD FLOOR on the abeam extent (02b §3.1), and it is not a tuning choice.
#
# The C1 returns nothing inside 1 m.  If 05's turning-circle identification
# produces an abeam domain below that, **the ship domain falls entirely inside
# the sensor's blind zone** -- and because `R-1` evaluates domain intrusion on
# ground-truth geometry, the agent would be penalised for intrusions it is
# physically incapable of perceiving, in simulation and in the field alike.
# `r_dom` would stop being a shaping signal and become an unlearnable term.
#
# A domain smaller than the sensor can resolve is not a domain, it is a blind
# spot.
DOMAIN_ABEAM_FLOOR = LIDAR_MIN_RANGE + 0.5 * BREADTH     # 1.25 m

# **F21 RESOLVED -- the floor is applied.**  02a §1 states the abeam extent as
# `0.75*Lpp = 1.18 m`, which does not clear this floor.  02b §3.1 anticipates
# exactly that case and says what to do: "If the measured manoeuvring
# performance implies a smaller one, the domain is floored at 1.25 m and the
# paper states why."  The provisional value is smaller, so the floor binds.
#
# Applying it is executing 02b's decision, not overriding 02a's: 02b is the
# later document, declares itself a companion that amends 02a §1, and §3.1 is
# stated as a decision rather than a recommendation.  Consequence, which belongs
# in the paper: `d_abeam` becomes 0.796*Lpp, `d_req` 2.50 m, and all four Study 1
# thresholds move up by 14-29 cm (see `predicted_thresholds()`).  02b C1's
# bracket collision survives -- 6.80 m and 6.30 m still share the (6, 7)
# bracket, so the sweep stays at seven levels.
DOMAIN_FORE = 2.00 * LBP                 # 3.14 m, TODO(05): provisional
DOMAIN_AFT = 1.00 * LBP                  # 1.57 m, TODO(05): provisional
DOMAIN_LATERAL = max(0.75 * LBP, DOMAIN_ABEAM_FLOOR)     # 1.25 m, TODO(05)


def check_domain(d_abeam=None) -> list:
    """Validator for the ship domain.  Returns a list of problems, empty if ok.

    02b §3.1 asks for the floor to be asserted in the config validator, and
    `reward.config.RewardConfig` now does raise on it.  This function returns
    rather than raises so that a caller sweeping candidate domains (05, after
    the turning-circle identification) can enumerate the problems with a
    proposed value instead of catching an exception per candidate.
    """
    d = DOMAIN_LATERAL if d_abeam is None else float(d_abeam)
    problems = []
    if d < DOMAIN_ABEAM_FLOOR:
        problems.append(
            f"d_abeam {d:.3f} m is below the sensor-resolution floor "
            f"{DOMAIN_ABEAM_FLOOR:.3f} m (= LIDAR_MIN_RANGE + B/2): the domain "
            f"would sit inside the sensor blind zone and r_dom becomes "
            f"unlearnable (02b §3.1)")
    return problems
# Lateral footprint 2.36 m, about 24% of a 10 m channel.
#
# **The principle matters more than the numbers.**  These are a provisional
# INPUT.  The final values are an OUTPUT of 05: derive them from measured
# manoeuvring performance -- advance and tactical diameter from the
# turning-circle tests, stopping distance from the stop test -- so the domain is
# "sized to this vessel's demonstrated ability to avoid", which is the argument
# Thyri & Breivik make for confined water.  Do not defend them as a scaled copy
# of someone else's domain.  Szlapczynski & Szlapczynska (2017) is the reference
# for justifying the compression.
# TODO(05): finalise from the identified turning circle.

# DCPA is normalised by the domain radius rather than by metres (01 §6.1), which
# is undefined for an asymmetric domain.  Convention: the **lateral** semi-axis,
# because DCPA is a closest-approach distance and closest approach in a channel
# is overwhelmingly a beam-on passing geometry.
#
# RESOLVED (02b §2), and the resolution is that observation and reward normalise
# **differently on purpose**: this constant scales the observation feature only.
# The reward uses the directional `d_dom(beta)` evaluated at the target's actual
# bearing (02a §5.3) and gates `rho_t` on the constant `d_req`.  Documented
# rather than unified, because the two serve different jobs.
DOMAIN_RADIUS_DCPA = DOMAIN_LATERAL

# Snapshot of §2's `predicted_thresholds()` at the provisional domain, for
# reference and for the tests.  Recompute by calling the function after 05.
PREDICTED_THRESHOLDS_M = predicted_thresholds()

# ===========================================================================
# 9. Collision Risk Index  (01 §5.2, after Waltz & Okhrin 2023 §3.3)
# ===========================================================================
#   CR = 1                     if the TS is inside the OS ship domain
#   CR = max(CR_CPA, CR_ED)    otherwise
#
# APPROVED (02b §2.1).  The sensor-horizon anchoring below is adopted, and the
# supporting observation is worth stating in the paper: a 320 m ship with 2 NM
# of radar sees 11.6 hull lengths ahead; the Bluefin with 16 m of LiDAR sees
# 10.2.  **The perceptual horizon in ship lengths transfers almost exactly even
# though the absolute range does not** -- it is the channel that is small at
# model scale, not the sensing.  That is a stronger justification than
# ship-length scaling and it generalises to every other re-derived constant.
#
# Defusing note for the methods: `R-6` removed CRI from the reward, so these
# affect an observation feature only.  They are not safety-critical.
#
# Waltz & Okhrin scale their
# decay to 2 NM = 3704 m for a 320 m KVLCC2, i.e. 11.6 Lpp.  Scaled to
# LBP = 1.57 m that is 18.2 m -- LARGER THAN THE 16 m SENSOR HORIZON, so a
# straight re-derivation in ship lengths produces a risk that never decays
# within anything the vessel can see.  The constants below are therefore
# anchored to the **sensor horizon** instead of to ship lengths, which is a
# different choice from the one 01 §5.2 asks for and needs sign-off.
CRI_DCPA_SCALE = 4.0                     # m
CRI_TCPA_SCALE_BEFORE = 20.0             # s, approaching CPA
CRI_TCPA_SCALE_AFTER = 6.0               # s, past CPA
# Asymmetric by construction: risk must fall away quickly once the CPA is
# behind, which is the whole point of the two-rate form.

# CR_ED: plain Euclidean-distance risk.  This is the patch for the
# near-parallel failure mode (01 §5.1) and is NOT optional in a channel, where
# near-parallel geometry is the normal case rather than the exception.
CRI_ED_SCALE = 5.0                       # m

# Bow-crossing factor: inflates risk when the CPA would put the OS across the
# target's bow.
CRI_BOW_CROSSING_GAIN = 1.3
CRI_BOW_CROSSING_HALF_DEG = 45.0

# ===========================================================================
# 10. Encounter classifier -- FIVE classes  (01 §5.3, S4)
# ===========================================================================
# alpha = relative bearing OS->TS, CT = heading intersection angle, both deg.
# Baseline thresholds are Waltz & Okhrin Table 1 (after Xu et al. 2020) with the
# three modifications 01 §5.3 requires.

# Modification 2 (RESOLVED, 01 §5.3): the source band of +/-5 deg is tight
# enough that a small heading error flips the classification.  Widened to
# +/-10 deg, which is within common practice and gives the hysteresis room to
# work.
HEAD_ON_BEARING_HALF_DEG = 10.0          # was 5.0 in the source table
# 01 resolves "the head-on band" without separating bearing from heading.  Kept
# symmetric: courses within 10 deg of reciprocal count as head-on, which is the
# reading that matches the stated rationale (a small *heading* error).
HEAD_ON_CT_HALF_DEG = 10.0

# Sector boundaries shared with the crossing and overtaking classes.
CROSSING_STBD_MAX_DEG = 112.5
CROSSING_PORT_MIN_DEG = 247.5
OVERTAKING_CT_HALF_DEG = 67.5

# Modification 3: the "being overtaken" class.  Not in the source table --
# Waltz & Okhrin assume linear deterministic targets and cover only give-way
# cases, so Rule 17(a)(i) passive course-keeping has no representation there.
# Mirror of the overtaking condition with U_TS > U_OS.
BEING_OVERTAKEN_BEARING_MIN_DEG = 112.5  # alpha OS->TS, stern arc lower bound
BEING_OVERTAKEN_BEARING_MAX_DEG = 247.5  # ... upper bound
# A fraction of cruise, not an absolute (02b §2): 0.10 m/s meant 5.6% of cruise
# at the simulator's old speed and 18% at the field's, i.e. two different things.
# Must exceed the tracker's speed-estimation noise, which 05 measures.
# TODO(05): confirm against the measured tracker residual.
BEING_OVERTAKEN_SPEED_MARGIN = 0.15 * U_REF              # m/s

# Hysteresis, applied ONCE inside the classifier module (01 §5.3).  The same
# function feeds the observation and 02's reward gate; if they diverge even at a
# sector boundary the agent is penalised for a role it was never shown.
ENCOUNTER_HOLD_STEPS = 8                 # steps a new class must persist
ENCOUNTER_BEARING_HYSTERESIS_DEG = 3.0   # band around every threshold

# Modification 1: port and starboard crossing COLLAPSE into a single class under
# Rule 9(b) -- the own ship gives way either way (S3).  This replaces the
# Rule 18 route used by Meyer et al., whose premise (own ship much smaller than
# the vessels it meets) fails here: own ship and target are similarly sized
# model vessels.
#
# The geometric side is still computed and exposed as `crossing_side`, because
# 02's passing-side reward term needs it -- but the observation one-hot has a
# single crossing class.
#
# Frozen one-hot order.  Every checkpoint depends on it.
ENCOUNTER_CLASSES = (
    "none",
    "head_on",
    "crossing",
    "overtaking",
    "being_overtaken",
)
N_ENCOUNTER_CLASSES = len(ENCOUNTER_CLASSES)
assert N_ENCOUNTER_CLASSES == 5

# ===========================================================================
# 11. Target slot and observation scales  (01 §6)
# ===========================================================================
# S1: two-vessel encounters.  `N_MAX_TARGETS` stays a config parameter so a
# multi-vessel extension costs a retrain rather than a redesign -- the slot
# machinery below is parameterised but not exercised at 1.
N_MAX_TARGETS = 1

# 15 features + 1 presence bit.  See OBSERVATION_SPEC.md for the frozen order.
TARGET_FEATURES = 16

# Normalisers.
# `d_scale` is the sensor horizon, the largest distance the perception stack can
# report.  With the workspace now fixed at the basin size (O4 resolved), this no
# longer floats.
D_SCALE = LIDAR_RANGE                    # m

# 02b §2: 40 s, was 60.  `T_engage` is 25 s, so anything past ~30 s is
# unactionable -- 60 s spent a third of the feature range on values the policy
# can never use.
TCPA_CLIP = 40.0                         # s, symmetric clip

# Normalises `ego` u/v and both target-speed features.  Expressed as a multiple
# of U_REF so it survives recalibration (02b §6).
#
# The earlier saturation bug was never the factor -- 2 x cruise is the right
# shape -- it was that `U_CRUISE` held 0.55 while the simulator ran at 1.77, so
# the scale sat below the whole operating range.  With one U_REF that cannot
# recur.  At the widest curriculum stage the hull reaches 2.16 m/s, which is
# 0.95 of this scale, so nothing clips.
SPEED_SCALE = 2.0 * U_REF                # m/s

# DCPA is normalised in domain radii, not metres.  02b §2 replaces the free
# constant with a clip at the sensor horizon: a DCPA beyond what the vessel can
# see is not an estimate, it is an extrapolation.  Derived, not chosen.
DCPA_CLIP_DOMAINS = LIDAR_RANGE / DOMAIN_RADIUS_DCPA

# ===========================================================================
# 12. Policy architecture  (01 §6.3)
# ===========================================================================
# Plain concatenation of the five branches into the SAC MultiInputPolicy.  The
# shared per-slot encoder and the DeepSets/attention aggregation from Revision 1
# are **not needed at one target** and are not built: superseded decision D3.
#
# The target branch keeps a small encoder so the multi-vessel extension path
# exists, but there is no aggregation comparison to defend.
SCENE_ENCODER_HIDDEN = 128
SLOT_ENCODER_HIDDEN = (64, 64)
SLOT_EMBED_DIM = 32

# RESOLVED (02b §2), and this closes 01's open item.  The headline architecture
# stays feedforward.  Recurrence enters only as the RecurrentPPO comparator and,
# contingent on that ranking top-two in selection, the M7 frame-stacked
# ablation.
#
# The reason is not compute: adding recurrence to the headline would **confound
# N1**.  An occlusion result would no longer isolate perception from memory, and
# perception is the contribution.
USE_RECURRENCE = False

# ===========================================================================
# 13. Reward  (02a Rev 2.2, with 02b C1-C4 applied)  -- T4
# ===========================================================================
# Eight dense terms plus terminals.  Every dense term is normalised to [-1, 0]
# before weighting (`r_prog` to [-1, +1]), so **the weight is the maximum
# per-step contribution** and the 02a §7 hierarchy holds by construction rather
# than being discovered empirically.  That is the direct fix for the Paper 2
# failure, where a path term out-scaled the avoidance term through a hidden
# scale factor nobody had computed.
#
# The coefficients live here; `reward/` consumes them.  `reward/config.py` reads
# these as its dataclass defaults, so there is exactly one place to change a
# coefficient and the ablation switches cannot fork the values.
#
# The three below are the structural terminal payoffs, decided in 02b §2.
#
# -300 rather than 02a's original -200 (`R-7`): at -200 the margin between a
# collision episode and a maximally non-compliant one is 32 points, which
# violates the 02 §5 ordering that a COLREGs-compliant collision must be worse
# than a non-compliant near-miss.
R_COLLISION = -300.0
R_GOAL = 100.0

# **Zero, and that is load-bearing.**  Timeout is handled by value bootstrapping,
# which requires the env to return `truncated=True, terminated=False` at the step
# limit so SB3 bootstraps the value of the final state.  A large negative
# terminal here would make the agent treat running out of time as catastrophic
# and prefer almost anything to it -- voiding the "no loitering incentive"
# argument in 02a §8.1.
R_TIMEOUT = 0.0

# --- 13.1 Weights (02a §7) -------------------------------------------------
# 300 >> 3.0 > 2.5 > 2.2 > 1.8 > 0.6 > 0.3 > 0.10 > 0.05
# collision >> --- safety --- > COLREGs > --- task ---
W_BND = 3.00                             # channel boundary -- a hard constraint
W_DOM = 2.50                             # target ship domain
W_OBS = 2.20                             # static obstacle proximity
W_COL = 1.80                             # COLREGs group, after group clipping
W_PF = 0.60                              # path following
W_PROG = 0.30                            # progress
W_SMOOTH = 0.10                          # action smoothness
W_EXIST = 0.05                           # existence cost

# The longest encounter the ordering assertion has to survive.  02a §8.1 prices
# the compliance-cost ratio over a 100-step encounter, and the validator
# `abs(R_COLLISION) > W_COL * MAX_ENCOUNTER_STEPS` is what keeps a COLREGs-
# compliant collision worse than a maximally non-compliant near-miss (02 §5).
MAX_ENCOUNTER_STEPS = 100

# --- 13.2 Path following, r_pf (02a §5.1) ----------------------------------
# Width normalisation is load-bearing.  Paper 2 used exp(-0.05*|e_y|), inherited
# from a 60 x 150 m map; across a 10 m channel it varies by under 10% of its own
# value.  Normalising by the LOCAL half-width fixes that and holds the term's
# range constant across the Study 1 sweep, so the path-following gradient does
# not change with corridor width and confound the study.
PF_GAMMA_E = 4.0                         # cross-track decay
PF_W_E = 0.70                            # cross-track vs course weighting
PF_OMEGA_LA = 0.25                       # lookahead share of the course error

# `R-2`: a give-way obligation whose compliant alteration is inadmissible
# discharges under Rule 8(e) by slackening speed.  Without dropping the speed
# gate's reference the path term would charge full penalty for the compliant
# action, and 8(e) would be structurally unlearnable.
U_REF_SLOW_FACTOR = 0.40

# --- 13.3 Safety geometry (02a §5.2-5.4) -----------------------------------
# `c_wall` is HEAD_ON_WALL_CLEARANCE in §2 -- one constant, not two.
#
# TODO(decision) -- **02a's own d_safe breaches 02a's own invariant.**  §2
# asserts `d_safe < c_wall - B/2`, so that the geometry defining a compliant
# narrow-channel manoeuvre cannot itself trigger the boundary penalty.  With
# c_wall = 0.65 and B = 0.50 the ceiling is 0.40 m, and §5.2's stated
# d_safe = 0.50 m does not clear it -- in the same sentence that says 0.50 was
# chosen *because of* this invariant.
#
# 0.35 m is the largest 5 cm value that clears the ceiling with margin.  d_safe
# is the free parameter of the pair: c_wall drives all four Study 1 thresholds
# and is a TODO(05) measurement, so moving it would move published predictions,
# whereas d_safe only sets where the boundary penalty begins.
D_SAFE = 0.35                            # m, TODO(decision): see above
D_OA = 0.60                              # m, static-obstacle decay scale
D_CUT = 2.00                             # m, beyond which r_obs is exactly zero
OBS_SWATH_HALF_DEG = POOL_SWATH_HALF_DEG  # +/-135 deg, matching the c_t swath

# --- 13.4 Progress, r_prog (02b C3, corrected by T1) -----------------------
# 02b C3 replaces 02a's `(s_t - s_{t-1})/(U_ref*dt)` with a path-fraction form,
# so the episode integral stops depending on the unresolved cruise speed:
#
#     r_prog = clip(N_REF_PROG * (s_t - s_{t-1}) / L_path, -1, +1)
#     Sum r_prog = N_REF_PROG exactly, provided the clip never binds
#
# **F22 -- C3's literal `N_ref = 250` inverts `R-9` at the measured speed.**
# The clip binds when `u > L_path / (N_ref*dt)`.  At N_ref = 250 over a 20 m
# path that is 0.80 m/s -- which is 02a's *assumed* cruise, not the 1.14 m/s T1
# measured.  Every step at cruise would clip, and slowing down would then
# *increase* the progress integral (175 -> 250).  That is the creep exploit
# 02 §4.4 warns about, arriving through the very term meant to remove it.
#
# The fix is 02b C2's own rule: a speed-scaled constant is derived from U_REF,
# not written down.  Deriving it restores 02a §5.5's stated intent exactly --
# telescoping is exact for `u <= U_REF`, so a legal 8(e) slowdown costs zero
# progress reward, and speeding gains nothing.
#
# Consequence for the §8.1 audit table: `W_PROG * Sum r_prog` is +52.6, not the
# +75 tabulated.  All three orderings survive with room (nominal success ~ +83
# against -242 and -297), and the compliance-cost ratio is untouched, because
# progress contributes zero to it by telescoping.
L_REF_PATH = 20.0                        # m, the 02a §8.1 design-point path
N_REF_PROG = L_REF_PATH / (U_REF * UPDATE_RATE)          # 175.4 steps

# --- 13.5 Smoothness, r_smooth (02a §5.6) ----------------------------------
# `kappa_delta` is the actuator's per-step rate limit in normalised action
# units, so the term saturates at exactly the physical limit and self-calibrates
# when 05 delivers the actuator model.  Derived rather than written down.
KAPPA_DELTA = MAX_RUD_RATE_DPS * UPDATE_RATE / MAX_RUD_ANGLE     # 0.05
KAPPA_N = 0.30                           # throttle rate scale, TODO(05)
SMOOTH_W_N = 0.50                        # throttle share of the penalty

# `sigma_t` resolves the Rule 8 tension: 8(b) wants ONE large alteration and
# forbids a succession of small ones, but a plain smoothness penalty suppresses
# both.  The first two seconds after engagement are charged at a quarter rate,
# so the committed alteration is affordable; everything after is full rate, so
# dithering is not.
SIGMA_ENC = 0.25
N_FREE_STEPS = 20                        # 2.0 s at 10 Hz

# --- 13.6 Encounter state machine (02a §6.1) -------------------------------
T_ENGAGE = 25.0                          # s, TCPA within which engagement fires
# Scaled on `d_req`, NOT on the domain.  In 02a Revision 1 it evaluated to
# exactly the compliant separation, putting the engagement threshold on a
# knife-edge at the geometry the agent is supposed to achieve.  At 1.5*d_req
# engagement fires before the obligation does -- watch first, then act.
KAPPA_ENG = 1.5
KAPPA_REL = 2.5                          # DCPA multiple at which the encounter clears
N_CLEAR_STEPS = 30                       # steps in CLEARING before returning to IDLE
N_SWITCH_STEPS = 10                      # steps a new class must hold to re-latch

# --- 13.7 COLREGs sub-weights and thresholds (02a §6) ----------------------
# Pre-clip group maxima: head-on 1.45, crossing 1.60, overtaking 1.50, being
# overtaken 0.45.  Two concurrent severe violations saturate the group; one
# does not.
V_PORT_W = 0.55                          # turning the wrong way while give-way
V_BOW_W = 0.55                           # crossing ahead of the target
V_SIDE_W = 0.40                          # wrong-side passing
V_HOLD_W = 0.45                          # failing to hold course while stand-on
V_R8_W = 0.50                            # late or insufficient action

# TODO(05): `R_REF` from the identified turning circle, `R_DEAD` from the
# measured gyro noise floor.  The IMU is confirmed (05 §4.7), so `r` is measured
# rather than differentiated, and the yaw-rate-not-rudder criterion is directly
# checkable in the field instead of inferred.
R_REF = 0.20                             # rad/s, full-severity excess yaw rate
R_DEAD = 0.02                            # rad/s, below which a turn is not a turn
BETA_BOW_DEG = 67.5                      # bow arc for the crossing-ahead severity
R_HOLD = 0.05                            # rad/s, yaw tolerance while standing on
DU_HOLD = 0.10                           # m/s, speed tolerance, TODO(05)
T_EXTREMIS = 5.0                         # s, 17(b) release of the hold penalty

# Rule 8 deficit accounting.  `DU_MIN` is 30% of cruise, expressed as a multiple
# so it survives T1 rather than needing re-derivation (02b C2).
DPSI_MIN_DEG = 20.0                      # deg, the alteration that counts as "one"
DU_MIN = 0.30 * U_REF                    # m/s
# 2.50 m of lateral offset at cruise with a 30 deg alteration needs ~5.9 s of
# running plus ~3.5 s of turn-in and turn-out, ~10 s, plus margin.  Worth
# stating non-dimensionally too: T_ACT * U_REF / LBP ~ 10.9 ship lengths of
# advance, which is the form that answers the Froude question before it is asked.
T_ACT = 15.0                             # s

U_MIN_REACHABLE = 0.20                   # m/s, lowest steady speed without reverse

# --- 13.8 Open-water fallback, `R-10` (02a §5.5a) --------------------------
# 04 §4.1 runs "Around the Clock" in an open-water variant, where three terms
# are otherwise undefined: `r_pf` normalises on W_local, `r_bnd` needs a
# boundary polygon, and the admissibility predicate needs `d_bnd_*`.
#
# 10.0 m rather than an arbitrary large value, because it is the widest trained
# width: `e_y` then stays on the same scale as the widest sweep condition, and
# the open-water score is directly comparable to the 20 B row of Study 1.  Any
# other choice silently changes the path-following gradient between the two
# variants and makes the comparison meaningless.
W_REF_OPEN_WATER = 10.0                  # m

# ===========================================================================
# 14. Static obstacles (carried from Paper 2, replaced by 03/04)
# ===========================================================================
# S1 caps static obstacles at 3 alongside the single dynamic target.
MAX_OBS = 3
OBSTACLE_SIZE = 1.0

TRAIN_OBS_COUNTS = [0, 1, 2, 3]
TRAIN_OBS_PROBS = [0.20, 0.25, 0.35, 0.20]

TRAIN_SCENARIO_MODES = ["normal", "target_side", "field_repair", "gate", "offpath"]
TRAIN_SCENARIO_PROBS = [0.40, 0.35, 0.15, 0.05, 0.05]

OBSTACLE_PATH_START_FRAC = 0.25
OBSTACLE_PATH_END_FRAC = 0.70
OBSTACLE_CENTER_PROB = 0.30
OBSTACLE_LATERAL_OFFSET_MIN = 0.25
OBSTACLE_LATERAL_OFFSET_MAX = 0.95

GATE_GAP_RANGE = (1.35, 2.25)
GATE_PATH_FRAC_RANGE = (0.35, 0.70)
GATE_CENTER_JITTER_ALONG = 0.45
GATE_CENTER_JITTER_LATERAL = 0.20
GATE_LATERAL_EXTRA = (0.05, 0.30)

FIELD_REPAIR_PATH_FRACS = (0.43, 0.66, 0.66)
FIELD_REPAIR_LATERALS = (0.0, +1.95, -1.95)
FIELD_REPAIR_FRAC_JITTER = 0.035
FIELD_REPAIR_LAT_JITTER = 0.25

TARGET_SIDE_PATH_FRAC_RANGE = (0.38, 0.68)
TARGET_SIDE_CORRIDOR_OFFSET_RANGE = (0.65, 1.05)
TARGET_SIDE_BLOCKED_OFFSET_RANGE = (1.40, 2.30)
TARGET_SIDE_ALONG_JITTER = 0.45
TARGET_SIDE_LATERAL_JITTER = 0.20
TARGET_SIDE_RIGHT_PROB = 0.50

OFFPATH_LATERAL_MIN = 1.4
OFFPATH_LATERAL_MAX = 3.2

# ===========================================================================
# 15. Dynamic target  (03)
# ===========================================================================
# D1: constant velocity in training; reactive and non-compliant in evaluation
# only.  Training against a reactive opponent makes the environment
# non-stationary and destroys attribution.
TARGET_SPAWN_BEYOND_RANGE = True         # 03 §3: acquire it as it approaches
TARGET_SPAWN_MARGIN = 1.0                # m beyond LIDAR_RANGE
# Must **bracket** the own ship's cruise, or whole encounter classes become
# unreachable: a range entirely below cruise means no target can ever overtake,
# so `being_overtaken` -- the class carrying the Rule 17 contribution -- cannot
# occur at all.  Expressed as multiples of U_REF so it survives recalibration
# (02b §2).
# TODO(03): 03 owns the real distribution; bracketing cruise is the minimum
# property it must have.
TARGET_SPEED_RANGE = (0.35 * U_REF, 1.35 * U_REF)        # m/s

# Radius within which a dynamic track counts as "this target", for the
# perception metrics only -- never for the observation.  A cluster centroid sits
# on the visible face of the hull rather than at its centre, so the offset can
# reach half the LOA.
TARGET_MATCH_RADIUS = LOA

# Fraction of training episodes with no dynamic target at all.  Without this the
# static-only configuration is out of distribution (01 §6.2).
NO_TARGET_EPISODE_PROB = 0.25            # approved 02b §2

# Fraction of spawns placing the target on its own starboard side of the
# fairway, i.e. positionally Rule 9(a)-compliant with DCPA >= d_req.  02a §11.1
# makes sampling this a blocking requirement: without it every episode has the
# target on the own ship's projected track, the "holding course is correct"
# branch never fires, and the agent learns "always alter" instead of "when".
# TODO(04): 04 owns the real stratification and must report the realised
# distribution.
TARGET_COMPLIANT_SPAWN_PROB = 0.5        # approved 02b §2; 04 reports realised


# ===========================================================================
# 16. Corridor and the out-of-corridor world  (03a §1.2, §3)
# ===========================================================================
# Three distinct geometries, and conflating them is the whole point of keeping
# them named separately (03a §3.1):
#
#   | Polygon         | Role                                  | LiDAR sees it? |
#   | Corridor        | hard constraint for the own ship      | NO  -- map     |
#   | Basin envelope  | limit of the physical water           | no             |
#   | Facility walls  | basin + 1.5 m                         | YES -- gated   |
#
# **03a §1.2: simulate the out-of-corridor world.**  As built until now the
# simulated LiDAR returned only panels and the target, so the boundary gate had
# nothing to remove and passed everything through.  In the field it discards
# facility-wall returns, operators standing at scan height, and clutter beyond
# the pool edge -- so the gate was a no-op in simulation and load-bearing in the
# field, which is a sim-to-real gap in the exact component 01 §3 exists to
# remove one from.
FACILITY_WALL_MARGIN = 1.5               # m beyond the basin envelope
SIMULATE_FACILITY_WALLS = True

# When the corridor is narrower than the basin, the water between the corridor
# edge and the basin edge is open and returns nothing.  That is the real
# situation: the corridor is a map polygon and is invisible to the sensor.

CORRIDOR_LENGTH_M = MAP_HEIGHT           # 25 m, matches the basin (O4)
SPAWN_SPAN_M = 21.0                      # usable along-corridor spawn span
REF_PATH_LENGTH_M = L_REF_PATH           # 20 m, unchanged from Paper 2

# --- 16.1 Generator ranges (04a §3.2) --------------------------------------
# The boundary branch only earns its 7 dimensions when width varies, the path is
# off-centre, or the channel bends (01 §3.3).  A hard requirement on the
# generator, not a nicety: in a straight centred constant-width corridor the
# port and starboard rays are affine in `e_y` and the branch is decorative.
CORRIDOR_WIDTH_RANGE = (3.50, 10.00)     # m  (7 - 20 B)
CORRIDOR_WIDTH_VARIATION = (1.0, 1.8)    # W_max / W_min
CORRIDOR_WIDTH_CONTROL_POINTS = (2, 4)   # piecewise-linear along s
CORRIDOR_BEND_RANGE_DEG = (0.0, 60.0)    # total heading change
CORRIDOR_BEND_MIN_DEG = 20.0             # what counts as "a bend"
CORRIDOR_BEND_FRACTION = 0.40            # >= 40% of episodes must carry one
CORRIDOR_BEND_CENTRE_FRAC = (0.30, 0.70)  # where along s the bend sits
CORRIDOR_BEND_SPAN_FRAC = 0.40           # fraction of the length it occupies
PATH_OFFSET_FRAC_RANGE = (-0.30, 0.30)   # of the local half-width
# Positive = toward the starboard wall.  **The mean must be positive** -- that
# is Rule 9(a) station, and it is what makes the port and starboard boundary
# rays carry different information.
PATH_OFFSET_MEAN_FRAC = 0.12

# The assertion 04a §3.2 and 03a §3.2 both demand, enforced in the test suite.
BOUNDARY_DECORRELATION_MAX = 0.90        # |corr(e_y, b_i)| over 1000 episodes

# **F25 -- bends and wide corridors do not both fit a 10 m basin.**  A bend of
# total heading change `dpsi` over the corridor length deviates laterally by
# roughly `L * dpsi / 8`; the corridor also needs `W/2` either side.  At W = 10
# no bend fits at all, and >= 20 deg needs W <~ 7.8 m.  The generator therefore
# clamps the sampled bend to what the basin admits and records the realised
# distribution, rather than producing a corridor that leaves the water.
#
# 04a's ">= 40% of episodes with a >= 20 deg bend" is still reachable, but only
# because the width distribution puts enough mass below 7.8 m -- in stage 5
# (3.5-10 m uniform) about 66% of draws can carry one.  It is not reachable in
# stage 3 (7-10 m) at all.  Reported per stage rather than assumed.
CORRIDOR_BASIN_MARGIN = 0.25             # m of slack when fitting to the basin

# ===========================================================================
# 17. Scenario generator  (04a §3)
# ===========================================================================
# One generator, three consumers: the training distribution, the frozen suite,
# and the sweeps.  One seed-namespace scheme (§18).

ENCOUNTER_SAMPLE_CLASSES = ("head_on", "crossing", "overtaking",
                            "being_overtaken", "null", "no_target")

# --- 17.1 Class-conditional intervals (04a §3.4) ---------------------------
# **Written as formulae of `U_NOM`, not as the table.**  04a's own status note
# says every derived threshold "recomputes when 05 lands"; the same applies to
# F24.  `k = U_TS / U_OS`.
#
# 03a §1.3 raised the overtaking floor from 0.25 to 0.40, and the reasoning is
# worth keeping because it is physics rather than preference: pose error imparts
# an apparent velocity `sigma_v = sqrt(2) * sigma_p / T_w` to genuinely static
# objects, so the static/dynamic threshold needs a velocity window of order
# seconds -- and that window *is* the classification latency.  Requiring
# `v_hi = 5 sigma_v` below 30% of the slowest target speed gives
# `T_w >= 5 sqrt(2) sigma_p / (0.3 k_min U_nom)`, which at k_min = 0.25 demands
# 3.4 s, longer than several encounters are usable for.  At k_min = 0.40 it
# falls to about 2.1 s and fits every class's acquisition-to-CPA budget.
#
# The overtaking window is therefore squeezed from both ends by physics:
#     k >= 0.40  from pose noise and the static/dynamic classifier  (03a §1.3)
#     k <= 0.55  from basin length and the completed-pass constraint (04a §1.4)
CLASS_CT_DEG = {
    "head_on": (170.0, 190.0),
    "crossing": ((67.5, 175.0), (185.0, 292.5)),
    "overtaking": (-67.5, 67.5),
    "being_overtaken": (-67.5, 67.5),
    "null": (-20.0, 20.0),
}
CLASS_SPEED_RATIO = {
    "head_on": (0.70, 1.30),
    "crossing": (0.60, 1.40),
    "overtaking": (0.40, 0.55),          # floor raised by 03a §1.3
    "being_overtaken": (1.50, 2.20),
    "null": (0.85, 1.15),
}
# TCPA intervals in **seconds**, scaled off the nominal speed so they stay the
# same geometry when F24 is decided.  The 04a §3.4 table is these numbers at
# U_nom = 0.55.
_TCPA_REF_SPEED = 0.55


def class_tcpa_range(name: str, u_nom: float = None) -> tuple:
    """Spawn-TCPA interval for a class, in seconds at the operating speed.

    04a §3.4's table was derived at 0.55 m/s.  A TCPA is a *time* to cover a
    *distance*, so at a different operating speed the same geometry takes a
    proportionally shorter time -- which is why this is a formula and not a
    lookup.  Getting it wrong would silently shrink or stretch every encounter.
    """
    u = U_NOM if u_nom is None else float(u_nom)
    base = {
        "head_on": (12.5, 17.3),
        "crossing": (8.0, 15.0),
        "overtaking": (20.0, 32.0),
        "being_overtaken": (10.0, 16.0),
        "null": (0.0, 0.0),
    }[name]
    scale = _TCPA_REF_SPEED / max(u, 1e-6)
    return (base[0] * scale, base[1] * scale)


def class_spawn_range(name: str, k: float = None, u_nom: float = None) -> tuple:
    """Spawn range `R_0` interval, metres (04a §3.4, §1.4).

    **Spawning outside sensor range is only possible for head-on** (04a §1.4),
    because closing speed is a difference rather than a sum for every other
    class.  Forcing it everywhere would need a target under 0.16 m/s for
    overtaking -- below steerageway for a model hull -- or a corridor two to
    three times the basin length, which forfeits the physical-reproducibility
    argument O4 was resolved to protect.
    """
    if name == "head_on":
        return (1.15 * LIDAR_RANGE_EFFECTIVE, 19.0)
    if name == "crossing":
        return (5.0, 10.0)               # bounded by basin width
    if name == "overtaking":
        kk = 0.5 if k is None else float(k)
        return (6.0, min(12.0, 16.0 * (1.0 - kk)))
    if name == "being_overtaken":
        return (4.0, 6.0)                # channel astern of the own start
    if name == "null":
        return (8.0, 15.0)
    raise ValueError(f"unknown class {name!r}")


# TODO(04-2): `D_max` effective, including the matte-black wall side, from the
# retained logs.  The C1 nominal is 12 m; `LIDAR_RANGE` is the simulated sensor
# horizon and this is the range the *spawn rule* is written against.
LIDAR_RANGE_EFFECTIVE = 12.0             # m, TODO(04-2)

CLASS_DCPA_MAX = {
    "head_on": 2.0, "crossing": 2.5, "overtaking": 2.0,
    "being_overtaken": 2.0, "null": None,   # null requires DCPA > 4 m
}
NULL_MIN_DCPA = 4.0

# **The null class is mandatory** (04a §3.4).  A target on a similar course at a
# similar speed never emerges from a class-conditional spawner but is common in
# practice, and it is the case where a policy that has learned "target present
# => manoeuvre" will visibly overreact.
CLASS_SAMPLE_WEIGHTS = {
    "head_on": 0.20, "crossing": 0.22, "overtaking": 0.16,
    "being_overtaken": 0.14, "null": 0.11, "no_target": 0.17,
}

GENERATOR_MAX_ATTEMPTS = 200             # 04a §3.5; record the cap-out rate

# --- 17.2 Static obstacles (04a §3.6) --------------------------------------
OBSTACLE_CPA_GUARD_FRAC = 0.40           # +/- 0.4 * T_0 around CPA kept clear
OCCLUSION_DURATIONS_S = (1.0, 2.0, 4.0)

# --- 17.3 Curriculum (04a §3.7) --------------------------------------------
# Stage 3 is restricted to head-on and null deliberately: it is the only class
# pair where the compliant response is available at *every* width, so the agent
# learns the encounter machinery before it meets a geometry where the textbook
# manoeuvre is inadmissible.
CURRICULUM_STAGES = {
    1: {"width": (8.0, 10.0), "vary": False, "bend": False,
        "classes": ("no_target",), "clutter": (0, 1), "tcpa": "full"},
    2: {"width": (5.0, 10.0), "vary": True, "bend": True,
        "classes": ("no_target",), "clutter": (0, 3), "tcpa": "full"},
    3: {"width": (7.0, 10.0), "vary": True, "bend": True,
        "classes": ("head_on", "null", "no_target"), "clutter": (0, 1),
        "tcpa": "upper"},
    4: {"width": (4.5, 10.0), "vary": True, "bend": True,
        "classes": ENCOUNTER_SAMPLE_CLASSES, "clutter": (0, 2), "tcpa": "full"},
    5: {"width": (3.5, 10.0), "vary": True, "bend": True,
        "classes": ENCOUNTER_SAMPLE_CLASSES, "clutter": (0, 3), "tcpa": "full"},
}
CURRICULUM_STAGE_STEPS = None            # TODO(04-3): after throughput measurement

NO_TARGET_TRAINING_FRACTION = (0.15, 0.20)   # 04a §3.4

# ===========================================================================
# 18. Evaluation suite  (04a §4-§9)
# ===========================================================================
# --- 18.1 Width thresholds (04a §1.1) --------------------------------------
# **04a §1.1 supersedes 02a §2.2 for crossing, and the difference is 0.8 m.**
#
#   02a §2.2:  own ship has `W/2` of starboard room    -> W >= 2(d_req + c_wall + B/2)
#   04a §1.1:  own ship is stationed at the starboard
#              quarter-width under Rule 9(a), so it has
#              only `W/4`                               -> W >= 4(a_abeam + w_wall)
#
# 04a's is the more correct reading and it is 02a's own N2 insight carried one
# step further: a vessel already keeping starboard under 9(a) has *spent* its
# starboard room before the Rule 14 alteration becomes tight.  Both are computed
# in `predicted_thresholds()` and the divergence is reported rather than hidden
# -- 04a §11 lists "confirm the §1.1 threshold ordering against 02a" as an open
# item owned by 02, and it is not Claude Code's to close.
W_WALL = HEAD_ON_WALL_CLEARANCE          # 0.65 m, TODO(05); 04a calls it w_wall

# Pose drift rate used by the overtaking exposure margin.  04a §1.1 quotes
# `2 * rho_pose * t_exp = 0.46 m` at `t_exp = 22.8 s`, which implies this value.
# TODO(05): replaces the implied figure once the rf2o walk rate is measured --
# at which point `BOUNDARY_POSE_NOISE_WALK` becomes the source and this constant
# is deleted rather than updated.
RHO_POSE_DRIFT = 0.0101                  # m/s, TODO(05), implied by 04a §1.1

# --- 18.2 Study 1 (04a §6) -------------------------------------------------
# Two levels added to the original six: 7.0 m brackets the crossing transition
# at 7.60 m and 4.5 m brackets the overtaking transition at 4.26 m, both of
# which the six-level sweep stepped straight over.
STUDY1_WIDTHS_M = (10.0, 8.0, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5)
STUDY1_BASE_CONSTELLATIONS = 12

# --- 18.3 Tier B strata (04a §4.3) -----------------------------------------
TIER_B_EPISODES_PER_CELL = 25
TIER_B_BEHAVIOURS = ("cv", "re", "nc")
TIER_A_ROLLOUTS = 10
AROUND_THE_CLOCK_SPOKES = 24
AROUND_THE_CLOCK_WIDTHS_M = (10.0, 6.0, 4.26)

# --- 18.4 Study 2 (04a §7) -------------------------------------------------
STUDY2_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0, 4.0)
STUDY2_AXES = ("pose_drift", "dropout_rate", "occlusion_duration",
               "velocity_noise")

# --- 18.5 Seed namespaces (04a §9.2) ---------------------------------------
# **Disjoint by construction, and asserted in a test.**  The development suite
# exists so that reward iteration and algorithm selection never touch the frozen
# suite; using one suite for both introduces a selection bias that is invisible
# in the results and fatal if noticed.
SEED_NAMESPACES = {
    "training": (0, 99_999),
    "development": (200_000, 209_999),
    "frozen_eval": (300_000, 309_999),
    "study1": (400_000, 409_999),
    "study2": (500_000, 509_999),
}

SUITE_VERSION = "paper3-suite-v0"        # bumped on any generator change
