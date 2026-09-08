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

from ship import VESSEL_LENGTH, VESSEL_WIDTH  # noqa: F401  (re-exported below)

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
MAX_EPISODE_STEPS = 700                  # 70 s episode cap
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

# TODO(decision) -- **THE FLOOR ALREADY EXCLUDES THE PROVISIONAL VALUE.**
# 02b §3.1 observes that 0.75*Lpp = 1.18 m "sits 0.18 m outside the sensor blind
# zone", which compares it to LIDAR_MIN_RANGE alone -- and then defines the
# floor as LIDAR_MIN_RANGE + hull half-breadth = 1.25 m, which 1.18 m does not
# clear.  The two halves of that section disagree.
#
# Not resolved unilaterally: 02a §1 states the abeam extent as 0.75*Lpp
# explicitly, and raising it to 1.25 m (0.796*Lpp) moves d_req and all four
# Study 1 thresholds.  The value below stays as 02a specifies; `check_domain()`
# reports the breach so it cannot be missed.
DOMAIN_FORE = 2.00 * LBP                 # 3.14 m, TODO(05): provisional
DOMAIN_AFT = 1.00 * LBP                  # 1.57 m, TODO(05): provisional
DOMAIN_LATERAL = 0.75 * LBP              # 1.18 m, TODO(05)/TODO(decision)


def check_domain(d_abeam=None) -> list:
    """Validator for the ship domain.  Returns a list of problems, empty if ok.

    02b §3.1 asks for the floor to be asserted in the config validator.  It is
    returned rather than raised, because the provisional value breaches it today
    and hard-failing at import would block everything on a decision that is not
    yet made.  `T4`'s config validator should raise on this once the domain is
    final.
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
# 13. Reward -- NOT DESIGNED HERE
# ===========================================================================
# 02 owns the entire reward: six carried-over terms, five COLREGs terms, the
# Rule 9 precedence table, and the mandatory per-term scale audit.  Per D10 it
# is redesigned, not patched, and per kickoff §8 no Paper 2 reward term name may
# appear in this tree.
#
# The three below are the structural terminal payoffs, decided in 02b §2.
# Everything dense is still 02's.
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
