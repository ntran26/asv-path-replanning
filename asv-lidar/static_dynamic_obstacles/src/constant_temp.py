"""Classical comparator settings — staging file, to be merged into `constants.py`.

**Why this is separate (your call, 2026-09-23).** Every constant in this project
lives in `constants.py`, and these belong there too. They are staged here until
the baseline-v2 campaign finishes, for one mechanical reason: the frozen
formulation's digest is computed over *every* upper-case name in `constants.py`
(`src/baseline_config.py`), so adding or tuning a value there would change the
digest, fail the pre-run check and halt the campaign mid-flight.

Nothing in training reads this file. It holds the parameters of the classical
comparators (04a §5):

* **LOS-PID** — the shared path follower and its controller gains;
* **LOS-PID + DWA** — dynamic-window sampling, horizons and clearance scoring;
* **Encounter-specific VO** — velocity-obstacle horizons, candidate sampling,
  the ship-domain scaling and the cost weights.

**Values are ported unchanged** from `src/classical/` as of 2026-09-23, so
behaviour is identical; the port only moves where they are written down.

**Still to do (C3-C6, after the campaign):**
1. tune on the **development set** against the same score the RL checkpoints use
   (goal rate − 2 × collision rate), with the search reported;
2. record the tuned values and their digest with every comparator result, so a
   table says what the baselines were;
3. merge this file into `constants.py`, excluding these names from the
   formulation digest (they are not part of the RL formulation);
4. build COLREGs-VO (Kuwata), which does not exist yet (B8).

Derived quantities (`HALF_L`, `SUBSTEPS`, `DELAY_STEPS`, `U_PER_RPM`) stay in
`src/classical/common.py`: they follow from the hull and the identified plant,
and are not free parameters.
"""
import math

import numpy as np

import constants as cfg

# ---------------------------------------------------------------------------
# Shared: actuation limits and the LOS path follower (src/classical/common.py)
# ---------------------------------------------------------------------------
CLASSICAL_MAX_RUDDER_DEG = 40.0          # deg, command limit
CLASSICAL_LOS_LOOKAHEAD_M = 2.5          # m, look-ahead distance
CLASSICAL_MAX_SIDESLIP_DEG = 20.0        # deg, sideslip compensation cap

# Heading PID, on the LOS course error.
CLASSICAL_PID_KP = 1.0 / 35.0            # rudder units per deg
CLASSICAL_PID_KD = 1.0 / 18.0            # per deg/s of measured yaw rate
CLASSICAL_PID_KI = 1.0 / 600.0           # per deg.s
CLASSICAL_PID_I_GATE_DEG = 10.0          # integral acts only inside this error
CLASSICAL_PID_I_LIMIT = 0.15             # rudder units

# Yaw-rate and speed loops.
CLASSICAL_YAW_RATE_GAIN_DPS = 10.4       # deg/s at full rudder, identified plant
CLASSICAL_YAW_RATE_KP = 0.08             # rudder units per deg/s of error
CLASSICAL_SPEED_KP_RPM = 6.0             # rpm-units per m/s of speed error

# Prediction and perception memory.
CLASSICAL_PRED_DT = 0.125                # s, rollout step
CLASSICAL_SCAN_MEMORY_FRAMES = 4         # scans retained for static structure
CLASSICAL_TRACK_EXCLUSION_M = 1.8        # returns this close to a track are the target's

# ---------------------------------------------------------------------------
# LOS-PID + DWA (src/classical/los_dwa.py)
# ---------------------------------------------------------------------------
CLASSICAL_DWA_TRAVEL_M = 5.5             # static obstacles judged over this track (10 s at cruise)
CLASSICAL_DWA_TARGET_HORIZON_S = 10.0    # tracked targets judged over this time
CLASSICAL_DWA_MAX_HORIZON_S = 16.0       # rollout length for the slowest candidate
CLASSICAL_DWA_YAW_RATES_DPS = tuple(np.linspace(-9.0, 9.0, 9))   # inside the 10.4 deg/s turn
CLASSICAL_DWA_SPEED_FRACTIONS = (1.0, 0.5, 0.0)                  # of U_NOM; 0 = propulsion floor
CLASSICAL_DWA_COMMIT_S = (3.0, 6.0)      # how long a candidate is held
CLASSICAL_DWA_LOS_SPEED_FRACTIONS = (1.0, 0.5)
CLASSICAL_DWA_SAFE_GAP_M = 0.20          # hull-to-hull, beyond the hull margin
CLASSICAL_DWA_BOUNDARY_GAP_M = 0.05      # the map polygon is exact, the scan is not
CLASSICAL_DWA_DIST_CAP_M = 1.5           # clearance beyond this earns nothing more
CLASSICAL_DWA_PATH_SCALE_M = 2.0         # mean cross-track error scoring zero
CLASSICAL_DWA_ENGAGE_RANGE_M = 9.0       # target range at which avoidance engages

# ---------------------------------------------------------------------------
# Encounter-specific VO (src/classical/encounter_vo.py)
# ---------------------------------------------------------------------------
CLASSICAL_VO_COURSE_OFFSETS_DEG = tuple(np.arange(-90.0, 90.01, 5.0))   # about the present heading
CLASSICAL_VO_SPEED_FRACTIONS = (1.0, 0.75, 0.5, 0.25, 0.0)
CLASSICAL_VO_TAU_S = 20.0                # horizon for tracked targets
CLASSICAL_VO_TAU_STATIC_S = 10.0         # and for scan points and the map polygon
CLASSICAL_VO_DT = 0.25                   # s, cone sampling step
CLASSICAL_VO_HARD_GAP_M = 0.20           # hull-to-hull clearance treated as a hard limit
CLASSICAL_VO_STATIC_GAP_M = 0.15
CLASSICAL_VO_BOUNDARY_GAP_M = 0.05
CLASSICAL_VO_SIDE_FREE_DCPA_M = 2.0 * cfg.DOMAIN_LATERAL   # 2.5 m: the pass is clear either side
CLASSICAL_VO_STAND_ON_RELEASE_S = 12.0   # 17(a)(ii): act when the give-way vessel has not
CLASSICAL_VO_TURN_LAG_S = 1.0            # assumed response lag of the other vessel
CLASSICAL_VO_TURN_RATE_DPS = 6.0         # assumed turn rate when projecting a response
CLASSICAL_VO_SPEED_TAU_DOWN_S = 60.0     # speed-change time constants
CLASSICAL_VO_SPEED_TAU_UP_S = 20.0

# Ship-domain scaling per encounter class, as (fore, aft, lateral) multipliers:
# crossing stretches ahead so a candidate keeps the target further off the own
# bow (a clear pass astern); head-on and overtaking widen abeam, where those
# passes happen.
CLASSICAL_VO_DOMAIN_SCALE = {
    "head_on": (1.0, 1.0, 1.2),
    "crossing": (1.5, 1.0, 1.0),
    "overtaking": (1.0, 1.0, 1.2),
    "being_overtaken": (1.0, 1.0, 1.0),
    "none": (1.0, 1.0, 1.0),
}

# Cost weights over the candidate (course, speed) set.
CLASSICAL_VO_W_COURSE = 1.0              # per pi rad from the preferred course
CLASSICAL_VO_W_SPEED = 0.5               # per cruise speed below preferred
CLASSICAL_VO_W_CHANGE = 0.3              # per pi rad from the previous course
CLASSICAL_VO_W_DOMAIN = 3.0              # per unit of ship-domain penetration
CLASSICAL_VO_STATIC_SOFT_M = 0.6         # static cost reaches zero here
CLASSICAL_VO_W_STATIC = 1.0              # at the hard gap

# ---------------------------------------------------------------------------
# COLREGs-VO, Kuwata et al. 2014 (src/classical/colregs_vo.py)
# ---------------------------------------------------------------------------
# The published method's own rule model: open-water roles, and the give-way
# constraint that the relative velocity stays to starboard of the bearing line.
CLASSICAL_KVO_HEAD_ON_HALF_DEG = 22.5    # deg, reciprocal sector (Rule 14)
CLASSICAL_KVO_OVERTAKING_DEG = 112.5     # deg, the overtaking sector (Rule 13)
CLASSICAL_KVO_COURSE_OFFSETS_DEG = tuple(np.arange(-90.0, 90.01, 5.0))
CLASSICAL_KVO_SPEED_FRACTIONS = (1.0, 0.75, 0.5, 0.25, 0.0)
CLASSICAL_KVO_TAU_S = 20.0               # horizon for tracked targets
CLASSICAL_KVO_TAU_STATIC_S = 10.0        # and for scan points and the map polygon
CLASSICAL_KVO_DT = 0.25                  # s, sampling step along the horizon
CLASSICAL_KVO_HARD_GAP_M = 0.20          # hull-to-hull clearance treated as a hard limit
CLASSICAL_KVO_STATIC_GAP_M = 0.15
CLASSICAL_KVO_BOUNDARY_GAP_M = 0.05
CLASSICAL_KVO_SIDE_FREE_DCPA_M = 2.0 * cfg.DOMAIN_LATERAL   # wider passes are unconstrained
CLASSICAL_KVO_STAND_ON_RELEASE_S = 12.0  # Rule 17: hold while the preferred velocity stays clear
CLASSICAL_KVO_TURN_RATE_DPS = 6.0        # the turn rate the candidate sweep assumes
CLASSICAL_KVO_W_COURSE = 1.0             # per pi rad from the preferred course
CLASSICAL_KVO_W_SPEED = 0.5              # per cruise speed below preferred
CLASSICAL_KVO_W_CHANGE = 0.3             # per pi rad from the previous course

# Radians, for the code that wants them.
CLASSICAL_MAX_RUDDER_RAD = math.radians(CLASSICAL_MAX_RUDDER_DEG)
CLASSICAL_MAX_SIDESLIP_RAD = math.radians(CLASSICAL_MAX_SIDESLIP_DEG)
