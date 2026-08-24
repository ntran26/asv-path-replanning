"""Task, reward and curriculum constants for the ASV local-planner environment.

Vessel geometry and hydrodynamics live in `ship.py`; sensor constants live in
`lidar.py`.  Everything here describes the *task*: the basin, the reward, and
the distribution of training scenarios.
"""

# --- Simulation ------------------------------------------------------------
UPDATE_RATE = 0.1                 # control period [s] -> 10 Hz
MAX_EPISODE_STEPS = 700           # 70 s episode cap
RENDER_FPS = 10
RENDER_SCALE = 25                 # pixels per metre

# --- Basin and reference path ----------------------------------------------
MAP_WIDTH = 10.0
MAP_HEIGHT = 25.0
MAX_OBS = 5

PATH_MODE = "straight"            # "straight" | "curve" | "mixed"
CURVE_PROB = 0.0                  # probability of a curve when PATH_MODE == "mixed"
LOOKAHEAD_FRACTION = 0.25         # lookahead point at 25% of the path length

VERTICAL_PATH_PROB = 0.70         # rest are slanted start/goal pairs
START_Y = 2.0
GOAL_Y_MARGIN = 3.0
START_X_MARGIN_FRAC = 0.25
START_X_MARGIN_MIN = 2.0

# --- Goal acceptance region -------------------------------------------------
GOAL_RADIUS = 0.5                 # plain distance-to-goal acceptance
GOAL_ALONG_DIST = 1.25            # ...or within this arc length of the path end
GOAL_CTE_RADIUS = 1.60            # ...and this far off it laterally

# --- Actuation ---------------------------------------------------------------
# action = [rudder, throttle], both in [-1, 1].
# rudder is passed to the hull as a percentage; throttle trims around cruise.
CRUISE_RPM = 12.0
FIXED_RPM = False                 # True ignores throttle and holds CRUISE_RPM

# Speed-authority curriculum: (delta, floor, ceiling) in RPM.
RPM_STAGE = 1
RPM_STAGES = {
    1: (3.0, 9.0, 15.0),
    2: (4.0, 8.0, 16.0),
    3: (6.0, 6.0, 18.0),
    4: (12.0, 0.0, 24.0),
}
RPM_DELTA, RPM_FLOOR, RPM_CEIL = RPM_STAGES[RPM_STAGE]

# --- Reward ------------------------------------------------------------------
DEFAULT_EVAL_LAMBDA = 0.5         # fixed path-following vs obstacle-avoidance blend

R_COLLISION = -1000.0             # replaces all dense terms
R_TIMEOUT = -1000.0               # added on truncation
R_GOAL = 50.0
R_EXIST = -0.5                    # living cost

# Path following.  gamma_e is blended between the two values below by
# block_alpha, so tracking is strict in open water and tolerant near obstacles.
GAMMA_E_CLEAR = 0.20
GAMMA_E_BLOCKED = 0.05
U_REWARD_REF = 0.8                # speed at which the path/heading gate saturates
W_HEADING = 0.35

# Anti-stall and effort terms.
K_PROGRESS = 0.7
K_SLOW = 0.10
U_MIN_REWARD = 0.30
K_THRUST_DEV = 0.025

# Basin walls.
K_BORDER_SOFT = 0.40
SOFT_BORDER_SAFE_DIST = 1.0

# Side-choice repair terms.  Both are deliberately small: they nudge the policy
# back toward the path after an avoidance manoeuvre and penalise rudder that
# contradicts an unambiguously better side.
K_CTE_RECOVERY = 0.35
K_WRONG_SIDE_ACTION = 0.12
WRONG_SIDE_CTE_MIN = 0.25
WRONG_SIDE_DIFF_MIN = 1.00
WRONG_SIDE_FRONT_MIN = 1.20

# Sector penalty reported in `info["mean_sector_pen"]`; diagnostic only.
GAMMA_X = 0.005
EPSILON_X = 1.0

# --- LiDAR-derived local planner features ------------------------------------
BLOCK_D_SAFE = 4.5                # front clearance at which blockage starts
BLOCK_D_CRIT = 2.0                # ...and at which it saturates
BLOCK_FRONT_DEG = 15.0            # half-width of the "ahead" arc
SIDE_ARC_MIN_DEG = 15.0
SIDE_ARC_MAX_DEG = 100.0
SIDE_CLEAR_TIE = 0.15             # below this, treat both sides as equally clear
BYPASS_CTE = 0.5                  # lateral offset suggested by the bypass cue

# --- Basin walls in the observation LiDAR ------------------------------------
# "none"       walls invisible to the policy
# "asymmetric" left pool edge visible, right edge invisible, far right wall visible
# "both"       both true pool edges visible
# "mixed"      sample one of the above per episode
OBS_BORDER_MODE = "none"
OBS_BORDER_P_NONE = 0.10          # "mixed" only; the remainder is "both"
OBS_BORDER_P_ASYMMETRIC = 0.60
RIGHT_WALL_OFFSET = 1.0

# --- Obstacle curriculum ------------------------------------------------------
OBSTACLE_SIZE = 1.0

TRAIN_OBS_COUNTS = [0, 1, 2, 3, 4]
TRAIN_OBS_PROBS = [0.15, 0.15, 0.45, 0.15, 0.10]

# Layout families.  "target_side" and "field_repair" oversample the side-choice
# failure mode; enough "normal" episodes remain to limit forgetting when
# fine-tuning from an existing checkpoint.
TRAIN_SCENARIO_MODES = ["normal", "target_side", "field_repair", "gate", "offpath"]
TRAIN_SCENARIO_PROBS = [0.40, 0.35, 0.15, 0.05, 0.05]

# Single obstacle placed near the reference path.
OBSTACLE_PATH_START_FRAC = 0.25
OBSTACLE_PATH_END_FRAC = 0.70
OBSTACLE_CENTER_PROB = 0.30
OBSTACLE_LATERAL_OFFSET_MIN = 0.25
OBSTACLE_LATERAL_OFFSET_MAX = 0.95

# Gate: two obstacles either side of the path.  Gap is between inner faces.
GATE_GAP_RANGE = (1.35, 2.25)
GATE_PATH_FRAC_RANGE = (0.35, 0.70)
GATE_CENTER_JITTER_ALONG = 0.45
GATE_CENTER_JITTER_LATERAL = 0.20
GATE_LATERAL_EXTRA = (0.05, 0.30)

# Field repair: perturbations of a recorded layout where the policy committed to
# the wide side while a path-side corridor was still open.
FIELD_REPAIR_PATH_FRACS = (0.43, 0.66, 0.66)
FIELD_REPAIR_LATERALS = (0.0, +1.95, -1.95)
FIELD_REPAIR_FRAC_JITTER = 0.035
FIELD_REPAIR_LAT_JITTER = 0.25

# Target side: the path-recovery side is deliberately left passable.
TARGET_SIDE_PATH_FRAC_RANGE = (0.38, 0.68)
TARGET_SIDE_CORRIDOR_OFFSET_RANGE = (0.65, 1.05)
TARGET_SIDE_BLOCKED_OFFSET_RANGE = (1.40, 2.30)
TARGET_SIDE_ALONG_JITTER = 0.45
TARGET_SIDE_LATERAL_JITTER = 0.20
TARGET_SIDE_RIGHT_PROB = 0.50     # mirror the layout so the repair is two-sided

# Off-path distractors: visible, but not a reason to leave the path.
OFFPATH_LATERAL_MIN = 1.4
OFFPATH_LATERAL_MAX = 3.2
