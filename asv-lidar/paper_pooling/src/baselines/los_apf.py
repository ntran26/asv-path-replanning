"""LOS + PID + APF classical baseline.

A conventional guidance/control stack: line-of-sight path following, an
artificial-potential-field avoidance layer driven by the pooled LiDAR sectors,
PID heading control, and a reactive speed rule.  It exposes
`predict(obs, deterministic=True) -> action` so it drops straight into the
shared evaluation harness alongside the SB3 policies.

The fairness constraint
-----------------------
**This controller reads the 34-dimensional observation and nothing else.**  It
never touches `env.obstacles`, `env.map_border`, the vessel's world pose, or any
other privileged geometry -- the constructor does not even receive the env.
Granting a classical stack ground-truth obstacle geometry would make the
comparison meaningless and would contradict the observation-interface-parity
argument the manuscript makes elsewhere.

It does use `front_clearance`, `side_clearance_diff` and `local_target_cte`,
because those are components of the observation vector the SAC policy receives.
Withholding them would handicap the baseline rather than make it fair.  Note
that `side_clearance_diff` and `local_target_cte` carry partial *boundary*
information (they are computed against a wall-only LiDAR), so the claim that
both methods are equally boundary-blind is false -- see BASELINES_NOTES.md
section 1.1.  Both methods have the same partial boundary information, which is
the actual fair comparison.

Sign conventions, all verified against the environment source
-------------------------------------------------------------
* Heading is compass-style: 0 deg = +y, clockwise positive.
* `cross_track_error > 0` means the vessel is **left (port) of the path**, and
  recovery is to starboard.  (`env.py:334` comment, and `train.side_path_guard`,
  agree: "Positive CTE means the vessel is left of the path and should recover
  to starboard".)
* **Positive `action[0]` turns to starboard**, i.e. increases heading.
* `side_clearance_diff = right - left`, so positive means starboard is clearer.
* `course_error = path_course - course`, so positive means the path heads to
  starboard of the current course.

Because every angular observation is already expressed *relative to the current
course*, the whole controller works in relative angles and never needs absolute
heading.  The LOS law

    chi_d = chi_path + atan2(e, Delta)

therefore becomes a heading **error**

    chi_err = course_error + degrees(atan2(e, Delta))

Note the sign of `e`: the textbook form is `atan2(-e, Delta)` under the
convention that positive cross-track error is to starboard.  This environment
defines positive as **port**, so the sign flips.  Getting this backwards makes
the controller steer away from the path, which is the single easiest way to
build an accidental straw man.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from lidar import LIDAR_BEAMS, LIDAR_SECTORS, LIDAR_SWATH, sector_angle_grid

# Nominal sector centre bearings, as the environment itself uses them.
_NOMINAL_BEARINGS = sector_angle_grid().astype(np.float64)

# True angular centre of each pooled chunk.  Pooling splits 225 beams into 25
# chunks of 9, so chunk i is centred on beam 9i+4, not on the nominal grid
# point.  The two differ by up to 4.82 deg at the swath edges.  Exposed as a
# tunable so the search can choose rather than the author assuming.
_RAW_ANGLES = np.linspace(-LIDAR_SWATH / 2.0, LIDAR_SWATH / 2.0, LIDAR_BEAMS)
_ACTUAL_BEARINGS = np.array(
    [_RAW_ANGLES[c].mean() for c in np.array_split(np.arange(LIDAR_BEAMS), LIDAR_SECTORS)],
    dtype=np.float64,
)

DEFAULTS: Dict[str, Any] = {
    # --- LOS guidance ---
    "delta_lookahead": 6.0,     # m; larger = gentler convergence to the path
    "w_lookahead": 0.0,         # blend toward the env's own lookahead bearing
    "max_los_deg": 75.0,        # clamp on the LOS approach angle

    # --- APF repulsion ---
    "k_rep": 25.0,              # overall repulsion gain, deg per unit potential
    "c_threshold": 0.55,        # closeness below this contributes nothing
    "rep_power": 2.0,           # how sharply repulsion grows with closeness
    "rep_sigma_deg": 55.0,      # angular decay width about the heading
    "max_rep_deg": 70.0,        # clamp on the total deflection

    # --- head-on symmetry breaking ---
    "k_headon": 45.0,           # deg of deflection per unit frontal potential
    "headon_deg": 25.0,         # half-width of the "dead ahead" arc
    "side_tie": 0.20,           # |side_clearance_diff| below this is a tie
    "default_side": 1.0,        # tie-break: +1 starboard, matching the env's bias

    # --- heading PID ---
    "kp": 0.020,                # action units per degree of heading error
    "ki": 0.0008,
    "kd": 0.010,                # on yaw rate, not on the error (no derivative kick)
    "integral_limit": 200.0,    # deg*s, anti-windup clamp

    # --- speed control ---
    "throttle_base": 1.0,       # commanded throttle in clear water
    "k_speed_obs": 1.2,         # slow down as the nearest sector closes
    "k_speed_head": 0.8,        # slow down when badly off the desired heading
    "min_throttle": -1.0,

    # --- sector geometry ---
    "bearing_mode": "nominal",  # "nominal" (as the env uses) | "actual" (true chunk centres)
}


class LosApfController:
    """LOS guidance + APF avoidance + PID heading + reactive speed.

    Stateful: the PID integrator and the head-on side commitment persist across
    steps within an episode.  `reset()` clears them and **must** be called at the
    start of every episode -- the evaluation harness does this automatically.
    """

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown LOS+APF parameters: {sorted(unknown)}")
        self.p: Dict[str, Any] = {**DEFAULTS, **params}

        mode = str(self.p["bearing_mode"])
        if mode not in ("nominal", "actual"):
            raise ValueError(f"bearing_mode must be 'nominal' or 'actual', got {mode!r}")
        self.bearings = _NOMINAL_BEARINGS if mode == "nominal" else _ACTUAL_BEARINGS

        # Precompute the angular decay weights; they depend only on parameters.
        sigma = max(float(self.p["rep_sigma_deg"]), 1e-6)
        self.angular_weight = np.exp(-(self.bearings / sigma) ** 2)
        self.lateral = -np.sin(np.radians(self.bearings))   # push away from the sector
        self.frontal_mask = np.abs(self.bearings) <= float(self.p["headon_deg"])

        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear integrator and side commitment.  Call once per episode."""
        self.integral = 0.0
        self.committed_side = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _scalar(obs: Dict[str, Any], key: str) -> float:
        value = obs[key]
        return float(np.asarray(value).reshape(-1)[0])

    def _los_heading_error(self, obs: Dict[str, Any]) -> float:
        """Desired course, expressed as an error against the current course."""
        e = self._scalar(obs, "cross_track_error")
        course_error = self._scalar(obs, "course_error")
        lookahead_error = self._scalar(obs, "lookahead_course_error")

        delta = max(float(self.p["delta_lookahead"]), 1e-3)
        # +e (port of path) -> positive approach angle -> steer to starboard.
        approach = math.degrees(math.atan2(e, delta))
        approach = float(np.clip(approach, -self.p["max_los_deg"], self.p["max_los_deg"]))

        w = float(np.clip(self.p["w_lookahead"], 0.0, 1.0))
        return (1.0 - w) * (course_error + approach) + w * lookahead_error

    def _side_preference(self, obs: Dict[str, Any]) -> float:
        """+1 to bypass to starboard, -1 to port.

        Uses `side_clearance_diff` (right minus left) with a tie band, falling
        back to the env's own bypass cue `local_target_cte`, then to a fixed
        default.  The choice is latched for as long as the vessel stays blocked,
        so the controller cannot dither between sides -- the classic APF failure.
        """
        diff = self._scalar(obs, "side_clearance_diff")
        tie = float(self.p["side_tie"])

        if diff > tie:
            side = 1.0
        elif diff < -tie:
            side = -1.0
        else:
            # local_target_cte < 0 means the env's own cue points to starboard.
            cue = self._scalar(obs, "local_target_cte")
            if cue < -1e-6:
                side = 1.0
            elif cue > 1e-6:
                side = -1.0
            else:
                side = float(self.p["default_side"])

        if self.committed_side == 0.0:
            self.committed_side = side
        return self.committed_side

    def _repulsion(self, obs: Dict[str, Any]) -> Tuple[float, float]:
        """Course deflection from the pooled sectors, and the frontal potential.

        Each sector is treated as a repulsive source at its centre bearing, with
        a magnitude that grows with closeness above an influence threshold and a
        weight that decays with angular distance from the heading.  The lateral
        components sum to a deflection; the frontal component is returned
        separately because a source dead ahead produces no lateral push and
        needs the side-preference tie-break instead.
        """
        closeness = np.asarray(obs["lidar"], dtype=np.float64).reshape(-1)

        c0 = float(np.clip(self.p["c_threshold"], 0.0, 0.999))
        excess = np.clip((closeness - c0) / (1.0 - c0), 0.0, 1.0)
        magnitude = excess ** float(self.p["rep_power"])

        weighted = magnitude * self.angular_weight
        deflection = float(self.p["k_rep"]) * float(np.sum(weighted * self.lateral))
        frontal = float(np.sum(weighted[self.frontal_mask]))
        return deflection, frontal

    # ------------------------------------------------------------------
    def predict(self, obs: Dict[str, Any], deterministic: bool = True,
                state: Optional[Any] = None, episode_start: Optional[Any] = None):
        """Return `(action, None)`, matching the SB3 `predict` signature.

        `deterministic` is accepted and ignored: the controller has no
        stochastic component, so it is deterministic either way.
        """
        chi_los = self._los_heading_error(obs)
        deflection, frontal = self._repulsion(obs)

        if frontal > 1e-9:
            side = self._side_preference(obs)
            deflection += float(self.p["k_headon"]) * frontal * side
        else:
            # Clear ahead again: release the commitment so the next encounter
            # is decided on its own merits.
            self.committed_side = 0.0

        deflection = float(np.clip(deflection, -self.p["max_rep_deg"],
                                   self.p["max_rep_deg"]))
        heading_error = chi_los + deflection

        # --- PID on heading error, derivative taken on yaw rate ---
        yaw_rate = self._scalar(obs, "yaw_rate")
        raw = (float(self.p["kp"]) * heading_error
               + float(self.p["ki"]) * self.integral
               - float(self.p["kd"]) * yaw_rate)
        rudder = float(np.clip(raw, -1.0, 1.0))

        # Conditional integration: stop winding up while the command saturates.
        if abs(raw) < 1.0:
            limit = float(self.p["integral_limit"])
            self.integral = float(np.clip(
                self.integral + heading_error * 0.1, -limit, limit))

        # --- speed: back off near obstacles and when badly misaligned ---
        max_closeness = float(np.max(np.asarray(obs["lidar"], dtype=np.float64)))
        heading_penalty = min(abs(heading_error) / 90.0, 1.0)
        throttle = (float(self.p["throttle_base"])
                    - float(self.p["k_speed_obs"]) * max_closeness
                    - float(self.p["k_speed_head"]) * heading_penalty)
        throttle = float(np.clip(throttle, self.p["min_throttle"], 1.0))

        action = np.array([rudder, throttle], dtype=np.float32)
        self.last_action = action
        return action, None

    # ------------------------------------------------------------------
    def params(self) -> Dict[str, Any]:
        return dict(self.p)

    def __repr__(self) -> str:
        changed = {k: v for k, v in self.p.items() if v != DEFAULTS[k]}
        return f"LosApfController({changed or 'defaults'})"
