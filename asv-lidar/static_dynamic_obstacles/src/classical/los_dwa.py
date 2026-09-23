"""LOS-PID + DWA (the first classical comparator, B8; 04 §5).

LOS guidance and a heading PID follow the path.  A dynamic window approach
(Fox, Burgard & Thrun 1997) sits over it and chooses the velocity to execute
whenever something is within reach:

* **search space** -- yaw-rate and speed set-points `(r_d, u_d)` across the
  vessel's limits, plus the LOS-PID command itself at two speeds, so plain
  path following is always one of the candidates;
* **dynamic window** -- each candidate is predicted closed-loop on the
  identified hull (rudder delay, servo, surge and yaw lags) rather than as an
  ideal constant-curvature arc.  On a hull that needs ~5 s to build its yaw
  rate, a one-step `(u, r)` window around the present velocity would admit
  arcs the vessel cannot fly; predicting through the dynamics admits exactly
  the trajectories it can.  A set-point is held for a commit time
  (`COMMIT_S`), after which the prediction hands back to LOS-PID -- which is
  what the controller will do once the obstacle is passed.  Held for the
  whole horizon, every arc ends pointing far from the path and the heading
  term cannot tell them apart (the first version chattered between them);
* **admissibility** -- a candidate whose predicted hull comes within
  `SAFE_GAP_M` of a scan point or the map polygon over its first `TRAVEL_M`
  of track, or of a tracked target (constant velocity) within
  `TARGET_HORIZON_S`, is inadmissible;
* **objective** -- Fox's `alpha * heading + beta * dist + gamma * velocity`,
  with the heading term measured against the LOS heading at the predicted end
  point (so a trajectory aimed back at the path scores well), plus two terms
  Fox did not need: path deviation (`DELTA`, the mean cross-track error over
  the prediction -- the goal here is a path, not a point, and without it a
  "turn away, return later" candidate is re-chosen every step and the return
  never comes) and smoothness against chattering between set-points.

**No COLREGs.**  DWA knows obstacles, not rules; the comparator exists to show
what collision avoidance without them does in these encounters.  If no
candidate is admissible it takes the one whose first violation comes latest
and, among those, the clearest -- DWA's "brake" fallback, adapted to a hull
that cannot brake quickly.
"""
from __future__ import annotations

import math

import numpy as np

import constant_temp as ct
import constants as cfg
from classical import common as cc

TRAVEL_M = ct.CLASSICAL_DWA_TRAVEL_M          # static obstacles, over this much track
TARGET_HORIZON_S = ct.CLASSICAL_DWA_TARGET_HORIZON_S   # tracked targets, over this much time
MAX_HORIZON_S = ct.CLASSICAL_DWA_MAX_HORIZON_S         # rollout length for the slowest candidate
YAW_RATES_DPS = np.asarray(ct.CLASSICAL_DWA_YAW_RATES_DPS)   # inside the full-rudder turn
SPEED_FRACTIONS = ct.CLASSICAL_DWA_SPEED_FRACTIONS     # of U_NOM; 0 = propulsion floor
COMMIT_S = ct.CLASSICAL_DWA_COMMIT_S
LOS_SPEED_FRACTIONS = ct.CLASSICAL_DWA_LOS_SPEED_FRACTIONS
SAFE_GAP_M = ct.CLASSICAL_DWA_SAFE_GAP_M      # hull-to-hull or hull-to-point, beyond HULL_MARGIN
BOUNDARY_GAP_M = ct.CLASSICAL_DWA_BOUNDARY_GAP_M   # the map polygon is exact, the scan is not
DIST_CAP_M = ct.CLASSICAL_DWA_DIST_CAP_M      # clearance beyond this earns nothing more
# Clearance is judged over equal travelled distance (below), so it no longer
# rewards slowing down, and can carry real weight against the path term.
ALPHA, BETA, GAMMA, DELTA, SMOOTH = 1.0, 1.0, 0.5, 0.6, 0.25
PATH_SCALE_M = ct.CLASSICAL_DWA_PATH_SCALE_M  # mean cross-track error scoring zero on DELTA
# DWA runs when anything is this close; otherwise LOS-PID alone.
ENGAGE_RANGE_M = ct.CLASSICAL_DWA_ENGAGE_RANGE_M


class LosDwaController:
    name = "los_dwa"

    def __init__(self):
        self.perception = cc.Perception()
        self.actuators = cc.Actuators()
        self.pid = cc.HeadingPID()
        self.prev_choice = None                    # (r_d dps or None for LOS, u fraction)
        self.last_diagnostics = {}

    # ------------------------------------------------------------------
    def action(self, env, observation=None) -> np.ndarray:
        snap = self.perception.snapshot(env)
        if not self._threat_in_reach(snap):
            self.prev_choice = None
            self.last_diagnostics = {"mode": "los", "admissible": None}
            return self._los_command(env, snap, 1.0)

        grid_r, grid_u, grid_c = np.meshgrid(np.radians(YAW_RATES_DPS), SPEED_FRACTIONS, COMMIT_S, indexing="ij")
        r_des = np.concatenate([grid_r.ravel(), np.full(len(LOS_SPEED_FRACTIONS), np.nan)])
        u_frac = np.concatenate([grid_u.ravel(), LOS_SPEED_FRACTIONS])
        commit = np.concatenate([grid_c.ravel(), np.zeros(len(LOS_SPEED_FRACTIONS))])
        is_los = np.isnan(r_des)
        u_des = u_frac * cfg.U_NOM
        n = len(r_des)

        def law(k, state):
            rudder_rate = cc.yaw_rate_rudder(np.nan_to_num(r_des), state[2])
            los = cc.los_heading(snap, state[5:7].T,
                                 np.clip(np.arctan2(state[1], np.maximum(0.15, state[0])),
                                         -cc.MAX_SIDESLIP_RAD, cc.MAX_SIDESLIP_RAD))
            rudder_los = cc.pd_rudder(cc.wrap_pi(los - state[3]), state[2])
            holding = k * cfg.UPDATE_RATE < commit
            return cc.speed_rpm(u_des, state[0]), np.where(holding, rudder_rate, rudder_los)

        ro = cc.rollout(snap, self.actuators, n, MAX_HORIZON_S, law)
        # Static obstacles and the map are judged over the same travelled
        # distance for every candidate, not the same time: in a fixed time
        # window a slow candidate looks clear just by not arriving, and the
        # first versions crawled up to obstacles doing exactly that.  Fox's
        # `dist` is likewise a distance along the arc.
        step = np.linalg.norm(np.diff(ro.positions, axis=0, prepend=np.broadcast_to(snap.position, (1, n, 2))), axis=-1)
        window = np.cumsum(step, axis=0) <= TRAVEL_M
        window[0] = True
        last = window.shape[0] - 1 - np.argmax(window[::-1], axis=0)
        cols = np.arange(n)
        clearance, first_violation = self._clearance(snap, ro, window)

        end_pos, end_heading = ro.positions[last, cols], ro.headings[last, cols]
        heading_score = 1.0 - np.abs(cc.wrap_pi(cc.los_heading(snap, end_pos) - end_heading)) / np.pi
        dist_score = np.clip(clearance, 0.0, DIST_CAP_M) / DIST_CAP_M
        timed = ro.times <= TARGET_HORIZON_S
        vel_score = np.clip(np.mean(ro.speeds[timed], axis=0) / cfg.U_NOM, 0.0, 1.0)
        lateral = np.abs((ro.positions - snap.centre) @ snap.right)
        path_score = 1.0 - np.clip(np.sum(lateral * window, axis=0) / np.maximum(window.sum(axis=0), 1)
                                   / PATH_SCALE_M, 0.0, 1.0)
        smooth = np.zeros(n)
        if self.prev_choice is not None:
            prev_r, prev_u = self.prev_choice
            same_kind = is_los if prev_r is None else ~is_los
            dr = 0.0 if prev_r is None else np.abs(np.degrees(np.nan_to_num(r_des)) - prev_r) / 18.0
            smooth = np.where(same_kind, dr, 1.0) + np.abs(u_frac - prev_u)
        score = (ALPHA * heading_score + BETA * dist_score + GAMMA * vel_score
                 + DELTA * path_score - SMOOTH * smooth)

        admissible = first_violation == np.inf
        if np.any(admissible):
            best = int(np.argmax(np.where(admissible, score, -np.inf)))
            fallback = False
        else:
            # Latest first violation, then most clearance.  Not "slowest", Fox's
            # stop: this hull coasts for tens of metres, and a candidate that
            # drops the speed also drops the rudder authority it needs.
            key = np.lexsort((-score, -clearance, -first_violation))
            best = int(key[0])
            fallback = True

        self.prev_choice = (None if is_los[best] else float(np.degrees(r_des[best])), float(u_frac[best]))
        self.last_diagnostics = {"mode": "dwa", "admissible": int(admissible.sum()), "fallback": fallback,
                                 "r_des_dps": None if is_los[best] else float(np.degrees(r_des[best])),
                                 "u_frac": float(u_frac[best]), "clearance_m": float(clearance[best])}
        if is_los[best]:
            return self._los_command(env, snap, float(u_frac[best]))
        rudder = float(cc.yaw_rate_rudder(r_des[best], snap.r))
        self.pid.reset()
        return self._issue(env, rudder, float(cc.speed_rpm(u_des[best], snap.u)))

    # ------------------------------------------------------------------
    def _threat_in_reach(self, snap) -> bool:
        if any(np.hypot(*(t.position - snap.position)) < ENGAGE_RANGE_M for t in snap.tracks):
            return True
        if len(snap.points):
            rel = snap.points - snap.position
            ahead = rel @ np.array([math.sin(snap.heading), math.cos(snap.heading)])
            near = np.linalg.norm(rel, axis=1)
            if np.any((near < ENGAGE_RANGE_M) & (ahead > -1.0)):
                return True
        # The map polygon: within a hull length of any edge.
        pos = snap.position[None, None, :]
        return bool(cc.boundary_clearance(pos, np.array([[snap.heading]]), snap.edges_a, snap.edges_b)[0, 0] < 1.0)

    def _clearance(self, snap, ro, window):
        """(min clearance, time of first violation) per candidate.

        Static returns and the map over `window` (the first TRAVEL_M of each
        trajectory); tracked targets over the first TARGET_HORIZON_S, since a
        moving obstacle's threat is a matter of time, not distance.
        """
        K, n = ro.headings.shape
        reach = cc.HALF_L + DIST_CAP_M
        c_static = np.minimum(cc.point_clearance(ro.positions, ro.headings, snap.points, reach=reach),
                              cc.boundary_clearance(ro.positions, ro.headings, snap.edges_a, snap.edges_b)
                              + (SAFE_GAP_M - BOUNDARY_GAP_M))
        c_static = np.where(window, c_static, np.inf)
        c_target = np.full((K, n), np.inf)
        timed = ro.times <= TARGET_HORIZON_S
        for track in snap.tracks:
            c_target = np.minimum(c_target, cc.target_gap(ro.positions, ro.headings, ro.times, track))
        c_target[~timed] = np.inf
        c = np.minimum(c_static, c_target)
        violated = c < SAFE_GAP_M
        first = np.where(violated.any(axis=0), ro.times[np.argmax(violated, axis=0)], np.inf)
        return c.min(axis=0), first

    def _los_command(self, env, snap, speed_fraction: float) -> np.ndarray:
        error = cc.wrap_pi(cc.los_heading(snap) - snap.heading)
        rudder = self.pid(float(error), snap.r)
        return self._issue(env, rudder, float(cc.speed_rpm(speed_fraction * cfg.U_NOM, snap.u)))

    def _issue(self, env, rudder: float, rpm: float) -> np.ndarray:
        self.actuators.issue(env, rudder)
        return np.array([rudder, cc.throttle_for_rpm(rpm)], dtype=np.float32)
