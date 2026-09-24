"""COLREGs-VO (Kuwata et al., 2014) — the published comparator.

Kuwata, Wolf, Zarzhitsky and Huntsberger, *Safe Maritime Autonomous Navigation
With COLREGS, Using Velocity Obstacles*, IEEE J. Oceanic Eng. 39(1), 2014.

The method, as published:

1. **Candidate velocities.** A set of (course, speed) pairs around the present
   state is sampled.
2. **Velocity obstacles.** A candidate is rejected if, holding it, the own hull
   would come within the hard gap of a target inside the horizon, the targets
   travelling at constant velocity.
3. **The COLREGS constraint.** The encounter is classified **by the open-water
   rules** — head-on, crossing with the other to starboard (give-way), crossing
   with the other to port (stand-on), overtaking, being overtaken — and where
   the own ship gives way, the candidate must put the **relative velocity to
   starboard of the line of sight**: the vessel alters to starboard and passes
   astern. That single sign test is the paper's contribution; everything else is
   ordinary VO.
4. **Stand-on.** Under Rule 17 the own ship holds course and speed, and acts
   only when the give-way vessel plainly has not.
5. **Safety first.** If nothing satisfies both the VO and the COLREGS
   constraint, the constraint is dropped rather than the collision accepted --
   Kuwata's own fallback, and the behaviour the narrow-channel claim is about.

**This is deliberately not `encounter_vo.py`.** That comparator follows *this
paper's* narrow-channel convention (A17: give way to a crossing target from
either side, turn toward its side). Kuwata follows the open-water role table:
a crossing target to port makes the own ship **stand on**. Where the channel is
too narrow for a starboard alteration, the difference between the two is the
measurement the Rule 9 precedence claim rests on, so both are kept.

Perception, actuators, the LOS reference and the static-obstacle treatment are
shared with the other comparators (`classical/common.py`), so what differs
between comparator rows is the avoidance logic and nothing else.

Parameters live in `src/constant_temp.py` (`CLASSICAL_KVO_*`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

import constant_temp as ct
import constants as cfg
from classical import common as cc
from classical.encounter_vo import sweep
from reference_controller import hull_separation
from ship import HULL_MARGIN

# Encounter classification (open water, Rules 13-17).
HEAD_ON_HALF_DEG = ct.CLASSICAL_KVO_HEAD_ON_HALF_DEG
OVERTAKING_DEG = ct.CLASSICAL_KVO_OVERTAKING_DEG
# Candidate set and horizons.
COURSE_OFFSETS_DEG = np.asarray(ct.CLASSICAL_KVO_COURSE_OFFSETS_DEG)
SPEED_FRACTIONS = ct.CLASSICAL_KVO_SPEED_FRACTIONS
TAU_S = ct.CLASSICAL_KVO_TAU_S
TAU_STATIC_S = ct.CLASSICAL_KVO_TAU_STATIC_S
VO_DT = ct.CLASSICAL_KVO_DT
# Clearances.
HARD_GAP_M = ct.CLASSICAL_KVO_HARD_GAP_M
STATIC_GAP_M = ct.CLASSICAL_KVO_STATIC_GAP_M
BOUNDARY_GAP_M = ct.CLASSICAL_KVO_BOUNDARY_GAP_M
SIDE_FREE_DCPA_M = ct.CLASSICAL_KVO_SIDE_FREE_DCPA_M
# Rule 17 and the manoeuvre model.
STAND_ON_RELEASE_S = ct.CLASSICAL_KVO_STAND_ON_RELEASE_S
TURN_RATE_DPS = ct.CLASSICAL_KVO_TURN_RATE_DPS
# Cost weights.
W_COURSE = ct.CLASSICAL_KVO_W_COURSE
W_SPEED = ct.CLASSICAL_KVO_W_SPEED
W_CHANGE = ct.CLASSICAL_KVO_W_CHANGE
STATIC_SOFT_M = ct.CLASSICAL_KVO_STATIC_SOFT_M
W_STATIC = ct.CLASSICAL_KVO_W_STATIC

HEAD_ON, CROSSING_GIVE, CROSSING_STAND, OVERTAKING, BEING_OVERTAKEN, NONE = (
    "head_on", "crossing_give_way", "crossing_stand_on", "overtaking",
    "being_overtaken", "none")
GIVE_WAY = (HEAD_ON, CROSSING_GIVE, OVERTAKING)


@dataclass
class Target:
    position: np.ndarray
    velocity: np.ndarray
    heading: float                  # rad, compass
    cls: str = NONE


def classify(p_own: np.ndarray, heading: float, u_own: float,
             p_ts: np.ndarray, v_ts: np.ndarray, heading_ts: float) -> str:
    """The open-water role, from geometry alone (Rules 13-17).

    `alpha` is the target's bearing relative to the own heading; `beta` is the
    own ship's bearing relative to the target's heading. Both in degrees,
    wrapped to +/-180.
    """
    los = p_ts - p_own
    if not np.any(np.abs(los) > 1e-9):
        return NONE
    alpha = math.degrees(cc.wrap_pi(math.atan2(los[0], los[1]) - heading))
    beta = math.degrees(cc.wrap_pi(math.atan2(-los[0], -los[1]) - heading_ts))
    speed_ts = float(np.hypot(*v_ts))

    if abs(alpha) >= OVERTAKING_DEG:
        return BEING_OVERTAKEN                      # it comes up from astern: stand on
    if abs(beta) >= OVERTAKING_DEG and u_own > speed_ts:
        return OVERTAKING                           # we come up from astern: keep clear
    if abs(alpha) <= HEAD_ON_HALF_DEG and abs(beta) <= HEAD_ON_HALF_DEG:
        return HEAD_ON                              # reciprocal: both alter to starboard
    return CROSSING_GIVE if alpha > 0.0 else CROSSING_STAND


def _colregs_ok(target: Target, p_own: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Kuwata's constraint: as give-way, keep the relative velocity to starboard
    of the line of sight, which is "alter to starboard and pass astern".

    Applied only while the encounter is live -- approaching, and closing inside
    `SIDE_FREE_DCPA_M`. A pass that is already wide is not constrained, or the
    own ship would keep altering for a vessel it is never going to meet.
    """
    los = target.position - p_own
    v_rel = vel - target.velocity                                  # (n, 2), own w.r.t. target
    # The bearing line closes at `-v_rel` (the target's motion as the own ship
    # sees it), so approaching is `los . v_rel > 0`, not the other way round.
    vv = np.maximum(np.sum(v_rel * v_rel, axis=1), 1e-9)
    closing = (v_rel @ los) > 0.0
    t_cpa = np.clip((v_rel @ los) / vv, 0.0, TAU_S)
    dcpa = np.linalg.norm(los - t_cpa[:, None] * v_rel, axis=1)
    live = closing & (dcpa < SIDE_FREE_DCPA_M)
    # cross(los, v_rel) < 0 puts the relative velocity to starboard of the
    # bearing line (x east, y north, heading clockwise from north).
    cross = los[0] * v_rel[:, 1] - los[1] * v_rel[:, 0]
    return ~live | (cross < 0.0)


@dataclass
class Choice:
    course: float
    speed: float
    cost: float
    feasible: bool
    released: bool              # COLREGS constraint dropped to stay clear
    stand_on: bool


def select_velocity(position, heading, speed, chi_pref, u_pref, targets: Sequence[Target], *,
                    prev_course: Optional[float] = None, static_clearance=None) -> Choice:
    courses = np.concatenate([heading + np.radians(COURSE_OFFSETS_DEG), [chi_pref]])
    courses, speeds = np.meshgrid(courses, np.asarray(SPEED_FRACTIONS) * u_pref)
    courses, speeds = cc.wrap_pi(courses.ravel()), speeds.ravel()
    n = len(courses)
    vel = speeds[:, None] * np.stack([np.sin(courses), np.cos(courses)], axis=1)
    times = VO_DT * np.arange(1, int(TAU_S / VO_DT) + 1)
    own, own_h = sweep(position, heading, speed, courses, speeds, times)

    hard_t = np.full(n, np.inf)
    forbidden = np.zeros(n, dtype=bool)
    stand_on = False
    for tgt in targets:
        future = tgt.position + times[:, None] * tgt.velocity
        gap = hull_separation(own, own_h, future[:, None, :], tgt.heading, margin=HULL_MARGIN)
        hit = gap < HARD_GAP_M
        hard_t = np.minimum(hard_t, np.where(hit.any(axis=0), times[np.argmax(hit, axis=0)], np.inf))
        if tgt.cls in GIVE_WAY:
            forbidden |= ~_colregs_ok(tgt, position, vel)
        if tgt.cls in (CROSSING_STAND, BEING_OVERTAKEN):
            stand_on = True

    static_t = np.full(n, np.inf)
    static_pen = np.zeros(n)
    if static_clearance is not None:
        ks = times <= TAU_STATIC_S + 1e-9
        clear = static_clearance(own[ks], own_h[ks])
        bad = clear < STATIC_GAP_M
        static_t = np.where(bad.any(axis=0), times[ks][np.argmax(bad, axis=0)], np.inf)
        # The same soft margin the other comparators keep: rejecting only hard
        # violations leaves the choice indifferent between grazing an obstacle
        # and clearing it, and the controller takes the grazing one whenever it
        # is closer to the LOS course.
        static_pen = np.maximum(0.0, STATIC_SOFT_M - clear.min(axis=0)) / (STATIC_SOFT_M - STATIC_GAP_M)

    cost = (W_COURSE * np.abs(cc.wrap_pi(courses - chi_pref)) / np.pi
            + W_SPEED * np.maximum(0.0, u_pref - speeds) / max(u_pref, 1e-6)
            + W_STATIC * static_pen)
    if prev_course is not None:
        cost = cost + W_CHANGE * np.abs(cc.wrap_pi(courses - prev_course)) / np.pi

    safe = (hard_t == np.inf) & (static_t == np.inf)
    pref = int(np.argmin(np.abs(cc.wrap_pi(courses - chi_pref)) + np.abs(speeds - u_pref)))
    # Rule 17(a)(i): hold course and speed while that stays clear for long enough.
    if stand_on and min(hard_t[pref], static_t[pref]) > STAND_ON_RELEASE_S:
        return Choice(float(courses[pref]), float(speeds[pref]), float(cost[pref]), True, False, True)

    ok = safe & ~forbidden
    if np.any(ok):
        best = int(np.argmin(np.where(ok, cost, np.inf)))
        return Choice(float(courses[best]), float(speeds[best]), float(cost[best]), True, False, False)
    # Kuwata's fallback: drop the COLREGS constraint before accepting a collision.
    if np.any(safe):
        best = int(np.argmin(np.where(safe, cost, np.inf)))
        return Choice(float(courses[best]), float(speeds[best]), float(cost[best]), True, True, False)
    first = np.minimum(hard_t, static_t)
    best = int(np.lexsort((cost, -first))[0])
    return Choice(float(courses[best]), float(speeds[best]), float(cost[best]), False, True, False)


class ColregsVOController:
    name = "colregs_vo"

    def __init__(self):
        self.perception = cc.Perception()
        self.actuators = cc.Actuators()
        self.prev_course: Optional[float] = None
        self.last_diagnostics = {}

    def action(self, env, observation=None) -> np.ndarray:
        snap = self.perception.snapshot(env)
        chi_los = float(cc.los_heading(snap, sideslip=0.0))
        targets = []
        for track in snap.tracks:
            cls = classify(snap.position, snap.heading, snap.u,
                           track.position, track.velocity, track.heading)
            targets.append(Target(track.position, track.velocity, track.heading, cls))

        def static(pos, hdg):
            c = cc.point_clearance(pos, hdg, snap.points, reach=cc.HALF_L + STATIC_GAP_M + 0.5)
            b = cc.boundary_clearance(pos, hdg, snap.edges_a, snap.edges_b)
            return np.minimum(c, b + (STATIC_GAP_M - BOUNDARY_GAP_M))

        choice = select_velocity(snap.position, snap.heading, snap.u, chi_los, cfg.U_NOM,
                                 targets, prev_course=self.prev_course, static_clearance=static)
        self.prev_course = choice.course
        self.last_diagnostics = {
            "course_offset_deg": math.degrees(float(cc.wrap_pi(choice.course - chi_los))),
            "speed": choice.speed, "feasible": choice.feasible,
            "released": choice.released, "stand_on": choice.stand_on,
            "encounters": [t.cls for t in targets]}
        desired = choice.course - snap.sideslip
        rudder = float(cc.course_rudder(float(cc.wrap_pi(desired - snap.heading)), snap.r,
                                        max_rate_dps=TURN_RATE_DPS))
        rpm = float(cc.speed_rpm(choice.speed, snap.u))
        self.actuators.issue(env, rudder)
        return np.array([rudder, cc.throttle_for_rpm(rpm)], dtype=np.float32)
