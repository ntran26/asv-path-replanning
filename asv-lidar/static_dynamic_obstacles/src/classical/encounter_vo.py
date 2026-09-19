"""Encounter-specific velocity obstacles (the third classical comparator, B8).

After Thyri & Breivik (2022): a velocity-obstacle search whose constraints
depend on the encounter the own ship is in, rather than one COLREGs rule set
applied to every target (Kuwata et al. 2014, the COLREGs-VO comparator, which
is not built here).  **This is our reading of that method, not a port of
their code**; every choice below is a named constant.

Search space: straight-line velocities, course x speed -- courses around the
present heading and the LOS course, speeds from stop to cruise.  For each
candidate and each tracked target (constant velocity, over `TAU_S`):

* **hard VO** -- the own hull, held on the candidate course, comes within
  `HARD_GAP_M` of the target's hull (separating-axis gap).  Excluded always.
* **encounter domain (soft)** -- the target enters the own ship's domain
  (`DOMAIN_FORE` / `DOMAIN_AFT` / `DOMAIN_LATERAL`, 01 §5.2), scaled per
  encounter class.  Penalised in proportion to the deepest penetration, so
  in a narrow channel a small intrusion is allowed where a clear pass is not
  available -- the confined-water relaxation that motivates the method.
* **passing side (hard, released in extremis)** -- give-way classes only:
    - head-on: the target passes down the own ship's port side (Rule 14);
    - crossing: pass astern of the target (Rules 15/16; this project's S3
      makes the own ship give way from either side, A17);
    - overtaking: pass on the side `compliant_turn_sense` names (Rule 13).
  A candidate whose CPA is outside `SIDE_FREE_DCPA_M` is free of it.
* **static margin (soft)** -- scan points and the map inside `STATIC_SOFT_M`
  are penalised, as the domain is for targets;
* **stand-on** -- being overtaken: hold the LOS course at cruise (Rule 17(a))
  while that velocity is outside every hard VO within `STAND_ON_RELEASE_S`;
  otherwise act as for any other obstacle (Rule 17(b)).

Static returns and the map polygon are velocity obstacles too, over the
shorter `TAU_STATIC_S`.

**Reachability.**  A textbook VO assumes the new velocity is taken at once.
On this hull that assumption is badly wrong (full rudder builds ~6 deg/s only
after ~5 s), and the first version showed the consequence: every decision the
VO found a slightly larger alteration "still clear", the heading barely moved,
and it procrastinated into static obstacles.  Each candidate is therefore
swept along a simple manoeuvre model -- hold the present heading for
`TURN_LAG_S`, turn at `TURN_RATE_DPS` onto the candidate course, speed
approaching the candidate with the hull's surge lags -- and every VO
test runs on that swept path.  The passing-side test keeps the candidate
velocity itself, since it states intent, not the transient.

Cost (lower is better): deviation from the LOS course, from cruise speed,
from the previous choice, plus the domain penalty.  The chosen `(course,
speed)` is then flown by a cascaded course autopilot (at the sweep's turn
rate) and the speed loop the LOS-PID comparator uses.

`select_velocity` is free of the environment on purpose: 03a §5.3 asks that
`T-RE`'s reactive target and this comparator be one implementation, so a
target can call it with its own state and the own ship as its one obstacle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

import constants as cfg
from classical import common as cc
from reference_controller import hull_separation
from ship import HULL_MARGIN

COURSE_OFFSETS_DEG = np.arange(-90.0, 90.01, 5.0)   # about the present heading
SPEED_FRACTIONS = (1.0, 0.75, 0.5, 0.25, 0.0)
TAU_S = 20.0                     # VO horizon for targets
TAU_STATIC_S = 10.0              # and for scan points and the map polygon
VO_DT = 0.25
HARD_GAP_M = 0.20
STATIC_GAP_M = 0.15
BOUNDARY_GAP_M = 0.05
SIDE_FREE_DCPA_M = 2.0 * cfg.DOMAIN_LATERAL          # 2.5 m: the pass is clear either side
# Rule 17(a)(ii)/(b): stand on until a hard violation is this close.  8 s was
# too late for this hull to sidestep a faster overtaker coming up its track
# (no clear velocity left by the time it released); 12 s leaves one.
STAND_ON_RELEASE_S = 12.0
# Manoeuvre model for the reachability sweep, from the identified hull at
# cruise (probed in simulation): ~1 s of rudder delay, servo and yaw lag
# before a turn is under way, ~6 deg/s at full rudder; on the propulsion
# floor it coasts 0.55 -> 0.41 m/s in 20 s (an effective time constant near
# 60 s), and at double thrust gains 0.55 -> 1.0 m/s in 20 s (~20 s).
TURN_LAG_S = 1.0
TURN_RATE_DPS = 6.0
SPEED_TAU_DOWN_S = 60.0
SPEED_TAU_UP_S = 20.0

# Domain scaling per encounter, (fore, aft, lateral).  Give-way crossing keeps
# the target further off the own bow (a clear pass astern); head-on and
# overtaking widen abeam, where those passes happen.
DOMAIN_SCALE = {
    "head_on": (1.0, 1.0, 1.2),
    "crossing": (1.5, 1.0, 1.0),
    "overtaking": (1.0, 1.0, 1.2),
    "being_overtaken": (1.0, 1.0, 1.0),
    "none": (1.0, 1.0, 1.0),
}

W_COURSE = 1.0                   # per pi rad from the preferred course
W_SPEED = 0.5                    # per cruise speed below preferred
W_CHANGE = 0.3                   # per pi rad from the previous course
W_DOMAIN = 3.0                   # per unit of domain penetration
# Static returns: the hard gap plus a soft margin, the static counterpart of the
# domain.  With the hard gap alone the VO scraped past obstacles on the margin
# and flipped back to the LOS course the moment it looked clear again.
STATIC_SOFT_M = 0.6
W_STATIC = 1.0                   # at the hard gap; zero at STATIC_SOFT_M


@dataclass
class Obstacle:
    """A moving vessel as the VO sees it."""

    position: np.ndarray
    velocity: np.ndarray
    heading: float               # rad, compass
    cls: str = "none"            # encounter class from the own ship's view
    turn_sense: int = 0          # +1 starboard / -1 port, for overtaking
    in_extremis: bool = False
    engaged: bool = False


@dataclass
class Choice:
    course: float                # rad
    speed: float                 # m/s
    cost: float
    feasible: bool
    released: bool               # passing-side constraints dropped
    stand_on: bool


def sweep(position: np.ndarray, heading: float, speed: float, courses: np.ndarray,
          speeds: np.ndarray, times: np.ndarray):
    """Positions (K, n, 2) and headings (K, n) along the manoeuvre model."""
    turn = cc.wrap_pi(courses - heading)
    progress = np.radians(TURN_RATE_DPS) * np.maximum(0.0, times[:, None] - TURN_LAG_S)
    hdg = heading + np.sign(turn) * np.minimum(np.abs(turn), progress)                 # (K, n)
    tau = np.where(speeds < speed, SPEED_TAU_DOWN_S, SPEED_TAU_UP_S)
    spd = speeds + (speed - speeds) * np.exp(-times[:, None] / tau)
    dt = np.diff(times, prepend=0.0)[:, None]
    step = (spd * dt)[..., None] * np.stack([np.sin(hdg), np.cos(hdg)], axis=-1)
    return position + np.cumsum(step, axis=0), hdg


def select_velocity(position: np.ndarray, heading: float, speed: float, chi_pref: float,
                    u_pref: float, obstacles: Sequence[Obstacle], *,
                    prev_course: Optional[float] = None, static_clearance=None) -> Choice:
    """Pick a velocity outside the encounter-specific VOs.

    `static_clearance(positions (K, n, 2), headings (K, n)) -> (K, n)` adds the
    static VOs; `None` when the caller has none (a target in open water).
    """
    courses = np.concatenate([heading + np.radians(COURSE_OFFSETS_DEG), [chi_pref]])
    courses, speeds = np.meshgrid(courses, np.asarray(SPEED_FRACTIONS) * u_pref)
    courses, speeds = cc.wrap_pi(courses.ravel()), speeds.ravel()
    n = len(courses)
    vel = speeds[:, None] * np.stack([np.sin(courses), np.cos(courses)], axis=1)      # (n, 2)
    times = VO_DT * np.arange(1, int(TAU_S / VO_DT) + 1)
    own, own_h = sweep(position, heading, speed, courses, speeds, times)               # (K, n, 2)

    hard_t = np.full(n, np.inf)              # first time inside a hard VO
    side_bad = np.zeros(n, dtype=bool)
    domain_pen = np.zeros(n)
    stand_on, extremis = False, False
    for ob in obstacles:
        future = ob.position + times[:, None] * ob.velocity                            # (K, 2)
        gap = hull_separation(own, own_h, future[:, None, :], ob.heading, margin=HULL_MARGIN)
        hit = gap < HARD_GAP_M
        hard_t = np.minimum(hard_t, np.where(hit.any(axis=0), times[np.argmax(hit, axis=0)], np.inf))

        rel = future[:, None, :] - own                                                  # target from own
        fwd = np.stack([np.sin(own_h), np.cos(own_h)], axis=-1)
        stbd = np.stack([np.cos(own_h), -np.sin(own_h)], axis=-1)
        lon = np.sum(rel * fwd, axis=-1)
        lat = np.sum(rel * stbd, axis=-1)
        f, a, l = DOMAIN_SCALE.get(ob.cls, DOMAIN_SCALE["none"])
        ax_lon = np.where(lon >= 0.0, f * cfg.DOMAIN_FORE, a * cfg.DOMAIN_AFT)
        ell = np.hypot(lon / ax_lon, lat / (l * cfg.DOMAIN_LATERAL))
        domain_pen = np.maximum(domain_pen, np.max(np.maximum(0.0, 1.0 - ell), axis=0))

        extremis |= bool(ob.in_extremis)
        if ob.cls == "being_overtaken":
            stand_on = True
        if ob.cls in ("head_on", "crossing", "overtaking"):
            side_bad |= _wrong_side(ob, position, vel, courses)

    def cost():
        c = (W_COURSE * np.abs(cc.wrap_pi(courses - chi_pref)) / np.pi
             + W_SPEED * np.maximum(0.0, u_pref - speeds) / max(u_pref, 1e-6)
             + W_DOMAIN * domain_pen)
        if prev_course is not None:
            c = c + W_CHANGE * np.abs(cc.wrap_pi(courses - prev_course)) / np.pi
        return c

    static_t = np.full(n, np.inf)
    static_pen = np.zeros(n)
    if static_clearance is not None:
        ks = times <= TAU_STATIC_S + 1e-9
        clear = static_clearance(own[ks], own_h[ks])
        bad = clear < STATIC_GAP_M
        static_t = np.where(bad.any(axis=0), times[ks][np.argmax(bad, axis=0)], np.inf)
        static_pen = np.maximum(0.0, STATIC_SOFT_M - clear.min(axis=0)) / (STATIC_SOFT_M - STATIC_GAP_M)
    safe = (hard_t == np.inf) & (static_t == np.inf)
    J = cost() + W_STATIC * static_pen

    # Rule 17: stand on while the preferred velocity is clear for long enough.
    pref = int(np.argmin(np.abs(cc.wrap_pi(courses - chi_pref)) + np.abs(speeds - u_pref)))
    if stand_on and min(hard_t[pref], static_t[pref]) > STAND_ON_RELEASE_S and not side_bad[pref]:
        return Choice(float(courses[pref]), float(speeds[pref]), float(J[pref]), True, False, True)

    # Rule 17(b) / 8: in extremis, or with no lawful clear velocity, any clear one.
    ok, released = safe & ~side_bad, False
    if extremis or not np.any(ok):
        ok, released = safe, True
    if np.any(ok):
        best = int(np.argmin(np.where(ok, J, np.inf)))
        return Choice(float(courses[best]), float(speeds[best]), float(J[best]), True, released, False)
    # Nothing clear: the latest first violation, then cost.  Not "slowest":
    # on the propulsion floor this hull coasts for tens of metres and loses
    # the rudder authority it needs, and the sweep already charges slowing
    # at that rate.
    first = np.minimum(hard_t, static_t)
    best = int(np.lexsort((J, -first))[0])
    return Choice(float(courses[best]), float(speeds[best]), float(J[best]), False, True, False)


def _wrong_side(ob: Obstacle, position, vel, courses) -> np.ndarray:
    """Candidates that would pass the target on the side its class forbids."""
    p_rel = ob.position - position                     # target from own, now
    v_rel = ob.velocity - vel                          # (n, 2)
    vv = np.maximum(np.sum(v_rel * v_rel, axis=1), 1e-9)
    t_cpa = np.clip(-(v_rel @ p_rel) / vv, 0.0, TAU_S)
    cpa = p_rel + t_cpa[:, None] * v_rel               # target from own at the CPA
    dcpa = np.linalg.norm(cpa, axis=1)
    approaching = (v_rel @ p_rel) < 0.0
    constrained = approaching & (dcpa < SIDE_FREE_DCPA_M)
    stbd = np.stack([np.cos(courses), -np.sin(courses)], axis=1)
    target_to_starboard = np.sum(cpa * stbd, axis=1) > 0.0
    if ob.cls == "head_on":
        wrong = target_to_starboard                    # must pass port to port
    elif ob.cls == "crossing":
        speed = float(np.hypot(*ob.velocity))
        if speed < 0.05:
            return np.zeros(len(courses), dtype=bool)
        t_hat = ob.velocity / speed
        wrong = (-cpa) @ t_hat > 0.0                   # own ahead of the target at the CPA
    else:                                              # overtaking
        sense = int(ob.turn_sense) or -1
        # A port pass (sense -1) leaves the target to starboard; a starboard pass to port.
        wrong = target_to_starboard if sense > 0 else ~target_to_starboard
    return constrained & wrong


class EncounterVOController:
    name = "encounter_vo"

    def __init__(self):
        self.perception = cc.Perception()
        self.actuators = cc.Actuators()
        self.prev_course: Optional[float] = None
        self.last_diagnostics = {}

    def action(self, env, observation=None) -> np.ndarray:
        snap = self.perception.snapshot(env)
        chi_los = float(cc.los_heading(snap, sideslip=0.0))
        obstacles = [self._obstacle(t) for t in snap.tracks]

        def static(pos, hdg):
            c = cc.point_clearance(pos, hdg, snap.points, reach=cc.HALF_L + STATIC_GAP_M + 0.5)
            b = cc.boundary_clearance(pos, hdg, snap.edges_a, snap.edges_b)
            # The polygon is exact; hold it to its own smaller margin.
            return np.minimum(c, b + (STATIC_GAP_M - BOUNDARY_GAP_M))

        choice = select_velocity(snap.position, snap.heading, snap.u, chi_los, cfg.U_NOM, obstacles,
                                 prev_course=self.prev_course, static_clearance=static)
        self.prev_course = choice.course
        self.last_diagnostics = {"course_offset_deg": math.degrees(float(cc.wrap_pi(choice.course - chi_los))),
                                 "speed": choice.speed, "feasible": choice.feasible,
                                 "released": choice.released, "stand_on": choice.stand_on,
                                 "encounters": [(o.cls, o.turn_sense) for o in obstacles]}
        # The VO chose a ground course; the course autopilot flies a heading,
        # at the turn rate the reachability sweep assumed.
        desired = choice.course - snap.sideslip
        rudder = float(cc.course_rudder(float(cc.wrap_pi(desired - snap.heading)), snap.r,
                                        max_rate_dps=TURN_RATE_DPS))
        rpm = float(cc.speed_rpm(choice.speed, snap.u))
        self.actuators.issue(env, rudder)
        return np.array([rudder, cc.throttle_for_rpm(rpm)], dtype=np.float32)

    @staticmethod
    def _obstacle(track: cc.TrackView) -> Obstacle:
        ctx = track.ctx
        if ctx is None:
            return Obstacle(track.position, track.velocity, track.heading)
        return Obstacle(track.position, track.velocity, track.heading, str(ctx.cls),
                        int(getattr(ctx, "compliant_turn_sense", 0)) if str(ctx.cls) == "overtaking" else 0,
                        bool(getattr(ctx, "in_extremis", False)), bool(getattr(ctx, "engaged", False)))
