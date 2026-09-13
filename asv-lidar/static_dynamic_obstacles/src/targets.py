"""The dynamic target: hull geometry, behaviour models, confinement (03a §5).

Five behaviour models.  **Only `T-CV` is used in training** (decision D1); the
other four are evaluation-only, and the split matters — a policy trained against
reactive targets learns to rely on the other vessel co-operating, which is the
assumption COLREGs exists because you cannot make.

| ID | Model | Used in |
|---|---|---|
| `T-CV` | Constant velocity, constant heading | Training and evaluation |
| `T-RE` | Compliant reactive — encounter-specific VO | Evaluation only |
| `T-NC1` | Stands on when it is the give-way vessel | Evaluation only |
| `T-NC2` | Alters to **port** in a head-on | Evaluation only |
| `T-NC3` | Positionally non-compliant — holds the wrong side of the fairway | Evaluation only |

**`T-NC3` is new in 03a §5.3 and it is not optional.**  The head-on precedence
argument (02 §3.2) is that Rule 9(a) channel-keeping satisfies Rule 14 *without*
an alteration.  If every head-on target is where 9(a) says it should be, the
policy is never asked to execute Rule 14 at all, and the head-on class is never
exercised as an avoidance problem.  `T-NC3` is the only case in the suite where
a head-on alteration is actually required, which makes it the case that decides
whether the precedence argument is testable.

**Confinement is class-conditional (03a §5.2, 04a §1.3).**  This supersedes
03 §3's blanket "the target must respect the channel", which is unsatisfiable
for a crossing target: a vessel crossing a 4 m fairway at 90° runs into the far
wall.  The resolution is in the rulebook rather than the geometry — **Rule 9(d)
is specifically about vessels crossing a narrow channel**, and such a vessel is
not a channel user at all.  It is field-reproducible without modification,
because the corridor is a virtual map polygon inside a 10 m basin: a target can
physically cross a 4 m virtual corridor and continue into water the own ship
treats as non-navigable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

import constants as cfg

# Behaviour identifiers, as 03a §5.3 names them.
T_CV = "T-CV"
T_RE = "T-RE"
T_NC1 = "T-NC1"
T_NC2 = "T-NC2"
T_NC3 = "T-NC3"

BEHAVIOURS = (T_CV, T_RE, T_NC1, T_NC2, T_NC3)
TRAINING_BEHAVIOURS = (T_CV,)                      # D1
NON_COMPLIANT = (T_NC1, T_NC2, T_NC3)

# Which classes confine the target to the corridor (03a §5.2).
CONFINED_CLASSES = ("head_on", "overtaking", "being_overtaken", "null")
UNCONFINED_CLASSES = ("crossing",)                 # Rule 9(d)

# Oriented hull, body frame at midships, x forward, y starboard (03a §5.1).
# LOA 1.73 m, maximum breadth 0.50 m.  Own ship and target share this polygon --
# they are the same class of vessel, which is the premise on which S3 rejected
# the Rule 18 route in favour of Rule 9(b).
#
# TODO(03-5): replace with traced offsets if hull lines exist.  Centimetre
# accuracy matters only for the LiDAR return pattern, which feeds cluster-
# centroid bias in the tracker -- so it is a perception-fidelity question rather
# than a dynamics one.
HULL_OFFSETS = (
    (0.865, 0.000), (0.780, 0.130), (0.600, 0.215), (0.250, 0.250),
    (-0.550, 0.250), (-0.865, 0.230), (-0.865, -0.230), (-0.550, -0.250),
    (0.250, -0.250), (0.600, -0.215), (0.780, -0.130),
)


def hull_polygon(x: float, y: float, heading_deg: float,
                 offsets: Sequence = HULL_OFFSETS) -> List[Tuple[float, float]]:
    """`HULL_OFFSETS` placed in the world at a pose.

    Headings are compass-style: 0 is +y, clockwise positive.  Forward is
    `(sin psi, cos psi)` and starboard is `(cos psi, -sin psi)`.
    """
    a = math.radians(float(heading_deg))
    sin_a, cos_a = math.sin(a), math.cos(a)
    return [(float(x) + fwd * sin_a + lat * cos_a,
             float(y) + fwd * cos_a - lat * sin_a)
            for fwd, lat in offsets]


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------
@dataclass
class Target:
    """One COLREGs target vessel, with a behaviour model.

    Replaces the circle-plus-heading placeholder.  An oriented polygon is
    required for aspect-angle computation, for ship-domain metrics, and for a
    LiDAR return pattern that looks like a vessel rather than like a buoy --
    all three feed N1.
    """

    x: float
    y: float
    heading: float                       # deg, compass
    speed: float                         # m/s
    behaviour: str = T_CV
    encounter_class: str = "null"
    confined: bool = True
    corridor: object = None              # set for confined targets

    # Latched on first engagement, for the reactive and non-compliant models.
    _reacted: bool = field(default=False, repr=False)
    _t: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------
    @property
    def heading_deg(self) -> float:
        return float(self.heading)

    @property
    def velocity(self) -> np.ndarray:
        a = math.radians(self.heading)
        return np.array([self.speed * math.sin(a), self.speed * math.cos(a)])

    def hull(self) -> List[Tuple[float, float]]:
        return hull_polygon(self.x, self.y, self.heading)

    # ------------------------------------------------------------------
    def step(self, dt: float, *, own=None) -> None:
        """Advance one control step, applying the behaviour model.

        `own` is the own-ship state the reactive and non-compliant models need.
        Training targets (`T-CV`) ignore it entirely, which is why the training
        environment never has to supply it.
        """
        self._t += float(dt)
        if self.behaviour != T_CV and own is not None:
            self._react(dt, own)

        v = self.velocity
        self.x += float(v[0]) * dt
        self.y += float(v[1]) * dt

    # ------------------------------------------------------------------
    def _react(self, dt: float, own) -> None:
        """Behaviour models other than constant velocity.

        `T-NC3` is deliberately absent: it is a *positional* violation, fixed at
        spawn by placing the target on the wrong side of the fairway, and it
        holds course from there.  Encoding it as a manoeuvre would make it a
        behavioural violation and it would stop testing what it exists to test.
        """
        if self.behaviour in (T_NC1, T_NC3):
            return                        # stands on; the violation is the point

        import cpa_cri as cc
        p_os = (float(own["x"]), float(own["y"]))
        p_ts = (self.x, self.y)
        dcpa, tcpa = cc.cpa(p_os, own["velocity"], p_ts, self.velocity)

        engaged = 0.0 < tcpa < cfg.T_ENGAGE and dcpa < cfg.KAPPA_ENG * 2.0 * cfg.DOMAIN_LATERAL
        if not engaged:
            return

        if self.behaviour == T_NC2:
            # Alters to PORT in a head-on -- the classic wrong-way violation,
            # and the reason 04a §11.4 requires passing-side correctness to be
            # reported conditioned on target compliance.  Scored against the own
            # ship it would otherwise read as a policy failure.
            if self.encounter_class == "head_on":
                self._turn(-TURN_RATE_DPS * dt)
            return

        if self.behaviour == T_RE:
            # Compliant reactive: alter to starboard, which satisfies Rule 14
            # and Rule 15 alike in this domain.  Deliberately simple -- 03a §5.3
            # requires `T-RE` and the C3 velocity-obstacle comparator to be one
            # implementation, so the full VO lives with the comparator and this
            # is the interface it will be swapped into.
            self._turn(+TURN_RATE_DPS * dt)
            self._reacted = True

    def _turn(self, delta_deg: float) -> None:
        self.heading = (self.heading + float(delta_deg)) % 360.0


# Reactive-target turn rate.  A model vessel's sustained rate of turn, not a
# free parameter: it should come from 05's identified turning circle at the same
# time `R_REF` does.
TURN_RATE_DPS = 8.0                      # deg/s, TODO(05)


# ---------------------------------------------------------------------------
# Confinement
# ---------------------------------------------------------------------------
def is_confined(encounter_class: str) -> bool:
    """Does this class keep the target inside the corridor? (03a §5.2)"""
    return str(encounter_class) not in UNCONFINED_CLASSES


def confinement_violation(target: Target, corridor) -> Optional[float]:
    """How far a confined target has strayed outside its corridor, metres.

    `None` when the target is unconfined (crossing, under Rule 9(d)) or inside.
    Used by acceptance test T7, which asserts the rule holds both ways round:
    crossing targets *must* leave and every other class *must not*.
    """
    if not target.confined or corridor is None:
        return None
    import boundary_raycast as br
    poly = corridor.polygon()
    outside = [p for p in target.hull()
               if not br.point_in_polygon(p[0], p[1], poly)]
    if not outside:
        return None
    dist = br.points_boundary_distance(
        np.array([p[0] for p in outside]), np.array([p[1] for p in outside]), poly)
    return float(np.max(dist))


def clamp_to_corridor(target: Target, corridor) -> None:
    """Keep a confined target inside the channel.

    Confined targets run constant-velocity down a channel that may bend, so
    without this a long episode walks them into the wall.  Rather than steer
    them -- which would make them reactive and break D1 -- the heading is nudged
    back toward the local channel tangent, which is what a vessel keeping the
    fairway does anyway.
    """
    if not target.confined or corridor is None:
        return
    if confinement_violation(target, corridor) is None:
        return

    # Nearest centreline station, then adopt its tangent direction.
    deltas = corridor.centre - np.array([target.x, target.y])
    index = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
    tangent = corridor.tangent(index)
    along = math.degrees(math.atan2(float(tangent[0]), float(tangent[1])))
    # A target running *against* the channel keeps its own sense of direction.
    if abs(((target.heading - along + 180.0) % 360.0) - 180.0) > 90.0:
        along = (along + 180.0) % 360.0
    target.heading = along
    target.x = float(corridor.centre[index][0])
    target.y = float(corridor.centre[index][1])
