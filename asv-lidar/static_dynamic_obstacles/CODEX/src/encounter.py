"""COLREGs encounter classification.  **One module, two consumers.**

**Revision 2** — five classes.  Port and starboard crossing collapse into one.

This is the only place the encounter geometry is defined.  The observation
(01 §6) and the reward gate (02 §4.2) must both call `classify` or
`EncounterClassifier` -- never reimplement the thresholds.  If the two diverge,
even only at a sector boundary, the agent is penalised for a role it was never
shown, and that failure is close to undiagnosable from training curves.
Hysteresis is applied **once**, here, for the same reason.

The five classes and what each obliges the own ship to do (01 §5.3):

| Class           | Governing rule | Own-ship obligation                     |
|-----------------|----------------|-----------------------------------------|
| none            | --             | Follow path                             |
| head_on         | 14             | Alter to starboard, subject to width    |
| crossing        | 15, 16, 9(b)   | Give way **regardless of approach side**|
| overtaking      | 13, 16, 9(e)   | Keep clear of the vessel overtaken      |
| being_overtaken | 13, 17(a)(i)   | Hold course and speed                   |

Baseline thresholds are Waltz & Okhrin (2023) Table 1, after Xu et al. (2020),
with the three modifications 01 §5.3 requires:

1. **Port and starboard crossing collapse into one class** under Rule 9(b): a
   vessel under 20 m shall not impede a vessel that can navigate only within a
   narrow channel, so the own ship gives way either way (S3).  This replaces the
   Rule 18 route used by Meyer et al., whose premise -- own ship much smaller
   than the vessels it meets -- fails here, because own ship and target are
   similarly sized model vessels.  The geometric side is still computed and
   returned by `crossing_side`, because 02's passing-side reward term needs it.
2. **The head-on band is widened** from +/-5 deg toward the +/-6-10 deg of
   common practice.
3. **A "being overtaken" class is added.**  The source table has no equivalent:
   Waltz & Okhrin assume linear deterministic targets and cover only give-way
   cases, so Rule 17(a)(i) passive course-keeping is unrepresented there.  Only
   17(a)(i) is in scope -- active release under 17(a)(ii) is future work (S5).

Angle conventions
-----------------
`alpha` is the relative bearing OS->TS in [0, 360): 0 dead ahead, 90 abeam to
starboard.  `ct` is the heading intersection angle in [0, 360): 180 means
reciprocal courses, 0 means the same course.
"""

from __future__ import annotations

from typing import Dict, Optional

import constants as cfg
from cpa_cri import heading_intersection_deg, relative_bearing_deg

# Public class names, in the frozen one-hot order used by the observation.
CLASSES = cfg.ENCOUNTER_CLASSES
NONE, HEAD_ON, CROSSING, OVERTAKING, BEING_OVERTAKEN = CLASSES
CLASS_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

# Geometric sub-distinction, retained for 02 but NOT part of the one-hot.
SIDE_NONE = "none"
SIDE_STARBOARD = "starboard"
SIDE_PORT = "port"


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------
def _wrap360(angle: float) -> float:
    return float(angle) % 360.0


def _in_arc(angle: float, lo: float, hi: float, widen: float = 0.0) -> bool:
    """Is `angle` inside the arc [lo, hi], walking clockwise from lo to hi?

    Handles wraparound, so `_in_arc(355, 350, 10)` is True.  `widen` grows the
    arc symmetrically at both ends, which is how the sticky hysteresis below
    keeps a held class from flickering at its own boundary.
    """
    lo = _wrap360(lo - widen)
    hi = _wrap360(hi + widen)
    a = _wrap360(angle)
    if lo <= hi:
        return lo <= a <= hi
    return a >= lo or a <= hi          # the arc crosses 0


# ---------------------------------------------------------------------------
# The pure classifier
# ---------------------------------------------------------------------------
def classify(p_os, heading_os_deg: float, speed_os: float,
             p_ts, heading_ts_deg: float, speed_ts: float,
             *, widen: float = 0.0, sticky: Optional[str] = None) -> str:
    """Classify one encounter.  Pure: no state, no side effects.

    `widen` and `sticky` exist only for `EncounterClassifier`.  When `sticky`
    names a class, that class's angular bands are widened by `widen` degrees so
    a held classification survives small excursions across its own threshold.
    Callers wanting a plain geometric answer should pass neither.
    """
    alpha = relative_bearing_deg(p_os, heading_os_deg, p_ts)          # OS -> TS
    beta = relative_bearing_deg(p_ts, heading_ts_deg, p_os)           # TS -> OS
    ct = heading_intersection_deg(heading_os_deg, heading_ts_deg)

    def w(name: str) -> float:
        return float(widen) if sticky == name else 0.0

    stern_lo = cfg.BEING_OVERTAKEN_BEARING_MIN_DEG                    # 112.5
    stern_hi = cfg.BEING_OVERTAKEN_BEARING_MAX_DEG                    # 247.5
    ct_half = cfg.OVERTAKING_CT_HALF_DEG                              # 67.5
    margin = cfg.BEING_OVERTAKEN_SPEED_MARGIN
    bearing_half = cfg.HEAD_ON_BEARING_HALF_DEG
    ct_head_half = cfg.HEAD_ON_CT_HALF_DEG

    # --- Rule 13 first -----------------------------------------------------
    # Overtaking overrides the crossing rules ("notwithstanding anything
    # contained in the Rules of Part B, Sections I and II"), so it is tested
    # before them.  The CT bands are in fact disjoint from the crossing bands,
    # but the precedence should be structural rather than accidental.
    near_parallel = _in_arc(ct, 360.0 - ct_half, ct_half,
                            w(OVERTAKING) or w(BEING_OVERTAKEN))

    if near_parallel:
        # OS overtaking TS: the OS lies in the TS's stern arc, and is faster.
        if _in_arc(beta, stern_lo, stern_hi, w(OVERTAKING)) and speed_os > speed_ts + margin:
            return OVERTAKING
        # OS being overtaken: the TS lies in the OS's stern arc, and is faster.
        # This is the class the source table has no room for.
        if _in_arc(alpha, stern_lo, stern_hi, w(BEING_OVERTAKEN)) and speed_ts > speed_os + margin:
            return BEING_OVERTAKEN

    # --- Rule 14, head-on --------------------------------------------------
    if (_in_arc(alpha, 360.0 - bearing_half, bearing_half, w(HEAD_ON))
            and _in_arc(ct, 180.0 - ct_head_half, 180.0 + ct_head_half, w(HEAD_ON))):
        return HEAD_ON

    # --- Rule 15 crossing, collapsed under Rule 9(b) -----------------------
    # Both approach sides return the same class.  The own ship gives way either
    # way, so the observation does not need to distinguish them; 02 gets the
    # side from `crossing_side` when the passing-side term needs it.
    if _crossing_starboard(alpha, ct, bearing_half, ct_head_half, w(CROSSING)):
        return CROSSING
    if _crossing_port(alpha, ct, bearing_half, ct_head_half, w(CROSSING)):
        return CROSSING

    return NONE


def _crossing_starboard(alpha, ct, bearing_half, ct_head_half, widen) -> bool:
    """Target on the starboard bow, crossing left to right."""
    return (_in_arc(alpha, bearing_half, cfg.CROSSING_STBD_MAX_DEG, widen)
            and _in_arc(ct, 180.0 + ct_head_half, 292.5, widen))


def _crossing_port(alpha, ct, bearing_half, ct_head_half, widen) -> bool:
    """Target on the port bow, crossing right to left."""
    return (_in_arc(alpha, cfg.CROSSING_PORT_MIN_DEG, 360.0 - bearing_half, widen)
            and _in_arc(ct, cfg.OVERTAKING_CT_HALF_DEG, 180.0 - ct_head_half, widen))


def crossing_side(p_os, heading_os_deg: float, p_ts, heading_ts_deg: float) -> str:
    """Which bow the target crosses from: "starboard", "port", or "none".

    The observation one-hot deliberately does **not** carry this -- Rule 9(b)
    makes the own ship give way either way, so the side is not a different
    obligation.  02's passing-side reward term does need it, which is why the
    geometric distinction is kept here rather than discarded.
    """
    alpha = relative_bearing_deg(p_os, heading_os_deg, p_ts)
    ct = heading_intersection_deg(heading_os_deg, heading_ts_deg)
    bearing_half = cfg.HEAD_ON_BEARING_HALF_DEG
    ct_head_half = cfg.HEAD_ON_CT_HALF_DEG

    if _crossing_starboard(alpha, ct, bearing_half, ct_head_half, 0.0):
        return SIDE_STARBOARD
    if _crossing_port(alpha, ct, bearing_half, ct_head_half, 0.0):
        return SIDE_PORT
    return SIDE_NONE


def one_hot(name: str):
    """Five-element one-hot in the frozen `cfg.ENCOUNTER_CLASSES` order."""
    import numpy as np
    vec = np.zeros(cfg.N_ENCOUNTER_CLASSES, dtype=np.float32)
    vec[CLASS_INDEX[name]] = 1.0
    return vec


# ---------------------------------------------------------------------------
# The hysteresis wrapper -- the only stateful part
# ---------------------------------------------------------------------------
class EncounterClassifier:
    """Holds a class per track until a different one has persisted.

    Two mechanisms, both from `constants.py` §10:

    * **Sticky bands.**  The currently-held class is re-tested with its angular
      bands widened by `ENCOUNTER_BEARING_HYSTERESIS_DEG`, so a target sitting
      exactly on a threshold does not oscillate.
    * **Hold steps.**  A genuinely different class must persist for
      `ENCOUNTER_HOLD_STEPS` consecutive updates before it is adopted.

    State is keyed by track id, so slot re-use cannot leak one target's history
    into another's.
    """

    def __init__(self, *, hold_steps: int = cfg.ENCOUNTER_HOLD_STEPS,
                 widen_deg: float = cfg.ENCOUNTER_BEARING_HYSTERESIS_DEG) -> None:
        self.hold_steps = int(hold_steps)
        self.widen_deg = float(widen_deg)
        self._held: Dict[int, str] = {}
        self._pending: Dict[int, str] = {}
        self._pending_count: Dict[int, int] = {}
        self._first_seen: Dict[int, int] = {}
        self._steps = 0

    def reset(self) -> None:
        self._held.clear()
        self._pending.clear()
        self._pending_count.clear()
        self._first_seen.clear()
        self._steps = 0

    def forget(self, track_id: int) -> None:
        """Drop a lost track's history."""
        self._held.pop(track_id, None)
        self._pending.pop(track_id, None)
        self._pending_count.pop(track_id, None)
        self._first_seen.pop(track_id, None)

    def held(self, track_id: int) -> str:
        return self._held.get(track_id, NONE)

    def classification_latency(self, track_id: int) -> Optional[int]:
        """Steps between first sight and the first non-`none` class.

        A reported perception metric (04 §7): classification latency and
        stability are part of the N1 evidence.
        """
        return self._first_seen.get(track_id)

    def update(self, track_id: int, p_os, heading_os_deg: float, speed_os: float,
               p_ts, heading_ts_deg: float, speed_ts: float) -> str:
        """Classify with hysteresis and return the class now in force."""
        self._steps += 1
        current = self._held.get(track_id)
        observed = classify(p_os, heading_os_deg, speed_os,
                            p_ts, heading_ts_deg, speed_ts,
                            widen=self.widen_deg if current else 0.0,
                            sticky=current)

        if current is None:
            self._held[track_id] = observed
            self._pending.pop(track_id, None)
            self._pending_count.pop(track_id, None)
            if observed != NONE:
                self._first_seen.setdefault(track_id, 0)
            return observed

        if observed == current:
            self._pending.pop(track_id, None)
            self._pending_count.pop(track_id, None)
            return current

        if self._pending.get(track_id) == observed:
            self._pending_count[track_id] += 1
        else:
            self._pending[track_id] = observed
            self._pending_count[track_id] = 1

        if self._pending_count[track_id] >= self.hold_steps:
            self._held[track_id] = observed
            self._pending.pop(track_id, None)
            self._pending_count.pop(track_id, None)
            if observed != NONE:
                self._first_seen.setdefault(track_id, self._steps)
            return observed

        return current
