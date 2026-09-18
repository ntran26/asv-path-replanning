"""Closest-point-of-approach geometry, the ship domain, and the risk index.

Follows Waltz & Okhrin (2023, *Neural Networks* 165:634-653) §3.3 in structure.
**None of their constants are reused** -- see `constants.py` §9 for why a
straight re-derivation in ship lengths does not work at this scale.

The known failure mode, and the patch (01 §5.1)
----------------------------------------------
CPA assumes both vessels hold course and speed.  Two ships on near-parallel
courses have a CPA far in the past or future, so CPA-based risk reads low --
but a slight turn by either collapses |TCPA| and the situation becomes urgent
instantly.  In a narrow channel near-parallel geometry is the *normal* case, not
the exception, so this matters far more here than in the open-water literature
the formulation comes from.

`CR_ED`, the plain Euclidean-distance term, is what covers that gap.  It is not
optional in a channel: without it a vessel 2 m abeam on a parallel course scores
near-zero risk.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

import constants as cfg


# ---------------------------------------------------------------------------
# CPA geometry
# ---------------------------------------------------------------------------
def cpa(p_os, v_os, p_ts, v_ts) -> Tuple[float, float]:
    """Return (DCPA, TCPA) in metres and seconds.

    With relative position p = p_TS - p_OS and relative velocity v = v_TS - v_OS:

        TCPA = -(p . v) / |v|^2
        DCPA = |p + v * TCPA|

    TCPA > 0 means the CPA lies ahead; TCPA < 0 means it is already passed and
    the range is opening.  When the relative velocity vanishes the geometry is
    frozen, so the CPA is the current separation, now, and TCPA is 0.
    """
    p = np.asarray(p_ts, dtype=np.float64) - np.asarray(p_os, dtype=np.float64)
    v = np.asarray(v_ts, dtype=np.float64) - np.asarray(v_os, dtype=np.float64)

    v_sq = float(v @ v)
    if v_sq < 1e-12:
        return float(np.linalg.norm(p)), 0.0

    tcpa = float(-(p @ v) / v_sq)
    dcpa = float(np.linalg.norm(p + v * tcpa))
    return dcpa, tcpa


def relative_bearing_deg(p_os, heading_os_deg: float, p_ts) -> float:
    """Bearing of the target from the own ship, relative to its heading.

    Returned in [0, 360): 0 is dead ahead, 90 is abeam to starboard.  This is
    the convention the encounter table in 01 §5.3 is written in.
    """
    d = np.asarray(p_ts, dtype=np.float64) - np.asarray(p_os, dtype=np.float64)
    absolute = math.degrees(math.atan2(float(d[0]), float(d[1])))
    return (absolute - float(heading_os_deg)) % 360.0


def heading_intersection_deg(heading_os_deg: float, heading_ts_deg: float) -> float:
    """Heading intersection angle CT, in [0, 360).

    CT near 180 means reciprocal courses (head-on); CT near 0 means the two are
    on the same course, which is the overtaking geometry.
    """
    return (float(heading_ts_deg) - float(heading_os_deg)) % 360.0


# ---------------------------------------------------------------------------
# Ship domain
# ---------------------------------------------------------------------------
def domain_scale(bearing_deg: float, *, fore: float = cfg.DOMAIN_FORE,
                 aft: float = cfg.DOMAIN_AFT,
                 lateral: float = cfg.DOMAIN_LATERAL) -> float:
    """Radius of the asymmetric ship domain at a given relative bearing.

    A half-ellipse fore and a half-ellipse aft, sharing the lateral semi-axis,
    after Chun et al. as used by Waltz & Okhrin -- but compressed.  Their
    3*Lpp fore-aft is 4.71 m at LBP = 1.57 m, which does not fit a 10 m channel
    (`constants.py` §8 carries the reasoning and the TODO).
    """
    a = math.radians(float(bearing_deg))
    # Body frame: +y is ahead, +x is starboard.
    ahead = math.cos(a)
    abeam = math.sin(a)
    longitudinal = fore if ahead >= 0.0 else aft
    denom = (ahead / longitudinal) ** 2 + (abeam / lateral) ** 2
    return float(1.0 / math.sqrt(denom)) if denom > 1e-12 else float(lateral)


def distance_to_domain(p_os, heading_os_deg: float, p_ts) -> float:
    """Range from the domain boundary to the target; 0 when inside.

    DCPA and the distance feature are both measured to the **domain**, not to
    the hull (01 §5.2).
    """
    d = float(np.linalg.norm(np.asarray(p_ts, dtype=np.float64)
                             - np.asarray(p_os, dtype=np.float64)))
    bearing = relative_bearing_deg(p_os, heading_os_deg, p_ts)
    return max(0.0, d - domain_scale(bearing))


def inside_domain(p_os, heading_os_deg: float, p_ts) -> bool:
    return distance_to_domain(p_os, heading_os_deg, p_ts) <= 0.0


# ---------------------------------------------------------------------------
# Collision Risk Index
# ---------------------------------------------------------------------------
def bow_crossing_factor(p_os, p_ts, heading_ts_deg: float, *,
                        gain: float = cfg.CRI_BOW_CROSSING_GAIN,
                        half_deg: float = cfg.CRI_BOW_CROSSING_HALF_DEG) -> float:
    """Inflate risk when the own ship sits across the target's bow.

    Crossing ahead of a vessel is more dangerous than crossing astern of it at
    the same distance, and COLREGs treats the two very differently.  A plain
    distance metric cannot see the difference.
    """
    bearing_os_from_ts = relative_bearing_deg(p_ts, heading_ts_deg, p_os)
    off_bow = abs((bearing_os_from_ts + 180.0) % 360.0 - 180.0)
    if off_bow >= float(half_deg):
        return 1.0
    # Linear taper from `gain` dead ahead to 1.0 at the edge of the arc.
    return float(1.0 + (float(gain) - 1.0) * (1.0 - off_bow / float(half_deg)))


def cri(p_os, v_os, heading_os_deg: float, p_ts, v_ts, heading_ts_deg: float) -> float:
    """Collision Risk Index in [0, 1].

        CR = 1                     if the TS is inside the OS ship domain
        CR = max(CR_CPA, CR_ED)    otherwise

    `CR_CPA` decays exponentially in DCPA and |TCPA|, with **different rates
    before and after** the CPA so risk falls away quickly once it is passed.
    `CR_ED` is the Euclidean-distance patch described in the module docstring.
    """
    if inside_domain(p_os, heading_os_deg, p_ts):
        return 1.0

    dcpa, tcpa = cpa(p_os, v_os, p_ts, v_ts)
    # DCPA is measured to the domain boundary, not to the hull (01 §5.2).
    bearing = relative_bearing_deg(p_os, heading_os_deg, p_ts)
    dcpa_eff = max(0.0, dcpa - domain_scale(bearing))

    tcpa_scale = cfg.CRI_TCPA_SCALE_BEFORE if tcpa >= 0.0 else cfg.CRI_TCPA_SCALE_AFTER
    cr_cpa = math.exp(-dcpa_eff / cfg.CRI_DCPA_SCALE) * math.exp(-abs(tcpa) / tcpa_scale)
    cr_cpa *= bow_crossing_factor(p_os, p_ts, heading_ts_deg)

    distance = distance_to_domain(p_os, heading_os_deg, p_ts)
    cr_ed = math.exp(-distance / cfg.CRI_ED_SCALE)

    return float(min(1.0, max(cr_cpa, cr_ed)))
