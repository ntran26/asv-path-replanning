"""CPA products, the ship domain, and the Rule 8 admissibility predicate.

`02 §3.2` fixes the structure of the precedence table and defers the width
thresholds to Study 1.  The reward therefore needs a **per-step geometric
predicate**, not a width lookup: one predicate drives every row of the table,
and the width thresholds fall out of the sweep as *results* rather than going in
as inputs.  That is what makes Study 1 a measurement instead of a restatement.

```
Dy_req  = max(0, d_req - DCPA)              # lateral deficit
r_stbd  = d_bnd_stbd - B/2 - c_wall         # usable room to starboard
r_port  = d_bnd_port - B/2 - c_wall         # usable room to port
A_stbd  = (r_stbd >= Dy_req)                # starboard alteration admissible
A_port  = (r_port >= Dy_req)                # port alteration admissible
```

Everything here is a pure function.  The hysteresis band on `r_* - Dy_req` is
stateful and lives in `context.py`.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

import boundary_raycast as br
import constants as cfg
import cpa_cri as cc

# Stations sampled along the projected passage when taking the minimum channel
# width between here and the CPA.  Nine is enough to catch a pinch in a channel
# whose width varies on the scale of a ship length; it costs nine raycasts.
_ROOM_SAMPLES = 9


def d_req(d_abeam: float = None) -> float:
    """Required separation for two identical vessels abeam, metres.

    `2 * d_abeam` -- each vessel keeps its own abeam domain clear, so a
    compliant pass leaves twice the abeam semi-axis between the two centres.
    Centre-to-centre, matching `cpa()` and the engagement gate.
    """
    d = cfg.DOMAIN_LATERAL if d_abeam is None else float(d_abeam)
    return 2.0 * d


def body_offset(p_from, heading_deg: float, p_to) -> Tuple[float, float]:
    """`p_to` in the body frame of a vessel at `p_from`, as (ahead, starboard).

    Headings are compass-style: 0 deg is +y and rotation is clockwise, so the
    forward unit vector is `(sin psi, cos psi)` and starboard is `(cos psi,
    -sin psi)`.
    """
    a = math.radians(float(heading_deg))
    dx = float(p_to[0]) - float(p_from[0])
    dy = float(p_to[1]) - float(p_from[1])
    ahead = dx * math.sin(a) + dy * math.cos(a)
    starboard = dx * math.cos(a) - dy * math.sin(a)
    return ahead, starboard


def cpa_products(p_os, v_os, heading_os_deg: float,
                 p_ts, v_ts, heading_ts_deg: float) -> dict:
    """DCPA, TCPA and the two geometry products evaluated *at* the CPA.

    `y_rel_cpa` is the lateral offset of the target from the own ship at the
    projected CPA, in the own ship's body frame, **positive when the target is
    to starboard**.  `beta_cpa` is the bearing of the own ship from the target
    at that same moment -- near zero means the own ship is crossing the
    target's bow.

    Both are evaluated at the constant-velocity projected CPA rather than now,
    consistently with DCPA and TCPA, because the question a passing-side term
    has to answer is which side the vessels *will* pass on.  A realised version
    measured at actual minimum range is a reported metric (02a §6.3) and must
    not be fed back into the reward -- it only exists after the fact.
    """
    dcpa, tcpa = cc.cpa(p_os, v_os, p_ts, v_ts)

    p_os = np.asarray(p_os, dtype=np.float64)
    p_ts = np.asarray(p_ts, dtype=np.float64)
    os_at = p_os + np.asarray(v_os, dtype=np.float64) * tcpa
    ts_at = p_ts + np.asarray(v_ts, dtype=np.float64) * tcpa

    _, y_rel_cpa = body_offset(os_at, heading_os_deg, ts_at)
    beta_cpa = cc.relative_bearing_deg(ts_at, heading_ts_deg, os_at)

    return {
        "dcpa": float(dcpa),
        "tcpa": float(tcpa),
        "y_rel_cpa": float(y_rel_cpa),
        "beta_cpa": float(beta_cpa),
        "alpha": cc.relative_bearing_deg(p_os, heading_os_deg, p_ts),
        "ct": cc.heading_intersection_deg(heading_os_deg, heading_ts_deg),
        "range": float(np.linalg.norm(p_ts - p_os)),
    }


def sigma_bow(beta_cpa_deg: float, beta_bow_deg: float = cfg.BETA_BOW_DEG) -> float:
    """Severity of crossing ahead, smooth in the bearing (02a §6.3).

    ```
    sigma_bow = clip( (cos beta_CPA - cos beta_bow) / (1 - cos beta_bow), 0, 1 )
    ```

    1 when the own ship sits dead ahead of the target at the CPA, tapering to 0
    at the edge of the bow arc.  Deliberately smooth rather than a beam test:
    a step at 90 deg would put a discontinuity in the reward exactly where
    crossing geometries cluster.
    """
    cos_bow = math.cos(math.radians(float(beta_bow_deg)))
    denom = 1.0 - cos_bow
    if denom < 1e-9:
        return 0.0
    value = (math.cos(math.radians(float(beta_cpa_deg))) - cos_bow) / denom
    return float(np.clip(value, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Channel room, and the admissibility predicate
# ---------------------------------------------------------------------------
def channel_room(path, polygon, s_from: float, s_to: float, cross_track: float,
                 *, samples: int = _ROOM_SAMPLES) -> Tuple[float, float]:
    """Beam-on distance to the channel limits, minimised over the passage.

    Returns `(d_bnd_stbd, d_bnd_port)` in metres, measured from the own ship's
    centre to the boundary polygon abeam, from the **map** -- ground truth,
    per `R-1`, because this is a physical fact about where the vessel fits.

    **Minimised over the along-path interval from now to the projected CPA**,
    not evaluated instantaneously.  The instantaneous value would let the agent
    commit to an alteration that fits where it starts and stops fitting before
    the CPA arrives, which is the manoeuvre Rule 8(b) exists to prevent.  While
    the corridor is a straight inset rectangle the two agree; 03's generator is
    what makes the distinction bite.

    The vessel's current cross-track offset is carried forward along the
    passage rather than assuming it returns to the path, so the room reported
    is the room available to a vessel where it actually is.
    """
    lo, hi = (float(s_from), float(s_to)) if s_to >= s_from else (float(s_to), float(s_from))
    lo = float(np.clip(lo, 0.0, path.length))
    hi = float(np.clip(hi, 0.0, path.length))

    d_stbd, d_port = float("inf"), float("inf")
    for s in np.linspace(lo, hi, max(2, int(samples))):
        frac = s / path.length if path.length > 1e-9 else 0.0
        point, tangent, left = path.frame_at_frac(frac)
        # `left_normal` is to port, so the starboard normal is its negation.
        px = float(point[0]) + float(cross_track) * -float(left[0])
        py = float(point[1]) + float(cross_track) * -float(left[1])
        course = math.degrees(math.atan2(float(tangent[0]), float(tangent[1])))
        d_stbd = min(d_stbd, float(br.raycast_polygon((px, py), course + 90.0, polygon,
                                                      cfg.BOUNDARY_MAX_RANGE)))
        d_port = min(d_port, float(br.raycast_polygon((px, py), course - 90.0, polygon,
                                                      cfg.BOUNDARY_MAX_RANGE)))
    return d_stbd, d_port


def usable_room(d_bnd: float, *, c_wall: float = None, breadth: float = None) -> float:
    """Room usable for an alteration: the beam-on distance less hull and margin."""
    c_wall = cfg.HEAD_ON_WALL_CLEARANCE if c_wall is None else float(c_wall)
    b = cfg.BREADTH if breadth is None else float(breadth)
    return float(d_bnd) - 0.5 * b - c_wall


def lateral_deficit(dcpa: float, d_required: float = None) -> float:
    """`Dy_req = max(0, d_req - DCPA)`: how much lateral offset is still owed.

    Zero when the projected pass already clears the required separation -- the
    case `02 §3.2` calls the normal one in a channel, where Rule 9(a)
    compliance satisfies Rule 14 without any alteration at all.
    """
    d_required = d_req() if d_required is None else float(d_required)
    return max(0.0, d_required - float(dcpa))
