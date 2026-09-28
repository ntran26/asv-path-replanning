"""The nominal encounter: both tracks with the own ship holding its leg at cruise.

Shared by the scenario galleries (`tools/diagnostics/frozen_gallery.py`,
`paper2_gallery.py`) and the Paper 2 layout set's target-clearance check
(`paper2_set.py`).  Moved out of `frozen_gallery.py` on 2026-09-27 so `src/` code
can use it.
"""
from __future__ import annotations

import copy
import math

import numpy as np

import constants as cfg
import targets as tgt


def nominal_encounter(env) -> dict:
    """Both tracks with the own ship holding the leg at cruise (`U_NOM`).

    Call after `env.reset(...)`.  The target is stepped exactly as the
    environment steps it -- its behaviour model sees the own ship's state, and
    confined classes are clamped to the fairway -- so the track is what that
    target would do if the own ship stood on.  A policy that manoeuvres changes
    a reactive target's track; the trajectory under a policy is `run_test.py`'s.
    """
    pts = np.asarray(env.path.points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_cum = np.r_[0.0, np.cumsum(seg)]
    length = float(s_cum[-1])
    horizon = min(length / cfg.U_NOM, cfg.MAX_EPISODE_STEPS * cfg.UPDATE_RATE)
    dt = cfg.PHYSICS_DT
    n = int(round(horizon / dt))
    target = copy.deepcopy(env.targets[0]) if env.targets else None
    own, trk = [], []
    for k in range(n + 1):
        s = min(cfg.U_NOM * k * dt, length)
        x, y = np.interp(s, s_cum, pts[:, 0]), np.interp(s, s_cum, pts[:, 1])
        j = min(int(np.searchsorted(s_cum, s, side="right")) - 1, len(seg) - 1)
        tangent = (pts[j + 1] - pts[j]) / max(seg[j], 1e-9)
        heading = math.degrees(math.atan2(tangent[0], tangent[1])) % 360.0
        own.append((x, y, heading))
        if target is not None:
            trk.append((target.x, target.y, target.heading))
            if k < n:
                state = {"x": x, "y": y, "velocity": cfg.U_NOM * tangent, "heading": heading}
                target.step(dt, own=state)
                tgt.clamp_to_corridor(target, env._confine_geom or env.channel,
                                      env._confine_poly or env.boundary_polygon)
    own, trk = np.array(own), np.array(trk)
    out = {"own": own, "target": trk, "dt": dt, "cpa_k": None, "cpa_range": float("nan"),
           "turned_deg": 0.0}
    if len(trk):
        rng = np.hypot(trk[:, 0] - own[:, 0], trk[:, 1] - own[:, 1])
        k = int(np.argmin(rng))
        out.update(cpa_k=k, cpa_range=float(rng[k]), cpa_t=k * dt,
                   turned_deg=float(np.max(np.abs((trk[:, 2] - trk[0, 2] + 180.0) % 360.0 - 180.0))))
    return out
