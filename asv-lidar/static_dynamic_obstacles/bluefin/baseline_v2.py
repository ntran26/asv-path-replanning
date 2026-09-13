"""Replay field runs through the *existing* `ship_model.py` (v2), unmodified.

This is the baseline every v3 claim is measured against. The v2 model is used
exactly as the simulator uses it: `update(rpm, rud, dt)` with `rud` in
*simulator* rudder percent, zero-order held between control samples, no
transport delay (v2 has none), started from rest at the measured initial
heading.

The logged `$CMD` value is the transmitted command, which is the negative of
the simulator rudder percent (`udp_live_rl.py` applies `rudder_sign = -1`), so
the replay passes `-cmd`. Feeding `+cmd` makes the model turn the wrong way and
scores worse than a model that never turns.

The same free-run and windowed protocols as `simulate.py` are applied so the
two models are compared on identical footing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from simulate import Batch, residuals, rms
import ship_model as v2


def simulate_v2(batch: Batch, reset_every: Optional[float] = None) -> Dict[str, List[np.ndarray]]:
    dt = batch.dt
    F = batch.F
    reset_stride = int(round(reset_every / dt)) if reset_every else 0

    out_psi, out_x, out_y = [], [], []
    for i, run in enumerate(batch.runs):
        m = v2.ShipModel()
        m.reset()
        m._h = float(batch.psi0[i])

        if reset_stride:
            tg = np.arange(F) * dt
            mpsi = np.interp(tg, run.t, batch.meas_psi[i])
            mx = np.interp(tg, run.t, batch.meas_x[i])
            my = np.interp(tg, run.t, batch.meas_y[i])

        psi_h = np.zeros(F)
        x_h = np.zeros(F)
        y_h = np.zeros(F)
        psi_h[0], x_h[0], y_h[0] = m._h, m._x, m._y

        for k in range(1, F):
            rpm = batch.rpm[i, k - 1]
            rud = -batch.cmd[i, k - 1]   # transmitted -> simulator convention
            m.update(float(rpm), float(rud), dt)
            if reset_stride and (k % reset_stride == 0):
                m._h = float(mpsi[k])
                m._x = float(mx[k])
                m._y = float(my[k])
            psi_h[k], x_h[k], y_h[k] = m._h, m._x, m._y

        ii = batch.idx[i]
        out_psi.append(psi_h[ii])
        out_x.append(x_h[ii])
        out_y.append(y_h[ii])

    return {"psi": out_psi, "x": out_x, "y": out_y}


def report_v2(batch: Batch, window: float = 5.0) -> Dict:
    pw = simulate_v2(batch, reset_every=window)
    ew_psi, ew_pos = residuals(batch, pw, skip_after_reset=1)
    pf = simulate_v2(batch, reset_every=None)
    ef_psi, ef_pos = residuals(batch, pf)

    per_run = []
    for i, run in enumerate(batch.runs):
        per_run.append({
            "run": run.name,
            "win_psi_rms": float(np.sqrt(np.mean(ew_psi[i] ** 2))),
            "win_pos_rms": float(np.sqrt(np.mean(ew_pos[i] ** 2))),
            "free_psi_rms": float(np.sqrt(np.mean(ef_psi[i] ** 2))),
            "free_pos_rms": float(np.sqrt(np.mean(ef_pos[i] ** 2))),
            "free_pos_final": float(ef_pos[i][-1]),
        })
    return {
        "per_run": per_run,
        "win_psi_rms": rms(ew_psi),
        "win_pos_rms": rms(ew_pos),
        "free_psi_rms": rms(ef_psi),
        "free_pos_rms": rms(ef_pos),
        "free_pos_final_med": float(np.median([p["free_pos_final"] for p in per_run])),
    }
