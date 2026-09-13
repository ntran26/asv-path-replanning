"""Emit `ship_model_v3.py`: a drop-in replacement for `ship_model.py`.

Values are baked in from `out/params_final.json` so the deliverable has no
runtime dependency on the fitting artefacts, but it is generated rather than
hand-edited so the numbers cannot drift away from the fit that produced them.
"""

from __future__ import annotations

import json
import os

import dynamics as dyn

TEMPLATE = '''"""Bluefin 3-DOF manoeuvring model, v3 — identified from field logs.

Drop-in replacement for `ship_model.py`. Same public interface:

    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)

`rud` is percent of max rudder in [-100, 100] in the **simulator** convention,
exactly as v2 took it, so no call site needs changing. (The transmitted `$CMD`
value is the negative of this; see `dynamics.py`.)

Identified on the 2026-07-02 field session ({n_train} runs) and validated on the
untouched 2026-07-03 session ({n_test} runs). Against the v2 model, on the
holdout:

    free-run heading RMSE      {v2_free_psi:.1f} deg  ->  {v3_free_psi:.1f} deg
    windowed position RMSE     {v2_win_pos:.2f} m   ->  {v3_win_pos:.2f} m
    free-run path length error {v2_path:+.2f} m   ->  {v3_path:+.2f} m
    turn-rate KS distance      {v2_ks:.3f}    ->  {v3_ks:.3f}

What changed from v2, and why
-----------------------------
1. The nondimensional MMG hull block inherited from `Blue02.m` is gone. At
   model scale it produced ~0.09 N and ~0.07 N.m, three orders of magnitude
   below the tunable linear terms beside it, so it never influenced the
   solution — v2's behaviour came entirely from TURN_COEF, LINEAR_*_DAMP and
   the rudder scales. It is replaced by damping terms with physical units that
   were actually identified.
2. Damping is speed-dependent (`N_r*u*r`, `Y_v*u*v`), which is the MMG form.
   v2's speed-independent damping is why every early fit collapsed.
3. Speed loss in a turn comes from cross-flow drag (`X_vv`, `X_rr`), not from
   rudder drag alone.
4. The actuator is explicit: transport delay, rate limit, first-order lag.
   v2's `delta_dot = clip(delta_cmd - delta, +/-rate)` was a first-order lag
   with an implicit 1 s time constant, not a rate limiter.

Known limits (these are what the basin sessions are for)
--------------------------------------------------------
* RPM was 12.0 in 18 of 20 usable runs, so `T12` is anchored at one operating
  point and the square-law RPM exponent is assumed, not measured.
* `rud_delay` = {rud_delay:.2f} s is an *effective* delay: it lumps transport
  latency, the one-frame-stale pose the controller used, servo response and
  estimator lag. It is not a measured servo specification.
* `rud_tau` sits at its lower bound — once the rate limit and delay are in the
  model, a separate servo lag is not identifiable at a 2 Hz command rate.
* Sway damping (`Y_v`, `Y_vv`) has confidence intervals wider than its own
  value. Sway is only weakly observable through the drift angle.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Tuple

import numpy as np

import dynamics as dyn
from dynamics import (VESSEL_LENGTH, VESSEL_WIDTH, HULL_MARGIN,
                      HULL_FORWARD_SHIFT, LIDAR_OFFSET_M,
                      MASS, M_X, M_Y, IZ_JZ, MAX_RUD_ANGLE_DEG)

# Kept for import compatibility with code written against v2
MAX_RUD_ANGLE = MAX_RUD_ANGLE_DEG
MOMINERTIA = IZ_JZ

# ---------------------------------------------------------------------------
# Identified parameters (2026-07-02 session; 90% bootstrap interval alongside)
# ---------------------------------------------------------------------------
IDENTIFIED: Dict[str, float] = {params_block}

CI_LOW: Dict[str, float] = {ci_low_block}

CI_HIGH: Dict[str, float] = {ci_high_block}

# Whole fitted parameter vectors from the run-level bootstrap. Domain
# randomisation draws from these rather than from the per-parameter intervals.
BOOTSTRAP: np.ndarray = np.array({bootstrap_block})

PARAM_ORDER = {param_order}


def sample_params(rng=None, scale: float = 1.0, jitter: float = 0.05) -> Dict[str, float]:
    """Draw a parameter set for domain randomisation.

    Draws a convex blend of two bootstrap solutions, plus a small jitter.

    Sampling each parameter independently from its own interval would be wrong
    here. Several parameters are strongly correlated — `k_R`, `N_r` and `N_uv`
    all scale the yaw subsystem, and some have bootstrap intervals wider than
    their own value — so independent draws land off the solution manifold and
    produce simulators that are unstable or turn the wrong way. Blending whole
    bootstrap vectors keeps the correlation structure, and every draw starts
    from a parameter set that actually satisfied the manoeuvring constraints.

    `scale` widens the spread about the bootstrap mean for the case where
    run-to-run variation within one session understates true uncertainty — which
    it does: the bootstrap resamples runs from a single day and re-optimises
    locally from the full-data optimum, so these intervals are a lower bound.
    `scale` > 1 is the honest setting until basin data gives a second estimate.

    Draws are not validated here; `acceptance.py` test A5 checks that a sample
    of them is usable, and that check should be re-run whenever `scale` changes.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = BOOTSTRAP.shape[0]
    i, k = rng.integers(0, n), rng.integers(0, n)
    w = rng.random()
    vec = w * BOOTSTRAP[i] + (1.0 - w) * BOOTSTRAP[k]

    mean = BOOTSTRAP.mean(axis=0)
    vec = mean + (vec - mean) * scale
    if jitter > 0:
        sd = BOOTSTRAP.std(axis=0)
        vec = vec + rng.normal(0.0, jitter * scale, size=vec.shape) * sd
    vec = np.maximum(vec, 0.0) * (mean >= 0) + np.minimum(vec, 0.0) * (mean < 0)
    return {{k2: float(v) for k2, v in zip(PARAM_ORDER, vec)}}


class ShipModel:
    """3-DOF manoeuvring model with an explicit actuator path."""

    def __init__(self, params: Dict[str, float] = None, dt_hint: float = 0.05) -> None:
        self.p = dict(IDENTIFIED if params is None else params)
        self._dt_hint = float(dt_hint)
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self) -> None:
        self._s = np.zeros((7, 1))
        self._cmd_buf = deque()
        self._buf_dt = None

    def state_dict(self) -> Dict[str, float]:
        u, v, r, psi, delta, x, y = self._s[:, 0]
        return {{
            "u_body_mps": float(u),
            "v_body_mps": float(v),
            "yaw_rate_radps": float(r),
            "yaw_rate_degps": float(math.degrees(r)),
            "heading_rad": float(psi),
            "heading_deg": float(math.degrees(psi) % 360.0),
            "rudder_deg": float(math.degrees(delta)),
            "x_m": float(x),
            "y_m": float(y),
            "speed_mps": float(math.hypot(u, v)),
        }}

    # -- actuator delay ---------------------------------------------------
    def _delayed_command(self, delta_cmd: float, dt: float) -> float:
        """Pure transport delay, implemented as a FIFO sized by dt."""
        n = int(round(self.p["rud_delay"] / max(dt, 1e-6)))
        if self._buf_dt != dt:
            self._cmd_buf = deque([delta_cmd] * max(n, 0), maxlen=max(n, 1))
            self._buf_dt = dt
        if n <= 0:
            return delta_cmd
        out = self._cmd_buf[0] if len(self._cmd_buf) == self._cmd_buf.maxlen else delta_cmd
        self._cmd_buf.append(delta_cmd)
        return out

    # -- integration ------------------------------------------------------
    def update(self, rpm: float, rud: float, dt: float,
               *, thruster_rpm: float = 0.0) -> Tuple[float, float, float, float]:
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        x_prev, y_prev = float(self._s[5, 0]), float(self._s[6, 0])

        # simulator percent -> transmitted convention -> angle
        delta_cmd = float(dyn.cmd_percent_to_angle(np.array([-float(rud)]))[0])
        delta_cmd = self._delayed_command(delta_cmd, dt)

        pa = {{k: np.array([val]) for k, val in self.p.items()}}
        self._s = dyn.rk4_step(self._s, np.array([float(rpm)]),
                               np.array([delta_cmd]), pa, dt)

        dx = float(self._s[5, 0]) - x_prev
        dy = float(self._s[6, 0]) - y_prev
        heading_deg = math.degrees(float(self._s[3, 0])) % 360.0
        yaw_rate_degps = math.degrees(float(self._s[2, 0]))
        return dx, dy, heading_deg, yaw_rate_degps


if __name__ == "__main__":
    m = ShipModel()
    print("straight, 12 rpm")
    for k in range(200):
        m.update(12.0, 0.0, 0.05)
        if k % 40 == 0:
            st = m.state_dict()
            print(f"  t={{0.05*(k+1):5.2f}}s u={{st['u_body_mps']:5.2f}} hdg={{st['heading_deg']:6.2f}}")

    m.reset()
    print("hard turn, 12 rpm, full rudder")
    for k in range(400):
        m.update(12.0, 100.0, 0.05)
        if k % 80 == 0:
            st = m.state_dict()
            print(f"  t={{0.05*(k+1):5.2f}}s u={{st['u_body_mps']:5.2f}} "
                  f"r={{st['yaw_rate_degps']:6.2f}} deg/s rud={{st['rudder_deg']:6.2f}} deg")
'''


def fmt(d):
    return "{\n" + "".join(f'    "{k}": {v:.6g},\n' for k, v in d.items()) + "}"


def main():
    j = json.load(open("out/params_final.json"))
    v = json.load(open("out/validation.json")) if os.path.exists("out/validation.json") else None

    stats = dict(v2_free_psi=0, v3_free_psi=0, v2_win_pos=0, v3_win_pos=0,
                 v2_path=0, v3_path=0, v2_ks=0, v3_ks=0)
    if v:
        free = [r for r in v["v1_holdout"] if r["horizon"] == "free"][0]
        w3 = [r for r in v["v1_holdout"] if r["horizon"] == "3 s"][0]
        import numpy as np
        stats.update(v2_free_psi=free["v2_psi"], v3_free_psi=free["v3_psi"],
                     v2_win_pos=w3["v2_pos"], v3_win_pos=w3["v3_pos"],
                     v2_path=float(np.mean(v["v5_holdout"]["v2_err"])),
                     v3_path=float(np.mean(v["v5_holdout"]["v3_err"])),
                     v2_ks=v["v4_holdout"]["ks_v2"], v3_ks=v["v4_holdout"]["ks_v3"])

    import numpy as _np
    B = _np.array(j["bootstrap"])
    src = TEMPLATE.format(
        params_block=fmt(j["params"]),
        ci_low_block=fmt(j["ci_low"]),
        ci_high_block=fmt(j["ci_high"]),
        bootstrap_block=repr(B.round(6).tolist()),
        param_order=repr(list(dyn.PARAM_NAMES)),
        n_train=len(j["train_runs"]), n_test=len(j["holdout_runs"]),
        rud_delay=j["params"]["rud_delay"], **stats)
    open("out/ship_model_v3.py", "w").write(src)
    print("wrote out/ship_model_v3.py")


if __name__ == "__main__":
    main()
