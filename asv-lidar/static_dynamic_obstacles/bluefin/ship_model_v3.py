"""Bluefin 3-DOF manoeuvring model, v3 — identified from field logs.

Drop-in replacement for `ship_model.py`. Same public interface:

    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)

`rud` is percent of max rudder in [-100, 100] in the **simulator** convention,
exactly as v2 took it, so no call site needs changing. (The transmitted `$CMD`
value is the negative of this; see `dynamics.py`.)

Identified on the 2026-07-02 field session (14 runs) and validated on the
untouched 2026-07-03 session (6 runs). Against the v2 model, on the
holdout:

    free-run heading RMSE      33.2 deg  ->  23.5 deg
    windowed position RMSE     0.50 m   ->  0.26 m
    free-run path length error -3.89 m   ->  -0.79 m
    turn-rate KS distance      0.243    ->  0.171

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
* `rud_delay` = 0.73 s is an *effective* delay: it lumps transport
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
IDENTIFIED: Dict[str, float] = {
    "T12": 16.0623,
    "t_boost": 0,
    "u_boost": 0.5,
    "X_uu": 12.8879,
    "X_vv": 50.6376,
    "X_rr": 0.0579934,
    "X_delta": 0.116498,
    "Y_v": 0.923845,
    "Y_vv": 220.976,
    "k_R": 2.17636,
    "k_race": 0.0135134,
    "N_r": 1.03802,
    "N_rr": 0.0173081,
    "N_uv": -83.9691,
    "rud_rate": 2984.75,
    "rud_tau": 0.895208,
    "rud_delay": 0.730117,
}

CI_LOW: Dict[str, float] = {
    "T12": 14.7447,
    "t_boost": 0,
    "u_boost": 0.5,
    "X_uu": 11.3132,
    "X_vv": 47.6656,
    "X_rr": 0,
    "X_delta": 0,
    "Y_v": 0,
    "Y_vv": 205.669,
    "k_R": 1.1403,
    "k_race": 0,
    "N_r": 0.0139861,
    "N_rr": 0,
    "N_uv": -86.7988,
    "rud_rate": 2675.74,
    "rud_tau": 0.871623,
    "rud_delay": 0.669981,
}

CI_HIGH: Dict[str, float] = {
    "T12": 18.6148,
    "t_boost": 0,
    "u_boost": 0.5,
    "X_uu": 15.1144,
    "X_vv": 54.7757,
    "X_rr": 13.3826,
    "X_delta": 0.55476,
    "Y_v": 5.96434,
    "Y_vv": 236.018,
    "k_R": 2.55019,
    "k_race": 0.210766,
    "N_r": 8.68639,
    "N_rr": 5.85239,
    "N_uv": -55.3445,
    "rud_rate": 2997.94,
    "rud_tau": 0.982161,
    "rud_delay": 0.779586,
}

# Whole fitted parameter vectors from the run-level bootstrap. Domain
# randomisation draws from these rather than from the per-parameter intervals.
BOOTSTRAP: np.ndarray = np.array([[16.107266, 0.0, 0.5, 12.280455, 51.038097, 3.407267, 0.155186, 0.03838, 225.41686, 1.539305, 0.063242, 6.893017, 1.10896, -71.953826, 2942.735791, 0.883302, 0.727702], [19.290313, 0.0, 0.5, 17.525325, 59.217201, 7.926003, 0.128846, 6.482477, 237.818528, 0.426209, 0.259519, 2.94415, 0.0, -61.455688, 2820.515389, 0.888379, 0.586092], [15.947865, 0.0, 0.5, 13.266343, 52.100327, 3.445514, 0.318151, 0.0, 228.646287, 1.877132, 0.05601, 2.303963, 0.478897, -81.303159, 2990.215691, 0.874159, 0.75914], [16.305968, 0.0, 0.5, 12.886408, 51.324973, 0.0, 0.138785, 1.239285, 221.039155, 2.233684, 0.0, 1.44598, 0.0, -82.42048, 2942.378219, 0.885542, 0.722033], [15.859606, 0.0, 0.5, 12.641901, 50.396382, 0.014255, 0.04587, 0.0, 217.406141, 2.207825, 0.031428, 0.03996, 0.0, -89.352843, 3000.0, 0.898238, 0.72421], [16.68206, 0.0, 0.5, 11.274963, 52.219205, 16.12978, 0.494519, 0.0, 194.767578, 2.392141, 0.0, 12.212964, 2.977581, -48.889534, 2589.361179, 0.612544, 0.764339], [16.270957, 0.0, 0.5, 13.651335, 55.410571, 4.721946, 0.587197, 0.387054, 212.443715, 1.111171, 0.237572, 4.254326, 0.069228, -81.323179, 2597.788043, 0.955304, 0.742304], [15.745972, 0.0, 0.5, 12.552394, 50.768535, 0.022042, 0.160492, 0.139129, 220.082034, 2.474546, 0.0, 1.085963, 0.0, -87.064066, 2984.280265, 0.909978, 0.723711], [16.084605, 0.0, 0.5, 12.623047, 49.792723, 0.621888, 0.203211, 0.053917, 223.099462, 2.019156, 0.001995, 1.855539, 0.044552, -75.533544, 2932.566096, 0.919328, 0.739449], [15.806218, 0.0, 0.5, 12.749457, 50.525615, 0.0, 0.12514, 1.470016, 222.465716, 2.104217, 0.010468, 0.952703, 0.0, -81.695439, 2983.715101, 0.8854, 0.726069], [15.649928, 0.0, 0.5, 12.770401, 49.183509, 0.115656, 0.0, 1.61941, 221.036068, 1.194408, 0.082929, 3.978519, 0.03011, -70.402194, 2899.624129, 0.902138, 0.724455], [17.811091, 0.0, 0.5, 14.101295, 52.942176, 3.324513, 0.644764, 0.000189, 227.029714, 1.818335, 0.006818, 0.118256, 7.40036, -73.676295, 3000.0, 0.918023, 0.685497], [15.573253, 0.0, 0.5, 12.640742, 53.596631, 1.838409, 0.015898, 1.711715, 213.970361, 1.554413, 0.101186, 8.716688, 2.177937, -73.614688, 2990.88983, 0.904197, 0.791152], [15.762153, 0.0, 0.5, 12.965499, 50.519708, 0.07286, 0.187812, 5.002078, 222.210461, 2.064609, 0.009393, 0.253463, 0.443556, -83.509523, 2928.761982, 0.890385, 0.692576], [19.04762, 0.0, 0.5, 15.537583, 47.764182, 15.74226, 0.0, 0.0, 254.293504, 1.774984, 0.056383, 0.0, 0.120578, -85.7514, 2979.937036, 1.025283, 0.761232], [14.710097, 0.0, 0.5, 10.686808, 51.803918, 0.472504, 0.146925, 0.038393, 232.673893, 1.209017, 0.160985, 5.064309, 0.296081, -73.25649, 2993.393104, 0.8846, 0.755331], [15.297206, 0.0, 0.5, 13.267304, 50.056106, 2.683084, 0.228681, 8.299471, 220.903183, 2.207825, 0.0, 2.489892, 1.050799, -81.512184, 2910.869391, 0.877537, 0.663521], [14.00487, 0.0, 0.5, 11.384076, 48.545763, 1.353292, 0.12949, 0.0, 225.749823, 2.189706, 0.000215, 3.147255, 0.045023, -73.879645, 2991.277318, 0.937036, 0.716808], [16.08528, 0.0, 0.5, 12.790675, 50.542433, 0.0, 0.106466, 1.375354, 222.392298, 2.201224, 0.019256, 0.885322, 0.110521, -85.721394, 2962.673259, 0.892468, 0.72699], [15.57933, 0.0, 0.5, 13.342626, 50.366792, 0.009554, 0.080425, 0.718585, 224.833555, 2.513076, 0.007078, 1.039196, 0.321729, -85.370971, 2977.510444, 0.93481, 0.68236], [15.392805, 0.0, 0.5, 14.328433, 49.499754, 0.489832, 0.0, 1.577337, 219.60918, 1.617529, 0.116083, 3.585689, 0.124733, -82.841872, 2938.693518, 0.903221, 0.729826], [16.017961, 0.0, 0.5, 13.152439, 51.707359, 0.0, 0.021441, 1.8467, 218.786109, 2.336824, 0.0, 0.0, 0.064999, -85.487328, 2945.303631, 0.900254, 0.714435], [17.19814, 0.0, 0.5, 12.540786, 47.612454, 9.000332, 0.145701, 0.0, 202.021102, 2.830544, 4.3e-05, 8.63011, 8.452395, -52.05381, 2987.005279, 0.996622, 0.681978], [15.758545, 0.0, 0.5, 13.243716, 51.17649, 0.396141, 0.155484, 0.0, 222.726905, 1.877482, 0.056701, 1.905749, 0.151244, -82.055839, 2876.591103, 0.896804, 0.75565], [16.050278, 0.0, 0.5, 12.894054, 50.579629, 0.0, 0.124183, 0.888456, 222.756871, 2.17854, 0.013188, 1.006865, 0.0, -84.207105, 2990.793955, 0.897534, 0.73318], [14.809101, 0.0, 0.5, 12.125989, 47.50465, 3.025687, 0.157609, 0.0, 218.366361, 2.570175, 0.001058, 0.069124, 0.090643, -86.306285, 2967.591899, 0.935379, 0.720299], [15.925122, 0.0, 0.5, 12.691925, 50.685421, 0.87457, 0.179484, 0.072316, 217.708302, 2.10469, 0.047428, 2.211003, 1.633549, -81.086251, 2994.126257, 0.904876, 0.787796], [16.252853, 0.0, 0.5, 13.859119, 51.045407, 2.127005, 0.0, 1.959732, 223.211934, 2.082006, 0.0, 1.340678, 0.547863, -77.8772, 2962.011405, 0.870257, 0.76214]])

PARAM_ORDER = ['T12', 't_boost', 'u_boost', 'X_uu', 'X_vv', 'X_rr', 'X_delta', 'Y_v', 'Y_vv', 'k_R', 'k_race', 'N_r', 'N_rr', 'N_uv', 'rud_rate', 'rud_tau', 'rud_delay']


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
    return {k2: float(v) for k2, v in zip(PARAM_ORDER, vec)}


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
        return {
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
        }

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

        pa = {k: np.array([val]) for k, val in self.p.items()}
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
            print(f"  t={0.05*(k+1):5.2f}s u={st['u_body_mps']:5.2f} hdg={st['heading_deg']:6.2f}")

    m.reset()
    print("hard turn, 12 rpm, full rudder")
    for k in range(400):
        m.update(12.0, 100.0, 0.05)
        if k % 80 == 0:
            st = m.state_dict()
            print(f"  t={0.05*(k+1):5.2f}s u={st['u_body_mps']:5.2f} "
                  f"r={st['yaw_rate_degps']:6.2f} deg/s rud={st['rudder_deg']:6.2f} deg")
