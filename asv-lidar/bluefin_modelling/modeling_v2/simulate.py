"""Batch replay of field runs through the v3 dynamics, and the fitting objective.

Two evaluation modes, both used:

*free*    Simulate each run from t = 0 with the exactly-known initial condition
          (every field run starts from rest at the pose origin), open loop on
          the logged command sequence, for the whole 20-25 s. This is the
          honest sim-to-real number and the one reported as the headline.

*window*  Re-initialise the state from the measurement every `H` seconds — a
          multiple-shooting objective. It conditions the fit far better than
          free-run, because a single early heading error cannot dominate the
          residual, and it is what the parameters are fitted on.

Pose is quantised to 0.1 m / 0.1 deg, but the stream is smooth: the residual
about a local quadratic is 0.34 deg in yaw and 0.026 m in position, i.e. at the
quantisation floor rather than above it. A 5-point quadratic derivative
therefore gives yaw rate to roughly 0.5 deg/s against an 8-20 deg/s signal, and
surge to roughly 0.05 m/s, which is good enough to *initialise* a window.
Derived velocities are used only for initialisation, never as a fitting target:
the residual is always heading and position, as document 05 section 4.4
requires.

Simulation is vectorised over runs *and* over candidate parameter vectors, so a
whole optimiser population costs about the same as one candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

import dynamics as dyn
from build_dataset import Run

DT_SIM = 0.05
PSI_SCALE = 10.0   # deg
POS_SCALE = 1.0    # m
FREE_SCALE = 4.0   # free-run residuals are ~4x windowed ones


@dataclass
class Batch:
    """Pre-gridded runs, ready for repeated simulation."""
    runs: List[Run]
    dt: float
    cmd: np.ndarray          # (R, F) rudder command percent on the fine grid
    rpm: np.ndarray          # (R, F)
    idx: List[np.ndarray]    # per run: fine-grid index of each pose sample
    psi0: np.ndarray         # (R,) initial heading (rad)
    meas_psi: List[np.ndarray]
    meas_x: List[np.ndarray]
    meas_y: List[np.ndarray]
    mpsi_f: np.ndarray = field(default=None)   # (R, F) measured, fine grid
    mx_f: np.ndarray = field(default=None)
    my_f: np.ndarray = field(default=None)
    mr_f: np.ndarray = field(default=None)     # measured yaw rate (rad/s)
    mu_f: np.ndarray = field(default=None)     # measured speed along track (m/s)

    @property
    def R(self) -> int:
        return len(self.runs)

    @property
    def F(self) -> int:
        return self.cmd.shape[1]


def _smooth_derivative(t: np.ndarray, z: np.ndarray, half: int = 2) -> np.ndarray:
    """Derivative of z(t) from a local quadratic fit over +/- `half` samples."""
    out = np.zeros_like(z, dtype=float)
    n = len(z)
    for k in range(n):
        a = max(0, k - half)
        b = min(n, k + half + 1)
        if b - a < 3:
            out[k] = np.gradient(z[a:b], t[a:b])[min(k - a, b - a - 1)]
            continue
        c = np.polyfit(t[a:b] - t[k], z[a:b], 2)
        out[k] = c[1]
    return out


def make_batch(runs: Sequence[Run], dt: float = DT_SIM) -> Batch:
    R = len(runs)
    F = int(max(np.ceil(r.duration / dt) for r in runs)) + 2
    cmd = np.zeros((R, F))
    rpm = np.full((R, F), 12.0)
    idx, mpsi, mx, my = [], [], [], []
    psi0 = np.zeros(R)
    mpsi_f = np.zeros((R, F))
    mx_f = np.zeros((R, F))
    my_f = np.zeros((R, F))
    mr_f = np.zeros((R, F))
    mu_f = np.zeros((R, F))
    tg = np.arange(F) * dt

    for i, run in enumerate(runs):
        cmd[i] = run.rudder_at(tg)
        rp = run.rpm_at(tg)
        rpm[i] = np.where(np.isfinite(rp), rp, 12.0)
        idx.append(np.clip(np.round(run.t / dt).astype(int), 0, F - 1))
        psi = np.deg2rad(run.yaw)
        mpsi.append(psi)
        mx.append(run.x.copy())
        my.append(run.y.copy())
        psi0[i] = psi[0]
        mpsi_f[i] = np.interp(tg, run.t, psi)
        mx_f[i] = np.interp(tg, run.t, run.x)
        my_f[i] = np.interp(tg, run.t, run.y)
        r_meas = _smooth_derivative(run.t, psi)
        s_meas = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(run.x), np.diff(run.y)))])
        u_meas = np.clip(_smooth_derivative(run.t, s_meas), 0.0, 3.0)
        mr_f[i] = np.interp(tg, run.t, r_meas)
        mu_f[i] = np.interp(tg, run.t, u_meas)

    return Batch(runs=list(runs), dt=dt, cmd=cmd, rpm=rpm, idx=idx, psi0=psi0,
                 meas_psi=mpsi, meas_x=mx, meas_y=my,
                 mpsi_f=mpsi_f, mx_f=mx_f, my_f=my_f, mr_f=mr_f, mu_f=mu_f)


def _tile_params(theta: np.ndarray, R: int) -> Dict[str, np.ndarray]:
    """theta: (P, n_params) -> dict of (P*R,) arrays, run index fastest."""
    out = {}
    for j, name in enumerate(dyn.PARAM_NAMES):
        out[name] = np.repeat(theta[:, j], R)
    return out


def simulate_pop(batch: Batch, theta: np.ndarray,
                 reset_every: Optional[float] = None,
                 reset_state: bool = True):
    """Simulate P candidate parameter vectors over all runs.

    `reset_state` additionally re-initialises yaw rate and surge speed from the
    measurement at each window boundary; sway is never observable and is always
    carried forward from the simulation.

    Returns psi, x, y histories of shape (P, R, F).
    """
    theta = np.atleast_2d(theta)
    P, R, F, dt = theta.shape[0], batch.R, batch.F, batch.dt
    p = _tile_params(theta, R)

    # transport delay: per-candidate index shift of the command series
    shift = np.clip(np.round(np.maximum(theta[:, dyn.PARAM_NAMES.index("rud_delay")], 0.0) / dt),
                    0, F - 1).astype(int)
    cols = np.arange(F)[None, :] - shift[:, None]          # (P, F)
    cols = np.clip(cols, 0, F - 1)
    cmd = batch.cmd[None, :, :]                            # (1, R, F)
    cmd = np.take_along_axis(np.repeat(cmd, P, axis=0),
                             np.repeat(cols[:, None, :], R, axis=1), axis=2)
    delta_cmd = dyn.cmd_percent_to_angle(cmd).reshape(P * R, F)
    rpm = np.tile(batch.rpm, (P, 1))

    s = np.zeros((7, P * R))
    s[3] = np.tile(batch.psi0, P)

    if reset_every:
        stride = int(round(reset_every / dt))
        mpsi = np.tile(batch.mpsi_f, (P, 1))
        mx = np.tile(batch.mx_f, (P, 1))
        my = np.tile(batch.my_f, (P, 1))
        mr = np.tile(batch.mr_f, (P, 1))
        mu = np.tile(batch.mu_f, (P, 1))
    else:
        stride = 0

    psi_h = np.empty((P * R, F))
    x_h = np.empty((P * R, F))
    y_h = np.empty((P * R, F))
    psi_h[:, 0], x_h[:, 0], y_h[:, 0] = s[3], s[5], s[6]

    for k in range(1, F):
        s = dyn.rk4_step(s, rpm[:, k - 1], delta_cmd[:, k - 1], p, dt)
        if stride and (k % stride == 0):
            s[3] = mpsi[:, k]
            s[5] = mx[:, k]
            s[6] = my[:, k]
            if reset_state:
                s[2] = mr[:, k]
                s[0] = mu[:, k]
        psi_h[:, k], x_h[:, k], y_h[:, k] = s[3], s[5], s[6]

    return (psi_h.reshape(P, R, F), x_h.reshape(P, R, F), y_h.reshape(P, R, F))


def simulate(batch: Batch, params: Dict[str, float],
             reset_every: Optional[float] = None,
             reset_state: bool = True) -> Dict[str, List[np.ndarray]]:
    """Single-candidate convenience wrapper, sampled at pose times."""
    theta = dyn.as_vector(params)[None, :]
    psi, x, y = simulate_pop(batch, theta, reset_every, reset_state)
    out_psi, out_x, out_y = [], [], []
    for i in range(batch.R):
        ii = batch.idx[i]
        out_psi.append(psi[0, i, ii])
        out_x.append(x[0, i, ii])
        out_y.append(y[0, i, ii])
    return {"psi": out_psi, "x": out_x, "y": out_y}


def residuals(batch: Batch, pred: Dict[str, List[np.ndarray]],
              skip_after_reset: int = 0):
    e_psi, e_pos = [], []
    for i in range(batch.R):
        dp = np.rad2deg(pred["psi"][i] - batch.meas_psi[i])
        dx = pred["x"][i] - batch.meas_x[i]
        dy = pred["y"][i] - batch.meas_y[i]
        if skip_after_reset:
            dp, dx, dy = dp[skip_after_reset:], dx[skip_after_reset:], dy[skip_after_reset:]
        e_psi.append(dp)
        e_pos.append(np.hypot(dx, dy))
    return e_psi, e_pos


def rms(seq) -> float:
    a = np.concatenate(seq)
    return float(np.sqrt(np.mean(a ** 2)))


def objective_pop(batch: Batch, theta: np.ndarray, window: Optional[float] = 3.0,
                  w_pos: float = 1.0, run_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Prediction error for each candidate; `window=None` means free run.

    Free-run errors are an order of magnitude larger than windowed ones, so the
    free-run term is divided by FREE_SCALE to keep the pooled objective from
    being dominated by it.
    """
    theta = np.atleast_2d(theta)
    psi, x, y = simulate_pop(batch, theta, reset_every=window)
    scale = 1.0 if window is not None else FREE_SCALE
    P, R = theta.shape[0], batch.R
    w = np.ones(R) if run_weights is None else run_weights
    j = np.zeros(P)
    for i in range(R):
        ii = batch.idx[i][1:]                      # drop t=0 sample
        dp = np.rad2deg(psi[:, i, ii] - batch.meas_psi[i][1:][None, :])
        dx = x[:, i, ii] - batch.meas_x[i][1:][None, :]
        dy = y[:, i, ii] - batch.meas_y[i][1:][None, :]
        j += w[i] * (np.sqrt(np.mean(dp ** 2, axis=1)) / (PSI_SCALE * scale)
                     + w_pos * np.sqrt(np.mean(dx ** 2 + dy ** 2, axis=1)) / (POS_SCALE * scale))
    bad = ~np.isfinite(j)
    j[bad] = 1e6
    return j / max(np.sum(w), 1e-9)


HORIZONS = (3.0, 10.0)


def turn_rate_stats(batch: Batch, psi_hist: np.ndarray) -> np.ndarray:
    """Pooled p90 of |yaw rate| (deg/s) per candidate, sampled as the log is."""
    P = psi_hist.shape[0]
    allr = []
    for i in range(batch.R):
        ii = batch.idx[i]
        dpsi = np.rad2deg(np.diff(psi_hist[:, i, ii], axis=1))
        dt = np.diff(batch.runs[i].t)[None, :]
        allr.append(np.abs(dpsi / dt))
    allr = np.concatenate(allr, axis=1)
    return np.percentile(allr, 90, axis=1)


def measured_turn_rate_p90(batch: Batch) -> float:
    allr = [np.abs(np.diff(r.yaw) / np.diff(r.t)) for r in batch.runs]
    return float(np.percentile(np.concatenate(allr), 90))


def drift_iqr(batch: Batch, psi_hist, x_hist, y_hist) -> np.ndarray:
    """Pooled interquartile range of the drift angle (course minus heading).

    Robust to the near-stationary samples that make the plain standard
    deviation useless (measured sd is 36 deg on the training session and 6 deg
    on the holdout, purely because of low-speed wrap-around).
    """
    P = psi_hist.shape[0]
    vals = []
    for i in range(batch.R):
        ii = batch.idx[i]
        x = x_hist[:, i, ii]
        y = y_hist[:, i, ii]
        psi = np.rad2deg(psi_hist[:, i, ii])
        dx, dy = np.diff(x, axis=1), np.diff(y, axis=1)
        spd = np.hypot(dx, dy) / np.diff(batch.runs[i].t)[None, :]
        course = np.degrees(np.arctan2(dx, dy))
        mid = 0.5 * (psi[:, 1:] + psi[:, :-1])
        d = (course - mid + 180.0) % 360.0 - 180.0
        vals.append(np.where(spd > 0.5, d, np.nan))
    vals = np.concatenate(vals, axis=1)
    q = np.nanpercentile(vals, [25, 75], axis=1)
    return q[1] - q[0]


def measured_drift_iqr(batch: Batch) -> float:
    vals = []
    for r in batch.runs:
        dx, dy = np.diff(r.x), np.diff(r.y)
        spd = np.hypot(dx, dy) / np.diff(r.t)
        course = np.degrees(np.arctan2(dx, dy))
        mid = 0.5 * (r.yaw[1:] + r.yaw[:-1])
        d = (course - mid + 180.0) % 360.0 - 180.0
        vals.append(d[spd > 0.5])
    v = np.concatenate(vals)
    return float(np.percentile(v, 75) - np.percentile(v, 25))


# --- manoeuvring priors -----------------------------------------------------
# The field runs contain no sustained turn: the policy oscillates the rudder
# throughout, so the steady-turn regime is completely unconstrained by the data
# and an unregularised fit extrapolates into it badly (turning radius
# non-monotonic in rudder, vessel stalling to 0.54 m/s at full helm). Training
# episodes DO hold rudder, so a policy would learn to avoid sustained turns as a
# pure simulator artefact.
#
# These bands encode standard manoeuvring expectations for a hull with a 2.7%
# rudder area ratio. They are PRIORS, not measurements, and basin session 1
# (turning circles) replaces them with data.
TURN_RUDDERS = (0.0, 25.0, 50.0, 75.0, 100.0)
PRIOR_R_BAND = (10.0, 20.0)        # steady |yaw rate| at full rudder, deg/s
PRIOR_RADIUS_BAND_L = (1.5, 4.0)   # steady turning radius, in vessel lengths
PRIOR_SPEED_RATIO = (0.55, 0.85)   # speed at full helm / straight-line speed


def steady_turn(theta: np.ndarray, rudders=TURN_RUDDERS, rpm: float = 12.0,
                T: float = 30.0, dt: float = 0.1):
    """Steady-state surge speed and yaw rate for each candidate at each helm.

    Returns (u, r_degps), both shape (P, len(rudders)).
    """
    theta = np.atleast_2d(theta)
    P, K = theta.shape[0], len(rudders)
    p = {k: np.repeat(theta[:, j], K) for j, k in enumerate(dyn.PARAM_NAMES)}
    delta = np.tile(np.deg2rad(np.array(rudders) / 100.0 * dyn.MAX_RUD_ANGLE_DEG), P)
    s = np.zeros((7, P * K))
    rpm_v = np.full(P * K, rpm)
    for _ in range(int(T / dt)):
        s = dyn.rk4_step(s, rpm_v, delta, p, dt)
    return s[0].reshape(P, K), np.rad2deg(s[2]).reshape(P, K)


def prior_penalty(theta: np.ndarray) -> np.ndarray:
    """Penalty for unphysical steady-turn behaviour. Zero inside every band."""
    u, r = steady_turn(theta)
    ar = np.abs(r)
    pen = np.zeros(theta.shape[0])

    # 0. every non-zero helm must turn the same way (an equilibrium that
    #    reverses sign with helm is a course-instability artefact, not physics)
    sgn = np.sign(r[:, 1:])
    ref = np.sign(r[:, -1:])
    pen += np.sum(np.maximum(0.0, -sgn * ref) * np.abs(r[:, 1:]), axis=1) / 5.0

    # 1. turning authority must grow with helm
    pen += np.sum(np.maximum(0.0, ar[:, :-1] - ar[:, 1:]), axis=1) / 10.0

    # 2. steady yaw rate at full helm inside the expected band
    lo, hi = PRIOR_R_BAND
    pen += (np.maximum(0.0, lo - ar[:, -1]) + np.maximum(0.0, ar[:, -1] - hi)) / 10.0

    # 3. turning radius inside the expected band
    radius = u[:, -1] / np.maximum(np.deg2rad(ar[:, -1]), 1e-3) / dyn.VESSEL_LENGTH
    rlo, rhi = PRIOR_RADIUS_BAND_L
    pen += (np.maximum(0.0, rlo - radius) + np.maximum(0.0, radius - rhi)) / 4.0

    # 4. speed loss in the turn of a plausible size
    ratio = u[:, -1] / np.maximum(u[:, 0], 1e-3)
    slo, shi = PRIOR_SPEED_RATIO
    pen += (np.maximum(0.0, slo - ratio) + np.maximum(0.0, ratio - shi)) * 2.0

    pen[~np.isfinite(pen)] = 1e3
    return pen


def objective_multi(batch: Batch, theta: np.ndarray, w_pos: float = 1.0,
                    horizons=HORIZONS, w_free: float = 1.0,
                    w_dist: float = 1.0, w_drift: float = 1.0,
                    w_prior: float = 10.0) -> np.ndarray:
    """Prediction error pooled over horizons, plus a turn-rate distribution term.

    Two problems are being solved at once.

    *Horizon pooling.* Fitting on a single short window buys window accuracy at
    the cost of long-horizon drift: a 3 s fit reached 14 deg windowed but 55 deg
    free-run. Pooling 3 s, 10 s and the full open-loop replay forces one
    parameter set to work at every horizon the simulator actually uses.

    *Distribution matching.* A pure mean-squared-error fit against a response
    that is only partly predictable shrinks the modelled response: the
    MSE-optimal model under-turns. Measured on the training runs, an
    MSE-only fit produced a mean |yaw rate| of 3.5 deg/s against 8.7 deg/s
    measured. That is exactly the wrong bias here — a simulator that under-turns
    trains a policy that over-commands rudder, which is the Paper 2 field
    symptom (wider turns, larger oscillation, doubled cross-track error). The
    penalty on the p90 of |yaw rate| keeps the modelled turning authority
    honest at a modest cost in MSE.
    """
    theta = np.atleast_2d(theta)
    j = np.zeros(theta.shape[0])
    for H in horizons:
        j += objective_pop(batch, theta, window=H, w_pos=w_pos)
    n = len(horizons)

    if w_free > 0.0 or w_dist > 0.0:
        psi, x, y = simulate_pop(batch, theta, reset_every=None)
        if w_free > 0.0:
            jf = np.zeros(theta.shape[0])
            for i in range(batch.R):
                ii = batch.idx[i][1:]
                dp = np.rad2deg(psi[:, i, ii] - batch.meas_psi[i][1:][None, :])
                dx = x[:, i, ii] - batch.meas_x[i][1:][None, :]
                dy = y[:, i, ii] - batch.meas_y[i][1:][None, :]
                jf += (np.sqrt(np.mean(dp ** 2, axis=1)) / (PSI_SCALE * FREE_SCALE)
                       + w_pos * np.sqrt(np.mean(dx ** 2 + dy ** 2, axis=1))
                       / (POS_SCALE * FREE_SCALE))
            j += w_free * jf / max(batch.R, 1)
            n += 1
        if w_dist > 0.0:
            p90_sim = turn_rate_stats(batch, psi)
            p90_meas = measured_turn_rate_p90(batch)
            j += w_dist * np.abs(p90_sim - p90_meas) / p90_meas
            n += 1
        if w_drift > 0.0:
            # Sway is only observable through the drift angle, and an MSE fit
            # leaves it unconstrained: the first v3 fit over-drifted (IQR 14 deg
            # against 8.5 measured) and was the one validation test where v3
            # scored worse than v2. Matching the drift spread pins it down.
            iqr_sim = drift_iqr(batch, psi, x, y)
            iqr_meas = measured_drift_iqr(batch)
            iqr_sim = np.where(np.isfinite(iqr_sim), iqr_sim, 1e3)
            j += w_drift * np.abs(iqr_sim - iqr_meas) / iqr_meas
            n += 1

    if w_prior > 0.0:
        # Treated as a constraint rather than a soft term: with a weight
        # comparable to the data terms the optimiser simply pays the penalty and
        # keeps an unusable steady turn. The result is the best fit to the field
        # data *subject to* plausible manoeuvring behaviour.
        j += w_prior * prior_penalty(theta)
        n += 1

    j[~np.isfinite(j)] = 1e6
    return j / n


def objective(batch: Batch, params: Dict[str, float], window: float = 5.0,
              w_pos: float = 1.0) -> float:
    return float(objective_pop(batch, dyn.as_vector(params)[None, :], window, w_pos)[0])


def report(batch: Batch, params: Dict[str, float], window: float = 5.0) -> Dict:
    pw = simulate(batch, params, reset_every=window)
    ew_psi, ew_pos = residuals(batch, pw, skip_after_reset=1)
    pf = simulate(batch, params, reset_every=None)
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
        "free_pos_final_med": float(np.median([q["free_pos_final"] for q in per_run])),
    }
