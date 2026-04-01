
"""Utilities for validating the MATLAB-style Bluefin Python model.

This module is designed around the current project workflow:
- replay the real control logs (S1/S2) through the simulator
- export comparable metrics and debug plots
- run open-loop constant-command tests

It uses the new ship model interface:

    model = ShipModel()
    dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud_percent, dt, thruster_rpm=0.0)

where rud_percent is in [-100, 100].
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def wrap_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def unwrap_heading_deg(yaw_deg: np.ndarray) -> np.ndarray:
    yaw_deg = np.asarray(yaw_deg, dtype=float)
    if yaw_deg.size == 0:
        return yaw_deg.copy()
    out = np.empty_like(yaw_deg)
    out[0] = yaw_deg[0]
    for i in range(1, yaw_deg.size):
        out[i] = out[i - 1] + wrap_180(yaw_deg[i] - yaw_deg[i - 1])
    return out

def sample_at_time(t_rel: np.ndarray, values: np.ndarray, query_s: float) -> Optional[float]:
    if len(t_rel) == 0 or query_s < t_rel[0] or query_s > t_rel[-1]:
        return None
    return float(np.interp(query_s, t_rel, values))

def first_crossing_time(t_rel: np.ndarray, values: np.ndarray, threshold: float) -> Optional[float]:
    for i in range(1, len(values)):
        if values[i - 1] < threshold <= values[i]:
            return float(t_rel[i])
    return None

def first_abs_crossing_time(t_rel: np.ndarray, values: np.ndarray, threshold: float) -> Optional[float]:
    av = np.abs(values)
    for i in range(1, len(av)):
        if av[i - 1] < threshold <= av[i]:
            return float(t_rel[i])
    return None

def slope_over_window(t_rel: np.ndarray, values: np.ndarray, t1: float, t2: float) -> Optional[float]:
    mask = (t_rel >= t1) & (t_rel <= t2)
    if np.count_nonzero(mask) < 2:
        return None
    p = np.polyfit(t_rel[mask], values[mask], 1)
    return float(p[0])

def cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(ds)])

def circle_fit_radius(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 6:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = sol
    r2 = c0 + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return float(np.sqrt(r2))

def first_sustained_index(values: np.ndarray, threshold: float, count: int = 3) -> Optional[int]:
    run = 0
    for i, v in enumerate(values):
        if v > threshold:
            run += 1
            if run >= count:
                return i - count + 1
        else:
            run = 0
    return None

def first_sustained_abs_index(values: np.ndarray, threshold: float, count: int = 3) -> Optional[int]:
    run = 0
    for i, v in enumerate(values):
        if abs(v) > threshold:
            run += 1
            if run >= count:
                return i - count + 1
        else:
            run = 0
    return None

def first_sustained_deviation_index(values: np.ndarray, baseline: float, threshold: float, count: int = 3) -> Optional[int]:
    run = 0
    for i, v in enumerate(values):
        if abs(v - baseline) > threshold:
            run += 1
            if run >= count:
                return i - count + 1
        else:
            run = 0
    return None

def safe_rel_error(sim: Optional[float], real: Optional[float], floor: float = 1e-6) -> Optional[float]:
    if sim is None or real is None:
        return None
    denom = max(abs(real), floor)
    return abs(sim - real) / denom

# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class ReplaySeries:
    t_sec: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    yaw_rate_real: np.ndarray
    u_body_real: np.ndarray
    heading_real_deg: Optional[np.ndarray] = None

@dataclass
class ReplayMapping:
    s1_neutral: float
    s1_scale: float
    s2_neutral: float
    s2_full_fwd: float
    max_rudder_deg: float = 30.0
    rpm_max: float = 12.7

    def s1_to_rudder_percent(self, s1_val: float) -> float:
        # Map PWM to requested rudder angle in [-max_rudder_deg, +max_rudder_deg],
        # then convert that angle to percent of the ship model's max rudder angle.
        if self.s1_scale <= 0:
            return 0.0
        z = (s1_val - self.s1_neutral) / self.s1_scale
        z = float(np.clip(z, -1.0, 1.0))
        rud_deg = z * self.max_rudder_deg
        # ship model max rudder is 40 deg in the provided model
        return (rud_deg / 40.0) * 100.0

    def s1_to_rudder_deg(self, s1_val: float) -> float:
        if self.s1_scale <= 0:
            return 0.0
        z = (s1_val - self.s1_neutral) / self.s1_scale
        z = float(np.clip(z, -1.0, 1.0))
        return z * self.max_rudder_deg

    def s2_to_rpm(self, s2_val: float) -> float:
        denom = self.s2_full_fwd - self.s2_neutral
        if abs(denom) < 1e-9:
            return 0.0
        z = (s2_val - self.s2_neutral) / denom
        z = float(np.clip(z, 0.0, 1.0))
        return z * self.rpm_max

# ---------------------------------------------------------------------
# Loading / mapping
# ---------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def build_replay_series(data: Dict[str, Any]) -> ReplaySeries:
    series = data["series"]
    heading = None
    if "yaw_deg" in series:
        heading = np.asarray(series["yaw_deg"], dtype=float)
    return ReplaySeries(
        t_sec=np.asarray(series["t_sec"], dtype=float),
        s1=np.asarray(series["s1"], dtype=float),
        s2=np.asarray(series["s2"], dtype=float),
        yaw_rate_real=np.asarray(series["yaw_rate_degps"], dtype=float),
        u_body_real=np.asarray(series["u_body_mps"], dtype=float),
        heading_real_deg=heading,
    )

def infer_mapping(
    data: Dict[str, Any],
    series: ReplaySeries,
    *,
    max_rudder_deg: float = 30.0,
    rpm_max: float = 12.7,
    override_s1_neutral: Optional[float] = None,
    override_s2_neutral: Optional[float] = None,
    override_s1_scale: Optional[float] = None,
    override_s2_full_fwd: Optional[float] = None,
) -> ReplayMapping:
    sm = data.get("straight_metrics", {})
    tm = data.get("turn_metrics", {})

    s1_neutral = override_s1_neutral
    if s1_neutral is None:
        cand = tm.get("s1_neutral", None)
        if cand is None or cand == 0.0:
            nonzero = series.s1[np.isfinite(series.s1) & (series.s1 > 0)]
            cand = float(np.median(nonzero[:20])) if nonzero.size else 1500.0
        s1_neutral = float(cand)

    s2_neutral = override_s2_neutral
    if s2_neutral is None:
        cand = sm.get("s2_neutral", None)
        if cand is None or cand == 0.0:
            nonzero = series.s2[np.isfinite(series.s2) & (series.s2 > 0)]
            cand = float(np.median(nonzero[:20])) if nonzero.size else 1500.0
        s2_neutral = float(cand)

    s1_scale = override_s1_scale
    if s1_scale is None:
        # Ignore startup zeros / missing-latched values when inferring scale.
        valid_s1 = series.s1[np.isfinite(series.s1) & (series.s1 > 100.0)]
        if valid_s1.size:
            dev = np.abs(valid_s1 - s1_neutral)
            s1_scale = float(np.max(dev)) if dev.size else 500.0
        else:
            s1_scale = 500.0
        s1_scale = max(s1_scale, 1.0)

    s2_full_fwd = override_s2_full_fwd
    if s2_full_fwd is None:
        cand = sm.get("s2_peak", None)
        if cand is None or cand == 0.0:
            finite = series.s2[np.isfinite(series.s2)]
            cand = float(np.max(finite)) if finite.size else s2_neutral + 500.0
        s2_full_fwd = float(cand)

    return ReplayMapping(
        s1_neutral=float(s1_neutral),
        s1_scale=float(s1_scale),
        s2_neutral=float(s2_neutral),
        s2_full_fwd=float(s2_full_fwd),
        max_rudder_deg=float(max_rudder_deg),
        rpm_max=float(rpm_max),
    )

# ---------------------------------------------------------------------
# Model loading / simulation
# ---------------------------------------------------------------------

def load_ship_model_module(module_name: str):
    return importlib.import_module(module_name)

def apply_model_overrides(
    ship_model_module,
    *,
    mass: Optional[float] = None,
    thrust_coef: Optional[float] = None,
    drag_coef: Optional[float] = None,
    turn_coef: Optional[float] = None,
) -> None:
    if mass is not None and hasattr(ship_model_module, "MASS"):
        ship_model_module.MASS = float(mass)
    if thrust_coef is not None and hasattr(ship_model_module, "THRUST_COEF"):
        ship_model_module.THRUST_COEF = float(thrust_coef)
    if drag_coef is not None and hasattr(ship_model_module, "DRAG_COEF"):
        ship_model_module.DRAG_COEF = float(drag_coef)
    if turn_coef is not None and hasattr(ship_model_module, "TURN_COEF"):
        ship_model_module.TURN_COEF = float(turn_coef)
    # Recompute inertia if the module uses this convention
    if hasattr(ship_model_module, "MOMINERTIA") and hasattr(ship_model_module, "MASS") and hasattr(ship_model_module, "RUDDEROFFSET"):
        ship_model_module.MOMINERTIA = 0.5 * ship_model_module.MASS * ship_model_module.RUDDEROFFSET ** 2

def simulate_replay(
    series: ReplaySeries,
    mapping: ReplayMapping,
    *,
    model_module: str = "ship_model_bluefin_matlab_style",
    mass: Optional[float] = None,
    thrust_coef: Optional[float] = None,
    drag_coef: Optional[float] = None,
    turn_coef: Optional[float] = None,
    thruster_rpm: float = 0.0,
) -> Dict[str, np.ndarray]:
    ship_model = load_ship_model_module(model_module)
    apply_model_overrides(
        ship_model,
        mass=mass,
        thrust_coef=thrust_coef,
        drag_coef=drag_coef,
        turn_coef=turn_coef,
    )

    model = ship_model.ShipModel()

    t = series.t_sec
    n = len(t)

    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    heading_deg = np.zeros(n, dtype=float)
    yaw_rate_degps = np.zeros(n, dtype=float)
    u_body = np.zeros(n, dtype=float)
    v_body = np.zeros(n, dtype=float)

    rudder_deg_cmd = np.zeros(n, dtype=float)
    rudder_percent_cmd = np.zeros(n, dtype=float)
    rpm_cmd = np.zeros(n, dtype=float)

    xk = 0.0
    yk = 0.0

    for i in range(n):
        dt = 0.1 if i == 0 else max(t[i] - t[i - 1], 1e-3)

        rud_deg = mapping.s1_to_rudder_deg(series.s1[i])
        rud_percent = mapping.s1_to_rudder_percent(series.s1[i])
        rpm = mapping.s2_to_rpm(series.s2[i])

        dx, dy, hdg_deg, yawrate_deg = model.update(rpm, rud_percent, dt, thruster_rpm=thruster_rpm)
        xk += dx
        yk += dy

        x[i] = xk
        y[i] = yk
        heading_deg[i] = hdg_deg
        yaw_rate_degps[i] = yawrate_deg

        if hasattr(model, "_v"):
            u_body[i] = float(model._v)
        if hasattr(model, "_v_sway"):
            v_body[i] = float(model._v_sway)

        rudder_deg_cmd[i] = rud_deg
        rudder_percent_cmd[i] = rud_percent
        rpm_cmd[i] = rpm

    return {
        "t_sec": t.copy(),
        "x_m": x,
        "y_m": y,
        "heading_deg": heading_deg,
        "yaw_rate_degps": yaw_rate_degps,
        "u_body_mps": u_body,
        "v_body_mps": v_body,
        "rudder_deg_cmd": rudder_deg_cmd,
        "rudder_percent_cmd": rudder_percent_cmd,
        "rpm_cmd": rpm_cmd,
    }

def simulate_open_loop(
    *,
    duration_s: float,
    dt: float,
    rpm: float,
    rudder_deg: float,
    model_module: str = "ship_model_bluefin_matlab_style",
    thruster_rpm: float = 0.0,
    mass: Optional[float] = None,
    thrust_coef: Optional[float] = None,
    drag_coef: Optional[float] = None,
    turn_coef: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    ship_model = load_ship_model_module(model_module)
    apply_model_overrides(
        ship_model,
        mass=mass,
        thrust_coef=thrust_coef,
        drag_coef=drag_coef,
        turn_coef=turn_coef,
    )
    model = ship_model.ShipModel()

    n = int(np.floor(duration_s / dt)) + 1
    t = np.arange(n, dtype=float) * dt

    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    heading_deg = np.zeros(n, dtype=float)
    yaw_rate_degps = np.zeros(n, dtype=float)
    u_body = np.zeros(n, dtype=float)
    v_body = np.zeros(n, dtype=float)
    rudder_deg_cmd = np.full(n, float(rudder_deg), dtype=float)
    rudder_percent_cmd = np.full(n, (rudder_deg / 40.0) * 100.0, dtype=float)
    rpm_cmd = np.full(n, float(rpm), dtype=float)

    xk = 0.0
    yk = 0.0
    rudder_percent = (rudder_deg / 40.0) * 100.0

    for i in range(n):
        dx, dy, hdg_deg, yawrate_deg = model.update(rpm, rudder_percent, dt, thruster_rpm=thruster_rpm)
        xk += dx
        yk += dy

        x[i] = xk
        y[i] = yk
        heading_deg[i] = hdg_deg
        yaw_rate_degps[i] = yawrate_deg
        if hasattr(model, "_v"):
            u_body[i] = float(model._v)
        if hasattr(model, "_v_sway"):
            v_body[i] = float(model._v_sway)

    return {
        "t_sec": t,
        "x_m": x,
        "y_m": y,
        "heading_deg": heading_deg,
        "yaw_rate_degps": yaw_rate_degps,
        "u_body_mps": u_body,
        "v_body_mps": v_body,
        "rudder_deg_cmd": rudder_deg_cmd,
        "rudder_percent_cmd": rudder_percent_cmd,
        "rpm_cmd": rpm_cmd,
    }

# ---------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------

def extract_motion_metrics(sim: Dict[str, np.ndarray]) -> Dict[str, Any]:
    t = np.asarray(sim["t_sec"], dtype=float)
    x = np.asarray(sim["x_m"], dtype=float)
    y = np.asarray(sim["y_m"], dtype=float)
    u = np.asarray(sim["u_body_mps"], dtype=float)
    rpm_cmd = np.asarray(sim["rpm_cmd"], dtype=float)

    idx_rpm = first_sustained_deviation_index(rpm_cmd, baseline=0.0, threshold=0.1, count=3)
    if idx_rpm is None:
        idx_rpm = 0

    idx_motion = first_sustained_index(u, threshold=0.05, count=3)
    if idx_motion is None:
        idx_motion = idx_rpm

    t_rel_cmd = t[idx_rpm:] - t[idx_rpm]
    u_rel_cmd = u[idx_rpm:]
    dist_rel_cmd = cumulative_distance(x[idx_rpm:], y[idx_rpm:])

    t_rel_motion = t[idx_motion:] - t[idx_motion]
    u_rel_motion = u[idx_motion:]
    dist_rel_motion = cumulative_distance(x[idx_motion:], y[idx_motion:])

    peak_u = float(np.max(u_rel_motion)) if len(u_rel_motion) else None

    return {
        "rpm_start_idx": int(idx_rpm),
        "rpm_start_t_sec": float(t[idx_rpm]),
        "motion_start_idx": int(idx_motion),
        "motion_start_t_sec": float(t[idx_motion]),
        "motion_lag_s": float(t[idx_motion] - t[idx_rpm]),

        "u_body_at_2s_after_motion_mps": sample_at_time(t_rel_motion, u_rel_motion, 2.0),
        "u_body_at_5s_after_motion_mps": sample_at_time(t_rel_motion, u_rel_motion, 5.0),
        "u_body_at_10s_after_motion_mps": sample_at_time(t_rel_motion, u_rel_motion, 10.0),
        "distance_at_5s_after_motion_m": sample_at_time(t_rel_motion, dist_rel_motion, 5.0),
        "distance_at_10s_after_motion_m": sample_at_time(t_rel_motion, dist_rel_motion, 10.0),
        "initial_accel_0_2_after_motion_mps2": slope_over_window(t_rel_motion, u_rel_motion, 0.0, 2.0),
        "initial_accel_0_5_after_motion_mps2": slope_over_window(t_rel_motion, u_rel_motion, 0.0, 5.0),
        "peak_u_body_mps": peak_u,
        "time_to_50pct_peak_u_after_motion_s": None if peak_u is None else first_crossing_time(t_rel_motion, u_rel_motion, 0.5 * peak_u),
        "time_to_90pct_peak_u_after_motion_s": None if peak_u is None else first_crossing_time(t_rel_motion, u_rel_motion, 0.9 * peak_u),

        "u_body_at_2s_after_rpm_mps": sample_at_time(t_rel_cmd, u_rel_cmd, 2.0),
        "u_body_at_5s_after_rpm_mps": sample_at_time(t_rel_cmd, u_rel_cmd, 5.0),
        "u_body_at_10s_after_rpm_mps": sample_at_time(t_rel_cmd, u_rel_cmd, 10.0),
        "distance_at_5s_after_rpm_m": sample_at_time(t_rel_cmd, dist_rel_cmd, 5.0),
        "distance_at_10s_after_rpm_m": sample_at_time(t_rel_cmd, dist_rel_cmd, 10.0),
    }

def extract_turn_metrics(sim: Dict[str, np.ndarray]) -> Dict[str, Any]:
    t = np.asarray(sim["t_sec"], dtype=float)
    x = np.asarray(sim["x_m"], dtype=float)
    y = np.asarray(sim["y_m"], dtype=float)
    heading = np.asarray(sim["heading_deg"], dtype=float)
    yaw_rate = np.asarray(sim["yaw_rate_degps"], dtype=float)
    u = np.asarray(sim["u_body_mps"], dtype=float)
    rudder = np.asarray(sim["rudder_deg_cmd"], dtype=float)

    idx_rud = first_sustained_abs_index(rudder, threshold=1.0, count=3)
    if idx_rud is None:
        idx_rud = 0

    idx_turn = first_sustained_abs_index(yaw_rate, threshold=1.0, count=3)
    if idx_turn is None:
        idx_turn = idx_rud

    t_rel_rud = t[idx_rud:] - t[idx_rud]
    yaw_rate_rel_rud = yaw_rate[idx_rud:]
    u_rel_rud = u[idx_rud:]
    heading_rel_rud = unwrap_heading_deg(heading[idx_rud:])
    dpsi_rud = heading_rel_rud - heading_rel_rud[0]

    t_rel_turn = t[idx_turn:] - t[idx_turn]
    yaw_rate_rel_turn = yaw_rate[idx_turn:]
    u_rel_turn = u[idx_turn:]
    heading_rel_turn = unwrap_heading_deg(heading[idx_turn:])
    dpsi_turn = heading_rel_turn - heading_rel_turn[0]

    x_rel_turn = x[idx_turn:]
    y_rel_turn = y[idx_turn:]

    idx90 = np.where(np.abs(dpsi_turn) >= 90.0)[0]
    idx180 = np.where(np.abs(dpsi_turn) >= 180.0)[0]
    r90 = circle_fit_radius(x_rel_turn[:idx90[0] + 1], y_rel_turn[:idx90[0] + 1]) if idx90.size > 0 else None
    r180 = circle_fit_radius(x_rel_turn[:idx180[0] + 1], y_rel_turn[:idx180[0] + 1]) if idx180.size > 0 else None

    return {
        "rudder_start_idx": int(idx_rud),
        "rudder_start_t_sec": float(t[idx_rud]),
        "turn_start_idx": int(idx_turn),
        "turn_start_t_sec": float(t[idx_turn]),
        "turn_lag_s": float(t[idx_turn] - t[idx_rud]),

        "yaw_rate_at_2s_after_rudder_degps": sample_at_time(t_rel_rud, yaw_rate_rel_rud, 2.0),
        "yaw_rate_at_5s_after_rudder_degps": sample_at_time(t_rel_rud, yaw_rate_rel_rud, 5.0),
        "yaw_rate_at_10s_after_rudder_degps": sample_at_time(t_rel_rud, yaw_rate_rel_rud, 10.0),
        "u_body_2s_after_rudder_mps": sample_at_time(t_rel_rud, u_rel_rud, 2.0),
        "u_body_5s_after_rudder_mps": sample_at_time(t_rel_rud, u_rel_rud, 5.0),
        "u_body_10s_after_rudder_mps": sample_at_time(t_rel_rud, u_rel_rud, 10.0),
        "time_to_30deg_after_rudder_s": first_abs_crossing_time(t_rel_rud, dpsi_rud, 30.0),
        "time_to_60deg_after_rudder_s": first_abs_crossing_time(t_rel_rud, dpsi_rud, 60.0),
        "time_to_90deg_after_rudder_s": first_abs_crossing_time(t_rel_rud, dpsi_rud, 90.0),
        "time_to_180deg_after_rudder_s": first_abs_crossing_time(t_rel_rud, dpsi_rud, 180.0),

        "peak_abs_yaw_rate_degps": float(np.max(np.abs(yaw_rate_rel_turn))) if len(yaw_rate_rel_turn) else None,
        "yaw_rate_at_2s_after_turn_degps": sample_at_time(t_rel_turn, yaw_rate_rel_turn, 2.0),
        "yaw_rate_at_5s_after_turn_degps": sample_at_time(t_rel_turn, yaw_rate_rel_turn, 5.0),
        "yaw_rate_at_10s_after_turn_degps": sample_at_time(t_rel_turn, yaw_rate_rel_turn, 10.0),
        "u_body_2s_after_turn_mps": sample_at_time(t_rel_turn, u_rel_turn, 2.0),
        "u_body_5s_after_turn_mps": sample_at_time(t_rel_turn, u_rel_turn, 5.0),
        "u_body_10s_after_turn_mps": sample_at_time(t_rel_turn, u_rel_turn, 10.0),
        "time_to_30deg_after_turn_s": first_abs_crossing_time(t_rel_turn, dpsi_turn, 30.0),
        "time_to_60deg_after_turn_s": first_abs_crossing_time(t_rel_turn, dpsi_turn, 60.0),
        "time_to_90deg_after_turn_s": first_abs_crossing_time(t_rel_turn, dpsi_turn, 90.0),
        "time_to_180deg_after_turn_s": first_abs_crossing_time(t_rel_turn, dpsi_turn, 180.0),
        "radius_first_90deg_m": r90,
        "radius_first_180deg_m": r180,
        "diameter_first_90deg_m": None if r90 is None else 2.0 * r90,
        "diameter_first_180deg_m": None if r180 is None else 2.0 * r180,
    }

def extract_all_metrics(sim: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {
        "motion_metrics": extract_motion_metrics(sim),
        "turn_metrics": extract_turn_metrics(sim),
    }


# ---------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------

def build_comparison_report(real_data: Dict[str, Any], sim_metrics: Dict[str, Any]) -> Dict[str, Any]:
    real_motion = real_data.get("straight_metrics", {})
    real_turn = real_data.get("turn_metrics", {})
    sim_motion = sim_metrics.get("motion_metrics", {})
    sim_turn = sim_metrics.get("turn_metrics", {})

    comparison = {
        "motion": {},
        "turn": {},
    }

    motion_pairs = [
        ("peak_u_body_mps", "peak_u_body_mps"),
        ("initial_accel_0_2_after_motion_mps2", "initial_accel_0_2_after_motion_mps2"),
        ("initial_accel_0_5_after_motion_mps2", "initial_accel_0_5_after_motion_mps2"),
        ("distance_at_10s_after_motion_m", "distance_at_10s_after_motion_m"),
        ("time_to_50pct_peak_u_after_motion_s", "time_to_50pct_peak_u_after_motion_s"),
        ("time_to_90pct_peak_u_after_motion_s", "time_to_90pct_peak_u_after_motion_s"),
    ]
    for real_key, sim_key in motion_pairs:
        rv = real_motion.get(real_key, None)
        sv = sim_motion.get(sim_key, None)
        comparison["motion"][real_key] = {
            "real": rv,
            "sim": sv,
            "rel_error": safe_rel_error(sv, rv),
        }

    turn_pairs = [
        ("peak_abs_yaw_rate_degps", "peak_abs_yaw_rate_degps"),
        ("time_to_90deg_after_turn_s", "time_to_90deg_after_turn_s"),
        ("time_to_180deg_after_turn_s", "time_to_180deg_after_turn_s"),
        ("radius_first_90deg_m", "radius_first_90deg_m"),
        ("radius_first_180deg_m", "radius_first_180deg_m"),
        ("u_body_10s_after_turn_mps", "u_body_10s_after_turn_mps"),
    ]
    for real_key, sim_key in turn_pairs:
        rv = real_turn.get(real_key, None)
        sv = sim_turn.get(sim_key, None)
        comparison["turn"][real_key] = {
            "real": rv,
            "sim": sv,
            "rel_error": safe_rel_error(sv, rv),
        }

    return comparison

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_replay_debug(
    out_png: Path,
    real_series: ReplaySeries,
    sim: Dict[str, np.ndarray],
    title: str,
) -> None:
    t = np.asarray(sim["t_sec"], dtype=float)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(t, real_series.s1, label="S1 raw")
    axs[0].plot(t, sim["rudder_deg_cmd"], label="mapped rudder deg")
    axs[0].set_ylabel("Rudder")
    axs[0].legend()

    axs[1].plot(t, real_series.s2, label="S2 raw")
    axs[1].plot(t, sim["rpm_cmd"], label="mapped rpm")
    axs[1].set_ylabel("Throttle / RPM")
    axs[1].legend()

    axs[2].plot(t, real_series.u_body_real, label="real u_body")
    axs[2].plot(t, sim["u_body_mps"], label="sim u_body")
    axs[2].set_ylabel("u_body [m/s]")
    axs[2].legend()

    axs[3].plot(t, real_series.yaw_rate_real, label="real yaw rate")
    axs[3].plot(t, sim["yaw_rate_degps"], label="sim yaw rate")
    axs[3].set_ylabel("yaw rate [deg/s]")
    axs[3].set_xlabel("time [s]")
    axs[3].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_path(out_png: Path, sim: Dict[str, np.ndarray], title: str) -> None:
    x = np.asarray(sim["x_m"], dtype=float)
    y = np.asarray(sim["y_m"], dtype=float)
    h = np.asarray(sim["heading_deg"], dtype=float)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.scatter([x[-1]], [y[-1]])
    hh = np.deg2rad(h[-1])
    dx = 1.0 * np.sin(hh)
    dy = 1.0 * np.cos(hh)
    ax.arrow(x[-1], y[-1], dx, dy, length_includes_head=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_open_loop_response(out_png: Path, sim: Dict[str, np.ndarray], title: str) -> None:
    t = np.asarray(sim["t_sec"], dtype=float)

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axs[0].plot(t, sim["rpm_cmd"], label="rpm")
    axs[0].plot(t, sim["rudder_deg_cmd"], label="rudder deg")
    axs[0].set_ylabel("Command")
    axs[0].legend()

    axs[1].plot(t, sim["u_body_mps"], label="u_body")
    axs[1].plot(t, sim["v_body_mps"], label="v_body")
    axs[1].set_ylabel("Body speed [m/s]")
    axs[1].legend()

    axs[2].plot(t, sim["yaw_rate_degps"], label="yaw rate")
    axs[2].plot(t, sim["heading_deg"], label="heading deg")
    axs[2].set_ylabel("Heading / yaw")
    axs[2].set_xlabel("time [s]")
    axs[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
