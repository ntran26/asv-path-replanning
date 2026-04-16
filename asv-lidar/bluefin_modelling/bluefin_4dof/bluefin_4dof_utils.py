"""Utilities for validating and tuning ``ship_model_bluefin_4dof.py``.

These helpers mirror the metric style already used in the Bluefin modelling
workflow:
    - straight-line motion metrics
    - turning-circle metrics
    - comparison against real-vessel benchmark JSON files
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent


def _find_first_existing(names: Iterable[str]) -> Optional[Path]:
    for name in names:
        p = ROOT / name
        if p.exists():
            return p
    return None


def _extract_real_metrics_from_json(data: Dict[str, Any]) -> Dict[str, Any]:
    # Already in benchmark form
    if "straight_metrics" in data or "turn_metrics" in data:
        return data

    # Comparison-style file: {"motion": {metric: {real, sim, rel_error}}, "turn": ...}
    out: Dict[str, Any] = {}
    if "motion" in data:
        out["straight_metrics"] = {k: v.get("real") if isinstance(v, dict) else v for k, v in data["motion"].items()}
    if "turn" in data:
        out["turn_metrics"] = {k: v.get("real") if isinstance(v, dict) else v for k, v in data["turn"].items()}
    return out


def load_default_real_benchmarks() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    motion_path = _find_first_existing([
        "test_3_metrics.json",
        "test_3_comparison.json",
        "best_v2_motion_comparison.json",
        "best_joint_motion_comparison.json",
    ])
    turn_path = _find_first_existing([
        "test_4_metrics.json",
        "test_4_comparison.json",
        "best_v2_turn_comparison.json",
        "best_joint_turn_comparison.json",
    ])
    if motion_path is None or turn_path is None:
        raise FileNotFoundError(
            "Could not find real benchmark JSON files. Expected one of: "
            "test_3_metrics.json / test_3_comparison.json and "
            "test_4_metrics.json / test_4_comparison.json in the script folder."
        )
    with motion_path.open("r", encoding="utf-8") as f:
        motion_data = json.load(f)
    with turn_path.open("r", encoding="utf-8") as f:
        turn_data = json.load(f)
    return _extract_real_metrics_from_json(motion_data), _extract_real_metrics_from_json(turn_data)


def wrap_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def unwrap_heading_deg(yaw_deg: np.ndarray) -> np.ndarray:
    yaw_deg = np.asarray(yaw_deg, dtype=float)
    if yaw_deg.size == 0:
        return yaw_deg.copy()
    out = np.empty_like(yaw_deg)
    out[0] = yaw_deg[0]
    for i in range(1, len(yaw_deg)):
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


def safe_rel_error(sim: Optional[float], real: Optional[float], floor: float = 1e-6) -> float:
    if sim is None or real is None:
        return 10.0
    return abs(sim - real) / max(abs(real), floor)


def load_ship_model_module(module_name: str = "ship_model_bluefin_4dof"):
    mod = importlib.import_module(module_name)
    importlib.reload(mod)
    return mod


def apply_module_params(mod, params: Dict[str, float]) -> None:
    for k, v in params.items():
        if hasattr(mod, k):
            setattr(mod, k, v)


def run_open_loop(
    *,
    module_name: str = "ship_model_bluefin_4dof",
    params: Optional[Dict[str, float]] = None,
    rpm: float,
    rudder_deg: float,
    duration_s: float,
    dt: float = 0.1,
    warmup_s: float = 0.0,
    thruster_rpm: float = 0.0,
) -> Dict[str, np.ndarray]:
    mod = load_ship_model_module(module_name)
    if params:
        apply_module_params(mod, params)
    model = mod.ShipModel()
    rudder_percent = (rudder_deg / float(mod.MAX_RUD_ANGLE)) * 100.0

    steps = int(round(duration_s / dt)) + 1
    warmup_steps = int(round(warmup_s / dt))

    t = np.zeros(steps, dtype=float)
    x = np.zeros(steps, dtype=float)
    y = np.zeros(steps, dtype=float)
    u = np.zeros(steps, dtype=float)
    v = np.zeros(steps, dtype=float)
    yaw = np.zeros(steps, dtype=float)
    yaw_rate = np.zeros(steps, dtype=float)
    roll = np.zeros(steps, dtype=float)
    delta = np.zeros(steps, dtype=float)

    for i in range(steps):
        t[i] = i * dt
        st = model.state_dict()
        x[i] = st.get("x_m", 0.0)
        y[i] = st.get("y_m", 0.0)
        u[i] = st.get("u_body_mps", 0.0)
        v[i] = st.get("v_body_mps", 0.0)
        yaw[i] = st.get("heading_deg", 0.0)
        yaw_rate[i] = st.get("yaw_rate_degps", 0.0)
        roll[i] = st.get("roll_deg", 0.0)
        delta[i] = st.get("rudder_deg", 0.0)

        if i < steps - 1:
            rud = 0.0 if i < warmup_steps else rudder_percent
            model.update(rpm, rud, dt, thruster_rpm=thruster_rpm)

    return {
        "t_sec": t,
        "x_m": x,
        "y_m": y,
        "u_body_mps": u,
        "v_body_mps": v,
        "yaw_deg": yaw,
        "yaw_rate_degps": yaw_rate,
        "roll_deg": roll,
        "rudder_deg": delta,
        "rpm_cmd": np.full_like(t, float(rpm)),
        "turn_rudder_deg": np.full_like(t, float(rudder_deg)),
    }


def extract_motion_metrics(sim: Dict[str, np.ndarray]) -> Dict[str, Optional[float]]:
    t = np.asarray(sim["t_sec"], dtype=float)
    x = np.asarray(sim["x_m"], dtype=float)
    y = np.asarray(sim["y_m"], dtype=float)
    u = np.asarray(sim["u_body_mps"], dtype=float)

    idx_motion = first_sustained_index(u, threshold=0.05, count=3)
    if idx_motion is None:
        idx_motion = 0
    t_rel = t[idx_motion:] - t[idx_motion]
    u_rel = u[idx_motion:]
    dist_rel = cumulative_distance(x[idx_motion:], y[idx_motion:])
    peak_u = float(np.max(u_rel)) if len(u_rel) else None

    return {
        "motion_start_idx": int(idx_motion),
        "motion_start_t_sec": float(t[idx_motion]),
        "u_body_at_2s_after_motion_mps": sample_at_time(t_rel, u_rel, 2.0),
        "u_body_at_5s_after_motion_mps": sample_at_time(t_rel, u_rel, 5.0),
        "u_body_at_10s_after_motion_mps": sample_at_time(t_rel, u_rel, 10.0),
        "distance_at_5s_after_motion_m": sample_at_time(t_rel, dist_rel, 5.0),
        "distance_at_10s_after_motion_m": sample_at_time(t_rel, dist_rel, 10.0),
        "initial_accel_0_2_after_motion_mps2": slope_over_window(t_rel, u_rel, 0.0, 2.0),
        "initial_accel_0_5_after_motion_mps2": slope_over_window(t_rel, u_rel, 0.0, 5.0),
        "peak_u_body_mps": peak_u,
        "time_to_50pct_peak_u_after_motion_s": None if peak_u is None else first_crossing_time(t_rel, u_rel, 0.5 * peak_u),
        "time_to_90pct_peak_u_after_motion_s": None if peak_u is None else first_crossing_time(t_rel, u_rel, 0.9 * peak_u),
    }


def extract_turn_metrics(sim: Dict[str, np.ndarray]) -> Dict[str, Optional[float]]:
    t = np.asarray(sim["t_sec"], dtype=float)
    x = np.asarray(sim["x_m"], dtype=float)
    y = np.asarray(sim["y_m"], dtype=float)
    yaw = np.asarray(sim["yaw_deg"], dtype=float)
    yaw_rate = np.asarray(sim["yaw_rate_degps"], dtype=float)
    u = np.asarray(sim["u_body_mps"], dtype=float)

    idx_turn = first_sustained_abs_index(yaw_rate, threshold=1.0, count=3)
    if idx_turn is None:
        idx_turn = 0

    t_rel = t[idx_turn:] - t[idx_turn]
    x_rel = x[idx_turn:]
    y_rel = y[idx_turn:]
    yaw_rel = unwrap_heading_deg(yaw[idx_turn:])
    yaw_rate_rel = yaw_rate[idx_turn:]
    u_rel = u[idx_turn:]
    dpsi = yaw_rel - yaw_rel[0]

    idx90 = np.where(np.abs(dpsi) >= 90.0)[0]
    idx180 = np.where(np.abs(dpsi) >= 180.0)[0]
    r90 = circle_fit_radius(x_rel[: idx90[0] + 1], y_rel[: idx90[0] + 1]) if idx90.size > 0 else None
    r180 = circle_fit_radius(x_rel[: idx180[0] + 1], y_rel[: idx180[0] + 1]) if idx180.size > 0 else None

    return {
        "turn_start_idx": int(idx_turn),
        "turn_start_t_sec": float(t[idx_turn]),
        "peak_abs_yaw_rate_degps": float(np.max(np.abs(yaw_rate_rel))) if len(yaw_rate_rel) else None,
        "time_to_90deg_after_turn_s": first_abs_crossing_time(t_rel, dpsi, 90.0),
        "time_to_180deg_after_turn_s": first_abs_crossing_time(t_rel, dpsi, 180.0),
        "radius_first_90deg_m": r90,
        "radius_first_180deg_m": r180,
        "u_body_10s_after_turn_mps": sample_at_time(t_rel, u_rel, 10.0),
    }


def compare_metrics(sim_metrics: Dict[str, Any], real_metrics: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k in keys:
        sim = sim_metrics.get(k)
        real = real_metrics.get(k)
        out[k] = {
            "real": real,
            "sim": sim,
            "rel_error": None if sim is None or real is None else safe_rel_error(sim, real),
        }
    return out


def motion_comparison(sim_metrics: Dict[str, Any], real_bench: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "peak_u_body_mps",
        "initial_accel_0_2_after_motion_mps2",
        "initial_accel_0_5_after_motion_mps2",
        "distance_at_10s_after_motion_m",
        "time_to_50pct_peak_u_after_motion_s",
        "time_to_90pct_peak_u_after_motion_s",
    ]
    return {"motion": compare_metrics(sim_metrics, real_bench["straight_metrics"], keys)}


def turn_comparison(sim_metrics: Dict[str, Any], real_bench: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "peak_abs_yaw_rate_degps",
        "time_to_90deg_after_turn_s",
        "time_to_180deg_after_turn_s",
        "radius_first_90deg_m",
        "radius_first_180deg_m",
        "u_body_10s_after_turn_mps",
    ]
    return {"turn": compare_metrics(sim_metrics, real_bench["turn_metrics"], keys)}

