"""Automatic parameter sweep for the simplified Python ship model.

Loads test_3_metrics.json and test_4_metrics.json, replays the latched S1/S2
histories into the simplified ship_model.py, extracts comparable metrics, and
ranks candidate values of THRUST_COEF, DRAG_COEF, TURN_COEF while MASS is fixed.
"""

from __future__ import annotations

import csv
import importlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
REAL_TEST3_JSON = ROOT / "test_3_metrics.json"
REAL_TEST4_JSON = ROOT / "test_4_metrics.json"
OUTPUT_DIR = ROOT / "sweep_results"
SHIP_MODEL_MODULE = "ship_model"

FIXED_MASS = 64.55

THRUST_GRID = [0.03, 0.04, 0.05, 0.06, 0.07]
DRAG_GRID = [6.0, 8.0, 10.0, 12.0, 14.0]
TURN_GRID = [200.0, 400.0, 800.0, 1200.0, 1600.0, 2200.0]

RPM_MAX = 30.0
RUDDER_CMD_MAX_PERCENT = 100.0

OVERRIDE_S1_NEUTRAL: Optional[float] = None
OVERRIDE_S2_NEUTRAL: Optional[float] = None
OVERRIDE_S1_SCALE: Optional[float] = None
OVERRIDE_S2_FULL_FWD: Optional[float] = None

WEIGHTS = {
    "speed_peak": 2.0,
    "speed_accel_0_2": 2.0,
    "speed_accel_0_5": 1.5,
    "speed_dist_10": 1.5,
    "speed_t50": 1.0,
    "speed_t90": 1.0,
    "turn_t90": 2.0,
    "turn_t180": 2.0,
    "turn_peak_yaw": 2.0,
    "turn_r90": 1.5,
    "turn_r180": 1.0,
    "turn_u10": 1.0,
}

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

def safe_rel_error(sim: Optional[float], real: Optional[float], floor: float = 1e-6) -> float:
    if sim is None or real is None:
        return 10.0
    denom = max(abs(real), floor)
    return abs(sim - real) / denom

@dataclass
class ReplaySeries:
    t_sec: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    yaw_rate_real: np.ndarray
    u_body_real: np.ndarray

@dataclass
class ReplayMapping:
    s1_neutral: float
    s1_scale: float
    s2_neutral: float
    s2_full_fwd: float

    def s1_to_rudder_percent(self, s1_val: float) -> float:
        if self.s1_scale <= 0:
            return 0.0
        z = (s1_val - self.s1_neutral) / self.s1_scale
        z = float(np.clip(z, -1.0, 1.0))
        return z * RUDDER_CMD_MAX_PERCENT

    def s2_to_rpm(self, s2_val: float) -> float:
        denom = self.s2_full_fwd - self.s2_neutral
        if abs(denom) < 1e-9:
            return 0.0
        z = (s2_val - self.s2_neutral) / denom
        z = float(np.clip(z, 0.0, 1.0))
        return z * RPM_MAX

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_replay_series(data: Dict[str, Any]) -> ReplaySeries:
    series = data["series"]
    return ReplaySeries(
        t_sec=np.asarray(series["t_sec"], dtype=float),
        s1=np.asarray(series["s1"], dtype=float),
        s2=np.asarray(series["s2"], dtype=float),
        yaw_rate_real=np.asarray(series["yaw_rate_degps"], dtype=float),
        u_body_real=np.asarray(series["u_body_mps"], dtype=float),
    )

def infer_mapping(data: Dict[str, Any], series: ReplaySeries) -> ReplayMapping:
    sm = data.get("straight_metrics", {})
    tm = data.get("turn_metrics", {})

    s1_neutral = OVERRIDE_S1_NEUTRAL
    if s1_neutral is None:
        cand = tm.get("s1_neutral", None)
        if cand is None or cand == 0.0:
            nonzero = series.s1[np.isfinite(series.s1) & (series.s1 > 0)]
            cand = float(np.median(nonzero[:20])) if nonzero.size else 1500.0
        s1_neutral = float(cand)

    s2_neutral = OVERRIDE_S2_NEUTRAL
    if s2_neutral is None:
        cand = sm.get("s2_neutral", None)
        if cand is None or cand == 0.0:
            nonzero = series.s2[np.isfinite(series.s2) & (series.s2 > 0)]
            cand = float(np.median(nonzero[:20])) if nonzero.size else 1500.0
        s2_neutral = float(cand)

    s1_scale = OVERRIDE_S1_SCALE
    if s1_scale is None:
        dev = np.abs(series.s1 - s1_neutral)
        s1_scale = float(np.max(dev[np.isfinite(dev)])) if np.any(np.isfinite(dev)) else 500.0
        s1_scale = max(s1_scale, 1.0)

    s2_full_fwd = OVERRIDE_S2_FULL_FWD
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
    )

def load_ship_model_module():
    return importlib.import_module(SHIP_MODEL_MODULE)

def simulate_replay(series: ReplaySeries, mapping: ReplayMapping,
                    thrust_coef: float, drag_coef: float, turn_coef: float, mass: float) -> Dict[str, np.ndarray]:
    ship_model = load_ship_model_module()
    ship_model.MASS = float(mass)
    ship_model.THRUST_COEF = float(thrust_coef)
    ship_model.DRAG_COEF = float(drag_coef)
    ship_model.TURN_COEF = float(turn_coef)
    ship_model.MOMINERTIA = 0.5 * ship_model.MASS * ship_model.RUDDEROFFSET ** 2

    model = ship_model.ShipModel()

    t = series.t_sec
    n = len(t)
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    yaw_deg = np.zeros(n, dtype=float)
    yaw_rate_degps = np.zeros(n, dtype=float)
    u_body = np.zeros(n, dtype=float)

    xk = 0.0
    yk = 0.0

    for i in range(n):
        dt = 0.1 if i == 0 else max(t[i] - t[i - 1], 1e-3)
        rudder_percent = mapping.s1_to_rudder_percent(series.s1[i])
        rpm = mapping.s2_to_rpm(series.s2[i])

        dx, dy, hdg_deg, yawrate_deg = model.update(rpm, rudder_percent, dt)
        xk += dx
        yk += dy

        x[i] = xk
        y[i] = yk
        yaw_deg[i] = hdg_deg
        yaw_rate_degps[i] = yawrate_deg
        u_body[i] = model._v

    return {
        "t_sec": t.copy(),
        "x_m": x,
        "y_m": y,
        "yaw_deg": yaw_deg,
        "yaw_rate_degps": yaw_rate_degps,
        "u_body_mps": u_body,
    }

def extract_sim_metrics(sim: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Optional[float]]]:
    t = np.asarray(sim["t_sec"], dtype=float)
    x = np.asarray(sim["x_m"], dtype=float)
    y = np.asarray(sim["y_m"], dtype=float)
    yaw = np.asarray(sim["yaw_deg"], dtype=float)
    yaw_rate = np.asarray(sim["yaw_rate_degps"], dtype=float)
    u = np.asarray(sim["u_body_mps"], dtype=float)

    idx_motion = first_sustained_index(u, threshold=0.05, count=3)
    if idx_motion is None:
        idx_motion = 0

    t_rel_m = t[idx_motion:] - t[idx_motion]
    u_rel_m = u[idx_motion:]
    x_rel_m = x[idx_motion:]
    y_rel_m = y[idx_motion:]
    dist_rel_m = cumulative_distance(x_rel_m, y_rel_m)
    peak_u = float(np.max(u_rel_m)) if len(u_rel_m) else None

    straight = {
        "peak_u_body_mps": peak_u,
        "initial_accel_0_2_after_motion_mps2": slope_over_window(t_rel_m, u_rel_m, 0.0, 2.0),
        "initial_accel_0_5_after_motion_mps2": slope_over_window(t_rel_m, u_rel_m, 0.0, 5.0),
        "distance_at_10s_after_motion_m": sample_at_time(t_rel_m, dist_rel_m, 10.0),
        "time_to_50pct_peak_u_after_motion_s": None if peak_u is None else first_crossing_time(t_rel_m, u_rel_m, 0.5 * peak_u),
        "time_to_90pct_peak_u_after_motion_s": None if peak_u is None else first_crossing_time(t_rel_m, u_rel_m, 0.9 * peak_u),
        "u_body_at_10s_after_motion_mps": sample_at_time(t_rel_m, u_rel_m, 10.0),
    }

    idx_turn = first_sustained_abs_index(yaw_rate, threshold=1.0, count=3)
    if idx_turn is None:
        idx_turn = 0

    t_rel_t = t[idx_turn:] - t[idx_turn]
    x_rel_t = x[idx_turn:]
    y_rel_t = y[idx_turn:]
    yaw_rel_t = yaw[idx_turn:]
    yaw_rate_rel_t = yaw_rate[idx_turn:]
    u_rel_t = u[idx_turn:]

    yaw_u = unwrap_heading_deg(yaw_rel_t)
    dpsi = yaw_u - yaw_u[0]

    idx90 = np.where(np.abs(dpsi) >= 90.0)[0]
    idx180 = np.where(np.abs(dpsi) >= 180.0)[0]
    r90 = circle_fit_radius(x_rel_t[:idx90[0] + 1], y_rel_t[:idx90[0] + 1]) if idx90.size > 0 else None
    r180 = circle_fit_radius(x_rel_t[:idx180[0] + 1], y_rel_t[:idx180[0] + 1]) if idx180.size > 0 else None

    turn = {
        "peak_abs_yaw_rate_degps": float(np.max(np.abs(yaw_rate_rel_t))) if len(yaw_rate_rel_t) else None,
        "time_to_90deg_after_turn_s": first_abs_crossing_time(t_rel_t, dpsi, 90.0),
        "time_to_180deg_after_turn_s": first_abs_crossing_time(t_rel_t, dpsi, 180.0),
        "radius_first_90deg_m": r90,
        "radius_first_180deg_m": r180,
        "u_body_10s_after_turn_mps": sample_at_time(t_rel_t, u_rel_t, 10.0),
    }

    return {"straight_metrics": straight, "turn_metrics": turn}

def score_against_real(sim_metrics: Dict[str, Dict[str, Optional[float]]],
                       real3: Dict[str, Any], real4: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    s3 = real3["straight_metrics"]
    t4 = real4["turn_metrics"]

    sim_s = sim_metrics["straight_metrics"]
    sim_t = sim_metrics["turn_metrics"]

    parts = {
        "speed_peak": safe_rel_error(sim_s["peak_u_body_mps"], s3["peak_u_body_mps"]),
        "speed_accel_0_2": safe_rel_error(sim_s["initial_accel_0_2_after_motion_mps2"], s3["initial_accel_0_2_after_motion_mps2"]),
        "speed_accel_0_5": safe_rel_error(sim_s["initial_accel_0_5_after_motion_mps2"], s3["initial_accel_0_5_after_motion_mps2"]),
        "speed_dist_10": safe_rel_error(sim_s["distance_at_10s_after_motion_m"], s3["distance_at_10s_after_motion_m"]),
        "speed_t50": safe_rel_error(sim_s["time_to_50pct_peak_u_after_motion_s"], s3["time_to_50pct_peak_u_after_motion_s"]),
        "speed_t90": safe_rel_error(sim_s["time_to_90pct_peak_u_after_motion_s"], s3["time_to_90pct_peak_u_after_motion_s"]),
        "turn_t90": safe_rel_error(sim_t["time_to_90deg_after_turn_s"], t4["time_to_90deg_after_turn_s"]),
        "turn_t180": safe_rel_error(sim_t["time_to_180deg_after_turn_s"], t4["time_to_180deg_after_turn_s"]),
        "turn_peak_yaw": safe_rel_error(sim_t["peak_abs_yaw_rate_degps"], t4["peak_abs_yaw_rate_degps"]),
        "turn_r90": safe_rel_error(sim_t["radius_first_90deg_m"], t4["radius_first_90deg_m"]),
        "turn_r180": safe_rel_error(sim_t["radius_first_180deg_m"], t4["radius_first_180deg_m"]),
        "turn_u10": safe_rel_error(sim_t["u_body_10s_after_turn_mps"], t4["u_body_10s_after_turn_mps"]),
    }

    total = 0.0
    for k, err in parts.items():
        total += WEIGHTS[k] * err
    return total, parts

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real3 = load_json(REAL_TEST3_JSON)
    real4 = load_json(REAL_TEST4_JSON)

    series3 = build_replay_series(real3)
    series4 = build_replay_series(real4)

    mapping3 = infer_mapping(real3, series3)
    mapping4 = infer_mapping(real4, series4)

    results: List[Dict[str, Any]] = []

    combos = list(itertools.product(THRUST_GRID, DRAG_GRID, TURN_GRID))
    print(f"Running {len(combos)} combinations with MASS fixed at {FIXED_MASS} ...")

    for i, (thrust_coef, drag_coef, turn_coef) in enumerate(combos, start=1):
        sim3 = simulate_replay(series3, mapping3, thrust_coef, drag_coef, turn_coef, FIXED_MASS)
        sim4 = simulate_replay(series4, mapping4, thrust_coef, drag_coef, turn_coef, FIXED_MASS)

        sim3_metrics = extract_sim_metrics(sim3)
        sim4_metrics = extract_sim_metrics(sim4)

        fused_metrics = {
            "straight_metrics": sim3_metrics["straight_metrics"],
            "turn_metrics": sim4_metrics["turn_metrics"],
        }

        score, parts = score_against_real(fused_metrics, real3, real4)

        row = {
            "MASS": FIXED_MASS,
            "THRUST_COEF": thrust_coef,
            "DRAG_COEF": drag_coef,
            "TURN_COEF": turn_coef,
            "score_total": score,
            **parts,
            **{f"sim3_{k}": v for k, v in sim3_metrics["straight_metrics"].items()},
            **{f"sim4_{k}": v for k, v in sim4_metrics["turn_metrics"].items()},
        }
        results.append(row)

    results.sort(key=lambda r: r["score_total"])

    csv_path = OUTPUT_DIR / "parameter_sweep_ranked.csv"
    keys = list(results[0].keys()) if results else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "parameter_sweep_top10.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results[:10], f, indent=2)

    print("\nTop 10 candidates:")
    for rank, row in enumerate(results[:10], start=1):
        print(
            f"{rank:2d}. score={row['score_total']:.4f}  "
            f"T={row['THRUST_COEF']:.4f}  D={row['DRAG_COEF']:.4f}  TURN={row['TURN_COEF']:.2f}"
        )

    print(f"\nSaved ranked results to:\n- {csv_path}\n- {json_path}")

if __name__ == "__main__":
    main()
