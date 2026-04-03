from __future__ import annotations

import importlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bluefin_test_utils import (
    extract_motion_metrics,
    extract_turn_metrics,
    load_json,
    plot_open_loop_response,
    plot_path,
    save_json,
)

REAL_SPEED_JSON = "test_3_comparison.json"
REAL_TURN_JSON = "test_4_comparison.json"
OUT_DIR = "v2_focus_results"
MODEL_MODULE = "ship_model_bluefin_matlab_style_v2"
DT = 0.1
STRAIGHT_DURATION = 40.0
TURN_DURATION = 50.0
MASS = 64.55
TOP_SURGE_K = 4

MOTION_KEYS = [
    "peak_u_body_mps",
    "initial_accel_0_2_after_motion_mps2",
    "initial_accel_0_5_after_motion_mps2",
    "distance_at_10s_after_motion_m",
    "time_to_50pct_peak_u_after_motion_s",
    "time_to_90pct_peak_u_after_motion_s",
]
TURN_KEYS = [
    "peak_abs_yaw_rate_degps",
    "time_to_90deg_after_turn_s",
    "time_to_180deg_after_turn_s",
    "radius_first_90deg_m",
    "radius_first_180deg_m",
    "u_body_10s_after_turn_mps",
]
MOTION_WEIGHTS = {
    "peak_u_body_mps": 2.0,
    "initial_accel_0_2_after_motion_mps2": 2.0,
    "initial_accel_0_5_after_motion_mps2": 1.5,
    "distance_at_10s_after_motion_m": 2.0,
    "time_to_50pct_peak_u_after_motion_s": 1.5,
    "time_to_90pct_peak_u_after_motion_s": 1.0,
}
TURN_WEIGHTS = {
    "peak_abs_yaw_rate_degps": 2.5,
    "time_to_90deg_after_turn_s": 2.0,
    "time_to_180deg_after_turn_s": 1.5,
    "radius_first_90deg_m": 1.5,
    "radius_first_180deg_m": 1.0,
    "u_body_10s_after_turn_mps": 1.0,
}


def rel_error(sim: Optional[float], real: Optional[float], floor: float = 1e-6) -> float:
    if sim is None or real is None:
        return 10.0
    return abs(sim - real) / max(abs(real), floor)


def get_real_targets(data: Dict[str, Any], section: str, keys: List[str]) -> Dict[str, Optional[float]]:
    src = data[section]
    out = {}
    for k in keys:
        item = src.get(k)
        out[k] = item.get("real") if isinstance(item, dict) else item
    return out


def score_section(sim_metrics: Dict[str, Any], real_targets: Dict[str, Optional[float]], weights: Dict[str, float]) -> Dict[str, Any]:
    parts: Dict[str, float] = {}
    total = 0.0
    for key, wt in weights.items():
        err = rel_error(sim_metrics.get(key), real_targets.get(key))
        parts[key] = err
        total += wt * err
    return {"score_total": total, "parts": parts}


def simulate_open_loop(**cfg: float) -> Dict[str, np.ndarray]:
    ship_model = importlib.import_module(MODEL_MODULE)
    importlib.reload(ship_model)

    # patch globals
    for key, value in cfg.items():
        if hasattr(ship_model, key):
            setattr(ship_model, key, float(value))

    model = ship_model.ShipModel()

    duration_s = cfg["duration_s"]
    dt = cfg["dt"]
    rpm = cfg["rpm"]
    rudder_deg = cfg["rudder_deg"]
    n = int(np.floor(duration_s / dt)) + 1
    t = np.arange(n, dtype=float) * dt
    x = np.zeros(n)
    y = np.zeros(n)
    heading_deg = np.zeros(n)
    yaw_rate_degps = np.zeros(n)
    u_body = np.zeros(n)
    v_body = np.zeros(n)
    rudder_deg_cmd = np.full(n, rudder_deg)
    rudder_percent_cmd = np.full(n, (rudder_deg / 40.0) * 100.0)
    rpm_cmd = np.full(n, rpm)
    xk = 0.0
    yk = 0.0
    rudder_percent = (rudder_deg / 40.0) * 100.0
    for i in range(n):
        dx, dy, hdg, yaw = model.update(rpm, rudder_percent, dt)
        xk += dx
        yk += dy
        x[i] = xk
        y[i] = yk
        heading_deg[i] = hdg
        yaw_rate_degps[i] = yaw
        u_body[i] = getattr(model, "_v", 0.0)
        v_body[i] = getattr(model, "_v_sway", 0.0)
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


def build_comparison(section: str, keys: List[str], sim_metrics: Dict[str, Any], real_targets: Dict[str, Optional[float]]) -> Dict[str, Any]:
    return {
        section: {
            k: {
                "real": real_targets.get(k),
                "sim": sim_metrics.get(k),
                "rel_error": rel_error(sim_metrics.get(k), real_targets.get(k)),
            }
            for k in keys
        }
    }


def main() -> None:
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    real_speed = load_json(Path(REAL_SPEED_JSON))
    real_turn = load_json(Path(REAL_TURN_JSON))
    real_motion = get_real_targets(real_speed, "motion", MOTION_KEYS)
    real_turn_targets = get_real_targets(real_turn, "turn", TURN_KEYS)

    # Stage 1: fix surge shape
    straight_rpm_grid = [14.0, 15.0, 16.0]
    thrust_grid = [0.05, 0.06, 0.07]
    drag_grid = [1.0, 1.2, 1.5]
    low_speed_boost_grid = [0.8, 1.2, 1.6]
    high_speed_decay_grid = [0.10, 0.18, 0.26]
    linear_surge_damp_grid = [1.0, 1.5, 2.0]

    stage1_rows = []
    for rpm, thrust, drag, boost, decay, surge_damp in itertools.product(
        straight_rpm_grid, thrust_grid, drag_grid, low_speed_boost_grid, high_speed_decay_grid, linear_surge_damp_grid
    ):
        sim = simulate_open_loop(
            MASS=MASS,
            THRUST_COEF=thrust,
            DRAG_COEF=drag,
            THRUST_LOW_SPEED_BOOST=boost,
            THRUST_HIGH_SPEED_DECAY=decay,
            LINEAR_SURGE_DAMP=surge_damp,
            TURN_COEF=4.0,
            RUDDER_FORCE_SCALE=0.16,
            RUDDER_YAW_SCALE=1.35,
            RUDDER_X_DRAG_SCALE=0.18,
            LINEAR_YAW_DAMP=3.5,
            SURGE_INERTIA_SCALE=1.0,
            YAW_INERTIA_SCALE=1.0,
            rpm=rpm,
            rudder_deg=0.0,
            duration_s=STRAIGHT_DURATION,
            dt=DT,
        )
        mm = extract_motion_metrics(sim)
        score = score_section(mm, real_motion, MOTION_WEIGHTS)
        stage1_rows.append({
            "straight_rpm": rpm,
            "MASS": MASS,
            "THRUST_COEF": thrust,
            "DRAG_COEF": drag,
            "THRUST_LOW_SPEED_BOOST": boost,
            "THRUST_HIGH_SPEED_DECAY": decay,
            "LINEAR_SURGE_DAMP": surge_damp,
            "score_total": score["score_total"],
            **{f"err_{k}": v for k, v in score["parts"].items()},
            **mm,
        })
    stage1_rows.sort(key=lambda r: r["score_total"])
    save_json(out / "stage1_v2_top20.json", {"rows": stage1_rows[:20]})
    top_stage1 = stage1_rows[:TOP_SURGE_K]

    # Stage 2: turn refinement around top surge configs
    turn_rpm_grid = [20, 22, 24]
    turn_rudder_grid = [25, 30]
    turn_coef_grid = [1.5, 2.0, 2.5, 3.0]
    rudder_force_grid = [0.20, 0.24, 0.28, 0.32]
    rudder_yaw_grid = [1.7, 2.0, 2.3, 2.6]
    rudder_xdrag_grid = [0.02, 0.05, 0.08]
    yaw_damp_grid = [1.5, 2.0, 2.5, 3.0]

    joint_rows = []
    for base in top_stage1:
        for turn_rpm, turn_rudder_deg, turn_coef, rudder_force, rudder_yaw, rudder_xdrag, yaw_damp in itertools.product(
            turn_rpm_grid, turn_rudder_grid, turn_coef_grid, rudder_force_grid, rudder_yaw_grid, rudder_xdrag_grid, yaw_damp_grid
        ):
            # surge sim for joint scoring
            sim_straight = simulate_open_loop(
                MASS=MASS,
                THRUST_COEF=base["THRUST_COEF"],
                DRAG_COEF=base["DRAG_COEF"],
                THRUST_LOW_SPEED_BOOST=base["THRUST_LOW_SPEED_BOOST"],
                THRUST_HIGH_SPEED_DECAY=base["THRUST_HIGH_SPEED_DECAY"],
                LINEAR_SURGE_DAMP=base["LINEAR_SURGE_DAMP"],
                TURN_COEF=turn_coef,
                RUDDER_FORCE_SCALE=rudder_force,
                RUDDER_YAW_SCALE=rudder_yaw,
                RUDDER_X_DRAG_SCALE=rudder_xdrag,
                LINEAR_YAW_DAMP=yaw_damp,
                SURGE_INERTIA_SCALE=1.0,
                YAW_INERTIA_SCALE=1.0,
                rpm=base["straight_rpm"],
                rudder_deg=0.0,
                duration_s=STRAIGHT_DURATION,
                dt=DT,
            )
            sim_turn = simulate_open_loop(
                MASS=MASS,
                THRUST_COEF=base["THRUST_COEF"],
                DRAG_COEF=base["DRAG_COEF"],
                THRUST_LOW_SPEED_BOOST=base["THRUST_LOW_SPEED_BOOST"],
                THRUST_HIGH_SPEED_DECAY=base["THRUST_HIGH_SPEED_DECAY"],
                LINEAR_SURGE_DAMP=base["LINEAR_SURGE_DAMP"],
                TURN_COEF=turn_coef,
                RUDDER_FORCE_SCALE=rudder_force,
                RUDDER_YAW_SCALE=rudder_yaw,
                RUDDER_X_DRAG_SCALE=rudder_xdrag,
                LINEAR_YAW_DAMP=yaw_damp,
                SURGE_INERTIA_SCALE=1.0,
                YAW_INERTIA_SCALE=1.0,
                rpm=turn_rpm,
                rudder_deg=turn_rudder_deg,
                duration_s=TURN_DURATION,
                dt=DT,
            )
            mm = extract_motion_metrics(sim_straight)
            tm = extract_turn_metrics(sim_turn)
            sscore = score_section(mm, real_motion, MOTION_WEIGHTS)
            tscore = score_section(tm, real_turn_targets, TURN_WEIGHTS)
            joint_rows.append({
                "MASS": MASS,
                "straight_rpm": base["straight_rpm"],
                "turn_rpm": turn_rpm,
                "turn_rudder_deg": turn_rudder_deg,
                "THRUST_COEF": base["THRUST_COEF"],
                "DRAG_COEF": base["DRAG_COEF"],
                "THRUST_LOW_SPEED_BOOST": base["THRUST_LOW_SPEED_BOOST"],
                "THRUST_HIGH_SPEED_DECAY": base["THRUST_HIGH_SPEED_DECAY"],
                "LINEAR_SURGE_DAMP": base["LINEAR_SURGE_DAMP"],
                "TURN_COEF": turn_coef,
                "RUDDER_FORCE_SCALE": rudder_force,
                "RUDDER_YAW_SCALE": rudder_yaw,
                "RUDDER_X_DRAG_SCALE": rudder_xdrag,
                "LINEAR_YAW_DAMP": yaw_damp,
                "surge_score": sscore["score_total"],
                "turn_score": tscore["score_total"],
                "joint_score": sscore["score_total"] + tscore["score_total"],
                **{f"surge_err_{k}": v for k, v in sscore["parts"].items()},
                **{f"turn_err_{k}": v for k, v in tscore["parts"].items()},
                **{f"surge_{k}": v for k, v in mm.items()},
                **{f"turn_{k}": v for k, v in tm.items()},
            })
    joint_rows.sort(key=lambda r: r["joint_score"])
    save_json(out / "joint_v2_top20.json", {"rows": joint_rows[:20]})

    best = joint_rows[0]
    best_straight = simulate_open_loop(
        MASS=MASS,
        THRUST_COEF=best["THRUST_COEF"],
        DRAG_COEF=best["DRAG_COEF"],
        THRUST_LOW_SPEED_BOOST=best["THRUST_LOW_SPEED_BOOST"],
        THRUST_HIGH_SPEED_DECAY=best["THRUST_HIGH_SPEED_DECAY"],
        LINEAR_SURGE_DAMP=best["LINEAR_SURGE_DAMP"],
        TURN_COEF=best["TURN_COEF"],
        RUDDER_FORCE_SCALE=best["RUDDER_FORCE_SCALE"],
        RUDDER_YAW_SCALE=best["RUDDER_YAW_SCALE"],
        RUDDER_X_DRAG_SCALE=best["RUDDER_X_DRAG_SCALE"],
        LINEAR_YAW_DAMP=best["LINEAR_YAW_DAMP"],
        SURGE_INERTIA_SCALE=1.0,
        YAW_INERTIA_SCALE=1.0,
        rpm=best["straight_rpm"],
        rudder_deg=0.0,
        duration_s=STRAIGHT_DURATION,
        dt=DT,
    )
    best_turn = simulate_open_loop(
        MASS=MASS,
        THRUST_COEF=best["THRUST_COEF"],
        DRAG_COEF=best["DRAG_COEF"],
        THRUST_LOW_SPEED_BOOST=best["THRUST_LOW_SPEED_BOOST"],
        THRUST_HIGH_SPEED_DECAY=best["THRUST_HIGH_SPEED_DECAY"],
        LINEAR_SURGE_DAMP=best["LINEAR_SURGE_DAMP"],
        TURN_COEF=best["TURN_COEF"],
        RUDDER_FORCE_SCALE=best["RUDDER_FORCE_SCALE"],
        RUDDER_YAW_SCALE=best["RUDDER_YAW_SCALE"],
        RUDDER_X_DRAG_SCALE=best["RUDDER_X_DRAG_SCALE"],
        LINEAR_YAW_DAMP=best["LINEAR_YAW_DAMP"],
        SURGE_INERTIA_SCALE=1.0,
        YAW_INERTIA_SCALE=1.0,
        rpm=best["turn_rpm"],
        rudder_deg=best["turn_rudder_deg"],
        duration_s=TURN_DURATION,
        dt=DT,
    )
    best_motion_metrics = extract_motion_metrics(best_straight)
    best_turn_metrics = extract_turn_metrics(best_turn)
    best_motion_cmp = build_comparison("motion", MOTION_KEYS, best_motion_metrics, real_motion)
    best_turn_cmp = build_comparison("turn", TURN_KEYS, best_turn_metrics, real_turn_targets)
    best_cmp = {**best_motion_cmp, **best_turn_cmp}

    save_json(out / "best_v2_joint_config.json", best)
    save_json(out / "best_v2_motion_metrics.json", {"motion_metrics": best_motion_metrics})
    save_json(out / "best_v2_turn_metrics.json", {"turn_metrics": best_turn_metrics})
    save_json(out / "best_v2_motion_comparison.json", best_motion_cmp)
    save_json(out / "best_v2_turn_comparison.json", best_turn_cmp)
    save_json(out / "best_v2_joint_comparison.json", best_cmp)
    plot_open_loop_response(out / "best_v2_straight_response.png", best_straight, "Best v2 straight response")
    plot_path(out / "best_v2_straight_path.png", best_straight, "Best v2 straight path")
    plot_open_loop_response(out / "best_v2_turn_response.png", best_turn, "Best v2 turn response")
    plot_path(out / "best_v2_turn_path.png", best_turn, "Best v2 turn path")

    print("Best v2 joint config saved to", out / "best_v2_joint_config.json")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
