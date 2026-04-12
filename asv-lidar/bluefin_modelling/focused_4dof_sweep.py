from __future__ import annotations

import importlib
import itertools
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

REAL_SPEED_JSON = "test_3_metrics.json"
REAL_TURN_JSON = "test_4_metrics.json"
OUT_DIR = "fourdof_focus_results"
MODEL_MODULE = "ship_model_bluefin_4dof"
THIS_DIR = Path(__file__).resolve().parent
DT = 0.1
STRAIGHT_DURATION = 40.0
TURN_DURATION = 50.0
TURN_LEADIN_DURATION = 5.0
TOP_SURGE_K = 3

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

V2_BASELINE_JOINT_SCORE = 6.546219039410511
V2_BASELINE_CONFIG = {
    "straight_rpm": 15.0,
    "turn_rpm": 24.0,
    "turn_rudder_deg": 30.0,
}


def rel_error(sim: Optional[float], real: Optional[float], floor: float = 1e-6) -> float:
    if sim is None or real is None:
        return 10.0
    return abs(sim - real) / max(abs(real), floor)


def get_real_targets(data: Dict[str, Any], section: str, keys: List[str]) -> Dict[str, Optional[float]]:
    src = data[section]
    return {k: src.get(k) for k in keys}


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

    for key, value in cfg.items():
        if hasattr(ship_model, key):
            setattr(ship_model, key, float(value))

    model = ship_model.ShipModel()

    duration_s = cfg["duration_s"]
    dt = cfg["dt"]
    rpm = cfg["rpm"]
    rudder_deg = cfg["rudder_deg"]
    thruster_rpm = cfg.get("thruster_rpm", 0.0)

    n = int(np.floor(duration_s / dt)) + 1
    t = np.arange(n, dtype=float) * dt
    x = np.zeros(n)
    y = np.zeros(n)
    heading_deg = np.zeros(n)
    yaw_rate_degps = np.zeros(n)
    u_body = np.zeros(n)
    v_body = np.zeros(n)
    roll_deg = np.zeros(n)
    rudder_deg_cmd = np.full(n, rudder_deg)
    rudder_percent_cmd = np.full(n, (rudder_deg / 40.0) * 100.0)
    rpm_cmd = np.full(n, rpm)
    xk = 0.0
    yk = 0.0
    rudder_percent = (rudder_deg / 40.0) * 100.0

    for i in range(n):
        dx, dy, hdg, yaw = model.update(rpm, rudder_percent, dt, thruster_rpm=thruster_rpm)
        xk += dx
        yk += dy
        x[i] = xk
        y[i] = yk
        heading_deg[i] = hdg
        yaw_rate_degps[i] = yaw
        u_body[i] = getattr(model, "_u", getattr(model, "_v", 0.0))
        v_body[i] = getattr(model, "_v_sway", 0.0)
        roll_deg[i] = np.degrees(getattr(model, "_phi", 0.0))

    return {
        "t_sec": t,
        "x_m": x,
        "y_m": y,
        "heading_deg": heading_deg,
        "yaw_rate_degps": yaw_rate_degps,
        "u_body_mps": u_body,
        "v_body_mps": v_body,
        "roll_deg": roll_deg,
        "rudder_deg_cmd": rudder_deg_cmd,
        "rudder_percent_cmd": rudder_percent_cmd,
        "rpm_cmd": rpm_cmd,
    }


def simulate_turn_with_leadin(**cfg: float) -> Dict[str, np.ndarray]:
    ship_model = importlib.import_module(MODEL_MODULE)
    importlib.reload(ship_model)

    for key, value in cfg.items():
        if hasattr(ship_model, key):
            setattr(ship_model, key, float(value))

    model = ship_model.ShipModel()

    duration_s = cfg["duration_s"]
    dt = cfg["dt"]
    rpm = cfg["rpm"]
    rudder_deg = cfg["rudder_deg"]
    turn_leadin_s = cfg.get("turn_leadin_s", TURN_LEADIN_DURATION)
    thruster_rpm = cfg.get("thruster_rpm", 0.0)

    n = int(np.floor(duration_s / dt)) + 1
    t = np.arange(n, dtype=float) * dt
    x = np.zeros(n)
    y = np.zeros(n)
    heading_deg = np.zeros(n)
    yaw_rate_degps = np.zeros(n)
    u_body = np.zeros(n)
    v_body = np.zeros(n)
    roll_deg = np.zeros(n)
    rudder_deg_cmd = np.zeros(n)
    rudder_percent_cmd = np.zeros(n)
    rpm_cmd = np.full(n, rpm)
    xk = 0.0
    yk = 0.0

    for i in range(n):
        rudder_deg_i = 0.0 if t[i] < turn_leadin_s else float(rudder_deg)
        rudder_percent_i = (rudder_deg_i / 40.0) * 100.0
        dx, dy, hdg, yaw = model.update(rpm, rudder_percent_i, dt, thruster_rpm=thruster_rpm)
        xk += dx
        yk += dy
        x[i] = xk
        y[i] = yk
        heading_deg[i] = hdg
        yaw_rate_degps[i] = yaw
        u_body[i] = getattr(model, "_u", getattr(model, "_v", 0.0))
        v_body[i] = getattr(model, "_v_sway", 0.0)
        roll_deg[i] = np.degrees(getattr(model, "_phi", 0.0))
        rudder_deg_cmd[i] = rudder_deg_i
        rudder_percent_cmd[i] = rudder_percent_i

    return {
        "t_sec": t,
        "x_m": x,
        "y_m": y,
        "heading_deg": heading_deg,
        "yaw_rate_degps": yaw_rate_degps,
        "u_body_mps": u_body,
        "v_body_mps": v_body,
        "roll_deg": roll_deg,
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
    out = THIS_DIR / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    real_speed = load_json(THIS_DIR / REAL_SPEED_JSON)
    real_turn = load_json(THIS_DIR / REAL_TURN_JSON)
    real_motion = get_real_targets(real_speed, "straight_metrics", MOTION_KEYS)
    real_turn_targets = get_real_targets(real_turn, "turn_metrics", TURN_KEYS)

    # Stage 1: straight-line alignment while keeping turn scales tame.
    straight_rpm_grid = [12.0, 15.0, 18.0]
    rpm_scale_grid = [100.0, 120.0, 140.0]
    prop_thrust_grid = [1.0, 1.4, 1.8, 2.2]
    surge_damp_grid = [1.0, 1.5, 2.0]

    stage1_rows = []
    for rpm, rpm_scale, thrust_scale, surge_damp in itertools.product(
        straight_rpm_grid, rpm_scale_grid, prop_thrust_grid, surge_damp_grid
    ):
        sim = simulate_open_loop(
            RPM_COMMAND_SCALE=rpm_scale,
            PROPELLER_THRUST_SCALE=thrust_scale,
            RUDDER_FORCE_SCALE=0.15,
            RUDDER_YAW_SCALE=1.5,
            RUDDER_X_DRAG_SCALE=0.02,
            LINEAR_SURGE_DAMP=surge_damp,
            LINEAR_YAW_DAMP=0.0,
            ROLL_DAMP_SCALE=4.0,
            BOW_THRUSTER_SCALE=1.0,
            rpm=rpm,
            rudder_deg=0.0,
            duration_s=STRAIGHT_DURATION,
            dt=DT,
        )
        mm = extract_motion_metrics(sim)
        score = score_section(mm, real_motion, MOTION_WEIGHTS)
        stage1_rows.append({
            "straight_rpm": rpm,
            "RPM_COMMAND_SCALE": rpm_scale,
            "PROPELLER_THRUST_SCALE": thrust_scale,
            "LINEAR_SURGE_DAMP": surge_damp,
            "score_total": score["score_total"],
            **{f"err_{k}": v for k, v in score["parts"].items()},
            **mm,
        })

    stage1_rows.sort(key=lambda r: r["score_total"])
    save_json(out / "stage1_4dof_top20.json", {"rows": stage1_rows[:20]})
    top_stage1 = stage1_rows[:TOP_SURGE_K]

    # Stage 2: refine turn response around the best surge candidates.
    turn_rpm_grid = [15.0, 20.0, 24.0]
    turn_rudder_grid = [25.0, 30.0]
    rudder_force_grid = [0.10, 0.15]
    rudder_yaw_grid = [1.0, 1.5]
    rudder_xdrag_grid = [0.02, 0.05]
    yaw_damp_grid = [0.0, 0.5, 1.0]
    roll_damp_grid = [3.0, 4.0]

    joint_rows = []
    for base in top_stage1:
        for turn_rpm, turn_rudder_deg, rudder_force, rudder_yaw, rudder_xdrag, yaw_damp, roll_damp in itertools.product(
            turn_rpm_grid,
            turn_rudder_grid,
            rudder_force_grid,
            rudder_yaw_grid,
            rudder_xdrag_grid,
            yaw_damp_grid,
            roll_damp_grid,
        ):
            sim_straight = simulate_open_loop(
                RPM_COMMAND_SCALE=base["RPM_COMMAND_SCALE"],
                PROPELLER_THRUST_SCALE=base["PROPELLER_THRUST_SCALE"],
                RUDDER_FORCE_SCALE=rudder_force,
                RUDDER_YAW_SCALE=rudder_yaw,
                RUDDER_X_DRAG_SCALE=rudder_xdrag,
                LINEAR_SURGE_DAMP=base["LINEAR_SURGE_DAMP"],
                LINEAR_YAW_DAMP=yaw_damp,
                ROLL_DAMP_SCALE=roll_damp,
                BOW_THRUSTER_SCALE=1.0,
                rpm=base["straight_rpm"],
                rudder_deg=0.0,
                duration_s=STRAIGHT_DURATION,
                dt=DT,
            )
            sim_turn = simulate_turn_with_leadin(
                RPM_COMMAND_SCALE=base["RPM_COMMAND_SCALE"],
                PROPELLER_THRUST_SCALE=base["PROPELLER_THRUST_SCALE"],
                RUDDER_FORCE_SCALE=rudder_force,
                RUDDER_YAW_SCALE=rudder_yaw,
                RUDDER_X_DRAG_SCALE=rudder_xdrag,
                LINEAR_SURGE_DAMP=base["LINEAR_SURGE_DAMP"],
                LINEAR_YAW_DAMP=yaw_damp,
                ROLL_DAMP_SCALE=roll_damp,
                BOW_THRUSTER_SCALE=1.0,
                rpm=turn_rpm,
                rudder_deg=turn_rudder_deg,
                turn_leadin_s=TURN_LEADIN_DURATION,
                duration_s=TURN_DURATION,
                dt=DT,
            )
            mm = extract_motion_metrics(sim_straight)
            tm = extract_turn_metrics(sim_turn)
            sscore = score_section(mm, real_motion, MOTION_WEIGHTS)
            tscore = score_section(tm, real_turn_targets, TURN_WEIGHTS)
            joint_rows.append({
                "straight_rpm": base["straight_rpm"],
                "turn_rpm": turn_rpm,
                "turn_rudder_deg": turn_rudder_deg,
                "RPM_COMMAND_SCALE": base["RPM_COMMAND_SCALE"],
                "PROPELLER_THRUST_SCALE": base["PROPELLER_THRUST_SCALE"],
                "RUDDER_FORCE_SCALE": rudder_force,
                "RUDDER_YAW_SCALE": rudder_yaw,
                "RUDDER_X_DRAG_SCALE": rudder_xdrag,
                "LINEAR_SURGE_DAMP": base["LINEAR_SURGE_DAMP"],
                "LINEAR_YAW_DAMP": yaw_damp,
                "ROLL_DAMP_SCALE": roll_damp,
                "surge_score": sscore["score_total"],
                "turn_score": tscore["score_total"],
                "joint_score": sscore["score_total"] + tscore["score_total"],
                **{f"surge_err_{k}": v for k, v in sscore["parts"].items()},
                **{f"turn_err_{k}": v for k, v in tscore["parts"].items()},
                **{f"surge_{k}": v for k, v in mm.items()},
                **{f"turn_{k}": v for k, v in tm.items()},
            })

    joint_rows.sort(key=lambda r: r["joint_score"])
    save_json(out / "joint_4dof_top20.json", {"rows": joint_rows[:20]})

    best = joint_rows[0]
    best_straight = simulate_open_loop(
        RPM_COMMAND_SCALE=best["RPM_COMMAND_SCALE"],
        PROPELLER_THRUST_SCALE=best["PROPELLER_THRUST_SCALE"],
        RUDDER_FORCE_SCALE=best["RUDDER_FORCE_SCALE"],
        RUDDER_YAW_SCALE=best["RUDDER_YAW_SCALE"],
        RUDDER_X_DRAG_SCALE=best["RUDDER_X_DRAG_SCALE"],
        LINEAR_SURGE_DAMP=best["LINEAR_SURGE_DAMP"],
        LINEAR_YAW_DAMP=best["LINEAR_YAW_DAMP"],
        ROLL_DAMP_SCALE=best["ROLL_DAMP_SCALE"],
        BOW_THRUSTER_SCALE=1.0,
        rpm=best["straight_rpm"],
        rudder_deg=0.0,
        duration_s=STRAIGHT_DURATION,
        dt=DT,
    )
    best_turn = simulate_turn_with_leadin(
        RPM_COMMAND_SCALE=best["RPM_COMMAND_SCALE"],
        PROPELLER_THRUST_SCALE=best["PROPELLER_THRUST_SCALE"],
        RUDDER_FORCE_SCALE=best["RUDDER_FORCE_SCALE"],
        RUDDER_YAW_SCALE=best["RUDDER_YAW_SCALE"],
        RUDDER_X_DRAG_SCALE=best["RUDDER_X_DRAG_SCALE"],
        LINEAR_SURGE_DAMP=best["LINEAR_SURGE_DAMP"],
        LINEAR_YAW_DAMP=best["LINEAR_YAW_DAMP"],
        ROLL_DAMP_SCALE=best["ROLL_DAMP_SCALE"],
        BOW_THRUSTER_SCALE=1.0,
        rpm=best["turn_rpm"],
        rudder_deg=best["turn_rudder_deg"],
        turn_leadin_s=TURN_LEADIN_DURATION,
        duration_s=TURN_DURATION,
        dt=DT,
    )

    mm = extract_motion_metrics(best_straight)
    tm = extract_turn_metrics(best_turn)
    motion_comp = build_comparison("motion", MOTION_KEYS, mm, real_motion)
    turn_comp = build_comparison("turn", TURN_KEYS, tm, real_turn_targets)

    save_json(out / "best_4dof_joint_config.json", best)
    save_json(out / "best_4dof_motion_metrics.json", mm)
    save_json(out / "best_4dof_turn_metrics.json", tm)
    save_json(out / "best_4dof_motion_comparison.json", motion_comp)
    save_json(out / "best_4dof_turn_comparison.json", turn_comp)
    save_json(
        out / "best_4dof_vs_v2_summary.json",
        {
            "fourdof_joint_score": best["joint_score"],
            "fourdof_surge_score": best["surge_score"],
            "fourdof_turn_score": best["turn_score"],
            "v2_baseline_joint_score": V2_BASELINE_JOINT_SCORE,
            "beats_v2": bool(best["joint_score"] < V2_BASELINE_JOINT_SCORE),
            "delta_joint_score_vs_v2": best["joint_score"] - V2_BASELINE_JOINT_SCORE,
            "v2_baseline_config": V2_BASELINE_CONFIG,
        },
    )

    plot_open_loop_response(out / "best_4dof_straight_response.png", best_straight, f"4DOF straight: rpm={best['straight_rpm']}")
    plot_path(out / "best_4dof_straight_path.png", best_straight, "4DOF straight path")
    plot_open_loop_response(out / "best_4dof_turn_response.png", best_turn, f"4DOF turn: rpm={best['turn_rpm']}, rud={best['turn_rudder_deg']}")
    plot_path(out / "best_4dof_turn_path.png", best_turn, "4DOF turn path")

    print("saved:", out / "best_4dof_joint_config.json")
    print("saved:", out / "best_4dof_vs_v2_summary.json")
    print(f"best 4DOF joint score: {best['joint_score']:.6f}")
    print(f"v2 baseline joint score: {V2_BASELINE_JOINT_SCORE:.6f}")
    print(f"beats v2: {best['joint_score'] < V2_BASELINE_JOINT_SCORE}")


if __name__ == "__main__":
    main()
