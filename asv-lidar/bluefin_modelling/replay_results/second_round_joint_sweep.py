from __future__ import annotations

import argparse
import csv
import importlib
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
    "time_to_50pct_peak_u_after_motion_s": 1.0,
    "time_to_90pct_peak_u_after_motion_s": 1.0,
}

TURN_WEIGHTS = {
    "peak_abs_yaw_rate_degps": 2.0,
    "time_to_90deg_after_turn_s": 2.0,
    "time_to_180deg_after_turn_s": 1.5,
    "radius_first_90deg_m": 1.5,
    "radius_first_180deg_m": 1.0,
    "u_body_10s_after_turn_mps": 1.0,
}


def parse_grid(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def rel_error(sim: Optional[float], real: Optional[float], floor: float = 1e-6) -> float:
    if sim is None or real is None:
        return 10.0
    return abs(sim - real) / max(abs(real), floor)


def get_real_targets(data: Dict[str, Any], section: str, keys: List[str]) -> Dict[str, Optional[float]]:
    # Accept either *_comparison.json style or metrics style.
    if section in data and isinstance(data[section], dict):
        src = data[section]
        out: Dict[str, Optional[float]] = {}
        for k in keys:
            item = src.get(k, None)
            if isinstance(item, dict) and "real" in item:
                out[k] = item.get("real", None)
            else:
                out[k] = item
        return out

    alt_section = "straight_metrics" if section == "motion" else "turn_metrics"
    if alt_section in data and isinstance(data[alt_section], dict):
        src = data[alt_section]
        return {k: src.get(k, None) for k in keys}

    raise ValueError(f"Could not find section '{section}' or '{alt_section}' in {list(data.keys())}")


def score_section(sim_metrics: Dict[str, Any], real_targets: Dict[str, Optional[float]], weights: Dict[str, float]) -> Dict[str, Any]:
    parts: Dict[str, float] = {}
    total = 0.0
    for key, wt in weights.items():
        err = rel_error(sim_metrics.get(key, None), real_targets.get(key, None))
        parts[key] = err
        total += wt * err
    return {"score_total": total, "parts": parts}


def simulate_open_loop_with_overrides(
    *,
    model_module: str,
    duration_s: float,
    dt: float,
    rpm: float,
    rudder_deg: float,
    mass: float,
    thrust_coef: float,
    drag_coef: float,
    turn_coef: float,
    rudder_force_scale: float,
    linear_yaw_damp: float,
    linear_surge_damp: float,
) -> Dict[str, np.ndarray]:
    ship_model = importlib.import_module(model_module)

    if hasattr(ship_model, "MASS"):
        ship_model.MASS = float(mass)
    if hasattr(ship_model, "THRUST_COEF"):
        ship_model.THRUST_COEF = float(thrust_coef)
    if hasattr(ship_model, "DRAG_COEF"):
        ship_model.DRAG_COEF = float(drag_coef)
    if hasattr(ship_model, "TURN_COEF"):
        ship_model.TURN_COEF = float(turn_coef)
    if hasattr(ship_model, "RUDDER_FORCE_SCALE"):
        ship_model.RUDDER_FORCE_SCALE = float(rudder_force_scale)
    if hasattr(ship_model, "LINEAR_YAW_DAMP"):
        ship_model.LINEAR_YAW_DAMP = float(linear_yaw_damp)
    if hasattr(ship_model, "LINEAR_SURGE_DAMP"):
        ship_model.LINEAR_SURGE_DAMP = float(linear_surge_damp)
    if hasattr(ship_model, "MOMINERTIA") and hasattr(ship_model, "IZ") and hasattr(ship_model, "JZ"):
        # Keep explicit yaw inertia if the model defines it separately.
        ship_model.MOMINERTIA = float(getattr(ship_model, "IZ", 0.0) + getattr(ship_model, "JZ", 0.0))

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
        dx, dy, hdg_deg, yawrate_deg = model.update(rpm, rudder_percent, dt)
        xk += dx
        yk += dy
        x[i] = xk
        y[i] = yk
        heading_deg[i] = hdg_deg
        yaw_rate_degps[i] = yawrate_deg
        u_body[i] = float(getattr(model, "_v", 0.0))
        v_body[i] = float(getattr(model, "_v_sway", 0.0))

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


def build_motion_comparison(sim_metrics: Dict[str, Any], real_targets: Dict[str, Optional[float]]) -> Dict[str, Any]:
    return {
        "motion": {
            k: {
                "real": real_targets.get(k, None),
                "sim": sim_metrics.get(k, None),
                "rel_error": rel_error(sim_metrics.get(k, None), real_targets.get(k, None)),
            }
            for k in MOTION_KEYS
        }
    }


def build_turn_comparison(sim_metrics: Dict[str, Any], real_targets: Dict[str, Optional[float]]) -> Dict[str, Any]:
    return {
        "turn": {
            k: {
                "real": real_targets.get(k, None),
                "sim": sim_metrics.get(k, None),
                "rel_error": rel_error(sim_metrics.get(k, None), real_targets.get(k, None)),
            }
            for k in TURN_KEYS
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Second-round joint output-only sweep for the Bluefin MATLAB-style model")

    # No required args: everything has sensible defaults so it runs as
    #   python second_round_joint_sweep.py
    ap.add_argument("--speed-json", default="test_3_comparison.json", help="Real speed benchmark JSON")
    ap.add_argument("--turn-json", default="test_4_comparison.json", help="Real turn benchmark JSON")
    ap.add_argument("--out-dir", default="joint_round2_results", help="Output directory")
    ap.add_argument("--model-module", default="ship_model_bluefin_matlab_style")
    ap.add_argument("--mass", type=float, default=64.55)

    # Stage 1 focused surge search around the good region already found.
    ap.add_argument("--straight-rpm-grid", default="14,15,16")
    ap.add_argument("--thrust-grid", default="0.05,0.06,0.07")
    ap.add_argument("--drag-grid", default="0.75,1.0,1.25,1.5")
    ap.add_argument("--linear-surge-damp-grid", default="0.0,0.5,1.0,1.5,2.0")
    ap.add_argument("--stage1-turn-coef-fixed", type=float, default=3.0)
    ap.add_argument("--stage1-rudder-force-fixed", type=float, default=0.1)
    ap.add_argument("--stage1-yaw-damp-fixed", type=float, default=5.0)
    ap.add_argument("--top-surge-k", type=int, default=8)

    # Stage 2 turn refinement around the good turn region.
    ap.add_argument("--turn-rpm-grid", default="16,18,20")
    ap.add_argument("--turn-rudder-grid", default="25,30")
    ap.add_argument("--turn-coef-grid", default="3,4,5")
    ap.add_argument("--rudder-force-grid", default="0.1,0.15,0.2")
    ap.add_argument("--yaw-damp-grid", default="4,5,6")

    ap.add_argument("--straight-duration-s", type=float, default=40.0)
    ap.add_argument("--turn-duration-s", type=float, default=50.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--surge-weight", type=float, default=1.0, help="Weight of speed-test score in final joint score")
    ap.add_argument("--turn-weight", type=float, default=1.0, help="Weight of turn-test score in final joint score")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speed_data = load_json(Path(args.speed_json))
    turn_data = load_json(Path(args.turn_json))
    real_motion_targets = get_real_targets(speed_data, "motion", MOTION_KEYS)
    real_turn_targets = get_real_targets(turn_data, "turn", TURN_KEYS)

    straight_rpm_grid = parse_grid(args.straight_rpm_grid)
    thrust_grid = parse_grid(args.thrust_grid)
    drag_grid = parse_grid(args.drag_grid)
    linear_surge_damp_grid = parse_grid(args.linear_surge_damp_grid)

    turn_rpm_grid = parse_grid(args.turn_rpm_grid)
    turn_rudder_grid = parse_grid(args.turn_rudder_grid)
    turn_coef_grid = parse_grid(args.turn_coef_grid)
    rudder_force_grid = parse_grid(args.rudder_force_grid)
    yaw_damp_grid = parse_grid(args.yaw_damp_grid)

    # ------------------------------------------------------------------
    # Stage 1: focused surge sweep.
    # ------------------------------------------------------------------
    surge_rows: List[Dict[str, Any]] = []
    print("[1/2] Running focused surge sweep...")
    total_stage1 = len(straight_rpm_grid) * len(thrust_grid) * len(drag_grid) * len(linear_surge_damp_grid)
    done_stage1 = 0
    for rpm in straight_rpm_grid:
        for thrust in thrust_grid:
            for drag in drag_grid:
                for linear_surge_damp in linear_surge_damp_grid:
                    sim = simulate_open_loop_with_overrides(
                        model_module=args.model_module,
                        duration_s=args.straight_duration_s,
                        dt=args.dt,
                        rpm=rpm,
                        rudder_deg=0.0,
                        mass=args.mass,
                        thrust_coef=thrust,
                        drag_coef=drag,
                        turn_coef=args.stage1_turn_coef_fixed,
                        rudder_force_scale=args.stage1_rudder_force_fixed,
                        linear_yaw_damp=args.stage1_yaw_damp_fixed,
                        linear_surge_damp=linear_surge_damp,
                    )
                    motion = extract_motion_metrics(sim)
                    scored = score_section(motion, real_motion_targets, MOTION_WEIGHTS)

                    row: Dict[str, Any] = {
                        "straight_rpm": rpm,
                        "MASS": args.mass,
                        "THRUST_COEF": thrust,
                        "DRAG_COEF": drag,
                        "LINEAR_SURGE_DAMP": linear_surge_damp,
                        "TURN_COEF": args.stage1_turn_coef_fixed,
                        "RUDDER_FORCE_SCALE": args.stage1_rudder_force_fixed,
                        "LINEAR_YAW_DAMP": args.stage1_yaw_damp_fixed,
                        "surge_score": scored["score_total"],
                    }
                    row.update({f"surge_err_{k}": v for k, v in scored["parts"].items()})
                    row.update({f"surge_{k}": v for k, v in motion.items()})
                    surge_rows.append(row)
                    done_stage1 += 1
                    if done_stage1 % 50 == 0 or done_stage1 == total_stage1:
                        print(f"  stage1 progress: {done_stage1}/{total_stage1}")

    surge_rows.sort(key=lambda r: r["surge_score"])
    save_json(out_dir / "stage1_surge_top20.json", {"rows": surge_rows[:20]})
    with (out_dir / "stage1_surge_ranked.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(surge_rows[0].keys()))
        writer.writeheader()
        writer.writerows(surge_rows)

    top_surge = surge_rows[: max(1, args.top_surge_k)]

    # ------------------------------------------------------------------
    # Stage 2: turn refinement for each top surge candidate.
    # ------------------------------------------------------------------
    joint_rows: List[Dict[str, Any]] = []
    best_turn_sim = None
    best_straight_sim = None
    best_motion = None
    best_turn = None
    best_row = None

    total_stage2 = len(top_surge) * len(turn_rpm_grid) * len(turn_rudder_grid) * len(turn_coef_grid) * len(rudder_force_grid) * len(yaw_damp_grid)
    done_stage2 = 0
    print("[2/2] Running joint turn refinement...")

    for base in top_surge:
        for turn_rpm in turn_rpm_grid:
            for rudder_deg in turn_rudder_grid:
                for turn_coef in turn_coef_grid:
                    for rudder_force_scale in rudder_force_grid:
                        for linear_yaw_damp in yaw_damp_grid:
                            straight_sim = simulate_open_loop_with_overrides(
                                model_module=args.model_module,
                                duration_s=args.straight_duration_s,
                                dt=args.dt,
                                rpm=float(base["straight_rpm"]),
                                rudder_deg=0.0,
                                mass=args.mass,
                                thrust_coef=float(base["THRUST_COEF"]),
                                drag_coef=float(base["DRAG_COEF"]),
                                turn_coef=float(turn_coef),
                                rudder_force_scale=float(rudder_force_scale),
                                linear_yaw_damp=float(linear_yaw_damp),
                                linear_surge_damp=float(base["LINEAR_SURGE_DAMP"]),
                            )
                            motion = extract_motion_metrics(straight_sim)
                            surge_scored = score_section(motion, real_motion_targets, MOTION_WEIGHTS)

                            turn_sim = simulate_open_loop_with_overrides(
                                model_module=args.model_module,
                                duration_s=args.turn_duration_s,
                                dt=args.dt,
                                rpm=float(turn_rpm),
                                rudder_deg=float(rudder_deg),
                                mass=args.mass,
                                thrust_coef=float(base["THRUST_COEF"]),
                                drag_coef=float(base["DRAG_COEF"]),
                                turn_coef=float(turn_coef),
                                rudder_force_scale=float(rudder_force_scale),
                                linear_yaw_damp=float(linear_yaw_damp),
                                linear_surge_damp=float(base["LINEAR_SURGE_DAMP"]),
                            )
                            turn = extract_turn_metrics(turn_sim)
                            turn_scored = score_section(turn, real_turn_targets, TURN_WEIGHTS)

                            total_score = args.surge_weight * surge_scored["score_total"] + args.turn_weight * turn_scored["score_total"]

                            row: Dict[str, Any] = {
                                "MASS": args.mass,
                                "straight_rpm": float(base["straight_rpm"]),
                                "turn_rpm": float(turn_rpm),
                                "turn_rudder_deg": float(rudder_deg),
                                "THRUST_COEF": float(base["THRUST_COEF"]),
                                "DRAG_COEF": float(base["DRAG_COEF"]),
                                "LINEAR_SURGE_DAMP": float(base["LINEAR_SURGE_DAMP"]),
                                "TURN_COEF": float(turn_coef),
                                "RUDDER_FORCE_SCALE": float(rudder_force_scale),
                                "LINEAR_YAW_DAMP": float(linear_yaw_damp),
                                "surge_score": surge_scored["score_total"],
                                "turn_score": turn_scored["score_total"],
                                "joint_score": total_score,
                            }
                            row.update({f"surge_err_{k}": v for k, v in surge_scored["parts"].items()})
                            row.update({f"turn_err_{k}": v for k, v in turn_scored["parts"].items()})
                            row.update({f"surge_{k}": v for k, v in motion.items()})
                            row.update({f"turn_{k}": v for k, v in turn.items()})
                            joint_rows.append(row)

                            if best_row is None or row["joint_score"] < best_row["joint_score"]:
                                best_row = row
                                best_straight_sim = straight_sim
                                best_turn_sim = turn_sim
                                best_motion = motion
                                best_turn = turn

                            done_stage2 += 1
                            if done_stage2 % 100 == 0 or done_stage2 == total_stage2:
                                print(f"  stage2 progress: {done_stage2}/{total_stage2}")

    joint_rows.sort(key=lambda r: r["joint_score"])
    save_json(out_dir / "joint_sweep_top20.json", {"rows": joint_rows[:20]})
    with (out_dir / "joint_sweep_ranked.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(joint_rows[0].keys()))
        writer.writeheader()
        writer.writerows(joint_rows)

    if best_row is not None and best_motion is not None and best_turn is not None and best_straight_sim is not None and best_turn_sim is not None:
        save_json(out_dir / "best_joint_config.json", best_row)
        save_json(out_dir / "best_joint_motion_metrics.json", {"motion_metrics": best_motion})
        save_json(out_dir / "best_joint_turn_metrics.json", {"turn_metrics": best_turn})
        save_json(out_dir / "best_joint_motion_comparison.json", build_motion_comparison(best_motion, real_motion_targets))
        save_json(out_dir / "best_joint_turn_comparison.json", build_turn_comparison(best_turn, real_turn_targets))
        save_json(
            out_dir / "best_joint_comparison.json",
            {
                **build_motion_comparison(best_motion, real_motion_targets),
                **build_turn_comparison(best_turn, real_turn_targets),
            },
        )

        plot_open_loop_response(
            out_dir / "best_joint_straight_response.png",
            best_straight_sim,
            title=(
                f"best joint straight: rpm={best_row['straight_rpm']}, T={best_row['THRUST_COEF']}, "
                f"D={best_row['DRAG_COEF']}, LSD={best_row['LINEAR_SURGE_DAMP']}"
            ),
        )
        plot_path(out_dir / "best_joint_straight_path.png", best_straight_sim, title="best joint straight path")
        plot_open_loop_response(
            out_dir / "best_joint_turn_response.png",
            best_turn_sim,
            title=(
                f"best joint turn: rpm={best_row['turn_rpm']}, rud={best_row['turn_rudder_deg']}, "
                f"TURN={best_row['TURN_COEF']}, RF={best_row['RUDDER_FORCE_SCALE']}, YD={best_row['LINEAR_YAW_DAMP']}"
            ),
        )
        plot_path(out_dir / "best_joint_turn_path.png", best_turn_sim, title="best joint turn path")

    print(f"Saved second-round joint sweep to {out_dir}")
    if best_row is not None:
        print("Best joint candidate:")
        print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
