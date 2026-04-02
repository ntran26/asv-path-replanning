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


TURN_KEYS = [
    "peak_abs_yaw_rate_degps",
    "time_to_90deg_after_turn_s",
    "time_to_180deg_after_turn_s",
    "radius_first_90deg_m",
    "radius_first_180deg_m",
    "u_body_10s_after_turn_mps",
]

WEIGHTS = {
    "peak_abs_yaw_rate_degps": 2.0,
    "time_to_90deg_after_turn_s": 2.0,
    "time_to_180deg_after_turn_s": 1.5,
    "radius_first_90deg_m": 1.5,
    "radius_first_180deg_m": 1.0,
    "u_body_10s_after_turn_mps": 1.0,
}


def parse_grid(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def get_real_turn_targets(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    if "turn_metrics" in data:
        src = data["turn_metrics"]
        return {k: src.get(k, None) for k in TURN_KEYS}
    if "turn" in data:
        out: Dict[str, Optional[float]] = {}
        for k in TURN_KEYS:
            item = data["turn"].get(k, None)
            out[k] = None if item is None else item.get("real", None)
        return out
    raise ValueError("Could not find real turn metrics in the supplied JSON file.")


def rel_error(sim: Optional[float], real: Optional[float], floor: float = 1e-6) -> float:
    if sim is None or real is None:
        return 10.0
    return abs(sim - real) / max(abs(real), floor)


def score_turn(sim_metrics: Dict[str, Any], real_targets: Dict[str, Optional[float]]) -> Dict[str, Any]:
    parts: Dict[str, float] = {}
    total = 0.0
    for key in TURN_KEYS:
        err = rel_error(sim_metrics.get(key, None), real_targets.get(key, None))
        parts[key] = err
        total += WEIGHTS[key] * err
    return {"score_total": total, "parts": parts}


def simulate_open_loop_turn_with_overrides(
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Output-only turning sweep for the MATLAB-style Bluefin model")
    ap.add_argument("--real-json", required=True, help="Real turn benchmark JSON (test_4_metrics.json or test_4_comparison.json)")
    ap.add_argument("--out-dir", default="turn_sweep_results", help="Output directory")
    ap.add_argument("--model-module", default="ship_model_bluefin_matlab_style")
    ap.add_argument("--mass", type=float, default=64.55)
    ap.add_argument("--thrust-coef", type=float, default=0.04)
    ap.add_argument("--drag-coef", type=float, default=1.0)
    ap.add_argument("--rpm-grid", default="12.7,14.0,16.0,18.0")
    ap.add_argument("--rudder-grid", default="25,30,35,40")
    ap.add_argument("--turn-coef-grid", default="1.0,1.5,2.0,3.0,4.0,5.0")
    ap.add_argument("--rudder-force-grid", default="0.1,0.2,0.3,0.5,0.7,1.0")
    ap.add_argument("--yaw-damp-grid", default="1.0,2.0,5.0,10.0")
    ap.add_argument("--duration-s", type=float, default=50.0)
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_data = load_json(Path(args.real_json))
    real_targets = get_real_turn_targets(real_data)

    rpm_grid = parse_grid(args.rpm_grid)
    rudder_grid = parse_grid(args.rudder_grid)
    turn_coef_grid = parse_grid(args.turn_coef_grid)
    rudder_force_grid = parse_grid(args.rudder_force_grid)
    yaw_damp_grid = parse_grid(args.yaw_damp_grid)

    rows: List[Dict[str, Any]] = []
    best_sim = None
    best_turn = None
    best_motion = None
    best_row = None

    for rpm in rpm_grid:
        for rudder_deg in rudder_grid:
            for turn_coef in turn_coef_grid:
                for rudder_force_scale in rudder_force_grid:
                    for linear_yaw_damp in yaw_damp_grid:
                        sim = simulate_open_loop_turn_with_overrides(
                            model_module=args.model_module,
                            duration_s=args.duration_s,
                            dt=args.dt,
                            rpm=rpm,
                            rudder_deg=rudder_deg,
                            mass=args.mass,
                            thrust_coef=args.thrust_coef,
                            drag_coef=args.drag_coef,
                            turn_coef=turn_coef,
                            rudder_force_scale=rudder_force_scale,
                            linear_yaw_damp=linear_yaw_damp,
                        )
                        motion = extract_motion_metrics(sim)
                        turn = extract_turn_metrics(sim)
                        scored = score_turn(turn, real_targets)

                        row: Dict[str, Any] = {
                            "rpm": rpm,
                            "rudder_deg": rudder_deg,
                            "MASS": args.mass,
                            "THRUST_COEF": args.thrust_coef,
                            "DRAG_COEF": args.drag_coef,
                            "TURN_COEF": turn_coef,
                            "RUDDER_FORCE_SCALE": rudder_force_scale,
                            "LINEAR_YAW_DAMP": linear_yaw_damp,
                            "score_total": scored["score_total"],
                        }
                        row.update({f"err_{k}": v for k, v in scored["parts"].items()})
                        row.update({f"motion_{k}": v for k, v in motion.items()})
                        row.update(turn)
                        rows.append(row)

                        if best_row is None or row["score_total"] < best_row["score_total"]:
                            best_row = row
                            best_sim = sim
                            best_turn = turn
                            best_motion = motion

    rows.sort(key=lambda r: r["score_total"])

    csv_path = out_dir / "turn_sweep_ranked.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top_json_path = out_dir / "turn_sweep_top20.json"
    save_json(top_json_path, {"rows": rows[:20]})

    if best_sim is not None and best_turn is not None and best_row is not None and best_motion is not None:
        save_json(out_dir / "best_turn_metrics.json", {"motion_metrics": best_motion, "turn_metrics": best_turn})

        comparison = {
            "turn": {
                k: {
                    "real": real_targets.get(k, None),
                    "sim": best_turn.get(k, None),
                    "rel_error": rel_error(best_turn.get(k, None), real_targets.get(k, None)),
                }
                for k in TURN_KEYS
            }
        }
        save_json(out_dir / "best_turn_comparison.json", comparison)
        save_json(out_dir / "best_turn_config.json", best_row)
        plot_open_loop_response(out_dir / "best_turn_response.png", best_sim, title=f"best turn: rpm={best_row['rpm']}, rud={best_row['rudder_deg']}, TURN={best_row['TURN_COEF']}, RF={best_row['RUDDER_FORCE_SCALE']}, YD={best_row['LINEAR_YAW_DAMP']}")
        plot_path(out_dir / "best_turn_path.png", best_sim, title="best turn path")

    print(f"Saved sweep to {out_dir}")
    if best_row is not None:
        print("Best candidate:")
        print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
