"""
python sweep_surge_output_only.py --real-json test_3_comparison.json --out-dir surge_sweep_results --rpm-grid 12.7,14.0,15.0,16.0,18.0 --thrust-grid 0.04,0.05,0.06,0.07,0.08,0.09 --drag-grid 0.5,0.75,1.0,1.25,1.5

python sweep_turn_output_only.py --real-json test_4_comparison.json --out-dir turn_sweep_results --thrust-coef 0.07 --drag-coef 0.75 --rpm-grid 12.7,14.0,16.0,18.0 --rudder-grid 25,30,35,40 --turn-coef-grid 1.0,1.5,2.0,3.0,4.0,5.0 --rudder-force-grid 0.1,0.2,0.3,0.5,0.7,1.0 --yaw-damp-grid 1.0,2.0,5.0,10.0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from bluefin_test_utils import (
    build_comparison_report,
    extract_motion_metrics,
    load_json,
    plot_open_loop_response,
    plot_path,
    save_json,
    simulate_open_loop,
)

MOTION_KEYS = [
    "peak_u_body_mps",
    "initial_accel_0_2_after_motion_mps2",
    "initial_accel_0_5_after_motion_mps2",
    "distance_at_10s_after_motion_m",
    "time_to_50pct_peak_u_after_motion_s",
    "time_to_90pct_peak_u_after_motion_s",
]

WEIGHTS = {
    "peak_u_body_mps": 2.0,
    "initial_accel_0_2_after_motion_mps2": 2.0,
    "initial_accel_0_5_after_motion_mps2": 1.5,
    "distance_at_10s_after_motion_m": 2.0,
    "time_to_50pct_peak_u_after_motion_s": 1.0,
    "time_to_90pct_peak_u_after_motion_s": 1.0,
}


def parse_grid(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def get_real_motion_targets(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    if "straight_metrics" in data:
        src = data["straight_metrics"]
        return {k: src.get(k, None) for k in MOTION_KEYS}
    if "motion" in data:
        out: Dict[str, Optional[float]] = {}
        for k in MOTION_KEYS:
            item = data["motion"].get(k, None)
            out[k] = None if item is None else item.get("real", None)
        return out
    raise ValueError("Could not find real motion metrics in the supplied JSON file.")


def rel_error(sim: Optional[float], real: Optional[float], floor: float = 1e-6) -> float:
    if sim is None or real is None:
        return 10.0
    return abs(sim - real) / max(abs(real), floor)


def score_motion(sim_metrics: Dict[str, Any], real_targets: Dict[str, Optional[float]]) -> Dict[str, Any]:
    parts: Dict[str, float] = {}
    total = 0.0
    for key in MOTION_KEYS:
        err = rel_error(sim_metrics.get(key, None), real_targets.get(key, None))
        parts[key] = err
        total += WEIGHTS[key] * err
    return {"score_total": total, "parts": parts}


def main() -> None:
    ap = argparse.ArgumentParser(description="Output-only surge sweep for the MATLAB-style Bluefin model")
    ap.add_argument("--real-json", required=True, help="Real speed benchmark JSON (test_3_metrics.json or test_3_comparison.json)")
    ap.add_argument("--out-dir", default="surge_sweep_results", help="Output directory")
    ap.add_argument("--model-module", default="ship_model_bluefin_matlab_style")
    ap.add_argument("--mass", type=float, default=64.55)
    ap.add_argument("--turn-coef", type=float, default=1.0, help="Fixed TURN_COEF during straight sweep")
    ap.add_argument("--rpm-grid", default="12.7,14.0,15.0,16.0,18.0")
    ap.add_argument("--thrust-grid", default="0.04,0.05,0.06,0.07,0.08,0.09")
    ap.add_argument("--drag-grid", default="0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--duration-s", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_data = load_json(Path(args.real_json))
    real_targets = get_real_motion_targets(real_data)

    rpm_grid = parse_grid(args.rpm_grid)
    thrust_grid = parse_grid(args.thrust_grid)
    drag_grid = parse_grid(args.drag_grid)

    rows: List[Dict[str, Any]] = []
    best_sim = None
    best_motion = None
    best_row = None

    for rpm in rpm_grid:
        for thrust in thrust_grid:
            for drag in drag_grid:
                sim = simulate_open_loop(
                    duration_s=args.duration_s,
                    dt=args.dt,
                    rpm=rpm,
                    rudder_deg=0.0,
                    model_module=args.model_module,
                    mass=args.mass,
                    thrust_coef=thrust,
                    drag_coef=drag,
                    turn_coef=args.turn_coef,
                )
                motion = extract_motion_metrics(sim)
                scored = score_motion(motion, real_targets)

                row: Dict[str, Any] = {
                    "rpm": rpm,
                    "THRUST_COEF": thrust,
                    "DRAG_COEF": drag,
                    "TURN_COEF": args.turn_coef,
                    "MASS": args.mass,
                    "score_total": scored["score_total"],
                }
                row.update({f"err_{k}": v for k, v in scored["parts"].items()})
                row.update(motion)
                rows.append(row)

                if best_row is None or row["score_total"] < best_row["score_total"]:
                    best_row = row
                    best_sim = sim
                    best_motion = motion

    rows.sort(key=lambda r: r["score_total"])

    # Save full ranking CSV
    csv_path = out_dir / "surge_sweep_ranked.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Save top 20 JSON
    top_json_path = out_dir / "surge_sweep_top20.json"
    save_json(top_json_path, {"rows": rows[:20]})

    if best_sim is not None and best_motion is not None and best_row is not None:
        best_metrics_path = out_dir / "best_surge_metrics.json"
        save_json(best_metrics_path, {"motion_metrics": best_motion})

        # Build a comparison report in the same style as previous tooling
        comparison = {
            "motion": {
                k: {
                    "real": real_targets.get(k, None),
                    "sim": best_motion.get(k, None),
                    "rel_error": rel_error(best_motion.get(k, None), real_targets.get(k, None)),
                }
                for k in MOTION_KEYS
            }
        }
        save_json(out_dir / "best_surge_comparison.json", comparison)
        save_json(out_dir / "best_surge_config.json", best_row)

        plot_open_loop_response(out_dir / "best_surge_response.png", best_sim, title=f"best surge: rpm={best_row['rpm']}, T={best_row['THRUST_COEF']}, D={best_row['DRAG_COEF']}")
        plot_path(out_dir / "best_surge_path.png", best_sim, title="best surge path")

    print(f"Saved sweep to {out_dir}")
    if best_row is not None:
        print("Best candidate:")
        print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
