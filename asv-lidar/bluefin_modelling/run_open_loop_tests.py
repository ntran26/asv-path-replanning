"""
python run_open_loop_tests.py --out-dir open_loop_results --straight-rpm 12.7 --turn-rpm 12.7 --turn-rudder-deg 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bluefin_test_utils import (
    extract_all_metrics,
    plot_open_loop_response,
    plot_path,
    save_json,
    simulate_open_loop,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="bluefin_open_loop_tests")
    ap.add_argument("--model-module", default="ship_model_bluefin")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--straight-duration", type=float, default=40.0)
    ap.add_argument("--turn-duration", type=float, default=50.0)
    ap.add_argument("--straight-rpm", type=float, default=12.7)
    ap.add_argument("--turn-rpm", type=float, default=12.7)
    ap.add_argument("--turn-rudder-deg", type=float, default=30.0)
    ap.add_argument("--mass", type=float, default=None)
    ap.add_argument("--thrust-coef", type=float, default=None)
    ap.add_argument("--drag-coef", type=float, default=None)
    ap.add_argument("--turn-coef", type=float, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    straight = simulate_open_loop(
        duration_s=args.straight_duration,
        dt=args.dt,
        rpm=args.straight_rpm,
        rudder_deg=0.0,
        model_module=args.model_module,
        mass=args.mass,
        thrust_coef=args.thrust_coef,
        drag_coef=args.drag_coef,
        turn_coef=args.turn_coef,
    )
    straight_metrics = extract_all_metrics(straight)
    save_json(out_dir / "straight_metrics.json", straight_metrics)
    plot_open_loop_response(out_dir / "straight_response.png", straight, f"straight: rpm={args.straight_rpm}")
    plot_path(out_dir / "straight_path.png", straight, "straight path")

    turn = simulate_open_loop(
        duration_s=args.turn_duration,
        dt=args.dt,
        rpm=args.turn_rpm,
        rudder_deg=args.turn_rudder_deg,
        model_module=args.model_module,
        mass=args.mass,
        thrust_coef=args.thrust_coef,
        drag_coef=args.drag_coef,
        turn_coef=args.turn_coef,
    )
    turn_metrics = extract_all_metrics(turn)
    save_json(out_dir / "turn_metrics.json", turn_metrics)
    plot_open_loop_response(out_dir / "turn_response.png", turn, f"turn: rpm={args.turn_rpm}, rud={args.turn_rudder_deg}")
    plot_path(out_dir / "turn_path.png", turn, "turn path")

    print(f"saved: {out_dir / 'straight_metrics.json'}")
    print(f"saved: {out_dir / 'straight_response.png'}")
    print(f"saved: {out_dir / 'straight_path.png'}")
    print(f"saved: {out_dir / 'turn_metrics.json'}")
    print(f"saved: {out_dir / 'turn_response.png'}")
    print(f"saved: {out_dir / 'turn_path.png'}")

if __name__ == "__main__":
    main()
