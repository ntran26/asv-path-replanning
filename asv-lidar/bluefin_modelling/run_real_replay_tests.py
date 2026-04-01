"""
python run_real_replay_tests.py --test3-json test_3_metrics.json --test4-json test_4_metrics.json --out-dir replay_results --rpm-max 12.7 --max-rudder-deg 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bluefin_test_utils import (
    build_comparison_report,
    build_replay_series,
    extract_all_metrics,
    infer_mapping,
    load_json,
    plot_path,
    plot_replay_debug,
    save_json,
    simulate_replay,
)

def process_one(
    real_json_path: Path,
    out_dir: Path,
    model_module: str,
    max_rudder_deg: float,
    rpm_max: float,
    mass,
    thrust_coef,
    drag_coef,
    turn_coef,
    tag: str,
) -> None:
    real_data = load_json(real_json_path)
    real_series = build_replay_series(real_data)

    mapping = infer_mapping(
        real_data,
        real_series,
        max_rudder_deg=max_rudder_deg,
        rpm_max=rpm_max,
    )

    sim = simulate_replay(
        real_series,
        mapping,
        model_module=model_module,
        mass=mass,
        thrust_coef=thrust_coef,
        drag_coef=drag_coef,
        turn_coef=turn_coef,
    )

    sim_metrics = extract_all_metrics(sim)
    comparison = build_comparison_report(real_data, sim_metrics)

    debug_json = {
        "mapping": {
            "s1_neutral": mapping.s1_neutral,
            "s1_scale": mapping.s1_scale,
            "s2_neutral": mapping.s2_neutral,
            "s2_full_fwd": mapping.s2_full_fwd,
            "max_rudder_deg": mapping.max_rudder_deg,
            "rpm_max": mapping.rpm_max,
        },
        "sim_metrics": sim_metrics,
        "comparison_to_real": comparison,
        "series": {
            "t_sec": sim["t_sec"].tolist(),
            "s1_raw": real_series.s1.tolist(),
            "s2_raw": real_series.s2.tolist(),
            "rudder_deg_cmd": sim["rudder_deg_cmd"].tolist(),
            "rudder_percent_cmd": sim["rudder_percent_cmd"].tolist(),
            "rpm_cmd": sim["rpm_cmd"].tolist(),
            "u_body_mps": sim["u_body_mps"].tolist(),
            "v_body_mps": sim["v_body_mps"].tolist(),
            "yaw_rate_degps": sim["yaw_rate_degps"].tolist(),
            "heading_deg": sim["heading_deg"].tolist(),
            "x_m": sim["x_m"].tolist(),
            "y_m": sim["y_m"].tolist(),
        },
    }

    save_json(out_dir / f"{tag}_sim_metrics.json", sim_metrics)
    save_json(out_dir / f"{tag}_comparison.json", comparison)
    save_json(out_dir / f"{tag}_debug.json", debug_json)

    plot_replay_debug(out_dir / f"{tag}_replay_debug.png", real_series, sim, f"{tag}: replayed commands and response")
    plot_path(out_dir / f"{tag}_path.png", sim, f"{tag}: simulated path")

    print(f"\n[{tag}]")
    print(f"mapping: s1_neutral={mapping.s1_neutral:.2f}, s1_scale={mapping.s1_scale:.2f}, "
          f"s2_neutral={mapping.s2_neutral:.2f}, s2_full_fwd={mapping.s2_full_fwd:.2f}")
    print(f"saved: {out_dir / f'{tag}_sim_metrics.json'}")
    print(f"saved: {out_dir / f'{tag}_comparison.json'}")
    print(f"saved: {out_dir / f'{tag}_debug.json'}")
    print(f"saved: {out_dir / f'{tag}_replay_debug.png'}")
    print(f"saved: {out_dir / f'{tag}_path.png'}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test3-json", default="test_3_metrics.json")
    ap.add_argument("--test4-json", default="test_4_metrics.json")
    ap.add_argument("--out-dir", default="bluefin_replay_validation")
    ap.add_argument("--model-module", default="ship_model_bluefin")
    ap.add_argument("--max-rudder-deg", type=float, default=30.0)
    ap.add_argument("--rpm-max", type=float, default=12.7)
    ap.add_argument("--mass", type=float, default=None)
    ap.add_argument("--thrust-coef", type=float, default=None)
    ap.add_argument("--drag-coef", type=float, default=None)
    ap.add_argument("--turn-coef", type=float, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    process_one(
        Path(args.test3_json), out_dir, args.model_module,
        args.max_rudder_deg, args.rpm_max,
        args.mass, args.thrust_coef, args.drag_coef, args.turn_coef,
        "test_3",
    )
    process_one(
        Path(args.test4_json), out_dir, args.model_module,
        args.max_rudder_deg, args.rpm_max,
        args.mass, args.thrust_coef, args.drag_coef, args.turn_coef,
        "test_4",
    )

if __name__ == "__main__":
    main()
