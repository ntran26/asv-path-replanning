"""Stage-1 motion sweep for the *faithful* 4DOF Bluefin model.

This version only sweeps parameters that actually exist in the faithful
`ship_model_bluefin_4dof.py` port.

Run with no arguments:

    python sweep_bluefin_4dof_motion_faithful.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

from bluefin_4dof_utils import (
    ROOT,
    compare_metrics,
    extract_motion_metrics,
    load_default_real_benchmarks,
    run_open_loop,
    safe_rel_error,
)

OUT_DIR = ROOT / "bluefin_4dof_motion_sweep_faithful"
MODULE_NAME = "ship_model_bluefin_4dof"

# Narrowed around the currently promising region.
RPM_GRID = [15.0, 16.0, 17.0]
RPM_INPUT_TO_SOLVER_RPM_GRID = [85.0, 90.0, 95.0]
PROPELLER_THRUST_SCALE_GRID = [1.3, 1.4, 1.5, 1.6, 1.7]

DT = 0.1
DURATION_S = 40.0

WEIGHTS = {
    "peak_u_body_mps": 2.0,
    "initial_accel_0_2_after_motion_mps2": 2.0,
    "initial_accel_0_5_after_motion_mps2": 1.5,
    "distance_at_10s_after_motion_m": 1.5,
    "time_to_50pct_peak_u_after_motion_s": 1.0,
    "time_to_90pct_peak_u_after_motion_s": 1.0,
}


def score_motion(sim_metrics: Dict[str, Any], real_metrics: Dict[str, Any]) -> float:
    total = 0.0
    for k, w in WEIGHTS.items():
        total += w * safe_rel_error(sim_metrics.get(k), real_metrics["straight_metrics"].get(k))
    return float(total)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real_motion, _ = load_default_real_benchmarks()
    rows: List[Dict[str, Any]] = []

    for rpm, rpm_scale, prop_scale in itertools.product(
        RPM_GRID,
        RPM_INPUT_TO_SOLVER_RPM_GRID,
        PROPELLER_THRUST_SCALE_GRID,
    ):
        params = {
            "RPM_INPUT_TO_SOLVER_RPM": rpm_scale,
            "PROPELLER_THRUST_SCALE": prop_scale,
            # faithful defaults for all other active scales
            "RUDDER_FORCE_SCALE": 1.0,
            "BOW_THRUSTER_SCALE": 1.0,
            "ROLL_DAMP_SCALE": 1.0,
            "ROLL_RESTORE_SCALE": 1.0,
        }
        sim = run_open_loop(
            module_name=MODULE_NAME,
            rpm=rpm,
            rudder_deg=0.0,
            duration_s=DURATION_S,
            dt=DT,
            params=params,
        )
        metrics = extract_motion_metrics(sim)
        score = score_motion(metrics, real_motion)
        row = {
            "rpm": rpm,
            "RPM_INPUT_TO_SOLVER_RPM": rpm_scale,
            "PROPELLER_THRUST_SCALE": prop_scale,
            "score_total": score,
            **metrics,
        }
        rows.append(row)

    rows.sort(key=lambda r: r["score_total"])
    best = rows[0]
    best_params = {
        "RPM_INPUT_TO_SOLVER_RPM": best["RPM_INPUT_TO_SOLVER_RPM"],
        "PROPELLER_THRUST_SCALE": best["PROPELLER_THRUST_SCALE"],
        "RUDDER_FORCE_SCALE": 1.0,
        "BOW_THRUSTER_SCALE": 1.0,
        "ROLL_DAMP_SCALE": 1.0,
        "ROLL_RESTORE_SCALE": 1.0,
    }
    best_sim = run_open_loop(
        module_name=MODULE_NAME,
        rpm=best["rpm"],
        rudder_deg=0.0,
        duration_s=DURATION_S,
        dt=DT,
        params=best_params,
    )
    best_metrics = extract_motion_metrics(best_sim)
    best_comparison = {
        "motion": compare_metrics(best_metrics, real_motion["straight_metrics"], WEIGHTS.keys())
    }

    with (OUT_DIR / "best_4dof_motion_config.json").open("w", encoding="utf-8") as f:
        json.dump({"rpm": best["rpm"], **best_params, "score_total": best["score_total"]}, f, indent=2)
    with (OUT_DIR / "best_4dof_motion_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"motion_metrics": best_metrics}, f, indent=2)
    with (OUT_DIR / "best_4dof_motion_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(best_comparison, f, indent=2)
    with (OUT_DIR / "stage1_4dof_motion_top20.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows[:20]}, f, indent=2)

    print("Saved motion sweep outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
