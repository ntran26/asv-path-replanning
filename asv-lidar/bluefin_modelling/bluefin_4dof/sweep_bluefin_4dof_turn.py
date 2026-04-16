"""Stage-2 turning sweep for the faithful 4DOF Bluefin model.

Focus: turning-circle behavior. If a motion sweep result exists, the best motion
settings are loaded automatically and kept fixed while turning-related
parameters are varied.

Run with no arguments:

    python sweep_bluefin_4dof_turn.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

from bluefin_4dof_utils import (
    ROOT,
    compare_metrics,
    extract_turn_metrics,
    load_default_real_benchmarks,
    run_open_loop,
    safe_rel_error,
)

OUT_DIR = ROOT / "bluefin_4dof_turn_sweep"
MOTION_SWEEP_DIR = ROOT / "bluefin_4dof_motion_sweep"

TURN_RPM_GRID = [18.0, 20.0, 22.0, 24.0]
TURN_RUDDER_DEG_GRID = [25.0, 30.0, 35.0, 40.0]
RUDDER_FORCE_SCALE_GRID = [0.8, 1.0, 1.2, 1.5]
ROLL_DAMP_SCALE_GRID = [0.8, 1.0, 1.2, 1.5]
ROLL_RESTORE_SCALE_GRID = [0.8, 1.0, 1.2]

DT = 0.1
DURATION_S = 50.0
WARMUP_S = 1.0

WEIGHTS = {
    "peak_abs_yaw_rate_degps": 2.0,
    "time_to_90deg_after_turn_s": 2.0,
    "time_to_180deg_after_turn_s": 2.0,
    "radius_first_90deg_m": 1.5,
    "radius_first_180deg_m": 1.0,
    "u_body_10s_after_turn_mps": 1.0,
}


def load_motion_best_params() -> Dict[str, float]:
    path = MOTION_SWEEP_DIR / "best_4dof_motion_config.json"
    if not path.exists():
        return {
            "RPM_INPUT_TO_SOLVER_RPM": 66.6666666667,
            "PROPELLER_THRUST_SCALE": 1.0,
            "RUDDER_FORCE_SCALE": 1.0,
            "BOW_THRUSTER_SCALE": 1.0,
            "ROLL_DAMP_SCALE": 1.0,
            "ROLL_RESTORE_SCALE": 1.0,
        }
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "RPM_INPUT_TO_SOLVER_RPM": data.get("RPM_INPUT_TO_SOLVER_RPM", 66.6666666667),
        "PROPELLER_THRUST_SCALE": data.get("PROPELLER_THRUST_SCALE", 1.0),
        "RUDDER_FORCE_SCALE": 1.0,
        "BOW_THRUSTER_SCALE": 1.0,
        "ROLL_DAMP_SCALE": 1.0,
        "ROLL_RESTORE_SCALE": 1.0,
    }


def score_turn(sim_metrics: Dict[str, Any], real_metrics: Dict[str, Any]) -> float:
    total = 0.0
    for k, w in WEIGHTS.items():
        total += w * safe_rel_error(sim_metrics.get(k), real_metrics["turn_metrics"].get(k))
    return float(total)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, real_turn = load_default_real_benchmarks()
    base_params = load_motion_best_params()
    rows: List[Dict[str, Any]] = []

    for rpm, rud_deg, rud_scale, roll_damp, roll_restore in itertools.product(
        TURN_RPM_GRID,
        TURN_RUDDER_DEG_GRID,
        RUDDER_FORCE_SCALE_GRID,
        ROLL_DAMP_SCALE_GRID,
        ROLL_RESTORE_SCALE_GRID,
    ):
        params = {
            **base_params,
            "RUDDER_FORCE_SCALE": rud_scale,
            "ROLL_DAMP_SCALE": roll_damp,
            "ROLL_RESTORE_SCALE": roll_restore,
        }
        sim = run_open_loop(
            rpm=rpm,
            rudder_deg=rud_deg,
            duration_s=DURATION_S,
            dt=DT,
            warmup_s=WARMUP_S,
            params=params,
        )
        metrics = extract_turn_metrics(sim)
        score = score_turn(metrics, real_turn)
        row = {
            "turn_rpm": rpm,
            "turn_rudder_deg": rud_deg,
            **params,
            "score_total": score,
            **metrics,
        }
        rows.append(row)

    rows.sort(key=lambda r: r["score_total"])
    best = rows[0]
    best_params = {
        "RPM_INPUT_TO_SOLVER_RPM": best["RPM_INPUT_TO_SOLVER_RPM"],
        "PROPELLER_THRUST_SCALE": best["PROPELLER_THRUST_SCALE"],
        "RUDDER_FORCE_SCALE": best["RUDDER_FORCE_SCALE"],
        "BOW_THRUSTER_SCALE": best["BOW_THRUSTER_SCALE"],
        "ROLL_DAMP_SCALE": best["ROLL_DAMP_SCALE"],
        "ROLL_RESTORE_SCALE": best["ROLL_RESTORE_SCALE"],
    }
    best_sim = run_open_loop(
        rpm=best["turn_rpm"],
        rudder_deg=best["turn_rudder_deg"],
        duration_s=DURATION_S,
        dt=DT,
        warmup_s=WARMUP_S,
        params=best_params,
    )
    best_metrics = extract_turn_metrics(best_sim)
    best_comparison = {
        "turn": compare_metrics(
            best_metrics,
            real_turn["turn_metrics"],
            WEIGHTS.keys(),
        )
    }

    with (OUT_DIR / "best_4dof_turn_config.json").open("w", encoding="utf-8") as f:
        json.dump({
            "turn_rpm": best["turn_rpm"],
            "turn_rudder_deg": best["turn_rudder_deg"],
            **best_params,
            "score_total": best["score_total"],
        }, f, indent=2)
    with (OUT_DIR / "best_4dof_turn_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"turn_metrics": best_metrics}, f, indent=2)
    with (OUT_DIR / "best_4dof_turn_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(best_comparison, f, indent=2)
    with (OUT_DIR / "stage2_4dof_turn_top20.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows[:20]}, f, indent=2)

    print("Saved turn sweep outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
