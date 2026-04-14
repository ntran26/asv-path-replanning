from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict

import focused_4dof_sweep as base

OUT_DIR = "fourdof_turn_timing_results"

# Keep the same observables as the existing turn score, but put more weight on
# the heading-change timings because that is the main remaining mismatch.
TIMING_TURN_WEIGHTS = {
    "peak_abs_yaw_rate_degps": 1.5,
    "time_to_90deg_after_turn_s": 4.0,
    "time_to_180deg_after_turn_s": 4.0,
    "radius_first_90deg_m": 1.0,
    "radius_first_180deg_m": 1.0,
    "u_body_10s_after_turn_mps": 1.0,
}


def build_straight_cfg(best: Dict[str, Any]) -> Dict[str, float]:
    return {
        "RPM_COMMAND_SCALE": best["RPM_COMMAND_SCALE"],
        "PROPELLER_THRUST_SCALE": best["PROPELLER_THRUST_SCALE"],
        "PROPELLER_ADVANCE_SCALE": best["PROPELLER_ADVANCE_SCALE"],
        "RUDDER_FORCE_SCALE": best["RUDDER_FORCE_SCALE"],
        "RUDDER_YAW_SCALE": best["RUDDER_YAW_SCALE"],
        "RUDDER_INFLOW_SCALE": best["RUDDER_INFLOW_SCALE"],
        "RUDDER_X_DRAG_SCALE": best["RUDDER_X_DRAG_SCALE"],
        "LINEAR_SURGE_DAMP": best["LINEAR_SURGE_DAMP"],
        "LINEAR_YAW_DAMP": best["LINEAR_YAW_DAMP"],
        "ROLL_DAMP_SCALE": best["ROLL_DAMP_SCALE"],
        "ROLL_RESTORE_SCALE": best["ROLL_RESTORE_SCALE"],
        "BOW_THRUSTER_SCALE": best["BOW_THRUSTER_SCALE"],
        "MAX_RUD_RATE_DPS": best.get("MAX_RUD_RATE_DPS", 20.0),
    }


def main() -> None:
    out = base.THIS_DIR / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    real_speed = base.load_json(base.THIS_DIR / base.REAL_SPEED_JSON)
    real_turn = base.load_json(base.THIS_DIR / base.REAL_TURN_JSON)
    real_motion = base.get_real_targets(real_speed, "straight_metrics", base.MOTION_KEYS)
    real_turn_targets = base.get_real_targets(real_turn, "turn_metrics", base.TURN_KEYS)

    fine_dir = base.THIS_DIR / "fourdof_fine_results"
    fine_best = base.load_json(fine_dir / "best_fine_4dof_config.json")
    fine_summary = base.load_json(fine_dir / "best_fine_4dof_summary.json")

    straight_cfg = build_straight_cfg(fine_best)
    straight_sim = base.simulate_open_loop(
        **straight_cfg,
        rpm=fine_best["straight_rpm"],
        rudder_deg=0.0,
        duration_s=base.STRAIGHT_DURATION,
        dt=base.DT,
    )
    straight_metrics = base.extract_motion_metrics(straight_sim)
    straight_score = base.score_section(straight_metrics, real_motion, base.MOTION_WEIGHTS)

    baseline_turn_sim = base.simulate_turn_with_leadin(
        **straight_cfg,
        rpm=fine_best["turn_rpm"],
        rudder_deg=fine_best["turn_rudder_deg"],
        turn_leadin_s=base.TURN_LEADIN_DURATION,
        duration_s=base.TURN_DURATION,
        dt=base.DT,
    )
    baseline_turn_metrics = base.extract_turn_metrics(baseline_turn_sim)
    baseline_turn_score = base.score_section(
        baseline_turn_metrics,
        real_turn_targets,
        base.TURN_WEIGHTS,
    )
    baseline_timing_turn_score = base.score_section(
        baseline_turn_metrics,
        real_turn_targets,
        TIMING_TURN_WEIGHTS,
    )

    turn_rpm_grid = [17.0, 18.0, 19.0]
    turn_rudder_grid = [28.0, 30.0, 32.0]
    rudder_rate_grid = [8.0, 10.0, 12.0, 15.0, 20.0]
    rudder_force_grid = [0.10, 0.12, 0.14]
    rudder_yaw_grid = [1.5, 1.7, 1.9]
    rudder_inflow_grid = [0.9, 1.0, 1.1]
    yaw_damp_grid = [0.0, 0.1, 0.2]
    roll_restore_grid = [1.0, 1.2]

    rows = []
    for (
        turn_rpm,
        turn_rudder_deg,
        rudder_rate_dps,
        rudder_force,
        rudder_yaw,
        rudder_inflow,
        yaw_damp,
        roll_restore,
    ) in itertools.product(
        turn_rpm_grid,
        turn_rudder_grid,
        rudder_rate_grid,
        rudder_force_grid,
        rudder_yaw_grid,
        rudder_inflow_grid,
        yaw_damp_grid,
        roll_restore_grid,
    ):
        turn_cfg = {
            **straight_cfg,
            "RUDDER_FORCE_SCALE": rudder_force,
            "RUDDER_YAW_SCALE": rudder_yaw,
            "RUDDER_INFLOW_SCALE": rudder_inflow,
            "LINEAR_YAW_DAMP": yaw_damp,
            "ROLL_RESTORE_SCALE": roll_restore,
            "MAX_RUD_RATE_DPS": rudder_rate_dps,
        }
        turn_sim = base.simulate_turn_with_leadin(
            **turn_cfg,
            rpm=turn_rpm,
            rudder_deg=turn_rudder_deg,
            turn_leadin_s=base.TURN_LEADIN_DURATION,
            duration_s=base.TURN_DURATION,
            dt=base.DT,
        )
        turn_metrics = base.extract_turn_metrics(turn_sim)
        turn_score = base.score_section(turn_metrics, real_turn_targets, base.TURN_WEIGHTS)
        timing_turn_score = base.score_section(
            turn_metrics,
            real_turn_targets,
            TIMING_TURN_WEIGHTS,
        )
        rows.append(
            {
                "straight_rpm": fine_best["straight_rpm"],
                "turn_rpm": turn_rpm,
                "turn_rudder_deg": turn_rudder_deg,
                **turn_cfg,
                "surge_score": straight_score["score_total"],
                "turn_score": turn_score["score_total"],
                "timing_turn_score": timing_turn_score["score_total"],
                "standard_joint_score": straight_score["score_total"] + turn_score["score_total"],
                "timing_joint_score": straight_score["score_total"] + timing_turn_score["score_total"],
                **{f"turn_err_{k}": v for k, v in turn_score["parts"].items()},
                **{f"timing_err_{k}": v for k, v in timing_turn_score["parts"].items()},
                **{f"turn_{k}": v for k, v in turn_metrics.items()},
            }
        )

    rows.sort(key=lambda row: (row["timing_joint_score"], row["standard_joint_score"]))
    base.save_json(out / "timing_turn_top20.json", {"rows": rows[:20]})

    best = rows[0]
    summary: Dict[str, Any] = {
        "baseline_fine_joint_score": fine_best["joint_score"],
        "baseline_fine_turn_score": fine_best["turn_score"],
        "baseline_fine_timing_turn_score": baseline_timing_turn_score["score_total"],
        "baseline_fine_timing_joint_score": straight_score["score_total"] + baseline_timing_turn_score["score_total"],
        "best_timing_joint_score": best["timing_joint_score"],
        "best_timing_turn_score": best["timing_turn_score"],
        "best_standard_joint_score": best["standard_joint_score"],
        "best_standard_turn_score": best["turn_score"],
        "timing_improvement_vs_baseline": best["timing_joint_score"] - (straight_score["score_total"] + baseline_timing_turn_score["score_total"]),
        "beats_baseline_on_timing_objective": best["timing_joint_score"] < (straight_score["score_total"] + baseline_timing_turn_score["score_total"]),
        "baseline_time_to_90deg_after_turn_s": baseline_turn_metrics["time_to_90deg_after_turn_s"],
        "baseline_time_to_180deg_after_turn_s": baseline_turn_metrics["time_to_180deg_after_turn_s"],
        "best_time_to_90deg_after_turn_s": best["turn_time_to_90deg_after_turn_s"],
        "best_time_to_180deg_after_turn_s": best["turn_time_to_180deg_after_turn_s"],
        "real_time_to_90deg_after_turn_s": real_turn_targets["time_to_90deg_after_turn_s"],
        "real_time_to_180deg_after_turn_s": real_turn_targets["time_to_180deg_after_turn_s"],
        "previous_fine_improvement_vs_focus": fine_summary["improvement_vs_previous"],
    }

    base.save_json(out / "best_timing_4dof_config.json", best)
    base.save_json(out / "best_timing_4dof_summary.json", summary)


if __name__ == "__main__":
    main()
