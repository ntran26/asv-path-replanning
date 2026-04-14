from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict

import focused_4dof_sweep as base

OUT_DIR = "fourdof_fine_results"
TOP_STAGE1_K = 4


def main() -> None:
    out = base.THIS_DIR / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    real_speed = base.load_json(base.THIS_DIR / base.REAL_SPEED_JSON)
    real_turn = base.load_json(base.THIS_DIR / base.REAL_TURN_JSON)
    real_motion = base.get_real_targets(real_speed, "straight_metrics", base.MOTION_KEYS)
    real_turn_targets = base.get_real_targets(real_turn, "turn_metrics", base.TURN_KEYS)

    prev_best = base.load_json(base.THIS_DIR / base.OUT_DIR / "best_4dof_joint_config.json")
    prev_summary = base.load_json(base.THIS_DIR / base.OUT_DIR / "best_4dof_vs_v2_summary.json")

    straight_rpm_grid = [13.0, 14.0, 15.0]
    rpm_scale_grid = [90.0, 95.0, 100.0]
    prop_thrust_grid = [1.8, 2.0, 2.2]
    prop_advance_grid = [0.75, 0.85, 0.95, 1.0]
    surge_damp_grid = [1.5, 1.75, 2.0]

    stage1_rows = []
    for rpm, rpm_scale, thrust_scale, advance_scale, surge_damp in itertools.product(
        straight_rpm_grid,
        rpm_scale_grid,
        prop_thrust_grid,
        prop_advance_grid,
        surge_damp_grid,
    ):
        sim = base.simulate_open_loop(
            RPM_COMMAND_SCALE=rpm_scale,
            PROPELLER_THRUST_SCALE=thrust_scale,
            PROPELLER_ADVANCE_SCALE=advance_scale,
            RUDDER_FORCE_SCALE=prev_best["RUDDER_FORCE_SCALE"],
            RUDDER_YAW_SCALE=prev_best["RUDDER_YAW_SCALE"],
            RUDDER_INFLOW_SCALE=1.0,
            RUDDER_X_DRAG_SCALE=prev_best["RUDDER_X_DRAG_SCALE"],
            LINEAR_SURGE_DAMP=surge_damp,
            LINEAR_YAW_DAMP=prev_best["LINEAR_YAW_DAMP"],
            ROLL_DAMP_SCALE=prev_best["ROLL_DAMP_SCALE"],
            ROLL_RESTORE_SCALE=1.0,
            BOW_THRUSTER_SCALE=1.0,
            rpm=rpm,
            rudder_deg=0.0,
            duration_s=base.STRAIGHT_DURATION,
            dt=base.DT,
        )
        mm = base.extract_motion_metrics(sim)
        score = base.score_section(mm, real_motion, base.MOTION_WEIGHTS)
        stage1_rows.append(
            {
                "straight_rpm": rpm,
                "RPM_COMMAND_SCALE": rpm_scale,
                "PROPELLER_THRUST_SCALE": thrust_scale,
                "PROPELLER_ADVANCE_SCALE": advance_scale,
                "LINEAR_SURGE_DAMP": surge_damp,
                "surge_score": score["score_total"],
                **{f"surge_err_{k}": v for k, v in score["parts"].items()},
                **{f"surge_{k}": v for k, v in mm.items()},
            }
        )

    stage1_rows.sort(key=lambda r: r["surge_score"])
    base.save_json(out / "stage1_fine_top20.json", {"rows": stage1_rows[:20]})
    top_stage1 = stage1_rows[:TOP_STAGE1_K]

    turn_rpm_grid = [15.0, 18.0]
    turn_rudder_grid = [28.0, 30.0, 32.0]
    rudder_force_grid = [0.10, 0.12, 0.15]
    rudder_yaw_grid = [1.3, 1.5, 1.7]
    rudder_inflow_grid = [1.0, 1.1, 1.2]
    rudder_xdrag_grid = [0.0, 0.01]
    yaw_damp_grid = [0.0, 0.1]
    roll_damp_grid = [4.0]
    roll_restore_grid = [1.0, 1.2]

    joint_rows = []
    for base_row in top_stage1:
        for turn_rpm, turn_rudder_deg, rudder_force, rudder_yaw, rudder_inflow, rudder_xdrag, yaw_damp, roll_damp, roll_restore in itertools.product(
            turn_rpm_grid,
            turn_rudder_grid,
            rudder_force_grid,
            rudder_yaw_grid,
            rudder_inflow_grid,
            rudder_xdrag_grid,
            yaw_damp_grid,
            roll_damp_grid,
            roll_restore_grid,
        ):
            common_cfg = {
                "RPM_COMMAND_SCALE": base_row["RPM_COMMAND_SCALE"],
                "PROPELLER_THRUST_SCALE": base_row["PROPELLER_THRUST_SCALE"],
                "PROPELLER_ADVANCE_SCALE": base_row["PROPELLER_ADVANCE_SCALE"],
                "RUDDER_FORCE_SCALE": rudder_force,
                "RUDDER_YAW_SCALE": rudder_yaw,
                "RUDDER_INFLOW_SCALE": rudder_inflow,
                "RUDDER_X_DRAG_SCALE": rudder_xdrag,
                "LINEAR_SURGE_DAMP": base_row["LINEAR_SURGE_DAMP"],
                "LINEAR_YAW_DAMP": yaw_damp,
                "ROLL_DAMP_SCALE": roll_damp,
                "ROLL_RESTORE_SCALE": roll_restore,
                "BOW_THRUSTER_SCALE": 1.0,
            }
            sim_straight = base.simulate_open_loop(
                **common_cfg,
                rpm=base_row["straight_rpm"],
                rudder_deg=0.0,
                duration_s=base.STRAIGHT_DURATION,
                dt=base.DT,
            )
            sim_turn = base.simulate_turn_with_leadin(
                **common_cfg,
                rpm=turn_rpm,
                rudder_deg=turn_rudder_deg,
                turn_leadin_s=base.TURN_LEADIN_DURATION,
                duration_s=base.TURN_DURATION,
                dt=base.DT,
            )
            mm = base.extract_motion_metrics(sim_straight)
            tm = base.extract_turn_metrics(sim_turn)
            sscore = base.score_section(mm, real_motion, base.MOTION_WEIGHTS)
            tscore = base.score_section(tm, real_turn_targets, base.TURN_WEIGHTS)
            joint_rows.append(
                {
                    "straight_rpm": base_row["straight_rpm"],
                    "turn_rpm": turn_rpm,
                    "turn_rudder_deg": turn_rudder_deg,
                    **common_cfg,
                    "surge_score": sscore["score_total"],
                    "turn_score": tscore["score_total"],
                    "joint_score": sscore["score_total"] + tscore["score_total"],
                    **{f"surge_err_{k}": v for k, v in sscore["parts"].items()},
                    **{f"turn_err_{k}": v for k, v in tscore["parts"].items()},
                    **{f"surge_{k}": v for k, v in mm.items()},
                    **{f"turn_{k}": v for k, v in tm.items()},
                }
            )

    joint_rows.sort(key=lambda r: r["joint_score"])
    base.save_json(out / "joint_fine_top20.json", {"rows": joint_rows[:20]})

    best = joint_rows[0]
    summary: Dict[str, Any] = {
        "fine_joint_score": best["joint_score"],
        "fine_surge_score": best["surge_score"],
        "fine_turn_score": best["turn_score"],
        "previous_joint_score": prev_best["joint_score"],
        "previous_turn_score": prev_best["turn_score"],
        "improvement_vs_previous": best["joint_score"] - prev_best["joint_score"],
        "v2_baseline_joint_score": prev_summary["v2_baseline_joint_score"],
        "beats_previous_4dof": best["joint_score"] < prev_best["joint_score"],
        "beats_v2": best["joint_score"] < prev_summary["v2_baseline_joint_score"],
    }

    base.save_json(out / "best_fine_4dof_config.json", best)
    base.save_json(out / "best_fine_4dof_summary.json", summary)


if __name__ == "__main__":
    main()
