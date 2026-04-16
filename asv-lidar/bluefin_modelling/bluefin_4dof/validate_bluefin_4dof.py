"""Quick open-loop validation for ``ship_model_bluefin_4dof.py``.

Runs:
    1) straight-line speed test
    2) turning-circle test

Outputs metrics, comparisons to the real benchmarks (if available), and plots.
Run with no arguments:

    python validate_bluefin_4dof.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from bluefin_4dof_utils import (
    ROOT,
    extract_motion_metrics,
    extract_turn_metrics,
    load_default_real_benchmarks,
    motion_comparison,
    run_open_loop,
    turn_comparison,
)

OUT_DIR = ROOT / "bluefin_4dof_validation"
STRAIGHT_RPM = 15.0
TURN_RPM = 24.0
TURN_RUDDER_DEG = 30.0
STRAIGHT_DURATION_S = 40.0
TURN_DURATION_S = 50.0
DT = 0.1
TURN_WARMUP_S = 1.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    real_motion, real_turn = load_default_real_benchmarks()

    straight = run_open_loop(rpm=STRAIGHT_RPM, rudder_deg=0.0, duration_s=STRAIGHT_DURATION_S, dt=DT)
    turn = run_open_loop(rpm=TURN_RPM, rudder_deg=TURN_RUDDER_DEG, duration_s=TURN_DURATION_S, dt=DT, warmup_s=TURN_WARMUP_S)

    motion_metrics = extract_motion_metrics(straight)
    turn_metrics = extract_turn_metrics(turn)

    motion_comp = motion_comparison(motion_metrics, real_motion)
    turn_comp = turn_comparison(turn_metrics, real_turn)

    with (OUT_DIR / "motion_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"motion_metrics": motion_metrics}, f, indent=2)
    with (OUT_DIR / "turn_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"turn_metrics": turn_metrics}, f, indent=2)
    with (OUT_DIR / "motion_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(motion_comp, f, indent=2)
    with (OUT_DIR / "turn_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(turn_comp, f, indent=2)

    # Speed plot
    plt.figure(figsize=(8, 4.5))
    plt.plot(straight["t_sec"], straight["u_body_mps"], lw=2, label="sim")
    plt.xlabel("Time [s]")
    plt.ylabel("Forward speed [m/s]")
    plt.title(f"4DOF speed test: rpm={STRAIGHT_RPM}, rudder=0 deg")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "speed_test_velocity.png", dpi=200)
    plt.close()

    # Turn trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(turn["x_m"], turn["y_m"], lw=2)
    plt.scatter(turn["x_m"][0], turn["y_m"][0], s=50, label="start")
    plt.scatter(turn["x_m"][-1], turn["y_m"][-1], s=50, label="end")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(f"4DOF turning circle: rpm={TURN_RPM}, rudder={TURN_RUDDER_DEG} deg")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "turning_circle_trajectory.png", dpi=200)
    plt.close()

    # Yaw / yaw rate / roll
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(turn["t_sec"], turn["yaw_deg"], lw=2)
    ax[0].set_ylabel("Yaw [deg]")
    ax[0].set_title("4DOF turning test: yaw, yaw rate, and roll")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(turn["t_sec"], turn["yaw_rate_degps"], lw=2)
    ax[1].set_ylabel("Yaw rate [deg/s]")
    ax[1].grid(True, alpha=0.3)
    ax[2].plot(turn["t_sec"], turn["roll_deg"], lw=2)
    ax[2].set_ylabel("Roll [deg]")
    ax[2].set_xlabel("Time [s]")
    ax[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "turning_circle_yaw_yawrate_roll.png", dpi=200)
    plt.close(fig)

    print("Saved validation outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
