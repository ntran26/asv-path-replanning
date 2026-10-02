"""Compare nominal safety integration with the plant, without running episodes.

Run from the project root. This holds the initial state, nominal identified
parameters, and future action sequence identical, isolating integration error.
No policy, scenario set, or environment is loaded.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import constants as cfg
from classical import common as cc
import safety_v3 as v3
import ship


def main() -> None:
    cfg.FIXED_RPM = True
    cfg.CRUISE_RPM = 12.0
    sequences = {
        "straight": [0.0] * 16,
        "hard_starboard": [1.0] * 16,
        "turn_2s_then_center": [1.0] * 4 + [0.0] * 12,
        "turn_2s_countersteer_2s": [1.0] * 4 + [-1.0] * 4 + [0.0] * 8,
        "alternating_full_rudder": [1.0, -1.0] * 8,
    }
    rows = []
    u0 = ship.steady_speed(cfg.CRUISE_RPM)
    for name, rudders in sequences.items():
        seq = np.column_stack((rudders, np.zeros(len(rudders))))
        plant = ship.ShipModel()
        plant._s[0, 0] = u0
        # A zero command was already in flight before the prediction starts.
        plant._delayed_command(0.0, ship.SUB_DT)
        truth = []
        for rudder in rudders:
            for _ in range(5):
                plant.update(cfg.CRUISE_RPM, 100.0 * rudder, 0.1)
            truth.append(plant._s[:, 0].copy())
        truth = np.asarray(truth)
        for dt in (0.125, 0.05):
            cc.PRED_DT = dt
            cc.SUBSTEPS = int(round(cfg.UPDATE_RATE / dt))
            cc.DELAY_STEPS = int(round(cc.IDENTIFIED["rud_delay"] / dt))
            act = cc.Actuators()
            act.buffer = [0.0] * cc.DELAY_STEPS
            snap = cc.Snapshot(0.0, 0.0, 0.0, u0, 0.0, 0.0,
                               np.array([0.0, 1.0]), np.array([1.0, 0.0]),
                               np.zeros(2), 0.0, 0.0, 100.0, np.empty((0, 2)))
            predicted = v3.rollout_seq(snap, act, seq[None])
            for horizon in (1, 2, 4, 8):
                pi = int(round(horizon / dt)) - 1
                ti = int(round(horizon / cfg.UPDATE_RATE)) - 1
                xy = predicted.positions[pi, 0]
                heading = predicted.headings[pi, 0]
                actual = truth[ti]
                error = xy - actual[5:7]
                rows.append({
                    "sequence": name, "prediction_dt_s": dt,
                    "horizon_s": horizon,
                    "position_error_m": float(np.linalg.norm(error)),
                    "x_error_m": float(error[0]), "y_error_m": float(error[1]),
                    "heading_error_deg": math.degrees(float(cc.wrap_pi(heading - actual[3]))),
                    "predicted_heading_deg": math.degrees(heading),
                    "actual_heading_deg": math.degrees(actual[3]),
                    "actual_x_m": float(actual[5]), "actual_y_m": float(actual[6]),
                })
    destination = ROOT / "results" / "safety_dev" / "integration_probe.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Fixed nominal initial surge {u0:.6f} m/s, rpm {cfg.CRUISE_RPM:g}")
    print("sequence,dt,horizon,position_error_m,heading_error_deg,actual_heading_deg")
    for row in rows:
        print(f"{row['sequence']},{row['prediction_dt_s']},{row['horizon_s']},"
              f"{row['position_error_m']:.9f},{row['heading_error_deg']:.9f},"
              f"{row['actual_heading_deg']:.6f}")
    print(destination)


if __name__ == "__main__":
    main()
