"""Replay recorded commands and raw ego measurements; never runs episodes.

The input must come from an oracle/oracle_state prediction audit, whose
snapshot body state is truth. Only raw measurements and issued commands enter
the observer; truth is used solely to score its errors. Current audit defaults
have fresh pose and no command limiter. If supplied, a pose_stale column is
respected. Input rudder is already the issued, limited command.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from classical import common as cc
from safety_observer import EgoObserver
import safety_v2 as v2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/safety_dev/wall_oracle_state_steps.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/safety_dev/observer_replay.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if not frame["mode"].isin(["oracle", "oracle_state"]).all():
        raise ValueError("Use oracle/oracle_state logs: snapshot ego state must be ground truth")
    records = []
    for (mode, case), steps in frame.groupby(["mode", "case"], sort=False):
        steps = steps.sort_values("step")
        if not np.array_equal(steps.step.to_numpy(), np.arange(len(steps))):
            raise ValueError(f"{case}: complete consecutive command history from step zero is required")
        estimator, actuators, ema = EgoObserver(), cc.Actuators(), None
        previous_rpm = np.nan
        for row in steps.itertuples(index=False):
            raw = np.array([row.measured_u_mps, row.measured_v_mps, row.measured_r_radps])
            truth = np.array([row.snapshot_u_mps, row.snapshot_v_mps, row.snapshot_r_radps])
            stale = str(getattr(row, "pose_stale", False)).lower() in ("true", "1")
            estimated = estimator.update(raw, fresh=not stale)
            ema = raw.copy() if ema is None else ema + v2.EGO_SMOOTHING * (raw - ema)
            for method, value in (("raw", raw), ("ema", ema), ("model_prior", estimated)):
                error = value - truth
                records.append({
                    "mode": mode, "case": case, "step": int(row.step),
                    "phase": row.phase, "method": method, "pose_stale": stale,
                    "previous_rpm": previous_rpm,
                    "u_error_mps": error[0], "v_error_mps": error[1],
                    "r_error_dps": math.degrees(error[2]),
                    "u_abs_error_mps": abs(error[0]), "v_abs_error_mps": abs(error[1]),
                    "r_abs_error_dps": abs(math.degrees(error[2])),
                    "true_u_mps": truth[0], "true_v_mps": truth[1],
                    "true_r_dps": math.degrees(truth[2]),
                    "estimated_u_mps": value[0], "estimated_v_mps": value[1],
                    "estimated_r_dps": math.degrees(value[2]),
                })
            snap = SimpleNamespace(x=0.0, y=0.0, heading=0.0,
                                   u=estimated[0], v=estimated[1], r=estimated[2])
            # Body equations do not depend on world pose, so the dummy pose
            # avoids feeding ground-truth heading or position to the observer.
            estimator.predict(snap, actuators, row.rudder, row.rpm)
            actuators.issue(SimpleNamespace(command_rate_limit=False), row.rudder)
            previous_rpm = row.rpm
    output = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = output.groupby("method").agg(
        samples=("step", "size"),
        u_mae=("u_abs_error_mps", "mean"), u_p95=("u_abs_error_mps", lambda x: x.quantile(.95)),
        v_mae=("v_abs_error_mps", "mean"), v_p95=("v_abs_error_mps", lambda x: x.quantile(.95)),
        r_mae_dps=("r_abs_error_dps", "mean"), r_p95_dps=("r_abs_error_dps", lambda x: x.quantile(.95)))
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"))
    print(f"Scored {frame['case'].nunique()} recorded cases / {len(frame)} decisions; no new episodes")
    print(summary.round(5).to_string())
    by_case = output.groupby(["case", "method"])[["u_abs_error_mps", "v_abs_error_mps", "r_abs_error_dps"]].mean().unstack()
    for metric in ("u_abs_error_mps", "v_abs_error_mps", "r_abs_error_dps"):
        won = (by_case[metric]["model_prior"] < by_case[metric]["ema"]).sum()
        print(f"Lower case-mean {metric} than EMA: {won}/{len(by_case)}")
    print(args.output)


if __name__ == "__main__":
    main()
