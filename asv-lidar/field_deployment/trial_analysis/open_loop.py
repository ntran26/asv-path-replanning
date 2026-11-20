"""Open-loop replay through the training and the calibrated model (FIELD_TRIALS_PLAN.md
section 7, level 1; claim C-18).

The logged command sequence (transmitted rudder and rpm, zero-order hold) drives each
parameter set with no feedback; heading and position errors are reported against the
prediction horizon, as `bluefin/validate.py` test V1 does: the state is re-initialised from
the measurement every H seconds (H = 1, 2, 3, 5, 10 s), and once at the start for the free
run.  The identified model's own machinery is used (`bluefin/simulate.make_batch`,
`residuals`, `dynamics.rk4_step`); the one change is that the first window starts from the
measured surge and yaw rate instead of rest, because a trial run does not start from rest.
For runs that do start from rest at the origin (the calibration sessions) the two agree.

Sources:
* a trial session folder (`trial_io` layout): each valid run up to its first command not
  from the policy (an operator stop, astern, which the identified dynamics do not carry);
* `--bluefin-data DIR --sessions S1,S2`: logs in the July format, read by
  `bluefin/build_dataset.load_all` (the calibration sessions; C-18's registered test is on
  the session-2 holdout).

C-18, open-loop part: met when the calibrated model's heading and position errors are below
the training model's at every horizon, the free run included.

    python open_loop.py --session <session_dir> --calibrated params.json
    python open_loop.py --bluefin-data <dir> --sessions <session-2 folder> --calibrated params.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import trial_io as tio

import build_dataset as bd  # noqa: E402  (bluefin)
import dynamics as dyn  # noqa: E402
import ship as shipmod  # noqa: E402
from simulate import _tile_params, make_batch, residuals, rms  # noqa: E402
from validate import naive_rms  # noqa: E402

HORIZONS = (1.0, 2.0, 3.0, 5.0, 10.0, None)       # None: free run


def bluefin_run(inp: tio.TrialInputs, session: str = "trial") -> Optional[bd.Run]:
    run = inp.run
    span = run.policy_span()
    if span < 4:
        return None
    t_end = float(run.t[span - 1])
    if inp.own_gt is not None:
        g = inp.own_gt
        keep = (g.t >= 0.0) & (g.t <= t_end)
        t, x, y, yaw = g.t[keep], g.x[keep], g.y[keep], g.heading_unwrapped[keep]
    else:
        t, x, y = run.t[:span], run.pose[:span, 0], run.pose[:span, 1]
        yaw = np.degrees(np.unwrap(np.radians(run.pose[:span, 2])))
    return bd.Run(name=run.run_id, session=session, t=np.asarray(t, float) - float(t[0]),
                  x=np.asarray(x, float) - float(x[0]), y=np.asarray(y, float) - float(y[0]),
                  yaw=np.asarray(yaw, float), tc=run.t[:span].copy(),
                  rudder_cmd=run.vessel_rudder[:span].copy(), rpm_cmd=run.rpm[:span].copy(),
                  meta={"source": "bridge log"})


def simulate_from_state(batch, params: Dict[str, float], reset_every: Optional[float]) -> Dict[str, List[np.ndarray]]:
    """`simulate.simulate_pop` for one parameter set, the first window from the measured state."""
    theta = dyn.as_vector(params)[None, :]
    R, F, dt = batch.R, batch.F, batch.dt
    p = _tile_params(theta, R)
    shift = int(np.clip(np.round(max(float(params["rud_delay"]), 0.0) / dt), 0, F - 1))
    cols = np.clip(np.arange(F) - shift, 0, F - 1)
    delta_cmd = dyn.cmd_percent_to_angle(batch.cmd[:, cols])
    rpm = batch.rpm
    s = np.zeros((7, R))
    s[0], s[2], s[3] = batch.mu_f[:, 0], batch.mr_f[:, 0], batch.psi0
    s[5], s[6] = batch.mx_f[:, 0], batch.my_f[:, 0]
    stride = int(round(reset_every / dt)) if reset_every else 0
    psi_h, x_h, y_h = np.empty((R, F)), np.empty((R, F)), np.empty((R, F))
    psi_h[:, 0], x_h[:, 0], y_h[:, 0] = s[3], s[5], s[6]
    for k in range(1, F):
        s = dyn.rk4_step(s, rpm[:, k - 1], delta_cmd[:, k - 1], p, dt)
        if stride and k % stride == 0:
            s[3], s[5], s[6] = batch.mpsi_f[:, k], batch.mx_f[:, k], batch.my_f[:, k]
            s[2], s[0] = batch.mr_f[:, k], batch.mu_f[:, k]
        psi_h[:, k], x_h[:, k], y_h[:, k] = s[3], s[5], s[6]
    return {"psi": [psi_h[i, batch.idx[i]] for i in range(R)],
            "x": [x_h[i, batch.idx[i]] for i in range(R)],
            "y": [y_h[i, batch.idx[i]] for i in range(R)]}


def horizon_table(runs: List[bd.Run], models: Dict[str, Dict[str, float]]) -> List[Dict]:
    batch = make_batch(runs)
    rows = []
    for H in HORIZONS:
        row = {"horizon": "free" if H is None else f"{H:.0f} s"}
        for name, params in models.items():
            e_psi, e_pos = residuals(batch, simulate_from_state(batch, params, H), skip_after_reset=1)
            row[f"{name}_heading_rms_deg"] = rms(e_psi)
            row[f"{name}_position_rms_m"] = rms(e_pos)
        horizon = 1e6 if H is None else H
        row["freeze_heading_rms_deg"] = naive_rms(runs, horizon, "freeze")
        row["const_r_heading_rms_deg"] = naive_rms(runs, horizon, "constr")
        rows.append(row)
    return rows


def per_run_table(runs: List[bd.Run], models: Dict[str, Dict[str, float]], horizon: Optional[float] = 3.0) -> List[Dict]:
    """Per-run errors at one horizon (F3's open-loop column)."""
    out = []
    for run in runs:
        batch = make_batch([run])
        row = {"run_id": run.name, "horizon": "free" if horizon is None else f"{horizon:.0f} s"}
        for name, params in models.items():
            e_psi, e_pos = residuals(batch, simulate_from_state(batch, params, horizon), skip_after_reset=1)
            row[f"{name}_heading_rms_deg"] = rms(e_psi)
            row[f"{name}_position_rms_m"] = rms(e_pos)
        out.append(row)
    return out


def c18_open_loop_verdict(table: List[Dict]) -> Dict:
    if not table or "calibrated_heading_rms_deg" not in table[0]:
        return {"met": None, "reason": "no calibrated model"}
    worse = [r["horizon"] for r in table
             if not (r["calibrated_heading_rms_deg"] < r["training_heading_rms_deg"]
                     and r["calibrated_position_rms_m"] < r["training_position_rms_m"])]
    return {"met": not worse, "horizons_not_better": worse,
            "rule": "calibrated below training in heading and position at every horizon, free run included"}


def load_models(calibrated: Optional[str]) -> Dict[str, Dict[str, float]]:
    models = {"training": dict(shipmod.IDENTIFIED)}
    if calibrated:
        data = json.loads(Path(calibrated).read_text())
        params = data.get("params", data)
        models["calibrated"] = {k: float(params[k]) for k in shipmod.IDENTIFIED}
    return models


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, help="trial session folder")
    ap.add_argument("--bluefin-data", type=Path, help="folder of July-format logs, one subfolder per session")
    ap.add_argument("--sessions", default="", help="comma-separated subfolders of --bluefin-data")
    ap.add_argument("--calibrated", help="parameter JSON of the session-1 fit")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.session:
        runs = [r for r in (bluefin_run(i) for i in tio.load_session(args.session) if i.sheet["valid"]) if r]
        out = args.out or args.session / "analysis" / "open_loop"
    elif args.bluefin_data:
        runs = [r for r in bd.load_all(str(args.bluefin_data), tuple(s for s in args.sessions.split(",") if s))
                if bd.classify(r) == "moving"]
        out = args.out or Path("open_loop_" + "_".join(s for s in args.sessions.split(",") if s))
    else:
        raise SystemExit("give --session or --bluefin-data")
    if not runs:
        raise SystemExit("no runs to replay")
    models = load_models(args.calibrated)
    table = horizon_table(runs, models)
    verdict = c18_open_loop_verdict(table)
    import pandas as pd
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(out / "horizon_table.csv", index=False)
    pd.DataFrame(per_run_table(runs, models)).to_csv(out / "per_run_3s.csv", index=False)
    (out / "c18_open_loop.json").write_text(json.dumps({"runs": len(runs), **verdict}, indent=2))
    print(pd.DataFrame(table).to_string(index=False))
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
