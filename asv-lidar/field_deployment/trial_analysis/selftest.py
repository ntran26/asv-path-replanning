"""Self-test of the trial analysis on the synthetic session (`synthetic.py`).

Checks, each against a simulated reference:

1. rule checks: `trial_metrics.encounter_behaviour` on the logged run and its ground truth
   against `colregs_compliance.episode` on the simulator's own trajectory;
2. twin reproduction: the twin of each run with the episode's own plant draw and seed, the
   own ship and target from the ground truth, must reproduce the episode (outcome, track);
3. metrics: the field metrics of each run against `run_episode`'s on that reproduction;
4. open loop: the logged commands through the episode's own plant must predict the track far
   better than through another parameter set.

    python selftest.py [--session DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import trial_io as tio
import trial_metrics as tm
import twin as twinmod
import open_loop as ol
import synthetic

import colregs_compliance as cc  # noqa: E402
import ship as shipmod  # noqa: E402

RULE_FIELDS = ("rule_compliant", "first_manoeuvre", "target_side_at_cpa", "crossed_ahead")
METRICS = ("min_target_range", "min_obstacle_clearance", "min_boundary_clearance", "path_efficiency", "rms_cte",
           "mean_speed")


def sim_reference(session: Path, inp: tio.TrialInputs) -> dict:
    return json.loads((session / "ground_truth" / f"{inp.run.run_id}_sim.json").read_text())


def check_rules(session: Path, inputs) -> list:
    out = []
    for inp in inputs:
        field = tm.encounter_behaviour(inp)
        if field is None:
            continue
        ref = sim_reference(session, inp)
        own, tgt = inp.own_gt, inp.target
        cls, side = tio.ENCOUNTER_CLASS[inp.sheet["encounter"]]
        tr = {"own_x": own.x, "own_y": own.y, "own_h": own.heading_unwrapped % 360.0, "own_u": own.speed,
              "tgt_x": tgt.x, "tgt_y": tgt.y, "tgt_h": tgt.heading_unwrapped % 360.0,
              "engaged": np.array(ref["engaged"], dtype=bool), "path": inp.run.path}
        sim = cc.episode({"class": cls, "crossing_side": side, "outcome": ref["outcome"], "variant": "FIX",
                          "test_id": inp.run.run_id}, tr)
        diffs = {k: (field[k], sim[k]) for k in RULE_FIELDS if not (field[k] == sim[k] or
                 (isinstance(field[k], float) and np.isnan(field[k]) and np.isnan(sim[k])))}
        out.append({"run_id": inp.run.run_id, "agree": not diffs, "differences": diffs,
                    "cpa_field": round(field["cpa_m"], 4), "cpa_sim": round(sim["cpa_m"], 4)})
    return out


def check_twins(session: Path, inputs) -> list:
    twinmod._init(twinmod.pb.DEFAULT_MODEL)
    out = []
    for inp in inputs:
        ref = sim_reference(session, inp)
        spec = twinmod.twin_spec(inp)
        row = twinmod._twin_job((spec, "episode_plant", None, int(ref["seed"]), True))
        tr = row.pop("_traj")
        n = min(len(tr["own_x"]), len(inp.own_gt.x))
        dev = float(np.max(np.hypot(tr["own_x"][:n] - inp.own_gt.x[:n], tr["own_y"][:n] - inp.own_gt.y[:n])))
        field = tm.run_metrics(inp)
        metric_diff = {m: round(float(field[m]) - float(row[m]), 4) for m in METRICS
                       if np.isfinite(field[m]) and np.isfinite(row[m])}
        out.append({"run_id": inp.run.run_id, "sim_outcome": ref["outcome"], "twin_outcome": row["outcome"],
                    "outcome_equal": ref["outcome"] == row["outcome"], "max_track_deviation_m": round(dev, 6),
                    "field_minus_sim_metrics": metric_diff})
    return out


def check_open_loop(session: Path, inputs) -> list:
    out = []
    rng = np.random.default_rng(1)
    other = shipmod.sample_params(rng, scale=1.0)
    for inp in inputs:
        run = ol.bluefin_run(inp)
        if run is None:
            continue
        ref = sim_reference(session, inp)
        models = {"episode": {k: float(ref["plant"][k]) for k in shipmod.IDENTIFIED}, "identified": shipmod.IDENTIFIED,
                  "other": other}
        row = ol.per_run_table([run], models, horizon=3.0)[0]
        out.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=synthetic.DEFAULT_OUT)
    args = ap.parse_args()
    inputs = tio.load_session(args.session)
    report = {"rules": check_rules(args.session, inputs), "open_loop": check_open_loop(args.session, inputs),
              "twins": check_twins(args.session, inputs)}
    report["summary"] = {
        "rule_checks_agree": f"{sum(r['agree'] for r in report['rules'])} / {len(report['rules'])}",
        "twin_outcomes_equal": f"{sum(r['outcome_equal'] for r in report['twins'])} / {len(report['twins'])}",
        "twin_max_track_deviation_m": max(r["max_track_deviation_m"] for r in report["twins"]),
    }
    out = args.session.parent / "selftest_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
