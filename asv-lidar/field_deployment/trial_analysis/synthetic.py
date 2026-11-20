"""A synthetic trial session from the simulator, to test the analysis.  Not a result.

Each selected test set v4 episode runs in the simulator (the deployed policy, layer off), and
its sensor stream goes through the bridge core exactly as `p3_parity_replay.py` does, with the
shadow layer inline.  The bridge's decision and shadow logs are written in the live bridge's
format, the simulator's true own and target tracks as the ground truth, and a run sheet from
the episode's outcome.  Every quantity the analysis computes then has a simulated reference:
the twin of a run with the episode's own plant and seed must reproduce it, and the rule checks
on the logged tracks must agree with `colregs_compliance.py` on the simulator's.

    python synthetic.py [--cases P2-L1-HO-FIX-03,...] [--out DIR]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import trial_io as tio

import p3_bridge as pb  # noqa: E402
import p3_parity_replay as pr  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

DEFAULT_OUT = tio.FIELD_DIR / "p3_bench" / "trial_analysis_selftest" / "session"
DEFINITION = tio.SIM_ROOT / "results" / "test_set" / "v4" / "definition.csv"
CODE_ENCOUNTER = {v: k for k, v in tio.CELL_CODE.items()}


def default_cases() -> List[str]:
    d = pd.read_csv(DEFINITION)
    picks = []
    for cell in ("L1-HO", "L1-CRS", "L1-CRP", "L1-OT", "L2-HO", "L3-HO", "L1-NT"):
        rows = d[d.cell == cell]
        fix = rows[rows.variant == "FIX"]
        picks.append((fix if len(fix) else rows).iloc[0].test_id)
    for cell in ("basin-head_on", "basin-crossing", "basin-overtaking", "basin-being_overtaken"):
        rows = d[(d.cell == cell) & (d.variant == "CV")]
        picks.append(rows.iloc[0].test_id)
        if cell == "basin-crossing":
            picks.append(rows.iloc[1].test_id)
    return picks


def sheet_fields(test_id: str, built) -> Dict:
    row = pd.read_csv(DEFINITION).set_index("test_id").loc[test_id]
    cell = str(row.cell)
    cls = str(built.encounter_class)
    if cls == "crossing":
        encounter = "crossing_port" if float(built.ct_deg) < 180.0 else "crossing_starboard"
    elif cls in ("no_target", "null"):
        encounter = "none"
    else:
        encounter = cls
    if cell.startswith("L"):
        layout = cell.split("-")[0]
        family = "A" if encounter == "none" else "C"
    else:
        layout, family = "open", ("A" if encounter == "none" else "B")
    scenario = f"{family}-{layout}" if family == "A" else (f"C-{cell}" if family == "C" else f"B-{encounter}")
    return {"family": family, "encounter": encounter, "layout": layout, "scenario_id": scenario}


def simulate(built, seed: int, policy) -> Dict:
    """The episode, as `p3_parity_replay.simulate` runs it in shadow mode, with the true tracks."""
    pr.restart_track_ids()
    pb.cfg.SAFETY_VERSION = 1
    env = ASVLidarEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0)
    obs, _ = env.reset(seed=seed, options={"generated": copy.deepcopy(built)})
    actor = pb.tf.EpisodeActor(policy.model)

    def truth(k: int) -> Dict:
        t = env.targets[0] if env.targets else None
        return {"own": (env.asv_x, env.asv_y, env.asv_h, env.u_body if k == 0 else env.speed_mps),
                "target": None if t is None else (t.x, t.y, t.heading, t.speed),
                "engaged": any(bool(c.engaged) for c in env.encounter_contexts.values())}

    rows, states, outcome = [], [truth(0)], "timeout"
    while True:
        rows.append({"sensor": pr.sensor_from_env(env)})
        obs, _, term, trunc, info = env.step(np.asarray(actor(obs), dtype=np.float32).reshape(2))
        states.append(truth(len(rows)))
        if term or trunc:
            if info.get("collided"):
                outcome = f"collision:{info['collision_kind']}"
            elif info.get("reached_goal"):
                outcome = "goal"
            break
    plant = dict(env.model.p)
    env.close()
    return {"rows": rows, "states": states, "outcome": outcome, "plant": plant}


def bridge_logs(built, seed: int, policy, sim: Dict, run_id: str, test_id: str, provenance: Dict):
    pr.restart_track_ids()
    bridge = pb.Bridge(built, policy, seed=seed)
    inline = pb.SafetyLayer("shadow")
    wall0, mono0 = time.time(), 1000.0
    records, shadows = [], []
    for row in sim["rows"]:
        d = bridge.decide(row["sensor"])
        rec = pb.decision_record(d, run_id=run_id)
        rec["perception"] = pb.perception_record(bridge.env)
        sh = pb.shadow_inline(inline, bridge.env, d.action, d.command.rudder_pct, d.command.rpm)
        rec.update(telemetry=None, other_lines=[], operator_stop={"state": "idle", "reason": ""}, sent=True,
                   clock={"t_wall_released": mono0 + d.sensor.t, "t_wall_sent": mono0 + d.sensor.t + 0.03})
        rec["seconds"]["lidar_line_to_sent"] = rec["seconds"]["observation"] + rec["seconds"]["policy"]
        records.append(rec)
        shadows.append({"type": "shadow", "run_id": run_id, "k": d.k, "t_frame": d.sensor.t,
                        "safety": pb.safety_record(sh)})
    header = pb.run_header(run_id=run_id, policy=policy, env=bridge.env, safety="shadow",
                           extra={**provenance, "args": {"case": test_id}, "geometry_source": f"test set v4 {test_id}",
                                  "synthetic": True, "seed": seed, "wall_utc_at_start": wall0,
                                  "monotonic_at_start": mono0})
    return header, records, shadows


def write_jsonl(path: Path, items) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(pb.jsonable(item)) + "\n")


def write_track(path: Path, states: List[Dict], key: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_run", "x", "y", "heading_deg", "speed"])
        for k, s in enumerate(states):
            if s[key] is not None:
                w.writerow([k * tio.DT, *s[key]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", help="comma-separated test set v4 ids (default: a spread of cells)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    cases = args.cases.split(",") if args.cases else default_cases()
    policy = pb.Policy()
    provenance = {"git_head": pb.git_head(), "source_sha256": pb.source_hashes()}
    out = args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    sheet = []
    for test_id in cases:
        built, seed = pb.load_test_case(test_id)
        run_id = f"syn-{test_id}"
        t0 = time.time()
        sim = simulate(built, seed, policy)
        header, records, shadows = bridge_logs(built, seed, policy, sim, run_id, test_id, provenance)
        folder = out / "runs" / run_id
        folder.mkdir(parents=True, exist_ok=True)
        write_jsonl(folder / "decisions.jsonl", [header] + records)
        write_jsonl(folder / "shadow.jsonl", shadows)
        write_track(out / "ground_truth" / f"{run_id}_own.csv", sim["states"], "own")
        if sim["states"][0]["target"] is not None:
            write_track(out / "ground_truth" / f"{run_id}_target.csv", sim["states"], "target")
        (out / "ground_truth" / f"{run_id}_sim.json").write_text(json.dumps(
            {"test_id": test_id, "seed": seed, "outcome": sim["outcome"], "plant": sim["plant"],
             "engaged": [s["engaged"] for s in sim["states"]]}))
        outcome = {"goal": "goal", "collision:obstacle": "contact_panel", "collision:boundary": "contact_boundary",
                   "collision:target": "contact_target", "timeout": "timeout"}[sim["outcome"]]
        sheet.append({"run_id": run_id, "session": "synthetic", **sheet_fields(test_id, built), "repeat": 1,
                      "valid": 1, "outcome": outcome, "notes": f"simulated {test_id}, seed {seed}"})
        print(f"{run_id}: {sim['outcome']}, {len(records)} decisions, {time.time() - t0:.0f} s", flush=True)
    cols = ["run_id", "session", "scenario_id", "family", "encounter", "layout", "repeat", "valid", "outcome", "notes"]
    with open(out / "run_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sheet)
    print(f"{len(sheet)} synthetic runs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
