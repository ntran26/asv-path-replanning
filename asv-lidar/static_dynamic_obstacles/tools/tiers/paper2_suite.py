"""The Paper 2 deployment-layout set on one policy -- a separate evaluation set.

The three published Paper 2 field layouts with and without a target ship
(`src/paper2_set.py`): 30 no-target episodes, and 300 target scenarios (five
COLREGs encounters x 3 layouts x 20), each run with a constant-velocity target
(FIX) and with the same target varying its speed once (VAR) -- 630 episodes per
supervisor mode.  **Not part of the frozen suite**, and not in the paper's
headline tables.

    python tools/tiers/paper2_suite.py --model runs/sac_formulation_seed0_bl2/best_model.zip
    python tools/tiers/paper2_suite.py --model runs/*_formulation_seed0_bl2/best_model.zip   # built once
    python tools/tiers/paper2_suite.py --policy colregs_vo --tag colregs_vo

Writes `results/paper2_set/<tag>/`: `episodes.csv`, `summary.txt`, `index.csv`
(every scenario: leg, encounter, target speed, drawn DCPA/TCPA, the VAR speed
change), `field_sheet.csv` (each run's field set-up: target start, heading to
hold, speed, when to change speed and to what) and `manifest.json` (the scenario
digests the run was scored against).  Every target scenario is field feasible
(set revision 2.0; `src/paper2_set.py`).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))

import curriculum  # noqa: E402
import paper2_set as p2  # noqa: E402
import train_formulation as tf  # noqa: E402
from common import run_pool  # noqa: E402

OUT = ROOT / "results" / "paper2_set"


def _summarise(d: pd.DataFrame, group) -> pd.DataFrame:
    g = d.groupby(group)
    return pd.DataFrame({
        "n": g.size(),
        "success": g.apply(lambda x: (x.outcome == "goal").mean(), include_groups=False),
        "coll_target": g.collided_target.mean(),
        "coll_other": g.apply(lambda x: (x.collided & ~x.collided_target).mean(), include_groups=False),
        "rms_cte": g.rms_cte.mean(),
        "min_range": g.min_target_range.median(),
        "estops": g.estops.mean(),
    }).round(3)


def index_frame(records) -> pd.DataFrame:
    rows = []
    for r in records:
        b = r["built"]
        prof = (b.flags or {}).get("speed_profile")
        rows.append({"test_id": r["test_id"], "layout": r["layout"], "encounter": r["encounter"],
                     "speed": r["speed"], "episode_seed": r["episode_seed"], "twin": r["twin"],
                     "start": list(p2.LAYOUTS[r["layout"]]["start"]), "goal": list(p2.LAYOUTS[r["layout"]]["goal"]),
                     "target_speed": round(float(b.target_speed), 3) if r["encounter"] != "NT" else "",
                     "dcpa_m": round(float(b.dcpa_m), 2) if r["encounter"] != "NT" else "",
                     "tcpa_s": round(float(b.tcpa_s), 1) if r["encounter"] != "NT" else "",
                     "vs_change_s": round(prof[0], 2) if prof else "",
                     "vs_final_speed": round(prof[1], 3) if prof else "",
                     "target_panel_clearance_m": r["target_clearance_m"]})
    return pd.DataFrame(rows)


def _tag_for(model: Path) -> str:
    """runs/sac_formulation_seed0_bl2/best_model.zip -> sacs0_bl2."""
    m = re.match(r"(.+)_formulation_seed(\d+)_(.+)", model.parent.name)
    return f"{m.group(1)}s{m.group(2)}_{m.group(3)}" if m else model.parent.name


def evaluate(records, short, manifest, *, tag, model=None, policy="model",
             supervisor="both", processes=None) -> str:
    """Run the set on one policy; write `results/paper2_set/<tag>/`."""
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    index_frame(records).to_csv(out / "index.csv", index=False)
    pd.DataFrame(p2.field_sheet(records)).to_csv(out / "field_sheet.csv", index=False)
    jobs = [(r["built"], r["episode_seed"], policy, None,
             {"set": "paper2", "test_id": r["test_id"], "layout": r["layout"],
              "encounter": r["encounter"], "speed": r["speed"], "twin": r["twin"]})
            for r in records]
    frames = []
    for mode in (("off", "on") if supervisor == "both" else (supervisor,)):
        rows = run_pool(jobs, model_path=model, processes=processes,
                        overrides={"EMERGENCY_STOP_ENABLED": mode == "on"})
        f = pd.DataFrame(rows)
        f["supervisor"] = mode
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(out / "episodes.csv", index=False)

    pd.set_option("display.width", 220)
    off = d[d.supervisor == d.supervisor.iloc[0]]
    tgt = off[off.encounter != "NT"]
    text = [f"Paper 2 deployment-layout set ({tag}) -- {model.relative_to(ROOT) if model else policy}",
            f"{len(d)} episodes, set revision {p2.SET_REVISION}, manifest {manifest['manifest_digest'][:16]}, "
            f"{time.time() - started:.0f} s", ""]
    if short:
        text += [f"== {len(short)} cell(s) short of their count: {short}", ""]
    text += ["-- by supervisor", _summarise(d, "supervisor").to_string(), "",
             f"-- no target, by layout (supervisor {off.supervisor.iloc[0]})",
             _summarise(off[off.encounter == "NT"], "layout").to_string(), "",
             "-- with a target, by encounter and speed", _summarise(tgt, ["encounter", "speed"]).to_string(), "",
             "-- with a target, by layout and speed", _summarise(tgt, ["layout", "speed"]).to_string(), ""]
    var = tgt[tgt.speed == "VAR"][["test_id", "twin", "outcome"]]
    fix = tgt[tgt.speed == "FIX"][["test_id", "outcome"]].rename(columns={"test_id": "twin", "outcome": "fix_outcome"})
    pair = var.merge(fix, on="twin")
    if len(pair):
        pair["enc"] = pair.twin.str.split("-").str[2]
        g = pair.groupby("enc")
        text += ["-- VAR against its FIX twin (same scenario and seed; only the speed differs)",
                 pd.DataFrame({"n": g.size(),
                               "success_fix": g.apply(lambda x: (x.fix_outcome == "goal").mean(), include_groups=False),
                               "success_var": g.apply(lambda x: (x.outcome == "goal").mean(), include_groups=False),
                               "lost": g.apply(lambda x: ((x.fix_outcome == "goal") & (x.outcome != "goal")).mean(), include_groups=False),
                               "gained": g.apply(lambda x: ((x.fix_outcome != "goal") & (x.outcome == "goal")).mean(), include_groups=False)}
                              ).round(3).to_string(), ""]
    body = "\n".join(text) + "\n"
    (out / "summary.txt").write_text(body, encoding="utf-8")
    print(body, flush=True)
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, nargs="+",
                    help="one or more trained policies (.zip); the set is built once for all")
    ap.add_argument("--policy", choices=("los_dwa", "colregs_vo", "encounter_vo", "reference"),
                    help="a classical comparator instead of a model")
    ap.add_argument("--tag", help="output folder (default for a model: e.g. sacs0_bl2 from its run folder)")
    ap.add_argument("--supervisor", choices=("off", "on", "both"), default="both")
    ap.add_argument("--processes", type=int, default=None)
    args = ap.parse_args()
    if bool(args.model) == bool(args.policy):
        ap.error("give exactly one of --model or --policy")
    if args.tag and args.model and len(args.model) > 1:
        ap.error("--tag names one output; leave it out with several models")
    if args.policy and not args.tag:
        args.tag = args.policy

    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    records, short = p2.build(ASVLidarEnv(render_mode=None, emergency_stop=False))
    digests = {r["test_id"]: r["built"].digest() for r in records}
    blob = json.dumps(digests, sort_keys=True, separators=(",", ":"))
    manifest = {"set": "paper2", "revision": p2.SET_REVISION, "n_episodes": len(records),
                "shortfall": short, "case_digests": digests,
                "manifest_digest": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
                "layouts": p2.LAYOUTS}
    if args.policy:
        evaluate(records, short, manifest, tag=args.tag, policy=args.policy,
                 supervisor=args.supervisor, processes=args.processes)
        return 0
    for m in args.model:
        model = m if m.is_absolute() else ROOT / m
        evaluate(records, short, manifest, tag=args.tag or _tag_for(model), model=model,
                 supervisor=args.supervisor, processes=args.processes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
