"""When does the port swerve in starboard crossings start?  (A28 option 2, F88)

Replays the development starboard crossings under a model and reports, at the
first step the heading leaves the initial leg heading by more than 10 deg:
whether the target is tracked, the encounter state and class, the latched
compliant sense, range and relative bearing to the target, and the turn's
direction; then the peak wrong-way excursion and the outcome.

    python tools/diagnostics/swerve_timing.py --model runs/ppo_formulation_seed0_v10/final_model.zip --tag run10
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tag", default="run10")
    args = ap.parse_args()
    import curriculum
    import train_formulation as tf
    from common import load_model
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    model = load_model(str(args.model if args.model.is_absolute() else ROOT / args.model))
    rows, traces = [], []
    for i, built in enumerate(tf.development_set(20)):
        if built.encounter_class != "crossing" or built.ct_deg < 180.0:
            continue
        obs, _ = env.reset(seed=900_000 + i, options={"generated": built})
        actor = tf.EpisodeActor(model)
        h0, first, step = float(env.asv_h), None, 0
        while True:
            ctx = next(iter(env.encounter_contexts.values()), None)
            t = env.targets[0]
            rng = math.hypot(t.x - env.asv_x, t.y - env.asv_y)
            brg = _wrap(math.degrees(math.atan2(t.x - env.asv_x, t.y - env.asv_y)) - env.asv_h)
            dpsi = _wrap(float(env.asv_h) - h0)
            rec = {"scenario": i, "t": step * 0.5, "dpsi": dpsi, "tracked": len(env.tracks),
                   "state": ctx.state if ctx else "none", "cls": str(ctx.cls) if ctx else "",
                   "sense": int(ctx.compliant_turn_sense) if ctx else 0,
                   "tcpa": float(ctx.tcpa) if ctx else np.nan, "range": rng, "bearing": brg,
                   "panels_near": sum(1 for p in env.obstacles
                                      if min(math.hypot(x - env.asv_x, y - env.asv_y) for x, y in p) < 3.0)}
            traces.append(rec)
            if first is None and abs(dpsi) > 10.0:
                first = rec
            obs, _, term, trunc, info = env.step(actor(obs))
            step += 1
            if term or trunc:
                break
        tr = pd.DataFrame([r for r in traces if r["scenario"] == i])
        outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
                   "goal" if info["reached_goal"] else "timeout")
        rows.append({"scenario": i, "outcome": outcome,
                     "first_turn_t": first["t"] if first else np.nan,
                     "first_turn_dir": ("port" if first["dpsi"] < 0 else "stbd") if first else "",
                     "tracked_at_turn": first["tracked"] if first else np.nan,
                     "state_at_turn": first["state"] if first else "",
                     "cls_at_turn": first["cls"] if first else "",
                     "range_at_turn": round(first["range"], 2) if first else np.nan,
                     "bearing_at_turn": round(first["bearing"], 1) if first else np.nan,
                     "panels_within_3m": first["panels_near"] if first else np.nan,
                     "first_tracked_t": float(tr[tr.tracked > 0].t.min()) if (tr.tracked > 0).any() else np.nan,
                     "first_engaged_t": float(tr[tr.state == "engaged"].t.min()) if (tr.state == "engaged").any() else np.nan,
                     "peak_port_deg": round(float(-tr.dpsi.min()), 1)})
    d = pd.DataFrame(rows)
    out = ROOT / "results" / "swerve_timing"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / f"summary_{args.tag}.csv", index=False)
    pd.DataFrame(traces).to_csv(out / f"traces_{args.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
