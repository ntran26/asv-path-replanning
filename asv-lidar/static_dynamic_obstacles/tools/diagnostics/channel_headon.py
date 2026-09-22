"""Why did head-on in channels regress after A27/A29/F88?  (F91)

Head-on scores 1.00 on the development set (mostly basin) while the channel
head-on width set worsened from 0-0.05 target collisions (run 8) to 0.10-0.25
(run 11, both seeds). This replays the width set under several models and
reports, per width: the outcome, whether the compliant starboard alteration was
admissible, the peak alteration in each sense, the speed at closest approach,
and where the encounter stood when the vessel first altered.

    python tools/diagnostics/channel_headon.py --models runs/ppo_formulation_seed0_v8/final_model.zip=run8 ...
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
    ap.add_argument("--models", nargs="+", required=True, help="path=tag pairs")
    args = ap.parse_args()
    import curriculum
    import train_formulation as tf
    from common import head_on_width_set, load_model, width_bin
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    cases = head_on_width_set()
    rows = []
    for spec in args.models:
        path, tag = spec.split("=")
        model = load_model(str(ROOT / path))
        for i, built in enumerate(cases):
            env.forced_num_obs = 0
            obs, _ = env.reset(seed=700_000 + i, options={"generated": built})
            actor = tf.EpisodeActor(model)
            h0, peak_s, peak_p, first = float(env.asv_h), 0.0, 0.0, None
            a_stbd_seen, min_rng, u_at_min = None, float("inf"), np.nan
            while True:
                ctx = next(iter(env.encounter_contexts.values()), None)
                if ctx is not None and ctx.engaged and a_stbd_seen is None:
                    a_stbd_seen = bool(ctx.a_stbd)
                obs, _, term, trunc, info = env.step(actor(obs))
                dpsi = _wrap(float(env.asv_h) - h0)
                peak_s, peak_p = max(peak_s, dpsi), max(peak_p, -dpsi)
                if first is None and abs(dpsi) > 10.0:
                    first = {"t": env.step_count * 0.5, "dir": "stbd" if dpsi > 0 else "port",
                             "state": ctx.state if ctx else "none"}
                t = env.targets[0]
                rng = math.hypot(t.x - env.asv_x, t.y - env.asv_y)
                if rng < min_rng:
                    min_rng, u_at_min = rng, float(info["speed_mps"])
                if term or trunc:
                    break
            rows.append({"tag": tag, "width": width_bin(built.nominal_width),
                         "outcome": (f"collision:{info['collision_kind']}" if info["collided"]
                                     else "goal" if info["reached_goal"] else "timeout"),
                         "stbd_admissible": a_stbd_seen, "peak_stbd": peak_s, "peak_port": peak_p,
                         "first_turn_t": first["t"] if first else np.nan,
                         "first_turn_dir": first["dir"] if first else "",
                         "min_range": min_rng, "speed_at_min": u_at_min})
    d = pd.DataFrame(rows)
    out = ROOT / "results" / "channel_headon"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "episodes.csv", index=False)
    pd.set_option("display.width", 220)
    table = d.groupby(["tag", "width"]).agg(
        n=("outcome", "size"),
        target_hit=("outcome", lambda s: (s == "collision:target").mean()),
        other_hit=("outcome", lambda s: s.isin(["collision:boundary", "collision:obstacle"]).mean()),
        stbd_admissible=("stbd_admissible", lambda s: s.dropna().mean()),
        peak_stbd=("peak_stbd", "median"), peak_port=("peak_port", "median"),
        turned=("first_turn_t", lambda s: s.notna().mean()),
        first_turn_t=("first_turn_t", "median"),
        stbd_first=("first_turn_dir", lambda s: (s == "stbd").mean()),
        min_range=("min_range", "median"), speed_at_min=("speed_at_min", "median")).round(2)
    (out / "summary.txt").write_text(table.to_string(), encoding="utf-8")
    print(table.to_string())


if __name__ == "__main__":
    main()
