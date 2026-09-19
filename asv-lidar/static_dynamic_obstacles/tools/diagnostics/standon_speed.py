"""Why do the policies outrun an overtaker?  (A28 option 3)

Run 7 and run 8 accelerate when being overtaken (max 0.94 and 0.80 m/s against a
0.558 m/s cruise), and the stand-on `v_hold` integral grew to 8.7-10.3 (F79,
F82). Rule 17(a)(i) asks the stand-on vessel to keep its course and speed. This
replays the 20 development being-overtaken episodes under a model, logs every
decision, and splits the reward by phase:

* **engaged** -- the being-overtaken context is ENGAGED (stand-on applies);
* **other**   -- everything else in the episode.

For each phase: speed, throttle, the COLREGs weighted term (where `v_hold`
lives), the path term (whose overspeed gate charges speed above
`U_REF * (1 + PF_OVERSPEED_TOL)`), the progress term (which pays metres along
the path, so faster pays more per step), and the per-step total. It also
counts the steps above the gate's tolerance and the steps `v_hold` fires, and
records how each episode ended and how close the overtaker came.

    python tools/diagnostics/standon_speed.py --model runs/ppo_formulation_seed0_v8/final_model.zip --tag run8
"""
import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))
OUT = ROOT / "results" / "standon_speed"
_W = {}


def _init(model_path):
    import curriculum
    import train_formulation as tf
    from common import load_model
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)
    _W["cases"] = [(i, b) for i, b in enumerate(tf.development_set(20))
                   if b.encounter_class == "being_overtaken"]
    _W["model"] = load_model(model_path)


def _episode(k):
    import constants as cfg
    import train_formulation as tf
    i, built = _W["cases"][k]
    env = _W["env"]
    obs, _ = env.reset(seed=900_000 + i, options={"generated": built})
    actor = tf.EpisodeActor(_W["model"])
    gate = cfg.U_REF * (1.0 + cfg.PF_OVERSPEED_TOL)
    rows, min_rng = [], float("inf")
    while True:
        engaged = any(c.engaged and str(c.cls) == "being_overtaken"
                      for c in env.encounter_contexts.values())
        action = actor(obs)
        obs, reward, term, trunc, info = env.step(action)
        t = env.targets[0]
        min_rng = min(min_rng, math.hypot(t.x - env.asv_x, t.y - env.asv_y))
        rows.append({"phase": "engaged" if engaged else "other", "u": float(info["speed_mps"]),
                     "throttle": float(action[1]), "reward": float(reward),
                     "col": float(info["reward/weighted/col"]), "pf": float(info["reward/weighted/pf"]),
                     "prog": float(info["reward/weighted/prog"]),
                     "exist": float(info["reward/weighted/exist"]),
                     "v_hold": float(info.get("colregs/v_hold", 0.0)),
                     "over_gate": float(info["speed_mps"]) > gate})
        if term or trunc:
            break
    d = pd.DataFrame(rows)
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    out = {"scenario": i, "outcome": outcome, "k": float(built.speed_ratio),
           "dcpa_drawn": float(built.dcpa_m), "min_range": min_rng, "steps": len(d),
           "max_u": float(d.u.max())}
    for phase in ("engaged", "other"):
        x = d[d.phase == phase]
        out[f"{phase}_steps"] = len(x)
        for c in ("u", "throttle", "reward", "col", "pf", "prog", "exist"):
            out[f"{phase}_{c}"] = float(x[c].mean()) if len(x) else np.nan
        out[f"{phase}_over_gate"] = float(x.over_gate.mean()) if len(x) else np.nan
        out[f"{phase}_v_hold_fires"] = float((x.v_hold > 0).mean()) if len(x) else np.nan
        out[f"{phase}_v_hold_mean"] = float(x.v_hold.mean()) if len(x) else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tag", default="run8")
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    model = args.model if args.model.is_absolute() else ROOT / args.model
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with ProcessPoolExecutor(args.processes, initializer=_init, initargs=(str(model),)) as pool:
        d = pd.DataFrame(list(pool.map(_episode, range(20))))
    d.to_csv(OUT / f"episodes_{args.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    cols = ["steps", "u", "throttle", "over_gate", "v_hold_fires", "v_hold_mean",
            "col", "pf", "prog", "exist", "reward"]
    table = pd.DataFrame({ph: [d[f"{ph}_{c}"].mean() for c in cols] for ph in ("engaged", "other")},
                         index=cols).round(3)
    text = (f"Being-overtaken development episodes, {args.tag}, supervisor off "
            f"({time.time() - started:.0f} s)\n"
            f"outcomes {d.outcome.value_counts().to_dict()}; max speed mean {d.max_u.mean():.2f} m/s; "
            f"min range to overtaker median {d.min_range.median():.2f} m\n\n"
            f"== per-step means by phase (weighted reward terms)\n{table}\n")
    (OUT / f"summary_{args.tag}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
