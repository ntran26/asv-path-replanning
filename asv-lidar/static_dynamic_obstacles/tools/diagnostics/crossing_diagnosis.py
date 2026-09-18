"""Why do learned policies fail crossings that a classical planner solves?  (F75)

Run 6 reached the goal on 0.40 of development crossings; the reference
controller on 0.70 (F71, F73). This replays the development crossings -- now
basin-default -- under a PPO model and the reference controller, logging every
decision, and reduces each episode to the quantities that separate a good
crossing from a collision:

* **when** each engages (TCPA at engagement) and first alters course (TCPA when
  the heading change from the start first exceeds 10 deg);
* **which way** (the first alteration against the latched compliant sense);
* **how much** (peak heading change) and **how slow** (minimum speed, speed at
  the closest point);
* the outcome and closest range.

    python tools/diagnostics/crossing_diagnosis.py --model runs/ppo_formulation_seed0_v6/final_model.zip --tag run6
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
OUT = ROOT / "results" / "crossing_diagnosis"
_W = {}


def _init(model_path):
    import curriculum
    import train_formulation as tf
    from common import load_model
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)
    dev = tf.development_set(20)
    _W["cases"] = [(i, b) for i, b in enumerate(dev) if b.encounter_class == "crossing"]
    _W["model"] = load_model(model_path) if model_path else None


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _episode(job):
    from reference_controller import ReferenceController
    k, policy = job
    i, built = _W["cases"][k]
    env = _W["env"]
    obs, _ = env.reset(seed=900_000 + i, options={"generated": built})
    ref = ReferenceController() if policy == "reference" else None
    import train_formulation as tf
    actor = tf.EpisodeActor(_W["model"]) if ref is None else None
    h0 = float(env.asv_h)
    rows = []
    while True:
        ctx = next(iter(env.encounter_contexts.values()), None)
        action = ref.action(env, obs) if ref is not None else actor(obs)
        target = env.targets[0] if env.targets else None
        rows.append({
            "t": env.step_count * 0.5, "dpsi": _wrap(float(env.asv_h) - h0),
            "u": float(env.u_body),
            "range": (math.hypot(target.x - env.asv_x, target.y - env.asv_y) if target else np.nan),
            "tcpa": float(ctx.tcpa) if ctx is not None else np.nan,
            "dcpa": float(ctx.dcpa) if ctx is not None else np.nan,
            "state": str(ctx.state) if ctx is not None else "none",
            "sense": int(ctx.compliant_turn_sense) if ctx is not None else 0,
            "rudder": float(action[0]), "throttle": float(action[1]),
        })
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    d = pd.DataFrame(rows)
    engaged = d[d.state == "engaged"]
    sense = int(engaged.sense.iloc[0]) if len(engaged) else 0
    altered = d[d.dpsi.abs() > 10.0]
    first = altered.iloc[0] if len(altered) else None
    closest = d.loc[d.range.idxmin()] if d.range.notna().any() else None
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    ct = float(getattr(built, "ct_deg", 0.0))
    return {
        "scenario": i, "policy": policy, "outcome": outcome, "mode": built.geometry_mode,
        "side": "port" if ct < 180.0 else "starboard", "dcpa_drawn": float(built.dcpa_m),
        "tcpa_drawn": float(built.tcpa_s), "sense": sense,
        "tcpa_at_engage": float(engaged.tcpa.iloc[0]) if len(engaged) else np.nan,
        "tcpa_at_first_turn": float(first.tcpa) if first is not None else np.nan,
        "first_turn_compliant": (bool(np.sign(first.dpsi) == sense) if first is not None and sense
                                 else None),
        "peak_dpsi_compliant": float((d.dpsi * (sense or 1)).max()),
        "peak_dpsi_wrong": float((-d.dpsi * (sense or 1)).max()),
        "min_speed": float(d.u.min()),
        "speed_at_closest": float(closest.u) if closest is not None else np.nan,
        "min_range": float(d.range.min()),
        "mean_throttle_engaged": float(engaged.throttle.mean()) if len(engaged) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tag", default="run6")
    ap.add_argument("--processes", type=int, default=10)
    args = ap.parse_args()
    model = args.model if args.model.is_absolute() else ROOT / args.model
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jobs = [(k, p) for p in ("model", "reference") for k in range(20)]
    with ProcessPoolExecutor(args.processes, initializer=_init, initargs=(str(model),)) as pool:
        d = pd.DataFrame(list(pool.map(_episode, jobs)))
    d["policy"] = d.policy.replace({"model": args.tag})
    d.to_csv(OUT / f"episodes_{args.tag}.csv", index=False)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    d["goal"] = d.outcome == "goal"
    summary = d.groupby("policy").agg(
        goal=("goal", "mean"), tcpa_engage=("tcpa_at_engage", "median"),
        tcpa_first_turn=("tcpa_at_first_turn", "median"),
        turned=("tcpa_at_first_turn", lambda s: s.notna().mean()),
        compliant_first=("first_turn_compliant", lambda s: s.dropna().astype(bool).mean()),
        peak_turn=("peak_dpsi_compliant", "median"), wrong_turn=("peak_dpsi_wrong", "median"),
        min_speed=("min_speed", "median"), speed_at_cpa=("speed_at_closest", "median"),
        min_range=("min_range", "median")).round(2)
    wide = d.pivot(index="scenario", columns="policy", values="goal")
    paired = wide.value_counts().rename("scenarios")
    by_outcome = d.groupby(["policy", "goal"]).agg(
        n=("goal", "size"), tcpa_first_turn=("tcpa_at_first_turn", "median"),
        peak_turn=("peak_dpsi_compliant", "median"), speed_at_cpa=("speed_at_closest", "median"),
        compliant_first=("first_turn_compliant", lambda s: s.dropna().astype(bool).mean())).round(2)
    text = (f"Development crossings ({len(d) // 2}), {args.tag} vs reference, supervisor off "
            f"({time.time() - started:.0f} s)\n\n== summary (medians)\n{summary}\n\n"
            f"== paired outcome (goal?)\n{paired}\n\n== by outcome\n{by_outcome}\n")
    (OUT / f"summary_{args.tag}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
