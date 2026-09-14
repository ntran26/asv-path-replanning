"""A18 + A19 end to end: replay the C14 head-on set under the current code.

The same 100 head-on scenarios as `c14_*` (20 per width at 5, 6, 7, 8, 10 m,
development seeds), obstacles off, nominal hull.  Reports, per width:

* target-collision rate, stops and minimum speed with the run 2 model -- against
  `c14_estop.csv`'s supervisor-on column (0.43, 0.25, 0.24, 0.16, 0.05);
* how often a head-on latches a **port** turn sense, for the model and a
  non-avoiding path follower -- against `c14_sense.csv` (0.67 and 0.20).

The run 2 model was trained under the old supervisor and classifier, so this
measures the mechanism, not a retrained policy.

    python tools/diagnostics/a18_a19_verify.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import curriculum  # noqa: E402
import scenario as scn  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

OUT = ROOT / "results" / "c14_narrow_head_on"
MODEL_CANDIDATES = (ROOT / "runs" / "ppo_formulation_seed0_v2" / "final_model.zip",)
WIDTHS = (5.0, 6.0, 7.0, 8.0, 10.0)
BEFORE_COLLISION = {"5": 0.43, "6": 0.25, "7": 0.24, "8": 0.16, "10": 0.05}


def scenarios():
    gen = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    out = []
    for w in WIDTHS:
        j = found = 0
        while found < 20 and j < 400:
            b = gen.sample(scn.seed_for("development", 300_000 + int(w * 10) * 1000 + j),
                           encounter_class="head_on", width=w)
            j += 1
            if b is not None:
                out.append(b)
                found += 1
    return out


def follower(env):
    rudder = float(np.clip(env.lookahead_course_error / 25.0 + 0.4 * env.cross_track_error, -1.0, 1.0))
    return np.array([rudder, 0.0], dtype=np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    model_path = next(p for p in MODEL_CANDIDATES if p.exists())
    model = PPO.load(model_path, device="cpu")
    scen = scenarios()
    env = ASVLidarEnv(render_mode=None)
    rows = []
    for policy in ("model", "follower"):
        for i, b in enumerate(scen):
            env.forced_num_obs = 0
            obs, _ = env.reset(seed=700_000 + i, options={"generated": b})
            speeds, engaged, port_frames, ever_port = [], 0, 0, False
            while True:
                action = model.predict(obs, deterministic=True)[0] if policy == "model" else follower(env)
                obs, _, term, trunc, info = env.step(action)
                speeds.append(info["speed_mps"])
                for ctx in env.encounter_contexts.values():
                    if ctx.engaged and ctx.tcpa > 0.0:
                        engaged += 1
                        if ctx.compliant_turn_sense < 0:
                            port_frames += 1
                            ever_port = True
                if term or trunc:
                    break
            rows.append(dict(policy=policy, idx=i, width=b.nominal_width,
                             coll_target=info["collision_kind"] == "target",
                             outcome=info["collision_kind"] or ("goal" if info["reached_goal"] else "timeout"),
                             estops=int(info["estop/events"]), u_min=min(speeds),
                             ever_port_sense=ever_port, p_port_sense=port_frames / max(engaged, 1)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "a18_a19_verify.csv", index=False)
    d["wbin"] = pd.cut(d.width, [0, 5.5, 6.5, 7.5, 9.0, 11.0], labels=["5", "6", "7", "8", "10"])
    pd.set_option("display.width", 250)
    print(f"model: {model_path}")
    m = d[d.policy == "model"].groupby("wbin", observed=True).agg(
        coll_target=("coll_target", "mean"), estops=("estops", "mean"), u_min=("u_min", "median"),
        ever_port_sense=("ever_port_sense", "mean"), p_port_sense=("p_port_sense", "mean")).round(2)
    m.insert(0, "coll_before", pd.Series(BEFORE_COLLISION))
    print("== run 2 model under A18 + A19 (coll_before = old supervisor, old classifier)")
    print(m)
    f = d[d.policy == "follower"].groupby("wbin", observed=True).agg(
        coll_target=("coll_target", "mean"), ever_port_sense=("ever_port_sense", "mean"),
        p_port_sense=("p_port_sense", "mean")).round(2)
    print("\n== path follower under A18 + A19")
    print(f)
    print("\noverall: model ever port sense", round(d[d.policy == "model"].ever_port_sense.mean(), 2),
          "(was 0.67); follower", round(d[d.policy == "follower"].ever_port_sense.mean(), 2), "(was 0.20)")


if __name__ == "__main__":
    main()
