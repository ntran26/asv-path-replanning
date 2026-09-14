"""C14: why narrow head-on collides.  Hypothesis: `clamp_to_corridor` teleports
a confined target onto the channel centreline when its hull touches the wall,
which happens early on the bends that only narrow corridors have (F25).

2 x 2: policy (scripted path follower with no avoidance, run 2 final model)
x clamp (as built, disabled).  Head-on only, obstacles off, nominal hull.
"""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import scenario as scn  # noqa: E402
import targets as tgtmod  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

OUT = ROOT / "results" / "c14_narrow_head_on"
WIDTHS = (5.0, 6.0, 7.0, 8.0, 10.0)
PER_WIDTH = 20

curriculum.apply_stage(tf.PROPULSION_STAGE)
MODEL = PPO.load(ROOT / "runs" / "ppo_formulation_seed0_v2" / "final_model.zip", device="cpu")

ORIGINAL_CLAMP = tgtmod.clamp_to_corridor
CLAMP_ON = [True]
STEP = [0]
SNAPS = []


def wrapped_clamp(target, corridor, polygon=None):
    if not CLAMP_ON[0]:
        return
    x0, y0, h0 = target.x, target.y, target.heading
    ORIGINAL_CLAMP(target, corridor, polygon)
    if (target.x, target.y) != (x0, y0):
        dh = ((target.heading - h0 + 180.0) % 360.0) - 180.0
        SNAPS.append({"step": STEP[0], "jump": math.hypot(target.x - x0, target.y - y0), "dh": dh})


tgtmod.clamp_to_corridor = wrapped_clamp


def follower(env):
    rudder = float(np.clip(env.lookahead_course_error / 25.0 + 0.4 * env.cross_track_error, -1.0, 1.0))
    return np.array([rudder, 0.0], dtype=np.float32)


def centre_distance(channel, x, y):
    d = channel.centre - np.array([x, y])
    return float(np.sqrt(np.min(np.einsum("ij,ij->i", d, d))))


def run(env, built, seed, policy):
    SNAPS.clear()
    env.forced_num_obs = 0
    obs, _ = env.reset(seed=seed, options={"generated": built})
    t0 = env.targets[0]
    spawn_off_centre = centre_distance(built.channel, t0.x, t0.y)
    min_rng, lat_at_min, classes, step = float("inf"), 0.0, Counter(), 0
    while True:
        STEP[0] = step
        action = follower(env) if policy == "follower" else MODEL.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(action)
        step += 1
        tgt = env.targets[0]
        dx, dy = tgt.x - env.asv_x, tgt.y - env.asv_y
        rng = math.hypot(dx, dy)
        if rng < min_rng:
            h = math.radians(env.asv_h)
            min_rng = rng
            lat_at_min = dx * math.cos(h) - dy * math.sin(h)   # + = target to own starboard
        if rng < 8.0:
            for ctx in env.encounter_contexts.values():
                classes[str(ctx.cls)] += 1
        if term or trunc:
            break
    coll_step = step if info["collided"] else None
    last_snap = max((s["step"] for s in SNAPS), default=None)
    n_cls = sum(classes.values())
    return {
        "width": built.nominal_width, "bend": built.bend_deg, "has_bend": abs(built.bend_deg) >= cfg.CORRIDOR_BEND_MIN_DEG,
        "dcpa_drawn": built.dcpa_m, "offset_frac": built.path_offset_frac,
        "spawn_off_centre": spawn_off_centre,
        "outcome": (f"collision:{info['collision_kind']}" if info["collided"] else
                    "goal" if info["reached_goal"] else "timeout"),
        "collided_target": info["collision_kind"] == "target",
        "steps": step, "min_range": min_rng, "lat_at_min": lat_at_min,
        "snaps": len(SNAPS), "first_snap_step": min((s["step"] for s in SNAPS), default=None),
        "max_jump": max((s["jump"] for s in SNAPS), default=0.0),
        "max_dh": max((abs(s["dh"]) for s in SNAPS), default=0.0),
        "coll_within_3s_of_snap": bool(coll_step is not None and last_snap is not None
                                       and coll_step - last_snap <= cfg.steps_for(3.0)),
        "frac_not_head_on_near": (1.0 - classes.get("head_on", 0) / n_cls) if n_cls else float("nan"),
    }


def main():
    gen = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    scenarios = []
    for w in WIDTHS:
        j = 0
        found = 0
        while found < PER_WIDTH and j < 20 * PER_WIDTH:
            built = gen.sample(scn.seed_for("development", 300_000 + int(w * 10) * 1000 + j),
                               encounter_class="head_on", width=w)
            j += 1
            if built is not None:
                scenarios.append(built)
                found += 1
    print(f"{len(scenarios)} head-on scenarios", flush=True)

    env = ASVLidarEnv(render_mode=None)
    rows = []
    for clamp in (True, False):
        CLAMP_ON[0] = clamp
        for policy in ("follower", "model"):
            for i, built in enumerate(scenarios):
                row = run(env, built, 700_000 + i, policy)
                row.update({"clamp": "on" if clamp else "off", "policy": policy, "idx": i})
                rows.append(row)
            print(f"done clamp={clamp} policy={policy}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "c14_diag.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    df["wbin"] = pd.cut(df.width, [0, 5.5, 6.5, 7.5, 9.0, 11.0], labels=["5", "6", "7", "8", "10"])

    print("\n== bend share and snap stats by width (clamp on, follower)")
    base = df[(df.clamp == "on") & (df.policy == "follower")]
    print(base.groupby("wbin", observed=True).agg(n=("idx", "size"), bent=("has_bend", "mean"),
                                                   mean_bend=("bend", lambda s: s.abs().mean()),
                                                   ep_with_snap=("snaps", lambda s: (s > 0).mean()),
                                                   max_jump=("max_jump", "median"),
                                                   first_snap_s=("first_snap_step", lambda s: s.dropna().median() * cfg.UPDATE_RATE),
                                                   spawn_off_centre=("spawn_off_centre", "median")).round(2))

    print("\n== target-collision rate: width x clamp x policy")
    print(df.pivot_table(index="wbin", columns=["policy", "clamp"], values="collided_target", aggfunc="mean", observed=True).round(2))

    print("\n== target-collision rate: bend x clamp x policy")
    print(df.pivot_table(index="has_bend", columns=["policy", "clamp"], values="collided_target", aggfunc=["mean", "size"], observed=True).round(2))

    print("\n== among target collisions with clamp on: share within 3 s of a snap")
    c = df[(df.clamp == "on") & df.collided_target]
    print(c.groupby(["policy", "wbin"], observed=True).coll_within_3s_of_snap.agg(["mean", "size"]).round(2))

    print("\n== perceived class not head-on (target within 8 m), clamp on vs off")
    print(df.pivot_table(index="wbin", columns=["clamp"], values="frac_not_head_on_near", aggfunc="mean", observed=True).round(2))

    print("\n== outcomes (all)")
    print(pd.crosstab([df.policy, df.clamp, df.wbin], df.outcome))


if __name__ == "__main__":
    main()
