"""A19 residual: why head-on encounters still latch a port sense under a
path-anchored classifier.

After A19 a non-avoiding path follower still latches a port turn sense in 24 %
of the C14 head-on episodes (`a18_a19_verify.csv`).  Its own manoeuvre cannot
be the cause, so this logs the geometry on those frames, measured against the
path tangent the classifier now uses: the heading-intersection angle, the
relative bearing, range, the corridor bend, and whether the encounter was first
classified head-on at all.  The target's heading is logged too, because a
confined target nudged along a bent channel changes heading.

    python tools/diagnostics/a19_residual.py
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
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

OUT = ROOT / "results" / "c14_narrow_head_on"
WIDTHS = (5.0, 6.0, 7.0, 8.0, 10.0)


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


def wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def follower(env):
    rudder = float(np.clip(env.lookahead_course_error / 25.0 + 0.4 * env.cross_track_error, -1.0, 1.0))
    return np.array([rudder, 0.0], dtype=np.float32)


def path_heading(env):
    idx = int(np.clip(np.searchsorted(env.path.s, float(env.s_along)), 0, len(env.path.points) - 1))
    t = env.path.tangent(idx)
    return math.degrees(math.atan2(float(t[0]), float(t[1])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None)
    frames, episodes = [], []
    for i, b in enumerate(scenarios()):
        env.forced_num_obs = 0
        env.reset(seed=700_000 + i, options={"generated": b})
        tgt0 = env.targets[0]
        first_cls, classes, step, port = None, Counter(), 0, 0
        spawn_ct_path = wrap180(tgt0.heading - path_heading(env) - 180.0)
        while True:
            _, _, term, trunc, info = env.step(follower(env))
            step += 1
            ph = path_heading(env)
            tgt = env.targets[0]
            for ctx in env.encounter_contexts.values():
                if not (ctx.engaged and ctx.tcpa > 0.0):
                    continue
                first_cls = first_cls or str(ctx.cls)
                classes[str(ctx.cls)] += 1
                if ctx.compliant_turn_sense < 0:
                    port += 1
                    dx, dy = tgt.x - env.asv_x, tgt.y - env.asv_y
                    frames.append(dict(
                        idx=i, width=b.nominal_width, bend=b.bend_deg, step=step,
                        cls=str(ctx.cls), side=str(ctx.crossing_side), rng=math.hypot(dx, dy),
                        # perceived, as the classifier saw it, relative to the path
                        ct_dev_perceived=wrap180(ctx.ct + (env.asv_h - ph) - 180.0),
                        # truth
                        ct_dev_true=wrap180(tgt.heading - ph - 180.0),
                        tgt_heading_change=wrap180(tgt.heading - tgt0.heading),
                        path_turn_since_start=wrap180(ph - b.own_heading),
                        alpha_path=wrap180(math.degrees(math.atan2(dx, dy)) - ph)))
            if term or trunc:
                break
        episodes.append(dict(idx=i, width=b.nominal_width, bend=b.bend_deg, ct_drawn=b.ct_deg,
                             spawn_ct_dev=spawn_ct_path, first_cls=first_cls or "never",
                             port_frames=port, ever_port=port > 0,
                             p_head_on=classes.get("head_on", 0) / max(sum(classes.values()), 1)))
    f = pd.DataFrame(frames)
    e = pd.DataFrame(episodes)
    f.to_csv(OUT / "a19_residual_frames.csv", index=False)
    e.to_csv(OUT / "a19_residual_episodes.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    e["bent"] = e.bend.abs() >= cfg.CORRIDOR_BEND_MIN_DEG
    e["wbin"] = pd.cut(e.width, [0, 5.5, 6.5, 7.5, 9.0, 11.0], labels=["5", "6", "7", "8", "10"])
    print("== follower: episodes that latch a port sense, by width x bend")
    print(e.pivot_table(index="wbin", columns="bent", values="ever_port", aggfunc=["mean", "size"], observed=True).round(2))
    print("\n== by first engaged class")
    print(e.groupby("first_cls").agg(n=("idx", "size"), ever_port=("ever_port", "mean")).round(2))
    print("\n== drawn CT deviation from reciprocal (|ct_drawn - 180|) vs ever_port")
    e["ct_dev_drawn"] = (e.ct_drawn - 180.0).abs()
    print(e.groupby(pd.cut(e.ct_dev_drawn, [-0.1, 3, 6, 8, 10.5]), observed=True).agg(
        n=("idx", "size"), ever_port=("ever_port", "mean"), first_head_on=("first_cls", lambda s: (s == "head_on").mean())).round(2))
    if len(f):
        print("\n== port-sense frames: geometry relative to the path (medians of |values|)")
        g = f.assign(abs_ct_p=f.ct_dev_perceived.abs(), abs_ct_t=f.ct_dev_true.abs(),
                     abs_tgt_turn=f.tgt_heading_change.abs(), abs_path_turn=f.path_turn_since_start.abs(),
                     abs_alpha=f.alpha_path.abs())
        print(g.groupby("cls").agg(frames=("idx", "size"), episodes=("idx", "nunique"), rng=("rng", "median"),
                                   ct_dev_perceived=("abs_ct_p", "median"), ct_dev_true=("abs_ct_t", "median"),
                                   tgt_heading_change=("abs_tgt_turn", "median"),
                                   path_turn=("abs_path_turn", "median"), alpha_path=("abs_alpha", "median")).round(1))
        print("\nshare of port-sense frames where |true ct dev| <= 10 deg (a truly head-on geometry):",
              round(float((f.ct_dev_true.abs() <= 10.0).mean()), 2))
        print("share where |perceived - true ct dev| > 5 deg (tracker course error):",
              round(float(((f.ct_dev_perceived - f.ct_dev_true).abs() > 5.0).mean()), 2))


if __name__ == "__main__":
    main()
