"""Snapshots of the three Paper 2 layouts with and without traffic, driven by a policy.

    python tools/diagnostics/paper2_snapshots.py
    python tools/diagnostics/paper2_snapshots.py --model runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip --case 01

One figure, layouts L1-L3 across, rows no target / head-on / crossing from port /
crossing from starboard / overtaking / being overtaken (the FIX form of case
`--case`).  Own-ship track in blue, target in red, both marked every 5 s;
outcome in each title.  Writes `results/paper2_snapshots/<tag>_case<NN>.png`.
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]
import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import paper2_set as p2  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402
from common import load_model  # noqa: E402

ROWS = (("NT", "No target"), ("HO", "Head-on"), ("CRP", "Crossing from port"),
        ("CRS", "Crossing from starboard"), ("OT", "Overtaking"), ("BO", "Being overtaken"))
COLOURS = {"goal": "#1a7f37", "collision:target": "#cf222e", "collision:obstacle": "#bc4c00",
           "collision:boundary": "#8250df", "timeout": "#6e7781"}


def rollout(env, model, built, seed):
    actor = tf.EpisodeActor(model)
    obs, _ = env.reset(seed=seed, options={"generated": built})
    own, tgt = [(env.asv_x, env.asv_y)], [[(t.x, t.y)] for t in env.targets]
    while True:
        obs, _, term, trunc, info = env.step(actor(obs))
        own.append((env.asv_x, env.asv_y))
        for k, t in enumerate(env.targets):
            tgt[k].append((t.x, t.y))
        if term or trunc:
            break
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    return own, tgt, outcome, env.hull_polygon()


def draw(ax, layout, own, tgt, outcome, hull, title):
    ax.add_patch(Rectangle((0, 0), cfg.MAP_WIDTH, cfg.MAP_HEIGHT, fill=False, lw=1.2, ec="#24292f"))
    inset = cfg.BASIN_NAV_INSET_M
    ax.add_patch(Rectangle((inset, inset), cfg.MAP_WIDTH - 2 * inset, cfg.MAP_HEIGHT - 2 * inset,
                           fill=False, lw=0.8, ls="--", ec="#8c959f"))
    for poly in p2.panel_polygons(layout):
        ax.add_patch(Polygon(poly, closed=True, fc="#57606a", ec="#24292f"))
    lay = p2.LAYOUTS[layout]
    (sx, sy), (gx, gy) = lay["start"], lay["goal"]
    ax.plot([sx, gx], [sy, gy], color="#8c959f", lw=0.8, ls=":")
    ax.plot(sx, sy, "o", ms=4, color="#24292f")
    ax.plot(gx, gy, "*", ms=9, color="#bf8700")
    every = int(round(5.0 / cfg.UPDATE_RATE))
    xs, ys = zip(*own)
    ax.plot(xs, ys, color="#0969da", lw=1.6)
    ax.plot(xs[::every], ys[::every], "o", ms=2.5, color="#0969da")
    for track in tgt:
        tx, ty = zip(*track)
        ax.plot(tx, ty, color="#cf222e", lw=1.4)
        ax.plot(tx[::every], ty[::every], "o", ms=2.5, color="#cf222e")
        ax.annotate("", xy=(tx[-1], ty[-1]), xytext=(tx[-2], ty[-2]),
                    arrowprops=dict(arrowstyle="->", color="#cf222e", lw=1.2))
    ax.add_patch(Polygon(hull, closed=True, fc="#0969da", ec="#0550ae", alpha=0.8))
    ax.set_title(f"{title}\n{outcome.replace('collision:', 'collision: ')}", fontsize=8,
                 color=COLOURS.get(outcome, "#24292f"))
    ax.set_xlim(-0.3, cfg.MAP_WIDTH + 0.3)
    ax.set_ylim(-0.3, cfg.MAP_HEIGHT + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.tick_params(labelsize=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path,
                    default=ROOT / "runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip")
    ap.add_argument("--case", default="01")
    ap.add_argument("--tag", default="sacs0_bl3_3M")
    args = ap.parse_args()
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    records, _ = p2.build(env)
    by_id = {r["test_id"]: r for r in records}
    model = load_model(args.model if args.model.is_absolute() else ROOT / args.model)
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(7.2, 16.5))
    for c, layout in enumerate(("L1", "L2", "L3")):
        for r, (code, name) in enumerate(ROWS):
            tid = f"P2-{layout}-NT-{args.case}" if code == "NT" else f"P2-{layout}-{code}-FIX-{args.case}"
            rec = by_id[tid]
            own, tgt, outcome, hull = rollout(env, model, rec["built"], rec["episode_seed"])
            draw(axes[r, c], layout, own, tgt, outcome, hull, f"{layout} · {name}")
            print(tid, outcome, flush=True)
    fig.suptitle(f"Paper 2 layouts L1-L3, {args.tag} (case {args.case}; own ship blue, target red, "
                 f"dots every 5 s)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = ROOT / "results" / "paper2_snapshots"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.tag}_case{args.case}.png"
    fig.savefig(path, dpi=150)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
