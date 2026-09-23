"""Trajectory figures for the paper: what the policy actually did, per encounter.

For each encounter class it replays development-set episodes with a trained
model, keeps the most representative one, and draws the track: the basin or
channel, the static panels, the reference leg, the own ship's path coloured by
speed, the target's path, both hulls at the closest point of approach, and the
stretch where the encounter was engaged (the window the COLREGs terms judge).

    python tools/diagnostics/paper_paths.py --model runs/ppo_formulation_seed0_bl2/best_model.zip --tag ppos0_bl2

Writes `results/paper_paths/<tag>/`: one PNG and one PDF (vector, for the
paper) per episode, a six-panel sheet, and `episodes.csv` with the numbers each
caption needs.  `--candidates N` replays N episodes per class (default 4) and
picks the one to draw; `--all` keeps a figure for every episode replayed.
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))

import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import targets as tgt  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

OUT = ROOT / "results" / "paper_paths"
COLOURS = {"nav": "#dfe9f5", "band": "#b9d3ee", "wall": "#555555", "panel": "#8c6d46",
           "path": "#2b6cb0", "own": "#1f7a3f", "target": "#c0392b", "engaged": "#d4a017"}
# One figure per rule the paper claims, plus a static-only case.
PANELS = (("head_on", None), ("crossing", "starboard"), ("crossing", "port"),
          ("overtaking", None), ("being_overtaken", None), ("no_target", None))


def _side(built) -> str:
    return "port" if float(getattr(built, "ct_deg", 0.0)) < 180.0 else "starboard"


def _replay(env, built, seed: int, actor) -> dict:
    obs, _ = env.reset(seed=seed, options={"generated": built})
    actor.reset()
    own, tgt_track, engaged, speeds = [], [], [], []
    min_range, cpa_step = float("inf"), 0
    while True:
        own.append((env.asv_x, env.asv_y, env.asv_h))
        speeds.append(float(env.model.u))
        engaged.append(any(c.engaged for c in env.encounter_contexts.values()))
        if env.targets:
            t = env.targets[0]
            tgt_track.append((t.x, t.y, float(getattr(t, "heading", built.target_heading))))
            rng = math.hypot(t.x - env.asv_x, t.y - env.asv_y)
            if rng < min_range:
                min_range, cpa_step = rng, len(own) - 1
        obs, _, term, trunc, info = env.step(actor(obs))
        if term or trunc:
            break
    outcome = ("target" if info["collision_kind"] == "target" else
               "other" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    return {"own": np.array(own), "target": np.array(tgt_track), "speeds": np.array(speeds),
            "engaged": np.array(engaged), "min_range": min_range, "cpa_step": cpa_step,
            "outcome": outcome, "steps": len(own), "built": built,
            "obstacles": [np.asarray(p) for p in env.obstacles],
            "boundary": np.asarray(env.boundary_polygon),
            "path": np.asarray(env.path.points),
            "goal": (env.goal_x, env.goal_y)}


def _draw(ax, ep, compact=False):
    built = ep["built"]
    w, h = cfg.MAP_WIDTH, cfg.MAP_HEIGHT
    ax.add_patch(MplPolygon([(0, 0), (w, 0), (w, h), (0, h)], closed=True, fill=False,
                            ec=COLOURS["wall"], lw=1.2))
    ax.add_patch(MplPolygon(ep["boundary"], closed=True, fc=COLOURS["nav"], ec="#7a9cc6", lw=0.8))
    if built.geometry_mode == "basin" and built.encounter_class == "head_on":
        ax.add_patch(MplPolygon(built.channel.band().polygon(), closed=True,
                                fc=COLOURS["band"], ec="none", alpha=0.7))
    for poly in ep["obstacles"]:
        ax.add_patch(MplPolygon(poly, closed=True, fc=COLOURS["panel"], ec="k", lw=0.5))

    ax.plot(ep["path"][:, 0], ep["path"][:, 1], color=COLOURS["path"], lw=1.0, ls="--",
            label="reference path")

    # The own ship's track, coloured by speed: the 8(e) slow-down is visible.
    own = ep["own"]
    seg = np.stack([own[:-1, :2], own[1:, :2]], axis=1)
    lc = LineCollection(seg, cmap="viridis", norm=plt.Normalize(0.0, 2.0 * cfg.U_REF),
                        linewidths=2.2, zorder=4)
    lc.set_array(ep["speeds"][:-1])
    ax.add_collection(lc)

    # The engaged stretch, which is where the COLREGs terms apply.
    if ep["engaged"].any():
        e = own[ep["engaged"], :2]
        ax.plot(e[:, 0], e[:, 1], color=COLOURS["engaged"], lw=5.0, alpha=0.35, zorder=3,
                solid_capstyle="round", label="encounter engaged")

    ax.plot(own[0, 0], own[0, 1], "o", color=COLOURS["own"], ms=5, zorder=6)
    ax.plot(*ep["goal"], "*", color="#d4a017", ms=11, mec="k", mew=0.4, zorder=6)
    k = ep["cpa_step"]
    ax.add_patch(MplPolygon(tgt.hull_polygon(own[k, 0], own[k, 1], own[k, 2]), closed=True,
                            fc=COLOURS["own"], ec="k", lw=0.4, alpha=0.85, zorder=6))
    if len(ep["target"]):
        tr = ep["target"]
        ax.plot(tr[:, 0], tr[:, 1], color=COLOURS["target"], lw=1.4, ls=":", zorder=4)
        ax.add_patch(MplPolygon(tgt.hull_polygon(tr[k, 0], tr[k, 1], tr[k, 2]), closed=True,
                                fc=COLOURS["target"], ec="k", lw=0.4, alpha=0.85, zorder=6))
        ax.plot([own[k, 0], tr[k, 0]], [own[k, 1], tr[k, 1]], color="k", lw=0.7, ls="-", zorder=5)

    ax.set_xlim(-0.4, w + 0.4)
    ax.set_ylim(-0.4, h + 0.4)
    ax.set_aspect("equal")
    if compact:
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    return lc


def _caption(ep) -> str:
    b = ep["built"]
    side = f", from {_side(b)}" if b.encounter_class == "crossing" else ""
    close = ("static obstacles only" if not np.isfinite(ep["min_range"])
             else f"min. range {ep['min_range']:.2f} m")
    return (f"{b.encounter_class.replace('_', ' ')}{side} — {b.geometry_mode}"
            f"\n{close}, {ep['outcome']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "runs" / "ppo_formulation_seed0_bl2" / "best_model.zip")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--candidates", type=int, default=4, help="episodes replayed per class")
    ap.add_argument("--all", action="store_true", help="a figure for every episode replayed")
    args = ap.parse_args()

    from common import load_model
    out = OUT / args.tag
    (out).mkdir(parents=True, exist_ok=True)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None, emergency_stop=False)      # supervisor off
    actor = tf.EpisodeActor(load_model(str(args.model)))
    scenarios = tf.development_set(20)

    episodes, rows = [], []
    for cls, side in PANELS:
        pool = [b for b in scenarios if b.encounter_class == cls
                and (side is None or _side(b) == side)][:args.candidates]
        got = []
        for i, built in enumerate(pool):
            ep = _replay(env, built, 900_000 + i, actor)
            rows.append({"class": cls, "side": side or "", "candidate": i,
                         "outcome": ep["outcome"], "min_range": round(ep["min_range"], 2),
                         "steps": ep["steps"], "mode": built.geometry_mode,
                         "dcpa_drawn": round(float(getattr(built, "dcpa_m", float("nan"))), 2)})
            if args.all:
                episodes.append((f"{cls}_{side or 'any'}_{i}", ep))
            got.append(ep)
        # Representative, neither flattering nor alarming: among the episodes
        # that reached the goal, the one with the **median** closest approach.
        # Taking the closest pass would cherry-pick the tightest case; taking
        # the widest would flatter the policy.
        pool_ok = [e for e in got if e["outcome"] == "goal"] or got
        pool_ok.sort(key=lambda e: e["min_range"])
        if pool_ok:
            episodes.append((f"{cls}_{side or 'any'}", pool_ok[len(pool_ok) // 2]))

    pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)
    for name, ep in episodes:
        fig, ax = plt.subplots(figsize=(3.4, 7.6))
        lc = _draw(ax, ep)
        ax.set_title(_caption(ep), fontsize=8)
        cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("speed (m/s)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    sheet = [e for e in episodes if not e[0].rsplit("_", 1)[-1].isdigit()][:6]
    # Each panel is the 10 x 25 m basin at equal aspect, so the height follows
    # the panel width; anything taller leaves a band of white under the row.
    fig, axes = plt.subplots(1, len(sheet), figsize=(2.3 * len(sheet), 6.4))
    for j, (ax, (name, ep)) in enumerate(zip(np.atleast_1d(axes), sheet)):
        lc = _draw(ax, ep, compact=True)
        ax.set_title(f"({chr(97 + j)}) " + _caption(ep), fontsize=7)
    handles = [plt.Line2D([], [], color=COLOURS["path"], ls="--", lw=1.0, label="reference path"),
               plt.Line2D([], [], color=COLOURS["target"], ls=":", lw=1.4, label="target track"),
               plt.Line2D([], [], color=COLOURS["engaged"], lw=4, alpha=0.4, label="encounter engaged"),
               plt.Line2D([], [], color=COLOURS["own"], marker="o", ls="", label="start"),
               plt.Line2D([], [], color="#d4a017", marker="*", ls="", mec="k", label="goal"),
               plt.Line2D([], [], color="k", lw=0.7, label="closest approach")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    cb = fig.colorbar(lc, ax=np.atleast_1d(axes).tolist(), fraction=0.012, pad=0.01)
    cb.set_label("own ship speed (m/s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sheet.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{len(episodes)} figures -> {out}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
