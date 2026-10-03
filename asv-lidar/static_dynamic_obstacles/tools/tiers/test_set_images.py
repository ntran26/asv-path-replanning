"""Draw test set v2 (2026-10-03): every episode's starting scene, as images.

Each episode is reset with its own seed (frozen-suite panels are placed at reset,
so the cache alone does not hold them) and drawn: map boundary, static panels,
reference path, start and goal, the own hull, and the target hull with its
heading and its next 10 s at constant velocity.  The title gives the test id,
the variant, and the outcome of SAC 3 M alone and of SAC + safety v8 (from the
trigger-counterfactual closed-loop results, where available).

    python tools/tiers/test_set_images.py [--processes 2]

Writes under `results/test_set/v<version>/images/` (default v3):
* `episodes/<test_id>.png`          -- one image per episode (1,000, one folder)
* `sheet_<cell>.png`                -- a contact sheet per cell (23)
* `test_set_v2_sheets.pdf`          -- every contact sheet, one page each
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]

import numpy as np
import pandas as pd

V2 = ROOT / "results" / "test_set" / "v3"          # set by --version
OUT = V2 / "images"
TRACK_S = 10.0
_W = {}


def _outcomes(with_v8=False):
    sac = pd.read_csv(V2 / "sacs0_bl3" / "episodes.csv", keep_default_na=False).set_index("test_id").outcome.to_dict()
    v8 = {}
    if not with_v8:                                   # v8 closed-loop outcomes exist for v2 only
        return sac, v8
    d = ROOT / "results" / "safety_dev" / "trigger_counterfactual"
    try:
        base = pd.read_csv(d / "episodes.csv").drop_duplicates("case").set_index("case")
        br = pd.read_csv(d / "v7_nohold" / "branches.csv")
        first = br[(br.version == "v7") & (br.fire_index == 1)].drop_duplicates("case").set_index("case")
        for case, row in base.iterrows():
            if case.startswith("TS2:"):
                v8[case[4:]] = first.branch_outcome.get(case, row.outcome)
    except FileNotFoundError:
        pass
    return sac, v8


def _init(out=None):
    global OUT
    if out is not None:
        OUT = Path(out)                 # spawned workers re-import the module: take the version's folder
    import curriculum
    import train_formulation as tf
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)


def _geometry(job):
    tid, built, seed, cell, variant = job
    env = _W["env"]
    env.reset(seed=seed, options={"generated": built})
    targets = [{"hull": [tuple(map(float, p)) for p in t.hull()], "x": float(t.x), "y": float(t.y),
                "heading": float(t.heading), "speed": float(t.speed), "behaviour": str(t.behaviour)}
               for t in env.targets]
    return {"test_id": tid, "cell": cell, "variant": variant,
            "boundary": [tuple(map(float, p)) for p in env.boundary_polygon],
            "obstacles": [[tuple(map(float, p)) for p in poly] for poly in env.obstacles],
            "path": np.asarray(env.path.points, dtype=float).tolist(),
            "start": (float(env.start_x), float(env.start_y)), "goal": (float(env.goal_x), float(env.goal_y)),
            "own": [tuple(map(float, p)) for p in env.hull_polygon()], "targets": targets}


def _short(outcome):
    if outcome is None or (isinstance(outcome, float) and math.isnan(outcome)):
        return "-"
    return "goal" if outcome == "goal" else str(outcome).replace("collision:", "")


def draw(ax, g, sac, v8, small=False):
    from matplotlib.patches import Polygon
    b = np.asarray(g["boundary"] + [g["boundary"][0]])
    ax.plot(b[:, 0], b[:, 1], color="black", lw=0.8 if small else 1.2)
    for poly in g["obstacles"]:
        ax.add_patch(Polygon(poly, closed=True, facecolor="0.55", edgecolor="0.25", lw=0.5))
    p = np.asarray(g["path"])
    ax.plot(p[:, 0], p[:, 1], ls="--", color="tab:blue", lw=0.6 if small else 1.0)
    ax.plot(*g["start"], "o", color="tab:green", ms=3 if small else 5)
    ax.plot(*g["goal"], "*", color="tab:green", ms=5 if small else 10)
    ax.add_patch(Polygon(g["own"], closed=True, facecolor="tab:blue", edgecolor="navy", lw=0.5))
    for t in g["targets"]:
        ax.add_patch(Polygon(t["hull"], closed=True, facecolor="tab:red", edgecolor="darkred", lw=0.5))
        a = math.radians(t["heading"])
        dx, dy = math.sin(a) * t["speed"] * TRACK_S, math.cos(a) * t["speed"] * TRACK_S
        ax.plot([t["x"], t["x"] + dx], [t["y"], t["y"] + dy], ls=":", color="tab:red", lw=0.8 if small else 1.2)
    xs, ys = b[:, 0], b[:, 1]
    ax.set_xlim(xs.min() - 0.3, xs.max() + 0.3)
    ax.set_ylim(ys.min() - 0.3, ys.max() + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    s = _short(sac.get(g["test_id"]))
    v = _short(v8.get(g["test_id"])) if v8 else None
    colour = "black" if s == "goal" else "firebrick"
    panels = len(g["obstacles"])
    leg = "straight" if abs(g["path"][-1][0] - g["path"][0][0]) < 0.75 else "slanted"
    if small:
        ax.set_title(f"{g['test_id']}\nSAC {s}" + (f" | v8 {v}" if v else ""), fontsize=5, color=colour)
    else:
        ax.set_title(f"{g['test_id']}  ({g['cell']}, {g['variant'] or '-'}, {panels} panels, {leg})\n"
                     f"SAC 3M: {s}" + (f"   SAC + safety v8: {v}" if v else ""), fontsize=9, color=colour)


def _render_one(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g, sac, v8 = args
    b = np.asarray(g["boundary"])
    w, h = np.ptp(b[:, 0]) + 0.6, np.ptp(b[:, 1]) + 0.6
    scale = 6.5 / max(w, h)
    fig, ax = plt.subplots(figsize=(max(2.6, w * scale + 0.4), max(2.6, h * scale + 0.8)))
    draw(ax, g, sac, v8)
    path = OUT / "episodes" / f"{g['test_id']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def _job_geometry_and_png(job):
    g = _geometry(job[:5])
    _render_one((g, job[5], job[6]))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--version", choices=("2", "3"), default="3")
    a = ap.parse_args()
    global V2, OUT
    V2 = ROOT / "results" / "test_set" / f"v{a.version}"
    OUT = V2 / "images"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with open(V2 / f"set_v{a.version}.0.pkl", "rb") as fh:
        kept, report = pickle.load(fh)
    sac, v8 = _outcomes(with_v8=(a.version == "2"))
    jobs = [(it["test_id"], it["built"], int(it["episode_seed"]), it["cell"], it["variant"], sac, v8) for it in kept]
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.processes, initializer=_init, initargs=(str(OUT),)) as pool:
        geoms = list(pool.map(_job_geometry_and_png, jobs, chunksize=4))
    print(f"{len(geoms)} episode images written", flush=True)
    order = pd.read_csv(V2 / "definition.csv", keep_default_na=False)[["test_id", "source", "cell"]]
    by_cell = {}
    for g in geoms:
        by_cell.setdefault(g["cell"], []).append(g)
    cells = list(dict.fromkeys(order.cell))
    with PdfPages(OUT / f"test_set_v{a.version}_sheets.pdf") as pdf:
        for cell in cells:
            gs = sorted(by_cell.get(cell, []), key=lambda g: g["test_id"])
            n = len(gs)
            basin = np.ptp(np.asarray(gs[0]["boundary"])[:, 1]) > np.ptp(np.asarray(gs[0]["boundary"])[:, 0])
            cols = 10 if n > 30 else 5
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * (1.25 if basin else 2.4), rows * (2.9 if basin else 1.6)))
            axes = np.atleast_1d(axes).ravel()
            for ax, g in zip(axes, gs):
                draw(ax, g, sac, v8, small=True)
            for ax in axes[n:]:
                ax.axis("off")
            fails = sum(1 for g in gs if sac.get(g["test_id"]) != "goal")
            extra = ""
            if v8:
                fails8 = sum(1 for g in gs if v8.get(g["test_id"], sac.get(g["test_id"])) != "goal")
                extra = f", SAC + v8 fails {fails8}"
            fig.suptitle(f"Test set v{a.version} -- {cell}: {n} episodes; SAC 3M fails {fails}{extra}  "
                         f"(blue own ship, red target with its next {TRACK_S:.0f} s at constant velocity)", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(OUT / f"sheet_{cell}.png", dpi=130)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"sheet {cell}: {n}", flush=True)
    print(f"done: {OUT}", flush=True)


if __name__ == "__main__":
    main()
