"""Trajectory snapshots of successful episodes for the paper (2026-10-06).

Replays chosen test-set v4 episodes with a policy, exactly as the test set ran
them (safety layer off, the episode's own seed), and draws each one top-down:
facility walls, the navigable area, static panels, the planned path, start and
goal, the own-ship track (blue) and target track (red).  Both hulls are drawn
together every `--every` seconds and numbered alike, so positions at the same
instant can be paired; the closest point of approach is marked with its range.

    python tools/paper_snapshots.py                       # the paper figure
    python tools/paper_snapshots.py --cases A,B,C --name candidates --cols 6

Writes `results/paper_snapshots/<name>.pdf` and `.png` (all panels), one PNG per
panel, and a per-step trace CSV per episode.  Each replay is checked against the
outcome recorded in the test set's `episodes.csv`; a mismatch stops the run.
The process runs at below-normal priority, so a training run keeps the CPU.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]

# The paper figure: one episode per encounter type, varied in layout, panel count
# and leg shape, each with the manoeuvre the formulation asks for (selection notes
# in results/paper_snapshots/README.txt).
DEFAULT_CASES = ("FS-NT-010", "P2-L3-HO-FIX-15", "CH-CR-CV-035",
                 "P2-L2-CRS-VAR-10", "BAS-OT-CV-073", "BAS-BO-RE-016")
TITLES = {"NT": "No target", "HO": "Head-on", "CRP": "Port crossing",
          "CRS": "Starboard crossing", "OT": "Overtaking", "BO": "Being overtaken"}
LAYOUT = {"BAS": "basin", "CH": "channel", "FS": "coupled layout", "L1": "layout L1",
          "L2": "layout L2", "L3": "layout L3"}
OWN, TGT = "#0969da", "#cf222e"


def _low_priority() -> None:
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)    # BELOW_NORMAL_PRIORITY_CLASS


def code_of(test_id: str, crossing_side: str) -> str:
    for code in ("CRP", "CRS", "NT", "HO", "OT", "BO"):
        if f"-{code}-" in test_id:
            return code
    if "-CR-" in test_id:
        return "CRP" if crossing_side == "port" else "CRS"
    raise ValueError(test_id)


def layout_of(test_id: str) -> str:
    for key, name in LAYOUT.items():
        if test_id.startswith(key + "-") or f"-{key}-" in test_id:
            return name
    return ""


def rollout(env, model, built, seed: int):
    import train_formulation as tf
    actor = tf.EpisodeActor(model)
    obs, _ = env.reset(seed=seed, options={"generated": built})
    scene = {"obstacles": [list(p) for p in env.obstacles],
             "boundary": [tuple(map(float, p)) for p in env.boundary_polygon],
             "path": [tuple(map(float, p)) for p in env.path.points],
             "start": (env.asv_x, env.asv_y), "goal": (env.goal_x, env.goal_y)}
    steps = []

    def record(t, info=None):
        steps.append({"t": t, "x": env.asv_x, "y": env.asv_y, "heading": env.asv_h,
                      "speed": float(info["speed_mps"]) if info else float("nan"),
                      "rudder_pct": float(env.rudder), "rpm": float(env.rpm),
                      "hull": env.hull_polygon(),
                      "targets": [(t_.x, t_.y, t_.hull()) for t_ in env.targets]})

    record(0.0)
    import constants as cfg
    k = 0
    while True:
        obs, _, term, trunc, info = env.step(actor(obs))
        k += 1
        record(k * cfg.UPDATE_RATE, info)
        if term or trunc:
            break
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    return scene, steps, outcome


def draw(ax, scene, steps, every_s: float, title: str, subtitle: str, map_w: float, map_h: float):
    ax.add_patch(Polygon([(0, 0), (map_w, 0), (map_w, map_h), (0, map_h)], closed=True,
                         fc="#d8dee4", ec="#24292f", lw=1.2, zorder=0))
    ax.add_patch(Polygon(scene["boundary"], closed=True, fc="white", ec="#8c959f", lw=0.6,
                         ls="--", zorder=1))
    for poly in scene["obstacles"]:
        ax.add_patch(Polygon(poly, closed=True, fc="#6e7781", ec="#24292f", lw=0.6, zorder=2))
    px, py = zip(*scene["path"])
    ax.plot(px, py, color="#57606a", lw=0.7, ls=(0, (3, 2)), zorder=3)
    ax.plot(*scene["start"], "o", ms=3, color="#24292f", zorder=6)
    ax.plot(*scene["goal"], "*", ms=8, color="#bf8700", mec="#7d4e00", mew=0.4, zorder=6)

    xs, ys = [s["x"] for s in steps], [s["y"] for s in steps]
    ax.plot(xs, ys, color=OWN, lw=1.3, zorder=5)
    n_tgt = len(steps[0]["targets"])
    for j in range(n_tgt):
        tx = [s["targets"][j][0] for s in steps]
        ty = [s["targets"][j][1] for s in steps]
        ax.plot(tx, ty, color=TGT, lw=1.1, zorder=5)
        ax.annotate("", xy=(tx[-1], ty[-1]), xytext=(tx[-3], ty[-3]),
                    arrowprops=dict(arrowstyle="-|>", color=TGT, lw=0.9, mutation_scale=7), zorder=6)

    # Synchronised snapshots: both hulls every `every_s`, plus the final state.
    t_end = steps[-1]["t"]
    marks = [i for i, s in enumerate(steps) if abs((s["t"] / every_s) - round(s["t"] / every_s)) < 1e-6]
    if marks[-1] != len(steps) - 1:
        marks.append(len(steps) - 1)
    last = {}                                             # last labelled position per ship

    def spaced(key, x, y):
        """True when a label at (x, y) keeps clear of the ship's previous one."""
        if key in last and math.hypot(x - last[key][0], y - last[key][1]) < 0.8:
            return False
        last[key] = (x, y)
        return True

    for n, i in enumerate(marks):
        s = steps[i]
        a = 0.30 + 0.65 * (s["t"] / max(t_end, 1e-6))
        ax.add_patch(Polygon(s["hull"], closed=True, fc=OWN, ec="#0550ae", lw=0.4, alpha=a, zorder=7))
        label = str(n + 1) if i != len(steps) - 1 or s["t"] % every_s < 1e-6 else ""
        if label and spaced("own", s["x"], s["y"]):
            ax.text(s["x"] - 0.75, s["y"], label, fontsize=5, color="#0550ae", ha="right", va="center",
                    zorder=9, clip_on=True)
        for j, (tx, ty, hull) in enumerate(s["targets"]):
            ax.add_patch(Polygon(hull, closed=True, fc=TGT, ec="#a40e26", lw=0.4, alpha=a, zorder=7))
            if label and 0.0 <= tx <= map_w and 0.0 <= ty <= map_h and spaced(j, tx, ty):
                ax.text(tx + 0.75, ty, label, fontsize=5, color="#a40e26", ha="left", va="center", zorder=9,
                        clip_on=True)

    # Closest point of approach.
    if n_tgt:
        best = min(((math.hypot(s["x"] - tx, s["y"] - ty), i, tx, ty)
                    for i, s in enumerate(steps) for (tx, ty, _) in s["targets"]))
        d, i, tx, ty = best
        s = steps[i]
        ax.plot([s["x"], tx], [s["y"], ty], color="#24292f", lw=0.6, ls=":", zorder=8)
        mx, my = (s["x"] + tx) / 2, (s["y"] + ty) / 2
        right = mx < map_w / 2                            # label on the side with more room
        ax.text(mx + 0.4 if right else mx - 0.4, my + 0.7, f"CPA {d:.1f} m", ha="left" if right else "right",
                fontsize=5.5, color="#24292f", zorder=9, clip_on=True,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_title(title, fontsize=7, pad=15)
    ax.text(0.5, 1.008, subtitle, transform=ax.transAxes, fontsize=5.5, ha="center", va="bottom",
            color="#424a53", linespacing=1.1)
    ax.set_xlim(-0.2, map_w + 0.2)
    ax.set_ylim(-0.2, map_h + 0.2)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10, 15, 20, 25])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path,
                    default=ROOT / "runs" / "sac_formulation_seed0_bl3" / "kept_best_3M" / "best_model.zip")
    ap.add_argument("--tag", default="sacs0_bl3", help="test-set results folder with the recorded outcomes")
    ap.add_argument("--cases", default=",".join(DEFAULT_CASES))
    ap.add_argument("--name", default="paper_snapshots")
    ap.add_argument("--every", type=float, default=5.0, help="snapshot interval, s")
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args()
    _low_priority()

    import pandas as pd
    import torch
    torch.set_num_threads(1)
    import constants as cfg
    cfg.EMERGENCY_STOP_ENABLED = False                    # as the test set ran: safety layer off
    import curriculum
    import train_formulation as tf
    from common import load_model
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    env = ASVLidarEnv(render_mode=None)
    env.estop_enabled = False
    model = load_model(args.model)

    sdir = ROOT / "results" / "test_set" / "v4"
    with open(sdir / "set_v4.0.pkl", "rb") as fh:
        kept, _ = pickle.load(fh)
    items = {it["test_id"]: it for it in kept}
    rec = pd.read_csv(sdir / args.tag / "episodes.csv", keep_default_na=False)
    rec = rec[rec.safety == "off"].set_index("test_id")
    legs = pd.read_csv(sdir / "definition.csv", keep_default_na=False).set_index("test_id").leg.to_dict()

    out = ROOT / "results" / "paper_snapshots"
    out.mkdir(parents=True, exist_ok=True)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    rows = math.ceil(len(cases) / args.cols)
    cols = min(args.cols, len(cases))
    height = 3.0 * rows + 0.5                             # 0.5 in for the legend
    fig, axes = plt.subplots(rows, cols, figsize=(1.18 * cols, height), squeeze=False)
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    summary = []
    for k, case in enumerate(cases):
        it = items[case]
        scene, steps, outcome = rollout(env, model, it["built"], int(it["episode_seed"]))
        recorded = rec.loc[case, "outcome"]
        if outcome != recorded:
            raise SystemExit(f"{case}: replay gave {outcome}, the test set recorded {recorded}")
        code = code_of(case, rec.loc[case, "crossing_side"])
        n_pan = len(scene["obstacles"])
        sub = (f"{layout_of(case)}, {legs.get(case, '')} leg\n{n_pan} panel{'s' if n_pan != 1 else ''}"
               + (", varying speed" if "-VAR-" in case else ""))
        letter = "abcdefghijklmnopqrstuvwxyz"[k]
        title = f"({letter}) {TITLES[code]}"
        draw(axes.flat[k], scene, steps, args.every, title, sub, cfg.MAP_WIDTH, cfg.MAP_HEIGHT)
        if k % cols:
            axes.flat[k].set_yticklabels([])
        # Single panel and trace.
        f1, a1 = plt.subplots(figsize=(1.8, 4.6))
        draw(a1, scene, steps, args.every, TITLES[code], sub + f"\n{case}", cfg.MAP_WIDTH, cfg.MAP_HEIGHT)
        f1.tight_layout()
        f1.savefig(out / f"{case}.png", dpi=300)
        plt.close(f1)
        with open(out / f"{case}_trace.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "x_m", "y_m", "heading_deg", "speed_mps", "rudder_pct", "rpm"]
                       + [f"target{j}_{a}" for j in range(len(steps[0]["targets"])) for a in ("x_m", "y_m")])
            for s in steps:
                w.writerow([f"{s['t']:.1f}", f"{s['x']:.3f}", f"{s['y']:.3f}", f"{s['heading']:.1f}",
                            f"{s['speed']:.3f}", f"{s['rudder_pct']:.1f}", f"{s['rpm']:.2f}"]
                           + [f"{v:.3f}" for (tx, ty, _) in s["targets"] for v in (tx, ty)])
        cpa = min(((math.hypot(s["x"] - tx, s["y"] - ty), s["t"]) for s in steps for (tx, ty, _) in s["targets"]),
                  default=(float("nan"), float("nan")))
        summary.append({"panel": letter, "case": case, "encounter": TITLES[code], "layout": layout_of(case),
                        "leg": legs.get(case, ""), "panels": n_pan, "varying_speed": "-VAR-" in case,
                        "duration_s": steps[-1]["t"], "cpa_m": round(cpa[0], 2), "t_cpa_s": cpa[1],
                        "min_speed_mps": round(min(s["speed"] for s in steps[1:]), 3)})
        print(f"{case}: {outcome}, {steps[-1]['t']:.1f} s, {n_pan} panels", flush=True)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=OWN, lw=1.3, label="Own ship"),
               Line2D([], [], color=TGT, lw=1.1, label="Target ship"),
               Line2D([], [], color="#57606a", lw=0.7, ls=(0, (3, 2)), label="Planned path"),
               Patch(fc="#6e7781", ec="#24292f", lw=0.6, label="Static panel"),
               Patch(fc="#d8dee4", ec="#8c959f", lw=0.6, ls="--", label="Outside navigable area"),
               Line2D([], [], marker="o", ls="", ms=3, color="#24292f", label="Start"),
               Line2D([], [], marker="*", ls="", ms=7, color="#bf8700", mec="#7d4e00", mew=0.4, label="Goal"),
               Line2D([], [], color="#24292f", lw=0.6, ls=":", label="Closest point of approach")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 0.16 / height), handlelength=2.0, columnspacing=1.2)
    fig.text(0.5, 0.04 / height, f"Hulls drawn every {args.every:g} s; equal numbers mark the same instant.",
             ha="center", fontsize=5.5, color="#424a53")
    fig.tight_layout(w_pad=0.4, h_pad=0.8, rect=(0, 0.5 / height, 1, 1))
    pd.DataFrame(summary).to_csv(out / f"{args.name}_summary.csv", index=False)
    fig.savefig(out / f"{args.name}.pdf")
    fig.savefig(out / f"{args.name}.png", dpi=300)
    print(f"-> {out / args.name}.pdf / .png", flush=True)


if __name__ == "__main__":
    main()
