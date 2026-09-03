"""Figure A: success rate by obstacle count, all methods, on the frozen 500.

    python src/plot_success_by_obstacles.py
    python src/plot_success_by_obstacles.py --out figures/fig_success --no-overall

Why this figure rather than a learning curve
--------------------------------------------
Evaluation return is a poor axis for comparing these three methods.  The
classical controller has no training curve at all, so a return-vs-timesteps plot
structurally excludes it; and with -1000 terminal penalties for collision and
timeout, episode return is essentially a rescaled collision rate on a scale no
reader can interpret.  Success rate is the metric every claim in the paper is
actually about, and all methods can appear on it on equal footing because they
share the same 500 evaluation layouts.

Obstacle count is the x-axis because it is the axis the paper's generalisation
claim rests on -- it shows not just which method is better overall, but how each
degrades as the layouts get harder.

What is drawn
-------------
* One line per method family, the mean across its runs (training seeds for the
  learned methods, tuning searches for LOS+APF).
* Individual run values as small open markers, so the seed spread is visible
  rather than hidden inside an interval.
* A shaded band spanning min-max across runs.
* The deployed SAC policy as a **separate** black reference line.  It is a
  single checkpoint, not a member of any seed set, and is drawn distinctly so it
  cannot be mistaken for one.

Every method uses a distinct line style *and* marker, not colour alone, per the
reviewer comment about an earlier figure.

Outputs `<out>.png`, `<out>.svg` and `<out>.csv`, the last so the figure can be
restyled for a journal template without re-reading the per-episode data.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BASE = os.path.join("eval_results", "baselines")

# (label, glob of per-episode CSVs, colour, linestyle, marker, is_reference)
#
# `is_reference` marks a single checkpoint rather than a family of runs -- it is
# drawn as a distinct reference line with no band and no per-run markers.
METHODS: List[Tuple[str, str, str, str, str, bool]] = [
    ("SAC (deployed policy)", os.path.join(BASE, "sac_1M", "episodes.csv"),
     "black", "-", "o", True),
    ("SAC (3 seeds)", os.path.join(BASE, "sac_gs4_seed*_best", "episodes.csv"),
     "tab:green", "-.", "^", False),
    ("PPO (3 seeds)", os.path.join(BASE, "ppo_fx_seed*_best", "episodes.csv"),
     "tab:blue", "--", "s", False),
    ("LOS+APF (3 searches)", os.path.join(BASE, "los_apf_s*", "episodes.csv"),
     "tab:red", ":", "D", False),
]


def load_success_by_group(path: str) -> Dict[Any, float]:
    """Success rate per obstacle count for one run, plus an 'all' entry.

    Per-count rates come from the run's `summary.json` (`by_obstacle_count`)
    when it is present, falling back to recomputing them from `episodes.csv`.

    The **'all' entry is the mean of the per-count success rates**, not a pool
    over every episode.  On a set with equal episodes per count the two agree
    exactly -- the frozen 500 is 100 per count, so every number in this figure
    is unchanged by the choice.  They diverge on an unbalanced set, where
    pooling would silently weight each obstacle count by its episode count
    rather than treating the counts as equally important conditions.  Averaging
    the per-count rates makes that weighting explicit and keeps the figure
    correct if a set with uneven groups is ever plotted.
    """
    summary_path = os.path.join(os.path.dirname(path), "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            by_count = json.load(f).get("by_obstacle_count", {})
        # keys look like "obs_3"
        per_count = {int(k.split("_")[1]): float(v["success_rate"])
                     for k, v in by_count.items()}
        if per_count:
            out: Dict[Any, float] = dict(per_count)
            out["all"] = float(np.mean(list(per_count.values())))
            return out

    # Fallback: recompute from the per-episode file.
    by_group: Dict[int, List[float]] = {}
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            by_group.setdefault(int(float(row["obstacle_count"])), []).append(
                float(row["success"]))
    per_count = {g: float(np.mean(v)) for g, v in by_group.items()}
    out = dict(per_count)
    out["all"] = float(np.mean(list(per_count.values())))
    return out


def collect() -> List[Dict[str, Any]]:
    """Resolve METHODS against what exists on disk."""
    out = []
    for label, pattern, colour, ls, marker, is_ref in METHODS:
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"  (skipping {label}: nothing matches {pattern})")
            continue
        runs = [{"name": os.path.basename(os.path.dirname(p)),
                 "success": load_success_by_group(p)} for p in paths]
        out.append({"label": label, "runs": runs, "colour": colour,
                    "linestyle": ls, "marker": marker, "is_reference": is_ref})
        print(f"  {label}: {len(runs)} run(s) — {', '.join(r['name'] for r in runs)}")
    return out


def obstacle_counts(methods: List[Dict[str, Any]]) -> List[int]:
    counts = set()
    for m in methods:
        for r in m["runs"]:
            counts |= {g for g in r["success"] if isinstance(g, int)}
    return sorted(counts)


def plot(methods: List[Dict[str, Any]], out: str, include_overall: bool,
         dpi: int, ylim: Optional[Tuple[float, float]] = None,
         jitter: float = 0.07) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: List[Any] = list(obstacle_counts(methods))
    if include_overall:
        groups.append("all")
    x = np.arange(len(groups), dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    # Offset each method slightly so overlapping per-run markers stay readable.
    n_fam = sum(1 for m in methods if not m["is_reference"])
    offsets, k = {}, 0
    for m in methods:
        if m["is_reference"]:
            offsets[m["label"]] = 0.0
        else:
            offsets[m["label"]] = (k - (n_fam - 1) / 2.0) * jitter
            k += 1

    for m in methods:
        dx = offsets[m["label"]]
        series = np.array([[r["success"].get(g, np.nan) for g in groups]
                           for r in m["runs"]], dtype=float)
        mean = np.nanmean(series, axis=0)

        if m["is_reference"]:
            ax.plot(x, mean, color=m["colour"], linestyle=m["linestyle"],
                    marker=m["marker"], markersize=7, linewidth=2.2,
                    label=m["label"], zorder=5)
            continue

        if series.shape[0] > 1:
            ax.fill_between(x + dx, np.nanmin(series, axis=0),
                            np.nanmax(series, axis=0),
                            color=m["colour"], alpha=0.15, linewidth=0, zorder=1)
            for row in series:
                ax.plot(x + dx, row, linestyle="none", marker=m["marker"],
                        markersize=4, markerfacecolor="none",
                        markeredgecolor=m["colour"], alpha=0.65, zorder=3)
        ax.plot(x + dx, mean, color=m["colour"], linestyle=m["linestyle"],
                marker=m["marker"], markersize=6.5, linewidth=1.9,
                label=m["label"], zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([("all" if g == "all" else str(g)) for g in groups])
    ax.set_xlabel("Number of obstacles")
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate by obstacle count (500-episode frozen evaluation set)")
    if ylim is None:
        # Auto-fit: a full 0-1 axis leaves every series crushed into the top
        # fifth of the panel and hides the differences the figure exists to
        # show.  The axis is zoomed to the data with padding, and the caption
        # should say so; pass --ylim 0,1 to force the full range.
        lows = [np.nanmin([[r["success"].get(g, np.nan) for g in groups]
                           for r in m["runs"]]) for m in methods]
        lo = max(0.0, float(np.floor((min(lows) - 0.04) * 20) / 20))
        ax.set_ylim(lo, 1.01)
    else:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, linestyle=":")
    if include_overall and len(groups) > 1:
        # Separate the pooled column from the per-count ones.
        ax.axvline(x[-1] - 0.5, color="0.6", linewidth=1.0, linestyle="-", alpha=0.8)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("png", "svg"):
        p = f"{out}.{ext}"
        fig.savefig(p, dpi=dpi)
        print(f"  wrote {p}")
    plt.close(fig)

    csv_path = f"{out}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "run", "obstacle_count", "success_rate"])
        for m in methods:
            for r in m["runs"]:
                for g in groups:
                    v = r["success"].get(g)
                    if v is not None:
                        w.writerow([m["label"], r["name"],
                                    "all" if g == "all" else g, f"{v:.6f}"])
    print(f"  wrote {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(BASE, "figures", "success_by_obstacles"),
                    help="output path without extension")
    ap.add_argument("--no-overall", action="store_true",
                    help="omit the pooled 'all' column")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--ylim", default=None,
                    help="y-axis range as 'lo,hi' (e.g. 0,1). Default auto-fits "
                         "to the data; the caption should note the zoom.")
    args = ap.parse_args()

    print("Collecting runs ...")
    methods = collect()
    if not methods:
        raise SystemExit("no per-episode CSVs found under " + BASE)
    print("Plotting ...")
    ylim = None
    if args.ylim:
        lo, hi = (float(v) for v in args.ylim.split(","))
        ylim = (lo, hi)
    plot(methods, args.out, not args.no_overall, args.dpi, ylim)


if __name__ == "__main__":
    main()
