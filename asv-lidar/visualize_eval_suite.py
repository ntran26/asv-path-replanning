#!/usr/bin/env python3
"""
Visualize a fixed ASV evaluation suite.

This script is intended for quickly checking whether a generated suite of
randomized obstacle layouts is visually reasonable/feasible before using it as
a holdout evaluation set.

Expected suite formats supported:
  1) {"cases": [ ... ]}
  2) [ ... ]

Each case should contain at least:
  - start: [x, y]
  - goal: [x, y]
  - obstacles: list of polygon points [[x,y], ...]

Optional fields used if available:
  - case_id
  - obstacle_count
  - map_width, map_height
  - path: [[x,y], ...]
  - feasibility: {path_length_ratio, approx_min_clearance_to_obstacle, ...}

Usage examples:
  python visualize_eval_suite.py
  python visualize_eval_suite.py data/env_setup/eval_suite_500/asv_eval_suite_500.json

Edit the constants below for your local project paths.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
# Default input if no command-line path is given.
SUITE_JSON = "data/env_setup/eval_suite/asv_eval_suite.json"

# Output directory. If the input is passed from command line, the default output
# folder will be created beside this script unless you edit this value.
OUT_DIR = "suite_visualization"

# Map defaults, used only if a case does not include map_width/map_height.
DEFAULT_MAP_WIDTH = 10.0
DEFAULT_MAP_HEIGHT = 25.0

# Plot settings.
CASES_PER_PAGE = 25        # 5x5 page by default
N_COLS = 5
DRAW_INDIVIDUAL_CASES = True
DRAW_GROUP_CONTACT_SHEETS = True
DRAW_ALL_CONTACT_SHEETS = True
DRAW_INFLATED_OBSTACLES = True
INFLATE_RADIUS = 0.45      # visual safety inflation radius [m]
SHOW_CASE_TEXT = True
DPI = 180

# If True, y-axis is normal map coordinates: y increases upward.
# This matches your trajectory paper plots.
Y_UP = True

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_suite(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "cases" in data:
        cases = data["cases"]
        meta = {k: v for k, v in data.items() if k != "cases"}
    elif isinstance(data, list):
        cases = data
        meta = {}
    else:
        raise ValueError("Suite JSON must be a list or a dict containing a 'cases' list.")

    if not isinstance(cases, list):
        raise ValueError("'cases' must be a list")

    return cases, meta


def case_id(case: Dict[str, Any], fallback: int) -> int:
    return int(case.get("case_id", fallback))


def obstacle_count(case: Dict[str, Any]) -> int:
    if "obstacle_count" in case:
        return int(case["obstacle_count"])
    return len(case.get("obstacles", []))


def map_size(case: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[float, float]:
    cfg = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    w = float(case.get("map_width", cfg.get("map_width", DEFAULT_MAP_WIDTH)))
    h = float(case.get("map_height", cfg.get("map_height", DEFAULT_MAP_HEIGHT)))
    return w, h


def polygon_bbox(poly: Iterable[Iterable[float]]) -> Tuple[float, float, float, float]:
    pts = list(poly)
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def draw_case(
    ax,
    case: Dict[str, Any],
    *,
    meta: Dict[str, Any],
    fallback_idx: int,
    title_prefix: str = "",
    show_text: bool = True,
    draw_inflated: bool = True,
) -> None:
    mw, mh = map_size(case, meta)
    cid = case_id(case, fallback_idx)
    nobs = obstacle_count(case)

    start = case.get("start", [mw / 2.0, 2.0])
    goal = case.get("goal", [mw / 2.0, mh - 3.0])
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])

    # Map boundary.
    ax.add_patch(Rectangle((0, 0), mw, mh, fill=False, edgecolor="0.35", linewidth=1.2))

    # Reference path.
    path = case.get("path", None)
    if path and len(path) >= 2:
        px = [float(p[0]) for p in path]
        py = [float(p[1]) for p in path]
    else:
        px = [sx, gx]
        py = [sy, gy]
    ax.plot(px, py, "k--", linewidth=1.2, alpha=0.9)

    # Obstacles.
    for obs in case.get("obstacles", []):
        poly = [(float(p[0]), float(p[1])) for p in obs]
        if draw_inflated:
            x0, y0, x1, y1 = polygon_bbox(poly)
            ax.add_patch(Rectangle(
                (x0 - INFLATE_RADIUS, y0 - INFLATE_RADIUS),
                (x1 - x0) + 2 * INFLATE_RADIUS,
                (y1 - y0) + 2 * INFLATE_RADIUS,
                facecolor="tab:red",
                edgecolor="none",
                alpha=0.12,
                zorder=1,
            ))
        ax.add_patch(Polygon(poly, closed=True, facecolor="tab:red", edgecolor="darkred", alpha=0.55, linewidth=0.8, zorder=2))

    # Start and goal.
    ax.scatter([sx], [sy], s=24, c="tab:green", edgecolor="white", linewidth=0.5, zorder=4)
    ax.scatter([gx], [gy], s=24, c="tab:red", edgecolor="white", linewidth=0.5, zorder=4)

    # Text annotations.
    title = f"{title_prefix}#{cid} | obs={nobs}"
    feas = case.get("feasibility", {}) if isinstance(case.get("feasibility", {}), dict) else {}
    ratio = feas.get("path_length_ratio", None)
    clear = feas.get("approx_min_clearance_to_obstacle", None)
    if ratio is not None:
        title += f" | ratio={float(ratio):.2f}"
    if clear is not None:
        title += f" | clr={float(clear):.2f}"
    ax.set_title(title, fontsize=7)

    if show_text:
        txt_lines = []
        if ratio is not None:
            txt_lines.append(f"ratio {float(ratio):.2f}")
        if clear is not None:
            txt_lines.append(f"clr {float(clear):.2f}m")
        if txt_lines:
            ax.text(0.03, 0.97, "\n".join(txt_lines), transform=ax.transAxes,
                    va="top", ha="left", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xlim(-0.2, mw + 0.2)
    ax.set_ylim(-0.2, mh + 0.2)
    if not Y_UP:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def save_individual_cases(cases: List[Dict[str, Any]], meta: Dict[str, Any], out_dir: Path) -> None:
    case_dir = out_dir / "individual_cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    for i, case in enumerate(cases):
        nobs = obstacle_count(case)
        cid = case_id(case, i)
        fig, ax = plt.subplots(figsize=(3.0, 7.0), dpi=DPI)
        draw_case(ax, case, meta=meta, fallback_idx=i, show_text=True, draw_inflated=DRAW_INFLATED_OBSTACLES)
        fig.tight_layout(pad=0.2)
        fig.savefig(case_dir / f"case_{cid:04d}_obs_{nobs}.png", bbox_inches="tight")
        plt.close(fig)


def save_contact_pages(
    cases: List[Dict[str, Any]],
    meta: Dict[str, Any],
    out_dir: Path,
    *,
    prefix: str,
    title: str,
) -> None:
    if not cases:
        return

    ncols = N_COLS
    nrows = int(math.ceil(CASES_PER_PAGE / ncols))
    page_count = int(math.ceil(len(cases) / CASES_PER_PAGE))

    for page in range(page_count):
        subset = cases[page * CASES_PER_PAGE : (page + 1) * CASES_PER_PAGE]
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 4.7), dpi=DPI)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax in axes:
            ax.axis("off")

        for j, case in enumerate(subset):
            ax = axes[j]
            ax.axis("on")
            draw_case(ax, case, meta=meta, fallback_idx=page * CASES_PER_PAGE + j,
                      show_text=SHOW_CASE_TEXT, draw_inflated=DRAW_INFLATED_OBSTACLES)

        fig.suptitle(f"{title} | page {page + 1}/{page_count}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(out_dir / f"{prefix}_page_{page + 1:02d}.png", bbox_inches="tight")
        plt.close(fig)


def save_summary_csv(cases: List[Dict[str, Any]], out_dir: Path) -> None:
    import csv
    path = out_dir / "suite_case_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "obstacle_count", "path_type", "start_x", "start_y", "goal_x", "goal_y",
            "path_length_ratio", "approx_min_clearance_to_obstacle", "n_obstacles",
        ])
        for i, c in enumerate(cases):
            start = c.get("start", [None, None])
            goal = c.get("goal", [None, None])
            feas = c.get("feasibility", {}) if isinstance(c.get("feasibility", {}), dict) else {}
            writer.writerow([
                case_id(c, i),
                obstacle_count(c),
                c.get("path_type", ""),
                start[0], start[1], goal[0], goal[1],
                feas.get("path_length_ratio", ""),
                feas.get("approx_min_clearance_to_obstacle", ""),
                len(c.get("obstacles", [])),
            ])


def main() -> None:
    suite_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(SUITE_JSON)
    if not suite_path.exists():
        print(f"Suite JSON not found: {suite_path}")
        print("Edit SUITE_JSON at the top of this script, or pass a path:")
        print("  python visualize_eval_suite.py path/to/suite.json")
        raise SystemExit(1)

    cases, meta = load_suite(suite_path)
    out_dir = Path(OUT_DIR)
    if len(sys.argv) > 2:
        out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(cases)} cases from {suite_path}")
    print(f"Saving visualization to {out_dir}")

    save_summary_csv(cases, out_dir)

    if DRAW_ALL_CONTACT_SHEETS:
        save_contact_pages(cases, meta, out_dir, prefix="all_cases", title="All evaluation cases")

    if DRAW_GROUP_CONTACT_SHEETS:
        groups = sorted(set(obstacle_count(c) for c in cases))
        for g in groups:
            group_cases = [c for c in cases if obstacle_count(c) == g]
            save_contact_pages(group_cases, meta, out_dir, prefix=f"obs_{g}", title=f"Cases with {g} obstacles")

    if DRAW_INDIVIDUAL_CASES:
        save_individual_cases(cases, meta, out_dir)

    print("Done.")
    print(f"Open contact sheets in: {out_dir}")
    print(f"Individual case images: {out_dir / 'individual_cases'}")


if __name__ == "__main__":
    main()
