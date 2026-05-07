"""Plot ASV trajectory JSON exported by train_test_asv.py.

This version is for the current small-map environment (e.g. 10 x 25 m).
It auto-scales the axes instead of using the old 400 x 600 limits.

Typical usage:
    python plot_trajectory.py asv_data.json --map-width 10 --map-height 25 --output traj.png

Multiple trajectories:
    python plot_trajectory.py run1.json run2.json --labels "Stage 7C" "Stage 7D" --output compare.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

try:
    from PIL import Image
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from images import BOAT_ICON
    _HAS_ICON = True
except Exception:
    _HAS_ICON = False

try:
    from ship_model import VESSEL_LENGTH, VESSEL_WIDTH, HULL_MARGIN, HULL_FORWARD_SHIFT
except Exception:
    VESSEL_LENGTH = 1.725
    VESSEL_WIDTH = 0.50
    HULL_MARGIN = 0.15
    HULL_FORWARD_SHIFT = 0.0

Color = Tuple[float, float, float]

def load_data(filepath: str | Path) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)

def as_xy_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    return arr[:, :2]

def hull_polygon_world(x: float, y: float, heading_deg: float) -> List[Tuple[float, float]]:
    """Match rl_env.py hull rectangle convention: heading 0 points +y."""
    L = VESSEL_LENGTH + 2.0 * HULL_MARGIN
    W = VESSEL_WIDTH + 2.0 * HULL_MARGIN
    half_L = 0.5 * L
    half_W = 0.5 * W
    shift = HULL_FORWARD_SHIFT

    h = math.radians(float(heading_deg))
    sin_h = math.sin(h)
    cos_h = math.cos(h)

    local = [
        (+half_L + shift, +half_W),
        (+half_L + shift, -half_W),
        (-half_L + shift, -half_W),
        (-half_L + shift, +half_W),
    ]

    poly = []
    for x_forward, y_left in local:
        wx = x + x_forward * sin_h - y_left * cos_h
        wy = y + x_forward * cos_h + y_left * sin_h
        poly.append((wx, wy))
    return poly

def draw_boat_icon(ax, x: float, y: float, heading_deg: float, zoom: float = 0.25) -> None:
    """Draw optional boat icon; hull polygon is usually more accurate for papers."""
    if not _HAS_ICON:
        return
    try:
        boat_img = Image.frombytes(BOAT_ICON["format"], BOAT_ICON["size"], BOAT_ICON["bytes"])
        # Image points upward by default; heading 0 = +y, so use -heading like rl_env render.
        rotated_img = boat_img.rotate(-float(heading_deg), expand=True, resample=Image.BICUBIC)
        imgbox = OffsetImage(rotated_img, zoom=zoom)
        ab = AnnotationBbox(imgbox, (x, y), frameon=False, zorder=10)
        ax.add_artist(ab)
    except Exception:
        pass

def draw_hull(ax, x: float, y: float, heading_deg: float, *, color: str = "black") -> None:
    poly = hull_polygon_world(x, y, heading_deg)
    patch = Polygon(poly, closed=True, fill=False, edgecolor=color, linewidth=1.5, zorder=11)
    ax.add_patch(patch)
    # Heading arrow/nose line.
    h = math.radians(float(heading_deg))
    nose = (x + 0.75 * VESSEL_LENGTH * math.sin(h), y + 0.75 * VESSEL_LENGTH * math.cos(h))
    ax.plot([x, nose[0]], [y, nose[1]], color=color, linewidth=1.2, zorder=12)

def infer_map_size(all_data: Sequence[dict], default_w: float, default_h: float) -> Tuple[float, float]:
    # Prefer exported map size if present.
    for d in all_data:
        if "map_width" in d and "map_height" in d:
            return float(d["map_width"]), float(d["map_height"])
        if "map_size" in d and len(d["map_size"]) >= 2:
            return float(d["map_size"][0]), float(d["map_size"][1])
    return float(default_w), float(default_h)

def plot_trajectories(
    files: Sequence[str],
    labels: Sequence[str] | None = None,
    map_width: float = 10.0,
    map_height: float = 25.0,
    output: str = "trajectory.png",
    title: str | None = None,
    invert_y: bool = False,
    no_icon: bool = False,
    draw_hull_geom: bool = True,
    margin: float = 0.5,
    show: bool = False,
) -> None:
    if not files:
        raise ValueError("At least one JSON trajectory file is required.")

    data_list = [load_data(f) for f in files]
    W, H = infer_map_size(data_list, map_width, map_height)

    if labels is None or len(labels) == 0:
        labels = [Path(f).stem for f in files]
    if len(labels) != len(files):
        raise ValueError("Number of --labels must match number of input files.")

    # Use the first file as static environment reference.
    ref = data_list[0]
    start = ref.get("start", None)
    goal = ref.get("goal", None)
    obstacles = ref.get("obstacles", [])
    path = as_xy_array(ref.get("path", []))

    # Figure size based on map aspect. This is the important fix for 10x25.
    base_w = 5.0
    fig_h = max(5.0, base_w * H / max(W, 1e-6))
    fig, ax = plt.subplots(figsize=(base_w, fig_h), constrained_layout=True)

    # Pool/map boundary.
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, edgecolor="0.35", linewidth=1.4, zorder=1))

    # Obstacles.
    for obs in obstacles:
        obs_arr = as_xy_array(obs)
        if len(obs_arr) >= 3:
            ax.add_patch(Polygon(obs_arr, closed=True, facecolor="tab:red", edgecolor="darkred", alpha=0.45, linewidth=1.0, zorder=3))

    # Reference path.
    if len(path) >= 2:
        ax.plot(path[:, 0], path[:, 1], color="black", linestyle="--", linewidth=1.8, label="Reference path", zorder=4)

    if start is not None:
        ax.scatter([start[0]], [start[1]], color="tab:green", s=55, label="Start", zorder=6)
    if goal is not None:
        ax.scatter([goal[0]], [goal[1]], color="tab:red", s=55, label="Goal", zorder=6)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue", "tab:orange", "tab:purple", "tab:brown"])
    linestyles = ["-", "-.", ":", "--"]

    for i, (label, d) in enumerate(zip(labels, data_list)):
        asv_path = as_xy_array(d.get("asv_path", []))
        if len(asv_path) < 2:
            print(f"Warning: {label} has no asv_path or too few points.")
            continue
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        ax.plot(asv_path[:, 0], asv_path[:, 1], label=label, color=color, linestyle=ls, linewidth=2.0, zorder=5)

        final_x, final_y = float(asv_path[-1, 0]), float(asv_path[-1, 1])
        heading = float(d.get("heading", d.get("final_heading", 0.0)))
        if not no_icon:
            draw_boat_icon(ax, final_x, final_y, heading, zoom=1)
        if draw_hull_geom:
            draw_hull(ax, final_x, final_y, heading, color=color)

    ax.set_xlim(-margin, W + margin)
    if invert_y:
        ax.set_ylim(H + margin, -margin)
    else:
        ax.set_ylim(-margin, H + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, color="#dddddd", alpha=0.8)
    ax.set_title(title or "ASV trajectory")
    ax.legend(loc="best", fontsize=8)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved plot -> {output}")
    if show:
        plt.show()
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot current ASV trajectory JSON files from train_test_asv.py test mode.")
    ap.add_argument("files", nargs="+", help="Trajectory JSON files, e.g. asv_data.json")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels, one per file")
    ap.add_argument("--map-width", type=float, default=10.0)
    ap.add_argument("--map-height", type=float, default=25.0)
    ap.add_argument("--output", "-o", default="trajectory.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--invert-y", action="store_true", help="Invert y-axis if you want image-coordinate style")
    ap.add_argument("--no-icon", action="store_true", help="Do not draw boat icon at final pose")
    ap.add_argument("--no-hull", action="store_true", help="Do not draw final hull collision geometry")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--show", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    plot_trajectories(
        files=args.files,
        labels=args.labels,
        map_width=args.map_width,
        map_height=args.map_height,
        output=args.output,
        title=args.title,
        invert_y=args.invert_y,
        no_icon=args.no_icon,
        draw_hull_geom=not args.no_hull,
        margin=args.margin,
        show=args.show,
    )
