#!/usr/bin/env python3
"""Plot simulation and field trajectories for ASV paper figures.

This script creates a clean three-panel figure comparing, for each scenario:
  - reference path and obstacle layout from test_run.py
  - simulation trajectory from train_test_asv.py output: asv_data.json -> asv_path
  - field trajectory from either:
      1) udp_live_rl.py logs containing #ACTION rows with x_rl/y_rl, or
      2) raw Bluefin telemetry logs parsed by log_parser.BluefinStreamDecoder

Recommended usage for paper figure:

python plot_trajectory.py \
  --cases 1 2 3 \
  --sim-jsons sim_case1/asv_data.json sim_case2/asv_data.json sim_case3/asv_data.json \
  --field-logs field_case1.log field_case2.log field_case3.log \
  --titles "Straight path" "Slanted path L-R" "Slanted path R-L" \
  --out sim_field_trajectory_comparison.png \
  --out-svg sim_field_trajectory_comparison.svg \

Notes:
  - If the field log comes from udp_live_rl.py, the script will automatically
    use the logged RL-frame coordinates x_rl/y_rl from #ACTION rows. This is
    the most reliable option because it uses the same coordinate mapping used
    during deployment.
  - If the field log is a raw telemetry log without #ACTION rows, the first
    decoded pose is mapped to the scenario start point, matching udp_live_rl.py.
    You can adjust --field-rotation-deg, --field-scale, --field-x-offset, and
    --field-y-offset if needed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

Point = Tuple[float, float]
Polygon = List[Point]


# ---------------------------------------------------------------------------
# Default data and configuration
# ---------------------------------------------------------------------------
DEFAULT_CASES = [1, 2, 3]
DEFAULT_SIM = ["sim_1.json", "sim_2.json", "sim_3.json"]
DEFAULT_FIELD = ["field_1.log", "field_2.log", "field_3.log"]
DEFAULT_TITLES = ["Straight path",
                  "Slanted path L-R",
                  "Slanted path R-L"]
DEFAULT_TEST_RUN = None
DEFAULT_FIELD_SOURCE = "action"
DEFAULT_OUT_PNG = "sim_field_trajectory_comparison.png"
DEFAULT_OUT_SVG = "sim_field_trajectory_comparison.svg"

DEFAULT_SHOW_SHIP_ICON = True
DEFAULT_SHIP_ICON_TARGET = "both"  # "field", "simulation", or "both"
DEFAULT_SHIP_ICON_ZOOM = 2

# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _none_if_missing(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if str(value).strip().lower() in {"", "none", "null", "-"}:
        return None
    return value


def _as_xy_array(points: Any, *, name: str = "points") -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} must be an array/list of [x, y] points; got shape {arr.shape}")
    return arr[:, :2].astype(float)


def _dense_straight_path(start: Point, goal: Point, points_per_m: float = 5.0) -> np.ndarray:
    sx, sy = start
    gx, gy = goal
    n = max(40, int(math.hypot(gx - sx, gy - sy) * points_per_m))
    return np.column_stack([
        np.linspace(sx, gx, n),
        np.linspace(sy, gy, n),
    ]).astype(float)


def _path_length(path: np.ndarray) -> float:
    if path is None or len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))


def _maybe_downsample(path: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(path) <= max_points:
        return path
    idx = np.linspace(0, len(path) - 1, max_points).round().astype(int)
    return path[idx]


def _smooth_xy(path: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(path) < window:
        return path
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.empty_like(path, dtype=float)
    for col in range(2):
        padded = np.pad(path[:, col], (pad, pad), mode="edge")
        out[:, col] = np.convolve(padded, kernel, mode="valid")
    return out

def _add_ship_marker(
    ax,
    path: np.ndarray,
    *,
    length: float = 0.85,
    width: float = 0.32,
    zoom: float = 1.0,
    facecolor: str = "white",
    edgecolor: str = "0.15",
    zorder: int = 10,
) -> None:
    """Draw a simple top-down ASV hull at the final point of a trajectory."""
    if path is None or len(path) < 2:
        return

    p_end = path[-1, :2]

    # Find last non-zero direction.
    direction = None
    for i in range(len(path) - 2, -1, -1):
        d = p_end - path[i, :2]
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            direction = d / n
            break

    if direction is None:
        direction = np.array([0.0, 1.0], dtype=float)

    # Forward and left vectors in world coordinates.
    fwd = direction
    left = np.array([-fwd[1], fwd[0]], dtype=float)

    L = float(length) * zoom
    W = float(width) * zoom

    # Simple boat shape: pointed bow, rectangular stern.
    local = [
        (+0.55 * L,  0.0),       # bow
        (+0.10 * L, +0.50 * W),
        (-0.45 * L, +0.50 * W),
        (-0.45 * L, -0.50 * W),
        (+0.10 * L, -0.50 * W),
    ]

    pts = []
    for xf, yl in local:
        p = p_end + xf * fwd + yl * left
        pts.append((float(p[0]), float(p[1])))

    hull = MplPolygon(
        pts,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        zorder=zorder,
    )
    ax.add_patch(hull)

    # Small centre line to make it visually read as a vessel.
    stern = p_end - 0.30 * L * fwd
    bow = p_end + 0.35 * L * fwd
    ax.plot(
        [stern[0], bow[0]],
        [stern[1], bow[1]],
        color=edgecolor,
        linewidth=0.8,
        zorder=zorder + 1,
    )


# ---------------------------------------------------------------------------
# test_run.py loading
# ---------------------------------------------------------------------------

def _import_test_run(test_run_path: Optional[str]):
    if test_run_path is None:
        try:
            import test_run  # type: ignore
            return test_run
        except Exception as exc:
            raise ImportError(
                "Could not import test_run.py from the current directory. "
                "Use --test-run /path/to/test_run.py."
            ) from exc

    test_run_path = os.path.abspath(test_run_path)
    spec = importlib.util.spec_from_file_location("test_run_for_plot", test_run_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load test_run.py from {test_run_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@dataclass
class ScenarioLayout:
    case_id: int
    start: Point
    goal: Point
    path: np.ndarray
    obstacles: List[Polygon]
    description: str = ""


def load_scenario_from_test_run(case_id: int, *, test_run_path: Optional[str] = None) -> ScenarioLayout:
    module = _import_test_run(test_run_path)
    tc = module.TestCase()

    sx, sy, gx, gy = tc.position(int(case_id))
    start = (float(sx), float(sy))
    goal = (float(gx), float(gy))

    obstacles_raw = tc.obstacles(int(case_id)) if hasattr(tc, "obstacles") else []
    obstacles: List[Polygon] = []
    for obs in obstacles_raw:
        arr = _as_xy_array(obs, name=f"obstacle in case {case_id}")
        obstacles.append([(float(x), float(y)) for x, y in arr])

    path: Optional[np.ndarray] = None
    if hasattr(tc, "path"):
        try:
            p = tc.path(int(case_id))
            if p is not None and len(p) >= 2:
                path = _as_xy_array(p, name=f"path for case {case_id}")
        except Exception:
            path = None
    if path is None:
        path = _dense_straight_path(start, goal)

    desc = ""
    if hasattr(tc, "description"):
        try:
            desc = str(tc.description(int(case_id)))
        except Exception:
            desc = ""

    return ScenarioLayout(case_id=int(case_id), start=start, goal=goal, path=path, obstacles=obstacles, description=desc)


# ---------------------------------------------------------------------------
# Simulation JSON loading
# ---------------------------------------------------------------------------

def load_sim_asv_path(json_path: Optional[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    json_path = _none_if_missing(json_path)
    if json_path is None:
        return np.zeros((0, 2), dtype=float), {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Simulation JSON must contain an object/dict: {json_path}")

    # train_test_asv.py writes env.asv_path here.
    candidates = ["asv_path", "trajectory", "traj", "path_world"]
    for key in candidates:
        if key in data and data[key] is not None:
            return _as_xy_array(data[key], name=f"{key} from {json_path}"), data
    raise KeyError(
        f"Could not find a trajectory in {json_path}. Expected one of: {candidates}. "
        "For train_test_asv.py output, use the asv_data.json file containing 'asv_path'."
    )


# ---------------------------------------------------------------------------
# Field log loading
# ---------------------------------------------------------------------------

_ACTION_PAIR_RE = re.compile(r"(?:^|,)([A-Za-z0-9_]+)=('(?:[^']*)'|[^,]*)")


def _parse_action_line(line: str) -> Dict[str, str]:
    """Parse #ACTION CSV-like key=value line.

    Handles quoted cmd='$CMD,5.00,50.00' without breaking on the internal comma.
    """
    if line.startswith("#ACTION,"):
        line = line[len("#ACTION,"):]
    out: Dict[str, str] = {}
    for m in _ACTION_PAIR_RE.finditer(line):
        key = m.group(1)
        val = m.group(2).strip()
        if len(val) >= 2 and val[0] == "'" and val[-1] == "'":
            val = val[1:-1]
        out[key] = val
    return out


def load_field_from_action_log(
    log_path: str,
    *,
    case_id: Optional[int] = None,
    max_points: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pts: List[Point] = []
    rows = 0
    used = 0
    first_case: Optional[int] = None
    last_case: Optional[int] = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#ACTION"):
                continue
            rows += 1
            d = _parse_action_line(line.strip())
            try:
                row_case = int(float(d.get("test_case", "nan")))
            except Exception:
                row_case = None  # type: ignore[assignment]
            if row_case is not None:
                if first_case is None:
                    first_case = row_case
                last_case = row_case
            if case_id is not None and row_case is not None and row_case != int(case_id):
                continue
            if "x_rl" not in d or "y_rl" not in d:
                continue
            try:
                x = float(d["x_rl"])
                y = float(d["y_rl"])
            except Exception:
                continue
            pts.append((x, y))
            used += 1

    arr = np.asarray(pts, dtype=float).reshape((-1, 2)) if pts else np.zeros((0, 2), dtype=float)
    arr = _maybe_downsample(arr, max_points)
    meta = {
        "source": "action_log",
        "action_rows": rows,
        "used_rows": used,
        "first_case": first_case,
        "last_case": last_case,
    }
    return arr, meta


def load_field_from_raw_log(
    log_path: str,
    *,
    start_xy: Point,
    pos_scale: float = 1.0,
    rotation_deg: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    max_points: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        from log_parser import BluefinStreamDecoder  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Raw log parsing requires log_parser.py in the current directory. "
            "For udp_live_rl.py logs, use --field-source action or auto."
        ) from exc

    decoder = BluefinStreamDecoder(lidar_out_beams=720)
    frames = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            frame = decoder.feed(line)
            if frame is not None:
                frames.append(frame)

    if not frames:
        return np.zeros((0, 2), dtype=float), {"source": "raw_log", "frames": 0}

    rx0, ry0 = float(frames[0].x_m), float(frames[0].y_m)
    c = math.cos(math.radians(rotation_deg))
    s = math.sin(math.radians(rotation_deg))
    sx, sy = start_xy

    pts: List[Point] = []
    for fr in frames:
        dx = float(fr.x_m) - rx0
        dy = float(fr.y_m) - ry0
        # Optional rotation in the horizontal plane.
        dxr = c * dx - s * dy
        dyr = s * dx + c * dy
        x = float(sx + pos_scale * dxr + x_offset)
        y = float(sy + pos_scale * dyr + y_offset)
        pts.append((x, y))

    arr = np.asarray(pts, dtype=float).reshape((-1, 2))
    arr = _maybe_downsample(arr, max_points)
    return arr, {"source": "raw_log", "frames": len(frames)}


def load_field_trajectory(
    log_path: Optional[str],
    *,
    case_id: int,
    start_xy: Point,
    source: str = "auto",
    pos_scale: float = 1.0,
    rotation_deg: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    smooth_window: int = 1,
    max_points: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    log_path = _none_if_missing(log_path)
    if log_path is None:
        return np.zeros((0, 2), dtype=float), {"source": "none"}

    source = str(source).lower()
    if source not in {"auto", "action", "raw"}:
        raise ValueError("--field-source must be one of: auto, action, raw")

    if source in {"auto", "action"}:
        arr, meta = load_field_from_action_log(log_path, case_id=case_id, max_points=max_points)
        # If the log contains #ACTION rows, trust that it is a deployment log.
        # Do not silently fall back to raw telemetry when the case_id does not
        # match, because that can overlay the wrong run on the wrong scenario.
        if len(arr) > 0 or source == "action" or int(meta.get("action_rows", 0)) > 0:
            return _smooth_xy(arr, smooth_window), meta

    arr, meta = load_field_from_raw_log(
        log_path,
        start_xy=start_xy,
        pos_scale=pos_scale,
        rotation_deg=rotation_deg,
        x_offset=x_offset,
        y_offset=y_offset,
        max_points=max_points,
    )
    return _smooth_xy(arr, smooth_window), meta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_layout(
    ax,
    layout: ScenarioLayout,
    *,
    map_width: float,
    map_height: float,
    show_grid: bool,
) -> None:
    # Workspace boundary
    ax.add_patch(Rectangle((0, 0), map_width, map_height, fill=False, edgecolor="0.15", linewidth=1.2))

    # Obstacles
    for obs in layout.obstacles:
        if len(obs) >= 3:
            patch = MplPolygon(obs, closed=True, facecolor="0.82", edgecolor="0.25", linewidth=1.0, zorder=2)
            ax.add_patch(patch)

    # Reference path
    if layout.path is not None and len(layout.path) >= 2:
        ax.plot(layout.path[:, 0], layout.path[:, 1], linestyle="--", color="0.05", linewidth=1.6, label="Reference path", zorder=3)

    # Start and goal markers
    ax.scatter([layout.start[0]], [layout.start[1]], s=42, marker="o", facecolor="white", edgecolor="0.05", linewidth=1.0, zorder=6)
    ax.scatter([layout.goal[0]], [layout.goal[1]], s=64, marker="*", facecolor="0.05", edgecolor="0.05", linewidth=0.8, zorder=6)
    ax.text(layout.start[0], layout.start[1] - 0.55, "Start", ha="center", va="top", fontsize=8)
    ax.text(layout.goal[0], layout.goal[1] + 0.55, "Goal", ha="center", va="bottom", fontsize=8)

    ax.set_xlim(-0.25, map_width + 0.25)
    ax.set_ylim(-0.25, map_height + 0.25)
    ax.set_aspect("equal", adjustable="box")
    if show_grid:
        ax.grid(True, linewidth=0.4, alpha=0.25)
    else:
        ax.grid(False)
    ax.tick_params(labelsize=8)


def plot_comparison(
    layouts: Sequence[ScenarioLayout],
    sim_paths: Sequence[np.ndarray],
    field_paths: Sequence[np.ndarray],
    *,
    titles: Sequence[str],
    map_width: float,
    map_height: float,
    out_png: str,
    out_svg: Optional[str] = None,
    dpi: int = 300,
    show_grid: bool = False,
    figure_title: Optional[str] = None,
    show_ship_icon: bool = DEFAULT_SHOW_SHIP_ICON,
    ship_icon_target: str = DEFAULT_SHIP_ICON_TARGET,
    ship_icon_zoom: float = DEFAULT_SHIP_ICON_ZOOM,
) -> None:
    n = len(layouts)
    if n != 3:
        print(f"[WARN] Expected 3 scenarios for the paper figure, got {n}.")

    fig_w = 8.0 if n == 3 else max(3.6 * n, 4.0)
    fig_h = 4.7
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), constrained_layout=False)
    if n == 1:
        axes = [axes]

    for i, (ax, layout, sim, field) in enumerate(zip(axes, layouts, sim_paths, field_paths)):
        _draw_layout(ax, layout, map_width=map_width, map_height=map_height, show_grid=show_grid)

        if sim is not None and len(sim) >= 2:
            ax.plot(sim[:, 0], sim[:, 1], color="#1f77b4", linewidth=2.0, label="Simulation", zorder=4)
            ax.scatter([sim[0, 0]], [sim[0, 1]], s=18, color="#1f77b4", zorder=5)
            ax.scatter([sim[-1, 0]], [sim[-1, 1]], s=24, color="#1f77b4", marker="s", zorder=5)
            if show_ship_icon and ship_icon_target in {"simulation", "both"}:
                _add_ship_marker(ax, sim, zoom=ship_icon_zoom, facecolor="white", edgecolor="#1f77b4", zorder=9)

        if field is not None and len(field) >= 2:
            ax.plot(field[:, 0], field[:, 1], color="#d62728", linewidth=2.0, label="Field", zorder=5)
            ax.scatter([field[0, 0]], [field[0, 1]], s=18, color="#d62728", zorder=6)
            ax.scatter([field[-1, 0]], [field[-1, 1]], s=24, color="#d62728", marker="s", zorder=6)
            if show_ship_icon and ship_icon_target in {"field", "both"}:
                _add_ship_marker(ax, field, zoom=ship_icon_zoom, facecolor="white", edgecolor="#d62728", zorder=10)

        title = titles[i] if i < len(titles) and titles[i] else f"Case {layout.case_id}"
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("x (m)", fontsize=9)
        if i == 0:
            ax.set_ylabel("y (m)", fontsize=9)
        else:
            ax.set_ylabel("")

    legend_handles = [
        Line2D([0], [0], linestyle="--", color="0.05", linewidth=1.6, label="Reference path"),
        Line2D([0], [0], color="#1f77b4", linewidth=2.0, label="Simulation"),
        Line2D([0], [0], color="#d62728", linewidth=2.0, label="Field experiment"),
        Rectangle((0, 0), 1, 1, facecolor="0.82", edgecolor="0.25", label="Static obstacle"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="0.05", label="Start"),
        Line2D([0], [0], marker="*", linestyle="None", color="0.05", markersize=9, label="Goal"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="0.4",
        fontsize=8,
        bbox_to_anchor=(0.5, 0.02),
    )

    if figure_title:
        fig.suptitle(figure_title, fontsize=12, fontweight="bold", y=0.98)
        top = 0.90
    else:
        top = 0.94
    fig.subplots_adjust(left=0.06, right=0.99, top=top, bottom=0.16, wspace=0.20)

    out_png = str(out_png)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"[SAVED] {out_png}")
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight")
        print(f"[SAVED] {out_svg}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _validate_three(name: str, values: Sequence[Any], expected: int) -> List[Any]:
    if len(values) != expected:
        raise ValueError(f"{name} must have {expected} values, got {len(values)}: {values}")
    return list(values)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ASV simulation vs field trajectories for three paper scenarios.")
    ap.add_argument("--cases", type=int, nargs="+", default=DEFAULT_CASES, help="Test-case IDs from test_run.py. Usually three values, e.g. 1 2 3.")
    ap.add_argument("--sim-jsons", nargs="+", default=DEFAULT_SIM, help="Simulation asv_data.json files. Use 'none' to skip a scenario.")
    ap.add_argument("--field-logs", nargs="+", default=DEFAULT_FIELD, help="Field logs. Use 'none' to skip a scenario.")
    ap.add_argument("--titles", nargs="*", default=None, help="Panel titles. Defaults to Case N.")
    ap.add_argument("--test-run", default=None, help="Path to test_run.py. Defaults to importing test_run from current directory.")

    ap.add_argument("--field-source", choices=["auto", "action", "raw"], default="auto", help="Field log source. auto prefers #ACTION x_rl/y_rl, then raw telemetry.")
    ap.add_argument("--field-scale", type=float, default=1.0, help="Scale factor for raw telemetry logs when #ACTION is absent.")
    ap.add_argument("--field-rotation-deg", type=float, default=0.0, help="Rotation applied to raw telemetry logs when #ACTION is absent.")
    ap.add_argument("--field-x-offset", type=float, default=0.0, help="Extra x offset for raw telemetry logs when #ACTION is absent.")
    ap.add_argument("--field-y-offset", type=float, default=0.0, help="Extra y offset for raw telemetry logs when #ACTION is absent.")
    ap.add_argument("--field-smooth-window", type=int, default=1, help="Optional moving-average window for field trajectory. Use 1 for no smoothing.")
    ap.add_argument("--max-field-points", type=int, default=0, help="Downsample field trajectory to at most this many points. 0 disables.")
    ap.add_argument("--max-sim-points", type=int, default=0, help="Downsample simulation trajectory to at most this many points. 0 disables.")

    ap.add_argument("--map-width", type=float, default=10.0)
    ap.add_argument("--map-height", type=float, default=25.0)
    ap.add_argument("--grid", action="store_true", help="Show light grid lines.")
    ap.add_argument("--figure-title", default=None)
    ap.add_argument("--out", default="sim_field_trajectory_comparison.png", help="Output PNG path.")
    ap.add_argument("--out-svg", default=None, help="Optional SVG output path.")
    ap.add_argument("--dpi", type=int, default=300)

    args = ap.parse_args()

    n = len(args.cases)
    sim_jsons = _validate_three("--sim-jsons", args.sim_jsons, n)
    field_logs = _validate_three("--field-logs", args.field_logs, n)
    if args.titles is None or len(args.titles) == 0:
        titles = [f"Case {c}" for c in args.cases]
    else:
        titles = _validate_three("--titles", args.titles, n)

    layouts: List[ScenarioLayout] = []
    sim_paths: List[np.ndarray] = []
    field_paths: List[np.ndarray] = []

    for case_id, sim_json, field_log in zip(args.cases, sim_jsons, field_logs):
        layout = load_scenario_from_test_run(case_id, test_run_path=args.test_run)
        layouts.append(layout)

        sim_path, sim_meta = load_sim_asv_path(sim_json)
        sim_path = _maybe_downsample(sim_path, args.max_sim_points)
        sim_paths.append(sim_path)

        field_path, field_meta = load_field_trajectory(
            field_log,
            case_id=None,
            start_xy=layout.start,
            source=args.field_source,
            pos_scale=args.field_scale,
            rotation_deg=args.field_rotation_deg,
            x_offset=args.field_x_offset,
            y_offset=args.field_y_offset,
            smooth_window=args.field_smooth_window,
            max_points=args.max_field_points,
        )
        field_paths.append(field_path)

        print(
            f"[LOAD] case={case_id} sim_points={len(sim_path)} field_points={len(field_path)} "
            f"field_source={field_meta.get('source')} title={titles[len(layouts)-1]!r}"
        )
        if field_meta.get("source") == "action_log" and int(field_meta.get("used_rows", 0)) == 0:
            print(f"[WARN] No #ACTION x_rl/y_rl rows used for case {case_id} in {field_log}")

    plot_comparison(
        layouts,
        sim_paths,
        field_paths,
        titles=titles,
        map_width=float(args.map_width),
        map_height=float(args.map_height),
        out_png=args.out,
        out_svg=args.out_svg,
        dpi=int(args.dpi),
        show_grid=bool(args.grid),
        figure_title=args.figure_title,
    )


if __name__ == "__main__":
    main()
