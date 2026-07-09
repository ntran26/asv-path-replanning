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
  
python plot_trajectory_updated_with_metrics.py \
  --cases 1 2 3 \
  --sim-jsons data/sim_1.json data/sim_2.json data/sim_3.json \
  --field-logs data/field_1.log data/field_2.log data/field_3.log \
  --titles "Scenario 1" "Scenario 2" "Scenario 3" \
  --test-run test_run.py \
  --out sim_field_trajectory_comparison.png \
  --out-svg sim_field_trajectory_comparison.svg \
  --metrics-csv trajectory_metrics.csv \
  --metrics-md trajectory_metrics.md \
  --mismatch-csv trajectory_mismatch.csv

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
import csv
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
DEFAULT_SIM = ["data/sim_1.json", "data/sim_2.json", "data/sim_3.json"]
DEFAULT_FIELD = ["data/field_1.log", "data/field_2.log", "data/field_3.log"]
DEFAULT_TITLES = ["Scenario 1",
                  "Scenario 2",
                  "Scenario 3"]
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




def load_layout_from_sim_json(case_id: int, sim_meta: Dict[str, Any], fallback: ScenarioLayout) -> ScenarioLayout:
    """Build a ScenarioLayout from sim JSON fields when available; otherwise use fallback."""
    try:
        start_raw = sim_meta.get("start", fallback.start)
        goal_raw = sim_meta.get("goal", fallback.goal)
        path_raw = sim_meta.get("path", fallback.path)
        obstacles_raw = sim_meta.get("obstacles", fallback.obstacles)
        start = (float(start_raw[0]), float(start_raw[1]))
        goal = (float(goal_raw[0]), float(goal_raw[1]))
        path = _as_xy_array(path_raw, name=f"path from sim JSON case {case_id}")
        obstacles: List[Polygon] = []
        for obs in obstacles_raw:
            arr = _as_xy_array(obs, name=f"obstacle from sim JSON case {case_id}")
            obstacles.append([(float(x), float(y)) for x, y in arr])
        return ScenarioLayout(case_id=int(case_id), start=start, goal=goal, path=path, obstacles=obstacles, description=fallback.description)
    except Exception as exc:
        print(f"[WARN] Could not use layout from sim JSON for case {case_id}: {exc}; using test_run layout.")
        return fallback


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
# Trajectory metrics
# ---------------------------------------------------------------------------

def _point_segment_distance_and_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Return distance from point p to segment ab, closest point, and segment parameter t in [0,1]."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        q = a.copy()
        return float(np.linalg.norm(p - q)), q, 0.0
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), q, t


def _signed_cte_to_path(points: np.ndarray, ref_path: np.ndarray) -> np.ndarray:
    """Signed cross-track error to a polyline reference path.

    Sign is based on the local segment tangent using the 2-D cross product.
    Magnitude is the minimum Euclidean distance to the polyline.
    """
    if points is None or len(points) == 0 or ref_path is None or len(ref_path) < 2:
        return np.asarray([], dtype=float)
    pts = np.asarray(points[:, :2], dtype=float)
    path = np.asarray(ref_path[:, :2], dtype=float)
    out: list[float] = []
    for p in pts:
        best_d = float("inf")
        best_signed = 0.0
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            d, q, _ = _point_segment_distance_and_projection(p, a, b)
            if d < best_d:
                tvec = b - a
                e = p - q
                cross_z = float(tvec[0] * e[1] - tvec[1] * e[0])
                sign = 1.0 if cross_z >= 0.0 else -1.0
                best_d = d
                best_signed = sign * d
        out.append(best_signed)
    return np.asarray(out, dtype=float)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        if ((yi > y) != (yj > y)):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _point_to_polygon_distance(point: np.ndarray, polygon_points: Sequence[Point]) -> float:
    """Distance from point to polygon boundary. Negative if point is inside polygon."""
    poly = np.asarray(polygon_points, dtype=float)
    if len(poly) < 2:
        return float("nan")
    min_d = float("inf")
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        d, _, _ = _point_segment_distance_and_projection(point, a, b)
        min_d = min(min_d, d)
    if len(poly) >= 3 and _point_in_polygon(point, poly):
        return -min_d
    return min_d


def _min_obstacle_clearance(path: np.ndarray, obstacles: Sequence[Polygon], vessel_radius: float = 0.0) -> float:
    """Minimum centre-point clearance to all obstacle polygon boundaries, optionally minus vessel_radius."""
    if path is None or len(path) == 0 or not obstacles:
        return float("nan")
    min_clearance = float("inf")
    for p in path[:, :2]:
        for obs in obstacles:
            d = _point_to_polygon_distance(p, obs)
            if math.isfinite(d):
                min_clearance = min(min_clearance, d)
    return (min_clearance - vessel_radius) if math.isfinite(min_clearance) else float("nan")


def _min_boundary_clearance(path: np.ndarray, map_width: float, map_height: float, vessel_radius: float = 0.0) -> float:
    """Minimum centre-point clearance to rectangular workspace boundary."""
    if path is None or len(path) == 0:
        return float("nan")
    x = path[:, 0]
    y = path[:, 1]
    vals = np.column_stack([x, map_width - x, y, map_height - y])
    return float(np.nanmin(vals) - vessel_radius)


def _resample_by_arclength(path: np.ndarray, n_samples: int = 200) -> np.ndarray:
    """Resample a trajectory by normalized arc length."""
    if path is None or len(path) == 0:
        return np.zeros((0, 2), dtype=float)
    if len(path) == 1:
        return np.repeat(path[:, :2], n_samples, axis=0)
    pts = np.asarray(path[:, :2], dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.insert(np.cumsum(seg), 0, 0.0)
    total = float(cum[-1])
    if total <= 1e-12:
        return np.repeat(pts[:1], n_samples, axis=0)
    target = np.linspace(0.0, total, n_samples)
    x = np.interp(target, cum, pts[:, 0])
    y = np.interp(target, cum, pts[:, 1])
    return np.column_stack([x, y])


def _trajectory_summary(
    path: np.ndarray,
    layout: ScenarioLayout,
    *,
    domain: str,
    map_width: float,
    map_height: float,
    vessel_radius: float = 0.0,
) -> Dict[str, Any]:
    """Compute trajectory metrics from the plotted path only."""
    cte = _signed_cte_to_path(path, layout.path)
    abs_cte = np.abs(cte)
    ref_len = _path_length(layout.path)
    traj_len = _path_length(path)
    final_goal_error = float(np.linalg.norm(path[-1, :2] - np.asarray(layout.goal))) if len(path) else float("nan")
    return {
        "scenario": layout.case_id,
        "domain": domain,
        "points": int(len(path)),
        "mean_abs_cte_m": float(np.nanmean(abs_cte)) if len(abs_cte) else float("nan"),
        "std_cte_m": float(np.nanstd(cte)) if len(cte) else float("nan"),
        "rms_cte_m": float(np.sqrt(np.nanmean(cte ** 2))) if len(cte) else float("nan"),
        "max_abs_cte_m": float(np.nanmax(abs_cte)) if len(abs_cte) else float("nan"),
        "path_length_m": traj_len,
        "path_length_ratio": float(traj_len / ref_len) if ref_len > 1e-12 else float("nan"),
        "min_obstacle_clearance_m": _min_obstacle_clearance(path, layout.obstacles, vessel_radius=vessel_radius),
        "min_boundary_clearance_m": _min_boundary_clearance(path, map_width, map_height, vessel_radius=vessel_radius),
        "final_goal_error_m": final_goal_error,
    }


def _sim_field_summary(sim: np.ndarray, field: np.ndarray, layout: ScenarioLayout, n_samples: int = 200) -> Dict[str, Any]:
    """Compute paired sim-field separation after arc-length resampling of both trajectories."""
    sim_r = _resample_by_arclength(sim, n_samples)
    field_r = _resample_by_arclength(field, n_samples)
    if len(sim_r) == 0 or len(field_r) == 0:
        sep = np.asarray([], dtype=float)
    else:
        n = min(len(sim_r), len(field_r))
        sep = np.linalg.norm(sim_r[:n] - field_r[:n], axis=1)
    return {
        "scenario": layout.case_id,
        "mean_sim_field_separation_m": float(np.nanmean(sep)) if len(sep) else float("nan"),
        "max_sim_field_separation_m": float(np.nanmax(sep)) if len(sep) else float("nan"),
        "resample_points": int(len(sep)),
    }


def _format_float(v: Any, ndigits: int = 3) -> str:
    try:
        f = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(f):
        return ""
    return f"{f:.{ndigits}f}"


def write_metrics_files(
    layouts: Sequence[ScenarioLayout],
    sim_paths: Sequence[np.ndarray],
    field_paths: Sequence[np.ndarray],
    *,
    map_width: float,
    map_height: float,
    metrics_csv: Optional[str] = None,
    mismatch_csv: Optional[str] = None,
    metrics_md: Optional[str] = None,
    vessel_radius: float = 0.0,
    separation_points: int = 200,
) -> None:
    """Write trajectory metrics to CSV and/or Markdown files."""
    rows: list[Dict[str, Any]] = []
    mismatch_rows: list[Dict[str, Any]] = []
    for layout, sim, field in zip(layouts, sim_paths, field_paths):
        if sim is not None and len(sim):
            rows.append(_trajectory_summary(sim, layout, domain="Simulation", map_width=map_width, map_height=map_height, vessel_radius=vessel_radius))
        if field is not None and len(field):
            rows.append(_trajectory_summary(field, layout, domain="Field", map_width=map_width, map_height=map_height, vessel_radius=vessel_radius))
        if sim is not None and field is not None and len(sim) and len(field):
            mismatch_rows.append(_sim_field_summary(sim, field, layout, n_samples=separation_points))

    metric_fields = [
        "scenario", "domain", "points", "mean_abs_cte_m", "std_cte_m", "rms_cte_m",
        "max_abs_cte_m", "path_length_m", "path_length_ratio",
        "min_obstacle_clearance_m", "min_boundary_clearance_m", "final_goal_error_m",
    ]
    mismatch_fields = ["scenario", "mean_sim_field_separation_m", "max_sim_field_separation_m", "resample_points"]

    if metrics_csv:
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=metric_fields)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"[SAVED] {metrics_csv}")

    if mismatch_csv:
        with open(mismatch_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=mismatch_fields)
            w.writeheader()
            for row in mismatch_rows:
                w.writerow(row)
        print(f"[SAVED] {mismatch_csv}")

    if metrics_md:
        lines: list[str] = []
        lines.append("### Trajectory metrics by scenario and domain")
        lines.append("")
        headers = ["Scenario", "Domain", "Mean |CTE| (m)", "Std CTE (m)", "RMS CTE (m)", "Max |CTE| (m)", "Path length (m)", "Length ratio", "Min obs. clearance (m)", "Min boundary clearance (m)"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "---|" * len(headers))
        for row in rows:
            vals = [
                str(row["scenario"]), row["domain"],
                _format_float(row["mean_abs_cte_m"]), _format_float(row["std_cte_m"]),
                _format_float(row["rms_cte_m"]), _format_float(row["max_abs_cte_m"]),
                _format_float(row["path_length_m"]), _format_float(row["path_length_ratio"]),
                _format_float(row["min_obstacle_clearance_m"]), _format_float(row["min_boundary_clearance_m"]),
            ]
            lines.append("| " + " | ".join(vals) + " |")
        if mismatch_rows:
            lines.append("")
            lines.append("### Sim-field trajectory separation")
            lines.append("")
            headers2 = ["Scenario", "Mean separation (m)", "Max separation (m)"]
            lines.append("| " + " | ".join(headers2) + " |")
            lines.append("|" + "---|" * len(headers2))
            for row in mismatch_rows:
                vals = [str(row["scenario"]), _format_float(row["mean_sim_field_separation_m"]), _format_float(row["max_sim_field_separation_m"])]
                lines.append("| " + " | ".join(vals) + " |")
        with open(metrics_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[SAVED] {metrics_md}")

        # Also print to console for quick copy/paste.
        print("\n" + "\n".join(lines))

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ASV simulation vs field trajectories and compute trajectory metrics.")
    ap.add_argument("--test-run", default=None, help="Path to test_run.py. Defaults to importing test_run from current directory.")
    ap.add_argument("--cases", type=int, nargs="+", default=DEFAULT_CASES, help="Scenario/test-case IDs to plot.")
    ap.add_argument("--sim-jsons", nargs="+", default=DEFAULT_SIM, help="Simulation JSON files, one per case.")
    ap.add_argument("--field-logs", nargs="+", default=DEFAULT_FIELD, help="Field log files, one per case.")
    ap.add_argument("--titles", nargs="+", default=DEFAULT_TITLES, help="Panel titles, one per case.")
    ap.add_argument("--layout-source", choices=["test-run", "sim-json"], default="test-run", help="Use layout from test_run.py or from simulation JSON fields.")
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
    ap.add_argument("--metrics-csv", default=None, help="Optional CSV output path for trajectory metrics by scenario/domain.")
    ap.add_argument("--mismatch-csv", default=None, help="Optional CSV output path for sim-field trajectory separation metrics.")
    ap.add_argument("--metrics-md", default=None, help="Optional Markdown output path for paper-ready trajectory metric tables.")
    ap.add_argument("--vessel-radius", type=float, default=0.0, help="Optional vessel radius/margin subtracted from obstacle and boundary clearances (m).")
    ap.add_argument("--separation-points", type=int, default=200, help="Number of arc-length samples for sim-field separation metrics.")

    args = ap.parse_args()

    cases = list(args.cases)
    sim_jsons = list(args.sim_jsons)
    field_logs = list(args.field_logs)
    titles = list(args.titles)
    n = len(cases)
    if not (len(sim_jsons) == len(field_logs) == n):
        raise ValueError("--cases, --sim-jsons, and --field-logs must have the same length.")
    if len(titles) < n:
        titles = titles + [f"Scenario {cid}" for cid in cases[len(titles):]]

    layouts: List[ScenarioLayout] = []
    sim_paths: List[np.ndarray] = []
    field_paths: List[np.ndarray] = []

    for case_id, sim_json, field_log in zip(cases, sim_jsons, field_logs):
        fallback_layout = load_scenario_from_test_run(case_id, test_run_path=args.test_run)

        sim_path, sim_meta = load_sim_asv_path(sim_json)
        layout = load_layout_from_sim_json(case_id, sim_meta, fallback_layout) if args.layout_source == "sim-json" else fallback_layout
        layouts.append(layout)

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

    if args.metrics_csv or args.mismatch_csv or args.metrics_md:
        write_metrics_files(
            layouts,
            sim_paths,
            field_paths,
            map_width=float(args.map_width),
            map_height=float(args.map_height),
            metrics_csv=args.metrics_csv,
            mismatch_csv=args.mismatch_csv,
            metrics_md=args.metrics_md,
            vessel_radius=float(args.vessel_radius),
            separation_points=int(args.separation_points),
        )


if __name__ == "__main__":
    main()
