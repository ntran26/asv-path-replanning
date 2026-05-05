"""
log_viewer.py

Offline Bluefin log replay + policy-observation viewer.

This version is aligned with the current SAC local-planner observation used by
udp_live_rl.py / rl_env.py. It displays both raw telemetry and the features the
policy observes:

    lidar                  : 25 pooled LiDAR sector closeness values
    u                      : body-frame surge velocity
    v                      : body-frame sway velocity
    yaw_rate               : heading/yaw rate
    cross_track_error      : signed path cross-track error
    course_error           : course error relative to the reference path tangent
    lookahead_course_error : course error to a look-ahead path point
    front_clearance        : front pooled LiDAR clearance
    side_clearance_diff    : right_clearance - left_clearance
    local_target_cte       : LiDAR-derived local bypass target
    log10_lambda           : fixed lambda value kept for model compatibility

Controls:
    Space   pause/resume
    F       toggle full 360 LiDAR list vs policy sector list
    M       toggle map panel
    G       toggle follow mode
    C       clear drawn trajectory
    O       reset log origin to current pose
    P       save screenshot
    Up/Down scroll LiDAR text list
    Esc     quit

Typical use:
    python log_viewer.py data/test_1.log --test-case 1
    python log_viewer.py data/test_1.log --test-case 1 --record --out-video replay.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pygame

from log_parser import BluefinFrame, BluefinStreamDecoder
from test_run import TestCase

try:
    from asv_lidar import (
        LIDAR_RANGE,
        LIDAR_SWATH,
        LIDAR_BEAMS,
        LIDAR_SECTORS,
        BEAMS_PER_SECTOR,
        Lidar as PolicyLidar,
    )
    from ship_model import VESSEL_WIDTH, HULL_MARGIN, VESSEL_LENGTH
except Exception:
    # Fallbacks keep the viewer usable if imported from a different folder.
    LIDAR_RANGE = 16.0
    LIDAR_SWATH = 240.0
    LIDAR_BEAMS = 225
    LIDAR_SECTORS = 25
    BEAMS_PER_SECTOR = 9
    PolicyLidar = None
    VESSEL_WIDTH = 0.50
    HULL_MARGIN = 0.15
    VESSEL_LENGTH = 1.725

# ---------------------------------------------------------------------------
# Policy-observation constants. Keep aligned with udp_live_rl.py / rl_env.py.
# ---------------------------------------------------------------------------
LIDAR_FULL_BEAMS = 720
LIDAR_FULL_STEP = 360.0 / LIDAR_FULL_BEAMS
LIDAR_INDEX_DEG = 0.0
LIDAR_OFFSET_M = float(VESSEL_LENGTH) / 2.0

DEFAULT_LAMBDA = 0.6
LOOKAHEAD_FRACTION = 0.25

BLOCK_D_SAFE = 6.0
BLOCK_D_CRIT = 2.0
BLOCK_FRONT_DEG = 25.0
SIDE_ARC_MIN_DEG = 15.0
SIDE_ARC_MAX_DEG = 100.0
SIDE_CLEAR_TIE = 0.25
BYPASS_CTE = 1.35

# ---------------------------------------------------------------------------
# Log streaming
# ---------------------------------------------------------------------------
class FrameStream:
    def __init__(self, filepath: str, decoder: Optional[BluefinStreamDecoder] = None):
        self.filepath = filepath
        self.decoder = decoder or BluefinStreamDecoder(lidar_out_beams=720)
        self._fh = open(filepath, "r", errors="ignore")
        self.frame_index = 0

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def restart(self) -> None:
        self.close()
        self._fh = open(self.filepath, "r", errors="ignore")
        self.frame_index = 0
        self.decoder = BluefinStreamDecoder(
            lidar_out_beams=self.decoder.lidar_out_beams,
            lidar_angle_offset_deg=self.decoder.lidar_angle_offset_deg,
            lidar_max_m=self.decoder.lidar_max_m,
            lidar_unit_scale=self.decoder.lidar_unit_scale,
            lidar_out_of_range=self.decoder.lidar_out_of_range,
        )

    def next_frame(self) -> Optional[BluefinFrame]:
        while True:
            line = self._fh.readline()
            if line == "":
                return None
            frame = self.decoder.feed(line)
            if frame is not None:
                self.frame_index += 1
                return frame

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def wrap_180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0

def format_lidar_lines(lidar_m: np.ndarray, *, per_line: int = 12, precision: int = 1) -> List[str]:
    arr = np.asarray(lidar_m).ravel()
    fmt = f"{{:.{precision}f}}"
    tokens = [fmt.format(float(x)) for x in arr]
    return [", ".join(tokens[i:i + per_line]) for i in range(0, len(tokens), per_line)]

def pick_lidar_swath(full_ranges_m: np.ndarray, angles_deg: np.ndarray, *, index0_deg: float) -> np.ndarray:
    """Pick a relative-angle swath from a full 360 degree LiDAR scan."""
    full_ranges_m = np.asarray(full_ranges_m, dtype=np.float32).ravel()
    n = full_ranges_m.size
    if n == 0:
        return full_ranges_m
    step = 360.0 / float(n)
    idx = np.round((angles_deg - float(index0_deg)) / step).astype(int) % n
    return full_ranges_m[idx]

def generate_reference_path(start_x: float, start_y: float, goal_x: float, goal_y: float) -> Tuple[np.ndarray, np.ndarray]:
    length_m = float(math.hypot(goal_x - start_x, goal_y - start_y))
    path_len = max(40, int(length_m * 5.0))
    xs = np.linspace(start_x, goal_x, path_len, dtype=np.float32)
    ys = np.linspace(start_y, goal_y, path_len, dtype=np.float32)
    path = np.column_stack((xs, ys)).astype(np.float32)
    diffs = np.diff(path, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    path_s = np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float32)
    return path, path_s

def path_tangent(path: np.ndarray, idx: int) -> np.ndarray:
    idx = int(np.clip(idx, 0, len(path) - 1))
    if len(path) < 2:
        return np.array([0.0, 1.0], dtype=np.float32)
    if idx == 0:
        vec = path[1] - path[0]
    elif idx == len(path) - 1:
        vec = path[-1] - path[-2]
    else:
        vec = path[idx + 1] - path[idx - 1]
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        return np.array([0.0, 1.0], dtype=np.float32)
    return (vec / n).astype(np.float32)

def bearing_deg(from_xy: np.ndarray, to_xy: np.ndarray) -> float:
    dx = float(to_xy[0] - from_xy[0])
    dy = float(to_xy[1] - from_xy[1])
    return float(math.degrees(math.atan2(dx, dy)))

def pool_lidar_to_policy_features(full_lidar_m: np.ndarray, *, lidar_index0_deg: float) -> Dict[str, np.ndarray]:
    """Convert full 360 LiDAR to policy raw swath and 25 pooled sector features."""
    raw_angles = np.linspace(-float(LIDAR_SWATH) / 2.0, float(LIDAR_SWATH) / 2.0, int(LIDAR_BEAMS), dtype=np.float32)
    raw = pick_lidar_swath(np.asarray(full_lidar_m, dtype=np.float32), raw_angles, index0_deg=lidar_index0_deg)
    raw = np.clip(raw.astype(np.float32), 0.0, float(LIDAR_RANGE))

    sector_ranges = np.empty(int(LIDAR_SECTORS), dtype=np.float32)
    vessel_width = float(VESSEL_WIDTH + 2.0 * HULL_MARGIN)
    sector_angle = math.radians(float(LIDAR_SWATH) / float(LIDAR_SECTORS))
    beams_per_sector = int(BEAMS_PER_SECTOR)

    for i in range(int(LIDAR_SECTORS)):
        start = i * beams_per_sector
        end = min(start + beams_per_sector, raw.size)
        sec = raw[start:end]
        if sec.size == 0:
            pooled = float(LIDAR_RANGE)
        elif PolicyLidar is not None:
            pooled = float(PolicyLidar._feasibility_pool(sec.astype(np.float64), vessel_width, sector_angle))
        else:
            pooled = float(np.min(sec))
        sector_ranges[i] = float(np.clip(pooled, 0.0, float(LIDAR_RANGE)))

    sector_closeness = np.clip(1.0 - sector_ranges / float(LIDAR_RANGE), 0.0, 1.0).astype(np.float32)
    sector_angles = np.linspace(-float(LIDAR_SWATH) / 2.0, float(LIDAR_SWATH) / 2.0, int(LIDAR_SECTORS), dtype=np.float32)
    return {
        "raw_ranges": raw,
        "raw_angles": raw_angles,
        "sector_ranges": sector_ranges,
        "sector_closeness": sector_closeness,
        "sector_angles": sector_angles,
    }

def compute_policy_obs_display(
    frame: BluefinFrame,
    *,
    origin_world: Tuple[float, float],
    start_xy: Tuple[float, float],
    reference_path: np.ndarray,
    reference_path_s: np.ndarray,
    pos_scale: float,
    lidar_index0_deg: float,
    lambda_value: float,
    lookahead_fraction: float,
) -> Dict[str, Any]:
    """Compute the policy observation for display, without loading a model."""
    mx = float(start_xy[0] + (float(frame.x_m) - float(origin_world[0])) * pos_scale)
    my = float(start_xy[1] + (float(frame.y_m) - float(origin_world[1])) * pos_scale)
    asv_pos = np.array([mx, my], dtype=np.float32)

    yaw_deg = float(frame.yaw_deg)
    if float(frame.speed_mps) > 1e-4:
        course_deg = float(math.degrees(math.atan2(float(frame.vx_mps), float(frame.vy_mps))))
    else:
        course_deg = yaw_deg

    d = np.linalg.norm(reference_path - asv_pos, axis=1)
    closest_idx = int(np.argmin(d))
    cte_abs = float(d[closest_idx])
    closest_pt = reference_path[closest_idx]
    tangent = path_tangent(reference_path, closest_idx)
    rel = asv_pos - closest_pt
    cross_z = float(tangent[0] * rel[1] - tangent[1] * rel[0])
    sign = 1.0 if cross_z > 0.0 else (-1.0 if cross_z < 0.0 else 0.0)
    cross_track_error = float(sign * cte_abs)

    path_course_deg = float(math.degrees(math.atan2(float(tangent[0]), float(tangent[1]))))
    course_error = wrap_180(path_course_deg - course_deg)

    total_len = float(reference_path_s[-1]) if reference_path_s.size else 1.0
    s_here = float(reference_path_s[closest_idx]) if reference_path_s.size else 0.0
    lookahead_distance = max(1.0, float(lookahead_fraction) * total_len)
    s_target = min(total_len, s_here + lookahead_distance)
    lookahead_idx = int(np.searchsorted(reference_path_s, s_target, side="left"))
    lookahead_idx = int(np.clip(lookahead_idx, 0, len(reference_path) - 1))
    lookahead_course_error = wrap_180(bearing_deg(asv_pos, reference_path[lookahead_idx]) - course_deg)

    lidar = pool_lidar_to_policy_features(frame.lidar_m, lidar_index0_deg=lidar_index0_deg)
    sector_ranges = lidar["sector_ranges"]
    sector_angles = lidar["sector_angles"]

    front_mask = np.abs(sector_angles) <= BLOCK_FRONT_DEG
    left_mask = (sector_angles <= -SIDE_ARC_MIN_DEG) & (sector_angles >= -SIDE_ARC_MAX_DEG)
    right_mask = (sector_angles >= SIDE_ARC_MIN_DEG) & (sector_angles <= SIDE_ARC_MAX_DEG)

    def pctl(mask: np.ndarray, p: float) -> float:
        vals = sector_ranges[mask]
        return float(np.percentile(vals, p)) if vals.size else float(LIDAR_RANGE)

    front_clearance = pctl(front_mask, 10.0)
    left_clearance = pctl(left_mask, 20.0)
    right_clearance = pctl(right_mask, 20.0)
    side_clearance_diff = float(right_clearance - left_clearance)
    block_alpha = float(np.clip((BLOCK_D_SAFE - front_clearance) / max(BLOCK_D_SAFE - BLOCK_D_CRIT, 1e-6), 0.0, 1.0))

    if block_alpha <= 1e-6:
        local_target_cte = 0.0
    else:
        if abs(side_clearance_diff) < SIDE_CLEAR_TIE:
            side_cte_sign = -1.0  # starboard/right default in current sim convention
        elif side_clearance_diff > 0.0:
            side_cte_sign = -1.0
        else:
            side_cte_sign = +1.0
        local_target_cte = float(side_cte_sign * BYPASS_CTE * block_alpha)

    return {
        "x": float(mx),
        "y": float(my),
        "u": float(frame.u_body_mps),
        "v": float(frame.v_body_mps),
        "yaw_rate": float(frame.yaw_rate),
        "course_deg": float(course_deg),
        "cross_track_error": float(cross_track_error),
        "course_error": float(course_error),
        "lookahead_course_error": float(lookahead_course_error),
        "front_clearance": float(front_clearance),
        "left_clearance": float(left_clearance),
        "right_clearance": float(right_clearance),
        "side_clearance_diff": float(side_clearance_diff),
        "block_alpha": float(block_alpha),
        "local_target_cte": float(local_target_cte),
        "log10_lambda": float(np.log10(float(lambda_value))),
        **lidar,
    }

# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------
def world_to_screen(
    xy_world: Tuple[float, float],
    *,
    view_center_world: Tuple[float, float],
    view_center_px: Tuple[int, int],
    px_per_m: float,
) -> Tuple[int, int]:
    x, y = xy_world
    cx_w, cy_w = view_center_world
    cx_px, cy_px = view_center_px
    sx = cx_px + (x - cx_w) * px_per_m
    sy = cy_px - (y - cy_w) * px_per_m
    return int(round(sx)), int(round(sy))

def draw_polyline(surface: pygame.Surface, pts_world: np.ndarray, *, color: Tuple[int, int, int], width: int, view_center_world, view_center_px, px_per_m):
    if len(pts_world) < 2:
        return
    pts = [world_to_screen((float(p[0]), float(p[1])), view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m) for p in pts_world]
    pygame.draw.lines(surface, color, False, pts, width)

def draw_map_panel(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    trajectory_world: List[Tuple[float, float]],
    current_world: Optional[Tuple[float, float]],
    yaw_deg: Optional[float],
    reference_path: Optional[np.ndarray],
    obstacles: List[List[Tuple[float, float]]],
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    view_center_world: Tuple[float, float],
    px_per_m: float,
    lidar_angles_deg: Optional[np.ndarray] = None,
    lidar_ranges_m: Optional[np.ndarray] = None,
) -> None:
    pygame.draw.rect(surface, (10, 10, 12), map_rect)
    pygame.draw.rect(surface, (80, 80, 90), map_rect, width=2)
    view_center_px = map_rect.center
    cx, cy = view_center_px
    pygame.draw.line(surface, (40, 40, 45), (map_rect.left, cy), (map_rect.right, cy), 1)
    pygame.draw.line(surface, (40, 40, 45), (cx, map_rect.top), (cx, map_rect.bottom), 1)

    prev_clip = surface.get_clip()
    surface.set_clip(map_rect)
    try:
        # Reference path and obstacles.
        if reference_path is not None:
            draw_polyline(surface, reference_path, color=(50, 220, 90), width=2,
                          view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
        for obs in obstacles:
            pts = [world_to_screen(p, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m) for p in obs]
            pygame.draw.polygon(surface, (180, 70, 70), pts)
            pygame.draw.polygon(surface, (255, 120, 120), pts, width=1)

        # Trajectory.
        if len(trajectory_world) >= 2:
            pts = [world_to_screen(p, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m) for p in trajectory_world]
            pygame.draw.lines(surface, (80, 180, 255), False, pts, 2)

        # LiDAR rays.
        if current_world is not None and yaw_deg is not None and lidar_angles_deg is not None and lidar_ranges_m is not None:
            h = math.radians(float(yaw_deg))
            sensor_world = (
                float(current_world[0] + LIDAR_OFFSET_M * math.sin(h)),
                float(current_world[1] + LIDAR_OFFSET_M * math.cos(h)),
            )
            s_px = world_to_screen(sensor_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
            for idx, (angle, r) in enumerate(zip(lidar_angles_deg, lidar_ranges_m)):
                if idx % 2 != 0:
                    continue
                rr = float(np.clip(r, 0.0, LIDAR_RANGE))
                a = math.radians(float(yaw_deg) + float(angle))
                end_world = (sensor_world[0] + rr * math.sin(a), sensor_world[1] + rr * math.cos(a))
                e_px = world_to_screen(end_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
                col = (80, 80, 180) if rr >= LIDAR_RANGE * 0.98 else (230, 180, 70)
                pygame.draw.line(surface, col, s_px, e_px, 1)

        # Start/goal/current.
        sp = world_to_screen(start_xy, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
        gp = world_to_screen(goal_xy, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
        pygame.draw.circle(surface, (40, 220, 40), sp, 5)
        pygame.draw.circle(surface, (220, 220, 40), gp, 6)

        if current_world is not None:
            p = world_to_screen(current_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
            pygame.draw.circle(surface, (255, 255, 255), p, 5)
            if yaw_deg is not None:
                h = math.radians(float(yaw_deg))
                tip_world = (current_world[0] + 1.2 * math.sin(h), current_world[1] + 1.2 * math.cos(h))
                tip = world_to_screen(tip_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
                pygame.draw.line(surface, (255, 200, 80), p, tip, 3)
    finally:
        surface.set_clip(prev_clip)

def surface_to_bgr(screen: pygame.Surface) -> np.ndarray:
    frame_rgb = pygame.surfarray.array3d(screen)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

def plot_trajectory(traj_xy: List[Tuple[float, float]], traj_yaw_deg: List[float], out_png: str) -> None:
    if not traj_xy:
        return
    import matplotlib.pyplot as plt
    xs = np.array([p[0] for p in traj_xy], dtype=float)
    ys = np.array([p[1] for p in traj_xy], dtype=float)
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys)
    plt.scatter([xs[-1]], [ys[-1]])
    if traj_yaw_deg:
        h = math.radians(float(traj_yaw_deg[-1]))
        plt.arrow(xs[-1], ys[-1], math.sin(h), math.cos(h), length_includes_head=True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {out_png}")

# ---------------------------------------------------------------------------
# Main UI loop
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Path to test_*.log")
    ap.add_argument("--rate", type=float, default=1.0, help="Playback speed multiplier")
    ap.add_argument("--fps", type=int, default=60, help="UI frame-rate cap")
    ap.add_argument("--full", action="store_true", help="Start with full 360 LiDAR list instead of policy sectors")
    ap.add_argument("--no-map", action="store_true", help="Start with map panel hidden")
    ap.add_argument("--zoom", type=float, default=30.0, help="Initial zoom in pixels per meter")
    ap.add_argument("--record", action="store_true", help="Record an MP4 of the pygame window")
    ap.add_argument("--out-video", default="bluefin_replay.mp4", help="Output video filename")
    ap.add_argument("--out-image", default="snapshot.png", help="Output screenshot filename")
    ap.add_argument("--video-fps", type=float, default=60.0, help="Video FPS")
    ap.add_argument("--plot", default="trajectory_plot.png", help="Trajectory plot output")
    ap.add_argument("--metrics", default="metrics.json", help="Optional JSON file for displayed policy-observation series")

    # Policy-observation display options.
    ap.add_argument("--test-case", type=int, default=1, help="Reference test case used for start/goal and obstacles")
    ap.add_argument("--lambda-value", type=float, default=DEFAULT_LAMBDA, help="Fixed lambda shown as log10_lambda")
    ap.add_argument("--lookahead-fraction", type=float, default=LOOKAHEAD_FRACTION)
    ap.add_argument("--pos-scale", type=float, default=1.0, help="Scale real log displacement into RL-map displacement")
    ap.add_argument("--lidar-index0-deg", type=float, default=LIDAR_INDEX_DEG, help="Full-scan angle offset for swath extraction")
    ap.add_argument("--hide-policy-obs", action="store_true", help="Hide policy-observation panel lines")
    args = ap.parse_args()

    if not os.path.exists(args.logfile):
        raise SystemExit(f"File not found: {args.logfile}")
    if args.rate <= 0:
        raise SystemExit("--rate must be > 0")

    # Scenario path/obstacles for policy-observation display.
    scenario = TestCase()
    try:
        sx, sy, gx, gy = scenario.position(test_case=args.test_case)
        obstacles = scenario.obstacles(test_case=args.test_case)
    except Exception as exc:
        print(f"[WARN] Could not load test_case={args.test_case}: {exc}. Falling back to case 0.")
        sx, sy, gx, gy = 5.0, 2.0, 5.0, 22.0
        obstacles = []
    start_xy = (float(sx), float(sy))
    goal_xy = (float(gx), float(gy))
    reference_path, reference_path_s = generate_reference_path(*start_xy, *goal_xy)

    pygame.init()
    pygame.display.set_caption("Bluefin log viewer + policy observation")
    win_w, win_h = 1280, 680
    text_w = 860
    map_w = win_w - text_w
    screen = pygame.display.set_mode((win_w, win_h))
    font = pygame.font.SysFont("consolas", 16) or pygame.font.Font(None, 16)
    small = pygame.font.SysFont("consolas", 14) or pygame.font.Font(None, 14)
    clock = pygame.time.Clock()

    video_writer = None
    video_fps = float(args.video_fps)
    capture_period = 1.0 / max(video_fps, 1e-9)
    next_capture_due = time.perf_counter()
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.out_video, fourcc, video_fps, (win_w, win_h))
        if not video_writer.isOpened():
            raise RuntimeError("Could not open video writer")
        print(f"[REC] Recording to {args.out_video}")

    decoder = BluefinStreamDecoder(lidar_out_beams=720)
    stream = FrameStream(args.logfile, decoder)

    paused = False
    show_full_lidar = bool(args.full)
    show_map = not bool(args.no_map)
    follow_mode = True
    lidar_scroll = 0
    px_per_m = float(args.zoom)
    view_center_world = (0.0, 0.0)
    origin_world: Optional[Tuple[float, float]] = None
    frame: Optional[BluefinFrame] = None
    prev_t_sec: Optional[float] = None
    next_due = time.perf_counter()
    dt_last = 0.1

    mapped_path: List[Tuple[float, float]] = []
    traj_real: List[Tuple[float, float]] = []
    traj_yaw: List[float] = []
    policy_series: List[Dict[str, Any]] = []
    policy_obs: Optional[Dict[str, Any]] = None
    cached_lidar_key = None
    cached_lidar_lines: List[str] = []

    running = True
    while running:
        now = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    show_full_lidar = not show_full_lidar
                    lidar_scroll = 0
                elif event.key == pygame.K_m:
                    show_map = not show_map
                elif event.key == pygame.K_g:
                    follow_mode = not follow_mode
                elif event.key == pygame.K_c:
                    mapped_path = []
                elif event.key == pygame.K_o and frame is not None:
                    origin_world = (float(frame.x_m), float(frame.y_m))
                    mapped_path = []
                    view_center_world = start_xy
                elif event.key == pygame.K_p:
                    pygame.image.save(screen, args.out_image)
                    print(f"[IMG] Saved: {args.out_image}")
                elif event.key == pygame.K_UP and (show_full_lidar or policy_obs is not None):
                    lidar_scroll = max(0, lidar_scroll - 1)
                elif event.key == pygame.K_DOWN and (show_full_lidar or policy_obs is not None):
                    lidar_scroll += 1
                elif not follow_mode:
                    pan_step_m = 20.0 / max(px_per_m, 1e-9)
                    if event.key == pygame.K_w:
                        view_center_world = (view_center_world[0], view_center_world[1] + pan_step_m)
                    elif event.key == pygame.K_s:
                        view_center_world = (view_center_world[0], view_center_world[1] - pan_step_m)
                    elif event.key == pygame.K_a:
                        view_center_world = (view_center_world[0] - pan_step_m, view_center_world[1])
                    elif event.key == pygame.K_d:
                        view_center_world = (view_center_world[0] + pan_step_m, view_center_world[1])

        # Playback timing / frame update.
        while not paused and now >= next_due:
            nf = stream.next_frame()
            if nf is None:
                paused = True
                break
            if prev_t_sec is None:
                dt_last = 0.1
            else:
                dt = float(nf.t_sec - prev_t_sec)
                if dt <= 0 or dt > 5:
                    dt = 0.1
                dt_last = dt
            frame = nf
            prev_t_sec = float(frame.t_sec)
            next_due += dt_last / float(args.rate)
            cached_lidar_key = None

            if origin_world is None:
                origin_world = (float(frame.x_m), float(frame.y_m))
                view_center_world = start_xy

            policy_obs = compute_policy_obs_display(
                frame,
                origin_world=origin_world,
                start_xy=start_xy,
                reference_path=reference_path,
                reference_path_s=reference_path_s,
                pos_scale=float(args.pos_scale),
                lidar_index0_deg=float(args.lidar_index0_deg),
                lambda_value=float(args.lambda_value),
                lookahead_fraction=float(args.lookahead_fraction),
            )

            mapped_xy = (float(policy_obs["x"]), float(policy_obs["y"]))
            mapped_path.append(mapped_xy)
            traj_real.append((float(frame.x_m), float(frame.y_m)))
            traj_yaw.append(float(frame.yaw_deg))
            policy_series.append({
                "t_sec": float(frame.t_sec),
                "x_rl": float(policy_obs["x"]),
                "y_rl": float(policy_obs["y"]),
                "cte": float(policy_obs["cross_track_error"]),
                "course_error": float(policy_obs["course_error"]),
                "lookahead_course_error": float(policy_obs["lookahead_course_error"]),
                "front_clearance": float(policy_obs["front_clearance"]),
                "side_clearance_diff": float(policy_obs["side_clearance_diff"]),
                "local_target_cte": float(policy_obs["local_target_cte"]),
            })
            if follow_mode:
                view_center_world = mapped_xy

        # Draw UI.
        screen.fill((20, 20, 25))
        text_rect = pygame.Rect(0, 0, text_w, win_h)
        map_rect = pygame.Rect(text_w, 0, map_w, win_h)
        y = 10
        line_h = 20

        header_lines = [
            f"File: {os.path.basename(args.logfile)}",
            f"Playback: {'PAUSED' if paused else 'RUNNING'} speed={args.rate:.2f}x frame={stream.frame_index:06d} dt~{dt_last:.3f}s",
            f"Policy ref: case={args.test_case} start=({start_xy[0]:.1f},{start_xy[1]:.1f}) goal=({goal_xy[0]:.1f},{goal_xy[1]:.1f}) lambda={args.lambda_value:.2f}",
            f"LiDAR policy swath={LIDAR_SWATH:.0f}deg beams={LIDAR_BEAMS} sectors={LIDAR_SECTORS} index0={args.lidar_index0_deg:.1f}deg",
        ]
        if frame is None:
            next_due = now
            header_lines.append("Waiting for first decoded LiDAR frame...")
        else:
            header_lines += [
                f"Real pose: x={frame.x_m:+.3f} y={frame.y_m:+.3f} yaw={frame.yaw_deg:+.2f} speed={frame.speed_mps:.3f} t={frame.t_sec:.2f}s",
                f"Body/ctrl: u={frame.u_body_mps:+.3f} v={frame.v_body_mps:+.3f} yaw_rate={frame.yaw_rate:+.2f} S1={frame.s1} S2={frame.s2}",
            ]
        if policy_obs is not None and not args.hide_policy_obs:
            header_lines += [
                " ",
                "Policy observation preview:",
                f"  map_xy=({policy_obs['x']:+.2f},{policy_obs['y']:+.2f})  u={policy_obs['u']:+.3f}  v={policy_obs['v']:+.3f}  yaw_rate={policy_obs['yaw_rate']:+.2f}",
                f"  cte={policy_obs['cross_track_error']:+.3f}  course_err={policy_obs['course_error']:+.2f}deg  lookahead_err={policy_obs['lookahead_course_error']:+.2f}deg",
                f"  front_clear={policy_obs['front_clearance']:.2f}  L/R={policy_obs['left_clearance']:.2f}/{policy_obs['right_clearance']:.2f}  side_diff={policy_obs['side_clearance_diff']:+.2f}",
                f"  block_alpha={policy_obs['block_alpha']:.2f}  local_target_cte={policy_obs['local_target_cte']:+.2f}  log10_lambda={policy_obs['log10_lambda']:+.3f}",
            ]

        for s in header_lines:
            color = (255, 230, 150) if s == "Policy observation preview:" else (235, 235, 245)
            screen.blit(font.render(s, True, color), (10, y))
            y += line_h
        y += 6

        # LiDAR list section.
        if frame is not None:
            if show_full_lidar:
                lidar_src = frame.lidar_m
                title = "Full 360 LiDAR ranges (F toggles policy sectors)"
                per_line = 15
                precision = 1
            else:
                lidar_src = policy_obs["sector_closeness"] if policy_obs is not None else np.array([], dtype=np.float32)
                title = "Policy lidar = 25 sector closeness values (0 clear, 1 close) (F toggles full scan)"
                per_line = 5
                precision = 3
            key = (stream.frame_index, show_full_lidar)
            if key != cached_lidar_key:
                cached_lidar_lines = format_lidar_lines(lidar_src, per_line=per_line, precision=precision)
                cached_lidar_key = key
            max_lines = max(1, (win_h - y - 20) // 18)
            max_scroll = max(0, len(cached_lidar_lines) - max_lines)
            lidar_scroll = min(lidar_scroll, max_scroll)
            screen.blit(font.render(f"{title} scroll {lidar_scroll}/{max_scroll}", True, (200, 200, 210)), (10, y))
            y += 22
            for s in cached_lidar_lines[lidar_scroll: lidar_scroll + max_lines]:
                screen.blit(small.render(s, True, (210, 210, 220)), (10, y))
                y += 18

        if show_map:
            current_world = (policy_obs["x"], policy_obs["y"]) if policy_obs is not None else None
            yaw_for_draw = float(frame.yaw_deg) if frame is not None else None
            raw_angles = policy_obs["raw_angles"] if policy_obs is not None else None
            raw_ranges = policy_obs["raw_ranges"] if policy_obs is not None else None
            draw_map_panel(
                screen,
                map_rect,
                trajectory_world=mapped_path,
                current_world=current_world,
                yaw_deg=yaw_for_draw,
                reference_path=reference_path,
                obstacles=obstacles,
                start_xy=start_xy,
                goal_xy=goal_xy,
                view_center_world=view_center_world,
                px_per_m=px_per_m,
                lidar_angles_deg=raw_angles,
                lidar_ranges_m=raw_ranges,
            )
            screen.blit(small.render(f"points={len(mapped_path)}  green=ref path  red=case obstacles", True, (230, 230, 230)), (map_rect.left + 8, map_rect.top + 8))

        if video_writer is not None and not paused:
            while now >= next_capture_due:
                video_writer.write(surface_to_bgr(screen))
                next_capture_due += capture_period

        pygame.display.flip()
        clock.tick(int(args.fps))

    stream.close()
    if video_writer is not None:
        video_writer.release()
        print(f"[REC] Video saved: {args.out_video}")
    plot_trajectory(traj_real, traj_yaw, args.plot)
    if args.metrics:
        data = {
            "logfile": args.logfile,
            "test_case": args.test_case,
            "start": start_xy,
            "goal": goal_xy,
            "n_frames": len(policy_series),
            "policy_observation_series": policy_series,
        }
        with open(args.metrics, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[JSON] Saved metrics: {args.metrics}")
    pygame.quit()

if __name__ == "__main__":
    main()
