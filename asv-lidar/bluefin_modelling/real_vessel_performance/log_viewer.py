"""
Practice tool (offline): replay a Bluefin log file and visualize decoded values.

What it shows per decoded LiDAR frame (10 Hz in the logs):
  - Log timestamp + t_sec (seconds since start)
  - Pose: x, y, yaw (from SLAM)
  - Derived velocity: vx, vy, speed
  - LiDAR stats + (optionally) the full list, scrollable
  - Map panel: trajectory polyline (relative to the chosen origin)

Controls:
  Space : pause / resume
  F     : toggle full LiDAR list on/off
  Up/Down : scroll LiDAR lines (when full LiDAR is enabled)
  R     : restart from beginning of file

  M     : toggle map panel on/off
  O     : set the map origin to the *current* position (re-zero)
  C     : clear the currently drawn path (keeps current origin)
  G     : toggle "follow" mode (camera centers on vessel)

  W/A/S/D : pan map up/left/down/right (only when follow mode is OFF)

  P     : take a snapshot
  Esc / window close : quit

Run:
  python log_viewer.py data/test_1.log
  python log_viewer.py data/test_2.log --record --video-fps 30

Notes:
  - This script imports log parser.
  - The map view is for *sensor sanity checking*, so it uses the SLAM pose
    directly. It does NOT use manual RC inputs (S1/S2).
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional, List, Tuple, Dict, Any
import json

import numpy as np
import pygame
import cv2

from log_parser import BluefinStreamDecoder, BluefinFrame

# -----------------------------
# LiDAR constants
# -----------------------------
LIDAR_FULL_BEAMS = 720
LIDAR_FULL_STEP = 360 / LIDAR_FULL_BEAMS    # 0.5 degrees

# LIDAR_SWATH = 360
# LIDAR_BEAMS = 720

LIDAR_SWATH = 270
LIDAR_BEAMS = 90

LIDAR_MAX = 16

# Angle of lidar relative to forward direction
LIDAR_INDEX_DEG = 0

# Vessel size 
VESSEL_LENGTH = 1.7
VESSEL_WIDTH = 0.5
LIDAR_OFFSET_M = VESSEL_LENGTH/2


# -----------------------------
#  Log decoding / streaming
# -----------------------------

class FrameStream:
    """Incremental decoder for a log file.
      - read file line-by-line
      - feed each line into BluefinStreamDecoder
      - only "yield" a frame when the decoder sees a LiDAR line (one full scan)
    """

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
        """Restart the file *and* reset the decoder's internal state."""
        self.close()
        self._fh = open(self.filepath, "r", errors="ignore")
        self.frame_index = 0

        # Recreate a fresh decoder with the same settings
        self.decoder = BluefinStreamDecoder(
            lidar_out_beams=self.decoder.lidar_out_beams,
            lidar_angle_offset_deg=self.decoder.lidar_angle_offset_deg,
            lidar_max_m=self.decoder.lidar_max_m,
            lidar_unit_scale=self.decoder.lidar_unit_scale,
            lidar_out_of_range=self.decoder.lidar_out_of_range,
        )

    def next_frame(self) -> Optional[BluefinFrame]:
        """Return the next decoded frame, or None at EOF."""
        while True:
            line = self._fh.readline()
            if line == "":
                return None  # EOF
            frame = self.decoder.feed(line)
            if frame is not None:
                self.frame_index += 1
                return frame


def format_lidar_lines(lidar_m: np.ndarray, *, per_line: int = 12, precision: int = 1) -> List[str]:
    """Format a LiDAR vector into multiple wrapped lines for text display."""
    if lidar_m.ndim != 1:
        lidar_m = np.asarray(lidar_m).ravel()

    fmt = f"{{:.{precision}f}}"
    tokens = [fmt.format(float(x)) for x in lidar_m]

    lines: List[str] = []
    for i in range(0, len(tokens), per_line):
        chunk = tokens[i : i + per_line]
        lines.append(", ".join(chunk))
    return lines

def pick_lidar_swath(full_ranges_m: np.ndarray, angles_deg: np.ndarray, *, index0_deg: float) -> np.ndarray:
    """
    Pick ranges from a 360 scan for the angles
    - full ranges_m has 720 beams covering 360 degrees
    - beam i corresponds to angle = index0_deg + i*0.5 degrees
    - angles_deg ranges from [-135, 135] for 270 lidar swath
    """
    full_ranges_m = np.asarray(full_ranges_m).ravel()   # flatten to 1D array
    n = full_ranges_m.size
    if n == 0:
        return full_ranges_m
    
    step = 360/n
    idx = np.round((angles_deg - index0_deg)/step).astype(int) % n

    return full_ranges_m[idx]

# -----------------------------
#  Map / trajectory rendering
# -----------------------------

def world_to_screen(
    xy_world: Tuple[float, float],
    *,
    view_center_world: Tuple[float, float],
    view_center_px: Tuple[int, int],
    px_per_m: float,
) -> Tuple[int, int]:
    """Convert (x,y) in *world meters* to pygame pixel coordinates.

    Convention here:
      - +x is to the right
      - +y is *up*

    Pygame convention:
      - +x is to the right
      - +y is *down*

    """
    x, y = xy_world
    cx_w, cy_w = view_center_world
    cx_px, cy_px = view_center_px

    sx = cx_px + (x - cx_w) * px_per_m
    sy = cy_px - (y - cy_w) * px_per_m  # inverted Y
    return sx, sy


def draw_map_panel(
    surface: pygame.Surface,
    map_rect: pygame.Rect,
    *,
    path_world: List[Tuple[float, float]],
    current_world: Optional[Tuple[float, float]] = None,
    yaw_deg: Optional[float] = None,
    view_center_world: Tuple[float, float],
    px_per_m: float,
    show_axes: bool = True,
    lidar_angles_deg: Optional[np.ndarray] = None,
    lidar_ranges_m: Optional[np.ndarray] = None,
    lidar_offset_m: float = LIDAR_OFFSET_M,
    lidar_index0_deg: float = 0,
    lidar_index0_range_m: Optional[float] = None,
    mark_index0: bool = True
    ) -> None:
    """Draw the trajectory polyline and the current vessel marker."""

    # Background
    pygame.draw.rect(surface, (10, 10, 12), map_rect)
    pygame.draw.rect(surface, (80, 80, 90), map_rect, width=2)

    view_center_px = map_rect.center

    # Local (X, Y) axes
    cx, cy = view_center_px
    pygame.draw.line(surface, (40, 40, 45), (map_rect.left, cy), (map_rect.right, cy), 1)
    pygame.draw.line(surface, (40, 40, 45), (cx, map_rect.top), (cx, map_rect.bottom), 1)

    bar_len_px = int(round(px_per_m))
    bar_x0 = map_rect.left + 20
    bar_y0 = map_rect.bottom - 25
    pygame.draw.line(surface, (180, 180, 190), (bar_x0, bar_y0), (bar_x0 + bar_len_px, bar_y0), 3)

    # Draw path
    if len(path_world) >= 2:
        pts = [
            world_to_screen(p, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
            for p in path_world
        ]
        # Clip drawing to the map panel area
        prev_clip = surface.get_clip()
        surface.set_clip(map_rect)
        try:
            pygame.draw.lines(surface, (80, 180, 255), False, pts, 2)
        finally:
            surface.set_clip(prev_clip)

    # Draw current position
    if current_world is not None:
        p = world_to_screen(current_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
        pygame.draw.circle(surface, (255, 255, 255), p, 5)
        pygame.draw.circle(surface, (0, 0, 0), p, 5, 1)

        # Heading arrow (assumption: yaw=0 points +Y, yaw=90 points +X)
        if yaw_deg is not None:
            yaw_rad = float(np.deg2rad(yaw_deg))
            arrow_len_m = 1.2
            tip_world = (
                float(current_world[0] + arrow_len_m * np.sin(yaw_rad)),
                float(current_world[1] + arrow_len_m * np.cos(yaw_rad)),
            )
            tip = world_to_screen(tip_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
            pygame.draw.line(surface, (255, 200, 80), p, tip, 3)
            pygame.draw.circle(surface, (255, 200, 80), tip, 4)

    # Draw lidar beams
    if (current_world is not None) and (yaw_deg is not None) and (lidar_angles_deg is not None) and (lidar_ranges_m is not None):
        # heading in radians
        h = float(np.deg2rad(yaw_deg))

        # place lidar in front of vessel
        sensor_world = (float(current_world[0] + lidar_offset_m * np.sin(h)),
                        float(current_world[1] + lidar_offset_m * np.cos(h)))
        s_px = world_to_screen(sensor_world, 
                               view_center_world=view_center_world,
                               view_center_px=view_center_px,
                               px_per_m=px_per_m)
        if mark_index0:
            a0 = float(np.deg2rad(yaw_deg + lidar_index0_deg))
            r0 = float(lidar_index0_range_m) if lidar_index0_range_m is not None else LIDAR_MAX
            r0 = float(np.clip(r0, 0, LIDAR_MAX))

            end0_world = (sensor_world[0] + r0*np.sin(a0), sensor_world[1] + r0*np.cos(a0))
            end0_px = world_to_screen(end0_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)

            pygame.draw.aaline(surface, (255,50,50), s_px, end0_px)
            pygame.draw.circle(surface, (255,50,50), end0_px, 4)

        prev_clip = surface.get_clip()
        surface.set_clip(map_rect)
        try:
            for angle, range in zip(lidar_angles_deg, lidar_ranges_m):
                r = float(np.clip(range, 0, LIDAR_MAX))
                a = float(np.deg2rad(yaw_deg + angle))

                end_world = (sensor_world[0] + r*np.sin(a), sensor_world[1] + r*np.cos(a))
                e_px = world_to_screen(end_world, view_center_world=view_center_world, view_center_px=view_center_px, px_per_m=px_per_m)
                pygame.draw.aaline(surface, (90,90,200), s_px, e_px)
        finally:
            surface.set_clip(prev_clip)

def surface_to_bgr(screen: pygame.Surface) -> np.ndarray:
    """
    Convert pygame Surface -> OpenCV BGR uint8 image.
    pygame.surfarray.array3d gives (W,H,3) in RGB.
    OpenCV expects (H,W,3) in BGR.
    """
    frame_rgb = pygame.surfarray.array3d(screen)            # (W,H,3)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))          # (H,W,3)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # BGR
    return frame_bgr

def plot_trajectory(traj_xy: List[Tuple[float, float]], traj_yaw_deg: List[float], out_png: str) -> None:
    import matplotlib.pyplot as plt

    xs = np.array([p[0] for p in traj_xy], dtype=float)
    ys = np.array([p[1] for p in traj_xy], dtype=float)

    plt.figure(figsize=(6,6))
    plt.plot(xs,ys)
    plt.scatter([xs[-1]], [ys[-1]]) # final point marker

    # final heading arrow
    h = np.deg2rad(traj_yaw_deg[-1])
    arrow_len = 1
    dx = arrow_len * np.sin(h)
    dy = arrow_len * np.cos(h)
    plt.arrow(xs[-1], ys[-1], dx, dy, length_includes_head=True)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {out_png}")

# -----------------------Analyzer helper functions---------------------------------
def wrap_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def unwrap_heading_deg(yaw_deg: np.ndarray) -> np.ndarray:
    yaw_deg = np.asarray(yaw_deg, dtype=float)
    if yaw_deg.size == 0:
        return yaw_deg.copy()

    out = np.empty_like(yaw_deg)
    out[0] = yaw_deg[0]
    for i in range(1, yaw_deg.size):
        out[i] = out[i - 1] + wrap_180(yaw_deg[i] - yaw_deg[i - 1])
    return out

def sample_at_time(t_rel: np.ndarray, values: np.ndarray, query_s: float):
    if len(t_rel) == 0 or query_s < t_rel[0] or query_s > t_rel[-1]:
        return None
    return float(np.interp(query_s, t_rel, values))

def first_crossing_time(t_rel: np.ndarray, values: np.ndarray, threshold: float):
    for i in range(1, len(values)):
        if values[i - 1] < threshold <= values[i]:
            return float(t_rel[i])
    return None

def first_abs_crossing_time(t_rel: np.ndarray, values: np.ndarray, threshold: float):
    vals = np.abs(values)
    for i in range(1, len(vals)):
        if vals[i - 1] < threshold <= vals[i]:
            return float(t_rel[i])
    return None

def slope_over_window(t_rel: np.ndarray, values: np.ndarray, t1: float, t2: float):
    mask = (t_rel >= t1) & (t_rel <= t2)
    if np.count_nonzero(mask) < 2:
        return None
    p = np.polyfit(t_rel[mask], values[mask], 1)
    return float(p[0])

def cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(ds)])

def circle_fit_radius(x: np.ndarray, y: np.ndarray):
    if len(x) < 6:
        return None

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x*x + y*y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = sol
    r2 = c0 + cx*cx + cy*cy
    if r2 <= 0:
        return None
    return float(np.sqrt(r2))

def first_sustained_index(values: np.ndarray, threshold: float, count: int = 3):
    run = 0
    for i, v in enumerate(values):
        if v > threshold:
            run += 1
            if run >= count:
                return i - count + 1
        else:
            run = 0
    return None

def first_sustained_deviation_index(values: np.ndarray, baseline: float, threshold: float, count: int = 3):
    run = 0
    for i, v in enumerate(values):
        if abs(v - baseline) > threshold:
            run += 1
            if run >= count:
                return i - count + 1
        else:
            run = 0
    return None

def first_sustained_abs_index(values: np.ndarray, threshold: float, count: int = 3):
    run = 0
    for i, v in enumerate(values):
        if abs(v) > threshold:
            run += 1
            if run >= count:
                return i - count + 1
        else:
            run = 0
    return None
#----------------------------------------------------------------------------------
#-----------------------------Analyzer class---------------------------------------
class RunAnalyzer:
    def __init__(self):
        self.t = []
        self.x = []
        self.y = []
        self.yaw = []
        self.yaw_rate = []
        self.speed = []
        self.u_body = []
        self.v_body = []
        self.s1 = []
        self.s2 = []

    def add_frame(self, frame: BluefinFrame):
        self.t.append(float(frame.t_sec))
        self.x.append(float(frame.x_m))
        self.y.append(float(frame.y_m))
        self.yaw.append(float(frame.yaw_deg))
        self.yaw_rate.append(float(frame.yaw_rate))
        self.speed.append(float(frame.speed_mps))
        self.u_body.append(float(frame.u_body_mps))
        self.v_body.append(float(frame.v_body_mps))
        self.s1.append(np.nan if frame.s1 is None else float(frame.s1))
        self.s2.append(np.nan if frame.s2 is None else float(frame.s2))

    def straight_metrics(self) -> Dict[str, Any]:
        if len(self.t) < 2:
            return {}

        t = np.asarray(self.t, dtype=float)
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        u = np.asarray(self.u_body, dtype=float)
        s2 = np.asarray(self.s2, dtype=float)

        # -----------------------------
        # 1) Detect S2 start (input start)
        # -----------------------------
        # Use early samples as neutral baseline
        valid_s2 = s2[np.isfinite(s2)]
        if valid_s2.size == 0:
            return {}

        s2_neutral = float(np.median(valid_s2[:min(20, valid_s2.size)]))

        # Threshold for detecting throttle command change
        s2_thresh = 30.0
        idx_s2 = first_sustained_deviation_index(s2, s2_neutral, s2_thresh, count=3)
        if idx_s2 is None:
            idx_s2 = 0

        # -----------------------------
        # 2) Detect motion start (response start)
        # -----------------------------
        motion_thresh = 0.05
        idx_motion = first_sustained_index(u, motion_thresh, count=3)
        if idx_motion is None:
            idx_motion = idx_s2

        # -----------------------------
        # 3) Build S2-referenced signals
        # -----------------------------
        t_rel_s2 = t[idx_s2:] - t[idx_s2]
        u_rel_s2 = u[idx_s2:]
        x_rel_s2 = x[idx_s2:]
        y_rel_s2 = y[idx_s2:]
        dist_rel_s2 = cumulative_distance(x_rel_s2, y_rel_s2)
        s2_rel = s2[idx_s2:]

        # -----------------------------
        # 4) Build motion-referenced signals
        # -----------------------------
        t_rel_motion = t[idx_motion:] - t[idx_motion]
        u_rel_motion = u[idx_motion:]
        x_rel_motion = x[idx_motion:]
        y_rel_motion = y[idx_motion:]
        dist_rel_motion = cumulative_distance(x_rel_motion, y_rel_motion)

        peak_u = float(np.max(u))
        peak_s2 = float(np.max(s2_rel)) if len(s2_rel) > 0 else None

        return {
            # input / motion alignment
            "s2_neutral": s2_neutral,
            "s2_start_idx": int(idx_s2),
            "s2_start_t_sec": float(t[idx_s2]),
            "motion_start_idx": int(idx_motion),
            "motion_start_t_sec": float(t[idx_motion]),
            "motion_lag_s": float(t[idx_motion] - t[idx_s2]),

            # S2 profile
            "s2_peak": peak_s2,
            "time_to_90pct_peak_s2_s": None if peak_s2 is None else first_crossing_time(
                t_rel_s2, np.abs(s2_rel - s2_neutral), 0.9 * np.max(np.abs(s2_rel - s2_neutral))
            ),

            # response relative to S2 start
            "u_body_at_2s_after_s2_mps": sample_at_time(t_rel_s2, u_rel_s2, 2.0),
            "u_body_at_5s_after_s2_mps": sample_at_time(t_rel_s2, u_rel_s2, 5.0),
            "u_body_at_10s_after_s2_mps": sample_at_time(t_rel_s2, u_rel_s2, 10.0),
            "distance_at_5s_after_s2_m": sample_at_time(t_rel_s2, dist_rel_s2, 5.0),
            "distance_at_10s_after_s2_m": sample_at_time(t_rel_s2, dist_rel_s2, 10.0),
            "initial_accel_0_2_after_s2_mps2": slope_over_window(t_rel_s2, u_rel_s2, 0.0, 2.0),
            "initial_accel_0_5_after_s2_mps2": slope_over_window(t_rel_s2, u_rel_s2, 0.0, 5.0),

            # response relative to motion start
            "u_body_at_2s_after_motion_mps": sample_at_time(t_rel_motion, u_rel_motion, 2.0),
            "u_body_at_5s_after_motion_mps": sample_at_time(t_rel_motion, u_rel_motion, 5.0),
            "u_body_at_10s_after_motion_mps": sample_at_time(t_rel_motion, u_rel_motion, 10.0),
            "distance_at_5s_after_motion_m": sample_at_time(t_rel_motion, dist_rel_motion, 5.0),
            "distance_at_10s_after_motion_m": sample_at_time(t_rel_motion, dist_rel_motion, 10.0),
            "initial_accel_0_2_after_motion_mps2": slope_over_window(t_rel_motion, u_rel_motion, 0.0, 2.0),
            "initial_accel_0_5_after_motion_mps2": slope_over_window(t_rel_motion, u_rel_motion, 0.0, 5.0),

            # overall
            "peak_u_body_mps": peak_u,
            "time_to_50pct_peak_u_after_motion_s": first_crossing_time(t_rel_motion, u_rel_motion, 0.5 * peak_u),
            "time_to_90pct_peak_u_after_motion_s": first_crossing_time(t_rel_motion, u_rel_motion, 0.9 * peak_u),
        }

    def turn_metrics(self) -> Dict[str, Any]:
        if len(self.t) < 2:
            return {}

        t = np.asarray(self.t, dtype=float)
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        yaw = np.asarray(self.yaw, dtype=float)
        yaw_rate = np.asarray(self.yaw_rate, dtype=float)
        u = np.asarray(self.u_body, dtype=float)
        s1 = np.asarray(self.s1, dtype=float)

        # -----------------------------
        # 1) Detect S1 start (rudder input start)
        # -----------------------------
        valid_s1 = s1[np.isfinite(s1)]
        if valid_s1.size == 0:
            return {}

        s1_neutral = float(np.median(valid_s1[:min(20, valid_s1.size)]))

        s1_thresh = 30.0
        idx_s1 = first_sustained_deviation_index(s1, s1_neutral, s1_thresh, count=3)
        if idx_s1 is None:
            idx_s1 = 0

        # -----------------------------
        # 2) Detect actual turn start from yaw rate
        # -----------------------------
        turn_thresh = 1.0
        idx_turn = first_sustained_abs_index(yaw_rate, turn_thresh, count=3)
        if idx_turn is None:
            idx_turn = idx_s1

        # -----------------------------
        # 3) S1-referenced signals
        # -----------------------------
        t_rel_s1 = t[idx_s1:] - t[idx_s1]
        x_rel_s1 = x[idx_s1:]
        y_rel_s1 = y[idx_s1:]
        yaw_rel_s1 = yaw[idx_s1:]
        yaw_rate_rel_s1 = yaw_rate[idx_s1:]
        u_rel_s1 = u[idx_s1:]
        s1_rel = s1[idx_s1:]

        yaw_u_s1 = unwrap_heading_deg(yaw_rel_s1)
        dpsi_s1 = yaw_u_s1 - yaw_u_s1[0]

        # -----------------------------
        # 4) Turn-response-referenced signals
        # -----------------------------
        t_rel_turn = t[idx_turn:] - t[idx_turn]
        x_rel_turn = x[idx_turn:]
        y_rel_turn = y[idx_turn:]
        yaw_rel_turn = yaw[idx_turn:]
        yaw_rate_rel_turn = yaw_rate[idx_turn:]
        u_rel_turn = u[idx_turn:]

        yaw_u_turn = unwrap_heading_deg(yaw_rel_turn)
        dpsi_turn = yaw_u_turn - yaw_u_turn[0]

        # -----------------------------
        # 5) Radius estimates from actual turn start
        # -----------------------------
        r90 = None
        r180 = None

        idx90 = np.where(np.abs(dpsi_turn) >= 90.0)[0]
        if idx90.size > 0:
            r90 = circle_fit_radius(x_rel_turn[:idx90[0] + 1], y_rel_turn[:idx90[0] + 1])

        idx180 = np.where(np.abs(dpsi_turn) >= 180.0)[0]
        if idx180.size > 0:
            r180 = circle_fit_radius(x_rel_turn[:idx180[0] + 1], y_rel_turn[:idx180[0] + 1])

        peak_abs_yaw_rate = float(np.max(np.abs(yaw_rate_rel_turn))) if len(yaw_rate_rel_turn) > 0 else None
        peak_s1 = float(np.max(np.abs(s1_rel - s1_neutral))) if len(s1_rel) > 0 else None

        return {
            # input / response alignment
            "s1_neutral": s1_neutral,
            "s1_start_idx": int(idx_s1),
            "s1_start_t_sec": float(t[idx_s1]),
            "turn_start_idx": int(idx_turn),
            "turn_start_t_sec": float(t[idx_turn]),
            "turn_lag_s": float(t[idx_turn] - t[idx_s1]),

            # S1 profile
            "peak_abs_s1_from_neutral": peak_s1,

            # turning response relative to S1 start
            "yaw_rate_at_2s_after_s1_degps": sample_at_time(t_rel_s1, yaw_rate_rel_s1, 2.0),
            "yaw_rate_at_5s_after_s1_degps": sample_at_time(t_rel_s1, yaw_rate_rel_s1, 5.0),
            "yaw_rate_at_10s_after_s1_degps": sample_at_time(t_rel_s1, yaw_rate_rel_s1, 10.0),
            "u_body_2s_after_s1_mps": sample_at_time(t_rel_s1, u_rel_s1, 2.0),
            "u_body_5s_after_s1_mps": sample_at_time(t_rel_s1, u_rel_s1, 5.0),
            "u_body_10s_after_s1_mps": sample_at_time(t_rel_s1, u_rel_s1, 10.0),
            "time_to_30deg_after_s1_s": first_abs_crossing_time(t_rel_s1, dpsi_s1, 30.0),
            "time_to_60deg_after_s1_s": first_abs_crossing_time(t_rel_s1, dpsi_s1, 60.0),
            "time_to_90deg_after_s1_s": first_abs_crossing_time(t_rel_s1, dpsi_s1, 90.0),
            "time_to_180deg_after_s1_s": first_abs_crossing_time(t_rel_s1, dpsi_s1, 180.0),

            # turning response relative to actual turn start
            "peak_abs_yaw_rate_degps": peak_abs_yaw_rate,
            "yaw_rate_at_2s_after_turn_degps": sample_at_time(t_rel_turn, yaw_rate_rel_turn, 2.0),
            "yaw_rate_at_5s_after_turn_degps": sample_at_time(t_rel_turn, yaw_rate_rel_turn, 5.0),
            "yaw_rate_at_10s_after_turn_degps": sample_at_time(t_rel_turn, yaw_rate_rel_turn, 10.0),
            "u_body_2s_after_turn_mps": sample_at_time(t_rel_turn, u_rel_turn, 2.0),
            "u_body_5s_after_turn_mps": sample_at_time(t_rel_turn, u_rel_turn, 5.0),
            "u_body_10s_after_turn_mps": sample_at_time(t_rel_turn, u_rel_turn, 10.0),
            "time_to_30deg_after_turn_s": first_abs_crossing_time(t_rel_turn, dpsi_turn, 30.0),
            "time_to_60deg_after_turn_s": first_abs_crossing_time(t_rel_turn, dpsi_turn, 60.0),
            "time_to_90deg_after_turn_s": first_abs_crossing_time(t_rel_turn, dpsi_turn, 90.0),
            "time_to_180deg_after_turn_s": first_abs_crossing_time(t_rel_turn, dpsi_turn, 180.0),

            # geometry
            "radius_first_90deg_m": r90,
            "radius_first_180deg_m": r180,
            "diameter_first_90deg_m": None if r90 is None else 2.0 * r90,
            "diameter_first_180deg_m": None if r180 is None else 2.0 * r180,
        }

    def export(self, out_json: str, logfile: str):
        data = {
            "logfile": logfile,
            "n_frames": len(self.t),
            "straight_metrics": self.straight_metrics(),
            "turn_metrics": self.turn_metrics(),
            "series": {
                "t_sec": self.t,
                "u_body_mps": self.u_body,
                "yaw_rate_degps": self.yaw_rate,
                "s1": self.s1,
                "s2": self.s2,
            }
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
#-----------------------------------------------------------------------------------------

# -----------------------------
#  Main UI loop
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Path to test_*.log")
    ap.add_argument("--rate", type=float, default=1.0, help="Playback speed multiplier (1.0 = realtime)")
    ap.add_argument("--fps", type=int, default=60, help="UI frame rate cap")
    ap.add_argument("--full", action="store_true", help="Start with full LiDAR list enabled")
    ap.add_argument("--no-map", action="store_true", help="Start with the map panel hidden")
    ap.add_argument("--zoom", type=float, default=20, help="Initial zoom in pixels per meter")
    ap.add_argument("--record", action="store_true", help="Record an MP4 of the pygame window")
    ap.add_argument("--out-video", default="bluefin_replay.mp4", help="Output video filename")
    ap.add_argument("--out-image", default="snapshot.png", help="Output final screenshot filename")
    ap.add_argument("--video-fps", type=float, default=60, help="Video FPS. If not set, defaults to --fps (UI rate).")
    ap.add_argument("--plot", default="trajectory_plot.png", help="Matplotlib trajectory plot output")

    ap.add_argument("--metrics", default="metrics.json", help="Optional JSON file to save extracted metrics")

    args = ap.parse_args()

    if not os.path.exists(args.logfile):
        raise SystemExit(f"File not found: {args.logfile}")
    if args.rate <= 0:
        raise SystemExit("--rate must be > 0")

    pygame.init()
    pygame.display.set_caption("Bluefin log viewer + trajectory")
    video_fps = float(args.video_fps) if args.video_fps is not None else float(args.fps)

    # Layout: left text panel + right map panel
    win_w, win_h = 1200, 600
    text_w = 800
    map_w = win_w - text_w

    screen = pygame.display.set_mode((win_w, win_h))

    video_writer = None
    capture_period = 1.0 / max(video_fps, 1e-9)
    next_capture_due = time.perf_counter()

    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.out_video, fourcc, video_fps, (win_w, win_h))
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer")
        print(f"[REC] Recording to {args.out_video} at {video_fps:.1f} fps, size={win_w}x{win_h}")

    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 18) or pygame.font.Font(None, 18)
    small = pygame.font.SysFont("consolas", 15) or pygame.font.Font(None, 15)

    decoder = BluefinStreamDecoder(lidar_out_beams=720)
    stream = FrameStream(args.logfile, decoder)

    paused = False
    show_full_lidar = bool(args.full)
    lidar_scroll = 0

    show_map = not bool(args.no_map)
    follow_mode = True

    # Map state
    px_per_m = args.zoom

    origin_world: Optional[Tuple[float, float]] = None
    path_world: List[Tuple[float, float]] = []  # stored relative to origin

    view_center_world = (0.0, 0.0)  # where the map camera is centered (relative coords)

    frame: Optional[BluefinFrame] = None
    prev_t_sec: Optional[float] = None
    next_due = time.perf_counter()
    dt_last = 0.1

    cached_lidar_lines: List[str] = []
    cached_lidar_key = None

    lidar_draw_angles = np.linspace(-LIDAR_SWATH/2, LIDAR_SWATH/2, LIDAR_BEAMS, dtype=np.float64)

    traj_xy: List[Tuple[float, float]] = []
    traj_yaw: List[float] = []

    # Add ship dynamics analyzer
    analyzer = RunAnalyzer()

    running = True
    while running:
        now = time.perf_counter()

        # -----------------
        # Handle events
        # -----------------
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
                elif event.key == pygame.K_r:
                    stream.restart()
                    frame = None
                    prev_t_sec = None
                    cached_lidar_lines = []
                    cached_lidar_key = None
                    lidar_scroll = 0
                    next_due = time.perf_counter()
                elif event.key == pygame.K_p:
                    pygame.image.save(screen, args.out_image)
                    print(f"[IMG] Saved: {args.out_image}")

                # LiDAR scrolling (only makes sense in full mode)
                elif event.key == pygame.K_UP:
                    if show_full_lidar:
                        lidar_scroll = max(0, lidar_scroll - 1)
                elif event.key == pygame.K_DOWN:
                    if show_full_lidar:
                        lidar_scroll = lidar_scroll + 1

                # Map toggles
                elif event.key == pygame.K_m:
                    show_map = not show_map
                elif event.key == pygame.K_g:
                    follow_mode = not follow_mode
                elif event.key == pygame.K_c:
                    path_world = []
                elif event.key == pygame.K_o:
                    # Re-zero origin at current frame
                    if frame is not None:
                        origin_world = (float(frame.x_m), float(frame.y_m))
                        path_world = [(0.0, 0.0)]
                        view_center_world = (0.0, 0.0)

                # Pan controls (only if not following)
                elif not follow_mode:
                    pan_step_m = 20.0 / px_per_m  # ~20px per keypress
                    if event.key == pygame.K_w:
                        view_center_world = (view_center_world[0], view_center_world[1] + pan_step_m)
                    elif event.key == pygame.K_s:
                        view_center_world = (view_center_world[0], view_center_world[1] - pan_step_m)
                    elif event.key == pygame.K_a:
                        view_center_world = (view_center_world[0] - pan_step_m, view_center_world[1])
                    elif event.key == pygame.K_d:
                        view_center_world = (view_center_world[0] + pan_step_m, view_center_world[1])

        # -----------------
        # Playback timing
        # -----------------
        while not paused and now >= next_due:
            next_frame = stream.next_frame()
            if next_frame is None:
                paused = True  # EOF
                break
            else:
                if prev_t_sec is None:
                    dt_last = 0.1
                else:
                    dt = float(next_frame.t_sec - prev_t_sec)
                    # Protect against weird gaps or equal timestamps
                    if dt <= 0 or dt > 5:
                        dt = 0.1
                    dt_last = dt

                frame = next_frame
                analyzer.add_frame(frame)
                prev_t_sec = float(next_frame.t_sec)
                # next_due = now + (dt_last / float(args.rate))
                next_due += (dt_last / float(args.rate))

                # Invalidate lidar cache for new frame
                cached_lidar_key = None

                # Update map path
                if origin_world is None:
                    origin_world = (float(frame.x_m), float(frame.y_m))
                    path_world = [(0.0, 0.0)]

                # Store relative-to-origin coordinates
                rel = (
                    float(frame.x_m - origin_world[0]),
                    float(frame.y_m - origin_world[1]),
                )
                path_world.append(rel)

                traj_xy.append((float(frame.x_m), float(frame.y_m)))
                traj_yaw.append(float(frame.yaw_deg))

                if follow_mode:
                    view_center_world = rel

        # -----------------
        # Draw UI
        # -----------------
        screen.fill((20, 20, 25))

        text_rect = pygame.Rect(0, 0, text_w, win_h)
        map_rect = pygame.Rect(text_w, 0, map_w, win_h)

        # --- Text panel ---
        y = 10
        line_h = 22

        lidar_raw = None
        lidar_view = None

        if frame is not None:
            lidar_raw = frame.lidar_m
            lidar_view = pick_lidar_swath(lidar_raw, lidar_draw_angles, index0_deg=LIDAR_INDEX_DEG)

        header_lines = [
            f"File: {os.path.basename(args.logfile)}",
            f"Playback: {'PAUSED' if paused else 'RUNNING'}   speed={args.rate:.2f}x   (Space=pause, F=full lidar, R=restart)",
            f"Map: {'ON' if show_map else 'OFF'}  follow={'ON' if follow_mode else 'OFF'}  zoom={px_per_m:0.1f}px/m",
        ]

        if frame is None:
            next_due = now
            header_lines.append("Waiting for first LiDAR frame...")
        else:
            lidar = lidar_view if lidar_view is not None else lidar_raw

            if origin_world is None:
                rel_x, rel_y = 0.0, 0.0
            else:
                rel_x = float(frame.x_m - origin_world[0])
                rel_y = float(frame.y_m - origin_world[1])

            header_lines += [
                f"Frame #{stream.frame_index:06d}    ts={frame.ts_str}    t_sec={frame.t_sec:9.3f}    dt~{dt_last:0.3f}s (~{(1.0/dt_last if dt_last>1e-6 else 0):0.1f} Hz)",
                f"Pose(SLAM):  x={frame.x_m:+0.3f} m   y={frame.y_m:+0.3f} m   yaw={frame.yaw_deg:0.2f} deg   (hdg_ref={frame.hdg_ref_deg})",
                f"Control: rudder: {frame.s1:0.2f}, thruster: {frame.s2:0.2f}",
                " ",
                f"Vel: vx={frame.vx_mps:+0.3f}  vy={frame.vy_mps:+0.3f}  speed={frame.speed_mps:0.3f}  ",
                f"u_body={frame.u_body_mps:+0.3f}  v_body={frame.v_body_mps:+0.3f}  yaw_rate={frame.yaw_rate:+0.2f}",
                " ",
                f"LiDAR: beams={lidar.size}   units=m (dm*0.1)",
            ]

        for s in header_lines:
            screen.blit(font.render(s, True, (235, 235, 245)), (10, y))
            y += line_h

        y += 10

        # --- LiDAR text ---
        if frame is not None:
            if show_full_lidar:
                lidar_src = lidar_raw
                title = "LiDAR full list (F)"
            else:
                lidar_src = lidar_view
                title = "Processed LiDAR list (F)"
            cached_key = (stream.frame_index, show_full_lidar)
            if cached_key != cached_lidar_key:
                cached_lidar_lines = format_lidar_lines(lidar_src, per_line=15, precision=1)
                cached_lidar_key = cached_key
            max_lines_on_screen = max(1, (win_h-y-20)//18)
            max_scroll = max(0, len(cached_lidar_lines) - max_lines_on_screen)
            lidar_scroll = min(lidar_scroll, max_scroll)
            
            if show_full_lidar:
                info = f"{title} (scroll {lidar_scroll}/{max_scroll})"
            else:
                info = title
            
            screen.blit(font.render(info, True, (200,200,210)), (10,y))
            y += 22

            for s in cached_lidar_lines[lidar_scroll : lidar_scroll + max_lines_on_screen]:
                screen.blit(small.render(s, True, (210,210,220)), (10,y))
                y += 18

        # --- Map panel ---
        if show_map:
            lidar_ranges_draw = pick_lidar_swath(frame.lidar_m, lidar_draw_angles, index0_deg=LIDAR_INDEX_DEG)
            if frame is None or origin_world is None:
                # Show an empty map
                draw_map_panel(
                    screen,
                    map_rect,
                    path_world=path_world,
                    current_world=None,
                    yaw_deg=None,
                    view_center_world=view_center_world,
                    px_per_m=px_per_m
                )
            else:
                current_rel = (
                    float(frame.x_m - origin_world[0]),
                    float(frame.y_m - origin_world[1]),
                )
                draw_map_panel(
                    screen,
                    map_rect,
                    path_world=path_world,
                    current_world=current_rel,
                    yaw_deg=float(frame.yaw_deg),
                    view_center_world=view_center_world,
                    px_per_m=px_per_m,
                    lidar_angles_deg=lidar_draw_angles,
                    lidar_ranges_m=lidar_ranges_draw,
                    lidar_index0_deg=LIDAR_INDEX_DEG,
                    lidar_index0_range_m=frame.lidar_m[0] if frame.lidar_m.size > 0 else None,
                    mark_index0=True
                )

                # A small status label in the map corner
                label = f"points={len(path_world)}"
                screen.blit(small.render(label, True, (210, 210, 220)), (map_rect.left + 8, map_rect.top + 8))

        if video_writer is not None and not paused:
            # Keep the output video time aligned with real time:
            # write as many frames as needed based on wall-clock schedule.
            while now >= next_capture_due:
                video_writer.write(surface_to_bgr(screen))
                next_capture_due += capture_period

        pygame.display.flip()
        clock.tick(args.fps)

    stream.close()

    # Release the video
    if video_writer is not None:
        video_writer.release()
        print(f"[REC] Video saved: {args.out_video}")
    
    plot_trajectory(traj_xy, traj_yaw, args.plot)

    straight_summary = analyzer.straight_metrics()
    turn_summary = analyzer.turn_metrics()

    print("\n[STRAIGHT METRICS]")
    print(json.dumps(straight_summary, indent=2))

    print("\n[TURN METRICS]")
    print(json.dumps(turn_summary, indent=2))

    if args.metrics:
        analyzer.export(args.metrics, args.logfile)
        print(f"[JSON] Saved metrics: {args.metrics}")

    pygame.quit()

if __name__ == "__main__":
    main()
