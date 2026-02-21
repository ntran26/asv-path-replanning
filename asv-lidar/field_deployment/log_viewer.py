"""bluefin_log_viewer_pygame_traj.py

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

  = / - : zoom map in / out
  0     : reset zoom to default
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
from typing import Optional, List, Tuple

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
    full_ranges_m = np.asarray(full_ranges_m).ravel()
    n = full_ranges_m.size
    if n == 0:
        return full_ranges_m
    
    step = 360/n
    idx = np.round((angles_deg - index0_deg)/step).astype(int) % n

    return full_ranges_m[idx]

# -----------------------------
#  Map / trajectory rendering
# -----------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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

    So we invert y when mapping to screen.
    """
    x, y = xy_world
    cx_w, cy_w = view_center_world
    cx_px, cy_px = view_center_px

    sx = cx_px + (x - cx_w) * px_per_m
    sy = cy_px - (y - cy_w) * px_per_m  # invert Y
    return int(round(sx)), int(round(sy))


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

    # Optional axes through the view center
    if show_axes:
        cx, cy = view_center_px
        pygame.draw.line(surface, (40, 40, 45), (map_rect.left, cy), (map_rect.right, cy), 1)
        pygame.draw.line(surface, (40, 40, 45), (cx, map_rect.top), (cx, map_rect.bottom), 1)

        # A simple scale bar: 1m
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
    ap.add_argument("--zoom", type=float, default=30.0, help="Initial zoom in pixels per meter")
    ap.add_argument("--record", action="store_true", help="Record an MP4 of the pygame window")
    ap.add_argument("--out-video", default="bluefin_replay.mp4", help="Output video filename")
    ap.add_argument("--out-image", default="bluefin_final.png", help="Output final screenshot filename")
    ap.add_argument("--video-fps", type=float, default=None, help="Video FPS. If not set, defaults to --fps (UI rate).")
    ap.add_argument("--plot", default="trajectory_plot.png", help="Matplotlib trajectory plot output")

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
    default_px_per_m = float(args.zoom)
    px_per_m = float(args.zoom)
    px_per_m = _clamp(px_per_m, 2.0, 400.0)

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

                    origin_world = None
                    path_world = []
                    view_center_world = (0.0, 0.0)

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

                # Zoom controls
                elif event.key == pygame.K_EQUALS:  # '=' key (also '+' with shift)
                    px_per_m = _clamp(px_per_m * 1.15, 2.0, 400.0)
                elif event.key == pygame.K_MINUS:
                    px_per_m = _clamp(px_per_m / 1.15, 2.0, 400.0)
                elif event.key == pygame.K_0:
                    px_per_m = default_px_per_m

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

        header_lines = [
            f"File: {os.path.basename(args.logfile)}",
            f"Playback: {'PAUSED' if paused else 'RUNNING'}   speed={args.rate:.2f}x   (Space=pause, F=full lidar, R=restart)",
            f"Map: {'ON' if show_map else 'OFF'}  follow={'ON' if follow_mode else 'OFF'}  zoom={px_per_m:0.1f}px/m  (M,G,=,-,O)",
        ]

        if frame is None:
            next_due = now
            header_lines.append("Waiting for first LiDAR frame...")
        else:
            lidar = frame.lidar_m
            lidar_min = float(np.min(lidar)) if lidar.size else float("nan")
            lidar_max = float(np.max(lidar)) if lidar.size else float("nan")
            lidar_mean = float(np.mean(lidar)) if lidar.size else float("nan")

            if origin_world is None:
                rel_x, rel_y = 0.0, 0.0
            else:
                rel_x = float(frame.x_m - origin_world[0])
                rel_y = float(frame.y_m - origin_world[1])

            header_lines += [
                f"Frame #{stream.frame_index:06d}    ts={frame.ts_str}    t_sec={frame.t_sec:9.3f}    dt~{dt_last:0.3f}s (~{(1.0/dt_last if dt_last>1e-6 else 0):0.1f} Hz)",
                f"Pose(SLAM):  x={frame.x_m:+0.3f} m   y={frame.y_m:+0.3f} m   yaw={frame.yaw_deg:0.2f} deg   (hdg_ref={frame.hdg_ref_deg})",
                f"Control: rudder: {frame.s1:0.2f}, thruster: {frame.s2:0.2f}",
                # f"Pose(rel):   x={rel_x:+0.3f} m   y={rel_y:+0.3f} m   origin=({origin_world[0]:+0.3f},{origin_world[1]:+0.3f})",
                f"Vel(derived): vx={frame.vx_mps:+0.3f} m/s   vy={frame.vy_mps:+0.3f} m/s   speed={frame.speed_mps:0.3f} m/s",
                f"LiDAR: beams={lidar.size}   units=m (dm*0.1)   min/mean/max={lidar_min:0.2f}/{lidar_mean:0.2f}/{lidar_max:0.2f}",
            ]

        for s in header_lines:
            screen.blit(font.render(s, True, (235, 235, 245)), (10, y))
            y += line_h

        y += 10

        # --- LiDAR text ---
        if frame is not None:
            if show_full_lidar:
                cache_key = (stream.frame_index,)
                if cache_key != cached_lidar_key:
                    cached_lidar_lines = format_lidar_lines(frame.lidar_m, per_line=12, precision=1)
                    cached_lidar_key = cache_key

                max_lines_on_screen = max(1, (win_h - y - 20) // 18)
                max_scroll = max(0, len(cached_lidar_lines) - max_lines_on_screen)
                lidar_scroll = min(lidar_scroll, max_scroll)

                info = f"LiDAR full list (scroll {lidar_scroll}/{max_scroll})"
                screen.blit(font.render(info, True, (200, 200, 210)), (10, y))
                y += 22

                for i in range(lidar_scroll, min(len(cached_lidar_lines), lidar_scroll + max_lines_on_screen)):
                    s = cached_lidar_lines[i]
                    screen.blit(small.render(s, True, (210, 210, 220)), (10, y))
                    y += 18
            else:
                lidar = frame.lidar_m
                first = ", ".join(f"{float(x):0.1f}" for x in lidar[:12])
                last = ", ".join(f"{float(x):0.1f}" for x in lidar[-12:])
                screen.blit(font.render("LiDAR summary (press F for full list)", True, (200, 200, 210)), (10, y))
                y += 22
                screen.blit(small.render(f"first 12: [{first}]", True, (210, 210, 220)), (10, y))
                y += 18
                screen.blit(small.render(f" last 12: [{last}]", True, (210, 210, 220)), (10, y))
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
                    px_per_m=px_per_m,
                    lidar_angles_deg=lidar_draw_angles,
                    lidar_ranges_m=lidar_ranges_draw,
                    lidar_index0_deg=LIDAR_INDEX_DEG,
                    lidar_index0_range_m=frame.lidar_m[0] if frame.lidar_m.size > 0 else None,
                    mark_index0=True
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

    # Save a final screenshot
    pygame.image.save(screen, args.out_image)
    print(f"[IMG] Saved final screenshot: {args.out_image}")

    # Release the video
    if video_writer is not None:
        video_writer.release()
        print(f"[REC] Video saved: {args.out_video}")
    
    plot_trajectory(traj_xy, traj_yaw, args.plot)

    pygame.quit()


if __name__ == "__main__":
    main()
