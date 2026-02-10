"""bluefin_log_viewer_pygame.py

Practice tool: replay a Bluefin log file and visualize decoded values in a
simple pygame window.

What it shows (per decoded LiDAR frame):
  - Log timestamp + t_sec
  - Pose: x, y, yaw
  - Derived velocity: vx, vy, speed
  - LiDAR stats + (optionally) the full list, scrollable

Controls:
  Space : pause / resume
  Up/Down : scroll LiDAR lines (when "full LiDAR" is enabled)
  F : toggle full LiDAR list on/off
  R : restart from beginning of file
  Esc / window close : quit

Run:
  python bluefin_log_viewer_pygame.py /path/to/test_1.log

Notes:
  - This script imports your existing parser (log_parser.py). Put this file in
    the same folder as log_parser.py, or ensure that folder is on PYTHONPATH.
  - Rendering ALL 720 beams as text is intentionally supported, but it can be
    heavy. Use the 'F' key to toggle.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional, List

import numpy as np
import pygame

from log_parser import BluefinStreamDecoder, BluefinFrame


class FrameStream:
    """Incremental decoder for a log file.

    We keep it streaming (instead of decoding the whole file upfront) so you can
    test large logs without loading everything into RAM.
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
        self.close()
        self._fh = open(self.filepath, "r", errors="ignore")
        self.frame_index = 0
        # Reset decoder state as well
        self.decoder = BluefinStreamDecoder(
            lidar_out_beams=self.decoder.lidar_out_beams,
            lidar_angle_offset_deg=self.decoder.lidar_angle_offset_deg,
            lidar_max_m=self.decoder.lidar_max_m,
            lidar_unit_scale=self.decoder.lidar_unit_scale,
            lidar_out_of_range=self.decoder.lidar_out_of_range,
        )

    def next_frame(self) -> Optional[BluefinFrame]:
        """Return the next decoded frame (on LiDAR line), or None at EOF."""
        while True:
            line = self._fh.readline()
            if line == "":
                return None
            frame = self.decoder.feed(line)
            if frame is not None:
                self.frame_index += 1
                return frame


def format_lidar_lines(lidar_m: np.ndarray, *, per_line: int = 12, precision: int = 1) -> List[str]:
    """Format a LiDAR vector into multiple wrapped lines."""
    if lidar_m.ndim != 1:
        lidar_m = np.asarray(lidar_m).ravel()

    fmt = f"{{:.{precision}f}}"
    tokens = [fmt.format(float(x)) for x in lidar_m]
    lines = []
    for i in range(0, len(tokens), per_line):
        chunk = tokens[i : i + per_line]
        lines.append(", ".join(chunk))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Path to test_*.log")
    ap.add_argument("--rate", type=float, default=1.0, help="Playback speed multiplier (1.0 = realtime)")
    ap.add_argument("--fps", type=int, default=60, help="UI frame rate cap")
    ap.add_argument("--full", action="store_true", help="Start with full LiDAR list enabled")
    args = ap.parse_args()

    if not os.path.exists(args.logfile):
        raise SystemExit(f"File not found: {args.logfile}")
    if args.rate <= 0:
        raise SystemExit("--rate must be > 0")

    pygame.init()
    pygame.display.set_caption("Bluefin log viewer")

    # A wider window helps a lot with LiDAR text
    w, h = 1200, 800
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 15)

    decoder = BluefinStreamDecoder(lidar_out_beams=720)
    stream = FrameStream(args.logfile, decoder)

    paused = False
    show_full_lidar = bool(args.full)
    lidar_scroll = 0

    frame: Optional[BluefinFrame] = None
    prev_t_sec: Optional[float] = None
    prev_wall = time.perf_counter()
    next_due = prev_wall  # when to fetch next frame
    dt_last = 0.1

    # Cached formatted lidar text for the current frame (so we don't reformat every UI tick)
    cached_lidar_lines: List[str] = []
    cached_lidar_key = None

    running = True
    while running:
        now = time.perf_counter()

        # --- Events ---
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
                elif event.key == pygame.K_UP:
                    lidar_scroll = max(0, lidar_scroll - 1)
                elif event.key == pygame.K_DOWN:
                    lidar_scroll = lidar_scroll + 1

        # --- Playback timing ---
        if not paused and now >= next_due:
            next_frame = stream.next_frame()
            if next_frame is None:
                # EOF
                paused = True
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
                next_due = now + (dt_last / float(args.rate))

                # Invalidate lidar cache for new frame
                cached_lidar_key = None

        # --- Draw ---
        screen.fill((20, 20, 25))

        y = 10
        line_h = 22

        header_lines = [
            f"File: {os.path.basename(args.logfile)}",
            f"Playback: {'PAUSED' if paused else 'RUNNING'}   speed={args.rate:.2f}x   (Space=pause, F=full lidar, R=restart)",
        ]

        if frame is None:
            header_lines.append("Waiting for first LiDAR frame...")
        else:
            # Basic sanity checks to help you verify conversions
            lidar = frame.lidar_m
            lidar_min = float(np.min(lidar)) if lidar.size else float('nan')
            lidar_max = float(np.max(lidar)) if lidar.size else float('nan')
            lidar_mean = float(np.mean(lidar)) if lidar.size else float('nan')

            header_lines += [
                f"Frame #{stream.frame_index:06d}    ts={frame.ts_str}    t_sec={frame.t_sec:9.3f}    dt~{dt_last:0.3f}s",
                f"Pose:  x={frame.x_m:+0.3f} m   y={frame.y_m:+0.3f} m   yaw={frame.yaw_deg:0.2f} deg   (hdg_ref={frame.hdg_ref_deg})",
                f"Vel:   vx={frame.vx_mps:+0.3f} m/s   vy={frame.vy_mps:+0.3f} m/s   speed={frame.speed_mps:0.3f} m/s",
                f"LiDAR: beams={lidar.size}   units=m (dm*0.1)   min/mean/max={lidar_min:0.2f}/{lidar_mean:0.2f}/{lidar_max:0.2f}",
            ]

        for s in header_lines:
            surf = font.render(s, True, (235, 235, 245))
            screen.blit(surf, (10, y))
            y += line_h

        y += 10

        # Lidar text
        if frame is not None:
            if show_full_lidar:
                # cache key: (id(array), scroll) isn't stable because array is recreated; use frame index
                cache_key = (stream.frame_index,)
                if cache_key != cached_lidar_key:
                    cached_lidar_lines = format_lidar_lines(frame.lidar_m, per_line=12, precision=1)
                    cached_lidar_key = cache_key

                # how many lines can we show
                max_lines_on_screen = max(1, (h - y - 20) // 18)
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
                # Summary view: show first + last few beams and a quick histogram-ish check
                lidar = frame.lidar_m
                first = ", ".join(f"{float(x):0.1f}" for x in lidar[:12])
                last = ", ".join(f"{float(x):0.1f}" for x in lidar[-12:])
                screen.blit(font.render("LiDAR summary (press F for full list)", True, (200, 200, 210)), (10, y))
                y += 22
                screen.blit(small.render(f"first 12: [{first}]", True, (210, 210, 220)), (10, y))
                y += 18
                screen.blit(small.render(f" last 12: [{last}]", True, (210, 210, 220)), (10, y))
                y += 18

        pygame.display.flip()
        clock.tick(args.fps)

    stream.close()
    pygame.quit()


if __name__ == "__main__":
    main()
