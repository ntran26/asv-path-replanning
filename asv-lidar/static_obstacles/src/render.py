"""Pygame view of the environment, plus optional MP4 capture.

Only imported when `render_mode="human"`, so headless training never touches
pygame or OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np
import pygame
import pygame.freetype

import config as cfg
from lidar import LIDAR_RANGE
from ship import VESSEL_LENGTH, VESSEL_WIDTH

VIDEO_PATH = "asv_lidar.mp4"

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
MAGENTA = (200, 0, 200)
CYAN = (0, 220, 220)
AMBER = (255, 180, 0)
DARK_RED = (100, 0, 0)
BEAM_CLEAR = (55, 55, 95)
BEAM_HIT = (90, 90, 160)


def boat_icon() -> pygame.Surface:
    """A small top-down hull sprite, drawn from an RGBA array."""
    w, h = 32, 64
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[8:58, 10:22] = (210, 210, 225, 255)      # hull
    for y in range(12):                          # bow taper
        img[y, max(0, 16 - y // 2):min(w, 16 + y // 2 + 1)] = (230, 230, 245, 255)
    img[8:58, 15:17] = (70, 70, 90, 255)         # centre line
    img[8:58, 9:10] = (40, 40, 55, 255)          # outline
    img[8:58, 22:23] = (40, 40, 55, 255)
    img[57:59, 10:22] = (40, 40, 55, 255)
    return pygame.image.frombytes(img.tobytes(), (w, h), "RGBA")


class Renderer:
    def __init__(self, map_width: float, map_height: float, *, record_video: bool = False) -> None:
        pygame.init()
        self.map_height = float(map_height)
        self.scale = float(cfg.RENDER_SCALE)
        self.window_size = (int(round(map_width * self.scale)), int(round(map_height * self.scale)))

        self.surface = pygame.Surface(self.window_size)
        self.font = pygame.freetype.SysFont(pygame.font.get_default_font(), size=10)
        self.clock = pygame.time.Clock()
        self.display = None

        self.icon = pygame.transform.smoothscale(boat_icon(), (
            max(1, int(round(VESSEL_WIDTH * self.scale))),
            max(1, int(round(VESSEL_LENGTH * self.scale))),
        ))

        self.record_video = bool(record_video)
        self.video_writer = None

    def to_screen(self, xy):
        """World metres -> screen pixels, with y flipped."""
        px = int(round(float(xy[0]) * self.scale))
        py = int(round((self.map_height - float(xy[1])) * self.scale))
        return px, max(0, min(self.window_size[1] - 1, py))

    def draw(self, env) -> None:
        if self.display is None:
            self.display = pygame.display.set_mode(self.window_size)

        self.surface.fill(BLACK)
        pygame.draw.rect(self.surface, RED,
                         pygame.Rect(0, 0, self.window_size[0] - 1, self.window_size[1] - 1), width=2)

        for obs in env.obstacles:
            pygame.draw.polygon(self.surface, RED, [self.to_screen(p) for p in obs])

        self._draw_lidar(env.lidar)

        path_px = [self.to_screen(p) for p in env.path.points]
        if len(path_px) >= 2:
            pygame.draw.lines(self.surface, GREEN, False, path_px, 2)

        pygame.draw.circle(self.surface, DARK_RED, self.to_screen((env.tgt_x, env.tgt_y)), 3)
        pygame.draw.circle(self.surface, CYAN, self.to_screen((env.lookahead_x, env.lookahead_y)), 3)
        pygame.draw.circle(self.surface, MAGENTA, self.to_screen((env.goal_x, env.goal_y)), 6)

        # The lateral bypass target chosen from the LiDAR sectors.
        if abs(env.local_target_cte) > 1e-4:
            target = env.path.points[env.closest_idx] + env.local_target_cte * env.path.left_normal(env.closest_idx)
            pygame.draw.circle(self.surface, AMBER, self.to_screen(target), 5)

        rotated = pygame.transform.rotozoom(self.icon, -env.asv_h, 1)
        self.surface.blit(rotated, rotated.get_rect(center=self.to_screen((env.asv_x, env.asv_y))))
        pygame.draw.polygon(self.surface, (255, 0, 0),
                            [self.to_screen(p) for p in env.hull_polygon()], width=2)

        self._draw_status(env)

        self.display.blit(self.surface, (0, 0))
        pygame.display.update()
        self.clock.tick(cfg.RENDER_FPS)

        if self.record_video:
            self._write_frame()

    def _draw_lidar(self, lidar) -> None:
        origin = self.to_screen(lidar.pos)

        # Pooled sectors: what the policy actually sees.  Blue = clear, red = close.
        for angle, dist, close in zip(lidar.sector_angles, lidar.sector_ranges, lidar.sector_closeness):
            colour = (int(80 + 175 * close), int(180 * (1.0 - close)), int(220 * (1.0 - close)))
            end = self._beam_end(lidar, angle, dist)
            pygame.draw.line(self.surface, colour, origin, end, 2)
            pygame.draw.circle(self.surface, colour, end, 2)

        # Every second raw beam, dim, to show the underlying geometry.
        for angle, dist in zip(lidar.angles[::2], lidar.ranges[::2]):
            colour = BEAM_CLEAR if dist >= LIDAR_RANGE - 1e-6 else BEAM_HIT
            pygame.draw.line(self.surface, colour, origin, self._beam_end(lidar, angle, dist), 1)

    def _beam_end(self, lidar, angle_deg, dist):
        bearing = np.radians(lidar.heading + float(angle_deg))
        return self.to_screen((lidar.pos[0] + float(dist) * np.sin(bearing),
                               lidar.pos[1] + float(dist) * np.cos(bearing)))

    def _draw_status(self, env) -> None:
        lines = [
            f"{env.elapsed_time:05.1f}s u:{env.u_body:0.2f} v:{env.v_body:+0.2f} r:{env.asv_w:+0.1f}",
            f"cte:{env.cross_track_error:+0.2f} tgt:{env.local_target_cte:+0.2f} "
            f"front:{env.front_clearance:0.1f} "
            f"L/R:{env.left_clearance:0.1f}/{env.right_clearance:0.1f} "
            f"bc:{env.true_border_clearance:0.2f}",
            f"rud:{env.rudder:+0.2f}   thr:{env.rpm:+0.2f}",
        ]
        height = self.window_size[1]
        for text, y in zip(lines, (height - 28, height - 14, height - 600)):
            surface, _ = self.font.render(text, WHITE, BLACK)
            self.surface.blit(surface, (10, y))

    def _write_frame(self) -> None:
        frame = pygame.surfarray.array3d(self.surface)
        frame = cv2.cvtColor(cv2.flip(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 1), cv2.COLOR_RGB2BGR)
        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(
                VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), cfg.RENDER_FPS, self.window_size)
        self.video_writer.write(frame)

    def close(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.display is not None:
            pygame.display.quit()
            self.display = None
