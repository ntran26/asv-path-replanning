import pygame
import numpy as np
from ship_model import VESSEL_LENGTH

# Raw simulated LiDAR
LIDAR_RANGE = 16.0
LIDAR_SWATH = 270.0
LIDAR_BEAMS = 90

# What the agent observes
LIDAR_SECTORS = 25


class Lidar:
    """Basic line-intersection LiDAR with sector pooling.

    The policy sees `sector_closeness` with shape (LIDAR_SECTORS,).
    Raw beams are kept for rendering/debugging.
    """

    def __init__(self):
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._hdg = 0.0
        self._angles = None
        self._ranges = None
        self._sector_ranges = None
        self._sector_closeness = None
        self._sector_angles = None
        self.reset()

    def reset(self):
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._hdg = 0.0
        self._angles = np.linspace(-LIDAR_SWATH / 2.0, LIDAR_SWATH / 2.0, LIDAR_BEAMS, dtype=np.float64)
        self._ranges = np.ones_like(self._angles, dtype=np.float64) * LIDAR_RANGE
        self._sector_angles = np.linspace(-LIDAR_SWATH / 2.0, LIDAR_SWATH / 2.0, LIDAR_SECTORS, dtype=np.float64)
        self._update_sectors()

    @property
    def angles(self):
        return self._angles.copy()

    @property
    def ranges(self):
        return self._ranges.copy()

    @property
    def sector_angles(self):
        return self._sector_angles.copy()

    @property
    def sector_ranges(self):
        return self._sector_ranges.copy()

    @property
    def sector_closeness(self):
        return self._sector_closeness.copy()

    def _update_sectors(self):
        sectors = np.array_split(self._ranges.astype(np.float64), LIDAR_SECTORS)
        pooled = []
        for sec in sectors:
            if sec.size == 0:
                pooled.append(LIDAR_RANGE)
            else:
                # Conservative local-planner pooling: closest beam per sector.
                pooled.append(float(np.min(sec)))
        self._sector_ranges = np.asarray(pooled, dtype=np.float32)
        self._sector_ranges = np.clip(self._sector_ranges, 0.0, LIDAR_RANGE)
        self._sector_closeness = (1.0 - self._sector_ranges / LIDAR_RANGE).astype(np.float32)
        self._sector_closeness = np.clip(self._sector_closeness, 0.0, 1.0)

    def scan(self, pos, hdg, obstacles=None, map_border=None) -> np.ndarray:
        self._hdg = float(hdg)

        # Sensor is mounted at bow/forward half-length.
        lidar_offset = VESSEL_LENGTH / 2.0
        self._pos_x = float(pos[0]) + lidar_offset * np.sin(np.radians(self._hdg))
        self._pos_y = float(pos[1]) + lidar_offset * np.cos(np.radians(self._hdg))

        obstacle_edges = []
        if obstacles:
            for obs in obstacles:
                for i in range(len(obs)):
                    obstacle_edges.append((obs[i], obs[(i + 1) % len(obs)]))
        if map_border:
            for border in map_border:
                for i in range(len(border)):
                    obstacle_edges.append((border[i], border[(i + 1) % len(border)]))

        for idx, angle in enumerate(self._angles):
            absolute_angle = np.radians(self._hdg + angle)
            end_x = self._pos_x + LIDAR_RANGE * np.sin(absolute_angle)
            end_y = self._pos_y + LIDAR_RANGE * np.cos(absolute_angle)
            closest_distance = LIDAR_RANGE

            for edge in obstacle_edges:
                intersection = self.line_intersection((self._pos_x, self._pos_y), (end_x, end_y), edge[0], edge[1])
                if intersection is not None:
                    dist = np.hypot(intersection[0] - self._pos_x, intersection[1] - self._pos_y)
                    closest_distance = min(closest_distance, dist)

            self._ranges[idx] = closest_distance

        self._update_sectors()
        return self._ranges.copy()

    def line_intersection(self, a1, a2, b1, b2):
        def cross_product(a, b):
            return a[0] * b[1] - a[1] * b[0]

        a = (a2[0] - a1[0], a2[1] - a1[1])
        b = (b2[0] - b1[0], b2[1] - b1[1])
        a_cross_b = cross_product(a, b)
        a_b = (b1[0] - a1[0], b1[1] - a1[1])
        a_b_cross_a = cross_product(a_b, a)

        if a_cross_b == 0 and a_b_cross_a == 0:
            return None
        if a_cross_b == 0:
            return None

        scalar_a = cross_product(a_b, b) / a_cross_b
        scalar_b = a_b_cross_a / a_cross_b

        if 0 <= scalar_a <= 1 and 0 <= scalar_b <= 1:
            return (a1[0] + scalar_a * a[0], a1[1] + scalar_a * a[1])
        return None

    def render(self, surface, world_to_screen):
        origin = world_to_screen((self._pos_x, self._pos_y))

        # Sector rays: what the agent actually sees.
        for angle, dist, close in zip(self._sector_angles, self._sector_ranges, self._sector_closeness):
            absolute_angle = np.radians(self._hdg + angle)
            x = self._pos_x + float(dist) * np.sin(absolute_angle)
            y = self._pos_y + float(dist) * np.cos(absolute_angle)
            end = world_to_screen((x, y))
            # clear = blue/green, close = red/yellow
            r = int(80 + 175 * float(close))
            g = int(180 * (1.0 - float(close)))
            b = int(220 * (1.0 - float(close)))
            pygame.draw.line(surface, (r, g, b), origin, end, 2)
            pygame.draw.circle(surface, (r, g, b), end, 2)

        # Raw beams, dim, for debugging full range geometry.
        for idx, angle in enumerate(self._angles):
            if idx % 2 != 0:
                continue
            absolute_angle = np.radians(self._hdg + angle)
            x = self._pos_x + self._ranges[idx] * np.sin(absolute_angle)
            y = self._pos_y + self._ranges[idx] * np.cos(absolute_angle)
            end = world_to_screen((x, y))
            color = (55, 55, 95) if self._ranges[idx] >= LIDAR_RANGE - 1e-6 else (90, 90, 160)
            pygame.draw.line(surface, color, origin, end, 1)
