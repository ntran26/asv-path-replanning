import math
from typing import List, Tuple, Optional

import numpy as np
import pygame
from ship_model import VESSEL_LENGTH

# Quick/paper-style LiDAR: raw beams are rendered, sectors are used by policy.
LIDAR_RANGE = 16.0
LIDAR_SWATH = 240.0
LIDAR_BEAMS = 225
LIDAR_SECTORS = 25
BEAMS_PER_SECTOR = LIDAR_BEAMS // LIDAR_SECTORS


class Lidar:
    def __init__(self) -> None:
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._hdg = 0.0
        self._angles = None
        self._ranges = None
        self._sector_ranges = None
        self._sector_closeness = None
        self.reset()

    def reset(self) -> None:
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._hdg = 0.0
        self._angles = np.linspace(-LIDAR_SWATH / 2.0, LIDAR_SWATH / 2.0, LIDAR_BEAMS, dtype=np.float64)
        self._ranges = np.ones(LIDAR_BEAMS, dtype=np.float32) * float(LIDAR_RANGE)
        self._sector_ranges = np.ones(LIDAR_SECTORS, dtype=np.float32) * float(LIDAR_RANGE)
        self._sector_closeness = np.zeros(LIDAR_SECTORS, dtype=np.float32)

    @property
    def angles(self):
        return self._angles.copy()

    @property
    def ranges(self):
        return self._ranges.copy()

    @property
    def sector_ranges(self):
        return self._sector_ranges.copy()

    @property
    def sector_closeness(self):
        return self._sector_closeness.copy()

    @staticmethod
    def _cross(a, b) -> float:
        return a[0] * b[1] - a[1] * b[0]

    def line_intersection(self, a1, a2, b1, b2):
        a = (a2[0] - a1[0], a2[1] - a1[1])
        b = (b2[0] - b1[0], b2[1] - b1[1])
        a_cross_b = self._cross(a, b)
        a_b = (b1[0] - a1[0], b1[1] - a1[1])
        a_b_cross_a = self._cross(a_b, a)

        if a_cross_b == 0 and a_b_cross_a == 0:
            return None
        if a_cross_b == 0:
            return None

        scalar_a = self._cross(a_b, b) / a_cross_b
        scalar_b = a_b_cross_a / a_cross_b
        if 0 <= scalar_a <= 1 and 0 <= scalar_b <= 1:
            return (a1[0] + scalar_a * a[0], a1[1] + scalar_a * a[1])
        return None

    def _pool_sectors(self) -> None:
        # Conservative min pooling is used here for simplicity and quick learning.
        ranges = self._ranges.reshape(LIDAR_SECTORS, BEAMS_PER_SECTOR)
        self._sector_ranges = np.min(ranges, axis=1).astype(np.float32)
        self._sector_closeness = np.clip(1.0 - self._sector_ranges / float(LIDAR_RANGE), 0.0, 1.0).astype(np.float32)

    def scan(self, pos, hdg, obstacles=None, map_border=None) -> np.ndarray:
        self._hdg = float(hdg)
        lidar_offset = VESSEL_LENGTH / 2.0
        self._pos_x = float(pos[0]) + lidar_offset * math.sin(math.radians(self._hdg))
        self._pos_y = float(pos[1]) + lidar_offset * math.cos(math.radians(self._hdg))

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
            absolute_angle = math.radians(self._hdg + float(angle))
            end_x = self._pos_x + LIDAR_RANGE * math.sin(absolute_angle)
            end_y = self._pos_y + LIDAR_RANGE * math.cos(absolute_angle)
            closest_distance = float(LIDAR_RANGE)
            for edge in obstacle_edges:
                intersection = self.line_intersection((self._pos_x, self._pos_y), (end_x, end_y), edge[0], edge[1])
                if intersection is not None:
                    dist = math.hypot(intersection[0] - self._pos_x, intersection[1] - self._pos_y)
                    if dist < closest_distance:
                        closest_distance = dist
            self._ranges[idx] = closest_distance

        self._pool_sectors()
        return self._ranges.copy()

    def render(self, surface, world_to_screen):
        origin = world_to_screen((self._pos_x, self._pos_y))
        # Raw beams, thinned for readability.
        for idx, angle in enumerate(self._angles):
            if idx % 3 != 0:
                continue
            absolute_angle = math.radians(self._hdg + float(angle))
            r = float(self._ranges[idx])
            x = self._pos_x + r * math.sin(absolute_angle)
            y = self._pos_y + r * math.cos(absolute_angle)
            color = (60, 70, 120) if r >= LIDAR_RANGE * 0.99 else (90, 140, 240)
            pygame.draw.line(surface, color, origin, world_to_screen((x, y)), 1)

        # Sector center rays, matching what the policy observes.
        sector_angles = np.linspace(-LIDAR_SWATH / 2.0, LIDAR_SWATH / 2.0, LIDAR_SECTORS, dtype=np.float64)
        for i, angle in enumerate(sector_angles):
            closeness = float(self._sector_closeness[i])
            r = float(self._sector_ranges[i])
            absolute_angle = math.radians(self._hdg + float(angle))
            x = self._pos_x + r * math.sin(absolute_angle)
            y = self._pos_y + r * math.cos(absolute_angle)
            # green clear -> red close
            color = (int(60 + 200 * closeness), int(220 * (1.0 - closeness)), 40)
            end = world_to_screen((x, y))
            pygame.draw.line(surface, color, origin, end, 2)
            pygame.draw.circle(surface, color, end, 3)
