
import math
import pygame
import numpy as np
from ship_model import VESSEL_LENGTH, VESSEL_WIDTH, HULL_MARGIN

# Paper-inspired sensor layout:
# 240 degree swath, 225 beams, pooled into 25 sectors.
LIDAR_RANGE = 16.0
LIDAR_SWATH = 240.0
LIDAR_BEAMS = 225
LIDAR_SECTORS = 25
BEAMS_PER_SECTOR = LIDAR_BEAMS // LIDAR_SECTORS


class Lidar:
    """LIDAR simulator with raw beams + sector pooling.

    Raw beam geometry is kept for reward/debug rendering, while the policy can
    consume the pooled sector features through `sector_closeness`.
    """

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
        self._angles = np.linspace(
            -LIDAR_SWATH / 2.0,
            LIDAR_SWATH / 2.0,
            LIDAR_BEAMS,
            dtype=np.float64,
        )
        self._ranges = np.ones(LIDAR_BEAMS, dtype=np.float32) * float(LIDAR_RANGE)
        self._sector_ranges = np.ones(LIDAR_SECTORS, dtype=np.float32) * float(LIDAR_RANGE)
        self._sector_closeness = np.zeros(LIDAR_SECTORS, dtype=np.float32)

    @property
    def angles(self) -> np.ndarray:
        return self._angles.copy()

    @property
    def ranges(self) -> np.ndarray:
        return self._ranges.copy()

    @property
    def sector_ranges(self) -> np.ndarray:
        return self._sector_ranges.copy()

    @property
    def sector_closeness(self) -> np.ndarray:
        return self._sector_closeness.copy()

    @staticmethod
    def _cross(a, b) -> float:
        return a[0] * b[1] - a[1] * b[0]

    def line_intersection(self, a1, a2, b1, b2):
        """Return (x, y) if the line segments intersect, else None."""
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
            return (
                a1[0] + scalar_a * a[0],
                a1[1] + scalar_a * a[1],
            )
        return None

    @staticmethod
    def _feasibility_pool(sector_ranges: np.ndarray, vessel_width: float, sector_angle_span_rad: float) -> float:
        """Paper-inspired feasibility pooling.

        The returned distance is the smallest range in the sector that still
        admits a feasible opening wider than the vessel. If no such opening is
        found, fall back to the maximum sector range.
        """
        n = int(len(sector_ranges))
        if n == 0:
            return float(LIDAR_RANGE)
        if n == 1:
            return float(sector_ranges[0])

        beam_spacing = sector_angle_span_rad / float(n - 1)
        order = np.argsort(sector_ranges)
        for idx in order:
            xi = float(sector_ranges[idx])
            arc_len = beam_spacing * xi
            opening_width = arc_len * 0.5
            found = False
            for j in range(n):
                if float(sector_ranges[j]) > xi:
                    opening_width += arc_len
                else:
                    opening_width += arc_len * 0.5
                if opening_width > vessel_width:
                    found = True
                    break
            if found:
                return xi
        return float(np.max(sector_ranges))

    def _pool_sectors(self) -> None:
        vessel_width = float(VESSEL_WIDTH + 2.0 * HULL_MARGIN)
        sector_angle = math.radians(LIDAR_SWATH / LIDAR_SECTORS)

        sector_ranges = np.empty(LIDAR_SECTORS, dtype=np.float32)
        for i in range(LIDAR_SECTORS):
            start = i * BEAMS_PER_SECTOR
            end = start + BEAMS_PER_SECTOR
            pooled = self._feasibility_pool(self._ranges[start:end], vessel_width, sector_angle)
            sector_ranges[i] = float(np.clip(pooled, 0.0, LIDAR_RANGE))

        self._sector_ranges = sector_ranges
        self._sector_closeness = np.clip(1.0 - (sector_ranges / float(LIDAR_RANGE)), 0.0, 1.0).astype(np.float32)

    def scan(self, pos, hdg, obstacles=None, map_border=None) -> np.ndarray:
        """Perform a scan and update both raw beams and pooled sectors."""
        self._hdg = float(hdg)

        lidar_offset = float(VESSEL_LENGTH / 2.0)
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
            end_x = self._pos_x + float(LIDAR_RANGE) * math.sin(absolute_angle)
            end_y = self._pos_y + float(LIDAR_RANGE) * math.cos(absolute_angle)
            closest_distance = float(LIDAR_RANGE)

            for edge in obstacle_edges:
                intersection = self.line_intersection(
                    (self._pos_x, self._pos_y),
                    (end_x, end_y),
                    edge[0],
                    edge[1],
                )
                if intersection is not None:
                    dist = math.hypot(intersection[0] - self._pos_x, intersection[1] - self._pos_y)
                    if dist < closest_distance:
                        closest_distance = dist

            self._ranges[idx] = float(np.clip(closest_distance, 0.0, LIDAR_RANGE))

        self._pool_sectors()
        return self._ranges.copy()

    def render(self, surface, world_to_screen) -> None:
        """Render raw beams for debugging."""
        origin = world_to_screen((self._pos_x, self._pos_y))
        # Draw every 2nd beam to keep the render readable.
        for idx, angle in enumerate(self._angles):
            if idx % 2 != 0:
                continue
            absolute_angle = math.radians(self._hdg + float(angle))
            x = self._pos_x + float(self._ranges[idx]) * math.sin(absolute_angle)
            y = self._pos_y + float(self._ranges[idx]) * math.cos(absolute_angle)
            end = world_to_screen((x, y))
            pygame.draw.line(surface, (90, 90, 200), origin, end, 1)
