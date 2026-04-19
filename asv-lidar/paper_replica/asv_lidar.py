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

    @property
    def sector_angles(self) -> np.ndarray:
        """Return the center angle of each pooled sector, in degrees."""
        sector_width = float(LIDAR_SWATH) / float(LIDAR_SECTORS)
        return np.linspace(
            -LIDAR_SWATH / 2.0 + sector_width / 2.0,
            LIDAR_SWATH / 2.0 - sector_width / 2.0,
            LIDAR_SECTORS,
            dtype=np.float32,
        )

    @property
    def sector_edges(self) -> np.ndarray:
        """Return sector boundary angles, in degrees."""
        return np.linspace(
            -LIDAR_SWATH / 2.0,
            LIDAR_SWATH / 2.0,
            LIDAR_SECTORS + 1,
            dtype=np.float32,
        )

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

    @staticmethod
    def _sector_color(closeness: float, *, alpha: int = 120) -> tuple[int, int, int, int]:
        """Color map for the pooled observation value seen by the policy.

        closeness = 0 means clear/max range; closeness = 1 means very close.
        The returned color moves from blue/green to yellow/red as danger grows.
        """
        c = float(np.clip(closeness, 0.0, 1.0))
        if c < 0.5:
            # Clear -> caution: blue/green to yellow.
            t = c / 0.5
            r = int(40 + 190 * t)
            g = int(180 + 40 * t)
            b = int(220 * (1.0 - t))
        else:
            # Caution -> close obstacle: yellow to red.
            t = (c - 0.5) / 0.5
            r = int(230 + 25 * t)
            g = int(220 * (1.0 - t))
            b = 0
        return (r, g, b, int(alpha))

    def _beam_endpoint(self, angle_deg: float, range_m: float) -> tuple[float, float]:
        absolute_angle = math.radians(self._hdg + float(angle_deg))
        r = float(np.clip(range_m, 0.0, LIDAR_RANGE))
        return (
            self._pos_x + r * math.sin(absolute_angle),
            self._pos_y + r * math.cos(absolute_angle),
        )

    def render(self, surface, world_to_screen) -> None:
        """Render raw beams plus the 25 pooled sector observations.

        Visual layers:
        1. Thin raw LiDAR beams: the full range scan before pooling.
        2. Semi-transparent sector wedges: the 25 pooled observations seen by the policy.
        3. Thick sector center rays + endpoint markers: pooled range/closeness per sector.

        The sector color encodes `sector_closeness`:
            blue/green = mostly clear, yellow = caution, red = close obstacle.
        """
        origin = world_to_screen((self._pos_x, self._pos_y))

        # 1) Raw full-resolution beams. Keep all beams visible, but dim them so
        # the pooled observation layer remains readable.
        for idx, angle in enumerate(self._angles):
            raw_range = float(self._ranges[idx])
            end = world_to_screen(self._beam_endpoint(float(angle), raw_range))
            if raw_range >= LIDAR_RANGE * 0.995:
                color = (55, 60, 85)
            else:
                color = (95, 115, 210)
            pygame.draw.line(surface, color, origin, end, 1)

        # 2) Pooled sector wedges. These show the compressed 25-sector view.
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        sector_edges = self.sector_edges
        sector_centers = self.sector_angles

        for i in range(LIDAR_SECTORS):
            pooled_range = float(self._sector_ranges[i])
            closeness = float(self._sector_closeness[i])

            # Use the pooled sector range because this is the compressed
            # distance represented by the observation feature.
            left_px = world_to_screen(self._beam_endpoint(float(sector_edges[i]), pooled_range))
            right_px = world_to_screen(self._beam_endpoint(float(sector_edges[i + 1]), pooled_range))

            fill_alpha = int(18 + 95 * closeness)
            fill_color = self._sector_color(closeness, alpha=fill_alpha)
            pygame.draw.polygon(overlay, fill_color, [origin, left_px, right_px])

        surface.blit(overlay, (0, 0))

        # 3) Sector boundaries and pooled center rays. The center ray is the
        # easiest way to read what one observation-sector contributes.
        for edge_angle in sector_edges:
            edge_end = world_to_screen(self._beam_endpoint(float(edge_angle), LIDAR_RANGE))
            pygame.draw.line(surface, (80, 80, 90), origin, edge_end, 1)

        for i, angle in enumerate(sector_centers):
            pooled_range = float(self._sector_ranges[i])
            closeness = float(self._sector_closeness[i])
            color = self._sector_color(closeness, alpha=255)[:3]

            center_end = world_to_screen(self._beam_endpoint(float(angle), pooled_range))
            pygame.draw.line(surface, color, origin, center_end, 3)
            pygame.draw.circle(surface, color, center_end, 3)

        pygame.draw.circle(surface, (245, 245, 245), origin, 3)
        pygame.draw.circle(surface, (20, 20, 20), origin, 3, 1)
