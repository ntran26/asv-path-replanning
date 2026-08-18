"""Simulated 2-D LiDAR and the sector pooling that turns it into an observation.

The raw sensor casts `LIDAR_BEAMS` rays over `LIDAR_SWATH` degrees.  The policy
never sees those directly: they are pooled down to `LIDAR_SECTORS` values.
Three pooling modes are available, all sharing the same output semantics so
simulation and deployment can be swapped freely:

"min"
    Each sector is the minimum raw beam range.  Conservative, and blind to
    whether a gap is actually wide enough to drive through.

"paper"
    Feasibility pooling after Meyer et al., and the mode used for the published
    results.  A distance level is reported as blocked unless the widest
    contiguous opening at that level exceeds the safety-adjusted vessel width.

"corridor"
    Footprint pooling.  Returns are projected into a frame aligned with the
    sector centreline; only those inside a vessel-width corridor count.
"""

from __future__ import annotations

import numpy as np

from ship import HULL_MARGIN, LIDAR_OFFSET_M, VESSEL_WIDTH

# Raw sensor.
LIDAR_RANGE = 16.0
LIDAR_SWATH = 270.0
LIDAR_BEAMS = 225

# What the policy observes.
LIDAR_SECTORS = 25

# Hardcoded for this branch so several branches can train at once without
# fighting over an environment variable.
LIDAR_POOLING_MODE = "paper"

# Safety-adjusted beam width used by the "paper" and "corridor" modes.
# Matches the inflated collision hull.
FEASIBILITY_SAFE_WIDTH = float(VESSEL_WIDTH + 2.0 * HULL_MARGIN)

POOLING_MODES = ("min", "paper", "corridor")


def check_pooling_mode(mode: str) -> str:
    if mode not in POOLING_MODES:
        raise ValueError(f"Unknown LiDAR pooling mode {mode!r}. Use one of {POOLING_MODES}.")
    return mode


def sector_angle_grid(swath_deg: float = LIDAR_SWATH, n_sectors: int = LIDAR_SECTORS) -> np.ndarray:
    """Sector centre bearings in the vessel body frame."""
    return np.linspace(-swath_deg / 2.0, swath_deg / 2.0, n_sectors, dtype=np.float32)


def closeness_from_ranges(sector_ranges: np.ndarray, lidar_range: float = LIDAR_RANGE) -> np.ndarray:
    """Map ranges to [0, 1] closeness: 1 = touching, 0 = clear to max range."""
    ranges = np.clip(np.asarray(sector_ranges, dtype=np.float32), 0.0, lidar_range)
    return np.clip(1.0 - ranges / lidar_range, 0.0, 1.0).astype(np.float32)


def meyer_feasibility_pool(sector_ranges, safe_width_m: float, neighbour_angle_rad: float) -> float:
    """Feasibility pooling for a single sector (Meyer et al.).

    Sort the sector beams by ascending range.  At each candidate distance level
    `xi`, sweep the sector in angular order and accumulate the widest contiguous
    opening formed by beams that reach past `xi`.  The arc between neighbouring
    beams is approximated as `neighbour_angle_rad * xi`.  The first level with
    no opening wider than `safe_width_m` is the maximum feasible distance.
    """
    x = np.asarray(sector_ranges, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float(LIDAR_RANGE)
    if x.size == 1:
        return float(x[0])

    for idx in np.argsort(x):
        xi = float(x[idx])
        if xi <= 0.0:
            continue

        arc = max(neighbour_angle_rad * xi, 0.0)
        opening = 0.5 * arc
        opening_found = False

        for xj in x:
            if xj > xi:
                opening += arc
                if opening > safe_width_m:
                    opening_found = True
                    break
            else:
                # A blocking beam contributes half an arc, then closes the gap.
                opening += 0.5 * arc
                if opening > safe_width_m:
                    opening_found = True
                    break
                opening = 0.0

        if not opening_found:
            return xi

    # Every level was feasible, so the sector is clear out to its farthest beam.
    return float(np.max(x))


def corridor_feasibility_pool(raw_ranges, raw_angles_deg, center_angle_deg: float,
                              safe_width_m: float, lidar_range: float = LIDAR_RANGE) -> float:
    """Footprint pooling for a single sector centre direction."""
    ranges = np.clip(np.asarray(raw_ranges, dtype=np.float64), 0.0, lidar_range)
    angles = np.asarray(raw_angles_deg, dtype=np.float64)
    if ranges.shape != angles.shape:
        raise ValueError(f"range/angle length mismatch: {ranges.shape} vs {angles.shape}")

    offset = np.radians((angles - center_angle_deg + 180.0) % 360.0 - 180.0)
    forward = ranges * np.cos(offset)
    lateral = ranges * np.sin(offset)

    blocking = (
        (ranges < lidar_range - 1e-6)
        & (forward > 0.0)
        & (forward <= lidar_range)
        & (np.abs(lateral) <= 0.5 * safe_width_m)
    )
    return float(np.min(forward[blocking])) if np.any(blocking) else float(lidar_range)


def pool_to_sectors(raw_ranges, raw_angles_deg, mode: str = LIDAR_POOLING_MODE,
                    n_sectors: int = LIDAR_SECTORS, safe_width_m: float = FEASIBILITY_SAFE_WIDTH,
                    lidar_range: float = LIDAR_RANGE, swath_deg: float = LIDAR_SWATH):
    """Pool raw ranges into (sector_ranges, sector_closeness, sector_angles)."""
    check_pooling_mode(mode)
    raw = np.clip(np.asarray(raw_ranges, dtype=np.float32).reshape(-1), 0.0, lidar_range)
    sector_angles = sector_angle_grid(swath_deg, n_sectors)

    if mode == "corridor":
        pooled = [
            corridor_feasibility_pool(raw, raw_angles_deg, a, safe_width_m, lidar_range)
            for a in sector_angles
        ]
    else:
        chunks = np.array_split(raw.astype(np.float64), n_sectors)
        if mode == "min":
            pooled = [float(np.min(c)) if len(c) else lidar_range for c in chunks]
        else:
            neighbour_angle = np.radians(swath_deg / max(float(raw.size - 1), 1.0))
            pooled = [meyer_feasibility_pool(c, safe_width_m, neighbour_angle) for c in chunks]

    sector_ranges = np.clip(np.asarray(pooled, dtype=np.float32), 0.0, lidar_range).astype(np.float32)
    return sector_ranges, closeness_from_ranges(sector_ranges, lidar_range), sector_angles


class Lidar:
    """Ray-cast LiDAR against polygon obstacles, with sector pooling applied.

    Attributes exposed for the environment and for rendering:
        angles, ranges                  raw beams (float64)
        sector_angles, sector_ranges    pooled beams (float32)
        sector_closeness                pooled beams as [0, 1] closeness
    """

    def __init__(self) -> None:
        self.angles = np.linspace(-LIDAR_SWATH / 2.0, LIDAR_SWATH / 2.0, LIDAR_BEAMS, dtype=np.float64)
        self.pos = (0.0, 0.0)
        self.heading = 0.0
        self.reset()

    def reset(self) -> None:
        self.pos = (0.0, 0.0)
        self.heading = 0.0
        self.ranges = np.full(LIDAR_BEAMS, LIDAR_RANGE, dtype=np.float64)
        self._pool()

    def _pool(self) -> None:
        self.sector_ranges, self.sector_closeness, self.sector_angles = pool_to_sectors(
            self.ranges.astype(np.float32),
            self.angles.astype(np.float32),
        )

    def scan(self, pos, heading_deg, obstacles=None, map_border=None) -> np.ndarray:
        """Cast all beams from `pos` and repool.  Returns the raw ranges."""
        self.heading = float(heading_deg)
        heading_rad = np.radians(self.heading)
        # The sensor is mounted forward of the vessel origin.
        origin_x = float(pos[0]) + LIDAR_OFFSET_M * np.sin(heading_rad)
        origin_y = float(pos[1]) + LIDAR_OFFSET_M * np.cos(heading_rad)
        self.pos = (origin_x, origin_y)

        beam_angles = np.radians(self.heading + self.angles)
        beam_x = (origin_x + LIDAR_RANGE * np.sin(beam_angles)) - origin_x
        beam_y = (origin_y + LIDAR_RANGE * np.cos(beam_angles)) - origin_y
        ranges = np.full(LIDAR_BEAMS, LIDAR_RANGE, dtype=np.float64)

        for (ex0, ey0), (ex1, ey1) in _polygon_edges(obstacles, map_border):
            edge_x = ex1 - ex0
            edge_y = ey1 - ey0
            denom = beam_x * edge_y - beam_y * edge_x
            if not np.any(denom != 0.0):
                continue

            to_edge_x = ex0 - origin_x
            to_edge_y = ey0 - origin_y
            with np.errstate(divide="ignore", invalid="ignore"):
                # t runs along the beam, s along the edge; both in [0, 1] on a hit.
                t = (to_edge_x * edge_y - to_edge_y * edge_x) / denom
                s = (to_edge_x * beam_y - to_edge_y * beam_x) / denom

            hit = (denom != 0.0) & (t >= 0.0) & (t <= 1.0) & (s >= 0.0) & (s <= 1.0)
            if not np.any(hit):
                continue

            # Zero the misses so parallel beams cannot spread NaN into the maths.
            t = np.where(hit, t, 0.0)
            dist = np.hypot((origin_x + t * beam_x) - origin_x,
                            (origin_y + t * beam_y) - origin_y)
            ranges = np.where(hit, np.minimum(ranges, dist), ranges)

        self.ranges = ranges
        self._pool()
        return self.ranges


def _polygon_edges(obstacles, map_border):
    """Yield every (start, end) segment of the given polygons/polylines."""
    for group in (obstacles, map_border):
        for poly in group or ():
            for i in range(len(poly)):
                yield poly[i], poly[(i + 1) % len(poly)]
