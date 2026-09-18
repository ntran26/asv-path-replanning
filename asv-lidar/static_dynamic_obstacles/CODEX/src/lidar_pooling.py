"""Sector pooling: raw LiDAR beams -> `c_t`, the pooled closeness vector.

Carried over from Paper 2 with **one substantive change**: sectors are no longer
uniform, so beams are assigned to sectors by *bearing* against explicit sector
edges rather than by index via `np.array_split`.

What did NOT change
-------------------
`feasibility_pool` -- Algorithm 1 -- is untouched.  It computes the arc width
between neighbouring beams from the per-beam angular resolution theta, which is
a constant 0.5 deg across the whole scan.  Only the sector *span* Phi varies, and
Phi never enters the algorithm: it only decides which beams are in which group.
That is the point 01 §2.2 makes, and it is why the pooling itself needs no edit.

What did change, and why it matters
-----------------------------------
Paper 2 pooled with `np.array_split(raw, n_sectors)`, which splits the beam
array by *index*.  That equals an angular split only on a uniform grid.  With
the non-uniform allocation in `constants.POOL_BANDS` it does not, so the
assignment is now done with `np.searchsorted` over `constants.sector_edges()`.

Paper 2 also computed theta as `swath / (n_beams - 1)` over the *full* scan and
passed the same value to every sector.  Numerically that was 270/224 = 1.205 deg.
The real sensor is 0.5 deg, so theta changes by 5.4x between Paper 2 and Paper 3.
The pooling *code* is unchanged; the pooled *values* are not comparable across
the two papers.  See PORTING_MANIFEST.md F4.

`c_t` carries **static obstacles only**.  Borders are gated out upstream and
dynamic targets go to the target branch.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

import constants as cfg


def beam_bearings(n_beams: int = cfg.LIDAR_BEAMS) -> np.ndarray:
    """Bearing of every raw beam, in degrees, wrapped to [-180, 180).

    Bin 0 is dead ahead and bins advance clockwise (to starboard), matching the
    field logs, whose config line records `lidar_index0_deg=0`.
    """
    raw = np.arange(int(n_beams), dtype=np.float64) * (360.0 / float(n_beams))
    return (raw + 180.0) % 360.0 - 180.0


def sector_assignment(bearings: np.ndarray | None = None) -> np.ndarray:
    """Sector index for every beam; -1 for beams outside the pooled swath.

    This is the whole of the per-sector-Phi change.  Everything downstream just
    groups by the returned label.
    """
    if bearings is None:
        bearings = beam_bearings()
    edges = cfg.sector_edges()
    # `right=False` puts a beam exactly on an edge into the sector above it, so
    # sectors are half-open [lo, hi) and no beam is double-counted.
    idx = np.searchsorted(edges, np.asarray(bearings, dtype=np.float64), side="right") - 1
    outside = (idx < 0) | (idx >= cfg.LIDAR_SECTORS)
    idx = idx.astype(np.int64)
    idx[outside] = -1
    return idx


def sector_centres() -> np.ndarray:
    """Centre bearing of each sector, degrees."""
    edges = cfg.sector_edges()
    return (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)


def sector_spans() -> np.ndarray:
    """Angular span Phi of each sector, degrees."""
    return np.diff(cfg.sector_edges()).astype(np.float32)


def closeness_from_ranges(sector_ranges, lidar_range: float = cfg.LIDAR_RANGE) -> np.ndarray:
    """Map ranges to [0, 1] closeness: 1 = touching, 0 = clear to max range.

    Identical to Paper 2, and reused verbatim by the boundary branch so the two
    normalisations cannot drift apart (01 §3.2).
    """
    ranges = np.clip(np.asarray(sector_ranges, dtype=np.float32), 0.0, float(lidar_range))
    return np.clip(1.0 - ranges / float(lidar_range), 0.0, 1.0).astype(np.float32)


def feasibility_pool(sector_ranges, safe_width_m: float, neighbour_angle_rad: float) -> float:
    """Algorithm 1, after Meyer et al.  Unchanged from Paper 2.

    Sort the sector's beams by ascending range.  At each candidate distance
    level `xi`, sweep the sector in angular order and accumulate the widest
    contiguous opening formed by beams reaching past `xi`.  The arc between
    neighbouring beams at that range is `neighbour_angle_rad * xi`.  The first
    level with no opening wider than `safe_width_m` is the maximum feasible
    distance: the vessel cannot fit through anything beyond it.
    """
    x = np.asarray(sector_ranges, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float(cfg.LIDAR_RANGE)
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


def pool_to_sectors(
    raw_ranges,
    bearings: np.ndarray | None = None,
    *,
    safe_width_m: float,
    lidar_range: float = cfg.LIDAR_RANGE,
    beam_res_deg: float = cfg.LIDAR_BEAM_RES_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool a full raw scan into (sector_ranges, sector_closeness).

    `raw_ranges` is the whole revolution (720 beams).  Beams outside the pooled
    +/-135 deg swath are dropped here: the aft 90 deg belongs to the tracker.
    """
    raw = np.clip(np.asarray(raw_ranges, dtype=np.float64).reshape(-1), 0.0, float(lidar_range))
    if bearings is None:
        bearings = beam_bearings(raw.size)
    labels = sector_assignment(bearings)

    # theta is per-beam and constant; only the grouping above varies by sector.
    neighbour_angle = np.radians(float(beam_res_deg))

    pooled = np.full(cfg.LIDAR_SECTORS, float(lidar_range), dtype=np.float32)
    for s in range(cfg.LIDAR_SECTORS):
        beams = raw[labels == s]
        if beams.size:
            pooled[s] = feasibility_pool(beams, float(safe_width_m), neighbour_angle)

    pooled = np.clip(pooled, 0.0, float(lidar_range)).astype(np.float32)
    return pooled, closeness_from_ranges(pooled, lidar_range)
