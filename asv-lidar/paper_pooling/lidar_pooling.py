"""Shared LiDAR sector pooling utilities for ASV simulation and UDP deployment.

Supported pooling modes
-----------------------
"min"
    Old conservative sector pooling: each sector is the minimum raw beam range.

"paper"
    Meyer/current-paper feasibility pooling. Raw beams are split into angular
    sectors. For each distance level, the algorithm checks whether the widest
    contiguous angular opening at that level is wider than the safety-adjusted
    vessel width. If not, that distance is returned as the maximum feasible
    sector distance.

"corridor"
    Corridor-based footprint pooling. For each sector centre direction, all raw
    returns are projected into a frame aligned with that centreline. A return
    blocks the sector only if it lies within a safety-adjusted vessel-width
    corridor. The sector distance is the nearest blocking forward projection.

The same helper can be imported by ``asv_lidar.py`` and ``udp_live_rl.py`` so
simulation and deployment use the same 25-sector observation semantics.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def normalise_pooling_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    aliases = {
        "meyer": "paper",
        "current_paper": "paper",
        "paper_feasibility": "paper",
        "feasibility": "paper",
        "corridor_feasibility": "corridor",
        "footprint": "corridor",
        "minimum": "min",
    }
    mode_norm = aliases.get(mode_norm, mode_norm)
    if mode_norm not in {"min", "paper", "corridor"}:
        raise ValueError(
            f"Unknown LiDAR pooling mode {mode!r}. Use 'min', 'paper', or 'corridor'."
        )
    return mode_norm


def sector_angle_grid(lidar_swath_deg: float, n_sectors: int) -> np.ndarray:
    """Return sector centre bearings in the ASV body frame."""
    return np.linspace(
        -float(lidar_swath_deg) / 2.0,
        float(lidar_swath_deg) / 2.0,
        int(n_sectors),
        dtype=np.float32,
    )


def closeness_from_ranges(sector_ranges: np.ndarray, lidar_range: float) -> np.ndarray:
    ranges = np.clip(np.asarray(sector_ranges, dtype=np.float32), 0.0, float(lidar_range))
    close = 1.0 - ranges / float(lidar_range)
    return np.clip(close, 0.0, 1.0).astype(np.float32)


def meyer_feasibility_pool(
    sector_ranges: np.ndarray,
    *,
    safe_width_m: float,
    neighbour_angle_rad: float,
    lidar_range: float,
) -> float:
    """Meyer/current-paper feasibility pooling for one angular sector.

    This implements the algorithm structure from Meyer et al.: sort sector
    sensor indices by ascending range. At each candidate distance level ``xi``,
    scan the sector in angular order and accumulate the widest contiguous
    opening whose beams are farther than ``xi``. If no opening wider than
    ``safe_width_m`` exists, ``xi`` is the maximum feasible distance.

    The arc length between neighbouring beams at distance ``xi`` is approximated
    as ``neighbour_angle_rad * xi``, matching the small-angle form used in the
    paper pseudocode.
    """
    x = np.clip(np.asarray(sector_ranges, dtype=np.float64).reshape(-1), 0.0, float(lidar_range))
    n = int(x.size)
    if n == 0:
        return float(lidar_range)
    if n == 1:
        return float(x[0])

    order = np.argsort(x)
    theta = float(neighbour_angle_rad)
    W = float(safe_width_m)

    for idx in order:
        xi = float(x[int(idx)])
        if xi <= 0.0:
            continue

        # Approximate opening width contribution between neighbouring beams
        # at this range level.
        di = max(theta * xi, 0.0)
        opening_width = 0.5 * di
        opening_found = False

        # Traverse beams in angular order. Beams farther than xi contribute a
        # full neighbour arc. Beams at/nearer than xi close the current opening,
        # but contribute half an arc before reset, matching Meyer pseudocode.
        for xj in x:
            if float(xj) > xi:
                opening_width += di
                if opening_width > W:
                    opening_found = True
                    break
            else:
                opening_width += 0.5 * di
                if opening_width > W:
                    opening_found = True
                    break
                opening_width = 0.0

        if not opening_found:
            return float(xi)

    # All distance levels were feasible: sector is clear up to the farthest
    # measured range in this sector.
    return float(np.max(x))


def _angle_diff_deg(a: np.ndarray, b: float) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) - float(b) + 180.0) % 360.0 - 180.0


def corridor_feasibility_pool(
    raw_ranges: np.ndarray,
    raw_angles_deg: np.ndarray,
    *,
    center_angle_deg: float,
    safe_width_m: float,
    lidar_range: float,
) -> float:
    """Corridor/footprint feasibility pooling for one sector centre direction."""
    ranges = np.clip(np.asarray(raw_ranges, dtype=np.float64).reshape(-1), 0.0, float(lidar_range))
    angles = np.asarray(raw_angles_deg, dtype=np.float64).reshape(-1)
    if ranges.size == 0 or angles.size == 0:
        return float(lidar_range)
    if ranges.shape[0] != angles.shape[0]:
        raise ValueError(
            "raw_ranges and raw_angles_deg must have the same length, got "
            f"{ranges.shape[0]} and {angles.shape[0]}"
        )

    diff_rad = np.radians(_angle_diff_deg(angles, float(center_angle_deg)))
    forward = ranges * np.cos(diff_rad)
    lateral = ranges * np.sin(diff_rad)
    half_width = 0.5 * float(safe_width_m)

    hit_mask = (
        (ranges < float(lidar_range) - 1e-6)
        & (forward > 0.0)
        & (forward <= float(lidar_range))
        & (np.abs(lateral) <= half_width)
    )
    if np.any(hit_mask):
        return float(np.min(forward[hit_mask]))
    return float(lidar_range)


def pool_lidar_to_sectors_shared(
    raw_ranges: np.ndarray,
    *,
    lidar_range: float,
    lidar_swath_deg: float,
    n_sectors: int,
    safe_width_m: float,
    mode: str,
    raw_angles_deg: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool raw LiDAR ranges into sector ranges, closeness, and sector angles."""
    raw = np.clip(np.asarray(raw_ranges, dtype=np.float32).reshape(-1), 0.0, float(lidar_range))
    n_sectors = int(n_sectors)
    sector_angles = sector_angle_grid(float(lidar_swath_deg), n_sectors)
    mode_norm = normalise_pooling_mode(mode)

    if mode_norm == "min":
        sectors = np.array_split(raw.astype(np.float64), n_sectors)
        sector_ranges = np.array(
            [float(np.min(sec)) if len(sec) else float(lidar_range) for sec in sectors],
            dtype=np.float32,
        )

    elif mode_norm == "paper":
        sectors = np.array_split(raw.astype(np.float64), n_sectors)
        # Use the actual number of beams in each split where possible, because
        # np.array_split may not create exactly equal sector lengths.
        pooled = []
        for sec in sectors:
            if len(sec) == 0:
                pooled.append(float(lidar_range))
                continue
            if len(sec) == 1:
                pooled.append(float(sec[0]))
                continue
            neighbour_angle = np.radians(float(lidar_swath_deg) / max(float(raw.size - 1), 1.0))
            pooled.append(
                meyer_feasibility_pool(
                    sec,
                    safe_width_m=float(safe_width_m),
                    neighbour_angle_rad=float(neighbour_angle),
                    lidar_range=float(lidar_range),
                )
            )
        sector_ranges = np.asarray(pooled, dtype=np.float32)

    elif mode_norm == "corridor":
        if raw_angles_deg is None:
            raw_angles_deg = np.linspace(
                -float(lidar_swath_deg) / 2.0,
                float(lidar_swath_deg) / 2.0,
                int(raw.size),
                dtype=np.float32,
            )
        raw_angles = np.asarray(raw_angles_deg, dtype=np.float32).reshape(-1)
        sector_ranges = np.array(
            [
                corridor_feasibility_pool(
                    raw,
                    raw_angles,
                    center_angle_deg=float(a),
                    safe_width_m=float(safe_width_m),
                    lidar_range=float(lidar_range),
                )
                for a in sector_angles
            ],
            dtype=np.float32,
        )

    else:  # pragma: no cover, normalise_pooling_mode already checks this.
        raise ValueError(mode_norm)

    sector_ranges = np.clip(sector_ranges, 0.0, float(lidar_range)).astype(np.float32)
    sector_closeness = closeness_from_ranges(sector_ranges, float(lidar_range))
    return sector_ranges, sector_closeness, sector_angles.astype(np.float32)
