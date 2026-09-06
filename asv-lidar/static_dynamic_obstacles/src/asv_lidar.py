"""Simulated RPLidar C1: 360 deg raycast against obstacle polygons.

Carried over from Paper 2 with **one substantive change**: this sensor sees
obstacles only.  The `map_border` argument and every border-visibility mode are
gone.  Channel boundaries reach the policy through `boundary_raycast.py`
instead, computed from the map rather than the sensor (01 §3, decision D5).

The raycast maths itself is unchanged from Paper 2's vectorised implementation.

Two sensor properties are modelled here that Paper 2 did not have:

* **360 deg / 720 beams at 0.5 deg.**  Paper 2 simulated 225 beams over 270 deg
  (1.205 deg).  The full sweep is required so the tracker can see overtaking
  vessels approaching from astern (Rule 13); the pooled `c_t` still uses only
  the forward +/-135 deg.
* **A 1 m minimum range.**  The C1 returns nothing closer than 1 m -- confirmed
  across all 30 field logs, where the smallest non-zero value is 10 dm.  Paper 2
  reported ranges down to 0.  Modelling it matters: with the sensor at the bow
  of a 1.57 m hull, a target alongside inside 1 m is invisible to the real
  sensor, and a policy trained without the dead zone would rely on returns that
  never arrive.  Collision termination stays geometric and is unaffected.
* **An aft self-occlusion mask.**  A 360 deg scanner on a hull with
  superstructure has a blind or degraded arc astern.  It is not detectable in
  the existing logs (see `constants.py` §4), so the mask half-width is 0.0 until
  a static-spin recording settles it -- but the hook is on the path, because
  this arc gates the **being-overtaken** class.  Train the tracker to see astern
  when the real mount cannot, and Rule 17 behaviour fails in the field for
  reasons that have nothing to do with the policy.
* **Per-beam dropout.**  A Study 2 axis and a stand-in for the no-return process
  the field logs show.  Nominal 0.0.
"""

from __future__ import annotations

import numpy as np

import constants as cfg
import lidar_pooling as lp
from ship import HULL_MARGIN, LIDAR_OFFSET_M, VESSEL_WIDTH

# Safety-adjusted width used by Algorithm 1.  Matches the inflated collision
# hull, exactly as in Paper 2.
FEASIBILITY_SAFE_WIDTH = float(VESSEL_WIDTH + 2.0 * HULL_MARGIN)


class Lidar:
    """Ray-cast LiDAR over obstacle polygons, with sector pooling applied.

    Attributes:
        bearings          (720,) beam bearings in the body frame, deg
        ranges            (720,) raw beam ranges, m.  `cfg.LIDAR_RANGE` = no return.
        sector_ranges     (27,)  pooled feasible ranges, m
        sector_closeness  (27,)  pooled closeness in [0, 1], the `lidar` branch
    """

    def __init__(self, *, aft_mask_half_deg: float = cfg.LIDAR_AFT_MASK_HALF_DEG,
                 dropout_p: float = cfg.LIDAR_DROPOUT_P,
                 rng=None) -> None:
        self.bearings = lp.beam_bearings()
        self.sector_angles = lp.sector_centres()
        self.aft_mask_half_deg = float(aft_mask_half_deg)
        self.dropout_p = float(dropout_p)
        self.rng = rng if rng is not None else np.random.default_rng()
        # Beams inside the masked aft arc, precomputed: |bearing| >= 180 - half.
        self.aft_mask = (np.abs(self.bearings) >= 180.0 - self.aft_mask_half_deg
                         if self.aft_mask_half_deg > 0.0
                         else np.zeros_like(self.bearings, dtype=bool))
        self.pos = (0.0, 0.0)
        self.heading = 0.0
        self.reset()

    def reset(self) -> None:
        self.pos = (0.0, 0.0)
        self.heading = 0.0
        self.ranges = np.full(cfg.LIDAR_BEAMS, cfg.LIDAR_RANGE, dtype=np.float64)
        self._pool()

    def _pool(self) -> None:
        self.sector_ranges, self.sector_closeness = lp.pool_to_sectors(
            self.ranges,
            self.bearings,
            safe_width_m=FEASIBILITY_SAFE_WIDTH,
        )

    def scan(self, pos, heading_deg: float, obstacles=None) -> np.ndarray:
        """Cast all 720 beams from `pos` against `obstacles`, then repool.

        `obstacles` is a sequence of polygons, each a sequence of (x, y).  Both
        static obstacles and target-ship hulls go in here -- the tracker
        separates them by estimated speed, not by how they were drawn.
        """
        self.heading = float(heading_deg)
        heading_rad = np.radians(self.heading)
        # The sensor is mounted forward of the vessel origin.
        origin_x = float(pos[0]) + LIDAR_OFFSET_M * np.sin(heading_rad)
        origin_y = float(pos[1]) + LIDAR_OFFSET_M * np.cos(heading_rad)
        self.pos = (origin_x, origin_y)

        beam_angles = np.radians(self.heading + self.bearings)
        beam_x = cfg.LIDAR_RANGE * np.sin(beam_angles)
        beam_y = cfg.LIDAR_RANGE * np.cos(beam_angles)
        ranges = np.full(cfg.LIDAR_BEAMS, cfg.LIDAR_RANGE, dtype=np.float64)

        for (ex0, ey0), (ex1, ey1) in _polygon_edges(obstacles):
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
            dist = np.hypot(t * beam_x, t * beam_y)
            ranges = np.where(hit, np.minimum(ranges, dist), ranges)

        self.ranges = self._degrade(apply_min_range(ranges))
        self._pool()
        return self.ranges

    def _degrade(self, ranges: np.ndarray) -> np.ndarray:
        """Apply the aft occlusion mask and per-beam dropout.

        Both produce no-returns, which are indistinguishable from "nothing out
        there" and therefore read as max range -- deliberately the
        unsafe-looking choice, because it is what the hardware does.
        """
        r = np.asarray(ranges, dtype=np.float64)
        if np.any(self.aft_mask):
            r = np.where(self.aft_mask, cfg.LIDAR_RANGE, r)
        if self.dropout_p > 0.0:
            dropped = self.rng.random(r.shape) < self.dropout_p
            r = np.where(dropped, cfg.LIDAR_RANGE, r)
        return r


def apply_min_range(ranges, min_range: float = cfg.LIDAR_MIN_RANGE,
                    max_range: float = cfg.LIDAR_RANGE) -> np.ndarray:
    """Model the sensor's near dead zone.

    A real return closer than `min_range` is not reported at all, so it becomes
    a no-return -- which is indistinguishable from "nothing out there" and
    therefore maps to max range.  That is deliberately the unsafe-looking
    choice: it is what the hardware does, and hiding it in simulation would
    train the policy to trust returns it will not get.

    Set `cfg.LIDAR_MIN_RANGE = 0.0` to disable.
    """
    r = np.asarray(ranges, dtype=np.float64)
    if min_range <= 0.0:
        return r
    return np.where(r < float(min_range), float(max_range), r)


def _polygon_edges(obstacles):
    """Yield every (start, end) segment of the given polygons."""
    for poly in obstacles or ():
        for i in range(len(poly)):
            yield poly[i], poly[(i + 1) % len(poly)]
