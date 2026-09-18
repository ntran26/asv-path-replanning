"""The navigable corridor: centreline, width profile, and the polygon (03a §3).

A corridor is a centreline polyline `C(s)` with a width profile `W(s)`.  The
navigable polygon is the offset of `C` by `±W(s)/2`.  Three distinct geometries
live in this project and conflating them is the failure this module exists to
prevent (03a §3.1):

| Polygon | Role | Visible to LiDAR? |
|---|---|---|
| **Corridor** | hard constraint for the own ship; termination on breach | **No** -- map-derived |
| **Basin envelope** | 10 x 25 m; limit of the physical water | No |
| **Facility walls** | basin envelope + 1.5 m | **Yes** -- returned, then gated |

**Why this replaces the straight inset rectangle.**  01 §3.3 records that in a
constant-width centred channel the port and starboard boundary rays are affine
functions of the cross-track error, so the 7-dimensional boundary branch carries
one number and an ablation of it would return a null result for a reason that
has nothing to do with the branch.  04a §3.2 makes variable width, bends and an
off-centre path a *hard requirement* on the generator for exactly that reason,
and asserts it: `|corr(e_y, b_i)| < 0.9` for every ray over 1000 episodes.

It also unblocks `r_path`.  With `kappa = 0` everywhere, 02a's `R-8` silently
reduces to the absolute yaw form and the term looks implemented while being
untested (02b §3.3).

**F25 -- bend magnitude is capped by channel width, and 04a does not say so.**
The corridor has to stay inside a 10 m basin, so a wide channel cannot bend.
Measured ceiling, with the chord aligned to the basin's long axis:

| `W` (m) | 10 | 9 | 8 | 7 | 6 | 5 | 4 | 3.5 |
|---|---|---|---|---|---|---|---|---|
| max `Δψ` (deg) | 0 | 5 | 16 | 27 | 39 | 51 | 60 | 60 |

Rather than emit a corridor that leaves the water -- which would forfeit the
physical-reproducibility argument O4 was resolved to protect -- the generator
**clamps the bend to what the basin admits** and records both the requested and
the realised value, so a silently straightened channel is visible rather than
inferred.

04a's "≥ 40% of episodes carry a ≥ 20° bend" is therefore a property of the
width distribution rather than a free parameter.  Measured over 300 samples of
the stage-5 range it comes out at **43%**, so the requirement is met -- but only
because narrow widths dominate.  **It is unreachable in curriculum stage 3**
(7-10 m), where the ceiling runs from 27° down to 0.  Stage 3 therefore trains
no `r_path` signal at all, which matters because it is the stage that
introduces the encounter machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np

import constants as cfg

# Centreline sample spacing.  0.05 m gives ~500 stations over a 25 m corridor,
# which is fine enough that the piecewise-linear polygon is visually smooth and
# coarse enough that `float32` differencing does not dominate the curvature --
# the trap T3 found, where curvature error *grows* with point density.
STATION_SPACING_M = 0.05

# **The boundary polygon is decimated, and it has to be.**  The centreline is
# sampled at 5 cm because `r_path` differentiates it twice and needs the
# resolution; the *polygon* does not.  Every consumer of the boundary iterates
# its edges -- 720 gated beams, 7 boundary rays, 9 channel-room raycasts per
# target, the hull containment test -- so a 1000-vertex polygon made the
# environment 250x slower than the four-corner rectangle it replaced and dropped
# it to 7 steps/s.
#
# At 0.5 m spacing a 10 m bend radius is represented to within 3 mm, which is
# two orders below the 0.25 m basin margin and four below anything the reward
# measures.
POLYGON_SPACING_M = 0.5


@dataclass
class Corridor:
    """A navigable channel: centreline, width profile, and what they imply."""

    centre: np.ndarray                  # (N, 2) centreline stations
    width: np.ndarray                   # (N,) channel width at each station
    s: np.ndarray                       # (N,) arclength along the centreline
    bend_deg: float = 0.0               # realised total heading change
    bend_requested_deg: float = 0.0     # before the basin clamp (F25)
    offset_frac: float = 0.0            # reference-path offset, +ve to starboard
    basin: Tuple[float, float] = (cfg.MAP_WIDTH, cfg.MAP_HEIGHT)

    # 06 M-1: which navigable geometry this is.  Not a dataclass field.
    mode = "channel"

    # ------------------------------------------------------------------
    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def nominal_width(self) -> float:
        return float(np.mean(self.width))

    @property
    def width_ratio(self) -> float:
        return float(np.max(self.width) / max(np.min(self.width), 1e-9))

    @property
    def has_bend(self) -> bool:
        return abs(self.bend_deg) >= cfg.CORRIDOR_BEND_MIN_DEG

    @property
    def was_clamped(self) -> bool:
        """True when the basin forced a smaller bend than was sampled (F25)."""
        return abs(self.bend_requested_deg) - abs(self.bend_deg) > 0.5

    # ------------------------------------------------------------------
    def tangent(self, index: int) -> np.ndarray:
        i = int(np.clip(index, 0, len(self.centre) - 1))
        if i == 0:
            vec = self.centre[1] - self.centre[0]
        elif i == len(self.centre) - 1:
            vec = self.centre[-1] - self.centre[-2]
        else:
            vec = self.centre[i + 1] - self.centre[i - 1]
        norm = float(np.linalg.norm(vec))
        return np.array([0.0, 1.0]) if norm < 1e-9 else vec / norm

    def left_normal(self, index: int) -> np.ndarray:
        t = self.tangent(index)
        return np.array([-t[1], t[0]])

    def index_at_s(self, s_query: float) -> int:
        return int(np.clip(np.searchsorted(self.s, float(s_query)),
                           0, len(self.s) - 1))

    def width_at_s(self, s_query: float) -> float:
        return float(np.interp(float(s_query), self.s, self.width))

    def frame_at_s(self, s_query: float):
        """(point, tangent, left_normal) at an arclength station."""
        i = self.index_at_s(s_query)
        return self.centre[i].copy(), self.tangent(i), self.left_normal(i)

    # ------------------------------------------------------------------
    def polygon(self, spacing: float = POLYGON_SPACING_M) -> list:
        """The navigable polygon: left edge forward, right edge back.

        Offsetting a curve by half its width self-intersects where the radius of
        curvature falls below the half-width, which is why `sample()` rejects
        that combination rather than emitting a polygon with a bow-tie in it --
        a self-intersecting boundary makes point-in-polygon tests return
        nonsense and the failure would present as an inexplicable termination.
        """
        left, right = self.edges()
        step = max(1, int(round(float(spacing) / STATION_SPACING_M)))
        # Keep the last station explicitly, or the channel is truncated by up to
        # one decimation step and the goal can sit outside its own corridor.
        keep = np.unique(np.append(np.arange(0, len(left), step), len(left) - 1))
        return ([tuple(p) for p in left[keep]]
                + [tuple(p) for p in right[keep][::-1]])

    def tangents(self) -> np.ndarray:
        """Unit tangents at every station, (N, 2) -- vectorised.

        The per-index `tangent()` above is the readable form and is fine for one
        lookup; calling it in a loop is not.  `edges()` and
        `reference_path_points()` both need all N, the bend-ceiling table builds
        ~1000 corridors, and the scalar version made that a multi-minute hang.
        """
        vec = np.gradient(self.centre, axis=0)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return np.divide(vec, norm, out=np.tile([0.0, 1.0], (len(vec), 1)),
                         where=norm > 1e-9)

    def normals(self) -> np.ndarray:
        """Unit port-side normals at every station, (N, 2)."""
        t = self.tangents()
        return np.column_stack([-t[:, 1], t[:, 0]])

    def edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """The two channel limits as (N, 2) arrays: (port edge, starboard edge)."""
        half = (0.5 * self.width)[:, None]
        n = self.normals()
        return self.centre + n * half, self.centre - n * half

    def lateral_extent(self) -> float:
        """Width of the corridor's bounding box across the basin, metres."""
        left, right = self.edges()
        xs = np.concatenate([left[:, 0], right[:, 0]])
        return float(xs.max() - xs.min())

    def min_radius(self) -> float:
        """Smallest radius of curvature along the centreline, metres."""
        kappa = np.abs(self.curvature())
        peak = float(np.max(kappa)) if kappa.size else 0.0
        return float("inf") if peak < 1e-9 else 1.0 / peak

    def curvature(self) -> np.ndarray:
        """Signed curvature at every station, 1/m, positive to starboard."""
        n = len(self.centre)
        if n < 3:
            return np.zeros(n)
        out = np.zeros(n)
        a, b, c = self.centre[:-2], self.centre[1:-1], self.centre[2:]
        ab, bc, ac = b - a, c - b, c - a
        cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
        denom = (np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1)
                 * np.linalg.norm(ac, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:-1] = np.where(denom > 1e-12, -2.0 * cross / denom, 0.0)
        return out

    # ------------------------------------------------------------------
    def reference_path_points(self, length: float = None,
                              start_s: float = 0.0) -> np.ndarray:
        """The reference path: the centreline offset to its Rule 9(a) station.

        `offset_frac` is a fraction of the **local** half-width, so a channel
        that narrows carries the path inward with it rather than letting it
        drift into the wall -- which is what makes the offset a station-keeping
        rule rather than a fixed displacement.
        """
        want = cfg.REF_PATH_LENGTH_M if length is None else float(length)
        start = float(np.clip(start_s, 0.0, max(0.0, self.length - 1.0)))
        keep = (self.s >= start) & (self.s <= min(start + want, self.length))
        if keep.sum() < 2:
            keep = np.ones(len(self.s), dtype=bool)

        # The normals are to port, so a *negative* multiple is to starboard.
        if cfg.STRAIGHT_REFERENCE_PATH:
            # F59: a constant offset, sized on the narrowest section so the
            # path stays inside it, keeps the path straight where the width
            # varies.  A local-width offset would bend it at every taper.
            lateral = np.full_like(self.width, -self.offset_frac * 0.5 * float(np.min(self.width)))
        else:
            lateral = -self.offset_frac * 0.5 * self.width
        points = self.centre + self.normals() * lateral[:, None]
        return points[keep].astype(np.float32)

    def contains(self, x: float, y: float) -> bool:
        import boundary_raycast as br
        return br.point_in_polygon(float(x), float(y), self.polygon())

    # --- the interface basin mode shares (06 §9) -------------------------
    def clearances_at_s(self, s_query: float) -> Tuple[float, float]:
        """(starboard, port) clearance at a station: `W/2` each in a channel."""
        half = 0.5 * self.width_at_s(s_query)
        return half, half

    def half_width_on_side(self, s_query: float, e_y: float) -> float:
        """06 M-5.  In a channel `h_+ = h_- = W/2`, so this is exactly the
        width normalisation every channel-mode result used (T14)."""
        return 0.5 * self.width_at_s(s_query)

    def band(self) -> "Corridor":
        """The water confined targets keep to: the channel itself."""
        return self

    def fits_basin(self, margin: float = 0.0) -> bool:
        left, right = self.edges()
        xs = np.concatenate([left[:, 0], right[:, 0]])
        ys = np.concatenate([left[:, 1], right[:, 1]])
        w, h = self.basin
        return bool(xs.min() >= margin - 1e-9 and xs.max() <= w - margin + 1e-9
                    and ys.min() >= -1e-6 and ys.max() <= h + 1e-6)

    def describe(self) -> dict:
        return {
            "nominal_width": self.nominal_width,
            "width_ratio": self.width_ratio,
            "bend_deg": self.bend_deg,
            "bend_requested_deg": self.bend_requested_deg,
            "clamped": self.was_clamped,
            "offset_frac": self.offset_frac,
            "length": self.length,
            "min_radius": self.min_radius(),
        }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def _heading_profile(s: np.ndarray, bend_rad: float, centre_frac: float,
                     span_frac: float) -> np.ndarray:
    """Heading along the corridor, from a raised-cosine turn rate.

    The rate starts and ends at exactly zero, so curvature is continuous at both
    ends of the bend.  A bend with a step change in curvature would put a
    discontinuity into `r_path`, and `R-8` differences the vessel's yaw rate
    against it -- a jump there would read as an evasive manoeuvre the agent
    never made.
    """
    length = float(s[-1])
    span = max(span_frac * length, 1e-6)
    start = float(np.clip(centre_frac * length - 0.5 * span, 0.0, length - span))
    inside = (s >= start) & (s <= start + span)

    rate = np.zeros_like(s)
    if bend_rad != 0.0:
        phase = 2.0 * np.pi * (s[inside] - start) / span
        # Integrates to exactly `bend_rad` over the span.
        rate[inside] = (bend_rad / span) * (1.0 - np.cos(phase))
    return np.concatenate(([0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(s))))


def _width_profile(s: np.ndarray, nominal: float, ratio: float,
                   n_control: int, rng) -> np.ndarray:
    """Piecewise-linear width over 2-4 control points, honouring `W_max/W_min`."""
    if ratio <= 1.0 + 1e-9 or n_control < 2:
        return np.full_like(s, nominal)

    knots = np.linspace(0.0, float(s[-1]), int(n_control))
    values = rng.uniform(0.0, 1.0, size=int(n_control))
    # Force the extremes onto the requested ratio so it is realised, not merely
    # bounded: without this a draw of three similar values silently produces a
    # constant-width channel and the boundary branch is decorative again.
    values[int(np.argmin(values))] = 0.0
    values[int(np.argmax(values))] = 1.0

    lo = 2.0 * nominal / (1.0 + ratio)
    hi = ratio * lo
    profile = np.interp(s, knots, lo + values * (hi - lo))

    # **Smooth the corners out of the profile.**  04a §3.2 asks for
    # "piecewise-linear over 2-4 control points", and taken literally that puts
    # a kink at every knot.  The reference path is the centreline offset by a
    # fraction of the *local* half-width, so a kink in the width becomes a
    # corner in the path, a delta in its curvature, and a spike in `r_path` --
    # measured at 0.49 rad/s on a dead-straight 10 m channel, which is a 2.3 m
    # turn radius that does not exist.
    #
    # `R-8` differences the vessel's yaw rate against `r_path`, so those spikes
    # would be charged to the agent as evasive manoeuvres it never made.  A 1 m
    # Hann window is far shorter than the 2-4 m between knots, so the profile
    # keeps its shape and its extremes and loses only the corners.
    span = max(3, int(round(1.0 / STATION_SPACING_M)) | 1)
    window = np.hanning(span)
    window /= window.sum()
    padded = np.pad(profile, span // 2, mode="edge")
    return np.convolve(padded, window, mode="valid")[:len(s)]


def build(nominal_width: float, *, bend_deg: float = 0.0,
          width_ratio: float = 1.0, offset_frac: float = 0.0,
          n_control: int = 3, bend_centre_frac: float = 0.5,
          bend_span_frac: float = cfg.CORRIDOR_BEND_SPAN_FRAC,
          length: float = None, basin: Tuple[float, float] = None,
          rng=None) -> Corridor:
    """Build one corridor from explicit parameters, centred in the basin."""
    rng = np.random.default_rng() if rng is None else rng
    basin = (cfg.MAP_WIDTH, cfg.MAP_HEIGHT) if basin is None else basin
    length = cfg.CORRIDOR_LENGTH_M if length is None else float(length)

    n = max(8, int(round(length / STATION_SPACING_M)) + 1)
    s = np.linspace(0.0, length, n)

    heading = _heading_profile(s, math.radians(bend_deg), bend_centre_frac,
                               bend_span_frac)
    # Heading 0 is +y (north), clockwise positive -- the project convention.
    step = np.diff(s, prepend=s[0])
    dx = np.sin(heading) * step
    dy = np.cos(heading) * step
    centre = np.column_stack([np.cumsum(dx), np.cumsum(dy)])

    width = _width_profile(s, float(nominal_width), float(width_ratio),
                           int(n_control), rng)

    corridor = Corridor(centre=centre, width=width, s=s, bend_deg=float(bend_deg),
                        bend_requested_deg=float(bend_deg),
                        offset_frac=float(offset_frac), basin=tuple(basin))
    _centre_in_basin(corridor)
    return corridor


def _centre_in_basin(corridor: Corridor) -> None:
    """Rotate the corridor's chord onto the basin's long axis, then centre it.

    **Rotation is what makes bends fit at all**, and it is worth more than a
    factor of three.  Laying the corridor out with its *start tangent* pointing
    up the basin turns a bend into a dogleg whose far end swings wide: a 25 deg
    turn halfway along a 25 m corridor puts the endpoint 5.3 m off axis, so only
    a 4 m channel fits and 04a's 40% bend requirement is unreachable.  Aligning
    the **chord** instead leaves only the sagitta, about `L*dpsi/8` = 1.4 m for
    the same turn.

    It is also the natural layout rather than a trick: a channel through a basin
    runs along the basin, and a surveyor laying one out would put its ends on the
    long axis, not its entry heading.
    """
    chord = corridor.centre[-1] - corridor.centre[0]
    norm = float(np.linalg.norm(chord))
    if norm > 1e-9:
        # Rotate the chord onto +y.
        cos_t, sin_t = chord[1] / norm, chord[0] / norm
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        corridor.centre = corridor.centre @ rotation.T

    left, right = corridor.edges()
    xs = np.concatenate([left[:, 0], right[:, 0]])
    ys = np.concatenate([left[:, 1], right[:, 1]])
    basin_w, basin_h = corridor.basin
    dx = 0.5 * basin_w - 0.5 * (xs.min() + xs.max())
    dy = 0.5 * basin_h - 0.5 * (ys.min() + ys.max())
    corridor.centre = corridor.centre + np.array([dx, dy])


# Width grid for the bend-ceiling table.  0.25 m is the basin margin itself, so
# resolving finer than that would be resolving noise.
_CEILING_GRID_M = 0.25


@lru_cache(maxsize=32)
def _ceiling_table(length: float, basin_width: float, margin: float):
    """Tabulate the bend ceiling once per (length, basin, margin).

    The ceiling is smooth and monotone in width, so a table plus linear
    interpolation is both faster and more honest than caching a bisection on a
    continuous key: widths are drawn from a uniform distribution, so a
    centimetre-quantised cache misses on nearly every call and the generator
    pays 24 corridor builds per attempt.  A 400-scenario batch went from minutes
    to under a second on this change alone.
    """
    widths = np.arange(0.5, basin_width + _CEILING_GRID_M, _CEILING_GRID_M)
    ceilings = np.array([_solve_max_bend(float(w), length, basin_width, margin)
                         for w in widths])
    return widths, ceilings


def max_bend_deg(nominal_width: float, *, length: float = None,
                 basin_width: float = None,
                 margin: float = cfg.CORRIDOR_BASIN_MARGIN) -> float:
    """Largest total heading change that still fits the basin (F25).

    Interpolated from `_ceiling_table`, and rounded **down** to the grid so the
    answer is never optimistic -- a corridor that overhangs the basin by a
    centimetre is still a corridor that cannot be sailed.
    """
    length = cfg.CORRIDOR_LENGTH_M if length is None else float(length)
    basin_width = cfg.MAP_WIDTH if basin_width is None else float(basin_width)
    widths, ceilings = _ceiling_table(float(length), float(basin_width),
                                      float(margin))
    index = int(np.searchsorted(widths, float(nominal_width)))
    index = int(np.clip(index, 0, len(ceilings) - 1))
    return float(ceilings[index])


def _solve_max_bend(nominal_width: float, length: float, basin_width: float,
                    margin: float) -> float:
    """Largest total heading change that still fits the basin (F25).

    Solved numerically rather than from the `L·Δψ/8` estimate, because the
    lateral excursion depends on where the bend sits and how wide the span is,
    and the estimate is only good to about 20% -- which at the wide end is the
    difference between fitting and not.
    """
    if nominal_width >= basin_width - 2.0 * margin:
        return 0.0

    lo, hi = 0.0, float(cfg.CORRIDOR_BEND_RANGE_DEG[1])
    if _bend_fits(hi, nominal_width, length, basin_width, margin):
        return hi
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if _bend_fits(mid, nominal_width, length, basin_width, margin):
            lo = mid
        else:
            hi = mid
    return lo


def _bend_fits(bend_deg: float, nominal_width: float, length: float,
               basin_width: float, margin: float) -> bool:
    trial = build(nominal_width, bend_deg=bend_deg, length=length,
                  basin=(basin_width, cfg.MAP_HEIGHT),
                  rng=np.random.default_rng(0))
    return bool(trial.lateral_extent() <= basin_width - 2.0 * margin)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample(rng, *, width_range: Tuple[float, float] = None,
           vary_width: bool = True, allow_bend: bool = True,
           force_bend: Optional[bool] = None,
           length: float = None,
           basin: Tuple[float, float] = None) -> Corridor:
    """Sample one corridor from the 04a §3.2 distribution.

    `force_bend` asks for a bend of at least `CORRIDOR_BEND_MIN_DEG` if the
    width admits one; the caller uses it to hit 04a's ">= 40% of episodes"
    requirement without rejection-sampling for it.  When the width does not
    admit one, the corridor comes back straight and `was_clamped` records it --
    a silent failure here would leave `r_path` inert and nobody would know.
    """
    width_range = cfg.CORRIDOR_WIDTH_RANGE if width_range is None else width_range
    basin = (cfg.MAP_WIDTH, cfg.MAP_HEIGHT) if basin is None else basin
    length = cfg.CORRIDOR_LENGTH_M if length is None else float(length)

    nominal = float(rng.uniform(*width_range))
    ratio = (float(rng.uniform(*cfg.CORRIDOR_WIDTH_VARIATION))
             if vary_width else 1.0)
    n_control = int(rng.integers(cfg.CORRIDOR_WIDTH_CONTROL_POINTS[0],
                                 cfg.CORRIDOR_WIDTH_CONTROL_POINTS[1] + 1))

    # The width profile's own maximum, not the nominal, is what has to fit.
    peak_width = nominal * (2.0 * ratio / (1.0 + ratio))
    ceiling = max_bend_deg(peak_width, length=length, basin_width=basin[0])

    requested = 0.0
    if allow_bend and cfg.CORRIDOR_BENDS:
        if force_bend:
            requested = float(rng.uniform(cfg.CORRIDOR_BEND_MIN_DEG,
                                          cfg.CORRIDOR_BEND_RANGE_DEG[1]))
        else:
            requested = float(rng.uniform(*cfg.CORRIDOR_BEND_RANGE_DEG))
        requested *= float(rng.choice([-1.0, 1.0]))

    bend = float(np.clip(requested, -ceiling, ceiling))

    # The reference-path offset carries a **positive mean** -- Rule 9(a) station.
    # A zero-mean offset would train a vessel that keeps the centreline, which
    # is the behaviour the whole precedence argument is about not doing.
    spread = 0.5 * (cfg.PATH_OFFSET_FRAC_RANGE[1] - cfg.PATH_OFFSET_FRAC_RANGE[0])
    offset = float(np.clip(rng.normal(cfg.PATH_OFFSET_MEAN_FRAC, 0.5 * spread),
                           *cfg.PATH_OFFSET_FRAC_RANGE))

    corridor = build(nominal, bend_deg=bend, width_ratio=ratio,
                     offset_frac=offset, n_control=n_control,
                     bend_centre_frac=float(rng.uniform(*cfg.CORRIDOR_BEND_CENTRE_FRAC)),
                     length=length, basin=basin, rng=rng)
    corridor.bend_requested_deg = requested

    # Offsetting a curve by half its width self-intersects once the radius of
    # curvature drops below the half-width.  Straighten rather than emit a
    # bow-tie; the clamp above makes this rare, and it is recorded either way.
    if corridor.min_radius() < 0.6 * np.max(corridor.width):
        corridor = build(nominal, bend_deg=0.0, width_ratio=ratio,
                         offset_frac=offset, n_control=n_control,
                         length=length, basin=basin, rng=rng)
        corridor.bend_requested_deg = requested
    return corridor


def rectangle(width: float, *, length: float = None,
              basin: Tuple[float, float] = None) -> Corridor:
    """A straight constant-width corridor -- Paper 2's channel, for comparison.

    Kept so the Study 1 sweep and the Tier A named cases can specify an exact
    width, and so tests that predate the generator still have their geometry.
    """
    return build(float(width), bend_deg=0.0, width_ratio=1.0, offset_frac=0.0,
                 length=length, basin=basin, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Basin mode  (06)
# ---------------------------------------------------------------------------
def nav_polygon(basin: Tuple[float, float] = None, inset: float = None) -> list:
    """`P_nav`: the basin envelope inset by `d_safe + 0.05` (06 §3.1)."""
    w, h = (cfg.MAP_WIDTH, cfg.MAP_HEIGHT) if basin is None else basin
    i = cfg.BASIN_NAV_INSET_M if inset is None else float(inset)
    return [(i, i), (w - i, i), (w - i, h - i), (i, h - i)]


def _ray_to_rectangle(points: np.ndarray, direction: np.ndarray, rect) -> np.ndarray:
    """Distance from each point along `direction` to the rectangle's boundary.

    The points lie inside; the exit is the smallest positive slab crossing.
    """
    (x0, y0), _, (x1, y1), _ = rect
    out = np.full(len(points), np.inf)
    for axis, lo, hi in ((0, x0, x1), (1, y0, y1)):
        d = float(direction[axis])
        if abs(d) < 1e-12:
            continue
        bound = hi if d > 0 else lo
        out = np.minimum(out, (bound - points[:, axis]) / d)
    return np.maximum(out, 0.0)


def _clip_convex(subject: list, clip: list) -> list:
    """Sutherland-Hodgman: `subject` clipped to the convex, CCW `clip`."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    def cut(p, q, a, b):
        dx, dy = q[0] - p[0], q[1] - p[1]
        ex, ey = b[0] - a[0], b[1] - a[1]
        denom = dx * ey - dy * ex
        t = ((a[0] - p[0]) * ey - (a[1] - p[1]) * ex) / denom
        return (p[0] + t * dx, p[1] + t * dy)

    out = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        src, out = out, []
        for j in range(len(src)):
            p, q = src[j], src[(j + 1) % len(src)]
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(cut(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(cut(p, q, a, b))
        if not out:
            break
    return [(float(x), float(y)) for x, y in out]


class _Band(Corridor):
    """The path band confined targets keep to in basin mode (06 §3.5).

    A constant-width strip along the leg's line, clipped to `P_nav`, so a
    head-on, overtaking, being-overtaken or null target behaves as channel
    traffic on the fairway the leg defines.  Its polygon is the clipped strip,
    not an offset of the centreline.
    """

    _clipped: list = None

    def polygon(self, spacing: float = POLYGON_SPACING_M) -> list:
        return list(self._clipped)


@dataclass
class Basin(Corridor):
    """A straight survey leg inside the whole basin (06 §3).

    `centre` is the leg itself, start to goal; `width` holds `W_eff(s) =
    h_+(s) + h_-(s)`, the clear width across the leg, so every consumer that
    reads a width gets the basin's honest one.  The navigable polygon is
    `P_nav`, not an offset of the centreline: the walls stay where the basin
    puts them and the leg runs obliquely between them.
    """

    mode = "basin"
    h_plus: np.ndarray = None           # starboard clearance to P_nav, per station
    h_minus: np.ndarray = None          # port clearance to P_nav, per station
    slant_requested_deg: float = 0.0
    slant_realised_deg: float = 0.0
    nav: list = None

    def polygon(self, spacing: float = POLYGON_SPACING_M) -> list:
        return list(self.nav)

    def edges(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.normals()
        return (self.centre + n * self.h_minus[:, None],
                self.centre - n * self.h_plus[:, None])

    def reference_path_points(self, length: float = None,
                              start_s: float = 0.0) -> np.ndarray:
        """The leg from `start_s` to the goal; there is no Rule 9(a) offset to
        apply, because the leg is already off-centre by its own endpoints."""
        start = float(np.clip(start_s, 0.0, max(0.0, self.length - 1.0)))
        # Paper 2's 0.2 m vertex spacing, not the 5 cm stations: on a slanted
        # line, float32 vertices 5 cm apart leave up to 3e-4 1/m of Menger
        # curvature -- above `CURVATURE_EPS`, so `r_path` would read a bend on a
        # straight leg.  At 0.2 m the residue is under the deadband.
        idx = np.flatnonzero(self.s >= start - 1e-9)
        step = max(1, int(round(0.2 / STATION_SPACING_M)))
        keep = np.unique(np.append(idx[::step], idx[-1]))
        return self.centre[keep].astype(np.float32)

    def clearances_at_s(self, s_query: float) -> Tuple[float, float]:
        s = float(s_query)
        return (float(np.interp(s, self.s, self.h_plus)),
                float(np.interp(s, self.s, self.h_minus)))

    def half_width_on_side(self, s_query: float, e_y: float) -> float:
        """06 M-5: the clearance on the side of the deviation, clipped."""
        h_plus, h_minus = self.clearances_at_s(s_query)
        h = h_plus if float(e_y) > 0.0 else h_minus
        return float(np.clip(h, *cfg.BASIN_H_SIDE_CLIP))

    def band(self) -> Corridor:
        cached = getattr(self, "_band", None)
        if cached is not None:
            return cached
        half = float(np.min(np.minimum(self.h_plus, self.h_minus)))
        t = self.tangent(0)
        n = np.array([-t[1], t[0]])
        # The leg's line across the whole basin, so a target ahead of the goal
        # or astern of the start is still on the fairway.
        (x0, y0), _, (x1, y1), _ = self.nav
        reach = math.hypot(x1 - x0, y1 - y0)
        a = self.centre[0] - t * reach
        b = self.centre[-1] + t * reach
        strip = [tuple(a - n * half), tuple(b - n * half), tuple(b + n * half), tuple(a + n * half)]
        clipped = _clip_convex(strip, self.nav)
        length = float(np.linalg.norm(b - a))
        n_st = max(8, int(round(length / STATION_SPACING_M)) + 1)
        s = np.linspace(0.0, length, n_st)
        centre = a[None, :] + s[:, None] * t[None, :]
        band = _Band(centre=centre, width=np.full(n_st, 2.0 * half), s=s, basin=self.basin)
        band._clipped = clipped
        self._band = band
        return band

    def fits_basin(self, margin: float = 0.0) -> bool:
        return True

    def describe(self) -> dict:
        out = super().describe()
        out.update({
            "geometry_mode": "basin",
            "slant_requested_deg": self.slant_requested_deg,
            "slant_realised_deg": self.slant_realised_deg,
            "path_start": [float(v) for v in self.centre[0]],
            "path_goal": [float(v) for v in self.centre[-1]],
            "path_midpoint": [float(v) for v in 0.5 * (self.centre[0] + self.centre[-1])],
            "clearance_profile": self.clearance_profile(),
        })
        return out

    def clearance_profile(self) -> dict:
        mid = 0.5 * self.length
        ends = (0.0, mid, self.length)
        return {"h_plus": [round(self.clearances_at_s(s)[0], 3) for s in ends],
                "h_minus": [round(self.clearances_at_s(s)[1], 3) for s in ends]}


def build_basin(start: Tuple[float, float], goal: Tuple[float, float], *,
                slant_requested_deg: float = None,
                basin: Tuple[float, float] = None) -> Basin:
    """A basin leg from explicit endpoints."""
    basin = (cfg.MAP_WIDTH, cfg.MAP_HEIGHT) if basin is None else tuple(basin)
    p0, p1 = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    n = max(8, int(round(length / STATION_SPACING_M)) + 1)
    s = np.linspace(0.0, length, n)
    t = (p1 - p0) / max(length, 1e-9)
    centre = p0[None, :] + s[:, None] * t[None, :]
    nav = nav_polygon(basin)
    port = np.array([-t[1], t[0]])
    h_minus = _ray_to_rectangle(centre, port, nav)
    h_plus = _ray_to_rectangle(centre, -port, nav)
    slant = math.degrees(math.atan2(float(t[0]), float(t[1])))
    return Basin(centre=centre, width=h_plus + h_minus, s=s, basin=basin,
                 h_plus=h_plus, h_minus=h_minus,
                 slant_requested_deg=float(slant if slant_requested_deg is None
                                           else slant_requested_deg),
                 slant_realised_deg=float(slant), nav=nav)


def sample_basin(rng, *, slant_max_deg: Optional[float] = None,
                 basin: Tuple[float, float] = None) -> Basin:
    """Paper 2's layout: fixed start and goal y, both x drawn independently.

    With a stage slant cap (06 §4, stage 1), the goal x is pulled toward the
    start x until the cap holds, and the requested and realised slants are both
    recorded -- the F25 clamp-and-record pattern.
    """
    y0, y1 = float(cfg.BASIN_START_Y), float(cfg.BASIN_GOAL_Y)
    x0 = float(rng.uniform(*cfg.BASIN_X_RANGE))
    x1 = float(rng.uniform(*cfg.BASIN_X_RANGE))
    requested = math.degrees(math.atan2(x1 - x0, y1 - y0))
    if slant_max_deg is not None and abs(requested) > float(slant_max_deg):
        x1 = x0 + math.copysign(math.tan(math.radians(float(slant_max_deg))) * (y1 - y0),
                                x1 - x0)
    return build_basin((x0, y0), (x1, y1), slant_requested_deg=requested, basin=basin)


# ---------------------------------------------------------------------------
# The out-of-corridor world  (03a §1.2)
# ---------------------------------------------------------------------------
def facility_walls(basin: Tuple[float, float] = None,
                   margin: float = None) -> list:
    """The physical room the vessel sits in: basin envelope plus a margin.

    **Returned in the raw scan and then gated**, which is the whole point.  Until
    03a §1.2 the simulated sensor saw only panels and the target, so the boundary
    gate had nothing to remove and passed everything through -- a no-op in
    simulation and load-bearing in the field, in the one component 01 §3 exists
    to remove a sim-to-real gap from.

    With walls present the gate does real work in training, its margin becomes a
    tunable with measurable failure modes in *both* directions (gating out real
    obstacles, or letting phantoms through), and the N1 claim covers the whole
    perception stack rather than the part after the gate.
    """
    basin = (cfg.MAP_WIDTH, cfg.MAP_HEIGHT) if basin is None else basin
    margin = cfg.FACILITY_WALL_MARGIN if margin is None else float(margin)
    w, h = basin
    return [(-margin, -margin), (w + margin, -margin),
            (w + margin, h + margin), (-margin, h + margin)]
