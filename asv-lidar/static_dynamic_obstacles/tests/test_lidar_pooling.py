"""Sector pooling: allocation, coverage, and Algorithm 1 behaviour.

Kickoff §7: "beam counts per sector, total = 27, no gaps or overlaps across
+/-135 deg".
"""

import numpy as np
import pytest

import constants as cfg
import lidar_pooling as lp


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
def test_sector_count():
    assert cfg.LIDAR_SECTORS == 27
    assert len(cfg.sector_edges()) == 28


def test_bands_sum_to_declared_sector_count():
    """15 bow + 8 mid + 4 outer = 27 (01 §2.2)."""
    counts = [int(round((hi - lo) / w)) for lo, hi, w in cfg.POOL_BANDS]
    assert counts == [2, 4, 15, 4, 2]
    assert sum(counts) == 27
    # Grouped the way 01 §2.2 tabulates them: bow / abeam / quarter.
    assert counts[2] == 15
    assert counts[1] + counts[3] == 8
    assert counts[0] + counts[4] == 4


def test_swath_is_exactly_plus_minus_135():
    edges = cfg.sector_edges()
    assert edges[0] == pytest.approx(-135.0)
    assert edges[-1] == pytest.approx(135.0)
    assert float(np.sum(lp.sector_spans())) == pytest.approx(270.0)


def test_no_gaps_or_overlaps():
    """Sector edges must be contiguous and strictly increasing."""
    edges = cfg.sector_edges()
    assert np.all(np.diff(edges) > 0.0)
    # Contiguity: each sector's upper edge is the next one's lower edge.  This
    # holds by construction from a single edge array, which is the point.
    spans = lp.sector_spans()
    assert np.sum(spans) == pytest.approx(edges[-1] - edges[0])


def test_sector_widths_match_spec():
    spans = lp.sector_spans()
    assert spans[:2] == pytest.approx([22.5, 22.5])          # port outer
    assert spans[2:6] == pytest.approx([11.25] * 4)          # port mid
    assert spans[6:21] == pytest.approx([6.0] * 15)          # bow
    assert spans[21:25] == pytest.approx([11.25] * 4)        # stbd mid
    assert spans[25:] == pytest.approx([22.5, 22.5])         # stbd outer


# ---------------------------------------------------------------------------
# Beam assignment
# ---------------------------------------------------------------------------
def test_beam_resolution_is_half_a_degree():
    assert cfg.LIDAR_BEAM_RES_DEG == pytest.approx(0.5)
    assert cfg.LIDAR_BEAMS == 720


def test_beams_per_sector():
    """01 §2.2: 12 beams in a 6 deg sector, 22-23 in 11.25 deg, 45 in 22.5 deg."""
    labels = lp.sector_assignment()
    counts = np.array([int(np.sum(labels == s)) for s in range(cfg.LIDAR_SECTORS)])

    assert list(counts[:2]) == [45, 45]                      # port outer
    assert list(counts[6:21]) == [12] * 15                   # bow
    assert list(counts[25:]) == [45, 45]                     # stbd outer

    # 11.25 deg / 0.5 deg = 22.5 beams, so these sectors MUST alternate 22/23.
    # 01 §2.2 writes "22-23"; the allocation cannot be constant here.
    mid = np.concatenate([counts[2:6], counts[21:25]])
    assert set(mid.tolist()) == {22, 23}
    assert int(mid.sum()) == 180                             # 90 deg at 0.5 deg


def test_every_beam_lands_in_at_most_one_sector():
    labels = lp.sector_assignment()
    assert labels.shape == (cfg.LIDAR_BEAMS,)
    assert labels.min() >= -1
    assert labels.max() == cfg.LIDAR_SECTORS - 1


def test_pooled_swath_covers_540_beams_and_aft_is_reserved():
    """270 deg of 360 deg is pooled; the aft 90 deg goes to the tracker."""
    labels = lp.sector_assignment()
    inside = int(np.sum(labels >= 0))
    assert inside == 540                                     # 270 deg / 0.5 deg
    assert cfg.LIDAR_BEAMS - inside == 180                   # aft 90 deg

    # Everything dropped really is astern.
    dropped = lp.beam_bearings()[labels < 0]
    assert np.all(np.abs(dropped) >= cfg.POOL_SWATH_HALF_DEG)


def test_bow_sector_is_centred_on_zero():
    """An odd bow sector count means one sector straddles dead ahead."""
    centres = lp.sector_centres()
    assert float(centres[13]) == pytest.approx(0.0)
    assert np.all(np.diff(centres) > 0.0)


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------
def test_clear_scan_pools_to_max_range_and_zero_closeness():
    raw = np.full(cfg.LIDAR_BEAMS, cfg.LIDAR_RANGE)
    ranges, close = lp.pool_to_sectors(raw, safe_width_m=0.8)
    assert ranges.shape == (27,)
    assert close.shape == (27,)
    assert np.allclose(ranges, cfg.LIDAR_RANGE)
    assert np.allclose(close, 0.0)


def test_closeness_normalisation_endpoints():
    assert lp.closeness_from_ranges([0.0])[0] == pytest.approx(1.0)
    assert lp.closeness_from_ranges([cfg.LIDAR_RANGE])[0] == pytest.approx(0.0)
    half = lp.closeness_from_ranges([0.5 * cfg.LIDAR_RANGE])[0]
    assert half == pytest.approx(0.5)


def test_wall_across_the_bow_is_seen_by_bow_sectors_only():
    raw = np.full(cfg.LIDAR_BEAMS, cfg.LIDAR_RANGE)
    bearings = lp.beam_bearings()
    raw[np.abs(bearings) <= 3.0] = 2.0                       # 6 deg wide, dead ahead
    ranges, close = lp.pool_to_sectors(raw, safe_width_m=0.8)

    assert ranges[13] < cfg.LIDAR_RANGE                      # centre bow sector
    assert close[13] > 0.0
    assert np.allclose(ranges[:6], cfg.LIDAR_RANGE)          # abeam untouched
    assert np.allclose(ranges[21:], cfg.LIDAR_RANGE)


def test_gap_narrower_than_the_vessel_is_reported_as_blocked():
    """The whole point of feasibility pooling over a plain minimum.

    A sector holding a narrow slot should pool to the near wall, not to the
    range seen through the slot.
    """
    near, far = 3.0, 12.0
    n = 12                                                   # one bow sector
    theta = np.radians(cfg.LIDAR_BEAM_RES_DEG)

    # One beam of open gap at 3 m spans theta*3 = 0.026 m -- far under any
    # plausible vessel width, so the slot is not drivable.
    sector = np.full(n, near)
    sector[n // 2] = far
    pooled = lp.feasibility_pool(sector, safe_width_m=0.8, neighbour_angle_rad=theta)
    assert pooled == pytest.approx(near)


def test_opening_wider_than_the_vessel_is_reported_as_open():
    near, far = 3.0, 12.0
    theta = np.radians(cfg.LIDAR_BEAM_RES_DEG)
    # 45 beams of gap at 12 m spans 45 * theta * 12 = 4.7 m: comfortably drivable.
    sector = np.full(45, far)
    sector[0] = near
    pooled = lp.feasibility_pool(sector, safe_width_m=0.8, neighbour_angle_rad=theta)
    assert pooled > near


def test_pooling_never_exceeds_the_true_minimum_by_construction():
    """Feasibility pooling is conservative: it can only pull a sector nearer."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        raw = rng.uniform(0.5, cfg.LIDAR_RANGE, size=cfg.LIDAR_BEAMS)
        ranges, _ = lp.pool_to_sectors(raw, safe_width_m=0.8)
        labels = lp.sector_assignment()
        for s in range(cfg.LIDAR_SECTORS):
            beams = raw[labels == s]
            assert ranges[s] <= beams.max() + 1e-6
            assert ranges[s] >= beams.min() - 1e-6


def test_pooling_is_deterministic():
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.5, cfg.LIDAR_RANGE, size=cfg.LIDAR_BEAMS)
    a, _ = lp.pool_to_sectors(raw, safe_width_m=0.8)
    b, _ = lp.pool_to_sectors(raw, safe_width_m=0.8)
    assert np.array_equal(a, b)
