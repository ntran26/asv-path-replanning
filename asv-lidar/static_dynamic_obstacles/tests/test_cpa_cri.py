"""CPA geometry, ship domain, and the risk index.

Kickoff §7: "head-on gives TCPA > 0 and DCPA ~ 0; already-passed geometry gives
TCPA < 0; near-parallel courses give a large |TCPA| and must be caught by the
Euclidean-distance risk term rather than by CPA".

Frame convention throughout: +y is north, +x is east, headings are compass
(0 = +y, clockwise positive).
"""

import numpy as np
import pytest

import constants as cfg
import cpa_cri as cc


# ---------------------------------------------------------------------------
# CPA
# ---------------------------------------------------------------------------
def test_head_on_gives_positive_tcpa_and_near_zero_dcpa():
    """Two vessels closing bow to bow on the same line."""
    p_os, v_os = (5.0, 0.0), (0.0, 0.5)          # heading north at 0.5 m/s
    p_ts, v_ts = (5.0, 20.0), (0.0, -0.5)        # heading south at 0.5 m/s

    dcpa, tcpa = cc.cpa(p_os, v_os, p_ts, v_ts)
    assert dcpa == pytest.approx(0.0, abs=1e-9)
    assert tcpa == pytest.approx(20.0)           # 20 m closing at 1.0 m/s


def test_already_passed_gives_negative_tcpa():
    """Same geometry, but the target is now astern and opening."""
    p_os, v_os = (5.0, 20.0), (0.0, 0.5)
    p_ts, v_ts = (5.0, 5.0), (0.0, -0.5)

    dcpa, tcpa = cc.cpa(p_os, v_os, p_ts, v_ts)
    assert tcpa < 0.0
    assert dcpa == pytest.approx(0.0, abs=1e-9)


def test_offset_head_on_gives_the_lateral_offset_as_dcpa():
    p_os, v_os = (4.0, 0.0), (0.0, 0.5)
    p_ts, v_ts = (6.0, 20.0), (0.0, -0.5)
    dcpa, tcpa = cc.cpa(p_os, v_os, p_ts, v_ts)
    assert dcpa == pytest.approx(2.0)
    assert tcpa == pytest.approx(20.0)


def test_zero_relative_velocity_freezes_the_geometry():
    """Two vessels on identical courses at identical speed never converge."""
    p_os, v_os = (4.0, 0.0), (0.0, 0.5)
    p_ts, v_ts = (6.0, 10.0), (0.0, 0.5)
    dcpa, tcpa = cc.cpa(p_os, v_os, p_ts, v_ts)
    assert tcpa == pytest.approx(0.0)
    assert dcpa == pytest.approx(float(np.hypot(2.0, 10.0)))


def test_crossing_geometry():
    """OS north, TS west-to-east; they meet at (5, 10)."""
    p_os, v_os = (5.0, 0.0), (0.0, 1.0)
    p_ts, v_ts = (-5.0, 10.0), (1.0, 0.0)
    dcpa, tcpa = cc.cpa(p_os, v_os, p_ts, v_ts)
    assert tcpa == pytest.approx(10.0)
    assert dcpa == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Bearings
# ---------------------------------------------------------------------------
def test_relative_bearing_cardinals():
    p_os = (0.0, 0.0)
    assert cc.relative_bearing_deg(p_os, 0.0, (0.0, 5.0)) == pytest.approx(0.0)     # ahead
    assert cc.relative_bearing_deg(p_os, 0.0, (5.0, 0.0)) == pytest.approx(90.0)    # stbd
    assert cc.relative_bearing_deg(p_os, 0.0, (0.0, -5.0)) == pytest.approx(180.0)  # astern
    assert cc.relative_bearing_deg(p_os, 0.0, (-5.0, 0.0)) == pytest.approx(270.0)  # port


def test_relative_bearing_follows_own_heading():
    """Turn 90 deg to starboard and a target due north is now on the port beam."""
    assert cc.relative_bearing_deg((0.0, 0.0), 90.0, (0.0, 5.0)) == pytest.approx(270.0)


def test_heading_intersection_angle():
    assert cc.heading_intersection_deg(0.0, 180.0) == pytest.approx(180.0)   # reciprocal
    assert cc.heading_intersection_deg(0.0, 0.0) == pytest.approx(0.0)       # same course
    assert cc.heading_intersection_deg(90.0, 0.0) == pytest.approx(270.0)


# ---------------------------------------------------------------------------
# Ship domain
# ---------------------------------------------------------------------------
def test_domain_is_asymmetric_fore_and_aft():
    assert cc.domain_scale(0.0) == pytest.approx(cfg.DOMAIN_FORE)
    assert cc.domain_scale(180.0) == pytest.approx(cfg.DOMAIN_AFT)
    assert cc.domain_scale(90.0) == pytest.approx(cfg.DOMAIN_LATERAL)
    assert cc.domain_scale(270.0) == pytest.approx(cfg.DOMAIN_LATERAL)
    assert cfg.DOMAIN_FORE > cfg.DOMAIN_AFT > cfg.DOMAIN_LATERAL


def test_domain_matches_the_resolved_multiples():
    """01 §5.2, resolved: 2.0 / 1.0 / 0.75 x Lpp (ahead / astern / abeam).

    Provisional -- the final values are an OUTPUT of 05, derived from the
    identified turning circle, not a scaled copy of Chun et al.
    """
    assert cfg.DOMAIN_FORE == pytest.approx(2.00 * cfg.LBP)
    assert cfg.DOMAIN_AFT == pytest.approx(1.00 * cfg.LBP)
    assert cfg.DOMAIN_LATERAL == pytest.approx(0.75 * cfg.LBP)
    assert cfg.DOMAIN_FORE == pytest.approx(3.14, abs=0.01)
    assert cfg.DOMAIN_LATERAL == pytest.approx(1.18, abs=0.01)


def test_chun_domain_would_not_fit_but_the_compressed_one_does():
    """Why the compression is necessary rather than convenient."""
    chun_lateral = 1.0 * cfg.LBP
    lateral_footprint = 2.0 * cfg.DOMAIN_LATERAL
    assert lateral_footprint == pytest.approx(2.36, abs=0.01)
    # ~24% of the widest channel, against ~31% for Chun.
    assert lateral_footprint / cfg.MAP_WIDTH < 0.25
    assert (2.0 * chun_lateral) / cfg.MAP_WIDTH > 0.30
    # Chun fore-aft is 4.7 m in a 25 m basin: nearly a fifth of the run.
    assert 3.0 * cfg.LBP == pytest.approx(4.71, abs=0.01)


def test_the_width_sweep_brackets_the_head_on_threshold():
    """03 §5: a compliant port-to-port head-on stops fitting between 4.0 and 3.5 m.

    Two non-overlapping domains abeam plus wall clearance each side.
    """
    # 2.36 m centre-to-centre separation + 0.65 m wall clearance each side.
    needed = 2.0 * cfg.DOMAIN_LATERAL + 2.0 * cfg.HEAD_ON_WALL_CLEARANCE
    assert needed == pytest.approx(3.66, abs=0.02)

    widths = sorted(cfg.CORRIDOR_WIDTHS_M)
    below = [w for w in widths if w < needed]
    above = [w for w in widths if w >= needed]
    assert below and above, "the sweep must bracket the threshold"
    assert max(below) == 3.5
    assert min(above) == 4.0


def test_distance_to_domain_is_zero_inside():
    p_os = (5.0, 10.0)
    just_inside = (5.0, 10.0 + 0.5 * cfg.DOMAIN_FORE)
    assert cc.distance_to_domain(p_os, 0.0, just_inside) == 0.0
    assert cc.inside_domain(p_os, 0.0, just_inside)


def test_distance_to_domain_outside():
    p_os = (5.0, 10.0)
    target = (5.0, 10.0 + cfg.DOMAIN_FORE + 3.0)
    assert cc.distance_to_domain(p_os, 0.0, target) == pytest.approx(3.0)
    assert not cc.inside_domain(p_os, 0.0, target)


# ---------------------------------------------------------------------------
# CRI
# ---------------------------------------------------------------------------
def test_cri_is_one_inside_the_domain():
    risk = cc.cri((5.0, 10.0), (0.0, 0.5), 0.0,
                  (5.0, 11.0), (0.0, -0.5), 180.0)
    assert risk == pytest.approx(1.0)


def test_cri_is_bounded():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p_os = rng.uniform(0, 10, 2)
        p_ts = rng.uniform(0, 10, 2)
        v_os = rng.uniform(-1, 1, 2)
        v_ts = rng.uniform(-1, 1, 2)
        r = cc.cri(p_os, v_os, rng.uniform(0, 360), p_ts, v_ts, rng.uniform(0, 360))
        assert 0.0 <= r <= 1.0
        assert np.isfinite(r)


def test_cri_falls_with_range_on_a_head_on_approach():
    risks = []
    for gap in (3.0, 6.0, 10.0, 15.0):
        risks.append(cc.cri((5.0, 0.0), (0.0, 0.5), 0.0,
                            (5.0, gap), (0.0, -0.5), 180.0))
    assert all(a > b for a, b in zip(risks, risks[1:])), risks


def test_cri_drops_quickly_once_the_cpa_is_passed():
    """The two-rate decay: risk must fall away faster astern than ahead."""
    approaching = cc.cri((5.0, 0.0), (0.0, 0.5), 0.0,
                         (5.0, 8.0), (0.0, -0.5), 180.0)
    # Mirror geometry, same 8 m separation, but the target is now astern
    # and opening.
    opening = cc.cri((5.0, 8.0), (0.0, 0.5), 0.0,
                     (5.0, 0.0), (0.0, -0.5), 180.0)
    assert opening < approaching
    assert cfg.CRI_TCPA_SCALE_AFTER < cfg.CRI_TCPA_SCALE_BEFORE


def test_near_parallel_is_caught_by_the_euclidean_term_not_by_cpa():
    """01 §5.1's known failure mode, and the reason CR_ED is not optional.

    Two vessels 2 m apart on near-parallel courses at nearly equal speed have a
    CPA far away in time, so a pure CPA risk reads almost nothing.  In a narrow
    channel this geometry is normal, not exceptional.
    """
    p_os, v_os, h_os = (4.0, 5.0), (0.0, 0.50), 0.0
    p_ts, v_ts, h_ts = (6.0, 6.0), (0.0, 0.51), 0.0

    dcpa, tcpa = cc.cpa(p_os, v_os, p_ts, v_ts)
    # CPA is genuinely useless here: hundreds of seconds out.
    assert abs(tcpa) > 60.0

    import math
    tcpa_scale = cfg.CRI_TCPA_SCALE_BEFORE if tcpa >= 0 else cfg.CRI_TCPA_SCALE_AFTER
    cr_cpa_only = math.exp(-max(0.0, dcpa) / cfg.CRI_DCPA_SCALE) \
        * math.exp(-abs(tcpa) / tcpa_scale)
    assert cr_cpa_only < 0.01                      # CPA alone says "no risk"

    risk = cc.cri(p_os, v_os, h_os, p_ts, v_ts, h_ts)
    assert risk > 0.5                              # the ED term rescues it
    assert risk > cr_cpa_only * 10


def test_bow_crossing_inflates_risk():
    """Crossing ahead of a vessel must score higher than crossing astern."""
    # TS heading north; OS sits dead ahead of it.
    ahead = cc.bow_crossing_factor((5.0, 12.0), (5.0, 5.0), 0.0)
    # OS sits dead astern of it.
    astern = cc.bow_crossing_factor((5.0, 2.0), (5.0, 5.0), 0.0)
    assert ahead > astern
    assert ahead == pytest.approx(cfg.CRI_BOW_CROSSING_GAIN)
    assert astern == pytest.approx(1.0)


def test_bow_crossing_factor_is_continuous_at_the_arc_edge():
    edge = cfg.CRI_BOW_CROSSING_HALF_DEG
    import math
    r = 5.0
    inside = cc.bow_crossing_factor(
        (5.0 + r * math.sin(math.radians(edge - 0.01)),
         5.0 + r * math.cos(math.radians(edge - 0.01))), (5.0, 5.0), 0.0)
    outside = cc.bow_crossing_factor(
        (5.0 + r * math.sin(math.radians(edge + 0.01)),
         5.0 + r * math.cos(math.radians(edge + 0.01))), (5.0, 5.0), 0.0)
    assert inside == pytest.approx(outside, abs=1e-3)
