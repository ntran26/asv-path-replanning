"""The frozen suite, revision 3.0 (06 §5, F75): named cases realise their names."""

import numpy as np
import pytest

import boundary_raycast as br
import constants as cfg
import feasibility as feas
import scenario as scn
import suite as ste
from env import ASVLidarEnv


@pytest.fixture(scope="module")
def tier_a():
    return {s.case_id: s for s in ste.build_tier_a()}


def test_tier_a_realises_all_but_the_reported_infeasible_cases(tier_a):
    shortfall = ste.tier_a_shortfall(list(tier_a.values()))
    # Being overtaken at 4 m and null at 6 and 4 m: the geometries 06 M-6
    # already calls infeasible.  Reported, not dropped silently.
    assert set(shortfall) <= {"A-BO-N", "A-NU-I", "A-NU-N"}
    assert len(tier_a) >= 35


def test_crossing_cases_cross_from_the_side_they_name(tier_a):
    for case in ste.tier_a():
        side = case.flag_dict.get("side")
        if side and case.case_id in tier_a:
            realised = "port" if tier_a[case.case_id].ct_deg < 180.0 else "starboard"
            assert realised == side, case.case_id


def test_named_speed_ratios_are_realised(tier_a):
    for case in ste.tier_a():
        if case.speed_ratio is not None and case.case_id in tier_a:
            assert tier_a[case.case_id].speed_ratio == pytest.approx(case.speed_ratio)


def test_basin_cases_run_the_fixed_14_degree_leg(tier_a):
    basin = [s for s in tier_a.values() if s.case_id.startswith("A-BSN")]
    assert len(basin) == 6
    for s in basin:
        assert s.geometry_mode == "basin" and s.field_replicable
        assert s.slant_realised_deg == pytest.approx(14.036, abs=0.01)


def test_offset_cases_set_the_rule_9a_station(tier_a):
    assert tier_a["A-OFF-CRP-I"].path_offset_frac == pytest.approx(-0.30)
    assert tier_a["A-OFF-OT-I"].path_offset_frac == pytest.approx(0.30)


@pytest.mark.parametrize("case_id", ["A-CLT-HO-I", "A-CLT-CRS-I", "A-BSN-CLT-CRS", "A-OCC-CRS-I"])
def test_flagged_panels_are_placed_and_leave_a_route(tier_a, case_id):
    built = tier_a[case_id]
    env = ASVLidarEnv(render_mode=None)
    env.forced_num_obs = 0
    env.reset(seed=1, options={"generated": built})
    assert getattr(env, "flagged_panels", 0) == 1, case_id
    assert feas.layout_feasible((env.start_x, env.start_y), (env.goal_x, env.goal_y),
                                env.boundary_polygon, env.obstacles)


def test_tier_a_and_tier_b_seeds_are_disjoint():
    b_hi = 48 * ste.TIER_B_SEEDS_PER_CELL
    assert ste.TIER_A_SEED_BASE >= b_hi
    top = ste.TIER_A_SEED_BASE + len(ste.tier_a()) * (ste.TIER_A_SEED_RETRIES + 1)
    lo, hi = cfg.SEED_NAMESPACES["frozen_eval"]
    assert top <= hi - lo + 1


def test_suite_31_floors_tier_b_channels_at_seven_and_a_half_metres():
    """Suite 3.1 (your call, 2026-09-23).  Below ~7.5 m a channel leaves a
    two-vessel encounter no room a lawful manoeuvre can use, so Tier B stops
    there and the 10 m basin carries the narrow-water case.  Pinned because the
    change moves claims C-2/C-3 onto R4, and because the counts feed every
    headline table."""
    import suite as ste
    assert ste.TIER_B_MIN_WIDTH_M == 7.5 and ste.SUITE_REVISION == "3.1"
    strata = ste.width_strata()
    assert set(strata) == {"wide", "intermediate"}
    assert min(lo for lo, _ in strata.values()) == 7.5
    assert max(hi for _, hi in strata.values()) == 10.0
    cells = ste.tier_b_cells()
    assert len(cells) == 39 and sum(c["episodes"] for c in cells) == 780
    assert {c["stratum"] for c in cells} == {"basin", "channel-wide", "channel-intermediate"}
    # Tier A is unchanged: it is the extended set, not the default.
    assert len(ste.tier_a()) == 38
