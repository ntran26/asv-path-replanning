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
    b_hi = len(ste.tier_b_cells()) * ste.TIER_B_SEEDS_PER_CELL
    assert ste.TIER_A_SEED_BASE >= b_hi
    top = ste.TIER_A_SEED_BASE + len(ste.tier_a()) * (ste.TIER_A_SEED_RETRIES + 1)
    lo, hi = cfg.SEED_NAMESPACES["frozen_eval"]
    assert top <= hi - lo + 1


def test_suite_34_holds_only_what_training_draws():
    """Suite 3.4 (your call, 2026-09-27): the headline is the development set's
    kind of scenario, drawn in the frozen namespace -- constant-velocity targets,
    and class x geometry only where training draws it (every class in the basin,
    `CHANNEL_CLASSES` in a channel), channels 7.5-10 m, 8 balanced cells of 100.
    Pinned because the counts feed every headline table."""
    import suite as ste
    assert ste.SUITE_REVISION == "3.4" and ste.TIER_B_EPISODES == 100
    assert ste.width_strata() == {"channel": (7.5, 10.0)}
    cells = ste.tier_b_cells()
    assert len(cells) == 8 and sum(c["episodes"] for c in cells) == 800
    assert {c["behaviour"] for c in cells} == {"cv"}
    assert {c["class"] for c in cells if c["stratum"] == "basin"} == {
        "head_on", "crossing", "overtaking", "being_overtaken", "null"}
    assert {c["class"] for c in cells if c["stratum"] == "channel"} == set(cfg.CHANNEL_CLASSES)


def test_robustness_variants_change_only_the_target_model():
    """The robustness set reruns headline scenarios: reactive for every encounter
    class, non-compliant only in head-on (T-NC1 would duplicate the headline)."""
    import suite as ste
    import targets as tgt
    built, short = ste.build_tier_b(cells=[0, 3, 4])      # basin head-on, being overtaken, null
    assert not short
    variants = ste.robustness_variants(built)
    kinds = {(v.encounter_class, b) for v, _, b in variants}
    assert kinds == {("head_on", "re"), ("head_on", "nc"), ("being_overtaken", "re")}
    for v, twin, b in variants:
        base = built[twin]
        assert v.target_behaviour == ste.target_model(b, base.encounter_class)
        assert base.target_behaviour == tgt.T_CV
        assert v.target_spawn == base.target_spawn and v.target_heading == base.target_heading
        assert v.case_id == f"{base.case_id}-{b.upper()}"
        assert ste.test_id(v.case_id) == ste.test_id(base.case_id).replace("-CV-", f"-{b.upper()}-")


def test_every_tier_b_test_has_a_unique_id_that_resolves_back():
    """Test IDs (2026-09-26) name a scenario for `run_test.py`.  All 800 must be
    distinct, and the ID, the case id and the index must all resolve to the
    same (cell, episode), or a replay would run a different test from the one
    asked for."""
    import suite as ste
    cells = ste.tier_b_cells()
    ids, index = set(), 0
    for c_index, cell in enumerate(cells):
        for n in range(cell["episodes"]):
            case_id = f"B-{c_index:02d}-{n:03d}"
            tid = ste.test_id(case_id)
            assert tid not in ids
            ids.add(tid)
            assert ste.resolve_test(tid) == (c_index, n, "cv")
            assert ste.resolve_test(case_id) == (c_index, n, "cv")
            assert ste.resolve_test(str(index)) == (c_index, n, "cv")
            assert ste.tier_b_index(c_index, n) == index
            index += 1
    assert len(ids) == 800
    assert ste.test_id("B-00-000") == "BAS-HO-CV-001"
    assert ste.resolve_test("BAS-HO-NC-001") == (0, 0, "nc")
    assert ste.resolve_test("CH-CR-RE-007") == (6, 6, "re")
    assert ste.resolve_test("B-06-006-RE") == (6, 6, "re")
    for bad in ("BAS-HO-CV-101", "800", "BAS-CR-NC-001", "BAS-NU-RE-001"):
        with pytest.raises(ValueError):
            ste.resolve_test(bad)
