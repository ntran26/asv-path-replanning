"""The Paper 2 deployment-layout set (2026-09-27): a separate evaluation set.

Pinned: the layouts are the published field layouts (Drones 10(9), 680, Fig. 8),
the set has the composition it claims, each VAR scenario differs from its FIX
twin only in the target's speed, every target scenario is field feasible
(revision 2.0), and the fixed-layout, varying-speed and stop-at-wall hooks leave
training untouched.
"""

import numpy as np
import pytest

import constant_temp as ct
import curriculum
import paper2_set as p2
import targets as tgt
import train_formulation as tf
from env import ASVLidarEnv


@pytest.fixture(scope="module")
def env():
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    return ASVLidarEnv(render_mode=None, emergency_stop=False)


@pytest.fixture(scope="module")
def cell(env):
    records, short = p2.build(env, layouts=["L3"], encounters=["BO"], include_no_target=False)
    assert not short
    return records


def test_layouts_are_the_published_field_layouts():
    assert p2.LAYOUTS["L1"]["start"] == (5.0, 2.0) and p2.LAYOUTS["L1"]["goal"] == (5.0, 22.0)
    assert p2.LAYOUTS["L2"]["start"] == (3.0, 2.0) and p2.LAYOUTS["L2"]["goal"] == (8.0, 22.0)
    assert p2.LAYOUTS["L3"]["start"] == (7.0, 2.0) and p2.LAYOUTS["L3"]["goal"] == (2.0, 22.0)
    assert sorted(p2.LAYOUTS["L1"]["panels"]) == [(2.0, 16.5), (5.0, 8.0), (7.2, 15.7)]
    assert sorted(p2.LAYOUTS["L2"]["panels"]) == [(2.0, 8.5), (6.0, 17.0), (7.2, 9.3)]
    assert sorted(p2.LAYOUTS["L3"]["panels"]) == [(1.5, 8.5), (5.0, 17.0), (6.5, 8.5)]
    assert p2.PANEL_SIZE_M == 1.0


def test_composition():
    assert len(p2.cells()) == 3 * 5
    assert set(p2.ENCOUNTERS) == {"HO", "CRP", "CRS", "OT", "BO"}
    total = len(p2.cells()) * p2.PER_CELL * len(p2.SPEEDS) + len(p2.LAYOUTS) * p2.NO_TARGET_EPISODES
    assert total == 630
    # generator seeds stay in the unused gap between the frozen and study1 namespaces
    top = p2.SEED_BASE + (len(p2.cells()) + 1) * p2.SEEDS_PER_CELL
    assert 309_999 < p2.SEED_BASE and top < 400_000


def test_var_twins_differ_only_in_speed(cell):
    fix = {r["test_id"]: r for r in cell if r["speed"] == "FIX"}
    var = [r for r in cell if r["speed"] == "VAR"]
    assert len(fix) == len(var) == p2.PER_CELL
    for r in var:
        f = fix[r["twin"]]
        a, b = f["built"], r["built"]
        assert a.target_behaviour == tgt.T_CV and b.target_behaviour == tgt.T_VS
        assert a.target_spawn == b.target_spawn and a.target_heading == b.target_heading
        assert a.target_speed == b.target_speed and r["episode_seed"] == f["episode_seed"]
        t_change, v_final, accel = b.flags["speed_profile"]
        ratio = v_final / a.target_speed
        assert (ct.TARGET_VS_SLOW_FACTOR[0] - 1e-9 <= ratio <= ct.TARGET_VS_SLOW_FACTOR[1] + 1e-9
                or ct.TARGET_VS_FAST_FACTOR[0] - 1e-9 <= ratio <= ct.TARGET_VS_FAST_FACTOR[1] + 1e-9)
        assert accel == ct.TARGET_VS_ACCEL_MPS2 and t_change > 0.0
        assert "speed_profile" not in a.flags


def test_targets_clear_the_panels_and_the_layout_is_fixed(env, cell):
    assert min(r["target_clearance_m"] for r in cell) >= ct.TARGET_PANEL_CLEARANCE_M
    r = cell[0]
    env.reset(seed=r["episode_seed"], options={"generated": r["built"]})
    assert sorted(env.obstacles) == sorted([list(map(tuple, p)) for p in p2.panel_polygons("L3")])
    assert (env.goal_x, env.goal_y) == pytest.approx(p2.LAYOUTS["L3"]["goal"])


def test_varying_speed_target_ramps_once_and_holds_course():
    t = tgt.Target(0.0, 0.0, 30.0, 0.6, behaviour=tgt.T_VS, speed_profile=(2.0, 0.3, 0.05))
    speeds, headings = [], []
    for _ in range(150):
        t.step(0.1, own={"x": 5.0, "y": 5.0, "velocity": np.array([0.0, 0.5]), "heading": 0.0})
        speeds.append(t.speed)
        headings.append(t.heading)
    assert speeds[18] == pytest.approx(0.6)                  # before t_change
    assert speeds[-1] == pytest.approx(0.3)                  # reached the final speed
    assert np.all(np.diff(speeds) <= 1e-12)                  # monotone ramp
    assert max(abs(np.diff(speeds))) <= 0.05 * 0.1 + 1e-12   # at the stated acceleration
    assert set(headings) == {30.0}                           # course never changes


def test_training_targets_have_no_speed_profile_or_stop_box(env):
    env.reset(seed=3)
    assert all(t.speed_profile is None and t.stop_box is None for t in env.targets)


def test_test_ids_resolve():
    assert p2.resolve_test("P2-L2-CRP-VAR-07") == {"layout": "L2", "encounter": "CRP", "speed": "VAR", "n": 7}
    assert p2.resolve_test("p2-l1-nt-03") == {"layout": "L1", "encounter": "NT", "speed": "", "n": 3}
    for bad in ("P2-L4-HO-FIX-01", "P2-L1-XX-FIX-01", "P2-L1-HO-FAST-01", "P2-L1-HO-FIX-21", "P2-L1-NT-11"):
        with pytest.raises(ValueError):
            p2.resolve_test(bad)


def test_polygon_distance():
    a = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert p2.polygon_distance(a, [(2, 0), (3, 0), (3, 1), (2, 1)]) == pytest.approx(1.0)
    assert p2.polygon_distance(a, [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]) == 0.0


def test_every_target_scenario_is_field_feasible(env, cell):
    """Revision 2.0: a Bluefin-class vessel could sail each target -- inside the
    basin at the start, one heading throughout, stopped short of the wall only
    after the encounter, at an achievable speed (the VAR final speed too)."""
    lo, hi = ct.FIELD_TARGET_SPEED_RANGE
    box = p2.stop_box()
    for r in cell:
        b = r["built"]
        assert lo <= b.target_speed <= hi
        if r["speed"] == "VAR":
            assert lo - 1e-9 <= b.flags["speed_profile"][1] <= hi + 1e-9
        assert b.flags["target_stop_box"] == box
        xs, ys = zip(*tgt.hull_polygon(b.target_spawn[0], b.target_spawn[1], b.target_heading))
        assert box[0] <= min(xs) and max(xs) <= box[1] and box[2] <= min(ys) and max(ys) <= box[3]
        check = p2.nominal_check(env, b, r["episode_seed"])
        assert p2.field_feasible(check)
        assert check["turned"] <= ct.FIELD_MAX_TURN_DEG


def test_field_sheet_gives_each_run_its_set_up(cell):
    sheet = p2.field_sheet(cell)
    assert len(sheet) == len(cell)
    var = [row for row in sheet if row["test_id"].split("-")[3] == "VAR"]
    assert all(row["speed_change_at_s"] != "" and row["speed_change_to_mps"] != "" for row in var)
    assert all(set(row) >= {"target_start", "target_heading_deg", "target_speed_mps"} for row in sheet)


def test_a_target_stops_short_of_the_wall():
    t = tgt.Target(5.0, 20.0, 0.0, 0.5, stop_box=(0.5, 9.5, 0.5, 24.5))
    for _ in range(200):
        t.step(0.1)
    assert t._stopped and t.speed == 0.0
    assert max(y for _, y in tgt.hull_polygon(t.x, t.y, t.heading)) <= 24.5
    free = tgt.Target(5.0, 20.0, 0.0, 0.5)
    for _ in range(200):
        free.step(0.1)
    assert free.y == pytest.approx(30.0)                      # no box: training targets sail on
