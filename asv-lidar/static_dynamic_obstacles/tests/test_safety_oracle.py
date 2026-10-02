"""Oracle geometry/state checks; no policy loading or episode evaluation."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from classical import common as cc
from env import _overlaps, _polygon_gap
from safety_v3 import SafetyFilterV3
from tools.diagnostics.safety.oracle import install_oracle, polygon_clearance, true_snapshot


def _environment():
    path = SimpleNamespace(
        project=lambda x, y, h: SimpleNamespace(closest_idx=0, target=np.array([5.0, y])),
        tangent=lambda index: np.array([0.0, 1.0]),
        points=np.array([[5.0, 0.0], [5.0, 25.0]]))
    return SimpleNamespace(
        asv_x=5.5, asv_y=8.0, asv_h=30.0, u_body=0.7, v_body=-0.1, asv_w=4.0,
        path=path, obstacles=[], targets=[], command_rate_limit=False,
        boundary_polygon=[[-100, -100], [100, -100], [100, 100], [-100, 100]])


def test_snapshot_uses_truth_and_converts_angles_without_reading_sensors():
    env = _environment()
    env.targets = [SimpleNamespace(x=9.0, y=10.0, velocity=np.array([0.4, -0.2]),
                                   heading_deg=120.0)]
    env.obstacles = [[(1.0, 2.0), (2.0, 2.0), (2.0, 3.0), (1.0, 3.0)]]
    snap = true_snapshot(env)
    assert (snap.x, snap.y, snap.u, snap.v) == (5.5, 8.0, 0.7, -0.1)
    assert snap.heading == pytest.approx(math.radians(30.0))
    assert snap.r == pytest.approx(math.radians(4.0))
    assert snap.lateral == pytest.approx(0.5)
    assert snap.remaining == pytest.approx(17.0)
    assert snap.tracks[0].heading == pytest.approx(math.radians(120.0))
    np.testing.assert_allclose(snap.tracks[0].velocity, [0.4, -0.2])
    np.testing.assert_allclose(snap.static_polygons[0], env.obstacles[0])
    assert snap.points.shape == (0, 2)


def test_filter_uses_current_truth_without_ema_lag_or_global_changes(monkeypatch):
    import safety_v2

    env = _environment()
    ordinary = SafetyFilterV3()
    oracle = install_oracle(SafetyFilterV3())
    smoothing = safety_v2.EGO_SMOOTHING
    monkeypatch.setattr(oracle, "_threat_in_reach", lambda snap: False)
    for u, v, yaw_dps in [(0.7, -0.1, 4.0), (0.3, 0.2, -8.0)]:
        env.u_body, env.v_body, env.asv_w = u, v, yaw_dps
        oracle.filter(env, [0.0, 0.0])
        np.testing.assert_allclose(oracle.ego, [u, v, math.radians(yaw_dps)])
    assert safety_v2.EGO_SMOOTHING == smoothing
    assert isinstance(ordinary.perception, cc.Perception)
    with pytest.raises(ValueError, match="already installed"):
        install_oracle(oracle)


def test_complete_panel_edges_detect_collision_when_all_vertices_are_distant():
    panel = np.array([[-20.0, -0.05], [20.0, -0.05], [20.0, 0.05], [-20.0, 0.05]])
    positions, headings = np.zeros((1, 1, 2)), np.zeros((1, 1))
    assert cc.point_clearance(positions, headings, panel)[0, 0] > 10.0
    assert polygon_clearance(positions, headings, [panel])[0, 0] < 0.0
    env = _environment()
    env.asv_x = env.asv_y = env.asv_h = 0.0
    env.obstacles = [panel]
    oracle = install_oracle(SafetyFilterV3())
    assert oracle._threat_in_reach(oracle.perception.snapshot(env))


def test_corner_distance_and_containment_are_exact_for_inflated_hull():
    corner = np.array([[cc.HALF_W + 3.0, cc.HALF_L + 4.0],
                       [cc.HALF_W + 4.0, cc.HALF_L + 4.0],
                       [cc.HALF_W + 4.0, cc.HALF_L + 5.0],
                       [cc.HALF_W + 3.0, cc.HALF_L + 5.0]])
    positions, headings = np.zeros((1, 1, 2)), np.zeros((1, 1))
    assert polygon_clearance(positions, headings, [corner])[0, 0] == pytest.approx(5.0)
    enclosure = np.array([[-10, -10], [-10, 10], [10, 10], [10, -10]])
    assert polygon_clearance(positions, headings, [enclosure])[0, 0] < 0.0
    assert np.isinf(polygon_clearance(positions, headings, [])[0, 0])


def test_vectorized_polygons_agree_with_environment_geometry():
    rng = np.random.default_rng(721)
    positions = rng.uniform(-3.0, 3.0, size=(4, 8, 2))
    headings = rng.uniform(-math.pi, math.pi, size=(4, 8))
    polygon = np.array([[-0.8, -0.2], [0.5, -0.6], [1.2, 0.5], [-0.2, 1.0]])
    actual = polygon_clearance(positions, headings, [polygon])
    reverse = polygon_clearance(positions, headings, [polygon[::-1]])
    np.testing.assert_allclose(actual, reverse, atol=1e-12)
    for index in np.ndindex(headings.shape):
        x, y = positions[index]
        h = headings[index]
        hull = [(x + f * math.sin(h) + s * math.cos(h),
                 y + f * math.cos(h) - s * math.sin(h))
                for f, s in [(cc.HALF_L, cc.HALF_W), (cc.HALF_L, -cc.HALF_W),
                             (-cc.HALF_L, -cc.HALF_W), (-cc.HALF_L, cc.HALF_W)]]
        if _overlaps(hull, polygon):
            assert actual[index] <= 0.0
        else:
            assert actual[index] == pytest.approx(_polygon_gap(hull, polygon)[0], abs=1e-12)


def test_oracle_evaluation_preserves_static_gap_and_terminal_extension():
    import safety_v2

    env = _environment()
    env.asv_x = env.asv_y = env.asv_h = 0.0
    env.obstacles = [[(-2.0, cc.HALF_L + 0.5), (2.0, cc.HALF_L + 0.5),
                      (2.0, cc.HALF_L + 0.6), (-2.0, cc.HALF_L + 0.6)]]
    oracle = install_oracle(SafetyFilterV3())
    snap = oracle.perception.snapshot(env)
    rollout = cc.Rollout(np.zeros((1, 1, 2)), np.zeros((1, 1)),
                         np.zeros((1, 1)), np.array([8.0]))
    first, gap = oracle._evaluate(snap, rollout)
    assert np.isinf(first[0])
    assert gap[0] == pytest.approx(0.5 - safety_v2.GAP_STATIC_M)
    rollout.speeds[:] = 0.6
    first, gap = oracle._evaluate(snap, rollout)
    assert first[0] == 8.0
    assert gap[0] < 0.0


def test_state_only_adapter_retains_scan_geometry():
    env = _environment()
    ordinary = SafetyFilterV3()
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    ordinary.perception = SimpleNamespace(snapshot=lambda env: SimpleNamespace(points=points))
    original_evaluator = ordinary._evaluate
    oracle = install_oracle(ordinary, exact_geometry=False)
    snap = oracle.perception.snapshot(env)
    np.testing.assert_array_equal(snap.points, points)
    assert oracle._evaluate == original_evaluator
    assert snap.x == env.asv_x
