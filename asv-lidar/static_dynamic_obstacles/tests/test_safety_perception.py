"""Conservative scan-memory clearing without running simulation episodes."""
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
import tracking
from classical import common as cc
from safety_perception import SafetyPerception


def _environment():
    bearings = np.arange(cfg.LIDAR_BEAMS, dtype=float) * (360.0 / cfg.LIDAR_BEAMS)
    ranges = np.full(len(bearings), cfg.LIDAR_RANGE)
    path = SimpleNamespace(
        project=lambda x, y, h: SimpleNamespace(closest_idx=0, target=np.array([0.0, y])),
        tangent=lambda index: np.array([0.0, 1.0]),
        points=np.array([[0.0, 0.0], [0.0, 20.0]]))
    return SimpleNamespace(
        lidar=SimpleNamespace(bearings=bearings, ranges=ranges.copy()),
        raw_ranges=ranges.copy(), gated_ranges=ranges.copy(),
        estimated_pose=lambda: (0.0, 0.0, 0.0), _measured_ego=lambda: (0.5, 0.0, 0.0),
        tracks=[], encounter_contexts={}, path=path, pose_stale=False,
        boundary_polygon=[[-20, -20], [20, -20], [20, 20], [-20, 20]])


def _remember(perception, *sensor_relative_points):
    origin = np.asarray(tracking.sensor_origin(0.0, 0.0, 0.0))
    points = np.asarray(sensor_relative_points, dtype=float) + origin
    perception.frames = 1
    perception.memory = [(1, points.copy())]
    return points


def test_finite_raw_wall_return_removes_vacated_ghost_permanently():
    env = _environment()
    perception = SafetyPerception(memory_frames=180)
    _remember(perception, (0.0, 3.0))
    env.raw_ranges[:] = 8.0      # Walls remain useful evidence after boundary gating.
    snap = perception.snapshot(env)
    assert snap.points.shape == (0, 2)
    assert perception.memory == []
    assert perception.last_memory_stats["cleared_points"] == 1
    env.raw_ranges[:] = cfg.LIDAR_RANGE
    assert len(perception.snapshot(env).points) == 0
    assert perception.total_cleared_points == 1


@pytest.mark.parametrize("kind", ["no_return", "occluded", "aft_mask", "dead_zone"])
def test_unknown_or_occluded_space_preserves_memory(kind):
    env = _environment()
    perception = SafetyPerception(memory_frames=180)
    point = (0.0, 3.0)
    if kind == "occluded":
        env.raw_ranges[:] = 2.0
    elif kind == "aft_mask":
        env.raw_ranges[:] = 8.0
        centre = len(env.raw_ranges) // 2
        env.raw_ranges[centre - 35:centre + 36] = cfg.LIDAR_RANGE
        point = (0.0, -3.0)
    elif kind == "dead_zone":
        env.raw_ranges[:] = 8.0
        point = (0.0, 0.5 * cfg.LIDAR_MIN_RANGE)
    expected = _remember(perception, point)
    np.testing.assert_allclose(perception.snapshot(env).points, expected)
    np.testing.assert_allclose(perception.memory[0][1], expected)
    assert perception.last_memory_stats["cleared_points"] == 0


def test_stale_pose_preserves_memory_and_does_not_ingest_fresh_scan():
    env = _environment()
    perception = SafetyPerception(memory_frames=180)
    expected = _remember(perception, (0.0, 3.0))
    env.raw_ranges[:] = 8.0       # Would clear the ghost if registered correctly.
    env.gated_ranges[0] = 2.0    # Must not become a new point under the held pose.
    env.pose_stale = True
    np.testing.assert_allclose(perception.snapshot(env).points, expected)
    assert perception.frames == 2
    assert len(perception.memory) == 1
    assert perception.last_memory_stats["cleared_points"] == 0
    assert perception.last_memory_stats["pose_stale"]


def test_current_obstacle_return_is_added_after_old_ghost_is_cleared():
    env = _environment()
    perception = SafetyPerception(memory_frames=180)
    _remember(perception, (0.0, 3.0))
    env.raw_ranges[:] = 8.0
    env.raw_ranges[0] = env.gated_ranges[0] = 4.0
    expected = np.asarray(tracking.sensor_origin(0.0, 0.0, 0.0)) + [0.0, 4.0]
    snap = perception.snapshot(env)
    np.testing.assert_allclose(snap.points, expected[None])
    assert perception.last_memory_stats["cleared_points"] == 1
    assert perception.last_memory_stats["remembered_after"] == 1


def test_lateral_grazing_return_blocks_clearing():
    env = _environment()
    perception = SafetyPerception(memory_frames=180)
    expected = _remember(perception, (0.0, 3.0))
    env.raw_ranges[:] = 8.0
    env.raw_ranges[1] = 2.9      # Central ray clears; neighbouring panel face does not.
    np.testing.assert_allclose(perception.snapshot(env).points, expected)
    assert perception.last_memory_stats["cleared_points"] == 0


def test_nearby_return_explanation_protects_a_passed_point():
    perception = SafetyPerception(memory_frames=180)
    expected = _remember(perception, (0.0, 3.0))
    # Exercise the additional explanation guard independently of radial/lateral
    # certificates; the tracker defines both as necessary for vacated space.
    scan = SimpleNamespace(passes_through=lambda points, tol: np.ones(len(points), dtype=bool),
                           explains=lambda points, radius: np.ones(len(points), dtype=bool))
    assert perception._clear_vacated(scan) == 0
    np.testing.assert_allclose(perception.memory[0][1], expected)


def test_dynamic_track_exclusion_does_not_permanently_erase_occluded_static_memory():
    env = _environment()
    perception = SafetyPerception(memory_frames=180)
    expected = _remember(perception, (0.0, 3.0))
    env.tracks = [SimpleNamespace(id=1, position=expected[0], velocity=np.zeros(2),
                                  last_fit_heading_deg=None)]
    assert len(perception.snapshot(env).points) == 0
    np.testing.assert_allclose(perception.memory[0][1], expected)
    env.tracks = []
    np.testing.assert_allclose(perception.snapshot(env).points, expected)


def test_memory_expiry_is_unchanged_on_stale_frames():
    env = _environment()
    env.pose_stale = True
    perception = SafetyPerception(memory_frames=2)
    _remember(perception, (0.0, 3.0))
    assert len(perception.snapshot(env).points) == 1
    assert len(perception.snapshot(env).points) == 0
    assert isinstance(perception, cc.Perception)
