"""Target geometry/return ownership; synthetic sensor snapshots, no episodes."""
from types import SimpleNamespace
import math

import numpy as np
import pytest

import constants as cfg
import tracking
from classical import common as cc
from safety_hull_perception import HullSafetyPerception, _HullFit, _explained_points
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


def _track(centre=(0.0, 5.0), heading=0.0):
    return SimpleNamespace(id=1, position=np.array([0.0, 4.5]),
                           velocity=np.array([0.0, 0.4]), misses=0,
                           last_fit_centre=np.asarray(centre), last_fit_heading_deg=heading)


def _scan_points(env, points):
    points = np.asarray(points, dtype=float)
    origin = np.asarray(tracking.sensor_origin(*env.estimated_pose()))
    rel = points - origin
    env.lidar.bearings = np.degrees(np.arctan2(rel[:, 0], rel[:, 1]))
    env.raw_ranges = env.gated_ranges = np.linalg.norm(rel, axis=1)
    env.lidar.ranges = env.raw_ranges.copy()


def _remember(perception, points):
    perception.frames = 1
    perception.memory = [(1, np.asarray(points, dtype=float).copy())]


def test_target_return_removed_but_neighbouring_static_panel_survives():
    env = _environment()
    env.tracks = [_track()]
    target = [0.0, 5.0 - 0.5 * cfg.LOA]
    panel = [1.6, 5.0]
    _scan_points(env, [target, panel])
    legacy = SafetyPerception(memory_frames=180).snapshot(env)
    assert len(legacy.points) == 0
    perception = HullSafetyPerception(memory_frames=180)
    snap = perception.snapshot(env)
    np.testing.assert_allclose(snap.points, [panel])
    np.testing.assert_allclose(perception.memory[0][1], [panel])
    np.testing.assert_allclose(snap.tracks[0].position, [0.0, 5.0])
    assert perception.last_hull_stats["fresh_returns_excluded"] == 1


@pytest.mark.parametrize("heading", [0.0, 37.0, 90.0, 180.0])
def test_oriented_footprint_uses_physical_hull_and_existing_tolerance(heading):
    angle = math.radians(heading)
    fwd = np.array([math.sin(angle), math.cos(angle)])
    right = np.array([fwd[1], -fwd[0]])
    centre = np.array([3.0, 6.0])
    tolerance = cfg.MOTION_EXPLAIN_M
    points = np.array([centre + 0.5 * cfg.LOA * fwd,
                       centre - 0.5 * cfg.BREADTH * right,
                       centre + (0.5 * cfg.BREADTH + tolerance + 0.01) * right,
                       centre + (0.5 * cfg.LOA + tolerance + 0.01) * fwd])
    np.testing.assert_array_equal(_explained_points(points, [_HullFit(centre, angle, 0)]),
                                  [True, True, False, False])


def test_geometry_view_does_not_modify_tracker_velocity_position_or_context():
    env = _environment()
    track = _track(heading=30.0)
    env.tracks = [track]
    context = object()
    env.encounter_contexts[1] = context
    original = (track.position.copy(), track.velocity.copy(), track.last_fit_centre.copy())
    snap = HullSafetyPerception().snapshot(env)
    np.testing.assert_allclose(snap.tracks[0].position, track.last_fit_centre)
    np.testing.assert_array_equal(snap.tracks[0].velocity, original[1])
    assert snap.tracks[0].ctx is context
    for actual, expected in zip((track.position, track.velocity, track.last_fit_centre), original):
        np.testing.assert_array_equal(actual, expected)
    snap.tracks[0].position[:] = 42.0
    np.testing.assert_array_equal(track.last_fit_centre, original[2])


@pytest.mark.parametrize("invalid", ["none", "nan_centre", "wrong_shape", "nan_heading", "missing_heading", "missed", "stale"])
def test_unreliable_fit_retains_unassigned_returns_and_legacy_track_position(invalid):
    env = _environment()
    track = _track()
    if invalid == "none": track.last_fit_centre = None
    elif invalid == "nan_centre": track.last_fit_centre = np.array([np.nan, 5.0])
    elif invalid == "wrong_shape": track.last_fit_centre = np.array([5.0])
    elif invalid == "nan_heading": track.last_fit_heading_deg = np.nan
    elif invalid == "missing_heading": track.last_fit_heading_deg = None
    elif invalid == "missed": track.misses = 1
    else: env.pose_stale = True
    env.tracks = [track]
    perception = HullSafetyPerception(memory_frames=180)
    expected = np.array([[1.6, 5.0]])
    _remember(perception, expected)
    snap = perception.snapshot(env)
    np.testing.assert_array_equal(snap.tracks[0].position, track.position)
    np.testing.assert_array_equal(snap.points, expected)
    assert perception.last_hull_stats["tracks_without_hull_fit"] == 1


@pytest.mark.parametrize("held_pose", [False, True])
def test_cached_fit_coasts_with_measured_velocity_then_expires(held_pose):
    env = _environment()
    track = _track()
    env.tracks = [track]
    perception = HullSafetyPerception()
    perception.snapshot(env)
    env.pose_stale = held_pose
    if not held_pose:
        track.last_fit_centre = track.last_fit_heading_deg = None
        track.misses = 1
    for age in range(1, cfg.TRACK_MAX_MISSES + 1):
        snap = perception.snapshot(env)
        np.testing.assert_allclose(snap.tracks[0].position,
                                   np.array([0.0, 5.0]) + age * cfg.UPDATE_RATE * track.velocity)
        assert perception.last_hull_stats["coasted_hull_fits"] == 1
    np.testing.assert_array_equal(perception.snapshot(env).tracks[0].position, track.position)
    assert perception.last_hull_stats["tracks_without_hull_fit"] == 1


def test_stale_pose_neither_clears_memory_nor_ingests_scan():
    env = _environment()
    perception = HullSafetyPerception(memory_frames=180)
    expected = np.array([[0.0, tracking.sensor_origin(0.0, 0.0, 0.0)[1] + 3.0]])
    _remember(perception, expected)
    env.pose_stale = True
    env.raw_ranges[:] = 8.0
    env.gated_ranges[0] = 2.0
    np.testing.assert_allclose(perception.snapshot(env).points, expected)
    np.testing.assert_allclose(perception.memory[0][1], expected)
    assert perception.last_memory_stats["cleared_points"] == 0


def test_finite_free_space_still_clears_memory_permanently():
    env = _environment()
    perception = HullSafetyPerception(memory_frames=180)
    _remember(perception, [[0.0, tracking.sensor_origin(0.0, 0.0, 0.0)[1] + 3.0]])
    env.raw_ranges[:] = 8.0
    assert len(perception.snapshot(env).points) == 0
    assert perception.memory == []
    assert perception.last_memory_stats["cleared_points"] == 1


def test_hull_mask_never_permanently_deletes_occluded_memory():
    env = _environment()
    env.tracks = [_track()]
    perception = HullSafetyPerception(memory_frames=180)
    expected = np.array([[0.0, 5.0]])
    _remember(perception, expected)
    assert len(perception.snapshot(env).points) == 0
    np.testing.assert_array_equal(perception.memory[0][1], expected)
    env.tracks = []
    np.testing.assert_array_equal(perception.snapshot(env).points, expected)
    assert perception._hull_fits == {}


@pytest.mark.parametrize("centre,mask", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_constructor_switches(centre, mask):
    env = _environment()
    env.tracks = [_track()]
    panel = [1.6, 5.0]
    _scan_points(env, [panel])
    snap = HullSafetyPerception(fitted_track_centre=centre, hull_return_mask=mask).snapshot(env)
    np.testing.assert_array_equal(snap.tracks[0].position,
                                  env.tracks[0].last_fit_centre if centre else env.tracks[0].position)
    assert len(snap.points) == int(mask)


def test_snapshot_requires_no_simulator_truth():
    class OnboardOnly:
        def __init__(self, env): self.env = env
        def __getattr__(self, name):
            assert name not in {"targets", "obstacles", "asv_x", "asv_y", "asv_h", "u_body", "v_body"}
            return getattr(self.env, name)
    env = _environment()
    env.tracks = [_track()]
    snap = HullSafetyPerception().snapshot(OnboardOnly(env))
    np.testing.assert_array_equal(snap.tracks[0].position, [0.0, 5.0])
