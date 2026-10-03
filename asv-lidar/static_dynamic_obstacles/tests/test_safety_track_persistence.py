"""Synthetic sensor snapshots only; no environment construction or episodes."""
from collections import deque
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
from safety_track_persistence import TrackPersistencePerception


class _Base:
    def __init__(self):
        self.calls = 0
        self.memory = [(1, np.array([[4., 8.], [5., 8.]]))]
        self.result = cc.Snapshot(0., 0., 0., .5, 0., 0.,
                                  np.array([0., 1.]), np.array([1., 0.]), np.zeros(2),
                                  0., 0., 20., self.memory[0][1].copy(), [])

    def snapshot(self, env):
        self.calls += 1
        return self.result


class _OnboardOnly(SimpleNamespace):
    @property
    def targets(self):
        raise AssertionError("Runtime read hidden targets")

    @property
    def asv_x(self):
        raise AssertionError("Runtime read true ego pose")


def _points(centre, length=None, width=None):
    length = cfg.LOA if length is None else length
    width = cfg.BREADTH if width is None else width
    x = np.linspace(-length / 2, length / 2, 10)
    return np.concatenate([np.column_stack((x, np.full(len(x), side * width / 2)))
                           for side in (-1, 1)]) + centre


def _setup(**kwargs):
    base = _Base()
    track = SimpleNamespace(id=7, hits=cfg.TRACK_MIN_HITS, misses=0,
                            velocity=np.array([.4, 0.]), position=np.array([4., 8.]),
                            last_fit_centre=np.array([4., 8.]), last_fit_heading_deg=90.,
                            last_evidence=None, is_dynamic=False, history=deque())
    env = _OnboardOnly(tracker=SimpleNamespace(tracks=[track]), tracks=[], pose_stale=False,
                       lidar=SimpleNamespace(bearings=np.arange(720) / 2),
                       raw_ranges=np.full(720, cfg.LIDAR_RANGE))
    return TrackPersistencePerception(base, **kwargs), base, env, track


def _observe(wrapper, env, track, serial, *, centre=None, points=None):
    centre = np.array([4. + .2 * (serial - 1), 8.]) if centre is None else np.asarray(centre, dtype=float)
    track.position = centre - [0., .2]
    track.last_fit_centre = centre.copy()
    track.history.append((serial, _points(centre) if points is None else points))
    return wrapper.snapshot(env)


def _admit(wrapper, env, track):
    assert not _observe(wrapper, env, track, 1).tracks
    assert not _observe(wrapper, env, track, 2).tracks
    snap = _observe(wrapper, env, track, 3)
    assert len(snap.tracks) == 1
    return snap.tracks[0]


def test_three_full_hull_translations_admit_without_free_space_motion_evidence():
    wrapper, base, env, track = _setup()
    first = _admit(wrapper, env, track)
    assert first.id < 0 and first.ctx is None
    np.testing.assert_allclose(first.position, [4.4, 8.])
    np.testing.assert_allclose(first.velocity, [.4, 0.], atol=1e-12)
    stats = wrapper.last_track_persistence_stats
    assert stats['updates'][0]['observation_frames'] == [1, 2, 3]
    assert stats['updates'][0]['base_admitted'] is False
    assert base.calls == 3 and not track.is_dynamic and not env.tracks


def test_raw_id_deletion_preserves_anchor_until_explicit_expiry():
    wrapper, _, env, track = _setup(max_coast_s=1.)
    first = _admit(wrapper, env, track)
    env.tracker.tracks = []
    for age in (.5, 1.):
        view = wrapper.snapshot(env).tracks[0]
        np.testing.assert_allclose(view.position, first.position + age * first.velocity)
        assert wrapper.last_track_persistence_stats['hypotheses'][0]['anchor_frame'] == 3
    assert not wrapper.snapshot(env).tracks
    assert wrapper.last_track_persistence_stats['expired_source_ids'] == [7]


def test_partial_occlusion_and_collapsing_raw_velocity_do_not_refresh_anchor():
    wrapper, _, env, track = _setup()
    first = _admit(wrapper, env, track)
    track.velocity[:] = 0.
    for serial in (4, 5, 6):
        centre = np.array([4.4, 8.])
        view = _observe(wrapper, env, track, serial, centre=centre,
                        points=_points(centre, length=.6)).tracks[0]
        np.testing.assert_allclose(view.velocity, first.velocity)
        np.testing.assert_allclose(view.position, first.position + (serial-3)*cfg.UPDATE_RATE*first.velocity)
        assert not wrapper.last_track_persistence_stats['updates']


@pytest.mark.parametrize('kind', ['short_panel', 'oversized_width', 'oversized_length', 'stationary', 'reverse_fit_axis'])
def test_static_or_incompatible_clusters_not_admitted(kind):
    wrapper, _, env, track = _setup()
    for serial in (1, 2, 3, 4):
        centre = np.array([4. + .2*(serial-1), 8.])
        if kind == 'stationary': centre[:] = [4., 8.]
        if kind == 'reverse_fit_axis': track.last_fit_heading_deg = 270.
        points = _points(centre,
                         length=1. if kind == 'short_panel' else (3. if kind == 'oversized_length' else None),
                         width=1. if kind == 'oversized_width' else None)
        assert not _observe(wrapper, env, track, serial, centre=centre, points=points).tracks


def test_visible_segment_growth_cannot_substitute_for_movement_of_both_ends():
    wrapper, _, env, track = _setup()
    # Fixed left end; extra returns reveal more of a stationary long surface.
    for serial, length in enumerate((cfg.LOA-.14, cfg.LOA, cfg.LOA+.14), 1):
        centre = np.array([4. + length/2, 8.])
        assert not _observe(wrapper, env, track, serial, centre=centre,
                            points=_points(centre, length=length)).tracks


def test_repeated_scan_does_not_manufacture_temporal_evidence():
    wrapper, _, env, track = _setup()
    _observe(wrapper, env, track, 1)
    for _ in range(4):
        assert not wrapper.snapshot(env).tracks
    assert wrapper.last_track_persistence_stats['rejections']['repeated_observation'] == 1


def test_stale_pose_cannot_admit_or_refresh():
    wrapper, _, env, track = _setup()
    env.pose_stale = True
    for serial in range(1, 5):
        assert not _observe(wrapper, env, track, serial).tracks


@pytest.mark.parametrize('condition', ['no_return', 'occluded', 'stale', 'dead_zone'])
def test_unknown_visibility_does_not_clear_an_anchor(condition):
    wrapper, base, env, track = _setup()
    _admit(wrapper, env, track)
    env.tracker.tracks = []
    if condition == 'occluded': env.raw_ranges[:] = 2.
    elif condition == 'stale':
        env.raw_ranges[:] = cfg.LIDAR_RANGE - 1
        env.pose_stale = True
    elif condition == 'dead_zone':
        base.result.x, base.result.y = 4.6, 7.14
        env.raw_ranges[:] = cfg.LIDAR_RANGE - 1
    assert len(wrapper.snapshot(env).tracks) == 1
    assert not wrapper.last_track_persistence_stats['contradicted_source_ids']


def test_finite_rays_through_predicted_hull_clear_unsupported_anchor():
    wrapper, _, env, track = _setup()
    _admit(wrapper, env, track)
    env.tracker.tracks = []
    env.raw_ranges[:] = cfg.LIDAR_RANGE - 1
    assert not wrapper.snapshot(env).tracks
    assert wrapper.last_track_persistence_stats['contradicted_source_ids'] == [7]


def test_admission_disabled_requires_existing_base_view():
    wrapper, base, env, track = _setup(enable_admission=False)
    for serial in (1, 2, 3):
        assert not _observe(wrapper, env, track, serial).tracks
    base.result.tracks = [cc.TrackView(track.id, np.array([4.6, 8.]), track.velocity.copy(), np.pi/2, object())]
    assert len(_observe(wrapper, env, track, 4).tracks) == 2
    assert wrapper.last_track_persistence_stats['updates'][0]['base_admitted'] is True


def test_persistence_disabled_keeps_current_admission_only():
    wrapper, _, env, track = _setup(enable_persistence=False)
    _admit(wrapper, env, track)
    env.tracker.tracks = []
    assert not wrapper.snapshot(env).tracks


def test_new_actual_measurements_refresh_age_and_keep_synthetic_id():
    wrapper, _, env, track = _setup()
    first = _admit(wrapper, env, track)
    next_view = _observe(wrapper, env, track, 4).tracks[0]
    assert first.id == next_view.id
    assert wrapper.last_track_persistence_stats['hypotheses'][0]['age_s'] == 0
    assert wrapper.last_track_persistence_stats['updates'][0]['kind'] == 'measured_refresh'


def test_retains_static_points_and_never_mutates_tracker_or_contexts():
    wrapper, base, env, track = _setup()
    _observe(wrapper, env, track, 1)
    _observe(wrapper, env, track, 2)
    original_points = base.result.points.copy()
    original_memory = copy.deepcopy(base.memory)
    snap = _observe(wrapper, env, track, 3)
    raw_copy = copy.deepcopy(track.__dict__)
    snap.tracks[0].position[:] = 999.
    snap.tracks[0].velocity[:] = 999.
    assert snap.points is base.result.points and not base.result.tracks
    np.testing.assert_array_equal(base.result.points, original_points)
    np.testing.assert_array_equal(base.memory[0][1], original_memory[0][1])
    for key in ('position', 'velocity', 'last_fit_centre'):
        np.testing.assert_array_equal(track.__dict__[key], raw_copy[key])
    assert not track.is_dynamic and not env.tracks
    env.tracker.tracks = []
    assert wrapper.snapshot(env).tracks[0].position[0] < 10


def test_deepcopy_delegate_does_not_recurse():
    wrapper, _, _, _ = _setup()
    cloned = copy.deepcopy(wrapper)
    assert cloned.base_perception is not wrapper.base_perception


@pytest.mark.parametrize('kwargs', [dict(max_coast_s=-1), dict(max_coast_s=float('inf')),
                                    dict(min_motion_observations=2), dict(min_motion_observations=3.5)])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        _setup(**kwargs)
