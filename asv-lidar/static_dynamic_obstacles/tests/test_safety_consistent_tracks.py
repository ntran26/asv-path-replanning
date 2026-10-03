"""Synthetic snapshots and algebraic shape checks; no environment episodes."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
from safety_consistent_tracks import ConsistentTrackPerception, centre_with_shape_prior
from safety_track_persistence import TrackPersistencePerception
from safety_v13 import SafetyFilterV13
import safety_v11
from test_safety_track_persistence import _setup as _legacy_setup, _observe, _admit, _points


def _setup(**kwargs):
    _, base, env, track = _legacy_setup()
    return ConsistentTrackPerception(base, **kwargs), base, env, track


@pytest.mark.parametrize('heading', [0., .71, np.pi/2, 3.4])
def test_partial_axis_keeps_feasible_prior_and_full_axis_uses_midpoint(heading):
    fwd = np.array([np.sin(heading), np.cos(heading)])
    axes = np.column_stack((fwd, [fwd[1], -fwd[0]]))
    centre = np.array([4., 8.])
    local = np.array([[-cfg.LOA/2, -.12], [-cfg.LOA/2, .12],
                      [cfg.LOA/2, -.12], [cfg.LOA/2, .12]])
    prior = centre + axes @ np.array([.3, .10])
    got, diag = centre_with_shape_prior(centre + local @ axes.T, heading, prior)
    np.testing.assert_allclose(got, centre + axes @ np.array([0., .10]), atol=1e-12)
    assert diag['full_axes'] == [True, False]


def test_partial_axis_clips_prior_to_feasible_known_width_interval():
    points = _points(np.array([4., 8.]), width=.24)
    got, _ = centre_with_shape_prior(points, np.pi/2, np.array([4., 9.]))
    np.testing.assert_allclose(got, [4., 8.13], atol=1e-12)


def test_tolerated_slightly_oversized_axis_uses_midpoint_not_inverted_interval():
    centre = np.array([4., 8.])
    points = _points(centre, length=cfg.LOA+.1, width=cfg.BREADTH+.1)
    got, diag = centre_with_shape_prior(points, np.pi/2, centre+[2., 2.])
    np.testing.assert_allclose(got, centre, atol=1e-12)
    assert diag['full_axes'] == [True, True]


def test_partial_side_fit_flip_cannot_jump_existing_anchor_centre():
    wrapper, _, env, track = _setup()
    first = _admit(wrapper, env, track)
    # Same moving visible footprint; the fitter chooses the opposite completion
    # of its partial lateral axis, shifting its returned centre by half a beam.
    actual = np.array([4.6, 8.])
    snap = _observe(wrapper, env, track, 4, centre=actual+[0., .25],
                    points=_points(actual, width=.24))
    np.testing.assert_allclose(snap.tracks[0].position, actual, atol=1e-12)
    assert snap.tracks[0].id == first.id
    geometry = wrapper.last_track_persistence_stats['updates'][0]['geometry_prior']
    np.testing.assert_allclose(geometry['uncorrected_centre'], actual+[0., .25])
    assert geometry['full_axes'] == [True, False]


def test_geometry_disabled_keeps_original_fit_completion():
    wrapper, _, env, track = _setup(geometry_prior=False)
    _admit(wrapper, env, track)
    actual = np.array([4.6, 8.])
    snap = _observe(wrapper, env, track, 4, centre=actual+[0., .25],
                    points=_points(actual, width=.24))
    np.testing.assert_allclose(snap.tracks[0].position, actual+[0., .25])


def test_first_admission_has_no_invented_geometry_prior():
    wrapper, _, env, track = _setup()
    _observe(wrapper, env, track, 1)
    _observe(wrapper, env, track, 2)
    centre = np.array([4.4, 8.25])
    snap = _observe(wrapper, env, track, 3, centre=centre,
                    points=_points(np.array([4.4, 8.]), width=.24))
    np.testing.assert_allclose(snap.tracks[0].position, centre)
    assert 'geometry_prior' not in wrapper.last_track_persistence_stats['updates'][0]


def test_independent_owner_replaces_only_matching_base_id_and_preserves_other_context():
    wrapper, base, env, track = _setup()
    first = _admit(wrapper, env, track)
    context = object()
    matching = cc.TrackView(7, np.array([9., 9.]), np.zeros(2), 0., object())
    other = cc.TrackView(20, np.array([4., 8.]), np.zeros(2), 0., context)
    base.result.tracks = [matching, other]
    env.tracker.tracks = []
    snap = wrapper.snapshot(env)
    assert [v.id for v in snap.tracks] == [20, first.id]
    assert snap.tracks[0] is other and snap.tracks[0].ctx is context
    assert base.result.tracks == [matching, other]
    assert snap.points is base.result.points
    stats = wrapper.last_track_persistence_stats
    assert stats['source_owners'] == {7: 'persistent'}
    assert stats['replaced_base_source_ids'] == [7]


def test_base_owner_keeps_existing_view_then_persists_after_source_disappears():
    wrapper, base, env, track = _setup()
    context = object()
    live = cc.TrackView(7, np.array([4., 8.]), np.zeros(2), 0., context)
    base.result.tracks = [live]
    for serial in (1, 2, 3):
        snap = _observe(wrapper, env, track, serial)
        assert snap.tracks == [live]
    assert wrapper.last_track_persistence_stats['source_owners'] == {7: 'base'}
    assert wrapper.last_track_persistence_stats['suppressed_source_ids'] == [7]
    base.result.tracks = []
    env.tracker.tracks = []
    snap = wrapper.snapshot(env)
    assert len(snap.tracks) == 1 and snap.tracks[0].id < 0
    assert live.ctx is context


def test_first_admission_ownership_survives_later_base_presence_and_refresh():
    wrapper, base, env, track = _setup()
    first = _admit(wrapper, env, track)
    base.result.tracks = [cc.TrackView(7, np.array([4., 8.]), np.zeros(2), 0.)]
    snap = _observe(wrapper, env, track, 4)
    assert [v.id for v in snap.tracks] == [first.id]
    assert wrapper.last_track_persistence_stats['source_owners'] == {7: 'persistent'}
    assert wrapper.last_track_persistence_stats['updates'][0]['base_admitted'] is True


def test_expired_anchor_releases_ownership_and_restores_matching_base_view():
    wrapper, base, env, track = _setup(max_coast_s=.5)
    _admit(wrapper, env, track)
    live = cc.TrackView(7, np.array([4., 8.]), np.zeros(2), 0.)
    base.result.tracks = [live]
    env.tracker.tracks = []
    assert wrapper.snapshot(env).tracks[0].id < 0
    snap = wrapper.snapshot(env)
    assert snap.tracks == [live]
    assert not wrapper.last_track_persistence_stats['source_owners']
    assert wrapper.last_track_persistence_stats['expired_source_ids'] == [7]


def test_contradicted_anchor_releases_ownership():
    wrapper, base, env, track = _setup()
    _admit(wrapper, env, track)
    live = cc.TrackView(7, np.array([4., 8.]), np.zeros(2), 0.)
    base.result.tracks = [live]
    env.tracker.tracks = []
    env.raw_ranges[:] = cfg.LIDAR_RANGE - 1
    assert wrapper.snapshot(env).tracks == [live]
    assert not wrapper._source_owners


def test_both_options_disabled_match_v11_values_order_and_anchor_evolution():
    _, base, env, track = _legacy_setup()
    other_base, other_env = copy.deepcopy((base, env))
    other_track = other_env.tracker.tracks[0]
    old = TrackPersistencePerception(base)
    new = ConsistentTrackPerception(other_base, geometry_prior=False, source_ownership=False)
    for serial in range(1, 7):
        if serial == 4:
            base.result.tracks = [cc.TrackView(7, np.array([4., 8.]), np.zeros(2), 0.)]
            other_base.result.tracks = copy.deepcopy(base.result.tracks)
        a = _observe(old, env, track, serial)
        b = _observe(new, other_env, other_track, serial)
        assert [t.id for t in a.tracks] == [t.id for t in b.tracks]
        for x, y in zip(a.tracks, b.tracks):
            np.testing.assert_array_equal(x.position, y.position)
            np.testing.assert_array_equal(x.velocity, y.velocity)
            assert x.heading == y.heading
        np.testing.assert_array_equal(a.points, b.points)
        assert old.last_track_persistence_stats['updates'] == new.last_track_persistence_stats['updates']


def test_truth_guards_and_source_lists_remain_unmodified():
    wrapper, base, env, track = _setup()
    # The imported fixture raises if targets or true own position are accessed.
    _admit(wrapper, env, track)
    before = copy.deepcopy(track.__dict__)
    points = base.result.points.copy()
    env.pose_stale = True
    wrapper.snapshot(env)
    np.testing.assert_array_equal(track.position, before['position'])
    np.testing.assert_array_equal(track.velocity, before['velocity'])
    np.testing.assert_array_equal(base.result.points, points)
    assert not base.result.tracks and not env.tracks and not track.is_dynamic


def test_v13_constructor_preserves_inherited_optional_features():
    filt = SafetyFilterV13(target_turn_rate_deg_s=5., policy_prefix_search=True,
                          max_coast_s=4., geometry_prior=False)
    assert isinstance(filt.perception, ConsistentTrackPerception)
    assert filt.perception is filt.track_persistence
    assert not filt.perception.geometry_prior and filt.perception.source_ownership
    assert filt.max_coast_s == filt.perception.max_coast_s == 4.
    assert filt.policy_prefix_search and np.isclose(filt.target_turn_rate_rad_s, np.radians(5.))
    disabled = SafetyFilterV13(enable_admission=False, enable_persistence=False)
    assert disabled.track_persistence is None


def test_v13_filter_delegates_without_altering_parent_action(monkeypatch):
    expected = np.array([.2, -.4])
    def parent(self, env, action):
        self.last = {'why': 'parent'}
        return expected, True
    monkeypatch.setattr(safety_v11.SafetyFilterV11, '_filter', parent)
    filt = SafetyFilterV13()
    result, changed = filt._filter(SimpleNamespace(), np.zeros(2))
    assert result is expected and changed is True and filt.last['why'] == 'parent'
    assert filt.last['v13_geometry_prior'] and filt.last['v13_source_ownership']
