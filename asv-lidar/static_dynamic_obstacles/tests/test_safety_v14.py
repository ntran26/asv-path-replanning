"""Synthetic target-view selection only; no environment or policy episodes."""
import copy
from types import SimpleNamespace

import numpy as np

import constants as cfg
from classical import common as cc
from safety_consistent_tracks import ConsistentTrackPerception
from safety_v14 import PreferredPersistentPerception, SafetyFilterV14
import safety_v13
from test_safety_track_persistence import _setup as _legacy_setup, _observe, _admit, _points


def _setup(**kwargs):
    _, base, env, track = _legacy_setup()
    return PreferredPersistentPerception(base, **kwargs), base, env, track


def _view(track_id=7, context=None):
    return cc.TrackView(track_id, np.array([8., 9.]), np.zeros(2), 0., context)


def test_base_first_source_is_replaced_only_after_original_admission_gates_pass():
    wrapper, base, env, track = _setup()
    original = _view()
    base.result.tracks = [original]
    for serial in (1, 2):
        assert _observe(wrapper, env, track, serial).tracks == [original]
    result = _observe(wrapper, env, track, 3)
    assert len(result.tracks) == 1 and result.tracks[0].id < 0
    stats = wrapper.last_track_persistence_stats
    assert stats['first_admission_owners'] == {7: 'base'}
    assert stats['source_owners'] == {7: 'persistent'}
    assert stats['replaced_base_source_ids'] == [7]
    assert stats['replaced_base_view_count'] == 1
    assert stats['published_source_ids'] == [7]
    assert stats['published_persistent_count'] == stats['added_hypotheses'] == 1


def test_independently_admitted_source_keeps_preference_after_base_appears():
    wrapper, base, env, track = _setup()
    first = _admit(wrapper, env, track)
    base.result.tracks = [_view()]
    env.tracker.tracks = []
    assert [t.id for t in wrapper.snapshot(env).tracks] == [first.id]
    assert wrapper.last_track_persistence_stats['first_admission_owners'] == {7: 'persistent'}


def test_expiry_restores_base_view_and_removes_published_source():
    wrapper, base, env, track = _setup(max_coast_s=.5)
    _admit(wrapper, env, track)
    original = _view()
    base.result.tracks = [original]
    env.tracker.tracks = []
    assert wrapper.snapshot(env).tracks[0].id < 0
    assert wrapper.snapshot(env).tracks == [original]
    stats = wrapper.last_track_persistence_stats
    assert stats['expired_source_ids'] == [7]
    assert stats['published_source_ids'] == [] and stats['source_owners'] == {}


def test_finite_ray_contradiction_restores_base_view():
    wrapper, base, env, track = _setup()
    _admit(wrapper, env, track)
    original = _view()
    base.result.tracks = [original]
    env.tracker.tracks = []
    env.raw_ranges[:] = cfg.LIDAR_RANGE - 1
    assert wrapper.snapshot(env).tracks == [original]
    assert wrapper.last_track_persistence_stats['contradicted_source_ids'] == [7]


def test_unmatched_ids_and_contexts_preserved_without_spatial_merging():
    wrapper, base, env, track = _setup()
    first = _admit(wrapper, env, track)
    context = object()
    same_place_different_id = cc.TrackView(90, first.position.copy(), first.velocity.copy(), 0., context)
    original = _view()
    base.result.tracks = [same_place_different_id, original]
    env.tracker.tracks = []
    result = wrapper.snapshot(env)
    assert [t.id for t in result.tracks] == [90, first.id]
    assert result.tracks[0] is same_place_different_id and result.tracks[0].ctx is context
    assert base.result.tracks == [same_place_different_id, original]


def test_static_points_tracker_and_snapshot_lists_are_unchanged():
    wrapper, base, env, track = _setup()
    _admit(wrapper, env, track)
    original = _view()
    base.result.tracks = [original]
    before_points = base.result.points.copy()
    before_velocity = track.velocity.copy()
    result = wrapper.snapshot(env)
    assert result.points is base.result.points
    np.testing.assert_array_equal(result.points, before_points)
    np.testing.assert_array_equal(track.velocity, before_velocity)
    assert base.result.tracks == [original] and not env.tracks and not track.is_dynamic
    # Fixture forbids hidden target/true-ego access throughout the calls above.


def test_partial_shape_correction_is_still_inherited():
    wrapper, _, env, track = _setup()
    _admit(wrapper, env, track)
    centre = np.array([4.6, 8.])
    result = _observe(wrapper, env, track, 4, centre=centre+[0., .25],
                      points=_points(centre, width=.24))
    np.testing.assert_allclose(result.tracks[0].position, centre, atol=1e-12)
    assert wrapper.last_track_persistence_stats['updates'][0]['geometry_prior']['full_axes'] == [True, False]


def test_opt_out_matches_v13_values_order_and_anchor_evolution():
    _, base, env, track = _legacy_setup()
    other_base, other_env = copy.deepcopy((base, env))
    other_track = other_env.tracker.tracks[0]
    old = ConsistentTrackPerception(base)
    new = PreferredPersistentPerception(other_base, prefer_persistent_estimates=False)
    base.result.tracks = [_view()]
    other_base.result.tracks = copy.deepcopy(base.result.tracks)
    for serial in range(1, 7):
        if serial == 5:
            base.result.tracks = []
            other_base.result.tracks = []
        a = _observe(old, env, track, serial)
        b = _observe(new, other_env, other_track, serial)
        assert [t.id for t in a.tracks] == [t.id for t in b.tracks]
        for x, y in zip(a.tracks, b.tracks):
            np.testing.assert_array_equal(x.position, y.position)
            np.testing.assert_array_equal(x.velocity, y.velocity)
            assert x.heading == y.heading
        assert old.last_track_persistence_stats['updates'] == new.last_track_persistence_stats['updates']
        assert old.last_track_persistence_stats['source_owners'] == new.last_track_persistence_stats['source_owners']


def test_v14_constructor_preserves_all_inherited_options():
    filt = SafetyFilterV14(target_turn_rate_deg_s=5., policy_prefix_search=True,
                          max_coast_s=4., source_ownership=False)
    assert isinstance(filt.perception, PreferredPersistentPerception)
    assert filt.track_persistence is filt.perception
    assert filt.prefer_persistent_estimates and filt.perception.geometry_prior
    assert not filt.perception.source_ownership
    assert filt.policy_prefix_search and np.isclose(filt.target_turn_rate_rad_s, np.radians(5.))
    assert filt.perception.max_coast_s == 4.
    disabled = SafetyFilterV14(enable_admission=False, enable_persistence=False)
    assert disabled.track_persistence is None


def test_v14_filter_delegates_action_and_diagnostics_without_posthoc_changes(monkeypatch):
    expected = np.array([.25, -.3])
    def parent(self, env, action):
        self.last = {'why': 'parent'}
        return expected, True
    monkeypatch.setattr(safety_v13.SafetyFilterV13, '_filter', parent)
    filt = SafetyFilterV14(prefer_persistent_estimates=False)
    out, changed = filt._filter(SimpleNamespace(), np.zeros(2))
    assert out is expected and changed is True
    assert filt.last == {'why': 'parent', 'v14_prefer_persistent_estimates': False}
