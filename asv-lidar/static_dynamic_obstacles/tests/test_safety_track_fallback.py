"""Synthetic snapshot selection and constructor checks; no policy episodes."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from classical import common as cc
from safety_track_fallback import TrackFallbackPerception
from safety_v12 import SafetyFilterV12
import safety_v11


def _view(track_id, position=(4., 8.), context=None):
    return cc.TrackView(track_id, np.array(position), np.array([.4, 0.]), 0., context)


class _Persistence:
    def __init__(self, tracks, hypotheses):
        self.calls = 0
        self.memory = [(1, np.array([[1., 2.], [3., 4.]]))]
        self._anchors = {h['source_id']: {'frame': 3} for h in hypotheses}
        self.last_track_persistence_stats = {
            'frame': 8, 'updates': [{'source_id': 4, 'kind': 'measured_refresh'}],
            'hypotheses': hypotheses, 'added_hypotheses': len(hypotheses),
            'expired_source_ids': [], 'contradicted_source_ids': [],
        }
        self.result = cc.Snapshot(0., 0., 0., .5, 0., 0., np.array([0., 1.]),
                                  np.array([1., 0.]), np.zeros(2), 0., 0., 20.,
                                  self.memory[0][1].copy(), tracks)

    def snapshot(self, env):
        self.calls += 1
        return self.result


def _hypothesis(source, synthetic, age=.5):
    return dict(source_id=source, id=synthetic, anchor_frame=3, age_s=age,
                position=[4., 8.], velocity=[.4, 0.])


def test_exact_source_live_view_suppresses_only_its_synthetic_alternative():
    context = object()
    live = _view(4, context=context)
    base = _Persistence([live, _view(-1000000), _view(-1000001)],
                        [_hypothesis(4, -1000000), _hypothesis(9, -1000001)])
    wrapper = TrackFallbackPerception(base)
    result = wrapper.snapshot(object())
    assert [t.id for t in result.tracks] == [4, -1000001]
    assert result.tracks[0] is live and result.tracks[0].ctx is context
    assert result.points is base.result.points
    assert base.calls == 1
    stats = wrapper.last_track_persistence_stats
    assert stats['base_track_ids'] == [4]
    assert stats['suppressed_source_ids'] == [4]
    assert stats['generated_hypotheses'] == 2 and stats['added_hypotheses'] == 1
    assert len(stats['hypotheses']) == stats['added_hypotheses']


def test_disabled_priority_returns_underlying_snapshot_and_order_unchanged():
    base = _Persistence([_view(4), _view(-1000000)], [_hypothesis(4, -1000000)])
    wrapper = TrackFallbackPerception(base, existing_track_priority=False)
    assert wrapper.snapshot(None) is base.result
    assert wrapper.last_track_persistence_stats['suppressed_source_ids'] == []
    assert wrapper.last_track_persistence_stats['added_hypotheses'] == 1


def test_new_id_at_same_position_does_not_trigger_spatial_association():
    base = _Persistence([_view(99), _view(-1000000)], [_hypothesis(4, -1000000)])
    wrapper = TrackFallbackPerception(base)
    assert wrapper.snapshot(None) is base.result
    assert wrapper.last_track_persistence_stats['suppressed_source_ids'] == []


def test_suppression_keeps_cached_anchor_available_when_source_disappears():
    base = _Persistence([_view(4), _view(-1000000)], [_hypothesis(4, -1000000)])
    original = copy.deepcopy(base._anchors)
    wrapper = TrackFallbackPerception(base)
    assert [t.id for t in wrapper.snapshot(None).tracks] == [4]
    assert base._anchors == original
    base.result = replace(base.result, tracks=[base.result.tracks[1]])
    assert [t.id for t in wrapper.snapshot(None).tracks] == [-1000000]
    assert base._anchors == original


def test_generated_diagnostics_and_snapshot_not_mutated():
    base = _Persistence([_view(4), _view(-1000000)], [_hypothesis(4, -1000000)])
    before = copy.deepcopy(base.last_track_persistence_stats)
    points, memory = base.result.points.copy(), base.memory[0][1].copy()
    wrapper = TrackFallbackPerception(base)
    wrapper.snapshot(None)
    wrapper.last_track_persistence_stats['suppressed_hypotheses'][0]['position'][0] = 999
    assert base.last_track_persistence_stats == before
    assert len(base.result.tracks) == 2
    np.testing.assert_array_equal(base.result.points, points)
    np.testing.assert_array_equal(base.memory[0][1], memory)


def test_empty_hypotheses_leave_base_views_unchanged():
    base = _Persistence([_view(4)], [])
    wrapper = TrackFallbackPerception(base)
    assert wrapper.snapshot(None) is base.result
    assert wrapper.last_track_persistence_stats['base_track_ids'] == [4]


def test_fresh_and_coasted_alternatives_are_both_suppressed_for_live_source():
    base = _Persistence([_view(4), _view(-1000000), _view(-1000001)],
                        [_hypothesis(4, -1000000, 0.), _hypothesis(4, -1000001, 4.)])
    wrapper = TrackFallbackPerception(base)
    assert [t.id for t in wrapper.snapshot(None).tracks] == [4]
    assert len(wrapper.last_track_persistence_stats['suppressed_hypotheses']) == 2


def test_negative_base_ids_are_preserved_and_not_assumed_synthetic():
    base = _Persistence([_view(-2), _view(-1000000)], [_hypothesis(4, -1000000)])
    assert TrackFallbackPerception(base).snapshot(None) is base.result


def test_deepcopy_and_attribute_delegation_safe():
    base = _Persistence([], [])
    wrapper = TrackFallbackPerception(base)
    cloned = copy.deepcopy(wrapper)
    assert wrapper.memory is base.memory
    assert cloned.base_perception is not base


def test_v12_installs_fallback_and_preserves_optional_target_envelope_options():
    filt = SafetyFilterV12(target_turn_rate_deg_s=5., policy_prefix_search=True)
    assert isinstance(filt.perception, TrackFallbackPerception)
    assert filt.track_persistence is filt.perception
    assert filt.existing_track_priority is True
    assert filt.policy_prefix_search is True
    assert np.isclose(filt.target_turn_rate_rad_s, np.radians(5.))


def test_v12_respects_disabled_perception_and_priority_options():
    disabled = SafetyFilterV12(enable_admission=False, enable_persistence=False)
    assert disabled.track_persistence is None
    parity = SafetyFilterV12(existing_track_priority=False)
    assert not parity.perception.existing_track_priority


def test_v12_filter_delegates_without_altering_action_or_parent_diagnostics(monkeypatch):
    expected = np.array([.4, -.2], dtype=np.float32)
    called = []
    def parent(self, env, action):
        called.append((env, action))
        self.last = {'why': 'unchanged parent', 'track_persistence': {'added_hypotheses': 0}}
        return expected, True
    monkeypatch.setattr(safety_v11.SafetyFilterV11, '_filter', parent)
    filt = SafetyFilterV12()
    env, action = SimpleNamespace(), np.zeros(2)
    out, changed = filt._filter(env, action)
    assert out is expected and changed is True
    assert called == [(env, action)]
    assert filt.last['why'] == 'unchanged parent'
    assert filt.last['v12_existing_track_priority'] is True
