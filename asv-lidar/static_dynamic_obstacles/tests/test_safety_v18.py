"""Synthetic same-source heading and controller-integration checks; no episodes."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
from safety_heading_memory import HeadingMemoryPerception
import safety_v16
import safety_v18
from safety_v18 import SafetyFilterV18
from safety_observer import EgoObserver


def track(source=21, heading=114., serial=1, *, age=10, hits=10, misses=0):
    angle = math.radians(114. if heading is None else heading)
    forward = np.array([math.sin(angle), math.cos(angle)])
    right = np.array([forward[1], -forward[0]])
    centre = np.array([4., 7.])
    points = np.array([centre+a*forward+b*right
                       for a in np.linspace(-.7, .7, 4) for b in (-.2, .2)])
    return SimpleNamespace(id=source, age=age, hits=hits, misses=misses,
                           last_fit_heading_deg=heading,
                           last_fit_centre=centre.copy() if heading is not None else None,
                           history=[(serial, points)], position=centre.copy(),
                           velocity=np.array([.1, -.1]))


def snapshot(views):
    return cc.Snapshot(1., 2., 0., .5, 0., 0., np.array([0., 1.]),
                       np.array([1., 0.]), np.zeros(2), 0., 0., 20.,
                       np.array([[8., 9.]]), views)


class Base:
    def __init__(self):
        self.calls = 0
        self.value = snapshot([cc.TrackView(21, np.array([4., 7.]),
                                               np.array([.1, -.1]), math.radians(114.), object())])
        self.last_motion_axis_stats = {}

    def snapshot(self, env):
        self.calls += 1
        return self.value


def frame(adapter, base, raw, *, heading=None, stale=False):
    if heading is not None:
        old = base.value.tracks[0]
        base.value = snapshot([cc.TrackView(old.id, old.position, old.velocity, heading, old.ctx)]
                              + base.value.tracks[1:])
    env = SimpleNamespace(pose_stale=stale, tracker=SimpleNamespace(tracks=raw))
    return adapter.snapshot(env)


def primed():
    base = Base()
    adapter = HeadingMemoryPerception(base)
    original = base.value
    assert frame(adapter, base, [track()]) is original
    return adapter, base


@pytest.mark.parametrize("velocity", [[0., 0.], [.1, -.15], [-1., 2.]])
def test_missing_fit_holds_shape_independently_of_course_without_mutation(velocity):
    adapter, base = primed()
    raw = track(heading=None, serial=2, age=11, hits=11)
    raw.velocity = np.asarray(velocity)
    old = base.value.tracks[0]
    base.value = snapshot([cc.TrackView(21, old.position, raw.velocity,
                                      math.radians(147.562), old.ctx)])
    original = base.value
    before = copy.deepcopy(vars(raw))
    result = frame(adapter, base, [raw])
    assert base.calls == 2 and result is not original
    assert result.tracks[0].heading == math.radians(114.)
    assert original.tracks[0].heading == math.radians(147.562)
    for key in ("points", "edges_a", "edges_b", "tangent", "right", "centre"):
        assert getattr(result, key) is getattr(original, key)
    assert result.tracks[0].position is old.position
    assert result.tracks[0].velocity is raw.velocity
    assert result.tracks[0].ctx is old.ctx
    np.testing.assert_array_equal(raw.history[0][1], before["history"][0][1])
    assert raw.last_fit_heading_deg is None
    assert adapter.last_heading_memory_stats["replacements"][0]["fit_age_steps"] == 1


def test_current_valid_turn_updates_memory_and_is_not_overridden():
    adapter, base = primed()
    result = frame(adapter, base, [track(heading=186., serial=2, age=11)], heading=math.radians(186.))
    assert result is base.value
    assert adapter.last_heading_memory_stats["updated_source_ids"] == [21]
    result = frame(adapter, base, [track(heading=None, serial=3, age=12)], heading=0.)
    assert result.tracks[0].heading == math.radians(186.)


def test_age_limit_counts_decisions_not_scan_serial_arithmetic(monkeypatch):
    monkeypatch.setattr(cfg, "TRACK_MAX_MISSES", 3)
    adapter, base = primed()
    for i in range(1, 5):
        result = frame(adapter, base, [track(heading=None, serial=900*i, age=10+i)], heading=.2)
        assert (result.tracks[0].heading == math.radians(114.)) is (i <= 3)
    assert adapter.last_heading_memory_stats["expired_source_ids"] == [21]


@pytest.mark.parametrize("stale,misses", [(True, 0), (False, 1)])
def test_stale_or_missed_detection_neither_refreshes_nor_applies(stale, misses):
    adapter, base = primed()
    original = base.value
    assert frame(adapter, base, [track(heading=None, serial=2, age=11, misses=misses)], stale=stale) is original
    assert adapter._headings[21].frame == 1
    assert adapter.last_heading_memory_stats["replaced_source_ids"] == []


def test_repeated_cluster_cannot_refresh_or_apply_a_heading():
    adapter, base = primed()
    result = frame(adapter, base, [track(heading=None, serial=1, age=11)], heading=.1)
    assert result is base.value
    assert adapter._headings[21].frame == 1
    assert adapter.last_heading_memory_stats["rejections"] == {"repeated_observation": 1}


def test_missing_id_forgets_memory_before_same_id_returns():
    adapter, base = primed()
    frame(adapter, base, [])
    assert not adapter._headings and not adapter._seen
    result = frame(adapter, base, [track(heading=None, serial=9)], heading=.1)
    assert result is base.value


@pytest.mark.parametrize("age,hits", [(1, 10), (10, 2)])
def test_detectable_id_reuse_discards_old_axis(age, hits):
    adapter, base = primed()
    result = frame(adapter, base, [track(heading=None, serial=2, age=age, hits=hits)], heading=.1)
    assert result is base.value
    assert adapter.last_heading_memory_stats["identity_reset_source_ids"] == [21]
    assert 21 not in adapter._headings


def test_different_id_and_negative_persistent_view_are_untouched():
    adapter, base = primed()
    synthetic = cc.TrackView(-1000000, np.ones(2), np.zeros(2), .7)
    other = cc.TrackView(22, np.ones(2), np.zeros(2), .8)
    base.value = snapshot([synthetic, other])
    result = frame(adapter, base, [track(heading=None, serial=2), track(22, None, 2)])
    assert result is base.value
    assert result.tracks == [synthetic, other]


def test_current_motion_axis_correction_has_priority():
    adapter, base = primed()
    base.last_motion_axis_stats = {"replaced_source_ids": [21]}
    result = frame(adapter, base, [track(heading=None, serial=2)], heading=.3)
    assert result is base.value
    assert result.tracks[0].heading == .3
    assert adapter.last_heading_memory_stats["rejections"] == {"current_motion_axis_geometry": 1}


def test_finite_but_oversized_fit_is_not_treated_as_missing():
    adapter, base = primed()
    raw = track(heading=90., serial=2)
    raw.history[0][1][:] *= 10.
    result = frame(adapter, base, [raw], heading=math.pi/2)
    assert result is base.value
    assert 21 not in adapter._headings
    assert adapter.last_heading_memory_stats["rejections"] == {"oversized_cluster": 1}
    result = frame(adapter, base, [track(heading=None, serial=3)], heading=.1)
    assert result is base.value


def test_source_zero_is_a_valid_same_source_identity():
    base = Base()
    base.value.tracks[0].id = 0
    adapter = HeadingMemoryPerception(base)
    frame(adapter, base, [track(0)])
    result = frame(adapter, base, [track(0, None, 2)], heading=.1)
    assert result.tracks[0].heading == math.radians(114.)
    assert adapter.last_heading_memory_stats["replaced_source_ids"] == [0]


@pytest.mark.parametrize("duplicate_raw", [True, False])
def test_ambiguous_duplicate_source_identity_is_rejected(duplicate_raw):
    adapter, base = primed()
    raw = [track(heading=None, serial=2)]
    if duplicate_raw:
        raw.append(track(heading=None, serial=3))
    else:
        base.value.tracks.append(base.value.tracks[0])
    with pytest.raises(ValueError, match="Duplicate source ID"):
        frame(adapter, base, raw)


@pytest.mark.parametrize("bad", ["unconfirmed", "invalid_cluster", "missing_cluster"])
def test_invalid_evidence_does_not_seed_or_apply_memory(bad):
    adapter, base = primed()
    raw = track(heading=None, serial=2)
    if bad == "unconfirmed":
        raw.hits = 1
    elif bad == "invalid_cluster":
        raw.history[0][1][0, 0] = np.nan
    else:
        raw.history = []
    assert frame(adapter, base, [raw], heading=.3) is base.value
    assert not adapter.last_heading_memory_stats["replaced_source_ids"]


def test_no_hidden_state_access_and_exact_single_base_call():
    class OnboardOnly:
        pose_stale = False
        tracker = SimpleNamespace(tracks=[track()])
        def __getattr__(self, name):
            raise AssertionError("Unexpected environment read: " + name)
    base = Base()
    adapter = HeadingMemoryPerception(base)
    assert adapter.snapshot(OnboardOnly()) is base.value and base.calls == 1
    assert copy.deepcopy(adapter)._frame == 1


def test_constructor_preserves_v16_observer_and_options():
    enabled = SafetyFilterV18(motion_axis_geometry=False, conditional_prefix=False)
    assert type(enabled.observer) is EgoObserver
    assert isinstance(enabled.perception, HeadingMemoryPerception)
    assert not enabled.motion_axis_geometry and not enabled.conditional_prefix
    assert enabled.track_persistence is not None


def test_disabled_option_keeps_exact_parent_perception_object(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(safety_v16.SafetyFilterV16, "__init__", lambda self, **kw: setattr(self, "perception", sentinel))
    disabled = SafetyFilterV18(hull_heading_memory=False)
    assert disabled.perception is sentinel and disabled.heading_memory_perception is None


def test_module_default_is_constructor_local_and_explicit_option_wins(monkeypatch):
    monkeypatch.setattr(safety_v18, "HULL_HEADING_MEMORY", False)
    assert not SafetyFilterV18().hull_heading_memory
    assert SafetyFilterV18(hull_heading_memory=True).hull_heading_memory


@pytest.mark.parametrize("enabled", [True, False])
def test_controller_delegates_action_state_and_detaches_diagnostics(monkeypatch, enabled):
    result = (np.array([.2, -.1]), True)
    def parent(self, env, action):
        self.last = {"why": "parent", "checked_clearance": .123}
        self.plan = np.array([[.2, -.1]])
        return result
    monkeypatch.setattr(safety_v16.SafetyFilterV16, "_filter", parent)
    filt = SafetyFilterV18(hull_heading_memory=enabled)
    if enabled:
        filt.heading_memory_perception.last_heading_memory_stats = {"replaced_source_ids": [21]}
    assert filt._filter(object(), np.zeros(2)) is result
    assert filt.last["why"] == "parent" and filt.last["checked_clearance"] == .123
    assert filt.last["v18_hull_heading_memory"] is enabled
    np.testing.assert_array_equal(filt.plan, [[.2, -.1]])
    if enabled:
        filt.heading_memory_perception.last_heading_memory_stats["replaced_source_ids"].append(22)
        assert filt.last["hull_heading_memory"]["replaced_source_ids"] == [21]
    else:
        assert filt.last["hull_heading_memory"] == {}
