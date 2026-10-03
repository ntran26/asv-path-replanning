"""Finite history from measured views; no simulator or policy episodes."""
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
from safety_track_history import TrackHistoryPerception


def setup():
    context = object()
    view = cc.TrackView(9, np.array([2., 9.]), np.array([.4, -.05]), 1.2, context)
    points = np.array([[3., 4.], [5., 6.]])
    snap = cc.Snapshot(0., 0., 0., 1., 0., 0., np.array([0., 1.]), np.array([1., 0.]),
                       np.zeros(2), 0., 0., 20., points, [view])
    base = SimpleNamespace(snapshot=lambda env: snap, last_memory_stats={"preserved": True})
    raw = SimpleNamespace(id=9, hits=int(cfg.TRACK_MIN_HITS), misses=0,
                          last_fit_centre=np.array([2.1, 9.1]), last_fit_heading_deg=90.,
                          last_evidence=SimpleNamespace(appear=int(cfg.MOTION_MIN_POINTS), vacate=0))
    env = SimpleNamespace(tracker=SimpleNamespace(tracks=[raw]), pose_stale=False)
    return TrackHistoryPerception(base), env, snap, raw, view


def extras(snap):
    return [t for t in snap.tracks if t.id < 0]


def test_exact_current_view_and_static_returns_preserved_with_independent_history():
    adapter, env, original, raw, view = setup()
    initial_position, initial_velocity = view.position.copy(), view.velocity.copy()
    first = adapter.snapshot(env)
    assert first is not original and first.tracks is not original.tracks
    assert first.tracks[0] is view and first.points is original.points
    assert first.tracks[0].ctx is view.ctx and not extras(first)
    view.position[:] = [20., 30.]
    view.velocity[:] = [-1., 0.]
    second = adapter.snapshot(env)
    history = extras(second)
    np.testing.assert_allclose(history[0].position, initial_position + cfg.UPDATE_RATE*initial_velocity)
    np.testing.assert_array_equal(history[0].velocity, initial_velocity)
    assert history[0].heading == 1.2 and history[0].ctx is None
    assert second.tracks[0] is view and second.points is original.points
    assert len(original.tracks) == 1
    assert adapter.last_memory_stats == {"preserved": True}


def test_propagation_expires_after_exact_age_even_if_raw_track_disappears():
    adapter, env, original, raw, view = setup()
    anchor_pos, anchor_vel = view.position.copy(), view.velocity.copy()
    adapter.snapshot(env)
    original.tracks = []
    env.tracker.tracks = []
    for age in range(1, int(cfg.TRACK_MAX_MISSES)+1):
        result = adapter.snapshot(env)
        assert len(result.tracks) == 1
        np.testing.assert_allclose(result.tracks[0].position, anchor_pos + age*cfg.UPDATE_RATE*anchor_vel)
        assert adapter.last_track_history_stats["hypotheses"][0]["age_decisions"] == age
    assert adapter.snapshot(env).tracks == []
    assert adapter._anchors == {}


def test_stale_pose_never_refreshes_and_existing_history_still_expires():
    adapter, env, original, raw, view = setup()
    adapter.snapshot(env)
    env.pose_stale = True
    for age in range(1, int(cfg.TRACK_MAX_MISSES)+2):
        result = adapter.snapshot(env)
        assert len(extras(result)) == (1 if age <= cfg.TRACK_MAX_MISSES else 0)
        assert adapter.last_track_history_stats["admitted_source_ids"] == []
        assert result.tracks[0] is view


def test_three_recent_anchors_and_unique_synthetic_ids_remain_bounded():
    adapter, env, original, raw, view = setup()
    ids_seen = set()
    first_id = None
    for frame in range(1, 10):
        result = adapter.snapshot(env)
        ghosts = extras(result)
        assert len(ghosts) == min(frame-1, cfg.TRACK_MAX_MISSES)
        assert len({t.id for t in ghosts}) == len(ghosts)
        assert adapter.last_track_history_stats["stored_anchor_count"] <= cfg.TRACK_MAX_MISSES
        if frame == 2:
            first_id = ghosts[0].id
        if frame > 1+cfg.TRACK_MAX_MISSES:
            assert first_id not in {t.id for t in ghosts}
        ids_seen.update(t.id for t in ghosts)
    assert len(ids_seen) == 8


@pytest.mark.parametrize("change,reason", [
    ({"last_fit_centre": None}, "missing_current_finite_fit"),
    ({"last_fit_centre": [np.nan, 1.]}, "missing_current_finite_fit"),
    ({"last_fit_heading_deg": None}, "missing_current_finite_fit"),
    ({"last_fit_heading_deg": np.inf}, "missing_current_finite_fit"),
    ({"misses": 1}, "missed_detection"),
    ({"hits": 0}, "unconfirmed"),
    ({"last_evidence": None}, "no_appearance_evidence"),
    ({"last_evidence": SimpleNamespace(appear=0, vacate=100)}, "no_appearance_evidence"),
    ({"last_evidence": SimpleNamespace(appear=1, vacate=0)}, "insufficient_motion_evidence"),
])
def test_missing_fit_or_motion_evidence_never_creates_anchor(change, reason):
    adapter, env, original, raw, view = setup()
    for name, value in change.items():
        setattr(raw, name, value)
    adapter.snapshot(env)
    assert adapter._anchors == {}
    assert adapter.last_track_history_stats["admission_rejections"] == {reason: 1}


def test_raw_unpublished_static_object_is_not_promoted_by_history():
    adapter, env, original, raw, view = setup()
    original.tracks = []  # Even a convincing raw fit/evidence is insufficient.
    assert adapter.snapshot(env).tracks == []
    assert adapter._anchors == {}


def test_synthetic_views_never_become_anchors():
    adapter, env, original, raw, view = setup()
    view.id = raw.id = -10
    adapter.snapshot(env)
    assert adapter._anchors == {}
    assert adapter.last_track_history_stats["admission_rejections"] == {"synthetic_or_duplicate_view": 1}


def test_held_or_coasted_motion_evidence_cannot_renew_anchor():
    adapter, env, original, raw, view = setup()
    adapter.snapshot(env)
    raw.misses = 1
    for _ in range(int(cfg.TRACK_MAX_MISSES)+1):
        last = adapter.snapshot(env)
    assert not extras(last) and not adapter._anchors


def test_invalid_admitted_velocity_is_rejected_without_modifying_current_view():
    adapter, env, original, raw, view = setup()
    view.velocity[:] = [np.nan, .1]
    result = adapter.snapshot(env)
    assert result.tracks[0] is view and np.isnan(view.velocity[0])
    assert not adapter._anchors
    assert adapter.last_track_history_stats["admission_rejections"] == {"invalid_admitted_state": 1}


@pytest.mark.parametrize("fitted_hulls", [False, True])
def test_v6_history_wraps_after_provisional_without_replacing_observer(monkeypatch, fitted_hulls):
    import safety_v4 as v4
    import safety_v6 as v6
    from safety_hull_perception import HullSafetyPerception
    from safety_provisional_tracks import ProvisionalTrackPerception
    monkeypatch.setattr(v4, "FREE_SPACE_MEMORY", False)
    monkeypatch.setattr(v4, "MODEL_EGO_OBSERVER", True)
    monkeypatch.setattr(v6, "CALIBRATED_BRAKING", False)
    monkeypatch.setattr(v6, "FITTED_HULL_PERCEPTION", fitted_hulls)
    monkeypatch.setattr(v6, "PROVISIONAL_TRACKS", True)
    monkeypatch.setattr(v6, "TRACK_HISTORY", True)
    candidate = v6.SafetyFilterV6()
    assert candidate.observer is not None
    assert isinstance(candidate.perception, TrackHistoryPerception)
    provisional = candidate.perception.base_perception
    assert isinstance(provisional, ProvisionalTrackPerception)
    expected_base = HullSafetyPerception if fitted_hulls else cc.Perception
    assert isinstance(provisional.base_perception, expected_base)
