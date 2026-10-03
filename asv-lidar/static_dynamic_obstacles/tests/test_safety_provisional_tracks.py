"""Synthetic onboard observations only; no policy/environment episodes."""
from collections import deque
from types import SimpleNamespace
import copy
import math

import numpy as np
import pytest

import constants as cfg
import tracking
from classical import common as cc
from safety_provisional_tracks import ProvisionalTrackPerception


def _track(track_id=1, *, heading=0.0, width=None, length=None):
    centre = np.array([2.0, 5.0])
    angle = math.radians(heading)
    forward = np.array([math.sin(angle), math.cos(angle)])
    right = np.array([forward[1], -forward[0]])
    length = cfg.LOA if length is None else length
    width = cfg.BREADTH if width is None else width
    longitudinal = np.linspace(-0.5 * length, 0.5 * length, 8)
    points = np.concatenate([centre + longitudinal[:, None] * forward + sign * 0.5 * width * right
                             for sign in (-1, 1)])
    track = tracking.Track(centre - 0.2 * forward)
    track.id = track_id
    track.state[2:] = 0.4 * forward
    track.hits = cfg.TRACK_MIN_HITS
    track.last_fit_centre = centre.copy()
    track.last_fit_heading_deg = heading
    track.history = deque([(1, points)], maxlen=cfg.MOTION_WINDOW_STEPS)
    return track


class _Base:
    def __init__(self):
        self.memory = [(1, np.array([[2.0, 5.0], [2.3, 5.0], [3.6, 5.0]]))]
        self.frames = 1
        self.last_memory_stats = {"cleared_points": 0}
        self.result = cc.Snapshot(0, 0, 0, 0.5, 0, 0,
                                  np.array([0, 1]), np.array([1, 0]), np.array([0, 0]),
                                  0, 0, 10, self.memory[0][1].copy(), [])
        self.calls = 0

    def snapshot(self, env):
        self.calls += 1
        return self.result


def _setup(track=None):
    track = _track() if track is None else track
    base = _Base()
    env = SimpleNamespace(tracks=[], tracker=SimpleNamespace(tracks=[track]),
                          pose_stale=False, encounter_contexts={})
    return ProvisionalTrackPerception(base), base, env, track


def test_confirmed_pre_dynamic_track_added_without_deleting_any_static_evidence():
    perception, base, env, track = _setup()
    before_points, before_memory = base.result.points.copy(), copy.deepcopy(base.memory)
    snap = perception.snapshot(env)
    assert len(snap.tracks) == 1
    np.testing.assert_array_equal(snap.tracks[0].position, track.last_fit_centre)
    assert snap.tracks[0].ctx is None
    assert snap.points is base.result.points
    np.testing.assert_array_equal(snap.points, before_points)
    np.testing.assert_array_equal(base.memory[0][1], before_memory[0][1])
    assert not base.result.tracks and not env.tracks and not track.is_dynamic
    assert perception.last_provisional_stats["current_fit_ids"] == [track.id]
    assert base.calls == 1


def test_no_policy_tracker_context_or_filter_mutation():
    perception, base, env, track = _setup()
    original = copy.deepcopy(track.__dict__)
    context = object()
    env.encounter_contexts[track.id] = context
    snap = perception.snapshot(env)
    snap.tracks[0].position[:] = 99
    snap.tracks[0].velocity[:] = 99
    for key in ("state", "cov", "fit_offset", "last_fit_centre", "_vel_noise"):
        np.testing.assert_array_equal(track.__dict__[key], original[key])
    np.testing.assert_array_equal(track.history[0][1], original["history"][0][1])
    assert env.encounter_contexts[track.id] is context
    assert (track.hits, track.misses, track.is_dynamic) == (cfg.TRACK_MIN_HITS, 0, False)
    assert perception.memory is base.memory
    assert perception.last_memory_stats is base.last_memory_stats


@pytest.mark.parametrize("invalid,reason", [
    ("young", "unconfirmed"), ("slow", "low_speed"), ("missed", "missed_detection"),
    ("no_centre", "invalid_fit"), ("nan_centre", "invalid_fit"),
    ("no_axis", "invalid_fit"), ("nan_axis", "invalid_fit"),
    ("no_history", "missing_cluster"), ("few_points", "invalid_cluster"),
    ("nan_points", "invalid_cluster"), ("stale", "pose_stale")])
def test_unreliable_current_track_is_not_admitted(invalid, reason):
    perception, _, env, track = _setup()
    if invalid == "young": track.hits = cfg.TRACK_MIN_HITS - 1
    elif invalid == "slow": track.state[2:] = [0, cfg.TRACK_FIT_PRIOR_SPEED]
    elif invalid == "missed": track.misses = 1
    elif invalid == "no_centre": track.last_fit_centre = None
    elif invalid == "nan_centre": track.last_fit_centre = np.array([np.nan, 5])
    elif invalid == "no_axis": track.last_fit_heading_deg = None
    elif invalid == "nan_axis": track.last_fit_heading_deg = np.nan
    elif invalid == "no_history": track.history.clear()
    elif invalid == "few_points": track.history[-1] = (1, track.history[-1][1][:cfg.TRACK_FIT_MIN_POINTS - 1])
    elif invalid == "nan_points": track.history[-1][1][0, 0] = np.nan
    else: env.pose_stale = True
    assert perception.snapshot(env).tracks == []
    assert perception.last_provisional_stats["rejections"][reason] == 1


@pytest.mark.parametrize("heading", [0, 37, 90, 180])
@pytest.mark.parametrize("axis", ["length", "width"])
def test_oversized_wall_cluster_rejected_in_fitted_axes(heading, axis):
    limit = (cfg.LOA if axis == "length" else cfg.BREADTH) + cfg.TRACK_FIT_FULL_EXTENT_TOL_M
    track = _track(heading=heading, **{axis: limit + 0.01})
    perception, _, env, _ = _setup(track)
    assert perception.snapshot(env).tracks == []
    assert perception.last_provisional_stats["rejections"] == {"oversized_cluster": 1}


@pytest.mark.parametrize("stale", [False, True])
def test_admitted_fit_coasts_without_refresh_then_expires(stale):
    perception, _, env, track = _setup()
    first = perception.snapshot(env).tracks[0]
    env.pose_stale = stale
    if not stale:
        track.last_fit_centre = track.last_fit_heading_deg = None
        track.misses = 1
    for age in range(1, cfg.TRACK_MAX_MISSES + 1):
        snap = perception.snapshot(env)
        np.testing.assert_allclose(snap.tracks[0].position,
                                   first.position + age * cfg.UPDATE_RATE * first.velocity)
        assert perception.last_provisional_stats["coasted_ids"] == [track.id]
    assert not perception.snapshot(env).tracks


def test_published_track_never_duplicated_then_retained_after_demotion_without_speed_gate():
    perception, base, env, track = _setup()
    context = object()
    view = cc.TrackView(track.id, track.last_fit_centre.copy(), track.velocity.copy(), 0, context)
    base.result.tracks = [view]
    env.tracks = [track]
    track.is_dynamic = True
    first = perception.snapshot(env)
    assert len(first.tracks) == 1 and first.tracks[0] is view
    assert first.tracks is not base.result.tracks
    assert not perception.last_provisional_stats["added_ids"]
    base.result.tracks = []
    env.tracks = []
    track.is_dynamic = False
    track.state[2:] = 0
    track.last_fit_centre = track.last_fit_heading_deg = None
    # Dynamic demotion does not delete a still-live safety hypothesis.
    for _ in range(cfg.TRACK_MAX_MISSES + 2):
        track.state[:2] += [0.1, 0]
        snap = perception.snapshot(env)
        np.testing.assert_allclose(snap.tracks[0].position, track.position + [0, 0.2])
        assert snap.tracks[0].ctx is None
        assert perception.last_provisional_stats["retained_dynamic_ids"] == [track.id]
    env.tracker.tracks = []
    assert not perception.snapshot(env).tracks
    assert not perception._hypotheses


def test_retained_dynamic_raw_prediction_not_integrated_twice():
    perception, base, env, track = _setup()
    base.result.tracks = [cc.TrackView(track.id, track.last_fit_centre.copy(), track.velocity.copy(), 0)]
    start = perception.snapshot(env).tracks[0].position.copy()
    base.result.tracks = []
    track.last_fit_centre = track.last_fit_heading_deg = None
    for misses in range(1, cfg.TRACK_MAX_MISSES + 1):
        track.predict(cfg.UPDATE_RATE)
        track.misses = misses
        snap = perception.snapshot(env)
        np.testing.assert_allclose(snap.tracks[0].position, start + misses * cfg.UPDATE_RATE * track.velocity)
    track.misses += 1
    assert not perception.snapshot(env).tracks


def test_stale_pose_does_not_refresh_previously_published_fit():
    perception, base, env, track = _setup()
    base.result.tracks = [cc.TrackView(track.id, track.last_fit_centre.copy(), track.velocity.copy(), 0)]
    first = perception.snapshot(env).tracks[0]
    base.result.tracks = []
    env.pose_stale = True
    for age in range(1, cfg.TRACK_MAX_MISSES + 1):
        np.testing.assert_allclose(perception.snapshot(env).tracks[0].position,
                                   first.position + age * cfg.UPDATE_RATE * first.velocity)
    assert not perception.snapshot(env).tracks


def test_missing_raw_tracker_leaves_base_snapshot_evidence_unchanged():
    perception, base, env, _ = _setup()
    del env.tracker
    result = perception.snapshot(env)
    assert result.points is base.result.points and result.tracks == []


def test_no_simulator_truth_access():
    perception, _, env, _ = _setup()

    class OnboardOnly:
        def __getattr__(self, name):
            assert name in {"tracker", "pose_stale"}, f"unexpected environment read: {name}"
            return getattr(env, name)

    assert len(perception.snapshot(OnboardOnly()).tracks) == 1


@pytest.mark.parametrize("count", [None, 0, cfg.MOTION_MIN_POINTS - 1, cfg.MOTION_MIN_POINTS])
def test_opt_in_motion_gate_requires_existing_minimum_count_for_new_id(count):
    _, base, env, track = _setup()
    track.last_evidence = (None if count is None else
                           tracking.MotionEvidence(count, 0, 1000))
    perception = ProvisionalTrackPerception(base, require_motion_evidence=True)
    snap = perception.snapshot(env)
    admitted = count is not None and count >= cfg.MOTION_MIN_POINTS
    assert bool(snap.tracks) is admitted
    if admitted:
        # A fraction below publication's requirement is intentionally allowed.
        assert not track.last_evidence.moving
        assert perception.last_provisional_stats["new_admissions"] == [{
            "id": track.id, "frame": 1,
            "motion_evidence": {"appear": count, "vacate": 0,
                                "compared": 1000, "violations": count}}]
    else:
        assert perception.last_provisional_stats["rejections"] == {"insufficient_motion_evidence": 1}
        assert perception.last_provisional_stats["new_admissions"] == []


@pytest.mark.parametrize("stale", [False, True])
def test_old_positive_evidence_cannot_admit_on_stale_or_missed_detection(stale):
    _, base, env, track = _setup()
    track.last_evidence = tracking.MotionEvidence(cfg.MOTION_MIN_POINTS, 0, 100)
    env.pose_stale = stale
    track.misses = 0 if stale else 1
    perception = ProvisionalTrackPerception(base, require_motion_evidence=True)
    assert not perception.snapshot(env).tracks
    assert perception.last_provisional_stats["new_admissions"] == []


def test_motion_evidence_is_for_new_admission_not_every_current_fit_refresh():
    _, base, env, track = _setup()
    perception = ProvisionalTrackPerception(base, require_motion_evidence=True)
    track.last_evidence = tracking.MotionEvidence(0, 0, 100)
    assert not perception.snapshot(env).tracks
    track.last_evidence = tracking.MotionEvidence(1, cfg.MOTION_MIN_POINTS - 1, 100)
    assert perception.snapshot(env).tracks
    assert perception.last_provisional_stats["new_admissions"][0]["frame"] == 2
    track.last_evidence = tracking.MotionEvidence(0, 0, 100)
    for _ in range(cfg.TRACK_MAX_MISSES + 2):
        track.last_fit_centre += [0, 0.1]
        snap = perception.snapshot(env)
        np.testing.assert_array_equal(snap.tracks[0].position, track.last_fit_centre)
        assert perception.last_provisional_stats["new_admissions"] == []


def test_published_dynamic_demotion_bypasses_optional_initial_evidence_gate():
    _, base, env, track = _setup()
    track.last_evidence = tracking.MotionEvidence(0, 0, 100)
    base.result.tracks = [cc.TrackView(track.id, track.last_fit_centre.copy(), track.velocity.copy(), 0)]
    perception = ProvisionalTrackPerception(base, require_motion_evidence=True)
    perception.snapshot(env)
    base.result.tracks = []
    track.last_fit_centre = track.last_fit_heading_deg = None
    track.state[2:] = 0
    assert len(perception.snapshot(env).tracks) == 1
    assert perception.last_provisional_stats["retained_dynamic_ids"] == [track.id]
    assert perception.last_provisional_stats["new_admissions"] == []


def test_dropped_id_loses_admission_and_must_supply_fresh_evidence_again():
    _, base, env, track = _setup()
    perception = ProvisionalTrackPerception(base, require_motion_evidence=True)
    track.last_evidence = tracking.MotionEvidence(cfg.MOTION_MIN_POINTS, 0, 100)
    assert perception.snapshot(env).tracks
    env.tracker.tracks = []
    perception.snapshot(env)
    env.tracker.tracks = [track]
    track.last_evidence = tracking.MotionEvidence(0, 0, 100)
    assert not perception.snapshot(env).tracks
