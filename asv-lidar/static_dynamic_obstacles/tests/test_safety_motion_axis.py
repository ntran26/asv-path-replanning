"""Saved observations and synthetic views only; no environment or plant calls."""
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
from safety_motion_axis import MotionAxisPerception
from safety_v16 import SafetyFilterV16
import safety_v15


class Base:
    def __init__(self):
        self.calls = 0
        self.result = cc.Snapshot(0., 0., 0., .5, 0., 0., np.array([0., 1.]),
            np.array([1., 0.]), np.zeros(2), 0., 0., 20., np.array([[5., 5.]]), [])

    def snapshot(self, env):
        self.calls += 1
        return self.result


class Onboard(SimpleNamespace):
    @property
    def targets(self): raise AssertionError('hidden targets')
    @property
    def asv_x(self): raise AssertionError('true pose')
    @property
    def raw_ranges(self): raise AssertionError('extra sensor access')


def setup():
    base = Base()
    track = SimpleNamespace(id=7, hits=4, confirmed=True, misses=0, velocity=np.array([0., .5]),
        position=np.array([0., 3.]), cov=np.eye(4), history=deque(),
        last_evidence=SimpleNamespace(violations=5, appear=5))
    env = Onboard(pose_stale=False, tracker=SimpleNamespace(tracks=[track], _scans=deque()))
    context = object()
    view = cc.TrackView(7, np.array([0., 3.]), track.velocity.copy(), .4, context)
    base.result.tracks = [view]
    return MotionAxisPerception(base), base, env, track, view


def observe(adapter, env, track, serial, centre_y, *, width=None, length=.2, count=8, origin=None):
    width = cfg.BREADTH if width is None else width
    points = np.column_stack((np.linspace(-width/2, width/2, count),
                             centre_y + np.linspace(-length/2, length/2, count)))
    track.history.append((serial, points))
    env.tracker._scans.append(SimpleNamespace(serial=serial, origin=np.array([0., 0.] if origin is None else origin)))
    return adapter.snapshot(env)


def admit(adapter, env, track):
    for i in range(3):
        result = observe(adapter, env, track, 1000+17*i, 3.+.25*i)
    return result


def test_fresh_beam_motion_corrects_only_geometry_and_preserves_inputs():
    adapter, base, env, track, view = setup()
    original = copy.deepcopy(track)
    result = admit(adapter, env, track)
    assert base.calls == 3 and adapter.last_motion_axis_stats['replaced_source_ids'] == [7]
    corrected = result.tracks[0]
    np.testing.assert_allclose(corrected.position, [0., 3.4+cfg.LOA/2])
    assert corrected.heading == 0 and corrected.id == 7 and corrected.ctx is view.ctx
    assert corrected.velocity is view.velocity and result.points is base.result.points
    assert base.result.tracks == [view]
    np.testing.assert_array_equal(track.position, original.position)
    np.testing.assert_array_equal(track.velocity, original.velocity)
    assert adapter.last_motion_axis_stats['updates'][0]['observation_times_s'] == [.5, 1., 1.5]
    assert adapter.last_motion_axis_stats['updates'][0]['scan_serials'] == [1000, 1017, 1034]


@pytest.mark.parametrize('problem', ['stationary', 'low_speed', 'no_evidence', 'vacate_only',
    'unconfirmed', 'missed', 'pose_stale', 'bad_cov', 'partial_beam', 'oversized_beam',
    'full_length', 'dead_zone', 'inconsistent_motion', 'too_few_points'])
def test_rejects_missing_static_or_incompatible_evidence(problem):
    adapter, base, env, track, view = setup()
    if problem == 'low_speed': track.velocity[:] = 0.
    if problem == 'no_evidence': track.last_evidence = None
    if problem == 'vacate_only': track.last_evidence.appear = 0
    if problem == 'unconfirmed': track.confirmed = False
    if problem == 'missed': track.misses = 1
    if problem == 'pose_stale': env.pose_stale = True
    if problem == 'bad_cov': track.cov[0, 0] = np.nan
    for i in range(4):
        y = 3. if problem == 'stationary' else 3.+.25*i
        if problem == 'inconsistent_motion': y += .4*(i % 2)
        kwargs = {'width': .1} if problem == 'partial_beam' else {}
        if problem == 'oversized_beam': kwargs['width'] = cfg.BREADTH + .2
        if problem == 'full_length': kwargs['length'] = cfg.LOA
        if problem == 'dead_zone': kwargs['origin'] = [0., y-.2]
        if problem == 'too_few_points': kwargs['count'] = cfg.TRACK_FIT_MIN_POINTS-1
        assert observe(adapter, env, track, i+1, y, **kwargs) is base.result
    assert adapter.last_motion_axis_stats['replaced_source_ids'] == []


def test_partial_older_clusters_can_support_motion_but_current_requires_fit_count():
    adapter, _, env, track, _ = setup()
    observe(adapter, env, track, 1, 3., count=4)
    observe(adapter, env, track, 2, 3.25, count=4)
    result = observe(adapter, env, track, 3, 3.5, count=8)
    assert result.tracks[0].heading == 0.
    assert adapter.last_motion_axis_stats['updates'][0]['point_counts'] == [4, 4, 8]


def test_unmatched_and_synthetic_views_are_identical_and_not_spatially_merged():
    adapter, base, env, track, view = setup()
    unrelated = cc.TrackView(8, view.position.copy(), np.zeros(2), 1., object())
    persistent = cc.TrackView(-1000001, view.position.copy(), np.zeros(2), 2., object())
    base.result.tracks = [persistent, view, unrelated]
    result = admit(adapter, env, track)
    assert result.tracks[0] is persistent and result.tracks[2] is unrelated
    base.result.tracks = [persistent, unrelated]
    assert observe(adapter, env, track, 1040, 3.75) is base.result


def test_no_backfill_and_no_refresh_from_repeated_or_old_scan():
    adapter, base, env, track, _ = setup()
    for i in range(3):
        # This history predates the adapter's first call and must not count.
        track.history.append((i, np.array([[-.25, 3.], [.25, 3.1], [-.25, 3.2], [.25, 3.3]])))
    observe(adapter, env, track, 5000, 3.5)
    assert len(adapter._observations[7]) == 1
    for _ in range(3): assert adapter.snapshot(env) is base.result
    assert adapter.last_motion_axis_stats['rejections'] == {'repeated_observation': 1}
    env.tracker._scans.append(SimpleNamespace(serial=5001, origin=np.zeros(2)))
    assert adapter.snapshot(env) is base.result
    assert adapter.last_motion_axis_stats['rejections'] == {'cluster_is_not_current_scan': 1}


def test_history_expires_without_coasting_corrected_view_and_keeps_elapsed_gaps():
    adapter, base, env, track, _ = setup()
    admit(adapter, env, track)
    env.tracker.tracks = []
    for _ in range(round(cfg.MOTION_WINDOW_S/cfg.UPDATE_RATE)+1):
        assert adapter.snapshot(env) is base.result
    assert adapter._observations == {} and adapter._last_serial == {}
    env.tracker.tracks = [track]
    assert observe(adapter, env, track, 9000, 6.) is base.result
    assert adapter.last_motion_axis_stats['stored_observation_counts'] == {7: 1}


def test_stale_gap_counts_as_time_not_as_an_observation():
    adapter, _, env, track, _ = setup()
    observe(adapter, env, track, 1, 3.)
    env.pose_stale = True
    observe(adapter, env, track, 200, 3.25)
    env.pose_stale = False
    observe(adapter, env, track, 900, 3.5)
    observe(adapter, env, track, 902, 3.75)
    stats = adapter.last_motion_axis_stats['updates'][0]
    assert stats['observation_frames'] == [1, 3, 4]
    assert stats['observation_times_s'] == [.5, 1.5, 2.]


def test_constructor_keeps_inner_persistence_and_all_v15_options():
    filt = SafetyFilterV16(conditional_prefix=False, policy_prefix_search=False, max_coast_s=4.)
    assert isinstance(filt.perception, MotionAxisPerception)
    assert filt.perception.base_perception is filt.track_persistence
    assert filt.track_persistence.max_coast_s == 4.
    assert not filt.conditional_prefix and not filt.policy_prefix_search
    disabled = SafetyFilterV16(motion_axis_geometry=False)
    assert disabled.motion_axis_perception is None
    assert disabled.perception is disabled.track_persistence


@pytest.mark.parametrize('enabled', [False, True])
def test_parent_filter_output_exact_and_diagnostics_detached(monkeypatch, enabled):
    expected = np.array([.2, -.4])
    def parent(self, env, action):
        self.last = {'why': 'parent', 'track_persistence': {'kept': True}}
        return expected, True
    monkeypatch.setattr(safety_v15.SafetyFilterV15, '_filter', parent)
    filt = SafetyFilterV16(motion_axis_geometry=enabled)
    if enabled: filt.motion_axis_perception.last_motion_axis_stats = {'updates': [1]}
    out, changed = filt._filter(SimpleNamespace(), np.zeros(2))
    assert out is expected and changed is True and filt.last['track_persistence'] == {'kept': True}
    assert filt.last['v16_motion_axis_geometry'] is enabled
    if enabled:
        filt.motion_axis_perception.last_motion_axis_stats['updates'].append(2)
        assert filt.last['motion_axis_geometry'] == {'updates': [1]}
    else:
        assert filt.last['motion_axis_geometry'] == {}


def test_exact_saved_ch_observations_qualify_before_first_intervention():
    root = Path(__file__).resolve().parents[1]
    audit_path = root/'results/safety_dev/v10_iterations/audits/v14_ch_ho_cv073_mechanism/audit.json'
    trace_path = root/'results/safety_dev/v10_iterations/persistent_prefix32/traces/009_v14.jsonl'
    if not audit_path.exists() or not trace_path.exists():
        pytest.skip('Optional committed saved-case diagnostic is unavailable')
    audit = json.loads(audit_path.read_text())
    raw = trace_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == audit['provenance']['inputs'][trace_path.relative_to(root).as_posix()]
    rows = [json.loads(line) for line in raw.splitlines()][:13]
    base, env = Base(), Onboard()
    adapter = MotionAxisPerception(base)
    admitted = []
    for row in rows:
        onboard = row['diagnostic_before']['onboard']
        env.pose_stale = onboard['pose_stale']
        tracks = []
        for value in onboard['raw_tracks']:
            item = copy.deepcopy(value)
            item['last_evidence'] = SimpleNamespace(**item.pop('evidence')) if item['evidence'] else None
            item['history'] = [(x['scan_serial'], np.array(x['points'])) for x in item['history']]
            tracks.append(SimpleNamespace(**item))
        env.tracker = SimpleNamespace(tracks=tracks, _scans=[SimpleNamespace(**x) for x in onboard['tracker_scans']])
        values = copy.deepcopy(row['diagnostic_decision']['snapshot'])
        values.pop('units', None)
        for key in ('tangent', 'right', 'centre', 'points', 'edges_a', 'edges_b'):
            values[key] = np.asarray(values[key])
        values['tracks'] = [cc.TrackView(t['id'], np.array(t['position']), np.array(t['velocity']), t['heading']) for t in values['tracks']]
        base.result = cc.Snapshot(**values)
        result = adapter.snapshot(env)
        if adapter.last_motion_axis_stats['updates']:
            admitted.append(row['step'])
        if row['step'] == 13:
            assert adapter.last_motion_axis_stats['replaced_source_ids'] == [67]
            expected = audit['geometry_sensitivity_on_saved_sac_commands']['motion_axis_face_centre_and_axis']
            np.testing.assert_allclose(result.tracks[0].position, expected['position'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.degrees(result.tracks[0].heading), expected['heading_deg'], atol=1e-12, rtol=0)
    assert admitted == [12, 13]
    assert base.calls == 13
