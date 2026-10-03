"""Synthetic and saved perception only; no simulator, reset or policy calls."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import constants as cfg
from classical import common as cc
from safety_source_points import SourcePointPerception
from safety_v19 import SafetyFilterV19
import safety_v16


class Base:
    def __init__(self):
        self.calls, self.frames = 0, 3
        self.memory = []
        self.result = cc.Snapshot(0., 0., 0., .5, 0., 0., np.array([0., 1.]),
            np.array([1., 0.]), np.zeros(2), 0., 0., 20., np.empty((0, 2)), [])

    def snapshot(self, env):
        self.calls += 1
        return self.result


class Onboard(SimpleNamespace):
    @property
    def targets(self): raise AssertionError('hidden target access')
    @property
    def asv_x(self): raise AssertionError('hidden pose access')
    @property
    def raw_ranges(self): raise AssertionError('extra sensor access')


def setup():
    base = Base()
    centre = np.array([5., 5.])
    cluster = centre + np.column_stack((np.linspace(-.1, .1, 8), np.linspace(-cfg.LOA/2, cfg.LOA/2, 8)))
    remembered, wall = np.array([8., 8.]), np.array([9., 9.])
    base.result.points = np.array([cluster[0], wall, cluster[4], remembered])
    base.memory = [(2, np.array([remembered])), (3, np.vstack((cluster, wall)))]
    track = SimpleNamespace(id=7, hits=5, confirmed=True, misses=0,
        position=centre.copy(), velocity=np.array([0., .5]), history=[(123, cluster)],
        last_fit_centre=centre.copy(), last_fit_heading_deg=0.,
        last_evidence=SimpleNamespace(appear=3, violations=4))
    target = cc.TrackView(-1000000, centre.copy(), track.velocity.copy(), 0., object())
    unrelated = cc.TrackView(88, np.array([10., 10.]), np.zeros(2), 1., object())
    base.result.tracks = [unrelated, target]
    stats = dict(frame=3, pose_stale=False, updates=[dict(source_id=7, synthetic_id=-1000000,
        kind='measured_refresh', observations=3, observation_frames=[1, 2, 3],
        endpoint_translation_m=[.5, .5], max_translation_residual_m=.01)],
        hypotheses=[dict(source_id=7, id=-1000000, age_s=0., anchor_frame=3,
                         position=centre.tolist(), velocity=track.velocity.tolist())])
    inner = SimpleNamespace(last_track_persistence_stats=stats, min_motion_observations=3)
    env = Onboard(pose_stale=False, tracker=SimpleNamespace(tracks=[track], _scans=[SimpleNamespace(serial=123)]))
    return SourcePointPerception(base, inner), base, env, track, stats


def test_transfer_exact_current_rows_preserves_other_points_views_memory_and_tracker():
    adapter, base, env, track, stats = setup()
    before = copy.deepcopy(track)
    memory_before = [(frame, points.copy()) for frame, points in base.memory]
    snapshot_before = base.result.points.copy()
    out = adapter.snapshot(env)
    assert base.calls == 1 and out is not base.result
    np.testing.assert_array_equal(out.points, snapshot_before[[1, 3]])
    np.testing.assert_array_equal(base.result.points, snapshot_before)
    assert out.tracks is base.result.tracks
    assert out.tracks[0] is base.result.tracks[0] and out.tracks[1].id == -1000000
    for (frame, points), (old_frame, old) in zip(base.memory, memory_before):
        assert frame == old_frame
        np.testing.assert_array_equal(points, old)
    np.testing.assert_array_equal(track.history[-1][1], before.history[-1][1])
    assert adapter.last_source_point_stats['removed_point_count'] == 2
    assert adapter.last_source_point_stats['source_ids_transferred'] == [7]


def test_identical_remembered_coordinate_retained_and_nearby_points_never_masked():
    adapter, base, env, track, _ = setup()
    base.memory[0][1][0] = track.history[-1][1][0]
    close = track.history[-1][1][4].copy()
    close[0] = np.nextafter(close[0], np.inf)
    farther = track.history[-1][1][4] + np.array([3e-9, 0.])
    base.result.points = np.vstack((base.result.points, close, farther))
    before = base.result.points.copy()
    out = adapter.snapshot(env)
    np.testing.assert_array_equal(out.points, before[[0, 1, 3, 4, 5]])
    assert adapter.last_source_point_stats['transfers'][0]['remembered_equal_points_retained'] == 1


@pytest.mark.parametrize('problem', ['stale', 'unconfirmed', 'few_hits', 'missed', 'no_evidence',
    'vacate_only', 'weak_evidence', 'oversized_length', 'merged_wall_width', 'partial_length',
    'duplicate_points', 'duplicate_raw_id', 'overlapping_raw_source', 'duplicate_update',
    'duplicate_hypothesis', 'coasted_hypothesis', 'unpublished_hypothesis', 'duplicate_view',
    'stale_serial', 'duplicate_serial', 'stationary_certificate', 'bad_residual',
    'no_memory', 'not_current_memory', 'wrong_view_position', 'wrong_hull_heading'])
def test_rejects_missing_stale_ambiguous_or_incompatible_evidence(problem):
    adapter, base, env, track, stats = setup()
    cluster = track.history[-1][1]
    if problem == 'stale': env.pose_stale = True
    if problem == 'unconfirmed': track.confirmed = False
    if problem == 'few_hits': track.hits = cfg.TRACK_MIN_HITS-1
    if problem == 'missed': track.misses = 1
    if problem == 'no_evidence': track.last_evidence = None
    if problem == 'vacate_only': track.last_evidence.appear = 0
    if problem == 'weak_evidence': track.last_evidence.violations = 0
    if problem == 'oversized_length': cluster[:, 1] += np.linspace(-1., 1., len(cluster))
    if problem == 'merged_wall_width': cluster[:, 0] += np.linspace(-1., 1., len(cluster))
    if problem == 'partial_length': cluster[:, 1] = 5.+np.linspace(-.1, .1, len(cluster))
    if problem == 'duplicate_points': track.history[-1] = (123, np.vstack((cluster, cluster[0])))
    if problem == 'duplicate_raw_id': env.tracker.tracks.append(copy.deepcopy(track))
    if problem == 'overlapping_raw_source':
        other = copy.deepcopy(track); other.id = 8; env.tracker.tracks.append(other)
    if problem == 'duplicate_update': stats['updates'].append(copy.deepcopy(stats['updates'][0]))
    if problem == 'duplicate_hypothesis': stats['hypotheses'].append(copy.deepcopy(stats['hypotheses'][0]))
    if problem == 'coasted_hypothesis': stats['hypotheses'][0]['age_s'] = .5
    if problem == 'unpublished_hypothesis': base.result.tracks.pop()
    if problem == 'duplicate_view': base.result.tracks.append(copy.copy(base.result.tracks[-1]))
    if problem == 'stale_serial': env.tracker._scans[-1].serial = 124
    if problem == 'duplicate_serial': env.tracker._scans.append(SimpleNamespace(serial=123))
    if problem == 'stationary_certificate': stats['updates'][0]['endpoint_translation_m'] = [0., 0.]
    if problem == 'bad_residual': stats['updates'][0]['max_translation_residual_m'] = np.inf
    if problem == 'no_memory': del base.memory
    if problem == 'not_current_memory': base.frames = 4
    if problem == 'wrong_view_position': base.result.tracks[-1].position = np.array([15., 15.])
    if problem == 'wrong_hull_heading': base.result.tracks[-1].heading = np.pi/2
    out = adapter.snapshot(env)
    assert out is base.result and base.calls == 1
    assert adapter.last_source_point_stats['removed_point_count'] == 0


def test_repeated_serial_never_transfers_twice_and_new_admission_supported():
    adapter, base, env, track, stats = setup()
    stats['updates'][0]['kind'] = 'admission'
    assert adapter.snapshot(env) is not base.result
    assert adapter.snapshot(env) is base.result
    assert adapter.last_source_point_stats['rejections']['repeated_observation'] == 1


def test_constructor_has_v16_parent_and_preserves_inner_diagnostics_reference():
    filt = SafetyFilterV19(max_coast_s=4., policy_prefix_search=False)
    assert isinstance(filt, safety_v16.SafetyFilterV16)
    assert filt.perception is filt.source_point_perception
    assert filt.perception.base_perception is filt.motion_axis_perception
    assert filt.perception.track_persistence is filt.track_persistence
    assert filt.track_persistence.max_coast_s == 4. and not filt.policy_prefix_search
    disabled = SafetyFilterV19(source_owned_points=False)
    assert disabled.perception is disabled.motion_axis_perception
    assert disabled.source_point_perception is None


@pytest.mark.parametrize('enabled', [False, True])
def test_disabled_parent_parity_and_detached_diagnostics(monkeypatch, enabled):
    expected = np.array([.3, -.2])
    def parent(self, env, action):
        self.last = {'why': 'parent', 'model_ego_observer': True}
        return expected, True
    monkeypatch.setattr(safety_v16.SafetyFilterV16, '_filter', parent)
    filt = SafetyFilterV19(source_owned_points=enabled)
    if enabled: filt.source_point_perception.last_source_point_stats = {'transfers': [7]}
    out, changed = filt._filter(Onboard(), np.zeros(2))
    assert out is expected and changed is True and filt.last['model_ego_observer']
    assert filt.last['v19_source_owned_points'] is enabled
    if enabled:
        filt.source_point_perception.last_source_point_stats['transfers'].append(8)
        assert filt.last['source_points'] == {'transfers': [7]}
    else:
        assert filt.last['source_points'] == {}


def test_saved_crs04_current_rows_qualify_with_conservative_old_memory_union():
    root = Path(__file__).resolve().parents[1]
    path = root/'results/safety_dev/v10_iterations/v16_broader40_paired/traces/011_v16.jsonl'
    if not path.exists():
        pytest.skip('Optional committed saved trace unavailable')
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            rows.append(json.loads(line))
            if rows[-1]['step'] == 13: break
    row = rows[-1]
    onboard = row['diagnostic_before']['onboard']
    values = copy.deepcopy(row['diagnostic_decision']['snapshot'])
    values.pop('units', None)
    for name in ('tangent', 'right', 'centre', 'points', 'edges_a', 'edges_b'):
        values[name] = np.array(values[name])
    values['tracks'] = [cc.TrackView(v['id'], np.array(v['position']), np.array(v['velocity']), v['heading']) for v in values['tracks']]
    base = Base(); base.result = cc.Snapshot(**values); base.frames = 13
    x, y, heading_deg = onboard['pose_hold_xy_heading_deg']
    heading = np.radians(heading_deg)
    ranges = np.array(onboard['scan']['gated_ranges'])
    mask = ranges < cfg.LIDAR_RANGE-1e-5
    bearings = np.radians(np.array(onboard['scan']['bearings_deg'])[mask])+heading
    origin = np.array([x, y])+cc.LIDAR_OFFSET_M*np.array([np.sin(heading), np.cos(heading)])
    current = origin+ranges[mask,None]*np.stack([np.sin(bearings), np.cos(bearings)], axis=1)
    # Conservative superset of prior memory: include every previously output
    # point, even one the real memory may already have cleared. No old point
    # can become removable due to this saved-test reconstruction.
    old = np.concatenate([np.array(r['diagnostic_decision']['snapshot']['points']) for r in rows[:-1]])
    base.memory = [(12, old), (13, current)]
    tracks = []
    for raw in onboard['raw_tracks']:
        raw = copy.deepcopy(raw)
        raw['last_evidence'] = SimpleNamespace(**raw.pop('evidence')) if raw['evidence'] else None
        raw['history'] = [(h['scan_serial'], np.array(h['points'])) for h in raw['history']]
        tracks.append(SimpleNamespace(**raw))
    env = Onboard(pose_stale=onboard['pose_stale'], tracker=SimpleNamespace(tracks=tracks,
                   _scans=[SimpleNamespace(**s) for s in onboard['tracker_scans']]))
    persistence = SimpleNamespace(last_track_persistence_stats=row['filter']['track_persistence'], min_motion_observations=3)
    adapter = SourcePointPerception(base, persistence)
    out = adapter.snapshot(env)
    assert adapter.last_source_point_stats['source_ids_transferred'] == [77]
    assert adapter.last_source_point_stats['removed_point_count'] == 9
    assert len(out.points) == 49 and out.tracks is base.result.tracks
    np.testing.assert_array_equal(out.points, base.result.points[[i for i in range(58) if i not in [30,31,32,33,34,35,36,37,39]]])
