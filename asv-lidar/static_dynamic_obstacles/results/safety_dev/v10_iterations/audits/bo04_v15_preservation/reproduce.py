"""Completed saved-state scoring only: no environment or policy calls."""
import ast
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src'))
OUT = Path(__file__).parent
BASE = ROOT / 'results/safety_dev'
RUN = BASE / 'v10_iterations/conditional_prefix32'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x]


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def main():
    paths = {
        'off': BASE / 'v9_paired_pilot/traces/008_off.jsonl',
        'v10': BASE / 'v10_iterations/conditional_policy_32/traces/003_v10.jsonl',
        'v14': BASE / 'v10_iterations/persistent_preference_remaining28/traces/002_v14.jsonl',
        'v14_prefix': BASE / 'v10_iterations/persistent_prefix32/traces/003_v14.jsonl',
        'v15': RUN / 'traces/003_v15.jsonl',
    }
    traces = {k: read(p) for k, p in paths.items()}
    manifest = json.loads((RUN / 'manifest.json').read_text())
    completed = json.loads((RUN / 'attempts/003_result.json').read_text())
    assert completed['case'] == 'DV3:DV3-BO-CV-04' and len(traces['v15']) == completed['steps']
    deps = ['src/constants.py', 'src/constant_temp.py', 'src/classical/common.py',
            'src/safety_v2.py', 'src/safety_v3.py', 'src/ship.py', 'bluefin/dynamics.py']
    source_hashes = {}
    for name in deps:
        source_hashes[name] = sha(ROOT / name)
        assert source_hashes[name] == manifest['source_sha256'][name], name
    off_manifest = json.loads((BASE / 'v9_paired_pilot/manifest.json').read_text())
    assert all(manifest[k] == off_manifest[k] for k in ['checkpoint_sha256', 'config_sha256'])
    cases = [next(c for c in m['cases'] if c['case'] == completed['case']) for m in [manifest, off_manifest]]
    assert all(cases[0][key] == cases[1][key] for key in ['seed', 'scenario_sha256'])
    import constants as cfg
    for key, value in manifest['constants'].items():
        try: value = ast.literal_eval(value)
        except (ValueError, SyntaxError): continue
        setattr(cfg, key, value)
    from classical import common as cc
    import safety_v2 as v2
    import safety_v3 as v3

    def command(r): return [r['rudder_command'], r['signed_rpm_command']]
    divergence = {}
    for name, rows in traces.items():
        first = next((i for i, (a, b) in enumerate(zip(rows, traces['off'])) if command(a) != command(b)), None)
        divergence[name] = {'steps': len(rows), 'changed_steps': [r['step'] for r in rows if r['changed']],
            'first_command_difference': None if first is None else first+1,
            'matched_recorded_prestate_at_difference': None if first is None else rows[first]['pre_state'] == traces['off'][first]['pre_state'],
            'matched_policy_at_difference': None if first is None else rows[first]['policy_action'] == traces['off'][first]['policy_action']}
    row = traces['v15'][11]
    saved = copy.deepcopy(row['diagnostic_decision']['snapshot']); saved.pop('units')
    for key in ['tangent', 'right', 'centre', 'points', 'edges_a', 'edges_b']:
        saved[key] = np.asarray(saved[key], dtype=float)
    saved['tracks'] = [cc.TrackView(t['id'], np.asarray(t['position']), np.asarray(t['velocity']), t['heading']) for t in saved['tracks']]
    snap = cc.Snapshot(**saved)
    a = row['diagnostic_decision']['actuators_before_decision']
    act = cc.Actuators(); act.servo, act.buffer, act.executed = a['servo'], a['buffer'], a['executed']
    off_rows = traces['off'][11:27]
    assert len(off_rows) == 16 and all(r['signed_rpm_command'] >= 0 for r in off_rows)
    seq = np.array([[r['rudder_command'], (r['signed_rpm_command']-cfg.CRUISE_RPM)/cfg.RPM_DELTA] for r in off_rows])[None]
    before = row['diagnostic_before']; truth = before['truth_scoring_only']; own = truth['own']

    def components(s, ro):
        arrays = {'static_memory': cc.point_clearance(ro.positions, ro.headings, s.points, reach=cc.HALF_L+1)-v2.GAP_STATIC_M,
                  'boundary': cc.boundary_clearance(ro.positions, ro.headings, s.edges_a, s.edges_b)-v2.GAP_BOUNDARY_M}
        arrays.update({f'target:{t.id}': cc.target_gap(ro.positions, ro.headings, ro.times, t)-v2.GAP_TARGET_M for t in s.tracks})
        return {key: {'minimum_clearance_m': float(value.min()), 'minimum_time_s': float(ro.times[np.argmin(value[:, 0])]),
                      'first_violation_s': float(ro.times[np.flatnonzero(value[:, 0]<0)[0]]) if (value<0).any() else None}
                for key, value in arrays.items()}

    def evaluate(s):
        ro = v3.rollout_seq(s, act, seq)
        first, clear = v2.SafetyFilterV2._evaluate(None, s, ro)
        return ro, {'first_violation_s_including_terminal': first[0], 'minimum_clearance_m_including_terminal': clear[0],
                    'hard_safe': bool(np.isposinf(first[0])), 'components_within_horizon': components(s, ro)}

    raw = row['diagnostic_decision']['raw_ego_u_v_yaw_rad_s']
    true_targets = [cc.TrackView(-9000-i, np.array([t['x'], t['y']]), np.array(t['velocity']), math.radians(t['heading']))
                    for i, t in enumerate(truth['targets'])]
    raw_track = next(t for t in before['onboard']['raw_tracks'] if t['id'] == snap.tracks[0].id)
    course = math.atan2(*raw_track['velocity'])
    fwd = np.array([math.sin(course), math.cos(course)])
    stbd = np.array([math.cos(course), -math.sin(course)])
    cluster_geometry = []
    for h in raw_track['history']:
        points = np.asarray(h['points'])
        projected = np.column_stack((points @ fwd, points @ stbd))
        cluster_geometry.append({'scan_serial': h['scan_serial'], 'points': len(points),
                                 'axis_min': projected.min(axis=0), 'axis_max': projected.max(axis=0),
                                 'longitudinal_extent_m': np.ptp(projected[:, 0]),
                                 'beam_extent_m': np.ptp(projected[:, 1])})
    from safety_motion_axis import MotionAxisPerception
    class SavedBase:
        def snapshot(self, env): return self.result
    base = SavedBase()
    adapter = MotionAxisPerception(base)
    adapter_decisions = []
    for previous in traces['v15'][:12]:
        onboard = previous['diagnostic_before']['onboard']
        raw_tracks = []
        for value in onboard['raw_tracks']:
            item = copy.deepcopy(value)
            evidence = item.pop('evidence')
            item['last_evidence'] = SimpleNamespace(**evidence) if evidence else None
            item['history'] = [(h['scan_serial'], np.asarray(h['points'])) for h in item['history']]
            raw_tracks.append(SimpleNamespace(**item))
        proxy = SimpleNamespace(pose_stale=onboard['pose_stale'], tracker=SimpleNamespace(
            tracks=raw_tracks, _scans=[SimpleNamespace(**s) for s in onboard['tracker_scans']]))
        values = copy.deepcopy(previous['diagnostic_decision']['snapshot']); values.pop('units')
        for key in ['tangent', 'right', 'centre', 'points', 'edges_a', 'edges_b']:
            values[key] = np.asarray(values[key], dtype=float)
        values['tracks'] = [cc.TrackView(t['id'], np.asarray(t['position']), np.asarray(t['velocity']), t['heading']) for t in values['tracks']]
        base.result = cc.Snapshot(**values)
        adapter_result = adapter.snapshot(proxy)
        adapter_decisions.append(copy.deepcopy(adapter.last_motion_axis_stats))
    variants = {
        'recorded_onboard': snap,
        'raw_yaw_only': replace(snap, r=raw[2]),
        'raw_ego_only': replace(snap, u=raw[0], v=raw[1], r=raw[2]),
        'true_current_ego_scoring_only': replace(snap, x=own['asv_x'], y=own['asv_y'], heading=math.radians(own['asv_h']),
                                                u=own['u_body'], v=own['v_body'], r=math.radians(own['asv_w'])),
        'true_current_targets_CV_sensitivity_only': replace(snap, tracks=true_targets),
        'onboard_fitted_centre_only': replace(snap, tracks=[replace(snap.tracks[0], position=np.asarray(raw_track['last_fit_centre']))]),
        'onboard_velocity_course_heading_only': replace(snap, tracks=[replace(snap.tracks[0], heading=course)]),
        'onboard_fit_centre_and_velocity_course': replace(snap, tracks=[replace(snap.tracks[0], position=np.asarray(raw_track['last_fit_centre']), heading=course)]),
        'true_target_centre_only_scoring': replace(snap, tracks=[replace(snap.tracks[0], position=true_targets[0].position)]),
        'true_target_heading_only_scoring': replace(snap, tracks=[replace(snap.tracks[0], heading=true_targets[0].heading)]),
        'true_target_velocity_only_scoring': replace(snap, tracks=[replace(snap.tracks[0], velocity=true_targets[0].velocity)]),
        'v16_adapter_sequential_saved_frames_only': adapter_result,
    }
    scores = {}; predictions = {}
    for name, s in variants.items(): predictions[name], scores[name] = evaluate(s)
    ro = predictions['recorded_onboard']
    actual_positions = np.asarray([r['post_state'][:2] for r in off_rows])[:, None, :]
    actual_headings = np.radians([r['post_state'][2] for r in off_rows])[:, None]
    actual = cc.Rollout(actual_positions, actual_headings, np.asarray([r['post_state'][3] for r in off_rows])[:, None], .5*np.arange(1,17))
    indices = np.arange(cc.SUBSTEPS-1, len(ro.times), cc.SUBSTEPS)
    position_errors = np.linalg.norm(ro.positions[indices, 0]-actual_positions[:, 0], axis=1)
    heading_errors = np.degrees(np.abs(cc.wrap_pi(ro.headings[indices, 0]-actual_headings[:, 0])))
    target_errors = []
    for t in snap.tracks:
        distances = [np.linalg.norm(t.position - x.position) for x in true_targets]
        i = int(np.argmin(distances)); q = true_targets[i]
        target_errors.append({'track_id': t.id, 'nearest_truth_index_scoring_only': i,
            'position_error_m': distances[i], 'velocity_error_mps': np.linalg.norm(t.velocity-q.velocity),
            'heading_error_deg': math.degrees(abs(cc.wrap_pi(t.heading-q.heading))),
            'position': t.position, 'velocity': t.velocity, 'heading_deg': math.degrees(t.heading)})
    linear_errors = []
    for future in traces['v15'][11:28]:
        elapsed = future['diagnostic_before']['elapsed_time']-before['elapsed_time']
        observed = future['diagnostic_before']['truth_scoring_only']['targets']
        for i,t in enumerate(true_targets):
            linear_errors.append(float(np.linalg.norm(t.position+elapsed*t.velocity-[observed[i]['x'], observed[i]['y']])))
    report = {'scope': 'Completed saved BO04 decision12 scoring only. Actual future SAC commands are noncausal diagnostic inputs, unavailable online. No episodes/controller calls/source edits.',
        'case': completed['case'], 'v15_outcome': completed['outcome'], 'paired_seed_geometry_checkpoint_config': True,
        'inputs': {k: {'path': str(p.relative_to(ROOT)).replace('\\','/'), 'sha256': sha(p)} for k,p in paths.items()},
        'model_source_hashes_match_evaluated_manifest': source_hashes, 'command_comparison': divergence,
        'decision11_v15_filter': traces['v15'][10]['filter'], 'decision12_v15_filter': row['filter'],
        'decision12_own_state': {'truth_scoring_only_u_v_yaw_deg_s': [own['u_body'], own['v_body'], own['asv_w']],
                              'raw_u_v_yaw_deg_s': [raw[0], raw[1], math.degrees(raw[2])],
                              'filtered_u_v_yaw_deg_s': [snap.u, snap.v, math.degrees(snap.r)]},
        'known_successful_off_steps': [r['step'] for r in off_rows], 'known_successful_off_issued_commands': [command(r) for r in off_rows],
        'scores': scores, 'actual_off_own_decision_endpoints_with_onboard_geometry': components(snap, actual),
        'actual_off_own_endpoints_with_true_current_targets_CV_sensitivity': components(replace(snap, tracks=true_targets), actual),
        'own_prediction_endpoint_position_errors_m': position_errors, 'own_prediction_endpoint_heading_errors_deg': heading_errors,
        'target_current_errors_scoring_only': target_errors,
        'source16_existing_onboard_evidence': {
            'raw_id': raw_track['id'], 'hits': raw_track['hits'], 'misses': raw_track['misses'],
            'is_dynamic': raw_track['is_dynamic'], 'confirmed': raw_track['confirmed'],
            'motion_evidence': raw_track['evidence'], 'evidence_run': raw_track['evidence_run'],
            'last_fit_centre': raw_track['last_fit_centre'], 'last_fit_heading_deg': raw_track['last_fit_heading_deg'],
            'current_velocity_course_deg': math.degrees(course) % 360.,
            'course_heading_error_deg_scoring_only': math.degrees(abs(cc.wrap_pi(course-true_targets[0].heading))),
            'fit_centre_error_m_scoring_only': np.linalg.norm(np.asarray(raw_track['last_fit_centre'])-true_targets[0].position),
            'history_clusters_in_current_velocity_course_axis': cluster_geometry,
            'persistent_diagnostics': row['filter']['track_persistence'],
        },
        'new_v16_adapter_saved_only_replay': {
            'module_sha256': sha(ROOT/'src/safety_motion_axis.py'),
            'scope': 'One wrapper call per recorded decision1-12, no history backfill; input base snapshots are recorded V15 snapshots. No future data or true state supplied to adapter.',
            'decisions': adapter_decisions,
        },
        'recorded_v15_true_target_max_deviation_from_current_CV_over8s_m': max(linear_errors),
        'limitations': ['Old successful OFF trace has no future target states. V15 future target observations validate CV on that recorded trajectory only; no unrecorded reactive response is assumed.',
                       'Actual-own endpoint scoring samples every .5s; predictor geometry samples .125s, while simulator contact checks use .1s. Positive endpoint margins do not establish continuous clearance.',
                       'True-current state replacements are isolated model sensitivities, not deployable inputs or evidence of a closed-loop rescue.',
                       'Per-component tables exclude terminal extension; aggregate evaluator includes the original terminal check.']}
    (OUT/'audit.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(clean({k:report[k] for k in ['decision12_own_state','scores','actual_off_own_decision_endpoints_with_onboard_geometry',
           'actual_off_own_endpoints_with_true_current_targets_CV_sensitivity','target_current_errors_scoring_only','recorded_v15_true_target_max_deviation_from_current_CV_over8s_m']}), indent=2))


if __name__ == '__main__': main()
