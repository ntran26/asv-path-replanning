"""Score committed CH-HO-CV-073 traces; no environments or policy calls.

The successful SAC command sequence is retrospective evidence, unavailable to
the online controller. Current target truth only scores estimation errors; it
never enters the collision checker or supplies assumed future motion.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from tools.diagnostics.safety.suite_status import read_shared_bytes
from tools.diagnostics.safety.report_safety_iteration import validate_archive, validate_trace

CASE = 'TS2:CH-HO-CV-073'
BASE = ROOT / 'results/safety_dev'
INPUTS = {'v14': (BASE / 'v10_iterations/persistent_prefix32', 9),
          'off': (BASE / 'v9_paired_pilot', 26)}


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def require(condition, message):
    if not condition: raise ValueError(message)


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b))
                 if x['rudder_command'] != y['rudder_command']
                 or x['signed_rpm_command'] != y['signed_rpm_command']), None)


def visible_face_centre(points, origin, heading, length, breadth, tolerance):
    """Fixed-axis form of tracking.hull_fit_centre's visible-face completion.

    Diagnostic hypothesis only. The caller checks no returns touch the blind
    zone. Motion provides the axis; no truth, outcome, future scan, or fit search
    supplies geometry. A course axis is not guaranteed to equal hull heading.
    """
    points, origin = np.asarray(points), np.asarray(origin)
    forward = np.array([np.sin(heading), np.cos(heading)])
    right = np.array([np.cos(heading), -np.sin(heading)])
    centre = np.zeros(2)
    for axis, size in ((forward, length), (right, breadth)):
        projection = points @ axis
        low, high = projection.min(), projection.max()
        middle = (low + high) / 2
        if high - low >= size - tolerance:
            coordinate = middle
        elif origin @ axis <= middle:
            coordinate = low + size / 2
        else:
            coordinate = high - size / 2
        centre += coordinate * axis
    return centre


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', help='New audit directory name; omitted means stdout only')
    args = parser.parse_args()
    if args.tag is not None:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.tag), 'Invalid tag')
    provenance = {}

    def read(path):
        data = read_shared_bytes(path)
        provenance[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
        return data

    manifests, results, traces, identities = {}, {}, {}, {}
    for mode, (folder, attempt) in INPUTS.items():
        manifest = json.loads(read(folder / 'manifest.json'))
        validate_archive(read(folder / 'evaluated_sources.zip'), manifest['source_sha256'])
        result = json.loads(read(folder / f'attempts/{attempt:03d}_result.json'))
        token = json.loads(read(folder / f'attempts/{attempt:03d}.json'))
        case = next(c for c in manifest['cases'] if c['case'] == CASE)
        require(all(result[k] == token[k] == v for k, v in
                    [('case', CASE), ('seed', case['seed']), ('mode', mode), ('attempt', attempt)]),
                'Result/token/manifest identity mismatch')
        raw = read(folder / f'traces/{attempt:03d}_{mode}.jsonl')
        validate_trace(raw, result, 'off' if mode == 'off' else 'idle')
        traces[mode] = [json.loads(line) for line in raw.splitlines() if line.strip()]
        manifests[mode], results[mode], identities[mode] = manifest, result, case
    for key in ('seed', 'scenario_sha256'):
        require(identities['off'][key] == identities['v14'][key], 'Scenario identity mismatch')
    for key in ('checkpoint_sha256', 'config_sha256'):
        require(manifests['off'][key] == manifests['v14'][key], 'Policy identity mismatch')
    require(results['off']['outcome'] == 'goal', 'Reference is not a saved SAC success')
    rows, off = traces['v14'], traces['off']
    index = first_difference(rows, off)
    require(index is not None, 'No command divergence')
    row = rows[index]
    for field in ('pre_state', 'policy_action'):
        require(row[field] == off[index][field], 'First-divergence recorded input mismatch')
    require(all(a['post_state'] == b['post_state'] for a, b in zip(rows[:index], off[:index])),
            'State paths already differ before the command divergence')

    dependencies = ['src/constants.py', 'src/constant_temp.py', 'src/classical/common.py',
                    'src/safety_v2.py', 'src/safety_v3.py', 'src/ship.py',
                    'src/reference_controller.py', 'src/tracking.py', 'bluefin/dynamics.py']
    checked_sources = {}
    for name in dependencies:
        digest = hashlib.sha256(read(ROOT / name)).hexdigest()
        require(digest == manifests['v14']['source_sha256'][name], 'Predictor source drift: ' + name)
        checked_sources[name] = digest
    import constants as cfg
    for key, value in manifests['v14']['constants'].items():
        try: value = ast.literal_eval(value)
        except (ValueError, SyntaxError): continue
        setattr(cfg, key, value)
    from classical import common as cc
    import safety_v2 as v2
    import safety_v3 as v3
    require(not row['filter'].get('calibrated_braking')
            and not row['filter'].get('dual_brake_sequences', 0)
            and not row['filter'].get('v10_target_turn_rate_deg_s', 0),
            'Replay requires recorded nominal CV/weak-brake settings')
    saved = copy.deepcopy(row['diagnostic_decision']['snapshot'])
    saved.pop('units', None)
    for key in ('tangent', 'right', 'centre', 'points', 'edges_a', 'edges_b'):
        saved[key] = np.asarray(saved[key], dtype=float)
    saved['tracks'] = [cc.TrackView(t['id'], np.array(t['position']), np.array(t['velocity']), t['heading'])
                       for t in saved['tracks']]
    snap = cc.Snapshot(**saved)
    act = cc.Actuators()
    saved_act = row['diagnostic_decision']['actuators_before_decision']
    act.servo, act.executed, act.buffer = saved_act['servo'], saved_act['executed'], copy.deepcopy(saved_act['buffer'])

    def command(record):
        rpm = record['signed_rpm_command']
        require(rpm >= 0 or np.isclose(rpm, v2.ASTERN_RPM), 'Unexpected partial reverse RPM')
        throttle = np.nan if rpm < 0 else (rpm - cfg.CRUISE_RPM) / cfg.RPM_DELTA
        require(not np.isfinite(throttle) or abs(throttle) <= 1 + 1e-9, 'Throttle transport mismatch')
        return [record['rudder_command'], throttle]

    def arrays(positions, headings, times):
        values = {'static': cc.point_clearance(positions, headings, snap.points,
                       reach=cc.HALF_L + 1.) - v2.GAP_STATIC_M,
                  'boundary': cc.boundary_clearance(positions, headings, snap.edges_a,
                       snap.edges_b) - v2.GAP_BOUNDARY_M}
        values.update({f'target_{t.id}': cc.target_gap(positions, headings, times, t) - v2.GAP_TARGET_M
                       for t in snap.tracks})
        return values

    def summarize_array(values, times):
        a = values[:, 0]
        return {'minimum_clearance_m': float(a.min()),
                'minimum_is_positive_infinity': bool(np.isposinf(a.min())),
                'minimum_time_s': float(times[int(a.argmin())]) if np.isfinite(a.min()) else None,
                'first_violation_s': float(times[np.flatnonzero(a < 0)[0]]) if (a < 0).any() else None}

    def components(ro):
        values = arrays(ro.positions, ro.headings, ro.times)
        horizon = {k: summarize_array(v, ro.times) for k, v in values.items()}
        terminal = {}
        if ro.speeds[-1, 0] > v2.TERMINAL_STOP_SPEED:
            distance = min(v2.TERMINAL_M, v2.TERMINAL_S * ro.speeds[-1, 0])
            h = ro.headings[-1, 0]
            ext = ro.positions[-1:] + np.linspace(1/8, 1., 8)[:, None, None] * distance * np.array([np.sin(h), np.cos(h)])
            hs = np.full(ext.shape[:2], h)
            terminal['static'] = float((cc.point_clearance(ext, hs, snap.points,
                                        reach=cc.HALF_L + 1.) - v2.GAP_STATIC_M).min())
            if v2.TERMINAL_BOUNDARY:
                terminal['boundary'] = float((cc.boundary_clearance(ext, hs, snap.edges_a,
                                                   snap.edges_b) - v2.GAP_BOUNDARY_M).min())
        minimum = min([v['minimum_clearance_m'] for v in horizon.values()] + list(terminal.values()))
        return {'horizon': horizon, 'terminal': terminal, 'minimum_clearance_m': minimum}, values

    proposal = np.asarray(row['plan'], dtype=float)
    require(len(proposal) == round(v2.HORIZON_S / cfg.UPDATE_RATE), 'Unexpected plan horizon')
    require(len(off) >= index + len(proposal), 'Insufficient recorded successful commands')
    same_tail = proposal.copy()
    same_tail[0] = row['policy_action']
    sac = np.asarray([command(r) for r in off[index:index + len(proposal)]])
    np.testing.assert_allclose(sac[0], row['policy_action'], atol=1e-12, rtol=0)
    plans = {'stored_v14_proposal': proposal, 'policy_first_same_v14_tail': same_tail,
             'saved_successful_sac_sequence': sac}
    scored, rollouts, timeline = {}, {}, []
    for name, sequence in plans.items():
        ro = v3.rollout_seq(snap, act, sequence[None])
        first, clearance = v2.SafetyFilterV2._evaluate(None, snap, ro)
        comp, values = components(ro)
        np.testing.assert_allclose(comp['minimum_clearance_m'], clearance[0], atol=1e-12, rtol=0)
        scored[name] = dict(first_violation_s=float(first[0]), hard_safe=bool(np.isposinf(first[0])),
                            clearance_m=float(clearance[0]), components=comp, commands=sequence)
        rollouts[name] = ro
        for j, t in enumerate(ro.times):
            timeline.append({'sequence': name, 'time_s': float(t), 'x': ro.positions[j, 0, 0],
                             'y': ro.positions[j, 0, 1], 'heading_deg': np.degrees(ro.headings[j, 0]),
                             **{k + '_clearance_m': v[j, 0] for k, v in values.items()}})
    np.testing.assert_allclose(scored['stored_v14_proposal']['clearance_m'], row['filter']['checked_clearance'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(scored['policy_first_same_v14_tail']['clearance_m'], row['filter']['v10_policy_tail_clearance'], atol=1e-12, rtol=0)

    actual = np.array([r['post_state'] for r in off[index:index + len(proposal)]])
    times = cfg.UPDATE_RATE * np.arange(1, len(proposal) + 1)
    actual_values = arrays(actual[:, None, :2], np.radians(actual[:, None, 2]), times)
    actual_scoring = {k: summarize_array(v, times) for k, v in actual_values.items()}
    errors = []
    ro = rollouts['saved_successful_sac_sequence']
    for horizon in (.5, 1., 2., 4., 8.):
        k = round(horizon / cfg.UPDATE_RATE) - 1
        j = (k + 1) * cc.SUBSTEPS - 1
        errors.append(dict(horizon_s=horizon, predicted_position=ro.positions[j, 0], actual_position=actual[k, :2],
            position_error_m=float(np.linalg.norm(ro.positions[j, 0] - actual[k, :2])),
            displacement_error_m=float(np.linalg.norm((ro.positions[j, 0] - snap.position) - (actual[k, :2] - row['pre_state'][:2]))),
            heading_error_deg=float(np.degrees(cc.wrap_pi(ro.headings[j, 0] - np.radians(actual[k, 2]))))))
    truths = row['diagnostic_before']['truth_scoring_only']['targets']
    require(len(truths) == 1 and len(snap.tracks) == 1, 'Current-truth matching requires one target')
    target, truth = snap.tracks[0], truths[0]
    target_errors = dict(track_id=target.id, position=target.position, velocity=target.velocity,
                        heading_deg=float(np.degrees(target.heading)), true_current_target=truth,
        position_error_m=float(np.linalg.norm(target.position - [truth['x'], truth['y']])),
        position_error_xy_m=target.position - [truth['x'], truth['y']],
        velocity_error_mps=float(np.linalg.norm(target.velocity - truth['velocity'])),
        axis_error_deg=float(np.degrees(.5 * cc.wrap_pi(2 * (target.heading - np.radians(truth['heading']))))))
    # Causal geometry is computed first, then compared with current truth only.
    # Only the actual completed SAC command sequence is noncausal in these checks.
    geometry_history = []
    latest_face_centre, latest_course, latest_raw = None, None, None
    for previous in rows[:index + 1]:
        onboard = previous['diagnostic_before']['onboard']
        raw = next((t for t in onboard['raw_tracks'] if t['id'] == target.id), None)
        if raw is None: continue
        current_truth = previous['diagnostic_before']['truth_scoring_only']['targets'][0]
        truth_xy = np.array([current_truth['x'], current_truth['y']])
        course = float(np.arctan2(*raw['velocity'])) if np.linalg.norm(raw['velocity']) > 0 else None
        points = np.asarray(raw['history'][-1]['points'])
        serial = raw['history'][-1]['scan_serial']
        scan = next(s for s in onboard['tracker_scans'] if s['serial'] == serial)
        origin = np.asarray(scan['origin'])
        require(np.linalg.norm(points - origin, axis=1).min() > cfg.LIDAR_MIN_RANGE + cfg.TRACK_FIT_DEAD_ZONE_TOL_M,
                'Visible-face sensitivity does not implement clipped returns')
        covariance = np.asarray(raw['cov'])
        centre_error = np.asarray(raw['position']) - truth_xy
        fit = raw['last_fit_centre']
        record = dict(step=previous['step'], hits=raw['hits'], misses=raw['misses'],
            points=len(points), source_scan_serial=serial, motion_evidence=raw['evidence'],
            position=raw['position'], velocity=raw['velocity'], covariance=covariance,
            position_covariance_std_m=np.sqrt(np.diag(covariance)[:2]),
            centroid_to_true_centre_error_m=float(np.linalg.norm(centre_error)),
            centroid_to_true_centre_mahalanobis_squared=float(centre_error @ np.linalg.solve(covariance[:2, :2], centre_error)),
            fitted_heading_deg=raw['last_fit_heading_deg'], fitted_centre=fit,
            fitted_centre_error_m=None if fit is None else float(np.linalg.norm(fit - truth_xy)),
            fit_offset_known=raw['fit_offset_known'])
        if course is not None:
            d = np.array([np.sin(course), np.cos(course)])
            e = np.array([np.cos(course), -np.sin(course)])
            face = visible_face_centre(points, origin, course, cfg.LOA, cfg.BREADTH, cfg.TRACK_FIT_FULL_EXTENT_TOL_M)
            record.update(course_deg=float(np.degrees(course)),
                course_axis_error_deg=float(np.degrees(.5 * cc.wrap_pi(2 * (course - np.radians(current_truth['heading']))))),
                course_axis_span_length_m=float(np.ptp(points @ d)),
                course_axis_span_beam_m=float(np.ptp(points @ e)),
                visible_face_centre=face, visible_face_centre_error_m=float(np.linalg.norm(face - truth_xy)))
            if previous['step'] == row['step']:
                latest_face_centre, latest_course, latest_raw = face, course, raw
        geometry_history.append(record)
    require(latest_raw is not None, 'No current moving raw track')
    geometry_variants = {'captured_view': (target.position, target.heading),
        'course_axis_only': (target.position, latest_course),
        'raw_fit_centre_only': (np.asarray(latest_raw['last_fit_centre']), target.heading),
        'raw_fit_centre_and_course_axis': (np.asarray(latest_raw['last_fit_centre']), latest_course),
        'motion_axis_face_centre_only': (latest_face_centre, target.heading),
        'motion_axis_face_centre_and_axis': (latest_face_centre, latest_course)}
    geometry_sensitivity = {}
    for name, (position, heading) in geometry_variants.items():
        modified = copy.copy(snap)
        modified.tracks = [cc.TrackView(target.id, position.copy(), target.velocity.copy(), heading)]
        first, clearance = v2.SafetyFilterV2._evaluate(None, modified, ro)
        gaps = cc.target_gap(ro.positions, ro.headings, ro.times, modified.tracks[0]) - v2.GAP_TARGET_M
        geometry_sensitivity[name] = dict(position=position, heading_deg=float(np.degrees(heading)),
            hard_safe=bool(np.isposinf(first[0])), first_violation_s=float(first[0]), clearance_m=float(clearance[0]),
            target_component=summarize_array(gaps, ro.times),
            current_centre_error_m=float(np.linalg.norm(position - [truth['x'], truth['y']])),
            current_axis_error_deg=float(np.degrees(.5 * cc.wrap_pi(2 * (heading - np.radians(truth['heading']))))))
    after = [{'step': r['step'], 'why': r['filter'].get('why', 'idle'), 'changed': r['changed'],
              'checked_clearance': r['filter'].get('checked_clearance'), 'rudder': r['rudder_command'],
              'signed_rpm': r['signed_rpm_command']} for r in rows[index:]]
    output = dict(scope='Saved completed-case scoring only; zero episodes, environment calls or policy calls.',
        case=CASE, seed=identities['v14']['seed'], scenario_sha256=identities['v14']['scenario_sha256'],
        outcomes={m: {'outcome': r['outcome'], 'steps': r['steps']} for m, r in results.items()},
        first_divergence={'step': row['step'], 'prior_commands_and_recorded_post_states_exact_equal': True,
            'pre_state_exact_equal': True, 'policy_action_exact_equal': True,
            'pre_state': row['pre_state'], 'policy_action': row['policy_action'],
            'off_rudder_rpm': [off[index]['rudder_command'], off[index]['signed_rpm_command']],
            'v14_rudder_rpm': [row['rudder_command'], row['signed_rpm_command']],
            'filter': row['filter'], 'onboard_snapshot': row['diagnostic_decision']['snapshot'],
            'pre_actuators': saved_act},
        plan_scores=scored, component_timeline=timeline,
        actual_successful_sac_endpoints_scored_against_captured_onboard_cv=actual_scoring,
        recorded_future_sac_command_conditioned_model_errors=errors,
        current_target_error_scoring_only=target_errors, v14_post_divergence=after,
        pre_intervention_raw_track_history=geometry_history,
        geometry_sensitivity_on_saved_sac_commands=geometry_sensitivity,
        geometry_sensitivity_method={'source': 'src/tracking.py:hull_fit_centre, visible-face known-size completion',
            'axis': 'current measured velocity course; no future measurement or truth',
            'LOA': cfg.LOA, 'BREADTH': cfg.BREADTH, 'existing_full_extent_tolerance': cfg.TRACK_FIT_FULL_EXTENT_TOL_M,
            'covariance_caution': 'Kalman centroid covariance does not include the centroid-to-hull-centre shape bias; Mahalanobis error is descriptive, not a calibrated confidence test.'},
        checked_predictor_sources=checked_sources,
        provenance={'inputs': provenance, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        limitations=[
            'The successful future SAC commands are retrospective evidence and unavailable to the online filter.',
            'Recorded state parity covers x,y,heading,surge and policy action; the old OFF trace lacks full sway/yaw/actuator observations.',
            'Current target truth only scores estimation error; no true target future is supplied or assumed.',
            'Scoring actual SAC endpoints against the captured onboard constant-velocity target separates own-motion error, but is not actual future target separation.',
            'Actual-trajectory checks use .5s endpoints and omit terminal extension; prediction checks use .125s samples and the original terminal rules.',
            'Motion-axis geometry sensitivities are hypotheses, not validated detections: course can differ from hull heading, visible returns may omit both faces, and a single case cannot establish a safe replacement rule.',
            'This is one completed case in an ongoing campaign, not a population performance claim.'])
    if args.tag:
        out = BASE / 'v10_iterations/audits' / args.tag
        out.mkdir(parents=True, exist_ok=False)
        with (out / 'audit.json').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(clean(output), indent=2, allow_nan=False) + '\n')
        with (out / 'prediction_components.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timeline[0]))
            writer.writeheader(); writer.writerows(clean(timeline))
        lines = ['# CH-HO-CV-073 saved-command audit', '',
            f"SAC reached the goal in {results['off']['steps']} decisions; V14 collided with the target in {len(rows)}.", '',
            f"The first command difference is decision {row['step']}. Earlier commands and recorded states match exactly, as do the current recorded state and SAC action. SAC issued rudder {off[index]['rudder_command']:.6f}, RPM {off[index]['signed_rpm_command']:.6f}; V14 issued rudder {row['rudder_command']:.6f}, RPM {row['signed_rpm_command']:.6f}.", '',
            '| Saved sequence | Hard passing | Minimum clearance (m) | First violation (s) |',
            '|---|---:|---:|---:|']
        for name, score in scored.items():
            first = score['first_violation_s']
            lines.append(f"| {name} | {score['hard_safe']} | {score['clearance_m']:.6f} | {first if math.isfinite(first) else 'none'} |")
        lines += ['', '| Sequence | Component | Horizon minimum (m) | At (s) | Terminal minimum (m) |',
                  '|---|---|---:|---:|---:|']
        for name, score in scored.items():
            for component, detail in score['components']['horizon'].items():
                terminal = score['components']['terminal'].get(component)
                when = detail['minimum_time_s']
                lines.append(f"| {name} | {component} | {detail['minimum_clearance_m']:.6f} | {when if when is not None else 'outside point broadphase'} | {terminal if terminal is not None else 'not checked'} |")
        lines += ['', f"The current onboard target centre error is {target_errors['position_error_m']:.3f}m, velocity error {target_errors['velocity_error_mps']:.3f}m/s, and hull-axis error {target_errors['axis_error_deg']:.2f}°. No persistent hypothesis was published at this decision." if not row['filter']['track_persistence']['added_hypotheses'] else '', '',
            'Both logged proposal and same-tail policy clearances reproduce to 1e-12. Every reported prediction component is checked against the unchanged evaluator.', '',
            'The known successful future SAC commands fail the captured target check, even when actual SAC endpoints replace predicted own motion. A larger maneuver search alone cannot make this same trajectory pass this target representation. This localizes the discrepancy to target representation/forecast or clearance conservatism; the old OFF trace does not contain future target truth to separate those fully.', '',
            '| Onboard geometry sensitivity | Current centre error (m) | Axis error (deg) | Saved SAC sequence margin (m) | Hard passing |',
            '|---|---:|---:|---:|---:|',
            *[f"| {k} | {v['current_centre_error_m']:.6f} | {v['current_axis_error_deg']:.3f} | {v['clearance_m']:.6f} | {v['hard_safe']} |" for k, v in geometry_sensitivity.items()], '',
            'Geometry sensitivities retain the measured velocity and original collision checker. Motion supplies the orientation, and the known hull dimensions complete unseen faces using the existing `src/tracking.py:hull_fit_centre` construction. Current truth scores errors after construction and never enters a forecast. This is diagnostic evidence, not an implemented replacement or a claim that course always reveals hull heading.', '',
            '[audit.json](audit.json) contains component minima, actual-endpoint scoring, model errors, full first-decision diagnostics, trace/source/archive hashes and exact provenance. [prediction_components.csv](prediction_components.csv) contains every predicted sample.', '',
            'Reproduce with `python -B tools/diagnostics/safety/audit_ch_ho_policy.py --tag NEW_TAG`.', '',
            'Limitations:', '', *['- ' + x for x in output['limitations']], '']
        with (out / 'audit.md').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write('\n'.join(lines))
        print('OUTPUT', out)
    print(json.dumps(clean({'first_step': row['step'], 'scores': {k: {x: v[x] for x in ('hard_safe', 'clearance_m', 'components')} for k, v in scored.items()},
        'actual_sac_endpoint_scoring': actual_scoring, 'model_errors': errors,
        'target_errors': target_errors, 'current_raw_geometry': geometry_history[-1],
        'geometry_sensitivity': geometry_sensitivity}), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
