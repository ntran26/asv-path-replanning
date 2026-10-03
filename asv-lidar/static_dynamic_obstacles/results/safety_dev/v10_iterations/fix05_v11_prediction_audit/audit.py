"""Saved-trace prediction scoring only: no environment, reset, step or episodes.

Run from the static_dynamic_obstacles project root with Python -B.
Truth labels score recorded predictions and an explicitly labelled target-only
oracle sensitivity check. They never enter a controller or generate actions.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np


ROOT = Path.cwd()
RUN = ROOT / 'results/safety_dev/v10_iterations/persistence_three'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from tools.diagnostics.safety.suite_status import read_shared_bytes


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def main():
    trace_bytes = read_shared_bytes(RUN / 'traces/001_v11.jsonl')
    manifest = json.loads(read_shared_bytes(RUN / 'manifest.json'))
    rows = [json.loads(line) for line in trace_bytes.splitlines() if line]
    result = json.loads(read_shared_bytes(RUN / 'attempts/001_result.json'))
    dependencies = ['src/constants.py', 'src/constant_temp.py', 'src/classical/common.py',
                    'src/safety_v2.py', 'src/safety_v3.py', 'src/ship.py',
                    'src/reference_controller.py', 'bluefin/dynamics.py']
    matched = {}
    for name in dependencies:
        digest = hashlib.sha256(read_shared_bytes(ROOT / name)).hexdigest()
        if digest != manifest['source_sha256'][name]:
            raise ValueError('Prediction source differs from evaluated source: ' + name)
        matched[name] = digest
    import constants as cfg
    restored = []
    for key, text in manifest['constants'].items():
        try: value = ast.literal_eval(text)
        except (ValueError, SyntaxError): continue
        setattr(cfg, key, value)
        restored.append(key)
    from classical import common as cc
    import safety_v2 as v2
    import safety_v3 as v3
    import ship
    from reference_controller import hull_separation

    def snapshot(row):
        saved = row['diagnostic_decision']['snapshot']
        values = {key: value for key, value in saved.items() if key != 'units'}
        for key in ['tangent', 'right', 'centre', 'points', 'edges_a', 'edges_b']:
            values[key] = np.asarray(values[key], dtype=float)
        values['tracks'] = [cc.TrackView(t['id'], np.array(t['position']), np.array(t['velocity']),
                                        t['heading']) for t in values['tracks']]
        return cc.Snapshot(**values)

    def actuator(row):
        saved = row['diagnostic_decision']['actuators_before_decision']
        act = cc.Actuators()
        act.servo, act.executed = saved['servo'], saved['executed']
        act.buffer = copy.deepcopy(saved['buffer'])
        return act

    def command(row):
        rpm = float(row['signed_rpm_command'])
        throttle = np.nan if rpm < 0. else (rpm - cfg.CRUISE_RPM) / cfg.RPM_DELTA
        if rpm < 0. and not np.isclose(rpm, v2.ASTERN_RPM):
            raise ValueError('Unexpected partial astern command')
        if np.isfinite(throttle) and abs(throttle) > 1. + 1e-9:
            raise ValueError('Issued command is outside the recorded throttle map')
        return np.array([row['rudder_command'], throttle])

    def body_forecast(snap, act, row, nominal_brake=False):
        state = np.zeros((7, 1))
        state[:, 0] = [max(0., snap.u), snap.v, snap.r, snap.heading, act.servo, snap.x, snap.y]
        params = {key: np.array([value]) for key, value in cc.IDENTIFIED.items()}
        delta = np.array([-cc.MAX_RUDDER_RAD * row['rudder_command']])
        pending = ([np.array([x]) for x in act.buffer] if act.buffer is not None
                   else [delta.copy() for _ in range(cc.DELAY_STEPS)])
        rpm = float(row['signed_rpm_command'])
        efficiency = ship.REVERSE_THRUST_EFFICIENCY if nominal_brake else v2.BRAKE_EFFICIENCY
        decel = ship.braking_thrust(rpm, efficiency=efficiency) / ship.M11
        for j in range(cc.SUBSTEPS):
            pending.append(delta)
            state = cc.dyn.rk4_step(state, np.array([max(0., rpm)]), pending.pop(0), params, cc.PRED_DT)
            if rpm < 0. and (nominal_brake or j * cc.PRED_DT >= v2.BRAKE_DELAY_S):
                state[0, 0] = max(0., state[0, 0] - decel * cc.PRED_DT)
        return state[:, 0]

    def target_truth(row):
        target = row['diagnostic_before']['truth_scoring_only']['targets'][0]
        return cc.TrackView(0, np.array([target['x'], target['y']]), np.array(target['velocity']),
                            np.radians(target['heading']))

    def endpoint_gap(position, heading, target):
        return float(hull_separation(np.array(position)[None, None], np.array([[heading]]),
                     target.position[None, None], target.heading, margin=cc.HULL_MARGIN)[0, 0])

    timeline, tracks, plans, replay = [], [], [], []
    for index, row in enumerate(rows):
        snap, act = snapshot(row), actuator(row)
        truth = row['diagnostic_before']['truth_scoring_only']['own']
        target = target_truth(row)
        ro = v3.rollout_seq(snap, act, command(row)[None, None])
        state = body_forecast(snap, act, row)
        np.testing.assert_allclose(state[5:], ro.positions[-1, 0], atol=1e-12)
        np.testing.assert_allclose(state[3], ro.headings[-1, 0], atol=1e-12)
        next_truth = (rows[index+1]['diagnostic_before']['truth_scoring_only']['own']
                      if index+1 < len(rows) else None)
        actual_position = np.array(row['post_state'][:2])
        pre_position = np.array(row['pre_state'][:2])
        increment_error = (state[5:] - snap.position) - (actual_position - pre_position)
        heading_error = np.degrees(cc.wrap_pi(state[3] - np.radians(row['post_state'][2])))
        heading_increment_error = np.degrees(cc.wrap_pi(
            (state[3] - snap.heading) - np.radians(row['post_state'][2] - row['pre_state'][2])))
        nominal = body_forecast(snap, act, row, nominal_brake=True)
        end_target = (target_truth(rows[index+1]) if index+1 < len(rows) else
                      cc.TrackView(0, target.position + cfg.UPDATE_RATE*target.velocity,
                                   target.velocity, target.heading))
        estimated_gaps = [float(cc.target_gap(ro.positions, ro.headings, ro.times, t)[-1, 0])
                          for t in snap.tracks]
        saved_filter = row['filter']
        record = dict(step=row['step'], time_s=row['diagnostic_before']['elapsed_time'],
            why=saved_filter.get('why'), mode=saved_filter.get('mode'), changed=row['changed'],
            rudder=row['rudder_command'], rpm=row['signed_rpm_command'], requested_brake=row['brake'],
            any_safe=saved_filter.get('any_safe'), checked_clearance=saved_filter.get('checked_clearance'),
            estimated_track_ids=[t.id for t in snap.tracks],
            predicted_end_position=state[5:], actual_end_position=actual_position,
            position_error_m=float(np.linalg.norm(state[5:]-actual_position)),
            displacement_error_m=float(np.linalg.norm(increment_error)),
            heading_error_deg=float(heading_error), heading_increment_error_deg=float(heading_increment_error),
            initial_u_error_mps=snap.u-truth['u_body'], predicted_end_u_mps=float(state[0]),
            actual_end_u_mps=row['post_state'][3], u_error_mps=float(state[0]-row['post_state'][3]),
            u_increment_error_mps=float(state[0]-snap.u-(row['post_state'][3]-truth['u_body'])),
            nominal_reverse_u_error_mps=float(nominal[0]-row['post_state'][3]),
            v_error_mps=None if next_truth is None else float(state[1]-next_truth['v_body']),
            yaw_error_deg_s=None if next_truth is None else float(np.degrees(state[2])-next_truth['asv_w']),
            minimum_estimated_end_target_gap_m=min(estimated_gaps, default=None),
            truth_target_gap_at_predicted_own_endpoint_m=endpoint_gap(state[5:], state[3], end_target),
            truth_endpoint_target_gap_m=endpoint_gap(actual_position, np.radians(row['post_state'][2]), end_target),
            final_target_endpoint_extrapolated=index+1 == len(rows))
        timeline.append(record)
        for t in snap.tracks:
            tracks.append(dict(step=row['step'], id=t.id, position=t.position, velocity=t.velocity,
                truth_position=target.position, truth_velocity=target.velocity,
                position_error_m=float(np.linalg.norm(t.position-target.position)),
                velocity_error_mps=float(np.linalg.norm(t.velocity-target.velocity)),
                heading_error_deg=float(np.degrees(cc.wrap_pi(t.heading-target.heading))),
                prediction_error_at_2s_m=float(np.linalg.norm(t.position+2*t.velocity-(target.position+2*target.velocity))),
                prediction_error_at_8s_m=float(np.linalg.norm(t.position+8*t.velocity-(target.position+8*target.velocity)))))
        if row['plan'] is not None:
            plan = np.asarray(row['plan'], dtype=float)
            pr = v3.rollout_seq(snap, act, plan[None])
            first, clear = v2.SafetyFilterV2._evaluate(None, snap, pr)
            target_only = copy.copy(snap)
            target_only.tracks = [target]  # Sensitivity scoring; never chooses a command.
            oracle_first, oracle_clear = v2.SafetyFilterV2._evaluate(None, target_only, pr)
            components = {'static': float((cc.point_clearance(pr.positions, pr.headings, snap.points,
                reach=cc.HALF_L+1.)-v2.GAP_STATIC_M).min()),
                'boundary': float((cc.boundary_clearance(pr.positions, pr.headings, snap.edges_a,
                    snap.edges_b)-v2.GAP_BOUNDARY_M).min())}
            for t in snap.tracks:
                components[str(t.id)] = float((cc.target_gap(pr.positions, pr.headings, pr.times, t)
                                                -v2.GAP_TARGET_M).min())
            plans.append(dict(step=row['step'], first_violation_s=float(first[0]), clearance=float(clear[0]),
                logged_clearance=saved_filter.get('checked_clearance'), components=components,
                oracle_target_scoring_first_violation_s=float(oracle_first[0]),
                oracle_target_scoring_clearance=float(oracle_clear[0])))
        if row['step'] in (13, 14, 18, 19, 23):
            sequences = np.array([command(future) for future in rows[index:]])[None]
            conditioned = v3.rollout_seq(snap, act, sequences)
            for horizon in (.5, 1., 2., 4.):
                offset = int(round(horizon/cfg.UPDATE_RATE))
                if index+offset > len(rows): continue
                j = offset*cc.SUBSTEPS-1
                actual = rows[index+offset-1]['post_state']
                replay.append(dict(start_step=row['step'], horizon_s=horizon,
                    position_error_m=float(np.linalg.norm(conditioned.positions[j, 0]-actual[:2])),
                    heading_error_deg=float(np.degrees(cc.wrap_pi(conditioned.headings[j, 0]-np.radians(actual[2]))))))

    def error_stats(key, selection):
        values = [abs(row[key]) for row in selection if row[key] is not None]
        return {'mae': float(np.mean(values)), 'maximum_absolute': float(max(values)), 'count': len(values)}
    summary = {'steps': len(rows), 'outcome': result['outcome'], 'changed_steps': [r['step'] for r in rows if r['changed']],
        'first_intervention_time_s': next(r['time_s'] for r in timeline if r['changed']),
        'last_currently_passing_stored_plan_step': max(p['step'] for p in plans if np.isposinf(p['first_violation_s']) and p['clearance'] >= 0),
        'target_truth_velocity_constant': all(np.allclose(target_truth(r).velocity, target_truth(rows[0]).velocity) for r in rows),
        'errors': {name: {key: error_stats(key, selection) for key in
                    ['position_error_m', 'displacement_error_m', 'heading_error_deg', 'heading_increment_error_deg',
                     'u_error_mps', 'nominal_reverse_u_error_mps', 'v_error_mps', 'yaw_error_deg_s']}
                   for name, selection in [('all', timeline), ('astern', [r for r in timeline if r['rpm'] < 0]),
                                            ('non_astern', [r for r in timeline if r['rpm'] >= 0])]}}
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope='saved-trace offline scoring; zero new episodes',
        case=manifest['cases'][0], trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
        manifest_sha256=hashlib.sha256(read_shared_bytes(RUN/'manifest.json')).hexdigest(),
        checked_prediction_source_hashes=matched, restored_constant_count=len(restored),
        model='recorded onboard snapshot and actuator history; identified parameters only',
        truth_usage='scoring, plus labelled target-only sensitivity of the already-recorded plan',
        final_endpoint_target='CV extrapolation by .5s; target velocity constant across all recorded pre-states',
        summary=summary, timeline=timeline, track_errors=tracks, stored_plan_checks=plans,
        conditioned_on_recorded_future_commands=replay)
    path = OUT/'audit.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(clean(output), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(clean(summary), indent=2))
    print('PLAN_CHECKS', json.dumps(clean(plans[-7:])))
    print('FINAL_TIMELINE', json.dumps(clean(timeline[-12:])))
    print('COMMAND_CONDITIONED',json.dumps(clean(replay)))


if __name__ == '__main__':
    main()
