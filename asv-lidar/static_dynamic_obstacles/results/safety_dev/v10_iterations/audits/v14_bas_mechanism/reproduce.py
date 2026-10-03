"""Saved completed-case scoring only; never construct/reset/step an environment."""
import ast
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np

OUT = Path(__file__).parent
BASE = ROOT / 'results/safety_dev'
RUN = BASE / 'v10_iterations/persistent_preference_probe4'
OLD = BASE / 'v10_iterations/consistent_track_probe8'
COMBINED = BASE / 'v10_iterations/persistent_prefix32'
OFF = BASE / 'v9_paired_pilot'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trace(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def command(row):
    return [row['rudder_command'], row['signed_rpm_command']]


def compare(rows, reference):
    fields = ['pre_state', 'post_state', 'policy_action', 'rudder_command', 'signed_rpm_command']
    return {
        'steps': len(rows), 'reference_steps': len(reference),
        'changed_steps': [r['step'] for r in rows if r.get('changed', False)],
        'current_policy_preserved_decisions': sum(not r.get('changed', False) for r in rows),
        'first_command_divergence': next((r['step'] for r, b in zip(rows, reference)
                                          if command(r) != command(b)), None),
        'exact_full_field_parity': {key: len(rows) == len(reference) and all(
            r[key] == b[key] for r, b in zip(rows, reference)) for key in fields},
        'first_field_divergence': {key: next((r['step'] for r, b in zip(rows, reference)
                                            if r[key] != b[key]), None) for key in fields},
    }


def main():
    manifests = {name: json.loads((folder / 'manifest.json').read_text())
                 for name, folder in [('v14', RUN), ('v13', OLD), ('combined', COMBINED), ('off', OFF)]}
    case = 'TS2:BAS-CR-RE-072'
    identity = {name: next(c for c in m['cases'] if c['case'] == case)
                for name, m in manifests.items()}
    for key in ['seed', 'scenario_sha256']:
        assert len({v[key] for v in identity.values()}) == 1
    for key in ['checkpoint_sha256', 'config_sha256']:
        assert len({m[key] for m in manifests.values()}) == 1
    inputs = {'off': OFF / 'traces/006_off.jsonl', 'v13': OLD / 'traces/002_v13.jsonl',
              'v14': RUN / 'traces/001_v14.jsonl', 'combined': COMBINED / 'traces/002_v14.jsonl'}
    rows = {name: read_trace(path) for name, path in inputs.items()}
    for folder, attempt, name in [(RUN, 1, 'v14'), (OLD, 2, 'v13'), (COMBINED, 2, 'combined')]:
        result = json.loads((folder / f'attempts/{attempt:03d}_result.json').read_text())
        assert result['case'] == case and result['outcome'] == 'goal'
        assert result['steps'] == len(rows[name])

    dependencies = ['src/constants.py', 'src/constant_temp.py', 'src/classical/common.py',
                    'src/safety_v2.py', 'src/safety_v3.py', 'src/ship.py', 'bluefin/dynamics.py']
    checked_sources = {}
    for name in dependencies:
        actual = digest(ROOT / name)
        assert actual == manifests['v14']['source_sha256'][name], name
        assert actual == manifests['v13']['source_sha256'][name], name
        checked_sources[name] = actual
    import constants as cfg
    for key, value in manifests['v14']['constants'].items():
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            continue
        setattr(cfg, key, value)
    from classical import common as cc
    import safety_v2 as v2
    import safety_v3 as v3

    def snapshot(row):
        values = copy.deepcopy(row['diagnostic_decision']['snapshot'])
        values.pop('units', None)
        for key in ['tangent', 'right', 'centre', 'points', 'edges_a', 'edges_b']:
            values[key] = np.asarray(values[key], dtype=float)
        values['tracks'] = [cc.TrackView(t['id'], np.asarray(t['position']),
                                         np.asarray(t['velocity']), t['heading']) for t in values['tracks']]
        return cc.Snapshot(**values)

    sequences = np.asarray([[r['rudder_command'], (r['signed_rpm_command'] - cfg.CRUISE_RPM) / cfg.RPM_DELTA]
                            for r in rows['off'][7:23]])[None]
    decision8 = {}
    for name in ['v13', 'v14']:
        row = rows[name][7]
        saved = row['diagnostic_decision']['actuators_before_decision']
        act = cc.Actuators()
        act.servo, act.executed, act.buffer = saved['servo'], saved['executed'], saved['buffer']
        snap = snapshot(row)
        ro = v3.rollout_seq(snap, act, sequences)
        first, clear = v2.SafetyFilterV2._evaluate(None, snap, ro)
        truth = row['diagnostic_before']['truth_scoring_only']['targets'][0]
        target = snap.tracks[0]
        decision8[name] = {
            'identical_recorded_pre_state_and_policy_to_off': row['pre_state'] == rows['off'][7]['pre_state']
                and row['policy_action'] == rows['off'][7]['policy_action'],
            'recorded_reason': row['filter']['why'],
            'recorded_checked_clearance_m': row['filter']['checked_clearance'],
            'saved_successful_16_command_sequence': {
                'first_violation_s': float(first[0]) if math.isfinite(first[0]) else None,
                'minimum_clearance_m': float(clear[0]), 'hard_safe': bool(np.isinf(first[0])),
            },
            'target_view': {'id': target.id, 'position': target.position.tolist(),
                            'velocity': target.velocity.tolist(), 'heading_rad': target.heading},
            'current_position_error_m_scoring_only': float(np.linalg.norm(target.position - [truth['x'], truth['y']])),
            'current_velocity_error_mps_scoring_only': float(np.linalg.norm(target.velocity - truth['velocity'])),
            'track_persistence': row['filter']['track_persistence'],
        }
    expected = json.loads((BASE / 'v10_iterations/bas_re_persistent_replacement_audit.json').read_text())
    assert decision8['v14']['target_view']['position'] == expected['current_state_errors'][0]['persistent_position']
    assert decision8['v14']['target_view']['velocity'] == expected['current_state_errors'][0]['persistent_velocity']
    assert abs(decision8['v14']['saved_successful_16_command_sequence']['minimum_clearance_m']
               - expected['persistent_replacement']['minimum_clearance_m']) < 1e-12

    outcomes = []
    old_results = list(csv.DictReader((OLD / 'episodes.csv').open(newline='')))
    for current in csv.DictReader((RUN / 'episodes.csv').open(newline='')):
        previous = next(r for r in old_results if r['case'] == current['case'] and r['mode'] == 'v13')
        current_rows = read_trace(RUN / f"traces/{int(current['attempt']):03d}_v14.jsonl")
        previous_rows = read_trace(OLD / f"traces/{int(previous['attempt']):03d}_v13.jsonl")
        outcomes.append({'case': current['case'], 'v13_outcome': previous['outcome'], 'v14_outcome': current['outcome'],
                         'v13_steps': int(previous['steps']), 'v14_steps': int(current['steps']),
                         'v13_interventions': int(previous['safety_v2_steps']),
                         'v14_interventions': int(current['safety_v2_steps']),
                         'v14_vs_v13_trace': compare(current_rows, previous_rows)})

    report = {
        'scope': 'Saved completed-case analysis only; no episodes, policy calls, environment calls or controller edits. Combined32 campaign is ongoing; only its completed BAS case is included.',
        'case': case,
        'provenance': {'same_seed_scene_checkpoint_config': True, 'seed': identity['v14']['seed'],
                       'scenario_sha256': identity['v14']['scenario_sha256'],
                       'input_traces': {k: {'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'sha256': digest(p)}
                                        for k, p in inputs.items()},
                       'matched_predictor_sources': checked_sources},
        'against_saved_successful_off': {name: compare(rows[name], rows['off']) for name in ['v13', 'v14', 'combined']},
        'decision8': decision8,
        'isolated_v14_only_change_step26': rows['v14'][25]['filter'],
        'combined_step26': rows['combined'][25]['filter'],
        'isolated_four_case_results': outcomes,
        'limitations': [
            'Exact state parity concerns recorded x,y,heading,surge fields; the old OFF trace lacks sway/yaw/actuator state and full observations.',
            'The future16 SAC commands are noncausal saved-data scoring only; unavailable to the online filter.',
            'Current truth scores estimate errors only. No future reactive-target truth or constant-velocity truth assumption is used.',
            'One matched case supports this mechanism; it does not establish broad safety improvement.',
        ],
    }
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'comparison': report['against_saved_successful_off'],
                      'decision8_margins': {k: v['saved_successful_16_command_sequence'] for k, v in decision8.items()},
                      'four_case_outcomes': [{k: v for k, v in x.items() if k != 'v14_vs_v13_trace'} for x in outcomes]}, indent=2))


if __name__ == '__main__':
    main()
