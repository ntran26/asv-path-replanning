"""Offline geometry-consistent centre sensitivity; zero episodes/actions.

For an observed interval [lo, hi] inside a known-size object, every enclosing
centre lies in [hi-size/2, lo+size/2]. Project the prior centre onto this set on
partially observed axes; use the observed midpoint on existing full-extent
axes. This derived geometric projection is a proposed engineering update,
not a calibrated estimator or a safety guarantee. Shape versus visible-subset
modelling background: Granstrom/Baum/Reuter, Sec. II-C and VI-C,
https://arxiv.org/html/1604.00970 . Truth is used only to score the result.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
from tools.diagnostics.safety.suite_status import read_shared_bytes
from tools.diagnostics.safety.audit_v11_prediction import clean, audit_arguments


def main():
    args = audit_arguments(__doc__, projection=True)
    RUN, OUT = args.run_dir, args.output_dir
    trace = read_shared_bytes(RUN/'traces'/args.trace)
    rows = [json.loads(line) for line in trace.splitlines() if line]
    manifest = json.loads(read_shared_bytes(RUN/'manifest.json'))
    result = json.loads(read_shared_bytes(RUN/'attempts'/f'{args.attempt:03d}_result.json'))
    if len(rows) != result['steps']:
        raise ValueError('Trace is incomplete or disagrees with its completed result')
    if not 2 <= args.step <= len(rows) or [r['step'] for r in rows] != list(range(1, len(rows)+1)):
        raise ValueError('Refresh step must have a preceding decision in a complete ordered trace')
    for row in rows:
        flags=row['filter']
        if (flags.get('calibrated_braking') or flags.get('dual_brake_sequences', 0)
                or flags.get('v10_target_turn_rate_deg_s', 0) or flags.get('countersteer_recovery_enabled')):
            raise ValueError('This bank replay supports the recorded nominal CV/weak-brake primitive configuration')
        if len(row['diagnostic_before']['truth_scoring_only']['targets']) != 1:
            raise ValueError('This scoring audit requires one recorded truth target')
    import constants as cfg
    for key, text in manifest['constants'].items():
        try: setattr(cfg, key, ast.literal_eval(text))
        except (ValueError, SyntaxError): pass
    from classical import common as cc
    import safety_v2 as v2
    import safety_v3 as v3
    import safety_v4 as v4
    import safety_v7 as v7
    sources = ['src/safety_v2.py', 'src/safety_v3.py', 'src/safety_v4.py', 'src/safety_v7.py',
               'src/classical/common.py', 'src/tracking.py', 'src/safety_track_persistence.py']
    hashes = {name: hashlib.sha256(read_shared_bytes(ROOT/name)).hexdigest() for name in sources}
    assert all(digest == manifest['source_sha256'][name] for name, digest in hashes.items())

    def snap(row):
        fields = {k: v for k,v in row['diagnostic_decision']['snapshot'].items() if k!='units'}
        for key in ['tangent','right','centre','points','edges_a','edges_b']:
            fields[key] = np.asarray(fields[key],dtype=float)
        fields['tracks'] = [cc.TrackView(t['id'],np.array(t['position']),np.array(t['velocity']),t['heading'])
                            for t in fields['tracks']]
        return cc.Snapshot(**fields)

    def act(row):
        out=cc.Actuators()
        for key,value in row['diagnostic_decision']['actuators_before_decision'].items():
            setattr(out,key,copy.deepcopy(value))
        return out

    source_id=args.source_id
    hypothesis=next(h for h in rows[args.step-1]['filter']['track_persistence']['hypotheses']
                    if h['source_id']==source_id)
    synthetic_id=hypothesis['id']
    prior_view=next(t for t in snap(rows[args.step-2]).tracks if t.id==synthetic_id)
    new_view=next(t for t in snap(rows[args.step-1]).tracks if t.id==synthetic_id)
    prior=prior_view.position+cfg.UPDATE_RATE*prior_view.velocity
    track=next(t for t in rows[args.step-1]['diagnostic_before']['onboard']['raw_tracks'] if t['id']==source_id)
    points=np.array(track['history'][-1]['points'])
    h=np.radians(track['last_fit_heading_deg'])
    axes=np.column_stack(([np.sin(h),np.cos(h)],[np.cos(h),-np.sin(h)]))
    local=points@axes
    updated=np.zeros(2)
    geometry=[]
    for j,size in enumerate([cfg.LOA,cfg.BREADTH]):
        lo,hi=local[:,j].min(),local[:,j].max()
        assert hi-lo <= size+cfg.TRACK_FIT_FULL_EXTENT_TOL_M
        full=hi-lo >= size-cfg.TRACK_FIT_FULL_EXTENT_TOL_M
        lower,upper=hi-size/2,lo+size/2
        coordinate=(lo+hi)/2 if full else float(np.clip(prior@axes[:,j],lower,upper))
        updated+=coordinate*axes[:,j]
        geometry.append(dict(axis='longitudinal' if j==0 else 'lateral',observed_extent=hi-lo,
            full_extent=full,centre_interval=[lower,upper],prior_coordinate=prior@axes[:,j],
            selected_coordinate=coordinate,old_fit_coordinate=new_view.position@axes[:,j]))
    target=rows[args.step-1]['diagnostic_before']['truth_scoring_only']['targets'][0]
    truth=np.array([target['x'],target['y']])
    centre=dict(step=args.step,source_id=source_id,prior=prior,old_fit=new_view.position,projected=updated,truth=truth,
        old_error_m=float(np.linalg.norm(new_view.position-truth)),
        projected_error_m=float(np.linalg.norm(updated-truth)),
        old_innovation_m=float(np.linalg.norm(new_view.position-prior)),
        projected_innovation_m=float(np.linalg.norm(updated-prior)),axes=geometry)

    sensitivities=[]
    for row in rows[args.step-2:]:
        original=snap(row)
        target=row['diagnostic_before']['truth_scoring_only']['targets'][0]
        truth_view=cc.TrackView(0,np.array([target['x'],target['y']]),np.array(target['velocity']),
                                np.radians(target['heading']))
        corrected=cc.TrackView(synthetic_id,updated+(row['step']-args.step)*cfg.UPDATE_RATE*new_view.velocity,
                               new_view.velocity.copy(),new_view.heading)
        variants={'onboard_all':original.tracks,
            'drop_same_source_base_only':[t for t in original.tracks if t.id!=source_id],
            'oracle_target_scoring_only':[truth_view]}
        if row['step']>=args.step:
            variants['projected_anchor_all']=[corrected if t.id==synthetic_id else t for t in original.tracks]
            variants['projected_anchor_without_same_source_base']=[corrected if t.id==synthetic_id else t
                                                        for t in original.tracks if t.id!=source_id]
        traffic=any(np.linalg.norm(t.position-original.position)<v2.ENGAGE_RANGE_M for t in original.tracks)
        candidates=[tuple(row['policy_action'])]+[(r,t) for r in v2.RUDDERS for t in v2.THROTTLES]
        candidates.append(tuple(v3.SafetyFilterV3._rejoin(None,original)))
        if traffic: candidates += [(r,np.nan) for r in v2.BRAKE_RUDDERS]
        recovery=[(r,np.nan if t is v2.BRAKE else t) for r,t in v2.RECOVERY if traffic or t is not v2.BRAKE]
        templates=v4.recovery_templates(recovery)
        bank=v4.sequence_bank(np.array(candidates),templates)
        bank_rollout=v3.rollout_seq(original,act(row),bank)
        stored=None
        if row['plan'] is not None:
            plan=v7.replace_first_action(np.asarray(row['plan'],dtype=float),row['filter']['chosen'])
            if row['brake']: plan[0,1]=np.nan
            stored=v3.rollout_seq(original,act(row),plan[None])
        record=dict(step=row['step'],logged_why=row['filter']['why'],
                    logged_clearance=row['filter'].get('checked_clearance'),variants={})
        for label,views in variants.items():
            variant=copy.copy(original)
            variant.tracks=views
            first,clear=v2.SafetyFilterV2._evaluate(None,variant,bank_rollout)
            safe=np.isposinf(first)&(clear>=0.)
            passing_trigger=safe&(clear>=v2.TRIGGER_MARGIN_M)
            item=dict(bank_plans=len(bank),hard_passing_bank_plans=int(safe.sum()),
                trigger_passing_bank_plans=int(passing_trigger.sum()),
                best_hard_passing_bank_clearance=float(clear[safe].max()) if safe.any() else None)
            if stored is not None:
                first,clear=v2.SafetyFilterV2._evaluate(None,variant,stored)
                item.update(padded_stored_first_violation_s=float(first[0]),
                            padded_stored_clearance=float(clear[0]))
            record['variants'][label]=item
        sensitivities.append(record)
    output=dict(scope='saved-state, same-plan/bank sensitivity; no episodes or action selection',
        trace_sha256=hashlib.sha256(trace).hexdigest(),source_hashes=hashes,
        case=result['case'],seed=result['seed'],working_reproducer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        projection=centre,variants=sensitivities,
        caveat='Changing recorded perception while holding own trajectories fixed is not a closed-loop rescue test. '
               'The projected centre uses only the previous onboard anchor, current onboard heading and returns. '
               'Truth-target substitution is labelled sensitivity scoring only. Old stored tails are padded '
               'with V7 convention for parity with the full-horizon logged proposal check.')
    if OUT is not None:
        path=OUT/'projection.json'
        with path.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(clean(output),indent=2,allow_nan=False)+'\n')
        print('OUTPUT',path)
    else:
        print('READ_ONLY: no output files written; use --tag NEW_TAG to save')
    print('PROJECTION',json.dumps(clean(centre)))
    for record in sensitivities:
        if record['step'] in (args.step-1,args.step,args.step+1,args.step+2,args.step+3,args.step+4,len(rows)): print(json.dumps(clean(record)))


if __name__=='__main__': main()
