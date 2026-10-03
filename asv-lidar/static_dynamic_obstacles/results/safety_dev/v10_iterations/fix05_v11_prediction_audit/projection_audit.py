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

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
RUN = ROOT/'results/safety_dev/v10_iterations/persistence_three'
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
from tools.diagnostics.safety.suite_status import read_shared_bytes
from audit import clean


def main():
    trace = read_shared_bytes(RUN/'traces/001_v11.jsonl')
    rows = [json.loads(line) for line in trace.splitlines() if line]
    manifest = json.loads(read_shared_bytes(RUN/'manifest.json'))
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

    prior_view=next(t for t in snap(rows[17]).tracks if t.id==-1000001)
    new_view=next(t for t in snap(rows[18]).tracks if t.id==-1000001)
    prior=prior_view.position+cfg.UPDATE_RATE*prior_view.velocity
    track=next(t for t in rows[18]['diagnostic_before']['onboard']['raw_tracks'] if t['id']==4)
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
    target=rows[18]['diagnostic_before']['truth_scoring_only']['targets'][0]
    truth=np.array([target['x'],target['y']])
    centre=dict(step=19,source_id=4,prior=prior,old_fit=new_view.position,projected=updated,truth=truth,
        old_error_m=float(np.linalg.norm(new_view.position-truth)),
        projected_error_m=float(np.linalg.norm(updated-truth)),
        old_innovation_m=float(np.linalg.norm(new_view.position-prior)),
        projected_innovation_m=float(np.linalg.norm(updated-prior)),axes=geometry)

    sensitivities=[]
    for row in rows[17:]:
        original=snap(row)
        target=row['diagnostic_before']['truth_scoring_only']['targets'][0]
        truth_view=cc.TrackView(0,np.array([target['x'],target['y']]),np.array(target['velocity']),
                                np.radians(target['heading']))
        corrected=cc.TrackView(-1000001,updated+(row['step']-19)*cfg.UPDATE_RATE*new_view.velocity,
                               new_view.velocity.copy(),new_view.heading)
        variants={'onboard_all':original.tracks,
            'drop_base_id4_only':[t for t in original.tracks if t.id!=4],
            'oracle_target_scoring_only':[truth_view]}
        if row['step']>=19:
            variants['projected_anchor_all']=[corrected if t.id==-1000001 else t for t in original.tracks]
            variants['projected_anchor_without_base4']=[corrected if t.id==-1000001 else t
                                                        for t in original.tracks if t.id!=4]
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
        projection=centre,variants=sensitivities,
        caveat='Changing recorded perception while holding own trajectories fixed is not a closed-loop rescue test. '
               'The projected centre uses only the previous onboard anchor, current onboard heading and returns. '
               'Truth-target substitution is labelled sensitivity scoring only. Old stored tails are padded '
               'with V7 convention for parity with the full-horizon logged proposal check.')
    path=OUT/'projection.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(clean(output),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PROJECTION',json.dumps(clean(centre)))
    for record in sensitivities:
        if record['step'] in (18,19,20,21,22,23,29): print(json.dumps(clean(record)))


if __name__=='__main__': main()
