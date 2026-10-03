"""Saved-trace scoring only; no environment, policy or episode calls."""
import ast
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src'))
OUT=Path(__file__).parent
BASE=ROOT/'results/safety_dev'
RUN=BASE/'v10_iterations/motion_axis_probe5'


def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x


def main():
    paths={'v16':RUN/'traces/001_v16.jsonl','off':BASE/'v9_paired_pilot/traces/001_off.jsonl',
           'v15':BASE/'v10_iterations/conditional_prefix32/traces/001_v15.jsonl',
           'broad_prefix':BASE/'v10_iterations/persistent_prefix32/traces/001_v14.jsonl'}
    rows={k:read(p) for k,p in paths.items()};manifest=json.loads((RUN/'manifest.json').read_text())
    result=json.loads((RUN/'attempts/001_result.json').read_text())
    assert result['case']=='TS2:BAS-HO-NC-059' and result['steps']==len(rows['v16'])
    other=json.loads((BASE/'v9_paired_pilot/manifest.json').read_text())
    assert all(manifest[k]==other[k] for k in ['checkpoint_sha256','config_sha256'])
    cases=[next(c for c in m['cases'] if c['case']==result['case']) for m in [manifest,other]]
    assert all(cases[0][k]==cases[1][k] for k in ['seed','scenario_sha256'])
    dependencies=['src/constants.py','src/constant_temp.py','src/classical/common.py','src/safety_v2.py',
                  'src/safety_v3.py','src/safety_v7.py','src/safety_v10.py','src/ship.py','bluefin/dynamics.py']
    hashes={k:sha(ROOT/k) for k in dependencies}
    assert all(v==manifest['source_sha256'][k] for k,v in hashes.items())
    import constants as cfg
    for k,v in manifest['constants'].items():
        try:v=ast.literal_eval(v)
        except (ValueError,SyntaxError):continue
        setattr(cfg,k,v)
    from classical import common as cc
    import safety_v2 as v2
    import safety_v3 as v3
    import safety_v7 as v7
    index=next(i for i,(a,b) in enumerate(zip(rows['v16'],rows['off'])) if
               [a['rudder_command'],a['signed_rpm_command']]!=[b['rudder_command'],b['signed_rpm_command']])
    row=rows['v16'][index];assert index==19
    assert row['pre_state']==rows['off'][index]['pre_state'] and row['policy_action']==rows['off'][index]['policy_action']
    prefix_manifest=json.loads((BASE/'v10_iterations/persistent_prefix32/manifest.json').read_text())
    assert all(manifest[k]==prefix_manifest[k] for k in ['checkpoint_sha256','config_sha256'])
    prefix_case=next(c for c in prefix_manifest['cases'] if c['case']==result['case'])
    assert all(cases[0][k]==prefix_case[k] for k in ['seed','scenario_sha256'])
    prefix_first=next(i for i,(a,b) in enumerate(zip(rows['broad_prefix'],rows['off'])) if
                     [a['rudder_command'],a['signed_rpm_command']]!=[b['rudder_command'],b['signed_rpm_command']])
    prefix_row=rows['broad_prefix'][prefix_first]
    assert prefix_first==index
    assert all(prefix_row[k]==row[k] for k in ['pre_state','policy_action','rudder_command','signed_rpm_command'])
    assert prefix_row['diagnostic_decision']['snapshot']==row['diagnostic_decision']['snapshot']
    d=copy.deepcopy(row['diagnostic_decision']['snapshot']);d.pop('units')
    for k in ['tangent','right','centre','points','edges_a','edges_b']:d[k]=np.asarray(d[k],dtype=float)
    d['tracks']=[cc.TrackView(t['id'],np.asarray(t['position']),np.asarray(t['velocity']),t['heading']) for t in d['tracks']]
    snap=cc.Snapshot(**d);a=row['diagnostic_decision']['actuators_before_decision'];act=cc.Actuators()
    act.servo,act.buffer,act.executed=a['servo'],a['buffer'],a['executed']
    future=rows['off'][index:index+16]
    seq=np.asarray([[r['rudder_command'],(r['signed_rpm_command']-cfg.CRUISE_RPM)/cfg.RPM_DELTA] for r in future])[None]
    assert seq.shape==(1,16,2)
    before=row['diagnostic_before'];truth=before['truth_scoring_only'];own=truth['own'];target=truth['targets'][0]
    q=cc.TrackView(-9000,np.array([target['x'],target['y']]),np.asarray(target['velocity']),math.radians(target['heading']))
    raw=row['diagnostic_decision']['raw_ego_u_v_yaw_rad_s']
    source=row['filter']['track_persistence']['hypotheses'][0]['source_id']
    track=next(t for t in before['onboard']['raw_tracks'] if t['id']==source)
    variants={
        'onboard':snap,
        'raw_yaw_only':replace(snap,r=raw[2]),
        'true_current_ego_scoring_only':replace(snap,x=own['asv_x'],y=own['asv_y'],heading=math.radians(own['asv_h']),
                    u=own['u_body'],v=own['v_body'],r=math.radians(own['asv_w'])),
        'true_current_target_CV_scoring_only':replace(snap,tracks=[q]),
        'onboard_fresh_fit_centre_only':replace(snap,tracks=[replace(snap.tracks[0],position=np.asarray(track['last_fit_centre']))]),
        'onboard_base_centroid_and_velocity':replace(snap,tracks=[cc.TrackView(source,np.asarray(track['position']),np.asarray(track['velocity']),snap.tracks[0].heading)]),
        'true_target_centre_only_scoring':replace(snap,tracks=[replace(snap.tracks[0],position=q.position)]),
        'true_target_velocity_only_scoring':replace(snap,tracks=[replace(snap.tracks[0],velocity=q.velocity)]),
    }
    def components(s,ro):
        arrays={'static':cc.point_clearance(ro.positions,ro.headings,s.points,reach=cc.HALF_L+1)-v2.GAP_STATIC_M,
                'boundary':cc.boundary_clearance(ro.positions,ro.headings,s.edges_a,s.edges_b)-v2.GAP_BOUNDARY_M}
        arrays.update({f'target:{t.id}':cc.target_gap(ro.positions,ro.headings,ro.times,t)-v2.GAP_TARGET_M for t in s.tracks})
        return {k:{'minimum':float(v.min()),'at_s':float(ro.times[np.argmin(v[:,0])]),
                   'first_violation_s':float(ro.times[np.flatnonzero(v[:,0]<0)[0]]) if (v<0).any() else None} for k,v in arrays.items()}
    scores={};pred={}
    for k,s in variants.items():
        ro=v3.rollout_seq(s,act,seq);f,c=v2.SafetyFilterV2._evaluate(None,s,ro);pred[k]=ro
        scores[k]={'minimum_including_terminal':c[0],'first_violation_s':f[0], 'hard_safe':bool(np.isposinf(f[0])),
                   'components_within_horizon':components(s,ro)}
    policy_tail=v7.replace_first_action(np.asarray(row['plan'],dtype=float),np.asarray(row['policy_action']))
    parent=policy_tail.copy();parent[0]=[row['rudder_command'],np.nan if row['signed_rpm_command']<0 else (row['signed_rpm_command']-cfg.CRUISE_RPM)/cfg.RPM_DELTA]
    f,c=v2.SafetyFilterV2._evaluate(None,snap,v3.rollout_seq(snap,act,np.stack([parent,policy_tail])))
    assert np.allclose(c,[row['filter']['v10_proposed_clearance'],row['filter']['v10_policy_tail_clearance']],rtol=0,atol=2e-6)
    actual=cc.Rollout(np.asarray([r['post_state'][:2] for r in future])[:,None,:],np.radians([r['post_state'][2] for r in future])[:,None],
                      np.asarray([r['post_state'][3] for r in future])[:,None],.5*np.arange(1,17))
    ix=np.arange(cc.SUBSTEPS-1,len(pred['onboard'].times),cc.SUBSTEPS)
    target_future=[]
    for r in rows['v16'][index:]:
        t=r['diagnostic_before']['truth_scoring_only']['targets'][0]
        dt=r['diagnostic_before']['elapsed_time']-before['elapsed_time']
        target_future.append({'step':r['step'],'dt_s':dt,'position_error_from_current_CV':np.linalg.norm(q.position+dt*q.velocity-[t['x'],t['y']]),
                              'heading_change_deg':math.degrees(cc.wrap_pi(math.radians(t['heading'])-q.heading)), 'speed':t['speed']})
    evolution=[]
    for r in rows['v16']:
        stats=r['filter'].get('track_persistence',{});ts=r['diagnostic_before']['truth_scoring_only']['targets'][0]
        for h in stats.get('hypotheses',[]):
            if h['source_id']!=source:continue
            rt=next((t for t in r['diagnostic_before']['onboard']['raw_tracks'] if t['id']==source),None)
            evolution.append({'step':r['step'],'age_s':h['age_s'],
                'truth_heading_deg':ts['heading'],
                'persistent_heading_deg':next((math.degrees(t['heading']) for t in r['diagnostic_decision']['snapshot']['tracks'] if t['id']==h['id']),None),
                'current_fit_heading_deg':None if rt is None else rt['last_fit_heading_deg'],
                'raw_velocity':None if rt is None else rt['velocity'],
                'rejections':stats.get('rejections'),
                'persistent_position_error':np.linalg.norm(np.asarray(h['position'])-[ts['x'],ts['y']]),
                'persistent_velocity_error':np.linalg.norm(np.asarray(h['velocity'])-ts['velocity']),
                'fresh_fit_position_error':None if rt is None or rt['last_fit_centre'] is None else np.linalg.norm(np.asarray(rt['last_fit_centre'])-[ts['x'],ts['y']]),
                'updates':stats.get('updates')})
    report={'scope':'Saved completed first-divergence diagnostic. No episodes, controller search or new rule. Future OFF commands are noncausal scoring inputs only.',
       'case':result['case'],'result':{k:result[k] for k in ['outcome','steps','safety_v2_steps']},'first_divergence':index+1,
       'off_prestate_policy_match':True,'v15_v16_recorded_commands_states_equal':all(a[k]==b[k] for a,b in zip(rows['v15'],rows['v16']) for k in ['pre_state','post_state','policy_action','rudder_command','signed_rpm_command']),
       'inputs':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths.values()},'model_sources_match_manifest':hashes,
       'decision_filter':row['filter'],'represented_tracks':row['diagnostic_decision']['snapshot']['tracks'],
       'current_raw_target':{k:v for k,v in track.items() if k!='history'},'target_current_truth_scoring_only':target,
       'current_ego':{'truth_u_v_r_deg_s':[own['u_body'],own['v_body'],own['asv_w']],
                      'raw_u_v_r_deg_s':[raw[0],raw[1],math.degrees(raw[2])],'filtered_u_v_r_deg_s':[snap.u,snap.v,math.degrees(snap.r)]},
       'known_successful_off_steps':[r['step'] for r in future],'known_successful_continuation_scores':scores,
       'parent_vs_same_tail_reproduction':{'first':f,'clear':c},'actual_off_own_endpoints_scored_with_onboard_geometry':components(snap,actual),
       'existing_broad_prefix':{'first_divergence':prefix_first+1,'same_snapshot_policy_prestate_and_command':True,
           'prefix_search':prefix_row['filter']['policy_prefix_search'],
           'result':json.loads((BASE/'v10_iterations/persistent_prefix32/attempts/001_result.json').read_text())},
       'existing_any_feasible_flag_at_saved_decision':{'would_preserve_this_policy_action':bool(np.isposinf(f[1]) and c[1]>=0),
           'reason':'Existing safety_v10.py passes the exact checked policy-tail plan when prefer_any_feasible_policy=True; no episode outcome implied.'},
       'own_endpoint_position_error_m':np.linalg.norm(pred['onboard'].positions[ix,0]-actual.positions[:,0],axis=1),
       'own_endpoint_heading_error_deg':np.degrees(np.abs(cc.wrap_pi(pred['onboard'].headings[ix,0]-actual.headings[:,0]))),
       'v16_recorded_target_future':target_future,'persistent_source_evolution':evolution,
       'limitations':['OFF lacks future target states; current-target CV sensitivities do not assert the unrecorded counterfactual future.',
                     'Actual-own endpoint scoring uses .5s sampling, not continuous contact checks.',
                     'A current passing model backup does not establish a closed-loop rescue or justify a general gate from one case.']}
    (OUT/'audit.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(clean({k:report[k] for k in ['current_ego','known_successful_continuation_scores','parent_vs_same_tail_reproduction',
         'actual_off_own_endpoints_scored_with_onboard_geometry','v16_recorded_target_future']}),indent=2))


if __name__=='__main__':main()
