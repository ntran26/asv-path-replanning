"""Saved completed-case diagnostic; no environments, policies or episodes."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT/'src'))
OUT = Path(__file__).parent
RUN = ROOT/'results/safety_dev/v10_iterations/conditional_prefix32'


def read(path): return [json.loads(x) for x in path.read_text().splitlines() if x]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x


def main():
    trace=RUN/'traces/014_v15.jsonl'
    offpath=ROOT/'results/safety_dev/v9_paired_pilot/traces/042_off.jsonl'
    rows,off=read(trace),read(offpath)
    manifest=json.loads((RUN/'manifest.json').read_text())
    result=json.loads((RUN/'attempts/014_result.json').read_text())
    assert result['case']=='TS2:BAS-NU-CV-070' and len(rows)==result['steps']
    dep=['src/constants.py','src/constant_temp.py','src/classical/common.py','src/safety_v2.py',
         'src/safety_v3.py','src/safety_policy_prefix.py','src/safety_v7.py','src/ship.py','bluefin/dynamics.py']
    source_hashes={name:sha(ROOT/name) for name in dep}
    assert all(value==manifest['source_sha256'][name] for name,value in source_hashes.items())
    old=json.loads((ROOT/'results/safety_dev/v9_paired_pilot/manifest.json').read_text())
    assert all(old[k]==manifest[k] for k in ['checkpoint_sha256','config_sha256'])
    cases=[next(c for c in m['cases'] if c['case']==result['case']) for m in [old,manifest]]
    assert all(cases[0][k]==cases[1][k] for k in ['seed','scenario_sha256'])
    import constants as cfg
    for key,value in manifest['constants'].items():
        try:value=ast.literal_eval(value)
        except (ValueError,SyntaxError):continue
        setattr(cfg,key,value)
    from classical import common as cc
    import safety_v2 as v2
    import safety_v3 as v3
    import safety_policy_prefix as prefix

    def snapshot(row):
        data=copy.deepcopy(row['diagnostic_decision']['snapshot']);data.pop('units')
        for key in ['tangent','right','centre','points','edges_a','edges_b']:data[key]=np.asarray(data[key],dtype=float)
        data['tracks']=[cc.TrackView(t['id'],np.asarray(t['position']),np.asarray(t['velocity']),t['heading']) for t in data['tracks']]
        snap=cc.Snapshot(**data)
        saved=row['diagnostic_decision']['actuators_before_decision'];act=cc.Actuators()
        act.servo,act.buffer,act.executed=saved['servo'],saved['buffer'],saved['executed']
        return snap,act

    # The independent prefix generator consumes normal draws only, with logged
    # round/sample counts. Means/stds affect transformations, not RNG consumption.
    # Advance exactly those recorded draw shapes; no parent-controller replay.
    rng=np.random.default_rng(0)
    earlier=[]
    for row in rows[:10]:
        f=row['filter']
        if not f.get('v11_policy_prefix_search_checked'):continue
        d=f['policy_prefix_search']; remaining=d['sampled_plans']
        blocks=int(np.ceil((d['horizon_decisions']-1)/prefix.TAIL_BLOCK_DECISIONS))
        batches=[]
        for iteration in range(d['rounds']):
            count=int(np.ceil(remaining/(d['rounds']-iteration)))
            rng.normal(size=(count,blocks,2));batches.append(count);remaining-=count
        earlier.append({'step':row['step'],'sampled_plans':d['sampled_plans'],'round_batches':batches,'blocks':blocks})
    row=rows[10];snap,act=snapshot(row)
    traffic=any(np.hypot(*(t.position-snap.position))<v2.ENGAGE_RANGE_M for t in snap.tracks)
    callback=lambda s,r:v2.SafetyFilterV2._evaluate(None,s,r)
    backup=np.asarray(row['plan'],dtype=float)
    # Stored CEM plan is float64; issued parent command is rounded to float32.
    assert np.array_equal(backup[0].astype(np.float32),np.asarray(row['filter']['v11_parent_action'],dtype=np.float32))
    rng_before_step11=copy.deepcopy(rng.bit_generator.state)
    hypothetical=prefix.search(snap,act,np.asarray(row['policy_action']),backup,v3.rollout_seq,callback,rng,traffic=traffic)
    # Independently verify the draw-count advancement by reproducing recorded
    # step10. The paired V14-prefix run rejected that prefix and consequently
    # retained the exact pre-prefix parent backup needed for this replay.
    parent_trace=ROOT/'results/safety_dev/v10_iterations/persistent_prefix32/traces/014_v14.jsonl'
    saved_parent=read(parent_trace)[9]
    assert saved_parent['pre_state']==rows[9]['pre_state'] and saved_parent['policy_action']==rows[9]['policy_action']
    assert saved_parent['filter']['why']=='turn' and not saved_parent['filter']['policy_prefix_search']['accepted']
    check_rng=np.random.default_rng(0)
    for history in earlier[:-1]:
        for count in history['round_batches']:check_rng.normal(size=(count,history['blocks'],2))
    s10,a10=snapshot(rows[9])
    check=prefix.search(s10,a10,np.asarray(rows[9]['policy_action']),np.asarray(saved_parent['plan'],dtype=float),
                        v3.rollout_seq,callback,check_rng,
                        traffic=any(np.hypot(*(t.position-s10.position))<v2.ENGAGE_RANGE_M for t in s10.tracks),
                        minimum_clearance=rows[9]['filter']['v11_prefix_minimum_clearance'])
    logged=rows[9]['filter']['policy_prefix_search']
    assert abs(check.clearance-logged['maximum_clearance'])<1e-12
    assert check.diagnostics['accepted_plans']==logged['accepted_plans']
    assert check.diagnostics['hard_passing_plans']==logged['hard_passing_plans']
    np.testing.assert_allclose(check.plan,np.asarray(rows[9]['plan'],dtype=float),rtol=0,atol=1e-12)
    assert check_rng.bit_generator.state==rng_before_step11
    seq=np.asarray([[r['rudder_command'],(r['signed_rpm_command']-cfg.CRUISE_RPM)/cfg.RPM_DELTA] for r in off[10:26]])[None]
    assert seq.shape==(1,16,2)
    actual_first,actual_clear=callback(snap,v3.rollout_seq(snap,act,seq))
    report={'scope':'Completed saved BAS-NU V15 trace. One offline call to the existing bounded prefix search at the skipped decision; no policy/environment/controller decision or episode. No new rule or threshold fitted.',
      'case':result['case'],'outcome':result['outcome'],'steps':len(rows),'changed_steps':[r['step'] for r in rows if r['changed']],
      'source_hashes_match_evaluated_manifest':source_hashes,'paired_seed_scene_checkpoint_config':True,
      'inputs':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [trace,offpath,parent_trace]},
      'first_actual_command_difference':next(i+1 for i,(a,b) in enumerate(zip(rows,off)) if [a['rudder_command'],a['signed_rpm_command']]!=[b['rudder_command'],b['signed_rpm_command']]),
      'step10_same_off_command':all(rows[9][k]==off[9][k] for k in ['rudder_command','signed_rpm_command']),
      'step11_same_recorded_prestate_policy':all(row[k]==off[10][k] for k in ['pre_state','policy_action']),
      'step10_filter':rows[9]['filter'],'step11_filter':row['filter'],
      'step11_represented_target_count':len(snap.tracks),'step11_represented_target_ids':[t.id for t in snap.tracks],
      'parent_clearance_above_trigger_m':row['filter']['v10_proposed_clearance']-v2.TRIGGER_MARGIN_M,
      'recorded_rng_history':earlier,'offline_skipped_prefix_search':hypothetical.diagnostics,
      'rng_validation':{'step10_plan_and_diagnostics_reproduced':True,'step11_rng_state_matches_after_actual_step10_search':True},
      'offline_skipped_prefix_plan':hypothetical.plan,
      'known_successful_future16_SAC':{'first_violation_s':actual_first[0],'minimum_clearance_m':actual_clear[0], 'hard_safe':bool(np.isposinf(actual_first[0]))},
      'limits':['Counterfactual search is a saved-state feasibility test, not an executed decision or episode rescue.',
                'The random state is reconstructed from recorded prefix draw counts with seed0; parent search RNG is independent.',
                'A passing backup cannot establish later estimates, recursive feasibility, or successful closed-loop policy behavior.',
                'This one case does not justify an outcome-separating rule or changing the existing adequate-parent gate.']}
    (OUT/'audit.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(clean({k:report[k] for k in ['first_actual_command_difference','parent_clearance_above_trigger_m',
        'step11_represented_target_count','recorded_rng_history','offline_skipped_prefix_search','known_successful_future16_SAC']}),indent=2))


if __name__=='__main__':main()
