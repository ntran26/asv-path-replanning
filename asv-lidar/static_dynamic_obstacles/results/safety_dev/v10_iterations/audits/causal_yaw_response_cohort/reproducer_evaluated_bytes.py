"""Frozen gain-only yaw fits on saved complete cases, zero episode calls.

Method fixed by the preceding BAS audit: fit g*identified_dr (no bias/bounds)
using the first ten measured nonbraking transitions, freeze at decision11.
No future/truth/outcome fitting or sample-count tuning. Repeated scenarios in
different runs remain distinct exposures. See audit_causal_yaw_response.py.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).resolve().parent)]
from audit_causal_yaw_response import clean, fit, forecast, initial, measured
from tools.diagnostics.safety.suite_status import read_shared_bytes


def metrics(predicted, observed):
    difference = np.asarray(predicted)-np.asarray(observed)
    return dict(rmse=float(np.sqrt(np.mean(difference**2))), mae=float(np.mean(np.abs(difference))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",default="conditional_prefix32,motion_axis_probe5")
    parser.add_argument("--tag",required=True)
    args=parser.parse_args()
    if Path(args.tag).name!=args.tag:
        raise ValueError("Use a simple new output tag")
    output=ROOT/"results/safety_dev/v10_iterations/audits"/args.tag
    if output.exists():
        raise FileExistsError(output)
    inputs={}
    def read(path):
        raw=read_shared_bytes(path)
        inputs[path.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
        return raw
    import constants as cfg
    run_manifests={}
    for name in args.runs.split(","):
        directory=ROOT/"results/safety_dev/v10_iterations"/name
        manifest=json.loads(read(directory/"manifest.json"))
        completion=json.loads(read(directory/"completion.json"))
        if completion["source_drift"] or completion["completed_runs"]!=manifest["planned_runs"]:
            raise ValueError("Incomplete or drifted source run: "+name)
        if not completion["checkpoint_unchanged"] or not completion["selection_unchanged"]:
            raise ValueError("Frozen input changed: "+name)
        for path in ("src/constants.py","src/constant_temp.py","src/classical/common.py","src/ship.py","bluefin/dynamics.py"):
            if hashlib.sha256(read(ROOT/path)).hexdigest()!=manifest["source_sha256"][path]:
                raise ValueError("Predictor source mismatch: "+path)
        if run_manifests and manifest["constants"]!=next(iter(run_manifests.values()))["constants"]:
            raise ValueError("Different effective constants")
        run_manifests[name]=manifest
    for key,value in next(iter(run_manifests.values()))["constants"].items():
        try: setattr(cfg,key,ast.literal_eval(value))
        except (ValueError,SyntaxError): pass
    from classical import common as cc
    records=[]
    for run,manifest in run_manifests.items():
        directory=ROOT/"results/safety_dev/v10_iterations"/run
        results=[json.loads(read(path)) for path in sorted((directory/"attempts").glob("*_result.json"))]
        if len(results)!=manifest["planned_runs"]:
            raise ValueError("Missing committed result")
        for result in results:
            path=directory/"traces"/f"{result['attempt']:03d}_{result['mode']}.jsonl"
            rows=[json.loads(line) for line in read(path).splitlines() if line]
            if len(rows)!=result["steps"]:
                raise ValueError("Incomplete trace")
            case=next(c for c in manifest["cases"] if c["case"]==result["case"])
            record=dict(run=run,case=result["case"],seed=case["seed"],scenario_sha256=case["scenario_sha256"],steps=len(rows),fit_endpoint_decision=11)
            records.append(record)
            if len(rows)<12:
                record.update(status="short_trace_before_first_validation_endpoint")
                continue
            if any("diagnostic_decision" not in r for r in rows):
                record.update(status="missing_snapshots")
                continue
            data=[measured(r) for r in rows]
            past=data[:11]
            rejected=Counter()
            for a,b in zip(past,past[1:]):
                if not a["fresh"] or not b["fresh"]: rejected["stale_endpoint"]+=1
                if a["command"][1]<0: rejected["braking_command"]+=1
                if not np.isfinite(a["raw"]).all() or not np.isfinite(b["raw"]).all(): rejected["nonfinite_measured_state"]+=1
            if rejected:
                record.update(status="first_ten_transitions_not_all_eligible",training_exclusions=dict(rejected))
                continue
            pairs=list(zip(past,past[1:]))
            answer=fit(pairs,"gain",cc,cfg)
            gain=answer["gain"]
            sensitivity=float(answer["jacobian_singular_values"][0])
            record.update(fit=answer,training_transition_count=10,
                gain_noise_scale_sensitivity=(np.radians(cfg.EGO_YAW_RATE_NOISE_DPS)*np.sqrt(2.)/sensitivity if sensitivity>0 else float("inf")),
                measured_past_yaw_span_deg_s=float(np.degrees(np.ptp([x["raw"][2] for x in past]))),
                issued_past_rudder_span=float(np.ptp([x["command"][0] for x in past[:-1]])))
            if not answer["success"] or not np.isfinite(gain):
                record.update(status="invalid_fit")
                continue
            if gain<=0:
                record.update(status="nonpositive_gain")
                continue
            if sensitivity==0 or not np.isfinite(sensitivity):
                record.update(status="no_numerical_gain_excitation")
                continue
            record["status"]="fitted"
            # No outcomes or truth enter any following validation metric either.
            available=list(zip(data[10:26],data[11:27]))
            held=[(a,b) for a,b in available if a["fresh"] and b["fresh"] and a["command"][1]>=0]
            record["held_forward_one_step_pairs"]=len(held)
            record["held_forward_one_step_skipped_pairs"]=len(available)-len(held)
            record["one_step"]={}
            for label,g in (("identified",1.),("gain",gain)):
                predicted=[float(forecast(initial(a),a["actuators"],[a["command"]],g,0.,cc,cfg)[-1,2]) for a,b in held]
                observed=[float(b["raw"][2]) for a,b in held]
                record["one_step"][label]=metrics(np.degrees(predicted),np.degrees(observed)) if held else None
            # A single continuous forecast from decision11; no later resets.
            length=0
            for a,b in available:
                if a["command"][1]<0: break
                length+=1
            record["continuous_horizon_s"]=length*cfg.UPDATE_RATE
            record["continuous_stop_reason"]=("first_braking_command" if length<len(available) else ("eight_seconds" if length==16 else "trace_end"))
            record["continuous"]={}
            if length:
                for kind in ("raw","filtered"):
                    record["continuous"][kind]={}
                    observed=data[11:11+length]
                    for label,g in (("identified",1.),("gain",gain)):
                        states=forecast(initial(data[10],kind),data[10]["actuators"],[x["command"] for x in data[10:10+length]],g,0.,cc,cfg)[cc.SUBSTEPS-1::cc.SUBSTEPS]
                        errors=[dict(horizon_s=(i+1)*cfg.UPDATE_RATE,fresh=b["fresh"],
                            yaw_error_deg_s=float(np.degrees(s[2]-b["raw"][2])),
                            heading_error_deg=float(np.degrees(cc.wrap_pi(s[3]-b["heading"]))),
                            position_error_m=float(np.linalg.norm(s[5:7]-b["position"]))) for i,(s,b) in enumerate(zip(states,observed))]
                        fresh=[e for e in errors if e["fresh"]]
                        record["continuous"][kind][label]=dict(errors=errors,scored_endpoints=len(fresh),
                            heading_rmse_deg=float(np.sqrt(np.mean([e["heading_error_deg"]**2 for e in fresh]))) if fresh else None,
                            yaw_rmse_deg_s=float(np.sqrt(np.mean([e["yaw_error_deg_s"]**2 for e in fresh]))) if fresh else None,
                            position_rmse_m=float(np.sqrt(np.mean([e["position_error_m"]**2 for e in fresh]))) if fresh else None)
            print(run,result['case'],record['status'],'gain',round(gain,4),'held',len(held),'horizon',length*cfg.UPDATE_RATE,flush=True)
    summary={}
    for run in run_manifests:
        selected=[r for r in records if r['run']==run]
        valid=[r for r in selected if r['status']=='fitted']
        item=dict(records=len(selected),status_counts=dict(Counter(r['status'] for r in selected)),fits=len(valid))
        if valid:
            item['gain_range']=[min(r['fit']['gain'] for r in valid),max(r['fit']['gain'] for r in valid)]
            item['gain_median']=float(np.median([r['fit']['gain'] for r in valid]))
        def comparisons(eligible,getter):
            rows=[dict(case=r['case'],identified=getter(r,'identified'),gain=getter(r,'gain')) for r in eligible]
            return dict(n=len(rows),improved=sum(x['gain']<x['identified'] for x in rows),degraded=sum(x['gain']>x['identified'] for x in rows),
                mean_identified=float(np.mean([x['identified'] for x in rows])) if rows else None,
                mean_gain=float(np.mean([x['gain'] for x in rows])) if rows else None,cases=rows)
        item['one_step_rmse_deg_s']=comparisons([r for r in valid if r['held_forward_one_step_pairs']],lambda r,k:r['one_step'][k]['rmse'])
        item['continuous']={}
        for kind in ('raw','filtered'):
            eligible=[r for r in valid if kind in r['continuous'] and r['continuous'][kind]['identified']['scored_endpoints']]
            item['continuous'][kind]={m:comparisons(eligible,lambda r,k:r['continuous'][kind][k][m]) for m in ('heading_rmse_deg','yaw_rmse_deg_s','position_rmse_m')}
        summary[run]=item
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),method=__doc__,new_episode_calls=0,
        training_rule='Exactly transitions1-10, all fresh measured endpoints and nonbraking; frozen at decision11. No truth or future fitting.',
        sensitivity_note='One-parameter condition number is always1 when nonzero; report absolute sensitivity/noise scale instead. Noise scale ignores other input noise and model error and is not a confidence interval.',
        scope='Selected saved episodes; repeated case IDs across controllers are not independent cases. No gain bounds, clipping, outcome tuning or runtime recommendation.',
        records=records,summary=summary,provenance=dict(inputs=inputs,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(Path(__file__).with_name('audit_causal_yaw_response.py').read_bytes()).hexdigest()))
    output.mkdir(exist_ok=False)
    (output/'audit.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    (output/'reproducer_evaluated_bytes.py').write_bytes(Path(__file__).read_bytes())
    print('OUTPUT',output)
    print(json.dumps(clean({k:{x:y for x,y in v.items() if x not in ('one_step_rmse_deg_s','continuous')} for k,v in summary.items()}),indent=2))


if __name__=='__main__':
    main()
