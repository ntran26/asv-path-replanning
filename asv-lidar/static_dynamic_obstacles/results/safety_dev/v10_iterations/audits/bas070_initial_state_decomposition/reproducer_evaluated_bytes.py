"""Saved-only decomposition of policy-future prediction error; zero episodes.

Truth initial state is used only in explicitly labelled sensitivity branches.
The baseline remains the captured onboard prediction. Plant actuator state is
not in these traces and is never inferred. Finer integration retains the same
captured actuator-history signal and identified dynamics, not hidden dynamics.
"""
import ast
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tools.diagnostics.safety.audit_v11_prediction import audit_arguments, clean
from tools.diagnostics.safety.suite_status import read_shared_bytes


def main():
    args = audit_arguments(__doc__)
    manifest_bytes = read_shared_bytes(args.run_dir / "manifest.json")
    manifest = json.loads(manifest_bytes)
    trace_bytes = read_shared_bytes(args.run_dir / "traces" / args.trace)
    rows = [json.loads(x) for x in trace_bytes.splitlines() if x]
    result = json.loads(read_shared_bytes(args.run_dir / "attempts" / f"{args.attempt:03d}_result.json"))
    if len(rows) != result["steps"]:
        raise ValueError("Require a completed committed episode")
    base = ROOT / "results/safety_dev/v9_paired_pilot"
    old_manifest = json.loads(read_shared_bytes(base / "manifest.json"))
    case = next(c for c in manifest["cases"] if c["case"] == result["case"])
    old_case = next(c for c in old_manifest["cases"] if c["case"] == result["case"])
    for key in ("scenario_sha256", "seed"):
        if case[key] != old_case[key]:
            raise ValueError("Case identity mismatch")
    for key in ("checkpoint_sha256", "config_sha256"):
        if manifest[key] != old_manifest[key]:
            raise ValueError("Model/config mismatch")
    ep = next(r for r in csv.DictReader(io.StringIO(read_shared_bytes(base / "episodes.csv").decode()))
              if r["case"] == result["case"] and r["mode"] == "off")
    if ep["outcome"] != "goal":
        raise ValueError("Reference must be a recorded SAC success")
    reference_bytes = read_shared_bytes(base / "traces" / f"{int(ep['attempt']):03d}_off.jsonl")
    off = [json.loads(x) for x in reference_bytes.splitlines() if x]
    command_keys = ("rudder_command", "signed_rpm_command")
    start = next(i for i,(a,b) in enumerate(zip(rows,off)) if not np.allclose(
        [a[k] for k in command_keys], [b[k] for k in command_keys], atol=1e-7, rtol=0))
    for a,b in zip(rows[:start+1],off[:start+1]):
        np.testing.assert_allclose(a["pre_state"], b["pre_state"], atol=1e-9, rtol=0)
    row = rows[start]
    source_hashes = {}
    for name in ("src/constants.py", "src/constant_temp.py", "src/classical/common.py",
                 "src/safety_v2.py", "src/safety_v3.py", "src/ship.py",
                 "src/reference_controller.py", "bluefin/dynamics.py"):
        digest = hashlib.sha256(read_shared_bytes(ROOT / name)).hexdigest()
        if manifest["source_sha256"][name] != digest:
            raise ValueError("Evaluated source mismatch: "+name)
        source_hashes[name] = digest
    import constants as cfg
    for key,text in manifest["constants"].items():
        try:
            setattr(cfg,key,ast.literal_eval(text))
        except (ValueError,SyntaxError):
            pass
    from classical import common as cc
    import safety_v2 as v2
    import safety_v3 as v3
    values = copy.deepcopy(row["diagnostic_decision"]["snapshot"])
    values.pop("units",None)
    for key in ("tangent","right","centre","points","edges_a","edges_b"):
        values[key]=np.asarray(values[key],dtype=float)
    values["tracks"]=[cc.TrackView(t["id"],np.array(t["position"]),np.array(t["velocity"]),t["heading"])
                      for t in values["tracks"]]
    snap=cc.Snapshot(**values)
    saved=row["diagnostic_decision"]["actuators_before_decision"]
    act=cc.Actuators()
    act.servo,act.executed,act.buffer=saved["servo"],saved["executed"],copy.deepcopy(saved["buffer"])
    truth=row["diagnostic_before"]["truth_scoring_only"]["own"]
    raw=np.asarray(row["diagnostic_decision"]["raw_ego_u_v_yaw_rad_s"])
    n=int(round(v2.HORIZON_S/cfg.UPDATE_RATE))
    commands=off[start:start+n]
    if len(commands)!=n or any(r["signed_rpm_command"]<0 for r in commands):
        raise ValueError("Requires a full nonbraking successful recorded command horizon")
    sequence=np.array([[r["rudder_command"],(r["signed_rpm_command"]-cfg.CRUISE_RPM)/cfg.RPM_DELTA]
                       for r in commands])[None]
    variants={"onboard":copy.copy(snap),"raw_measured_uvr":copy.copy(snap),
              "truth_r_only_scoring":copy.copy(snap),"truth_uv_only_scoring":copy.copy(snap),
              "truth_uvr_scoring":copy.copy(snap),"truth_pose_uvr_scoring":copy.copy(snap)}
    variants["raw_measured_uvr"].u,variants["raw_measured_uvr"].v,variants["raw_measured_uvr"].r=raw
    variants["truth_r_only_scoring"].r=np.radians(truth["asv_w"])
    variants["truth_uv_only_scoring"].u=truth["u_body"]
    variants["truth_uv_only_scoring"].v=truth["v_body"]
    for name in ("truth_uvr_scoring","truth_pose_uvr_scoring"):
        variants[name].u,variants[name].v,variants[name].r=truth["u_body"],truth["v_body"],np.radians(truth["asv_w"])
    variants["truth_pose_uvr_scoring"].x=truth["asv_x"]
    variants["truth_pose_uvr_scoring"].y=truth["asv_y"]
    variants["truth_pose_uvr_scoring"].heading=np.radians(truth["asv_h"])

    def integrate(initial,dt):
        state=np.array([max(0.,initial.u),initial.v,initial.r,initial.heading,act.servo,initial.x,initial.y])[:,None]
        params={key:np.array([value]) for key,value in cc.IDENTIFIED.items()}
        steps=int(round(cfg.UPDATE_RATE/dt))
        delay_s=cc.DELAY_STEPS*cc.PRED_DT
        history=act.buffer
        positions=[];headings=[];speeds=[]
        for decision,r in enumerate(commands):
            for sub in range(steps):
                now=decision*cfg.UPDATE_RATE+sub*dt
                if history is not None and now<delay_s-1e-10:
                    delayed=history[min(len(history)-1,int((now+1e-10)/cc.PRED_DT))]
                else:
                    earlier=max(0,int((now-delay_s+1e-10)/cfg.UPDATE_RATE)) if history is not None else decision
                    delayed=-cc.MAX_RUDDER_RAD*commands[earlier]["rudder_command"]
                state=cc.dyn.rk4_step(state,np.array([r["signed_rpm_command"]]),np.array([delayed]),params,dt)
                if (sub+1)%int(round(cc.PRED_DT/dt))==0:
                    positions.append(state[5:7,0].copy());headings.append(float(state[3,0]));speeds.append(float(np.hypot(state[0,0],state[1,0])))
        return cc.Rollout(np.array(positions)[:,None],np.array(headings)[:,None],np.array(speeds)[:,None],
                          cc.PRED_DT*np.arange(1,len(positions)+1))

    forecasts={}
    for name,initial in variants.items():
        forecasts[name]=v3.rollout_seq(initial,act,sequence)
    repeat=integrate(snap,cc.PRED_DT)
    np.testing.assert_allclose(repeat.positions,forecasts["onboard"].positions,atol=1e-12,rtol=0)
    np.testing.assert_allclose(repeat.headings,forecasts["onboard"].headings,atol=1e-12,rtol=0)
    for dt in (.0625,.025):
        forecasts[f"onboard_finer_dt_{dt}"]=integrate(snap,dt)
        forecasts[f"truth_uvr_finer_dt_{dt}_scoring"]=integrate(variants["truth_uvr_scoring"],dt)
    scored={}
    for name,forecast in forecasts.items():
        first,clear=v2.SafetyFilterV2._evaluate(None,snap,forecast)
        errors=[]
        for j,r in enumerate(commands):
            k=(j+1)*cc.SUBSTEPS-1
            errors.append(dict(horizon_s=(j+1)*cfg.UPDATE_RATE,
                position_error_m=float(np.linalg.norm(forecast.positions[k,0]-r["post_state"][:2])),
                heading_error_deg=float(np.degrees(cc.wrap_pi(forecast.headings[k,0]-np.radians(r["post_state"][2]))))))
        scored[name]=dict(first_violation_s=float(first[0]),clearance_m=float(clear[0]),errors=errors)
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope=__doc__,case=case,first_divergence_step=start+1,
        initial_onboard=dict(u=snap.u,v=snap.v,r_deg_s=float(np.degrees(snap.r)),heading_deg=float(np.degrees(snap.heading))),
        initial_raw_measured=dict(u=float(raw[0]),v=float(raw[1]),r_deg_s=float(np.degrees(raw[2]))),
        initial_truth_scoring_only=truth,captured_predictor_actuators=saved,
        plant_actuator_truth_available=False,
        actuator_limit="True plant rudder servo, delay history and dynamics parameters are not recorded; no comparison or reconstruction is claimed.",
        integration_limit="Same identified model, same captured piecewise actuator history, same delay duration; finer dt isolates numerical discretization only.",
        predictions=scored,source_sha256=source_hashes,
        trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),reference_trace_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.output_dir:
        with (args.output_dir/'audit.json').open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(clean(output),indent=2,allow_nan=False)+'\n')
        with (args.output_dir/'reproducer_evaluated_bytes.py').open('xb') as stream:
            stream.write(Path(__file__).read_bytes())
        print('OUTPUT',args.output_dir/'audit.json')
    print('INITIAL',json.dumps(clean({k:output[k] for k in ['initial_onboard','initial_raw_measured','initial_truth_scoring_only']})))
    for name,score in scored.items():
        print(name,'clear',score['clearance_m'],'errors',json.dumps([x for x in score['errors'] if x['horizon_s'] in (.5,1.,2.5,4.,8.)]))


if __name__=='__main__':
    main()
