"""Causal saved-data yaw-response audit; never reset/step an environment.

Fit only measured body states and commands available BEFORE --decision. Frozen
parameters are then scored forward; future commands are conditional validation
inputs, not online information. This is prediction-error identification in the
sense of Ljung (2002), DOI 10.1007/BF01211648, with an intentionally small
effective yaw-acceleration correction. It is not physical parameter recovery.
Shared noise in successive rate measurements and unobserved plant actuator
states limit interpretation. Leave-one-transition ranges are sensitivity, not
confidence intervals. No production/controller model is changed.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from tools.diagnostics.safety.suite_status import read_shared_bytes


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


def body_step(state, rpm, delayed, params, dt, gain, bias, dyn):
    """The identified RK4/servo scheme, changing only dr = gain*dr + bias."""
    next_delta = dyn.advance_rudder(state[4], delayed, params, dt)
    middle = state.copy()
    middle[4] = .5 * (state[4] + next_delta)
    def derivative(x):
        result = dyn.derivatives(x, rpm, delayed, params)
        result[2] = gain * result[2] + bias
        return result
    k1 = derivative(middle)
    k2 = derivative(middle + .5 * dt * k1)
    k3 = derivative(middle + .5 * dt * k2)
    k4 = derivative(middle + dt * k3)
    result = middle + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    result[0] = np.clip(result[0], 0., dyn.MAX_SURGE_SPEED)
    result[1] = np.clip(result[1], -dyn.MAX_SWAY_SPEED, dyn.MAX_SWAY_SPEED)
    result[2] = np.clip(result[2], -dyn.MAX_YAW_RATE_RAD, dyn.MAX_YAW_RATE_RAD)
    result[4] = next_delta
    return result


def forecast(initial, actuators, commands, gain, bias, cc, cfg):
    """Carry copied estimated servo/history continuously; nonbraking scope."""
    state = np.asarray(initial, float).reshape(7, 1).copy()
    params = {k: np.array([v]) for k, v in cc.IDENTIFIED.items()}
    pending = None if actuators["buffer"] is None else [np.array([v]) for v in actuators["buffer"]]
    output = []
    for command in commands:
        if command[1] < 0:
            raise ValueError("This yaw-only audit excludes braking commands")
        rpm = np.array([command[1]])
        delta = np.array([-cc.MAX_RUDDER_RAD * command[0]])
        if pending is None:
            pending = [delta.copy() for _ in range(cc.DELAY_STEPS)]
        for _ in range(cc.SUBSTEPS):
            pending.append(delta)
            delayed = pending.pop(0)
            previous = state
            state = body_step(state, rpm, delayed, params, cc.PRED_DT, gain, bias, cc.dyn)
            if gain == 1. and bias == 0.:
                np.testing.assert_allclose(state, cc.dyn.rk4_step(previous, rpm, delayed, params, cc.PRED_DT), atol=1e-13, rtol=0)
            output.append(state[:, 0].copy())
    return np.asarray(output)


def measured(row):
    decision = row["diagnostic_decision"]
    snap = decision["snapshot"]
    act = decision["actuators_before_decision"]
    raw = decision["raw_ego_u_v_yaw_rad_s"]
    return dict(step=row["step"], fresh=not row["diagnostic_before"]["onboard"]["pose_stale"],
        raw=np.asarray(raw, float), filtered=np.array([snap["u"], snap["v"], snap["r"]]),
        heading=snap["heading"], position=np.array([snap["x"], snap["y"]]),
        actuators=copy.deepcopy(act), command=np.array([row["rudder_command"], row["signed_rpm_command"]]))


def initial(item, kind="raw"):
    state = np.r_[item[kind], item["heading"], item["actuators"]["servo"], item["position"]]
    state[0] = max(0., state[0])  # Existing rollout initialization, including noisy raw u.
    return state


def fit(pairs, mode, cc, cfg):
    def decode(x):
        return (float(x[0]), 0.) if mode == "gain" else ((1., float(x[0])) if mode == "bias" else tuple(map(float, x)))
    def residual(x):
        gain, bias = decode(x)
        return np.array([forecast(initial(a), a["actuators"], [a["command"]], gain, bias, cc, cfg)[-1, 2]-b["raw"][2] for a,b in pairs])
    start = np.array([1., 0.]) if mode == "gain_bias" else np.array([1. if mode == "gain" else 0.])
    answer = least_squares(residual, start, max_nfev=80, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    gain, bias = decode(answer.x)
    singular = np.linalg.svd(answer.jac, compute_uv=False)
    return dict(gain=gain, bias_rad_s2=bias, success=bool(answer.success), nfev=answer.nfev,
        training_rmse_deg_s=float(np.degrees(np.sqrt(np.mean(answer.fun**2)))),
        jacobian_singular_values=singular, jacobian_condition=float(singular[0]/singular[-1]),
        residual_deg_s=np.degrees(answer.fun))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="conditional_prefix32")
    parser.add_argument("--trace", default="014_v15.jsonl")
    parser.add_argument("--decision", type=int, default=11)
    parser.add_argument("--tag", help="New audit directory; omitted means read-only output")
    args = parser.parse_args()
    folder = ROOT / "results/safety_dev/v10_iterations" / args.run
    inputs = {}
    def read(path):
        raw = read_shared_bytes(path)
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        return raw
    manifest = json.loads(read(folder / "manifest.json"))
    rows = [json.loads(line) for line in read(folder / "traces" / args.trace).splitlines() if line]
    attempt = int(args.trace.split("_")[0])
    result = json.loads(read(folder / "attempts" / f"{attempt:03d}_result.json"))
    if len(rows) != result["steps"]:
        raise ValueError("Require a completed committed episode")
    import constants as cfg
    for key, text in manifest["constants"].items():
        try:
            setattr(cfg, key, ast.literal_eval(text))
        except (ValueError, SyntaxError):
            pass
    sources = {}
    for name in ("src/constants.py", "src/constant_temp.py", "src/classical/common.py", "src/ship.py", "bluefin/dynamics.py", "src/safety_v2.py", "src/safety_v3.py"):
        digest = hashlib.sha256(read(ROOT / name)).hexdigest()
        if digest != manifest["source_sha256"][name]:
            raise ValueError("Frozen predictor source differs: " + name)
        sources[name] = digest
    from classical import common as cc
    import safety_v2 as v2
    data = [measured(row) for row in rows]
    cut = args.decision-1
    if cut < 3 or cut+16 >= len(data):
        raise ValueError("Need past transitions and sixteen held-forward decisions")
    # Only these stripped onboard records reach fitting. Truth/future rows do not.
    past = copy.deepcopy(data[:cut+1])
    pairs = [(a,b) for a,b in zip(past, past[1:]) if a["fresh"] and b["fresh"] and a["command"][1] >= 0]
    fits = {name: fit(pairs, name, cc, cfg) for name in ("gain", "bias", "gain_bias")}
    for name, item in fits.items():
        omitted = [fit(pairs[:i]+pairs[i+1:], name, cc, cfg) for i in range(len(pairs))]
        item["leave_one_transition_out_gain_range"] = [min(x["gain"] for x in omitted), max(x["gain"] for x in omitted)]
        item["leave_one_transition_out_bias_deg_s2_range"] = np.degrees([min(x["bias_rad_s2"] for x in omitted), max(x["bias_rad_s2"] for x in omitted)])
    models = {"identified": (1., 0.), **{k: (v["gain"],v["bias_rad_s2"]) for k,v in fits.items()}}
    baseline_residual = [forecast(initial(a), a["actuators"], [a["command"]], 1., 0., cc, cfg)[-1,2]-b["raw"][2] for a,b in pairs]
    held_pairs = [(a,b) for a,b in zip(data[cut:cut+16], data[cut+1:cut+17]) if a["fresh"] and b["fresh"] and a["command"][1] >= 0]
    one_step = {}
    forward = {}
    for name, (gain,bias) in models.items():
        errors = [forecast(initial(a),a["actuators"],[a["command"]],gain,bias,cc,cfg)[-1,2]-b["raw"][2] for a,b in held_pairs]
        one_step[name] = dict(rmse_deg_s=float(np.degrees(np.sqrt(np.mean(np.asarray(errors)**2)))), residual_deg_s=np.degrees(errors))
        forward[name] = {}
        for kind in ("raw","filtered"):
            states = forecast(initial(data[cut],kind),data[cut]["actuators"],[x["command"] for x in data[cut:cut+16]],gain,bias,cc,cfg)
            endpoints = states[cc.SUBSTEPS-1::cc.SUBSTEPS]
            observed = data[cut+1:cut+17]
            forward[name][kind] = [dict(horizon_s=(i+1)*cfg.UPDATE_RATE,
                measured_yaw_error_deg_s=float(np.degrees(s[2]-obs["raw"][2])),
                measured_heading_error_deg=float(np.degrees(cc.wrap_pi(s[3]-obs["heading"]))),
                measured_position_error_m=float(np.linalg.norm(s[5:7]-obs["position"]))) for i,(s,obs) in enumerate(zip(endpoints,observed))]
    # Independent conditional successful-SAC continuation; never used for fitting.
    reference = ROOT / "results/safety_dev/v9_paired_pilot"
    old_manifest = json.loads(read(reference / "manifest.json"))
    case = next(c for c in manifest["cases"] if c["case"]==result["case"])
    old_case = next(c for c in old_manifest["cases"] if c["case"]==result["case"])
    assert all(case[k]==old_case[k] for k in ("seed","scenario_sha256"))
    assert all(manifest[k]==old_manifest[k] for k in ("checkpoint_sha256","config_sha256"))
    ref_results = [json.loads(read(p)) for p in (reference/"attempts").glob("*_result.json")]
    off_result = next(x for x in ref_results if x["case"]==result["case"] and x["mode"]=="off")
    assert off_result["outcome"]=="goal"
    off = [json.loads(line) for line in read(reference/"traces"/f"{off_result['attempt']:03d}_off.jsonl").splitlines() if line]
    for a,b in zip(rows[:cut],off[:cut]):
        np.testing.assert_allclose([a["rudder_command"],a["signed_rpm_command"]],[b["rudder_command"],b["signed_rpm_command"]],atol=1e-7,rtol=0)
    np.testing.assert_allclose(rows[cut]["pre_state"],off[cut]["pre_state"],atol=1e-9,rtol=0)
    values=copy.deepcopy(rows[cut]["diagnostic_decision"]["snapshot"])
    values.pop("units",None)
    for key in ("tangent","right","centre","points","edges_a","edges_b"):
        values[key]=np.asarray(values[key])
    values["tracks"]=[cc.TrackView(t["id"],np.array(t["position"]),np.array(t["velocity"]),t["heading"]) for t in values["tracks"]]
    snap=cc.Snapshot(**values)
    successful_future={}
    commands=[[r["rudder_command"],r["signed_rpm_command"]] for r in off[cut:cut+16]]
    for name,(gain,bias) in models.items():
        successful_future[name]={}
        for kind in ("raw","filtered"):
            states=forecast(initial(data[cut],kind),data[cut]["actuators"],commands,gain,bias,cc,cfg)
            rollout=cc.Rollout(states[:,5:7,None].transpose(0,2,1),states[:,3,None],np.hypot(states[:,0],states[:,1])[:,None],cc.PRED_DT*np.arange(1,len(states)+1))
            first,clear=v2.SafetyFilterV2._evaluate(None,snap,rollout)
            errors=[dict(horizon_s=(i+1)*cfg.UPDATE_RATE,truth_heading_error_deg=float(np.degrees(cc.wrap_pi(s[3]-np.radians(r["post_state"][2])))),truth_position_error_m=float(np.linalg.norm(s[5:7]-r["post_state"][:2]))) for i,(s,r) in enumerate(zip(states[cc.SUBSTEPS-1::cc.SUBSTEPS],off[cut:cut+16]))]
            successful_future[name][kind]=dict(clearance_m=float(clear[0]),first_violation_s=float(first[0]),errors=errors)
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),case=case["case"],decision=args.decision,
        fit_endpoint_decision_max=max(b["step"] for a,b in pairs),fit_transition_count=len(pairs),
        training_command_steps=[a["step"] for a,b in pairs],held_forward_command_steps=[a["step"] for a,b in held_pairs],
        identified_training_rmse_deg_s=float(np.degrees(np.sqrt(np.mean(np.asarray(baseline_residual)**2)))),
        fits=fits,held_forward_one_step=one_step,held_forward_continuous=forward,
        successful_recorded_policy_future_scoring_only=successful_future,episode_calls=0,
        limitations=["Only one short, selected case; no calibrated uncertainty or population claim.","Raw rate noise appears in both transition endpoints; successive residuals are correlated.","Captured actuator history is the predictor estimate; actual plant servo/delay are unavailable.","Frozen future commands are conditional validation inputs, not known online.","Gain/bias correct total yaw acceleration, not separately identifiable body/actuator coefficients.","Past leave-one-transition ranges are sensitivity, not confidence intervals."],
        provenance=dict(inputs=inputs,source_sha256=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        method_reference="Ljung (2002), Prediction Error Estimation Methods, DOI 10.1007/BF01211648; section 1 prediction-error setup; this low-dimensional correction is a diagnostic adaptation.")
    if args.tag:
        if Path(args.tag).name!=args.tag:
            raise ValueError("Use a simple new tag")
        destination=ROOT/"results/safety_dev/v10_iterations/audits"/args.tag
        destination.mkdir(exist_ok=False)
        (destination/"audit.json").write_text(json.dumps(clean(output),indent=2,allow_nan=False)+"\n")
        (destination/"reproducer_evaluated_bytes.py").write_bytes(Path(__file__).read_bytes())
        print('OUTPUT',destination)
    print(json.dumps(clean({k:output[k] for k in ("fit_transition_count","identified_training_rmse_deg_s","fits","held_forward_one_step")}),indent=2))
    for name in models:
        for kind in ("raw","filtered"):
            score=successful_future[name][kind]
            print(name,kind,'clear',score['clearance_m'],'OFFheldfuture',json.dumps([v for v in score['errors'] if v['horizon_s'] in (.5,2.5,4.,8.)]))


if __name__=="__main__":
    main()
