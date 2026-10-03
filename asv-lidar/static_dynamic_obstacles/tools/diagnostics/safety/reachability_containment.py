"""Saved-trace conditional containment diagnostic; never execute an episode.

The recorded future controls are NONCAUSAL diagnostic inputs on the same saved
branch. This is neither an executable feedback policy nor a safety certificate.
The parameter/state boxes are declared hypotheses, not Gaussian full support,
confidence regions, fitted error bounds, or guarantees of continuous motion.
See Kochdumper et al., https://arxiv.org/abs/2210.10691 for the substantially
stronger reachable-set shield architecture; this script does not implement it.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np
import mpmath

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from safety_reachability import (ConditionalPlantTube, EnclosureFailure,
                                 IntervalBounds, PARAMETER_NAMES)

OWN_TRACE = ROOT / "results/safety_dev/v10_iterations/conditional_prefix32/traces/014_v15.jsonl"
TARGET_TRACE = ROOT / "results/safety_dev/v10_iterations/motion_axis_probe5/traces/001_v16.jsonl"
OUTPUT_ROOT = ROOT / "results/safety_dev/reachability_stage1"
STEP_S = .5
SOURCE_NAMES = ("src/constants.py", "src/constant_temp.py", "src/ship.py",
                "bluefin/dynamics.py", "bluefin/ship_model_v3.py", "src/targets.py")
EXPECTED = {"UPDATE_RATE": .5, "PHYSICS_DT": .1, "FIXED_RPM": False,
            "RUDDER_COMMAND_LIMIT": False, "VESSEL_RANDOMISATION_SCALE": 1.,
            "POSE_STALE_PROB": 0., "BOUNDARY_POSE_NOISE_XY": .03,
            "BOUNDARY_POSE_NOISE_HEADING_DEG": .2,
            "BOUNDARY_POSE_NOISE_WALK": 0., "EGO_SPEED_NOISE": .05,
            "EGO_YAW_RATE_NOISE_DPS": 1.}


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("require finite numeric inputs")
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial_bounds(bounds):
    return {key: {"lo": value.lo, "hi": value.hi} for key, value in bounds.items()}


def conditional_parameter_bounds(bootstrap, order, *, scale=1.,
                                 jitter_fraction=.05, sigma_multiple=3.):
    """Axis hull of scaled bootstrap convex blends plus declared finite jitter.

    Independent axes discard correlations conservatively. The three-sigma jitter
    truncation is a hypothesis; actual Gaussian jitter has unbounded support.
    """
    matrix = np.asarray(bootstrap, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != len(order)
            or len(set(order)) != len(order) or not np.isfinite(matrix).all()):
        raise ValueError("invalid bootstrap matrix/order")
    scale, jitter_fraction, sigma_multiple = map(finite, (scale, jitter_fraction, sigma_multiple))
    if min(scale, jitter_fraction, sigma_multiple) < 0:
        raise ValueError("bound scales must be nonnegative")
    mean = matrix.mean(axis=0)
    jitter = sigma_multiple * jitter_fraction * scale * matrix.std(axis=0)
    lower = mean + scale * (matrix.min(axis=0) - mean) - jitter
    upper = mean + scale * (matrix.max(axis=0) - mean) + jitter
    # sample_params clips by the sign of the bootstrap mean, including zero.
    lower = np.where(mean >= 0, np.maximum(0., lower), np.minimum(0., lower))
    upper = np.where(mean >= 0, np.maximum(0., upper), np.minimum(0., upper))
    return {name: IntervalBounds(lo, hi) for name, lo, hi in zip(order, lower, upper)}


def initial_state_bounds(causal_frame, effective_constants, *, sigma_multiple=3.):
    """Use current fresh onboard measurements only; no truth or future inputs."""
    if not causal_frame["fresh"]:
        raise ValueError("stale initial measurements are unsupported")
    if finite(effective_constants["BOUNDARY_POSE_NOISE_WALK"]) != 0:
        raise ValueError("nonzero pose random walk is unsupported")
    multiplier = finite(sigma_multiple)
    if multiplier < 0:
        raise ValueError("sigma multiplier must be nonnegative")
    pose, ego = causal_frame["pose"], causal_frame["ego"]
    if len(pose) != 3 or len(ego) != 3:
        raise ValueError("require three pose and body measurements")
    x, y, heading = map(finite, pose)
    u, v, yaw = map(finite, ego)
    noises = [finite(effective_constants[k]) for k in
              ("BOUNDARY_POSE_NOISE_XY", "BOUNDARY_POSE_NOISE_HEADING_DEG",
               "EGO_SPEED_NOISE", "EGO_YAW_RATE_NOISE_DPS")]
    if min(noises) < 0:
        raise ValueError("noise scales must be nonnegative")
    xy, hdg, speed, rate = [multiplier * value for value in noises]
    def interval(value, radius, low=-math.inf, high=math.inf):
        return IntervalBounds(max(low, value-radius), min(high, value+radius))
    # These are numerical-map endpoint clips, not additional fitted assumptions.
    return dict(x=interval(x, xy), y=interval(y, xy),
                heading=interval(math.radians(heading), math.radians(hdg)),
                u=interval(u, speed, 0., 5.), v=interval(v, speed, -3., 3.),
                r=interval(math.radians(yaw), math.radians(rate),
                           -math.radians(160.), math.radians(160.)))


def delay_step_branches(bound, step_s=.05):
    step_s = finite(step_s)
    if step_s <= 0 or bound.lo < 0:
        raise ValueError("require nonnegative delay and positive time step")
    # Outward division handles binary rounding at half-step ties; ties may admit
    # an extra branch, never silently prune an admissible banker's-round result.
    low = math.ceil(np.nextafter(bound.lo / step_s - .5, -np.inf))
    high = math.floor(np.nextafter(bound.hi / step_s + .5, np.inf))
    return list(range(max(0, low), high+1))


def strip_trace_rows(rows):
    """Explicit allowlists separate causal records from later truth scoring."""
    causal, scoring = [], []
    for row in rows:
        before = row["diagnostic_before"]
        onboard = before["onboard"]
        rudder = finite(row["rudder_command"])
        if not -1. <= rudder <= 1.:
            raise ValueError("saved rudder_command must be normalized to [-1,1]")
        causal.append(dict(step=int(row["step"]), elapsed_time=finite(before["elapsed_time"]),
                           pose=list(map(finite, onboard["pose_hold_xy_heading_deg"])),
                           ego=list(map(finite, onboard["ego_hold_u_v_yaw_deg_s"])),
                           fresh=not bool(onboard["pose_stale"]),
                           command=(100 * rudder,
                                    finite(row["signed_rpm_command"])),
                           tracks=copy.deepcopy(row.get("diagnostic_decision", {}).get("snapshot", {}).get("tracks", []))))
        truth = before["truth_scoring_only"]
        own = truth["own"]
        scoring.append(dict(step=int(row["step"]), own=dict(
            x=finite(own["asv_x"]), y=finite(own["asv_y"]),
            heading=math.radians(finite(own["asv_h"])),
            u=finite(own["u_body"]), v=finite(own["v_body"]),
            r=math.radians(finite(own["asv_w"]))),
            targets=copy.deepcopy(truth["targets"])))
    return causal, scoring


def score_interval(value, bound, *, periodic=False):
    value = finite(value)
    if periodic:
        if bound.hi-bound.lo >= 2*math.pi:
            return True
        value += 2*math.pi * round(((bound.lo+bound.hi)/2-value)/(2*math.pi))
    return bool(bound.lo <= value <= bound.hi)


def run_containment(causal, decision, horizon_s, parameters, constants, *,
                    max_seconds=60., max_position_width_m=10.,
                    max_heading_width_rad=2*math.pi, max_branches=32):
    """Propagate all delay branches without ever receiving scoring truth.

    A missing/failed branch makes that and every later endpoint unknown. Branches
    are advanced in lockstep, so retained earlier endpoints cover the full bank.
    """
    horizon_s = finite(horizon_s)
    limits = list(map(finite, (max_seconds, max_position_width_m, max_heading_width_rad)))
    if (decision < 1 or decision > len(causal) or horizon_s <= 0 or horizon_s > 8
            or min(limits) <= 0 or max_branches < 1
            or not math.isclose(horizon_s / STEP_S, round(horizon_s / STEP_S))):
        raise ValueError("invalid decision, horizon or resource guard")
    start = time.perf_counter()
    index = decision-1
    state = initial_state_bounds(causal[index], constants)
    branches = delay_step_branches(parameters["rud_delay"])
    history = [tuple(frame["command"]) for frame in causal[:index]]
    steps = min(int(round(horizon_s/STEP_S)), len(causal)-decision)
    result = dict(decision=decision, requested_horizon_s=horizon_s,
                  available_prestate_horizon_s=steps*STEP_S,
                  delay_branches=branches, history_command_count=len(history),
                  initial_bounds=serial_bounds(state), endpoints=[], status="unknown",
                  stop_reason=None, guards=dict(max_seconds=max_seconds,
                  max_position_width_m=max_position_width_m,
                  max_heading_width_rad=max_heading_width_rad, max_branches=max_branches),
                  future_controls="NONCAUSAL recorded same-branch controls, fixed offline",
                  terminal_poststate_used=False)
    if len(branches) > max_branches:
        result["stop_reason"] = "delay branch resource limit; no branch subset propagated"
        return result
    tubes = []
    try:
        for branch in branches:
            if time.perf_counter()-start >= max_seconds:
                raise EnclosureFailure("wall-time resource limit during history reconstruction")
            tubes.append(ConditionalPlantTube(parameters, state, branch, command_history=history))
        result["initial_actuator_bounds_by_delay"] = {
            str(branch): serial_bounds({"servo": tube.bounds()["servo"]})
            for branch, tube in zip(branches, tubes)}
        result["endpoints"].append(dict(decision=decision, horizon_s=0.,
                                         bounds=serial_bounds(state), status="enclosed",
                                         branches_completed=len(branches)))
        for offset in range(steps):
            frames = []
            try:
                for tube in tubes:
                    if time.perf_counter()-start >= max_seconds:
                        raise EnclosureFailure("wall-time resource limit")
                    frames.append(tube.advance(*causal[index+offset]["command"], dt=STEP_S))
                hull = {key: IntervalBounds(min(frame[key].lo for frame in frames),
                                           max(frame[key].hi for frame in frames))
                        for key in frames[0]}
                endpoint = dict(decision=decision+offset+1, horizon_s=(offset+1)*STEP_S,
                                bounds=serial_bounds(hull), status="enclosed",
                                branches_completed=len(branches))
                if (max(hull[k].hi-hull[k].lo for k in ("x", "y")) > max_position_width_m
                        or hull["heading"].hi-hull["heading"].lo > max_heading_width_rad):
                    endpoint["status"] = "unknown_width_limit"
                    result["endpoints"].append(endpoint)
                    raise EnclosureFailure("union box width resource limit; no narrowing applied")
                result["endpoints"].append(endpoint)
            except EnclosureFailure:
                if len(frames) < len(branches):
                    result["endpoints"].append(dict(decision=decision+offset+1,
                        horizon_s=(offset+1)*STEP_S, status="unknown_incomplete_branch_bank",
                        bounds=None, branches_completed=len(frames)))
                raise
        result["status"] = "completed_requested_endpoints" if steps*STEP_S == horizon_s else "unknown"
        if steps*STEP_S < horizon_s:
            result["stop_reason"] = "later PREdecision truth unavailable; terminal poststate excluded"
    except EnclosureFailure as error:
        result["stop_reason"] = str(error)
    result["wall_seconds"] = time.perf_counter()-start
    result["branch_diagnostics"] = [tube.diagnostics for tube in tubes]
    return result


def score_containment(result, scoring):
    """Only this post-propagation stage sees true endpoint states."""
    result = copy.deepcopy(result)
    truth = {frame["step"]: frame["own"] for frame in scoring}
    for endpoint in result["endpoints"]:
        if endpoint["status"] != "enclosed":
            endpoint["truth_contained"] = None
            continue
        own = truth[endpoint["decision"]]
        checks = {key: score_interval(own[key], IntervalBounds(**endpoint["bounds"][key]),
                                     periodic=key == "heading") for key in own}
        endpoint.update(scoring_truth=own, component_containment=checks,
                        truth_contained=all(checks.values()))
    result["known_endpoint_misses"] = [e["decision"] for e in result["endpoints"]
                                       if e.get("truth_contained") is False]
    result["mission_safety_certificate"] = False
    return result


def load_completed_trace(path):
    path = Path(path).resolve()
    folder = path.parent.parent
    manifest_path, completion_path = folder/"manifest.json", folder/"completion.json"
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads(completion_path.read_text())
    if completion.get("source_drift") or not completion.get("checkpoint_unchanged", False):
        raise ValueError("completed run records source/checkpoint drift")
    attempt, mode = path.stem.split("_", 1)
    result_path = folder/"attempts"/f"{int(attempt):03d}_result.json"
    saved = json.loads(result_path.read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if saved["attempt"] != int(attempt) or saved["mode"] != mode or saved["steps"] != len(rows):
        raise ValueError("trace lacks matching complete committed result")
    cases = [item for item in manifest["cases"] if item["case"] == saved["case"]]
    if len(cases) != 1 or cases[0]["seed"] != saved["seed"]:
        raise ValueError("result case/seed differs from its manifest")
    constants = {key: ast.literal_eval(manifest["constants"][key]) for key in EXPECTED}
    if constants != EXPECTED or manifest.get("low_speed_start_frac") != 0:
        raise ValueError("unsupported effective sensor, actuator, timing or randomization settings")
    causal, scoring = strip_trace_rows(rows)
    for index, frame in enumerate(causal):
        if frame["step"] != index+1 or not math.isclose(frame["elapsed_time"], index*STEP_S, abs_tol=1e-9):
            raise ValueError("nonsequential saved PREdecision time/step indices")
    sources = {}
    for name in SOURCE_NAMES:
        current, recorded = sha(ROOT/name), manifest.get("source_sha256", {}).get(name)
        if recorded is not None and current != recorded:
            raise ValueError("physics source differs from saved run: " + name)
        sources[name] = dict(current_sha256=current, recorded_sha256=recorded,
                             status="matched" if recorded else "unverified_missing_manifest_hash")
    # env is not imported/run by this diagnostic. Preserve its original archived
    # provenance without assuming any current difference is behavior-preserving.
    name = "src/env.py"
    current, recorded = sha(ROOT/name), manifest.get("source_sha256", {}).get(name)
    archive_path = folder/"evaluated_sources.zip"
    archived = None
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            if name in archive.namelist():
                archived = hashlib.sha256(archive.read(name)).hexdigest()
    if recorded is not None and archived is not None and archived != recorded:
        raise ValueError("archived environment differs from saved manifest")
    sources[name] = dict(current_sha256=current, recorded_sha256=recorded,
                        archived_sha256=archived,
                        status="matched" if current == recorded else "current_differs_not_executed",
                        archived_verified=archived is not None and archived == recorded)
    provenance = dict(case=saved["case"], mode=mode, seed=saved["seed"], steps=len(rows),
                      outcome=saved["outcome"], checkpoint_sha256=manifest["checkpoint_sha256"],
                      config_sha256=manifest["config_sha256"], effective_constants=constants,
                      scenario_sha256=cases[0]["scenario_sha256"],
                      evaluated_sources_sha256=sha(archive_path) if archive_path.exists() else None,
                      physics_sources=sources, input_sha256={str(p.relative_to(ROOT)): sha(p)
                          for p in (path, manifest_path, completion_path, result_path)})
    return causal, scoring, constants, provenance


def read_bootstrap():
    """Read literals and sampling default without constructing/sampling a plant."""
    tree = ast.parse((ROOT/"bluefin/ship_model_v3.py").read_text())
    values, jitter = {}, None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = node.targets if isinstance(node, ast.Assign) else [node.target]
            for name in names:
                if isinstance(name, ast.Name) and name.id in ("PARAM_ORDER", "BOOTSTRAP"):
                    value = node.value.args[0] if name.id == "BOOTSTRAP" else node.value
                    values[name.id] = ast.literal_eval(value)
        if isinstance(node, ast.FunctionDef) and node.name == "sample_params":
            defaults = dict(zip([arg.arg for arg in node.args.args][-len(node.args.defaults):], node.args.defaults))
            jitter = ast.literal_eval(defaults["jitter"])
    if jitter != .05 or tuple(values["PARAM_ORDER"]) != PARAMETER_NAMES:
        raise ValueError("sampling jitter/order differs from declared diagnostic assumptions")
    return values["BOOTSTRAP"], values["PARAM_ORDER"], jitter


def target_turn_audit(causal, scoring, decision=20, bound_deg_s=8.):
    """Audit a declared target turn bound using SAME-branch later truth only."""
    index = decision-1
    if index < 0 or index+1 >= len(scoring):
        raise ValueError("target audit needs two PREdecision states")
    earlier = {target["index"]: target for target in scoring[index]["targets"]}
    later = {target["index"]: target for target in scoring[index+1]["targets"]}
    pairs = []
    dt = finite(causal[index+1]["elapsed_time"]-causal[index]["elapsed_time"])
    bound_deg_s = finite(bound_deg_s)
    if dt <= 0 or bound_deg_s < 0:
        raise ValueError("target scoring needs positive duration and nonnegative turn bound")
    for target_id in sorted(earlier.keys() & later.keys()):
        before, after = earlier[target_id], later[target_id]
        change = (finite(after["heading"])-finite(before["heading"])+180.) % 360.-180.
        pairs.append(dict(scoring_target_index=target_id, heading_before_deg=before["heading"],
                          heading_after_deg=after["heading"], shortest_change_deg=change,
                          interval_s=dt, mean_turn_deg_s=change/dt,
                          declared_bound_deg_s=bound_deg_s, violates_bound=abs(change)>bound_deg_s*dt))
    return dict(decision=decision, next_decision=decision+1, pairs=pairs,
                current_onboard_views=causal[index]["tracks"],
                scope="same saved filtered branch; no OFF future inference or target certificate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="New output directory under reachability_stage1")
    parser.add_argument("--trace", type=Path, default=OWN_TRACE)
    parser.add_argument("--decision", type=int, default=11)
    parser.add_argument("--horizon", type=float, default=8.)
    parser.add_argument("--max-seconds", type=float, default=60.)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in (".", ".."):
        raise ValueError("require simple new tag")
    output = OUTPUT_ROOT/args.tag
    if output.exists():
        raise FileExistsError(output)
    causal, truth, constants, provenance = load_completed_trace(args.trace)
    target_causal, target_truth, _, target_provenance = load_completed_trace(TARGET_TRACE)
    bootstrap, order, jitter = read_bootstrap()
    parameters = conditional_parameter_bounds(bootstrap, order)
    propagated = run_containment(causal, args.decision, args.horizon, parameters, constants,
                                max_seconds=args.max_seconds)
    scored = score_containment(propagated, truth)
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), episode_calls=0,
        own=scored, target_turn=target_turn_audit(target_causal, target_truth),
        conditional_hypotheses=dict(name="bootstrap_axis_hull_plus_declared_3sigma_jitter_and_sensors",
            scale=1., jitter_fraction=jitter, sigma_multiple=3., parameter_bounds=serial_bounds(parameters),
            bootstrap_count=len(bootstrap), deterministic_full_support=False, probability_claim=False,
            initial_servo_at_reset_rad=0., all_past_issued_commands_used=True),
        provenance=dict(own=provenance, target=target_provenance,
            script_sha256=sha(Path(__file__)), core_sha256=sha(ROOT/"src/safety_reachability.py"),
            runtime=dict(python=platform.python_version(), numpy=np.__version__,
                         mpmath=mpmath.__version__, native_thread_environment={
                             key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})),
        limitations=["Gaussian sensor noise and parameter jitter have unbounded support; declared finite boxes are conditional hypotheses.",
            "Recorded future controls are NONCAUSAL same-branch diagnostic inputs, not an executable feedback policy.",
            "Truth is used only after propagation for PREdecision endpoint scoring, never to construct/tune bounds.",
            "Natural interval arithmetic loses parameter/state dependence; wide/failed boxes mean unknown, never safe.",
            "Experimental mpmath.iv real-arithmetic numerical-map enclosure; binary64 roundoff, continuous ODE and swept geometry are unvalidated.",
            "No obstacle/target occupancy certificate, recursive feasibility, terminal backup or mission safety guarantee.",
            "Actual plant parameters and servo are unlogged, so a containment miss cannot distinguish an assumption violation from an implementation error."],
        reference="Kochdumper et al., https://arxiv.org/abs/2210.10691 ; architecture inspiration only.")
    output.mkdir(parents=True, exist_ok=False)
    (output/"audit.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (output/"endpoints.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["decision", "horizon_s", "status", "component", "lo", "hi", "truth", "contained"])
        writer.writeheader()
        for point in scored["endpoints"]:
            for key, bounds in (point["bounds"] or {}).items():
                writer.writerow(dict(decision=point["decision"], horizon_s=point["horizon_s"], status=point["status"], component=key,
                    **bounds, truth=point.get("scoring_truth", {}).get(key), contained=point.get("component_containment", {}).get(key)))
    report = ["# Conditional saved-trace containment diagnostic", "",
        f"Case: {provenance['case']}; decision {args.decision}; requested horizon {args.horizon:g} s.",
        f"Status: **{scored['status']}**. Stop reason: {scored['stop_reason'] or 'none'}.",
        f"Delay branches: {scored['delay_branches']}; elapsed diagnostic time: {scored.get('wall_seconds', 0):.3f} s.",
        f"Scored endpoint misses: {scored['known_endpoint_misses']}. No new episodes were run.", "",
        "| Horizon (s) | Status | All six true endpoint states contained |",
        "|---:|---|---|"]
    report += [f"| {e['horizon_s']:g} | {e['status']} | {e.get('truth_contained')} |" for e in scored["endpoints"]]
    report += ["", "The BAS-HO target-turn check uses its own recorded filtered branch only:", ""]
    report += [f"- Target {p['scoring_target_index']}: {p['shortest_change_deg']:.6f} degrees over {p['interval_s']:g} s; declared 8 degrees/s bound violated: {p['violates_bound']}." for p in payload["target_turn"]["pairs"]]
    report += ["", *[f"- {item}" for item in payload["limitations"]], "",
               "Reachable-set architecture reference: [Kochdumper et al.](https://arxiv.org/abs/2210.10691). This prototype does not inherit its guarantees."]
    (output/"REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), status=scored["status"], stop_reason=scored["stop_reason"], episode_calls=0)))


if __name__ == "__main__":
    main()
