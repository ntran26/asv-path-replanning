"""Reproduce an offline oracle-plant audit of two recorded DEVELOPMENT starts.

No environment, policy inference, reset, or environment step is invoked. Only
static plant-state clones and vectorized dynamics are advanced. This is not a
new episode evaluation and does not write to an evaluation-attempt ledger.
Truth states, seed-derived simulator parameters and true obstacle polygons are
deliberate ORACLE inputs for diagnosis; they are not available to a deployable
filter and are not inputs to the measurement-only braking calibration.

The sampled first-command grid is numerical evidence, not a proof over the
continuous control space. Static-panel avoidance is not complete-episode
success. The original sensor blind-zone analysis is a separate audit.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import ship

CASES = ("DV3-BO-CV-10", "DV3-BO-VS-05")
MODEL_FILES = ("src/ship.py", "bluefin/dynamics.py", "bluefin/ship_model_v3.py")
INTEGRATION_DT, COLLISION_DT, DECISION_DT = .05, .1, .5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def state(row):
    return np.array([row["u_body"], row["v_body"], np.deg2rad(row["asv_w"]),
                     np.deg2rad(row["asv_h"]), 0., row["asv_x"], row["asv_y"]])


def plant(initial, params):
    """Initialize a static plant state without calling any reset method."""
    model = object.__new__(ship.ShipModel)
    model.sub_dt = INTEGRATION_DT
    model.reverse_efficiency = ship.REVERSE_THRUST_EFFICIENCY
    model.p = dict(params)
    model._pa = {k: np.array([v]) for k, v in params.items()}
    model._s = initial[:, None].copy()
    model._cmd_buf, model._buf_dt = deque(), None
    model.last_braking_force = model.last_astern_impulse = 0.
    return model


def static_gap(states, obstacles):
    """Minimum SAT axis gap: exact overlap sign for rectangle versus boxes.

    Positive SAT separation is not claimed to be Euclidean distance for
    arbitrary orientations. Geometry exactly uses env.hull_polygon's inflated
    rectangle and every fixed panel in the cached scene.
    """
    centres = obstacles.mean(axis=1)
    half = (obstacles.max(axis=1) - obstacles.min(axis=1)) / 2
    dx = states[5, :, None] - centres[None, :, 0]
    dy = states[6, :, None] - centres[None, :, 1]
    sn, cs = np.sin(states[3, :, None]), np.cos(states[3, :, None])
    hx, hy = half[None, :, 0], half[None, :, 1]
    hl = ship.VESSEL_LENGTH / 2 + ship.HULL_MARGIN
    hw = ship.VESSEL_WIDTH / 2 + ship.HULL_MARGIN
    return np.maximum.reduce([
        abs(dx) - (hl * abs(sn) + hw * abs(cs) + hx),
        abs(dy) - (hl * abs(cs) + hw * abs(sn) + hy),
        abs(sn * dx + cs * dy) - (hl + hx * abs(sn) + hy * abs(cs)),
        abs(cs * dx - sn * dy) - (hw + hx * abs(cs) + hy * abs(sn)),
    ]).min(axis=1)


def check_recorded_replay(initial, params, rows):
    model = plant(initial, params)
    records = []
    for row in rows:
        expected = state(row["after"])
        matched = False
        for substep in range(1, 6):
            model.update(row["issued_rpm_signed"], row["issued_rudder_percent"], COLLISION_DT)
            error = model._s[:, 0] - expected
            error[3] = (error[3] + np.pi) % (2 * np.pi) - np.pi
            error[4] = 0.  # The log does not contain true servo angle.
            if np.max(np.abs(error)) < 1e-9:
                records.append({"decision": row["decision"],
                                "matched_duration_s": substep * COLLISION_DT,
                                "position_error_m": float(np.linalg.norm(error[5:7])),
                                "maximum_state_component_error": float(np.max(abs(error)))})
                matched = True
                break
        if not matched:
            raise ValueError("Oracle reconstruction does not reproduce recorded state")
    return records


def immediate_grid(initial, params, obstacles):
    rudder, rpm = np.meshgrid(np.linspace(-100., 100., 401),
                             np.r_[-24., np.linspace(0., 12., 25)])
    rudder, rpm = rudder.ravel(), rpm.ravel()
    n = len(rudder)
    states = np.repeat(initial[:, None], n, axis=1)
    pa = {k: np.full(n, v) for k, v in params.items()}
    delta = ship.dyn.cmd_percent_to_angle(-rudder)
    # An empty plant delay line fills with the first delta command; a constant
    # first command therefore reaches the servo immediately, exactly as ship.py.
    decel = np.where(rpm < 0, ship.braking_thrust(-24., params) / ship.M11, 0.)
    first, minimum = np.full(n, np.inf), np.full(n, np.inf)
    timeline = []
    for substep in range(1, 25):
        states = ship.dyn.rk4_step(states, rpm, delta, pa, INTEGRATION_DT)
        states[0] = np.maximum(0., states[0] - decel * INTEGRATION_DT)
        if substep % 2 == 0:
            gap = static_gap(states, obstacles)
            minimum = np.minimum(minimum, gap)
            elapsed = substep * INTEGRATION_DT
            first = np.where((gap <= 0.) & np.isinf(first), elapsed, first)
            timeline.append({"time_s": elapsed,
                             "commands_without_previous_contact": int(np.isinf(first).sum()),
                             "maximum_current_sat_gap_m": float(gap.max())})
    best = int(np.argmax(minimum))
    return {"sampled_commands": n, "rudder_percent_count": 401,
            "rudder_percent_spacing": .5, "rpm_values": [-24.] + np.linspace(0., 12., 25).tolist(),
            "timeline": timeline, "noncontact_commands_at_1_2s": int(np.isinf(first).sum()),
            "noncontact_astern_commands_at_1_2s": int((np.isinf(first) & (rpm < 0)).sum()),
            "latest_first_contact_s": None if np.isinf(first).any() else float(first.max()),
            "best_minimum_sat_gap_through_1_2s_m": float(minimum[best]),
            "best_command": {"rudder_percent": float(rudder[best]), "rpm": float(rpm[best])}}


def brake_branches(initial, params, obstacles, first_command=None):
    base = plant(initial, params)
    start_time = 0.
    if first_command is not None:
        for _ in range(5):
            base.update(first_command["issued_rpm_signed"],
                        first_command["issued_rudder_percent"], COLLISION_DT)
        start_time = DECISION_DT
    results = []
    for rudder in (-100., 0., 100.):
        model = copy.deepcopy(base)
        first, minimum = None, np.inf
        trajectory = []
        for substep in range(1, 31):
            model.update(-24., rudder, COLLISION_DT)
            gap = float(static_gap(model._s, obstacles)[0])
            minimum = min(minimum, gap)
            elapsed = start_time + substep * COLLISION_DT
            if gap <= 0. and first is None:
                first = elapsed
            trajectory.append({"time_s": elapsed, "sat_gap_m": gap,
                               "state_u_v_r_hdg_servo_x_y": model._s[:, 0].tolist()})
        results.append({"rudder_percent": rudder, "rpm": -24.,
                        "brake_start_s": start_time, "first_contact_s": first,
                        "minimum_sat_gap_m": minimum, "trajectory": trajectory})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/safety_dev/development_v6_budget150/runs/v6_calibrated_hulls_pilot")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError("Tag must be one directory name")
    output = ROOT / "results/safety_dev/development_v6_budget150/offline" / args.tag
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())["configuration"]
    settings = manifest["settings"]
    for name in MODEL_FILES:
        if digest(ROOT / name) != settings["source_sha256"][name]:
            raise ValueError(f"Oracle model source changed since the recorded pilot: {name}")
    scale = float(settings["effective_constants"]["VESSEL_RANDOMISATION_SCALE"])
    if settings["effective_constants"]["RUDDER_COMMAND_LIMIT"] != "False":
        raise ValueError("This static-clone audit requires the recorded unlimited bridge")
    for name, expected in (("UPDATE_RATE", DECISION_DT), ("PHYSICS_DT", COLLISION_DT)):
        if float(settings["effective_constants"][name]) != expected:
            raise ValueError(f"Recorded timing differs from audit timing: {name}")
    result = {"utc": datetime.now(timezone.utc).isoformat(),
              "accounting": {"new_policy_episodes": 0, "environment_objects_constructed": 0,
                             "environment_reset_calls": 0, "environment_step_calls": 0,
                             "plant_reset_calls": 0, "evaluation_attempt_tokens_consumed": 0,
                             "operation": "Static oracle plant-state replay and batched control prediction only"},
              "parameter_access": {"plant": "ORACLE: exact randomized simulator parameters reconstructed from original evaluation reset seed + 104729",
                                   "initial_state": "ORACLE: recorded true before-state",
                                   "geometry": "ORACLE: canonical cached fixed-obstacle polygons",
                                   "learned_braking_model": "NOT USED: measurement-only calibration remains separate; its true-state fields were scoring-only",
                                   "controller_use": "These oracle inputs must not enter production safety decisions"},
              "provenance": {"script": relative(__file__), "script_sha256": digest(__file__),
                             "manifest": relative(manifest_path), "manifest_sha256": digest(manifest_path),
                             "model_source_sha256": {name: digest(ROOT / name) for name in MODEL_FILES},
                             "numpy_version": np.__version__, "python_version": sys.version.split()[0],
                             "integration_dt_s": INTEGRATION_DT, "collision_dt_s": COLLISION_DT},
              "cases": []}
    for case in CASES:
        record = next(c for c in manifest["cases"] if c["case"] == case)
        scene_files = sorted((ROOT / "results/safety_dev/scenario_cache").glob(f"*/{case}.pkl"))
        scene = scene_path = None
        for path in scene_files:
            candidate = pickle.loads(path.read_bytes())
            if candidate.digest() == record["scenario_sha256"]:
                scene, scene_path = candidate, path
                break
        if scene is None:
            raise ValueError(f"Canonical cached scene missing: {case}")
        episode_path = next(p for p in sorted(args.run.glob("episode_*.json"))
                            if json.loads(p.read_text())["case"] == case)
        trace_path = episode_path.with_name(episode_path.stem.replace("episode_", "trace_") + ".jsonl")
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
        initial = state(rows[0]["before"])
        if not np.allclose(initial[5:7], scene.own_spawn, rtol=0., atol=1e-6):
            raise ValueError(f"Recorded initial pose does not match cached scene: {case}")
        params = ship.sample_params(np.random.default_rng(record["seed"] + 104729), scale=scale)
        obstacles = np.asarray(scene.flags["fixed_obstacles"], dtype=float)
        for polygon in obstacles:
            if not np.all(np.isclose(polygon, polygon.min(axis=0))
                          | np.isclose(polygon, polygon.max(axis=0))):
                raise ValueError("This SAT implementation requires axis-aligned rectangular panels")
        replay = check_recorded_replay(initial, params, rows)
        result["cases"].append({"case": case, "reset_seed": record["seed"],
                                "hull_seed": record["seed"] + 104729,
                                "scenario_sha256": scene.digest(), "cache_file": relative(scene_path),
                                "cache_file_sha256": digest(scene_path),
                                "trace_file": relative(trace_path), "trace_sha256": digest(trace_path),
                                "episode_file_sha256": digest(episode_path), "oracle_parameters": params,
                                "oracle_reverse_efficiency": ship.REVERSE_THRUST_EFFICIENCY,
                                "initial_state_u_v_r_hdg_servo_x_y": initial.tolist(),
                                "fixed_obstacles": obstacles.tolist(),
                                "initial_sat_gap_m": float(static_gap(initial[:, None], obstacles)[0]),
                                "recorded_replay": replay,
                                "sampled_constant_command_grid": immediate_grid(initial, params, obstacles),
                                "immediate_brake": brake_branches(initial, params, obstacles),
                                "brake_after_recorded_first_decision": (
                                    brake_branches(initial, params, obstacles, rows[0]) if len(rows) > 1 else None)})
    result["limitations"] = [
        "The finite grid is not a continuous-control impossibility proof.",
        "Grid commands are constant for 1.2 s; for CV10 every sample already collides before the first 0.5 s decision ends.",
        "Only static panels are evaluated in the counterfactuals, not target trajectories or full mission success.",
        "The oracle simulator reverse law is an assumed simulated plant property, not a calibrated real-vessel braking guarantee.",
        "No obstacle observation is created; sensor visibility is established by the independent ray audit."]
    result["terminal_viability_note"] = {
        "observed_gap": "Earlier CRS-CV04 trace retains outward sway after surge reaches zero; existing terminal check omits boundaries, skips speeds <=0.10 m/s, and otherwise extends along hull heading.",
        "source_locations": ["src/safety_v2.py:61", "src/safety_v2.py:62", "src/safety_v2.py:179", "src/safety_v2.py:184"],
        "reference": "Wabersich and Zeilinger (2021), Section 4.1 Eq. (5f) and Assumption 4.2: terminal safe set with a known safety controller",
        "url": "https://arxiv.org/html/1812.05506v4",
        "limitation": "Neither an additional 3 s tail nor low surge speed establishes that terminal condition. A dynamics-based tail remains finite lookahead, not recursive feasibility.",
        "decision": "Terminal implementation deferred. Inspect the broad trajectory-search screen first; a simple longer tail largely repeats the previously tested longer-horizon approach."}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Offline blind-zone plant feasibility audit", "", result["utc"], "",
             "No new policy episodes, environment objects, reset/step calls, or evaluation tokens. "
             "This audit uses **oracle** recorded state, true geometry and seed-derived simulator parameters. "
             "It does not use or refit the measurement-only braking calibration.", "",
             "| Case | Initial inflated-hull SAT gap | Sampled first-command result | Immediate straight brake |",
             "|---|---:|---|---|"]
    for case in result["cases"]:
        grid = case["sampled_constant_command_grid"]
        branch = case["immediate_brake"][1]
        finding = (f"All {grid['sampled_commands']:,} contact by {grid['latest_first_contact_s']:.1f} s"
                   if grid["latest_first_contact_s"] is not None
                   else f"{grid['noncontact_commands_at_1_2s']} sampled commands remain clear through 1.2 s")
        brake = (f"Contact at {branch['first_contact_s']:.1f} s" if branch["first_contact_s"] is not None
                 else f"Clear for 3 s; minimum gap {branch['minimum_sat_gap_m']:.6f} m")
        lines.append(f"| {case['case']} | {case['initial_sat_gap_m']:.6f} m | {finding} | {brake} |")
    lines += ["", "BO-VS-05: taking the recorded first decision and then braking at 0.5 s contacts at 0.9 s "
              "for all three checked rudders. Its demonstrated immediate-stop clearance is below the "
              "filter's existing 0.15 m trigger even before the static gap subtraction. Initial astern is "
              "also excluded by the current no-traffic candidate gate. Neither limitation is changed here.", "",
              "The recorded partial decisions are reproduced to numerical precision; durations are inferred "
              "by matching each saved after-state at the original 0.1 s collision cadence. The full "
              "10,426-command grid uses 401 rudders and 26 RPM choices. These samples are **not a proof over "
              "continuous controls**. Static-panel survival is not full-episode success.", "",
              "An empty rudder delay line initially fills with the first command (ship.py:219–234); "
              "the initial test therefore already permits immediate arrival at the servo. Later commands "
              "retain the randomized actuator delay. All source, cache, trace and scenario hashes plus "
              "the exact seeds and oracle parameters are in results.json.", "",
              "The remaining lateral-drift issue is a terminal viability gap. A safe terminal region with "
              "a known controller is required by [Wabersich–Zeilinger, section 4.1 Eq. 5f and Assumption 4.2]"
              "(https://arxiv.org/html/1812.05506v4). Low surge speed or an additional 3 s predicted tail does "
              "not establish it. Terminal changes are deferred pending the broader trajectory-search "
              "screen; extending a tail alone largely repeats a previously unsuccessful horizon extension.", "",
              "Reproduce from the project directory with a fresh output tag:", "",
              "```powershell", "python -B tools/diagnostics/safety/blind_zone_feasibility.py --tag blind_zone_reproduction", "```", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(output)
    for case in result["cases"]:
        print(case["case"], json.dumps(case["sampled_constant_command_grid"]["timeline"]))


if __name__ == "__main__":
    main()
