"""Audit safety predictions on the development set, using executed command replay.

From the project root, for example::

    python tools/diagnostics/safety/prediction_audit.py --modes v3 \
        --cases DV3-OT-CV-16,DV3-BO-CV-04 --tag wall_prediction

No held-out sets are loaded. One process and one Torch thread are used. Every
sample replays the *actual future rpm and rudder commands*, not a held action
or the filter's hypothetical escape. This isolates prediction error from the
policy changing its mind. Two initial states are compared: the filter's held,
smoothed perception and the true ego state, with the SAME carried actuator
model in both. The latter still includes actuator/discretisation/model error.

Collision steps can stop physics before their nominal 0.5 s has elapsed. Such
steps are excluded from fixed-horizon errors, and reported separately through
the boundary contact's last ten decision modes. Error windows overlap; their
quantiles are descriptive, not independent confidence samples.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]

import numpy as np

import constants as cfg
import safety_v2 as v2
import safety_v3 as v3
import ship as shipmod
from classical import common as cc

HORIZONS = (1.0, 2.0, 4.0, 8.0)
DEVELOPMENT_CACHE_SCHEMA = 2          # bump if cache format or selective loader semantics change


def development_source_paths():
    """Local imports reachable from scene generation, including deferred imports.

    Environment/filter hooks do not determine a generated scene. Traversing the
    generator imports avoids invalidating 150 scenes when those hooks change,
    while newly imported generator helpers are included automatically.
    """
    roots = [ROOT / "src", ROOT / "bluefin"]
    queue = [ROOT / name for name in ("src/formulation_v3.py", "src/field_training.py",
                                    "src/scenario.py", "src/feasibility_st.py",
                                    "bluefin/dynamics.py", "bluefin/ship_model_v3.py")]
    found = set()

    def local_modules(name):
        path = Path(*name.split("."))
        for directory in roots:
            for candidate in (directory / path.with_suffix(".py"), directory / path / "__init__.py"):
                if candidate.is_file():
                    yield candidate

    while queue:
        path = queue.pop()
        if path in found or not path.is_file():
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    queue.extend(local_modules(alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                queue.extend(local_modules(node.module))
                for alias in node.names:
                    queue.extend(local_modules(node.module + "." + alias.name))
    return sorted(found)


def development_cache_digest():
    source_digest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest() for p in development_source_paths()}
    constants = {k: repr(v) for k, v in vars(cfg).items() if k.isupper()
                 and not k.startswith("SAFETY_")}
    return hashlib.sha256(json.dumps({"source": source_digest, "constants": constants,
                                     "schema": DEVELOPMENT_CACHE_SCHEMA}, sort_keys=True).encode()).hexdigest()[:20]


def import_known_cache(source_dir, *, rationale):
    """Explicitly import scene data whose generator/settings equivalence is known.

    This never searches for or trusts a legacy cache automatically. The caller
    must name it and explain why its scenes are valid under current settings.
    Existing destination entries are verified, never overwritten; sources stay
    intact. Repeating this operation imports newly completed source cases only.
    Apply the evaluation curriculum stage before calling, as for the loader.
    """
    import formulation_v3 as fv
    import field_training as ft
    if not rationale.strip():
        raise ValueError("An explicit cache-equivalence rationale is required")
    cache_root = (ROOT / "results" / "safety_dev" / "scenario_cache").resolve()
    source = Path(source_dir).resolve()
    if not source.is_relative_to(cache_root) or not source.is_dir():
        raise ValueError("The explicitly approved source must be a directory inside scenario_cache")
    destination = cache_root / development_cache_digest()
    if source == destination:
        raise ValueError("Source is already the current cache")
    plan = [("NT", fv.DEV_NO_TARGET, False)]
    for code in ft.ENCOUNTER_CODES:
        plan += [(code, fv.DEV_PER_ENCOUNTER_CV, False), (code, fv.DEV_PER_ENCOUNTER_VS, True)]
    known = {f"DV3-{code}-{'VS' if vary else 'CV'}-{i + 1:02d}"
             for code, count, vary in plan for i in range(count)}
    pending = []
    for path in sorted(source.glob("*.pkl")):
        data = path.read_bytes()
        built = pickle.loads(data)
        if path.stem not in known or built.case_id != path.stem:
            raise ValueError(f"Unexpected development case in approved cache: {path}")
        target = destination / path.name
        digest = built.digest()
        if target.exists():
            existing = pickle.loads(target.read_bytes())
            if existing.case_id != built.case_id or existing.digest() != digest:
                raise ValueError(f"Conflicting current-cache scene; refusing overwrite: {target}")
        pending.append((path, target, data, digest))
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    for path, target, data, digest in pending:
        try:
            with target.open("xb") as handle:
                handle.write(data)
            status = "imported"
        except FileExistsError:
            existing = pickle.loads(target.read_bytes())
            if existing.case_id != path.stem or existing.digest() != digest:
                raise ValueError(f"Conflicting concurrent cache write: {target}")
            status = "already present"
        cases.append({"case": path.stem, "scenario_sha256": digest,
                      "pickle_sha256": hashlib.sha256(data).hexdigest(), "status": status})
    manifest = {"source": str(source), "destination": str(destination),
                "cache_schema": DEVELOPMENT_CACHE_SCHEMA, "current_cache_digest": destination.name,
                "rationale": rationale, "source_files_preserved": True,
                "imported": sum(c["status"] == "imported" for c in cases),
                "already_present": sum(c["status"] == "already present" for c in cases),
                "generator_source_sha256_lf": {p.relative_to(ROOT).as_posix(): hashlib.sha256(
                    p.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest() for p in development_source_paths()},
                "cases": cases}
    manifest_path = destination / f"import_{source.name}_{uuid.uuid4().hex[:12]}.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest_path, manifest


def load_development_cases(cases=None):
    """Return (original full-set index, case), caching deterministic dev cases.

    This is the independent-case loop from formulation_v3.field_development_set:
    exact plan order, seeds, and sample kwargs, but unselected cases are skipped.
    All generator-related source and current public constants key the cache.
    The original full-set index is retained for the environment's reset seed.
    """
    import formulation_v3 as fv
    import field_training as ft
    import scenario as scn
    plan = [("NT", fv.DEV_NO_TARGET, False)]
    for code in ft.ENCOUNTER_CODES:
        plan += [(code, fv.DEV_PER_ENCOUNTER_CV, False), (code, fv.DEV_PER_ENCOUNTER_VS, True)]
    digest = development_cache_digest()
    cache = ROOT / "results" / "safety_dev" / "scenario_cache" / digest
    cache.mkdir(parents=True, exist_ok=True)
    wanted = set(cases or ())
    known = {f"DV3-{code}-{'VS' if vary else 'CV'}-{i + 1:02d}"
             for code, count, vary in plan for i in range(count)}
    if wanted - known:
        raise ValueError("Unknown development cases: " + ", ".join(sorted(wanted - known)))
    out, index = [], 0
    for block, (code, count, vary) in enumerate(plan):
        for i in range(count):
            case_id = f"DV3-{code}-{'VS' if vary else 'CV'}-{i + 1:02d}"
            case_index = index
            index += 1
            if wanted and case_id not in wanted:
                continue
            path = cache / f"{case_id}.pkl"
            if path.exists():
                with path.open("rb") as handle:
                    built = pickle.load(handle)
            else:
                base = fv.DEV_SEED_BASE + block * 1_000 + i * 20
                print(f"Building development case {case_id} (index {case_index})", flush=True)
                built = ft.sample(np.random.default_rng(base), namespace="dev_v3", encounter=code,
                                  varying=vary, generator=scn.ScenarioGenerator(stage=5, seed_namespace="development"),
                                  seed_fn=lambda k, base=base: base + 200_000 + k, solvable_only=True,
                                  near=(i % 10) < fv.DEV_NEAR_PER_TEN)
                built.case_id = case_id
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                with temporary.open("wb") as handle:
                    pickle.dump(built, handle, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, path)
            if built.case_id != case_id:
                raise RuntimeError(f"Scenario cache case mismatch: {path}")
            out.append((case_index, built))
    return out


def replay_commands(snap, actuators, commands):
    """Safety's identified dynamics for actual (rudder-unit, rpm) commands.

    The signed rpm is retained: clipped `_executed_action[1]` cannot represent
    zero thrust or astern and must not be used for this audit. Braking matches
    v2/v3's pessimistic efficiency and onset delay, restarting after release.
    """
    state = np.zeros((7, 1))
    state[:4, 0] = (max(0.0, snap.u), snap.v, snap.r, snap.heading)
    state[4, 0] = actuators.servo
    state[5:, 0] = (snap.x, snap.y)
    pending = ([np.array([x]) for x in actuators.buffer]
               if actuators.buffer is not None else None)
    params = {key: np.array([val]) for key, val in cc.IDENTIFIED.items()}
    pos, hdg = [], []
    brake_since = math.inf
    j = 0
    for rudder, rpm in commands:
        delta = np.array([-cc.MAX_RUDDER_RAD * float(rudder)])
        if pending is None:
            pending = [delta.copy() for _ in range(cc.DELAY_STEPS)]
        brake_decel = shipmod.braking_thrust(float(rpm), efficiency=v2.BRAKE_EFFICIENCY) / shipmod.M11
        for _ in range(cc.SUBSTEPS):
            pending.append(delta)
            state = cc.dyn.rk4_step(state, np.array([max(0.0, rpm)]),
                                    pending.pop(0), params, cc.PRED_DT)
            t_now = (j + 1) * cc.PRED_DT
            brake_since = min(brake_since, t_now) if rpm < 0 else math.inf
            if rpm < 0 and t_now - brake_since >= v2.BRAKE_DELAY_S:
                state[0, 0] = max(0.0, state[0, 0] - brake_decel * cc.PRED_DT)
            pos.append(state[5:7, 0].copy())
            hdg.append(float(state[3, 0]))
            j += 1
    return np.asarray(pos), np.asarray(hdg)


def classify_mode(last):
    why = last.get("why", "")
    if last.get("mode") == "idle":
        return "idle"
    if why == "no escape" or (last.get("mode") == "filter" and last.get("any_safe") is False):
        return "no escape"
    if why == "last certificate":
        return "last certificate"
    if last.get("mode") == "recovery" or last.get("changed", False):
        return "recovery"
    return "nominal"


def runtime_overrides():
    """Existing diagnostic syntax, parsed as literals rather than executable code."""
    result = {}
    modules = [("V2_SET", v2), ("V3_SET", v3)]
    if os.environ.get("V4_SET"):
        import safety_v4
        modules.append(("V4_SET", safety_v4))
    if os.environ.get("V5_SET"):
        import safety_v5
        modules.append(("V5_SET", safety_v5))
    for key, module in modules:
        for item in filter(None, os.environ.get(key, "").split(",")):
            name, raw = item.split("=", 1)
            name = name.strip()
            old = getattr(module, name)
            value = ast.literal_eval(raw)
            if not isinstance(value, type(old)):
                value = type(old)(value)
            setattr(module, name, value)
            result[f"{key}:{name}"] = value
    return result


def make_filter(mode):
    if mode == "v2":
        return v2.SafetyFilterV2()
    if mode in ("v3", "oracle", "oracle_state"):
        return v3.SafetyFilterV3()
    if mode in ("v4", "oracle4"):
        from safety_v4 import SafetyFilterV4
        return SafetyFilterV4()
    if mode in ("v5", "oracle5"):
        from safety_v5 import SafetyFilterV5
        return SafetyFilterV5()
    raise ValueError(mode)


def record_episode(env, built, seed, actor, mode, stride, max_steps, predictions=True):
    obs, _ = env.reset(seed=seed, options={"generated": built})
    env.estop_enabled = mode != "off"
    cfg.SAFETY_VERSION = (3 if mode in ("oracle", "oracle_state") else
                          int(mode[-1]) if mode != "off" else 1)
    filt = make_filter(mode) if mode != "off" else None
    if filt is not None:
        if mode.startswith("oracle"):
            from oracle import install_oracle
            install_oracle(filt, exact_geometry=mode != "oracle_state")
        env._safety_v2 = filt
        original_threat = filt._threat_in_reach
        capture = {}

        def capture_threat(snap):
            capture["snapshot"] = copy.copy(snap)
            return original_threat(snap)

        filt._threat_in_reach = capture_threat
        original_filter = filt.filter

        def timed_filter(scene, action):
            start = time.perf_counter()
            result = original_filter(scene, action)
            capture["filter_seconds"] = time.perf_counter() - start
            return result

        filt.filter = timed_filter
    else:
        shadow = cc.Perception(memory_frames=int(round(v2.MEMORY_S / cfg.UPDATE_RATE)))
        shadow_act = cc.Actuators()
        shadow_ego = None

    samples, commands, states, steps = [], [], [], []
    outcome = "diagnostic_limit"
    started = time.perf_counter()
    for k in range(max_steps):
        action = actor(obs)
        sampled = predictions and k % stride == 0
        if filt is not None:
            act = copy.deepcopy(filt.actuators) if sampled else None
            capture.clear()
        elif predictions:
            snap = shadow.snapshot(env)
            raw = np.array([snap.u, snap.v, snap.r])
            shadow_ego = raw if shadow_ego is None else shadow_ego + v2.EGO_SMOOTHING * (raw - shadow_ego)
            snap.u, snap.v, snap.r = map(float, shadow_ego)
            act = copy.deepcopy(shadow_act) if sampled else None
        # Capture simulator truth BEFORE issuing this decision's command.
        true = (float(env.asv_x), float(env.asv_y), math.radians(float(env.asv_h)),
                float(env.u_body), float(env.v_body), math.radians(float(env.asv_w)))
        measured = env._measured_ego() if filt is not None else None
        step_started = time.perf_counter()
        obs, _, term, trunc, info = env.step(action)
        step_seconds = time.perf_counter() - step_started
        commands.append((float(env.rudder) / 100.0, float(env.rpm)))
        states.append((float(env.asv_x), float(env.asv_y), math.radians(float(env.asv_h))))
        last = copy.deepcopy(filt.last) if filt is not None else {"mode": "off"}
        if filt is None and predictions:
            # No bridge limiter is applied a second time to its executed command.
            shadow_act.issue(type("Issued", (), {"command_rate_limit": False})(), commands[-1][0])
        elif sampled:
            if "snapshot" not in capture:
                raise RuntimeError("Filter no longer calls _threat_in_reach; update snapshot capture.")
            snap = capture["snapshot"]
        if sampled:
            perceived = copy.copy(snap)
            oracle = copy.copy(snap)
            oracle.x, oracle.y, oracle.heading, oracle.u, oracle.v, oracle.r = true
            # Replay does not use points/tracks: retain only geometry needed for
            # clearances and avoid keeping every growing scan-memory array alive.
            perceived.points = oracle.points = np.empty((0, 2))
            perceived.tracks = oracle.tracks = []
            samples.append((k, perceived, oracle, act, classify_mode(last)))
        row = {"mode": mode, "case": built.case_id, "step": k,
               "phase": classify_mode(last), "filter_mode": last.get("mode", ""),
               "why": last.get("why", ""), "changed": bool(last.get("changed", False)),
               "policy_margin": last.get("policy_margin", ""),
               "best_margin": last.get("best_margin", last.get("best_turn_margin", "")),
               "continuation_ok": last.get("continuation_ok", ""),
               "x_before": true[0], "y_before": true[1], "heading_before_deg": math.degrees(true[2]),
               "x_after": env.asv_x, "y_after": env.asv_y, "heading_after_deg": env.asv_h,
               "surge_before": true[3], "sway_before": true[4], "yaw_before_radps": true[5],
               "rudder": commands[-1][0], "rpm": commands[-1][1],
               "policy_rudder": float(action[0]), "policy_throttle": float(action[1]),
               "collision_kind": info.get("collision_kind") or ""}
        held = capture.get("snapshot") if filt is not None else None
        memory = getattr(filt.perception, "last_memory_stats", {}) if filt is not None else {}
        row.update({"snapshot_u_mps": held.u if held is not None else "",
                    "snapshot_v_mps": held.v if held is not None else "",
                    "snapshot_r_radps": held.r if held is not None else "",
                    "measured_u_mps": float(measured[0]) if measured is not None else "",
                    "measured_v_mps": float(measured[1]) if measured is not None else "",
                    "measured_r_radps": math.radians(float(measured[2])) if measured is not None else "",
                    "filter_seconds": capture.get("filter_seconds", 0.) if filt is not None else 0.,
                    "env_step_seconds": step_seconds,
                    "memory_cleared_points": memory.get("cleared_points", 0),
                    "memory_snapshot_points": memory.get("snapshot_points", len(held.points) if held is not None else 0),
                    "fast_brake_rejections": last.get("fast_brake_rejections", 0)})
        steps.append(row)
        if term or trunc:
            outcome = (f"collision:{info['collision_kind']}" if info.get("collided") else
                       "goal" if info.get("reached_goal") else "timeout")
            break
    valid_count = len(commands) - int(outcome.startswith("collision:"))
    errors = []
    for origin, perceived, oracle, act, phase in samples:
        n = min(int(round(max(HORIZONS) / cfg.UPDATE_RATE)), valid_count - origin)
        if n < int(round(min(HORIZONS) / cfg.UPDATE_RATE)):
            continue
        future_commands = commands[origin:origin + n]
        for initial, snap in (("perceived", perceived), ("true_ego", oracle)):
            positions, headings = replay_commands(snap, act, future_commands)
            for horizon in HORIZONS:
                decisions = int(round(horizon / cfg.UPDATE_RATE))
                if decisions > n:
                    continue
                j = decisions * cc.SUBSTEPS - 1
                actual = np.asarray(states[origin + decisions - 1])
                pp, ph = positions[j], headings[j]
                pred_clear = float(cc.boundary_clearance(pp[None, None], np.array([[ph]]),
                                                        snap.edges_a, snap.edges_b)[0, 0])
                true_clear = float(cc.boundary_clearance(actual[:2][None, None], np.array([[actual[2]]]),
                                                        snap.edges_a, snap.edges_b)[0, 0])
                # Compare running minima at identical decision boundaries, too.
                decision_indices = np.arange(cc.SUBSTEPS - 1, j + 1, cc.SUBSTEPS)
                actual_path = np.asarray(states[origin:origin + decisions])
                pmin = cc.boundary_clearance(positions[decision_indices, None],
                                             headings[decision_indices, None], snap.edges_a, snap.edges_b).min()
                amin = cc.boundary_clearance(actual_path[:, None, :2], actual_path[:, None, 2],
                                             snap.edges_a, snap.edges_b).min()
                heading_error = math.degrees(float(cc.wrap_pi(ph - actual[2])))
                errors.append({"mode": mode, "case": built.case_id, "origin_step": origin,
                               "phase": phase, "initial_state": initial, "horizon_s": horizon,
                               "position_error_m": float(np.linalg.norm(pp - actual[:2])),
                               "heading_error_deg": heading_error, "heading_abs_error_deg": abs(heading_error),
                               "pred_boundary_m": pred_clear, "actual_boundary_m": true_clear,
                               "boundary_error_m": pred_clear - true_clear,
                               "pred_min_boundary_m": float(pmin), "actual_min_boundary_m": float(amin),
                               "min_boundary_error_m": float(pmin - amin),
                               "initial_position_error_m": float(np.hypot(snap.x - oracle.x, snap.y - oracle.y)),
                               "brake_decisions": sum(rpm < 0 for _, rpm in future_commands[:decisions])})
    episode = {"mode": mode, "case": built.case_id, "seed": seed, "outcome": outcome,
               "scenario_sha256": built.digest(),
               "steps": len(commands), "interventions": sum(s["changed"] for s in steps),
               "seconds": round(time.perf_counter() - started, 3),
               "excluded_partial_collision_steps": int(outcome.startswith("collision:")),
               "plant_params_json": json.dumps(env.model.p, sort_keys=True)}
    boundary = steps[-10:] if outcome == "collision:boundary" else []
    return episode, errors, steps, boundary


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_metadata(args, modes, overrides):
    """Snapshot reproducibility data before any episodes run."""
    sources = ["src/safety_v2.py", "src/safety_v3.py", "src/classical/common.py", "src/env.py", "src/ship.py",
               "src/constants.py", "src/constant_temp.py", "src/train_formulation.py",
               "bluefin/dynamics.py", "bluefin/ship_model_v3.py", "tools/diagnostics/safety/prediction_audit.py"]
    if set(modes) & {"v4", "v5", "oracle4", "oracle5"}:
        sources.append("src/safety_v4.py")
        sources.extend(name for name in ("src/safety_prediction.py", "src/safety_perception.py", "src/safety_observer.py")
                       if (ROOT / name).exists())
    if set(modes) & {"v5", "oracle5"}:
        sources.append("src/safety_v5.py")
        sources.extend(name for name in ("src/safety_recovery.py", "src/safety_feedback.py")
                       if (ROOT / name).exists())
    if any(mode.startswith("oracle") for mode in modes):
        sources.append("tools/diagnostics/safety/oracle.py")
    model_config = args.model.parent / "config.json"
    modules = {"v2": v2, "v3": v3}
    if set(modes) & {"v4", "v5", "oracle4", "oracle5"} or os.environ.get("V4_SET"):
        import safety_v4
        modules["v4"] = safety_v4
    if set(modes) & {"v5", "oracle5"} or os.environ.get("V5_SET"):
        import safety_v5
        modules["v5"] = safety_v5
    return {"command": list(sys.argv), "development_only": True, "model": str(args.model.resolve()),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "model_config_sha256": hashlib.sha256(model_config.read_bytes()).hexdigest() if model_config.exists() else None,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
            "scenario_cache_digest": development_cache_digest(), "scenario_cache_schema": DEVELOPMENT_CACHE_SCHEMA,
            "runtime_overrides": dict(overrides), "horizons_s": HORIZONS, "stride": args.stride,
            "effective_safety_constants": {name: {key: repr(value) for key, value in vars(module).items()
                                                   if key.isupper()} for name, module in modules.items()},
            "prediction_dt": cc.PRED_DT, "update_rate": cfg.UPDATE_RATE,
            "source_hash_time": "before scenario generation, model loading and episode evaluation",
            "oracle_modes": {"oracle": "v3 true ego and targets, exact static geometry",
                             "oracle_state": "v3 true ego and targets, original remembered scan points unchanged",
                             "oracle4": "v4 true ego and targets, exact static geometry",
                             "oracle5": "v5 true ego and targets, exact static geometry"},
            "boundary_error_sign": "predicted minus actual; negative means pessimistic",
            "true_ego": "true initial position, heading and body velocity; same estimated actuator history",
            "boundary_geometry": "same inflated hull and common.boundary_clearance on known map polygon",
            "partial_collision_steps": "excluded from horizon errors, retained in mode traces",
            "limitations": ["No terminal-runout or hypothetical-backup-plan score is compared to actual motion.",
                            "True-ego replay is not an oracle closed-loop filter evaluation.",
                            "Running clearance minima are sampled at decision boundaries, not continuous time.",
                            "Overlapping windows are not independent statistical samples."]}


def reserve_run(output, tag, metadata):
    """Refuse existing artifacts and claim the tag with an exclusive metadata file.

    Completed episodes may update this run's CSV checkpoints; another process
    cannot claim its tag. An interrupted run retains its metadata and results.
    """
    output.mkdir(parents=True, exist_ok=True)
    suffixes = ("episodes.csv", "errors.csv", "steps.csv", "boundary_modes.csv", "summary.csv", "metadata.json")
    existing = [output / f"{tag}_{suffix}" for suffix in suffixes if (output / f"{tag}_{suffix}").exists()]
    if existing:
        raise FileExistsError(f"Run tag {tag!r} already has artifacts; choose a new --tag ({existing[0]}).")
    # Exclusive creation also handles two auditors racing for the same tag.
    with (output / f"{tag}_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def summaries(rows):
    result = []
    keys = sorted({(r["mode"], r["initial_state"], r["horizon_s"]) for r in rows})
    for mode, initial, horizon in keys:
        group = [r for r in rows if (r["mode"], r["initial_state"], r["horizon_s"]) == (mode, initial, horizon)]
        entry = {"mode": mode, "initial_state": initial, "horizon_s": horizon, "samples": len(group)}
        for metric in ("position_error_m", "heading_abs_error_deg", "boundary_error_m", "min_boundary_error_m"):
            values = np.array([r[metric] for r in group])
            entry[metric + "_mean"] = float(values.mean())
            for name, q in (("p05", .05), ("p50", .5), ("p95", .95)):
                entry[metric + "_" + name] = float(np.quantile(values, q))
        result.append(entry)
    return result


def self_check():
    """Cheap deterministic check against the existing v3 sequence predictor."""
    snap = cc.Snapshot(0., 0., .3, .7, .02, -.04, np.array([0., 1.]),
                       np.array([1., 0.]), np.zeros(2), 0., 0., 10., np.empty((0, 2)))
    act = cc.Actuators()
    controls = np.array([[.5, 0.], [-1., -.5], [1., np.nan], [1., np.nan],
                         [0., np.nan], [0., np.nan], [-.5, .3], [0., 0.]])
    expected = v3.rollout_seq(snap, act, controls[None])
    commands = list(zip(controls[:, 0], v2._rpm(controls[:, 1])))
    positions, headings = replay_commands(snap, act, commands)
    np.testing.assert_allclose(positions, expected.positions[:, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(headings, expected.headings[:, 0], rtol=0, atol=1e-12)
    assert classify_mode({"why": "no escape", "mode": "filter"}) == "no escape"
    assert classify_mode({"why": "last certificate", "mode": "recovery"}) == "last certificate"
    print("Self-check passed: replay matches v3 forward/braking sequence predictor.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", default="v3", help="comma-separated off,v2,v3,v4,v5,oracle,oracle_state,oracle4,oracle5; oracle_state retains scan points")
    parser.add_argument("--cases", default="", help="comma-separated development case IDs; empty = all 150")
    parser.add_argument("--tag", default="prediction_audit")
    parser.add_argument("--stride", type=int, default=4, help="sample one origin per N decisions")
    parser.add_argument("--max-steps", type=int, default=400, help="diagnostic cap; unfinished cases remain diagnostic_limit")
    parser.add_argument("--model", type=Path, default=ROOT / "runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--no-predictions", action="store_true", help="outcomes and mode traces only, without replay overhead")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.stride < 1 or args.max_steps < 1:
        parser.error("--stride and --max-steps must be positive")
    if Path(args.tag).name != args.tag or args.tag in (".", "..") or "\\" in args.tag:
        parser.error("--tag must be a plain filename stem")
    modes = args.modes.split(",")
    if any(mode not in ("off", "v2", "v3", "oracle", "oracle_state", "v4", "v5", "oracle4", "oracle5") for mode in modes):
        parser.error("--modes must contain off,v2,v3,v4,v5,oracle,oracle_state,oracle4,oracle5")
    import torch
    torch.set_num_threads(1)
    import curriculum
    import train_formulation as tf
    from common import load_model
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    overrides = runtime_overrides()
    # Load optional filter code before capturing its source version, rather than
    # importing it lazily after a long case-generation interval.
    if set(modes) & {"v4", "v5", "oracle4", "oracle5"}:
        import safety_v4
    if set(modes) & {"v5", "oracle5"}:
        import safety_v5
    if any(mode.startswith("oracle") for mode in modes):
        import oracle
    output = ROOT / "results" / "safety_dev"
    metadata = run_metadata(args, modes, overrides)
    try:
        reserve_run(output, args.tag, metadata)
    except FileExistsError as exc:
        parser.error(str(exc))
    selected = set(filter(None, args.cases.split(",")))
    try:
        cases = load_development_cases(selected)
    except ValueError as exc:
        parser.error(str(exc))
    metadata["cases"] = [{"case": built.case_id, "full_set_index": i, "reset_seed": 900_120 + i,
                          "scenario_sha256": built.digest()} for i, built in cases]
    # The source hashes above are retained; do not rehash files after evaluation.
    (output / f"{args.tag}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    model = load_model(args.model)
    episodes, errors, steps, boundaries = [], [], [], []
    env = ASVLidarEnv(render_mode=None, emergency_stop=True)
    try:
        for mode in modes:
            for i, built in cases:
                episode, er, st, bd = record_episode(env, built, 900_120 + i, tf.EpisodeActor(model),
                                                     mode, args.stride, args.max_steps, not args.no_predictions)
                episodes.append(episode)
                errors.extend(er)
                steps.extend(st)
                boundaries.extend(bd)
                print(json.dumps({key: episode[key] for key in
                                  ("mode", "case", "outcome", "steps", "interventions", "seconds")}),
                      flush=True)
                # Each completed episode is durable if a long audit is interrupted.
                for suffix, rows in (("episodes", episodes), ("errors", errors),
                                     ("steps", steps), ("boundary_modes", boundaries)):
                    write_csv(output / f"{args.tag}_{suffix}.csv", rows)
    finally:
        env.close()
    summary = summaries(errors)
    write_csv(output / f"{args.tag}_summary.csv", summary)
    print(f"Wrote {len(errors)} horizon comparisons to {output / (args.tag + '_summary.csv')}", flush=True)


if __name__ == "__main__":
    main()
