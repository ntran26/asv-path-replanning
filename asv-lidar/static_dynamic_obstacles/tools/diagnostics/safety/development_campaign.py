"""Freeze and run explicit DV3 development cases within a NEW 150-attempt cap.

  python -B tools/diagnostics/safety/development_campaign.py prepare --tag pilot --cases DV3-CRP-VS-04 --modes v6 --trace
  python -B tools/diagnostics/safety/development_campaign.py run --tag pilot

Alternatively prepare with --selection PATH containing a list, or {"cases":list},
of DV3 IDs/canonical case records. --v6-set accepts a JSON object (default V6_SET).
Preparation runs no episodes. Run never retries or resumes automatically. All
tags share development_v6_budget150/attempts; interrupted/failed attempts count.
Create development_v6_budget150/STOP to stop before the next attempt.
The previous quick campaign and its 100-attempt ledger are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import time
import zipfile

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/tiers")]
from suite_status import read_shared_bytes

CAMPAIGN = ROOT / "results/safety_dev/development_v6_budget150"
REFERENCE = ROOT / "results/safety_dev/quick_v5_budget100/runs/policy_feedback_v1/metadata.json"
INVENTORY = ROOT / "results/safety_dev/suites/inventory_v5/manifest.json"
LIMIT = 150
MODES = ("off", "v4", "v5", "v6")
RUNNER_SHA256 = hashlib.sha256(read_shared_bytes(Path(__file__))).hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reference_source_matches(name, expected, actual):
    """Accept exact bytes or the single audited native version6 dispatch edit."""
    # Structural equivalence is recorded in this campaign's integration_audit.json.
    # This directed exception does not approve any other file or environment edit.
    return expected == actual or (
        name == "src/env.py"
        and expected == "3664ee0c505fa18f26d7c0250d1ec0fff40f45d83f80c663d22cdda0ede6fb9b"
        and actual == "608d2dce1f3e17efa6ff0eadee0831cbb023482c7422f02ac73bdda4bd280ccb"
    )


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class StopRequested(RuntimeError):
    pass


def check_stop(campaign):
    if (campaign / "STOP").exists():
        raise StopRequested("Development STOP marker present; no next attempt reserved")


def reserve_attempt(directory, identity, *, limit=LIMIT):
    if type(limit) is not int or not 1 <= limit <= LIMIT:
        raise ValueError("Development attempt cap must be between1 and150")
    check_stop(directory.parent)
    directory.mkdir(parents=True, exist_ok=True)
    for number in range(1, limit + 1):
        path = directory / f"{number:03d}.json"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(identity, attempt=number), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return number
    raise RuntimeError(f"HARD DEVELOPMENT BUDGET EXHAUSTED: {limit} consumed attempt slots")


def remaining_budget():
    return sum(not (CAMPAIGN / "attempts" / f"{n:03d}.json").exists() for n in range(1, LIMIT + 1))


def select_cases(requested, inventory):
    canonical = {c["case"]: c for c in inventory if c["suite"] == "dev_field"}
    selected, seen = [], set()
    for entry in requested:
        supplied = {"case": entry} if isinstance(entry, str) else entry
        if not isinstance(supplied, dict) or supplied.get("suite", "dev_field") != "dev_field":
            raise ValueError("This development runner accepts DV3 dev_field cases only")
        name = supplied.get("case")
        if name not in canonical or name in seen:
            raise ValueError(f"Unknown or duplicate DV3 case: {name}")
        row = canonical[name]
        for key in ("seed", "scenario_sha256", "obstacles", "test_id"):
            if key in supplied and supplied[key] != row[key]:
                raise ValueError(f"Selection canonical {key} mismatch: {name}")
        selected.append(dict(row))
        seen.add(name)
    if not selected:
        raise ValueError("At least one explicit DV3 case is required")
    return selected


def parse_modes(value):
    modes = value.split(",") if isinstance(value, str) else list(value)
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MODES):
        raise ValueError("Use unique modes from off,v4,v5,v6")
    return modes


def prepare_runtime(modes, overrides):
    import torch
    torch.set_num_threads(1)
    import constants as cfg
    import curriculum
    import train_formulation as tf
    from prediction_audit import runtime_overrides
    from suite_eval import settings_snapshot
    reference_bytes = read_shared_bytes(REFERENCE)
    baseline = json.loads(reference_bytes)["settings"]
    for name, expected in baseline["source_sha256"].items():
        if not reference_source_matches(name, expected, sha(read_shared_bytes(ROOT / name))):
            raise ValueError(f"Frozen reference source changed: {name}")
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    model = Path(baseline["checkpoint"])
    config = model.with_name("config.json")
    if config.exists():
        for name, value in (json.loads(read_shared_bytes(config)).get("constant_overrides") or {}).items():
            setattr(cfg, name, value)
    runtime_overrides()
    candidate = importlib.import_module("safety_v6") if "v6" in modes else None
    if not isinstance(overrides, dict) or (overrides and candidate is None):
        raise ValueError("V6 overrides must be an object and require mode v6")
    if candidate is not None:
        for name, value in overrides.items():
            if not name.isupper() or not hasattr(candidate, name):
                raise ValueError(f"Unknown V6 module constant: {name}")
            setattr(candidate, name, value)
    # V6 intentionally uses the env's v5 hook; settings record both.
    settings = settings_snapshot(model, ["dev_field"], ["v5"])
    for key in ("checkpoint_sha256", "config_sha256", "effective_constants", "runtime_versions"):
        if settings[key] != baseline[key]:
            raise ValueError(f"Frozen reference {key} changed")
    for mode in ("v2", "v3", "v4", "v5"):
        if settings["effective_safety_constants"][mode] != baseline["effective_safety_constants"][mode]:
            raise ValueError(f"Frozen reference {mode} settings changed")
    for name in set(settings["source_sha256"]) - set(baseline["source_sha256"]):
        if not re.fullmatch(r"src/safety_[A-Za-z0-9_]+\.py", name):
            raise ValueError(f"Unapproved new non-safety source: {name}")
    settings["modes"] = modes
    if candidate is not None:
        from safety_v4 import SafetyFilterV4
        if not issubclass(candidate.SafetyFilterV6, SafetyFilterV4):
            raise ValueError("SafetyFilterV6 must inherit the frozen selected SafetyFilterV4")
        settings["effective_safety_constants"]["v6"] = {
            k: repr(v) for k, v in vars(candidate).items() if k.isupper()}
    classes = {"off": None, "v4": "safety_v4.SafetyFilterV4", "v5": "safety_v5.SafetyFilterV5",
               "v6": "safety_v6.SafetyFilterV6"}
    settings["actual_filter_classes"] = {m: classes[m] for m in modes}
    settings["v6_injection"] = "temporary safety_v5.SafetyFilterV5 replacement; env hook version5"
    settings["v6_overrides"] = overrides
    return cfg, model, candidate, settings, sha(reference_bytes)


def load_cached_cases(cases):
    from prediction_audit import development_cache_digest
    cache = ROOT / "results/safety_dev/scenario_cache" / development_cache_digest()
    scenes, hashes = {}, {}
    for case in cases:
        path = cache / (case["case"] + ".pkl")
        if not path.exists():
            raise FileNotFoundError(f"Canonical development cache missing; no regeneration attempted: {path}")
        raw = read_shared_bytes(path)
        built = pickle.loads(raw)
        if built.case_id != case["case"] or built.digest() != case["scenario_sha256"]:
            raise ValueError(f"Cached scene differs from canonical inventory: {case['case']}")
        scenes[case["case"]] = built
        hashes[case["case"]] = sha(raw)
    return scenes, {"digest": cache.name, "files_sha256": hashes}


def configuration(cases, modes, overrides, trace):
    cfg, model, candidate, settings, reference_sha = prepare_runtime(modes, overrides)
    inventory_raw = read_shared_bytes(INVENTORY)
    canonical = select_cases(cases, json.loads(inventory_raw)["cases"])
    scenes, cache = load_cached_cases(canonical)
    sources = dict(settings["source_sha256"])
    for path in (Path(__file__), Path(__file__).with_name("suite_status.py")):
        sources[path.relative_to(ROOT).as_posix()] = sha(read_shared_bytes(path))
    if sources[Path(__file__).relative_to(ROOT).as_posix()] != RUNNER_SHA256:
        raise ValueError("Development runner source changed after process import")
    frozen = {"schema": 1, "cases": canonical, "modes": modes, "settings": settings,
              "reference_metadata_sha256": reference_sha, "inventory_sha256": sha(inventory_raw),
              "cache": cache, "source_sha256": sources, "trace": bool(trace),
              "campaign": str(CAMPAIGN), "shared_attempt_cap": LIMIT,
              "planned_episodes": len(canonical) * len(modes), "scope": "explicit DV3 development diagnostics only"}
    return frozen, (cfg, model, candidate, scenes)


def prepare(tag, requested, modes, overrides, trace, selection_origin):
    check_stop(CAMPAIGN)
    frozen, _ = configuration(requested, modes, overrides, trace)
    if frozen["planned_episodes"] > remaining_budget():
        raise RuntimeError("Selected modes/cases exceed the remaining shared150 attempt budget")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, expected in frozen["source_sha256"].items():
            raw = read_shared_bytes(ROOT / name)
            if sha(raw) != expected:
                raise ValueError(f"Source changed while freezing: {name}")
            archive.writestr(name, raw)
    archive_bytes = archive_buffer.getvalue()
    directory = CAMPAIGN / "runs" / tag
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "evaluated_sources.zip").open("xb") as handle:
        handle.write(archive_bytes)
    write_json(directory / "manifest.json", {"configuration": frozen, "selection_origin": selection_origin,
        "archive_sha256": sha(archive_bytes), "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"Prepared {len(frozen['cases'])} DV3 cases x {len(modes)} modes = {frozen['planned_episodes']} attempts; {remaining_budget()}/150 remain. No episodes run.")


@contextmanager
def injected_filter(mode, candidate):
    if mode != "v6":
        yield
        return
    import safety_v5
    original = safety_v5.SafetyFilterV5
    safety_v5.SafetyFilterV5 = candidate.SafetyFilterV6
    try:
        yield
    finally:
        safety_v5.SafetyFilterV5 = original


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def tracker_state(env):
    """Copy existing tracks before the decision; include unpublished tracks."""
    result = []
    for track in getattr(getattr(env, "tracker", None), "tracks", []):
        evidence = getattr(track, "last_evidence", None)
        row = {name: jsonable(getattr(track, name, None)) for name in (
            "id", "position", "velocity", "hits", "misses", "age", "confirmed", "is_dynamic",
            "fit_offset", "fit_offset_known", "last_fit_centre", "last_fit_heading_deg",
            "evidence_run", "quiet_run", "_pending", "_pending_steps")}
        row["last_evidence"] = None if evidence is None else {
            name: jsonable(getattr(evidence, name, None))
            for name in ("appear", "vacate", "compared", "violations", "moving")}
        result.append(row)
    return result


def observed_environment(base, mode, trace_handle):
    class AuditEnv(base):
        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            self.development_counts = {"diagnostic_true_steps": Counter(), "filter_mode_counts": Counter(),
                "recovery_template_counts": Counter(), "actual_filter_class": None, "observed_decisions": 0}
            return result

        def step(self, action):
            before = {k: getattr(self, k, None) for k in ("asv_x", "asv_y", "asv_h", "u_body", "v_body", "asv_w")}
            policy_action = jsonable(action)
            if trace_handle is not None:
                # Existing held telemetry and actual simulator state only; no
                # new sensor call, random draw, target update or physics step.
                held_ego = jsonable(getattr(self, "_ego_hold", None))
                targets_before = [{"id": i, "position_m": [t.x, t.y],
                    "velocity_mps": jsonable(t.velocity), "heading_deg": t.heading_deg}
                    for i, t in enumerate(getattr(self, "targets", []))]
                raw_tracks_before = tracker_state(self)
                published_track_ids_before = [int(t.id) for t in getattr(self, "tracks", [])]
                static_geometry = None
                if self.development_counts["observed_decisions"] == 0:
                    static_geometry = {"obstacle_polygons_m": jsonable(getattr(self, "obstacles", [])),
                                       "boundary_polygon_m": jsonable(getattr(self, "boundary_polygon", None))}
            result = super().step(action)
            filt = getattr(self, "_safety_v2", None)
            actual = None if filt is None else type(filt).__module__ + "." + type(filt).__qualname__
            expected = None if mode == "off" else f"safety_{mode}.SafetyFilter{mode.upper()}"
            if actual != expected:
                raise ValueError(f"Actual filter dispatch mismatch: wanted {expected}, got {actual}")
            last = jsonable(getattr(filt, "last", {}))
            counts = self.development_counts
            counts["actual_filter_class"] = actual
            counts["observed_decisions"] += 1
            for key, value in last.items():
                if value is True:
                    counts["diagnostic_true_steps"][key] += 1
            counts["filter_mode_counts"][str(last.get("mode", "off"))] += 1
            if last.get("recovery_template") is not None:
                counts["recovery_template_counts"][str(last["recovery_template"])] += 1
            if trace_handle is not None:
                snap = getattr(filt, "_observer_snapshot", None)
                observer_snapshot = None if snap is None else {
                    "timing": "filter input before dynamics; read after the one real step",
                    "x_m": snap.x, "y_m": snap.y, "heading_rad": snap.heading,
                    "u_mps": snap.u, "v_mps": snap.v, "r_radps": snap.r,
                    "static_point_count": len(snap.points),
                    "tracks": [{"id": t.id, "position_m": jsonable(t.position),
                                "velocity_mps": jsonable(t.velocity), "heading_rad": t.heading}
                               for t in snap.tracks]}
                rudder_percent, rpm = getattr(self, "rudder", None), getattr(self, "rpm", None)
                perception = getattr(filt, "perception", None)
                row = {"decision": counts["observed_decisions"], "policy_action": policy_action, "before": before,
                       "after": {k: getattr(self, k, None) for k in before},
                       "state_units": {"asv_x": "m", "asv_y": "m", "asv_h": "deg compass",
                                       "u_body": "m/s", "v_body": "m/s", "asv_w": "deg/s"},
                       "measured_ego_before": None if held_ego is None else {
                           "u_mps": held_ego[0], "v_mps": held_ego[1], "r_radps": math.radians(held_ego[2])},
                       "targets_before": targets_before, "observer_snapshot": observer_snapshot,
                       "tracker_tracks_before": raw_tracks_before,
                       "published_dynamic_track_ids_before": published_track_ids_before,
                       "perception_memory": jsonable(getattr(getattr(filt, "perception", None), "last_memory_stats", {})),
                       "perception_hull": jsonable(getattr(perception, "last_hull_stats", {})),
                       "perception_provisional": jsonable(getattr(perception, "last_provisional_stats", {})),
                       "perception_statistics": {name: jsonable(value) for name, value in
                           (vars(perception).items() if perception is not None else [])
                           if name.startswith("last_") and name.endswith("_stats")},
                       "issued_rudder_percent": rudder_percent,
                       "issued_rudder_normalized": None if rudder_percent is None else rudder_percent / 100.0,
                       "issued_rpm_signed": rpm, "issued_propulsion_s2": getattr(self, "propulsion_s2", None),
                       "executed_action": jsonable(getattr(self, "_executed_action", None)), "actual_filter_class": actual,
                       "action_changed": bool(getattr(self, "safety_v2_changed", False)), "filter": last}
                if static_geometry is not None:
                    row["static_geometry_at_first_decision"] = static_geometry
                trace_handle.write(json.dumps(jsonable(row), allow_nan=True) + "\n")
                trace_handle.flush()
            return result
    return AuditEnv


def run(tag):
    check_stop(CAMPAIGN)
    directory = CAMPAIGN / "runs" / tag
    manifest_raw = read_shared_bytes(directory / "manifest.json")
    manifest = json.loads(manifest_raw)
    wanted = manifest["configuration"]
    frozen, runtime = configuration(wanted["cases"], wanted["modes"], wanted["settings"]["v6_overrides"], wanted["trace"])
    if frozen != wanted or sha(read_shared_bytes(directory / "evaluated_sources.zip")) != manifest["archive_sha256"]:
        raise ValueError("Frozen development configuration/source/cache changed; prepare a new tag")
    if frozen["planned_episodes"] > remaining_budget():
        raise RuntimeError("Insufficient shared development budget; no episode started")
    cfg, model_path, candidate, scenes = runtime
    lock = CAMPAIGN / "active_run.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "tag": tag}, handle)
    rows = []
    try:
        write_json(directory / "started.json", {"pid": os.getpid(), "manifest_sha256": sha(manifest_raw)})
        from common import load_model, run_episode
        from env import ASVLidarEnv
        model = load_model(model_path)
        for case in frozen["cases"]:
            for mode in frozen["modes"]:
                identity = {"tag": tag, "mode": mode, "suite": "dev_field", "case": case["case"],
                            "seed": case["seed"], "scenario_sha256": case["scenario_sha256"],
                            "run_manifest_sha256": sha(manifest_raw)}
                number = reserve_attempt(CAMPAIGN / "attempts", identity)
                env, trace = None, None
                try:
                    if frozen["trace"]:
                        trace = (directory / f"trace_{number:03d}.jsonl").open("x", encoding="utf-8", newline="\n")
                    cfg.SAFETY_VERSION = 1 if mode == "off" else 5 if mode == "v6" else int(mode[1:])
                    with injected_filter(mode, candidate):
                        env = observed_environment(ASVLidarEnv, mode, trace)(render_mode=None, emergency_stop=mode != "off")
                        started = time.perf_counter()
                        result = run_episode(env, scenes[case["case"]], case["seed"], "model", model, case["obstacles"])
                    row = dict(result, **identity, attempt=number, seconds=time.perf_counter()-started, **env.development_counts)
                    write_json(directory / f"episode_{number:03d}.json", row)
                    rows.append(row)
                    print(f"{len(rows)}/{frozen['planned_episodes']}; budget{number}/150; {mode} {case['case']}: {row['outcome']}", flush=True)
                except BaseException as exc:
                    write_json(directory / f"error_{number:03d}.json", dict(identity, attempt=number, error=repr(exc)))
                    raise
                finally:
                    if env is not None:
                        env.close()
                    if trace is not None:
                        trace.flush()
                        os.fsync(trace.fileno())
                        trace.close()
        write_json(directory / "complete.json", {"completed": len(rows), "expected": frozen["planned_episodes"],
            "outcomes": dict(Counter(r["mode"] + "/" + r["outcome"] for r in rows)), "rows": rows})
    except StopRequested:
        write_json(directory / "stopped.json", {"completed": len(rows), "expected": frozen["planned_episodes"], "rows": rows})
        print("Stopped before next attempt; completed results and consumed tokens preserved.", flush=True)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cases", help="comma-separated explicit DV3 case IDs")
    selection.add_argument("--selection", type=Path)
    parser.add_argument("--modes", default="v6")
    parser.add_argument("--v6-set", help="JSON object of existing V6 constants; prepare defaults to V6_SET or {}")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("Use a plain tag containing letters, digits, underscores or hyphens")
    if args.action == "run":
        if args.cases or args.selection or args.trace or args.modes != "v6" or args.v6_set is not None:
            parser.error("run uses only the frozen manifest; pass selection/settings to prepare")
        run(args.tag)
    else:
        if not args.cases and args.selection is None:
            parser.error("prepare requires --cases or --selection")
        if args.selection is not None:
            raw = read_shared_bytes(args.selection)
            selected = json.loads(raw)
            requested = selected["cases"] if isinstance(selected, dict) else selected
            origin = {"path": str(args.selection.resolve()), "sha256": sha(raw)}
        else:
            requested = args.cases.split(",")
            origin = {"explicit_case_ids": requested}
        prepare(args.tag, requested, parse_modes(args.modes), json.loads(args.v6_set or os.environ.get("V6_SET", "{}")), args.trace, origin)


if __name__ == "__main__":
    main()
