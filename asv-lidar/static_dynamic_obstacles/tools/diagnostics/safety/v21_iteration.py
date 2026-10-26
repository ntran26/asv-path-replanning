"""Immutable paired evaluation for the V21 phase: refining V16 and V16 with the boundary-and-targets stop.

The V20 runner (`v20_iteration.py`) with three changes, so its records stay as run:

* development outputs go to results/safety_dev/v21_development/runs/<tag>/ (test-set-v4
  outputs, run only on request, still go to results/safety_dev/testset_v4_main/runs/<tag>/);
* the DVD namespace: the 120 decoupled validation episodes, read from
  results/safety_dev/scenario_cache/dvd_v3/ (prepare_validation_set.py) and verified
  against the selection's seed and digest;
* a tag that holds a DVD case installs the `formulation_v3` overlay before any
  environment exists, because the environment draws those scenes' panels at reset
  under it (as the deployed policy's selection runs did).  DVD may share a tag with
  DV3 (fixed layouts, unaffected by the overlay) but not with TS2/TS3/TS4, whose
  panels are drawn without it.

Same episode protocol otherwise: explicit filter instance after reset, one Torch/native
thread, frozen sources and selection, reserved attempts, per-episode result JSON and
decision traces, no retries or result reuse, STOP marker.

Example (project root):
    python -B tools/diagnostics/safety/v21_iteration.py --tag val_off_v16_a         --selection results/safety_dev/v21_development/selection_validation.json         --modes off,v16,v16_estop_bt --compact-traces --gzip-traces
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import gzip
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import zipfile

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/tiers"), str(Path(__file__).resolve().parent)]
import v9_paired_pilot as pilot  # noqa: E402
import estop_protocol  # noqa: E402
from v9_paired_pilot import MODEL, EXPECTED_MODEL, jsonable, sha, write_json  # noqa: E402

DEVELOPMENT_OUT = ROOT / "results/safety_dev/v21_development/runs"
DVD_CACHE = ROOT / "results/safety_dev/scenario_cache/dvd_v3"
TS4_OUT = ROOT / "results/safety_dev/testset_v4_main/runs"
# Bulky per-decision diagnostics omitted by --compact-traces; decisions,
# commands, branches, V20 levels and certificates are always kept.
COMPACT_DROP = ("risk_monitor", "track_persistence", "motion_axis_geometry", "policy_search",
                "escape_search", "v11_track_admission", "v11_track_persistence")
VARIANTS = {
    "off": (1, None, {}),
    "v16": (16, "safety_v16", {}),
    "v20": (20, "safety_v20", {}),
    "v20_nominal": (20, "safety_v20", {"allowance_tables": "none"}),
    "v20_oocv16": (20, "safety_v20", {"out_of_contract": "v16"}),
    # Revision 1 (planning/SAFETY_V20_PLAN.md, section 13).
    "v20_r1": (20, "safety_v20", dict(importlib.import_module("safety_v20").REVISION1_OPTIONS)),
    # Revision 2 (planning/SAFETY_V20_PLAN.md, section 14).
    "v20_r2": (20, "safety_v20", dict(importlib.import_module("safety_v20").REVISION2_OPTIONS)),
    # Revision 3 (planning/SAFETY_V20_PLAN.md, section 15).
    "v20_r3": (20, "safety_v20", dict(importlib.import_module("safety_v20").REVISION3_OPTIONS)),
    # Emergency-stop protocol (planning/SAFETY_ESTOP_PROTOCOL.md).
    "v16_estop": (16, "safety_estop", {}),
    "v17_estop": (17, "safety_estop", {}),
    # Boundary-and-targets trigger (planning/SAFETY_ESTOP_PROTOCOL.md section 8).
    "v16_estop_bt": (16, "safety_estop", dict(importlib.import_module("safety_estop").BOUNDARY_TARGET_OPTIONS)),
    "v17_estop_bt": (17, "safety_estop", dict(importlib.import_module("safety_estop").BOUNDARY_TARGET_OPTIONS)),
    # V21 phase (safety_v21.py): braking fallback where V16 finds no escape.
    "v21": (21, "safety_v21", {}),
    "v21_lc": (21, "safety_v21", {"fallback_branches": ("no escape", "last certificate")}),
    "v21_estop_bt": (21, "safety_v21", dict(importlib.import_module("safety_estop").BOUNDARY_TARGET_OPTIONS)),
    "v21_lc_estop_bt": (21, "safety_v21", dict(importlib.import_module("safety_estop").BOUNDARY_TARGET_OPTIONS,
                                               fallback_branches=("no escape", "last certificate"))),
    # R2, graded buffers: alone, and with the braking fallback (R1 + R2).
    "v21_r2": (21, "safety_v21", {"braking_fallback": False, "graded_buffers": True}),
    "v21_r12": (21, "safety_v21", {"graded_buffers": True}),
    "v21_r2_estop_bt": (21, "safety_v21", dict(importlib.import_module("safety_estop").BOUNDARY_TARGET_OPTIONS,
                                               braking_fallback=False, graded_buffers=True)),
    "v21_r12_estop_bt": (21, "safety_v21", dict(importlib.import_module("safety_estop").BOUNDARY_TARGET_OPTIONS,
                                                graded_buffers=True)),
    # Parity: every V21 option off must reproduce V16.
    "v21_off": (21, "safety_v21", {"braking_fallback": False}),
    # R3 (scored boundary limits) and R4 (track-footprint exclusion), alone and together.
    "v21_r3": (21, "safety_v21", {"braking_fallback": False, "scored_boundary": True}),
    "v21_r4": (21, "safety_v21", {"braking_fallback": False, "track_trail": True}),
    "v21_r34": (21, "safety_v21", {"braking_fallback": False, "scored_boundary": True, "track_trail": True}),
    "v21_r34_estop_bt": (21, "safety_v21", dict(importlib.import_module("safety_estop").BOUNDARY_TARGET_OPTIONS,
                                                braking_fallback=False, scored_boundary=True, track_trail=True)),
    # Design B: parity (must equal v16_estop_bt), E1 (detector on the scored collision hull), R3 + E1.
    "v21_estop_off": (21, "safety_v21", dict(importlib.import_module("safety_estop").BOUNDARY_TARGET_OPTIONS,
                                             braking_fallback=False)),
    "v21_e1_estop_bt": (21, "safety_v21", dict(importlib.import_module("safety_estop").BOUNDARY_TARGET_OPTIONS,
                                               braking_fallback=False)),
    "v21_r3_e1_estop_bt": (21, "safety_v21", dict(importlib.import_module("safety_estop").BOUNDARY_TARGET_OPTIONS,
                                                  braking_fallback=False, scored_boundary=True)),
    # E1 with the remembered static points in the trigger (objective of 2026-10-25), W = 1.5 / 2.5 / 3.5 s.
    "v21_e1s15_estop": (21, "safety_v21", {"estop_window_s": 1.5, "use_static_points": True, "braking_fallback": False}),
    "v21_e1s25_estop": (21, "safety_v21", {"estop_window_s": 2.5, "use_static_points": True, "braking_fallback": False}),
    "v21_e1s35_estop": (21, "safety_v21", {"estop_window_s": 3.5, "use_static_points": True, "braking_fallback": False}),
}
# Filter class per mode where it is not SafetyFilterV<version> in the module.
VARIANT_CLASSES = {"v16_estop": "SafetyFilterV16E", "v17_estop": "SafetyFilterV17E",
                   "v16_estop_bt": "SafetyFilterV16E", "v17_estop_bt": "SafetyFilterV17E",
                   "v21_estop_bt": "SafetyFilterV21E", "v21_lc_estop_bt": "SafetyFilterV21E",
                   "v21_r2_estop_bt": "SafetyFilterV21E", "v21_r12_estop_bt": "SafetyFilterV21E",
                   "v21_r34_estop_bt": "SafetyFilterV21E", "v21_estop_off": "SafetyFilterV21E",
                   "v21_e1_estop_bt": "SafetyFilterV21EC", "v21_r3_e1_estop_bt": "SafetyFilterV21EC",
                   "v21_e1s15_estop": "SafetyFilterV21EC", "v21_e1s25_estop": "SafetyFilterV21EC",
                   "v21_e1s35_estop": "SafetyFilterV21EC"}
# Modes evaluated under the emergency-stop protocol's environment rules.
PROTOCOL_MODES = frozenset({"v16_estop", "v17_estop", "v16_estop_bt", "v17_estop_bt",
                            "v21_estop_bt", "v21_lc_estop_bt", "v21_r2_estop_bt", "v21_r12_estop_bt",
                            "v21_r34_estop_bt", "v21_estop_off", "v21_e1_estop_bt", "v21_r3_e1_estop_bt",
                            "v21_e1s15_estop", "v21_e1s25_estop", "v21_e1s35_estop"})


def load_ts4_cache():
    """Trusted saved test-set-v4 cache; never call a scenario generator."""
    directory = ROOT / "results/test_set/v4"
    paths = [directory / name for name in ("set_v4.0.pkl", "definition.json", "definition.csv")]
    raw, definition_raw, csv_raw = [path.read_bytes() for path in paths]
    kept, metadata = pickle.loads(raw)
    definition = json.loads(definition_raw)
    rows = list(csv.DictReader(io.StringIO(csv_raw.decode("utf-8"))))
    if (metadata != definition or definition.get("version") != "4.0"
            or len(kept) != definition.get("episodes") or len(rows) != len(kept)):
        raise ValueError("TS4 cache/definition metadata or count mismatch")
    entries = {item["test_id"]: item for item in kept}
    definitions = {item["test_id"]: item for item in rows}
    if (len(entries) != len(kept) or len(definitions) != len(rows)
            or entries.keys() != definitions.keys()):
        raise ValueError("TS4 cache/definition duplicate or mismatched IDs")
    digests = {}
    for case, entry in entries.items():
        row = definitions[case]
        digest = entry["built"].digest()
        if int(entry["episode_seed"]) != int(row["episode_seed"]) or digest != row["digest"]:
            raise ValueError(f"TS4 cache/definition seed or geometry mismatch: {case}")
        digests[case] = digest
    manifest_digest = sha(json.dumps(digests, sort_keys=True, separators=(",", ":")).encode())
    if manifest_digest != definition.get("manifest_digest"):
        raise ValueError("TS4 cache manifest digest mismatch")
    return ({"TS4:" + case: entry for case, entry in entries.items()},
            {path.relative_to(ROOT).as_posix(): sha(data)
             for path, data in zip(paths, (raw, definition_raw, csv_raw))})


def load_scenes(cases):
    ts4 = [row for row in cases if row["case"].startswith("TS4:")]
    other = [row for row in cases if not row["case"].startswith("TS4:")]
    if ts4 and other:
        raise ValueError("Do not mix test set v4 with development cohorts in one tag")
    if other:
        dvd = [row for row in other if row["case"].startswith("DVD:")]
        rest = [row for row in other if not row["case"].startswith("DVD:")]
        if dvd and any(not row["case"].startswith("DV3:") for row in rest):
            raise ValueError("DVD scenes share a tag only with DV3 (the overlay changes other panel draws)")
        scenes, provenance = pilot.load_scenes(rest) if rest else ({}, {})
        for row in dvd:
            if row.get("dataset") != "dvd":
                raise ValueError(f"Selection dataset/namespace mismatch: {row['case']}")
            path = DVD_CACHE / f"{row['case'].split(':', 1)[1]}.pkl"
            raw = path.read_bytes()
            built = pickle.loads(raw)
            if built.digest() != row["scenario_sha256"]:
                raise ValueError(f"Selection geometry mismatch: {row['case']}")
            scenes[row["case"]] = built
            provenance[path.relative_to(ROOT).as_posix()] = sha(raw)
        return scenes, provenance
    entries, provenance = load_ts4_cache()
    scenes = {}
    for row in ts4:
        if row.get("dataset") != "ts4":
            raise ValueError(f"Selection dataset/namespace mismatch: {row['case']}")
        entry = entries.get(row["case"])
        if entry is None:
            raise ValueError(f"No canonical cached scene for {row['case']}")
        built = entry["built"]
        if int(entry["episode_seed"]) != int(row["seed"]) or built.digest() != row["scenario_sha256"]:
            raise ValueError(f"Selection seed/geometry mismatch: {row['case']}")
        scenes[row["case"]] = built
    return scenes, provenance


def environment_freeze():
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "packages": dict(packages)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--modes", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cases", default="", help="Comma-separated canonical prefixed IDs")
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument("--compact-traces", action="store_true",
                        help="Omit bulky nested diagnostics from trace records (storage only)")
    parser.add_argument("--gzip-traces", action="store_true", help="Write traces as .jsonl.gz (storage only)")
    args = parser.parse_args()
    if args.snapshots and args.compact_traces:
        parser.error("Snapshots and compact traces are exclusive")
    modes = args.modes.split(",")
    if (Path(args.tag).name != args.tag or not args.tag or len(set(modes)) != len(modes)
            or not set(modes) <= set(VARIANTS)):
        parser.error("Use a unique simple tag and registered modes")
    selection_raw = args.selection.read_bytes()
    selection = json.loads(selection_raw)
    requested = set(filter(None, args.cases.split(",")))
    cases = [r for r in selection["cases"] if not requested or r["case"] in requested]
    if not cases or len({r["case"] for r in cases}) != len(cases) or requested - {r["case"] for r in cases}:
        parser.error("Empty, duplicate or unknown scenario selection")
    primary = [r["case"].startswith("TS4:") for r in cases]
    if any(primary) and not all(primary):
        parser.error("Do not mix test set v4 with development cohorts")
    out = (TS4_OUT if all(primary) else DEVELOPMENT_OUT) / args.tag
    if out.exists():
        parser.error("Existing tag; no automatic retries or result reuse")

    import torch
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits, threadpool_info
    except ImportError:
        limits, threadpool_info = None, (lambda: [])
    else:
        limits = threadpool_limits(limits=1)
    import constants as cfg
    import curriculum
    import train_formulation as tf
    from common import load_model, run_episode
    from env import ASVLidarEnv
    overlay = None
    if any(r["case"].startswith("DVD:") for r in cases):
        import formulation_v3
        formulation_v3.apply(cfg)          # before any environment: it draws the DVD panels at reset
        overlay = formulation_v3.ID
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    config = json.loads(MODEL.with_name("config.json").read_text())
    for name, value in (config.get("constant_overrides") or {}).items():
        setattr(cfg, name, value)
    if sha(MODEL.read_bytes()) != EXPECTED_MODEL:
        raise ValueError("Checkpoint identity changed")
    classes, options = {}, {}
    for mode in modes:
        version, module, defaults = VARIANTS[mode]
        classes[mode] = None if module is None else getattr(
            importlib.import_module(module), VARIANT_CLASSES.get(mode, f"SafetyFilterV{version}"))
        options[mode] = dict(defaults) if module else {}
        if module:
            classes[mode](**options[mode])  # Validate configuration without an episode.
    capture = importlib.import_module("trace_snapshot") if args.snapshots else None
    scenes, cache_hashes = load_scenes(cases)
    for name, expected in selection.get("input_sha256", {}).items():
        if name.startswith("results/test_set/") and cache_hashes.get(name) not in (None, expected):
            raise ValueError(f"Frozen selection input changed: {name}")
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "bluefin").rglob("*.py"))
    paths += [Path(__file__), Path(pilot.__file__), ROOT / "tools/tiers/common.py"]
    if capture:
        paths.append(Path(capture.__file__))
    sources = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(set(paths))}
    source_hashes = {name: sha(raw) for name, raw in sources.items()}
    out.mkdir(parents=True)
    with zipfile.ZipFile(out / "evaluated_sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sources.items():
            archive.writestr(name, raw)
    manifest = dict(
        prepared_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        authorization=("V21 phase: development on the validation set and the earlier development "
                       "cohorts; test set v4 only on request; no tuning on test-set outcomes"),
        overlay=overlay,
        selection_source=str(args.selection.relative_to(ROOT) if args.selection.is_absolute() else args.selection),
        selection_sha256=sha(selection_raw), cases=cases, modes=modes,
        planned_runs=len(cases) * len(modes), process_id=os.getpid(), processes=1,
        filter_classes={m: None if classes[m] is None else classes[m].__module__ + "." + classes[m].__name__
                        for m in modes},
        constructor_options=options, filter_installation="Explicit instance immediately after native reset",
        snapshots=args.snapshots, compact_traces=args.compact_traces, gzip_traces=args.gzip_traces,
        compact_trace_omitted_keys=list(COMPACT_DROP) if args.compact_traces else [], checkpoint=str(MODEL.relative_to(ROOT)), checkpoint_sha256=EXPECTED_MODEL,
        config_sha256=sha(MODEL.with_name("config.json").read_bytes()), source_sha256=source_hashes,
        cache_sha256=cache_hashes, threadpools=threadpool_info(), torch_threads=torch.get_num_threads(),
        native_thread_environment={k: os.environ[k] for k in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        constants={k: repr(v) for k, v in vars(cfg).items() if k.isupper()}, low_speed_start_frac=0.0,
        evaluation_role=("primary_test_set_v4" if all(primary) else selection.get("evaluation_role", "development")),
        selected_scenarios=len(cases), selection_scenarios=len(selection["cases"]),
        scope=selection.get("scope", "Development only; no claimed population estimate or formal safety guarantee"),
        runtime_environment=environment_freeze())
    write_json(out / "manifest.json", manifest)
    (out / "attempts").mkdir()
    (out / "traces").mkdir()
    model = load_model(MODEL)

    class TraceEnv(ASVLidarEnv):
        def reset(self, **reset_kwargs):
            result = super().reset(**reset_kwargs)
            self._safety_v2 = classes[self.mode](**options[self.mode]) if self.estop_enabled else None
            if self.mode in PROTOCOL_MODES:
                estop_protocol.install(self)
            self.last_observation = result[0]
            return result

        def step(self, action):
            before = [self.asv_x, self.asv_y, self.asv_h, self.u_body]
            diagnostic_before = capture.capture_before(self) if capture else None
            observation = self.last_observation
            start = time.perf_counter()
            result = super().step(action)
            if self.mode in PROTOCOL_MODES:
                result = estop_protocol.after_step(self, result, self.steps_logged + 1)
            seconds = time.perf_counter() - start
            filt = self._safety_v2
            if self.estop_enabled and type(filt) is not classes[self.mode]:
                raise AssertionError("Unexpected filter class")
            details = getattr(filt, "last", {})
            if args.compact_traces:
                details = {k: v for k, v in details.items() if k not in COMPACT_DROP}
            self.why_counts[details.get("why", "idle")] += 1
            if "v20_level" in details:
                self.level_counts[details["v20_level"]] += 1
            self.steps_logged += 1
            record = dict(step=self.steps_logged, step_seconds=seconds, observation=observation,
                          pre_state=before, post_state=[self.asv_x, self.asv_y, self.asv_h, self.u_body],
                          policy_action=action, rudder_command=self.rudder / 100., signed_rpm_command=self.rpm,
                          changed=bool(getattr(self, "safety_v2_changed", False)),
                          brake=bool(getattr(self, "_v2_brake", False)),
                          filter=details, plan=getattr(filt, "plan", None))
            if self.mode in PROTOCOL_MODES:
                record["estop_protocol"] = dict(self.estop_protocol,
                                                committed=bool(getattr(self, "emergency_stop_committed", False)))
            if capture:
                record["diagnostic_before"] = diagnostic_before
                record["diagnostic_decision"] = capture.capture_decision(self)
            self.trace.write(json.dumps(jsonable(record), allow_nan=False) + "\n")
            self.last_observation = result[0]
            return result

    count, results = 0, []
    started = time.perf_counter()
    for index, case in enumerate(cases):
        for mode in modes[index % len(modes):] + modes[:index % len(modes)]:
            if (out / "STOP").exists() or (out.parent / "STOP").exists():
                print("STOP marker; completed records preserved", flush=True)
                return
            for name, expected in source_hashes.items():
                if sha((ROOT / name).read_bytes()) != expected:
                    raise RuntimeError(f"Frozen source changed: {name}")
            count += 1
            identity = dict(attempt=count, case=case["case"], mode=mode, seed=int(case["seed"]))
            write_json(out / "attempts" / f"{count:03d}.json", identity)
            cfg.SAFETY_VERSION = VARIANTS[mode][0]
            env = TraceEnv(render_mode=None, emergency_stop=mode != "off", low_speed_start_frac=0.0)
            env.mode, env.why_counts, env.level_counts, env.steps_logged = mode, Counter(), Counter(), 0
            episode_start = time.perf_counter()
            try:
                name = f"{count:03d}_{mode}.jsonl" + (".gz" if args.gzip_traces else "")
                opener = (gzip.open(out / "traces" / name, "xt", encoding="utf-8", newline="\n")
                          if args.gzip_traces else (out / "traces" / name).open("x", encoding="utf-8", newline="\n"))
                with opener as trace:
                    env.trace = trace
                    result = run_episode(env, copy.deepcopy(scenes[case["case"]]), int(case["seed"]), "model", model)
                    if mode in PROTOCOL_MODES:
                        state = env.estop_protocol
                        result["unprotocol_outcome"] = result["outcome"]
                        result["outcome"] = state["outcome"] or result["outcome"]
                        result["estop_commit_step"] = state["commit_step"]
                        result["estop_commit_speed_mps"] = state["commit_speed_mps"]
                    trace.flush()
                    if not args.gzip_traces:
                        os.fsync(trace.fileno())
                row = dict(identity, dataset=case["dataset"], stratum=case.get("selection_stratum", "explicit"),
                           scenario_sha256=case["scenario_sha256"],
                           elapsed_s=time.perf_counter() - episode_start, why_counts=dict(env.why_counts),
                           v20_level_counts=dict(env.level_counts), **result)
                write_json(out / "attempts" / f"{count:03d}_result.json", row)
                results.append(row)
                with (out / "episodes.csv").open("a", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if count == 1:
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    os.fsync(stream.fileno())
                print(f"{count}/{manifest['planned_runs']} {mode} {case['case']}: {row['outcome']} "
                      f"({row['elapsed_s']:.1f}s, interventions={row['safety_v2_steps']})", flush=True)
            finally:
                env.close()
    drift = [name for name, h in source_hashes.items() if sha((ROOT / name).read_bytes()) != h]
    write_json(out / "completion.json", dict(
        completed_runs=count, elapsed_s=time.perf_counter() - started, source_drift=drift,
        checkpoint_unchanged=sha(MODEL.read_bytes()) == EXPECTED_MODEL,
        selection_unchanged=sha(args.selection.read_bytes()) == sha(selection_raw)))
    print(json.dumps({m: dict(Counter(r["outcome"] for r in results if r["mode"] == m)) for m in modes}), flush=True)
    if limits is not None:
        limits.restore_original_limits()


if __name__ == "__main__":
    main()
