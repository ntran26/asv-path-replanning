"""Resumable, one-process safety evaluation across the complete named suites.

Select/freeze the candidate on development data before running held-out sets.
Run at most TWO instances concurrently while PPO training continues. Each
instance evaluates serially with one Torch thread; it never spawns workers.

Examples, from the project root:
    python -B tools/diagnostics/safety/suite_eval.py --suites dev --modes off,v5 --tag dev_final
    python -B tools/diagnostics/safety/suite_eval.py --suites frozen --include-tier-a --modes off,v5 --tag frozen_final
    python -B tools/diagnostics/safety/suite_eval.py --suites field,fv --modes off,v5 --tag field_final

Repeat the exact command with --resume after interruption. The immutable run
manifest rejects changed code, module settings, checkpoint, scenario digests,
or reset seeds. JSONL is the durable source of completed cases; CSV and summary
are reviewable projections. Existing historical result directories are unused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version as distribution_version
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

CACHE_SCHEMA = 1
RUN_SCHEMA = 1
SUITE_ALIASES = {
    "dev": ("dev_field", "dev_legacy", "dev_width"),
    "frozen": ("frozen_b", "frozen_r"),
    "field": ("field_deployment",),
    "fv": ("field_validation",),
}
CSV_FIELDS = ("mode", "suite", "case", "test_id", "seed", "scenario_sha256", "outcome", "steps",
              "safety_v2_steps", "safety_v2_brake_steps", "estops", "seconds", "return",
              "collided", "collided_target", "rms_cte", "min_target_range")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expand_suites(names, include_tier_a=False):
    names = ["dev", "frozen", "field", "fv"] if names == ["all"] else names
    allowed = {item for group in SUITE_ALIASES.values() for item in group} | {"frozen_a"}
    result = []
    for name in names:
        expanded = SUITE_ALIASES.get(name, (name,))
        if not set(expanded) <= allowed:
            raise ValueError(f"Unknown suite {name!r}; use dev,frozen,field,fv,all or explicit component names")
        result.extend(item for item in expanded if item not in result)
    if include_tier_a and "frozen_a" not in result:
        result.append("frozen_a")
    if not result:
        raise ValueError("Select at least one suite")
    return result


def scenario_cache_provenance():
    import constants as cfg
    from prediction_audit import development_source_paths
    paths = set(development_source_paths()) | {ROOT / name for name in
             ("src/suite.py", "src/train_formulation.py", "src/curriculum.py", "tools/tiers/common.py", "src/env.py")}
    return {"schema": CACHE_SCHEMA,
            "builder_implementation_sha256": hashlib.sha256(inspect.getsource(build_jobs).replace("\r\n", "\n").encode()).hexdigest(),
            "source_sha256_lf": {p.relative_to(ROOT).as_posix(): hashlib.sha256(
                p.read_text(encoding="utf-8-sig").encode()).hexdigest() for p in sorted(paths)},
            "constants": {k: repr(v) for k, v in vars(cfg).items()
                          if k.isupper() and not k.startswith("SAFETY_")}}


def cached_component(name, provenance, builder):
    directory = ROOT / "results" / "safety_dev" / "suite_cache" / digest_json(provenance)[:20]
    path = directory / f"{name}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            saved = pickle.load(handle)
        if saved["provenance"] != provenance or saved["component"] != name:
            raise ValueError(f"Scenario cache provenance mismatch: {path}")
        return saved["payload"]
    print(f"Building deterministic suite component {name}", flush=True)
    payload = builder()
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"{name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        pickle.dump({"provenance": provenance, "component": name, "payload": payload}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return payload


def build_jobs(names):
    """Original suite builders and reset seeds; no policy episodes are run here."""
    import suite
    import field_training
    import paper2_set
    from common import development_set, head_on_width_set
    from env import ASVLidarEnv
    from prediction_audit import load_development_cases
    provenance = scenario_cache_provenance()
    jobs, details = [], {}

    def add(name, built, seed, case=None, test_id="", obstacles=None):
        jobs.append({"suite": name, "case": case or built.case_id, "test_id": test_id,
                     "seed": int(seed), "obstacles": obstacles, "built": built})

    headline = None
    if set(names) & {"frozen_b", "frozen_r"}:
        headline, shortfall = cached_component("frozen_b", provenance, suite.build_tier_b)
        details["frozen_b_shortfall"] = shortfall
    for name in names:
        if name == "dev_field":
            for i, built in load_development_cases():
                add(name, built, 900_120 + i)
        elif name == "dev_legacy":
            scenes = cached_component(name, provenance, lambda: development_set(20))
            for i, built in enumerate(scenes):
                add(name, built, 900_000 + i, case=f"DEV-{i:03d}")
        elif name == "dev_width":
            scenes = cached_component(name, provenance, head_on_width_set)
            for i, built in enumerate(scenes):
                add(name, built, 700_000 + i, case=f"HW-{i:03d}", obstacles=0)
        elif name == "frozen_a":
            scenes = cached_component(name, provenance, suite.build_tier_a)
            details["frozen_a_defined"] = len(suite.tier_a())
            details["frozen_a_shortfall"] = suite.tier_a_shortfall(scenes)
            for i, built in enumerate(scenes):
                add(name, built, 300_000 + i)
        elif name == "frozen_b":
            for i, built in enumerate(headline):
                add(name, built, 400_000 + i, test_id=suite.test_id(built.case_id))
        elif name == "frozen_r":
            for built, twin, _ in suite.robustness_variants(headline):
                add(name, built, 400_000 + twin, test_id=suite.test_id(built.case_id))
        elif name == "field_deployment":
            def make_field():
                env = ASVLidarEnv(render_mode=None, emergency_stop=False)
                try:
                    return paper2_set.build(env)
                finally:
                    env.close()
            records, shortfall = cached_component(name, provenance, make_field)
            details["field_deployment_shortfall"] = shortfall
            for record in records:
                add(name, record["built"], record["episode_seed"], case=record["test_id"],
                    test_id=record["test_id"])
        elif name == "field_validation":
            scenes = cached_component(name, provenance, field_training.validation_set)
            # This previously unused suite had no evaluation runner/reset-seed
            # convention. Freeze an explicit disjoint sequence for both modes.
            for i, built in enumerate(scenes):
                add(name, built, 950_000 + i)
            details["field_validation_reset_seed_policy"] = "new fixed convention: 950000 + original index"
    keys = [(j["suite"], j["case"]) for j in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate suite/case identity")
    details["counts"] = {name: sum(j["suite"] == name for j in jobs) for name in names}
    details["scenario_cache_digest"] = digest_json(provenance)[:20]
    return jobs, details


def settings_snapshot(model, names, modes, references=None):
    import constants as cfg
    modules = {f"v{version}": importlib.import_module(f"safety_v{version}")
               for version in range(2, max([int(m[1:]) for m in modes if m != "off"] + [3]) + 1)}
    source_paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "bluefin").glob("*.py"))
    source_paths += [ROOT / "tools/tiers/common.py", Path(__file__),
                     ROOT / "tools/diagnostics/safety/prediction_audit.py"]
    config = model.with_name("config.json")
    versions = {"python": sys.version, "platform": sys.platform}
    for package in ("numpy", "torch", "stable-baselines3", "sb3-contrib", "gymnasium"):
        try:
            versions[package] = distribution_version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"schema": RUN_SCHEMA, "suites": names, "modes": modes,
            "checkpoint": str(model.resolve()), "checkpoint_sha256": sha(model),
            "config_sha256": sha(config) if config.exists() else None,
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
            "effective_safety_constants": {name: {k: repr(v) for k, v in vars(module).items() if k.isupper()}
                                           for name, module in modules.items()},
            "effective_constants": {k: repr(v) for k, v in vars(cfg).items()
                                    if k.isupper() and not k.startswith("SAFETY_")},
            "reference_manifests": {key: {"path": str(path.resolve()), "sha256": sha(path)}
                                    for key, path in (references or {}).items()},
            "runtime_versions": versions, "threads": 1, "processes": 1}


def check_suite_coverage(details):
    expected = {"dev_field": 150, "dev_legacy": 120, "dev_width": 100,
                "frozen_b": 800, "frozen_r": 900, "field_deployment": 630, "field_validation": 155}
    for name, count in details["counts"].items():
        if name in expected and count != expected[name]:
            raise ValueError(f"Incomplete suite {name}: generated {count}, expected {expected[name]}; no episodes started")
    if "frozen_a" in details["counts"]:
        realised = details["counts"]["frozen_a"]
        missing = len(details["frozen_a_shortfall"])
        if realised + missing != details["frozen_a_defined"]:
            raise ValueError("Tier A realisations and declared shortfalls do not cover its definitions")


def check_reference_cases(cases, references):
    result = {}
    for kind, path in references.items():
        reference = json.loads(path.read_text(encoding="utf-8"))
        expected = reference["case_digests"]
        names = {"frozen_b", "frozen_r"} if kind == "frozen" else {"field_deployment"}
        selected = [case for case in cases if case["suite"] in names]
        if not selected:
            raise ValueError(f"Reference {kind} supplied without its suite")
        for case in selected:
            if expected.get(case["case"]) != case["scenario_sha256"]:
                raise ValueError(f"Historical {kind} scenario digest mismatch: {case['case']}; no episodes started")
        result[kind] = {"matched_cases": len(selected), "reference_cases": len(expected),
                        "manifest_digest": reference.get("manifest_digest")}
    return result


def create_or_check_manifest(directory, manifest, resume):
    path = directory / "manifest.json"
    if resume:
        if not path.exists():
            raise ValueError(f"No manifest to resume: {path}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Resume rejected: source/config/model/scenario/seed manifest differs; use a new tag")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing run: {directory}; use --resume or a new tag")
        directory.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")


def load_completed(path, expected):
    """Accept complete JSONL records only; an interrupted final write is retried."""
    rows, keys = [], set()
    if not path.exists():
        return rows
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    valid_bytes = 0
    for i, raw in enumerate(lines):
        if not raw.endswith(b"\n") and i == len(lines) - 1:
            # Preserve the partial tail as evidence before truncating the live
            # append log to its last fully committed record.
            archive = path.with_name(f"incomplete_tail_{uuid.uuid4().hex[:12]}.bin")
            archive.write_bytes(raw)
            with path.open("r+b") as handle:
                handle.truncate(valid_bytes)
            break
        row = json.loads(raw)
        key = (row["mode"], row["suite"], row["case"])
        if key not in expected or key in keys:
            raise ValueError(f"Unknown or duplicate completed case {key}")
        job = expected[key]
        if row["seed"] != job["seed"] or row["scenario_sha256"] != job["scenario_sha256"]:
            raise ValueError(f"Completed-case seed/digest mismatch {key}")
        if row.get("outcome") not in ("goal", "timeout", "collision:boundary", "collision:obstacle", "collision:target"):
            raise ValueError(f"Unfinished or invalid completed-case outcome {key}")
        rows.append(row)
        keys.add(key)
        valid_bytes += len(raw)
    return rows


def write_csv_projection(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(directory, rows, total):
    from collections import Counter
    counts = Counter((r["mode"], r["suite"], r["outcome"]) for r in rows)
    summary = {"completed": len(rows), "expected": total, "complete": len(rows) == total,
               "outcomes": [{"mode": mode, "suite": name, "outcome": outcome, "n": n}
                            for (mode, name, outcome), n in sorted(counts.items())]}
    (directory / "progress.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def pid_is_alive(pid):
    """Read process existence; never signal or terminate a Windows process."""
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE only
        if not handle:
            return ctypes.get_last_error() == 5  # access denied is not evidence of death
        try:
            return kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 258
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path, resume):
    if path.exists() and resume:
        pid = int(path.read_text(encoding="utf-8").strip())
        if pid_is_alive(pid):
            raise RuntimeError(f"This run is still active in process {pid}; refusing concurrent writes")
        # The explicitly resumed run's own lock is removed only after its PID
        # is proved absent. No processes are stopped and no results are removed.
        path.unlink()
    with path.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", default="dev", help="comma-separated dev,frozen,field,fv; all includes all four")
    parser.add_argument("--include-tier-a", action="store_true", help="also evaluate all realisable named frozen cases")
    parser.add_argument("--modes", default="off,v5", help="comma-separated off,v2,v3,v4,v5")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-frozen", type=Path, help="require generated B/R cases to match this historical manifest")
    parser.add_argument("--reference-field", type=Path, help="require deployment-field cases to match this historical manifest")
    parser.add_argument("--build-only", action="store_true", help="create/validate manifests and caches without policy episodes")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in (".", "..") or "\\" in args.tag:
        parser.error("--tag must be a plain filename stem")
    try:
        names = expand_suites(args.suites.split(","), args.include_tier_a)
    except ValueError as exc:
        parser.error(str(exc))
    modes = args.modes.split(",")
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= {"off", "v2", "v3", "v4", "v5"}:
        parser.error("Use unique modes from off,v2,v3,v4,v5")
    output = ROOT / "results" / "safety_dev" / "suites" / args.tag
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error(f"Existing run {output}; use --resume or a new tag")
    import torch
    torch.set_num_threads(1)
    import constants as cfg
    import curriculum
    import train_formulation as tf
    from common import load_model, run_episode
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    config = args.model.with_name("config.json")
    if config.exists():
        for key, value in (json.loads(config.read_text()).get("constant_overrides") or {}).items():
            setattr(cfg, key, value)
    from prediction_audit import runtime_overrides
    runtime_overrides()
    references = {key: path for key, path in (("frozen", args.reference_frozen), ("field", args.reference_field)) if path}
    settings = settings_snapshot(args.model, names, modes, references)
    # Reject changed source/settings before potentially expensive regeneration.
    if args.resume and (output / "manifest.json").exists():
        old = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if old.get("settings") != settings:
            parser.error("Resume rejected: source/config/model/settings changed; use a new tag")
    jobs, details = build_jobs(names)
    check_suite_coverage(details)
    cases = [{k: v for k, v in job.items() if k != "built"} | {"scenario_sha256": job["built"].digest()}
             for job in jobs]
    details["historical_manifest_checks"] = check_reference_cases(cases, references)
    manifest = {"settings": settings, "suite_details": details, "cases": cases,
                "scenario_seed_digest": digest_json(cases), "expected_episodes": len(cases) * len(modes)}
    create_or_check_manifest(output, manifest, args.resume)
    print(f"Suites {details['counts']}; {manifest['expected_episodes']} paired-mode episodes", flush=True)
    if args.build_only:
        return
    expected = {(mode, c["suite"], c["case"]): c for mode in modes for c in cases}
    # Exclusive live lock prevents accidental concurrent writes to one tag.
    lock = output / "running.lock"
    acquire_lock(lock, args.resume)
    rows = []
    try:
        rows = load_completed(output / "episodes.jsonl", expected)
        write_csv_projection(output / "episodes.csv", rows)
        done = {(r["mode"], r["suite"], r["case"]) for r in rows}
        model = load_model(args.model)
        from env import ASVLidarEnv
        with (output / "episodes.jsonl").open("a", encoding="utf-8") as journal, \
                (output / "episodes.csv").open("a", newline="", encoding="utf-8") as table:
            writer = csv.DictWriter(table, fieldnames=CSV_FIELDS, extrasaction="ignore")
            for mode in modes:
                cfg.SAFETY_VERSION = 1 if mode == "off" else int(mode[1:])
                for name in names:
                    pending = [j for j in jobs if j["suite"] == name and (mode, name, j["case"]) not in done]
                    if not pending:
                        continue
                    # A fresh env per component prevents the width diagnostic's
                    # obstacles=0 override leaking into subsequent suites.
                    env = ASVLidarEnv(render_mode=None, emergency_stop=mode != "off")
                    try:
                        for job in pending:
                            started = time.perf_counter()
                            try:
                                result = run_episode(env, job["built"], job["seed"], "model", model, job["obstacles"])
                            except Exception as exc:
                                with (output / "errors.jsonl").open("a", encoding="utf-8") as failures:
                                    failures.write(json.dumps({"mode": mode, "suite": name, "case": job["case"],
                                        "seed": job["seed"], "error": repr(exc), "traceback": traceback.format_exc()}) + "\n")
                                    failures.flush()
                                    os.fsync(failures.fileno())
                                raise  # fail visibly; an errored case is never counted or silently skipped
                            row = dict(result, mode=mode, suite=name, case=job["case"], test_id=job["test_id"],
                                       seed=job["seed"], scenario_sha256=job["built"].digest(),
                                       seconds=time.perf_counter() - started)
                            # Commit the full result first. On resume the CSV is
                            # reconstructed if the process died between writes.
                            journal.write(json.dumps(row, allow_nan=True) + "\n")
                            journal.flush()
                            os.fsync(journal.fileno())
                            writer.writerow(row)
                            table.flush()
                            rows.append(row)
                            done.add((mode, name, job["case"]))
                            write_summary(output, rows, len(expected))
                            print(f"{len(rows)}/{len(expected)} {mode} {name} {job['case']}: {row['outcome']}", flush=True)
                    finally:
                        env.close()
    finally:
        write_summary(output, rows, len(expected))
        lock.unlink()


if __name__ == "__main__":
    main()
