"""Fresh, bounded SAC/V8/V9 comparison on a preselected development cohort.

One process, one Torch thread; no result reuse, retries, or parameter tuning.
Selection is frozen before physics. Existing campaign ledgers are untouched.
Usage: python -B tools/diagnostics/safety/v9_paired_pilot.py
An output-directory STOP file stops before the next episode.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import zipfile

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/tiers")]
OUT = ROOT / "results/safety_dev/v9_paired_pilot"
MODEL = ROOT / "runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip"
MODES = ("off", "v8", "v9")
EXPECTED_MODEL = "993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(jsonable(value), handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_ts3_cache():
    """Read the trusted saved v3 cache; never call a scenario generator."""
    directory = ROOT / "results/test_set/v3"
    paths = [directory / name for name in ("set_v3.0.pkl", "definition.json", "definition.csv")]
    raw, definition_raw, csv_raw = [path.read_bytes() for path in paths]
    kept, metadata = pickle.loads(raw)
    definition = json.loads(definition_raw)
    rows = list(csv.DictReader(io.StringIO(csv_raw.decode("utf-8"))))
    if (metadata != definition or definition.get("version") != "3.0"
            or len(kept) != definition.get("episodes") or len(rows) != len(kept)):
        raise ValueError("TS3 cache/definition metadata or count mismatch")
    entries = {item["test_id"]: item for item in kept}
    definitions = {item["test_id"]: item for item in rows}
    if (len(entries) != len(kept) or len(definitions) != len(rows)
            or entries.keys() != definitions.keys()):
        raise ValueError("TS3 cache/definition duplicate or mismatched IDs")
    digests = {}
    for case, entry in entries.items():
        row = definitions[case]
        digest = entry["built"].digest()
        if int(entry["episode_seed"]) != int(row["episode_seed"]) or digest != row["digest"]:
            raise ValueError(f"TS3 cache/definition seed or geometry mismatch: {case}")
        digests[case] = digest
    manifest_digest = sha(json.dumps(digests, sort_keys=True, separators=(",", ":")).encode())
    if manifest_digest != definition.get("manifest_digest"):
        raise ValueError("TS3 cache manifest digest mismatch")
    return ({"TS3:" + case: entry for case, entry in entries.items()},
            {path.relative_to(ROOT).as_posix(): sha(data)
             for path, data in zip(paths, (raw, definition_raw, csv_raw))})


def load_scenes(cases):
    cases = list(cases)
    names = [row["case"] for row in cases]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate selection case")
    for row in cases:
        namespace = row["case"].split(":", 1)[0]
        if namespace not in ("TS2", "TS3", "DV3") or ":" not in row["case"]:
            raise ValueError(f"Unknown scenario namespace: {row['case']}")
        if row.get("dataset", namespace.lower()) != namespace.lower():
            raise ValueError(f"Selection dataset/namespace mismatch: {row['case']}")
    scenes, provenance, ts, dv3 = {}, {}, {}, {}
    if any(case.startswith(("TS2:", "DV3:")) for case in names):
        # Preserve the original TS2/DV3 cache selection and provenance.
        ts_path = ROOT / "results/test_set/v2/set_v2.0.pkl"
        raw = ts_path.read_bytes()
        kept, _ = pickle.loads(raw)
        ts = {"TS2:" + item["test_id"]: item for item in kept}
        inventory = json.loads((ROOT / "results/safety_dev/suites/inventory_v5/manifest.json").read_text())
        dv3 = {"DV3:" + item["case"]: item for item in inventory["cases"]
               if item["suite"] == "dev_field"}
        provenance[ts_path.relative_to(ROOT).as_posix()] = sha(raw)
    if any(case.startswith("TS3:") for case in names):
        ts3, hashes = load_ts3_cache()
        ts.update(ts3)
        provenance.update(hashes)
    for row in cases:
        case = row["case"]
        if case in ts:
            entry = ts[case]
            built, seed = entry["built"], entry["episode_seed"]
        else:
            if case not in dv3:
                raise ValueError(f"No canonical cached scene for {case}; refusing regeneration")
            entry = dv3[case]
            seed = entry["seed"]
            matches = []
            for path in sorted((ROOT / "results/safety_dev/scenario_cache").glob(f"*/{entry['case']}.pkl")):
                data = path.read_bytes()
                candidate = pickle.loads(data)
                if candidate.digest() == entry["scenario_sha256"]:
                    matches.append((path, data, candidate))
            if not matches:
                raise ValueError(f"No canonical cached scene for {case}; refusing regeneration")
            path, data, built = matches[0]
            provenance[path.relative_to(ROOT).as_posix()] = sha(data)
        if int(seed) != int(row["seed"]) or built.digest() != row["scenario_sha256"]:
            raise ValueError(f"Selection seed/geometry mismatch: {case}")
        scenes[case] = built
    return scenes, provenance


def main():
    import torch
    torch.set_num_threads(1)
    import constants as cfg
    import curriculum
    import train_formulation as tf
    from common import load_model, run_episode
    from env import ASVLidarEnv

    selection_raw = (OUT / "selection.json").read_bytes()
    selection = json.loads(selection_raw)
    cases = selection["cases"]
    if len(cases) != 32 or len({row["case"] for row in cases}) != 32:
        raise ValueError("This campaign is frozen to32 unique cases /96 runs")
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Campaign already prepared/started; no automatic retries")
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    model_cfg = json.loads(MODEL.with_name("config.json").read_text())
    for name, value in (model_cfg.get("constant_overrides") or {}).items():
        setattr(cfg, name, value)
    if sha(MODEL.read_bytes()) != EXPECTED_MODEL:
        raise ValueError("Unexpected checkpoint")
    scenes, cache_hashes = load_scenes(cases)
    modules = [importlib.import_module(f"safety_v{i}") for i in range(2, 10)]
    v9 = modules[-1]
    if (v9.TARGET_TURN_RATE_DEG_S != 0.0 or not v9.REQUIRE_CURRENT_PLAN
            or not v9.PREFER_POLICY_MARGIN):
        raise ValueError("Evaluate fixed default V9, with both guards and no turn ensemble")
    source_paths = sorted(set(
        list((ROOT / "src").rglob("*.py")) + list((ROOT / "bluefin").rglob("*.py"))
        + [Path(__file__), ROOT / "tools/tiers/common.py"]))
    sources = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in source_paths}
    source_hashes = {name: sha(raw) for name, raw in sources.items()}
    with zipfile.ZipFile(OUT / "evaluated_sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sources.items():
            archive.writestr(name, raw)
    manifest = {
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_sha256": sha(selection_raw), "cases": cases,
        "modes": MODES, "planned_runs": 96, "processes": 1, "torch_threads": 1,
        "checkpoint": str(MODEL.relative_to(ROOT)), "checkpoint_sha256": EXPECTED_MODEL,
        "config_sha256": sha(MODEL.with_name("config.json").read_bytes()),
        "source_sha256": source_hashes, "cache_sha256": cache_hashes,
        "effective_constants": {k: repr(v) for k, v in vars(cfg).items() if k.isupper()},
        "effective_safety_constants": {m.__name__: {k: repr(v) for k, v in vars(m).items()
                                                   if k.isupper()} for m in modules},
        "low_speed_start_frac": 0.0, "python": sys.version, "torch": torch.__version__,
        "scope": "Outcome-enriched development pilot; not a representative success-rate estimate",
        "new_authorization": "User: Run new episodes to evaluation and confirm v9 performs better or worse.",
    }
    write_json(OUT / "manifest.json", manifest)
    model = load_model(MODEL)
    (OUT / "attempts").mkdir()
    (OUT / "traces").mkdir()

    class TraceEnv(ASVLidarEnv):
        def step(self, action):
            before = [self.asv_x, self.asv_y, self.asv_h, self.u_body]
            result = super().step(action)
            filt = getattr(self, "_safety_v2", None)
            if self.estop_enabled and type(filt).__name__ != f"SafetyFilterV{cfg.SAFETY_VERSION}":
                raise AssertionError("Native dispatch selected an unexpected filter")
            details = getattr(filt, "last", {})
            self.why_counts[details.get("why", "off")] += 1
            self.suppressed += int(details.get("v9_override_suppressed", False))
            self.checked += int(details.get("v9_current_plan_checked", False))
            self.trace_steps += 1
            record = dict(step=self.trace_steps, pre_state=before,
                          post_state=[self.asv_x, self.asv_y, self.asv_h, self.u_body],
                          policy_action=action, rudder_command=self.rudder / 100.0,
                          signed_rpm_command=self.rpm,
                          changed=bool(getattr(self, "safety_v2_changed", False)),
                          brake=bool(getattr(self, "_v2_brake", False)),
                          filter=details, plan=getattr(filt, "plan", None))
            self.trace.write(json.dumps(jsonable(record), allow_nan=False) + "\n")
            return result

    from collections import Counter
    started = time.perf_counter()
    rows, attempt = [], 0
    for index, case in enumerate(cases):
        # Rotate order to avoid systematically assigning transient CPU load to V9.
        modes = MODES[index % 3:] + MODES[:index % 3]
        for mode in modes:
            if (OUT / "STOP").exists():
                print("STOP requested; completed results preserved", flush=True)
                return
            for name, expected in source_hashes.items():
                if sha((ROOT / name).read_bytes()) != expected:
                    raise ValueError(f"Frozen source changed during evaluation: {name}")
            attempt += 1
            identity = dict(attempt=attempt, case=case["case"], mode=mode, seed=case["seed"])
            write_json(OUT / "attempts" / f"{attempt:03d}.json", identity)
            cfg.SAFETY_VERSION = 1 if mode == "off" else int(mode[1:])
            env = TraceEnv(render_mode=None, emergency_stop=mode != "off", low_speed_start_frac=0.0)
            env.why_counts, env.suppressed, env.checked, env.trace_steps = Counter(), 0, 0, 0
            episode_started = time.perf_counter()
            try:
                with (OUT / "traces" / f"{attempt:03d}_{mode}.jsonl").open("x", encoding="utf-8", newline="\n") as trace:
                    env.trace = trace
                    result = run_episode(env, copy.deepcopy(scenes[case["case"]]),
                                         int(case["seed"]), "model", model)
                    trace.flush()
                    os.fsync(trace.fileno())
                row = dict(identity, dataset=case["dataset"], stratum=case["selection_stratum"],
                           elapsed_s=time.perf_counter() - episode_started,
                           suppressed_steps=env.suppressed, checked_steps=env.checked,
                           why_counts=dict(env.why_counts), **result)
                write_json(OUT / "attempts" / f"{attempt:03d}_result.json", row)
                rows.append(row)
                with (OUT / "episodes.csv").open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if attempt == 1:
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                print(f"{attempt}/96 {mode} {case['case']}: {row['outcome']} "
                      f"({row['elapsed_s']:.1f}s, interventions={row['safety_v2_steps']}, "
                      f"suppressed={row['suppressed_steps']})", flush=True)
            finally:
                env.close()
    drift = [name for name, expected in source_hashes.items()
             if sha((ROOT / name).read_bytes()) != expected]
    write_json(OUT / "completion.json", dict(completed_runs=len(rows),
        elapsed_s=time.perf_counter() - started, source_drift=drift,
        selection_unchanged=sha((OUT / "selection.json").read_bytes()) == sha(selection_raw),
        checkpoint_unchanged=sha(MODEL.read_bytes()) == EXPECTED_MODEL))
    print(f"Completed {len(rows)} fresh runs in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
