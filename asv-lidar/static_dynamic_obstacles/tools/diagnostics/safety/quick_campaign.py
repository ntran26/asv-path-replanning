"""Fixed 24-case paired diagnostic; a shared hard cap of 100 NEW attempts.

    python -B tools/diagnostics/safety/quick_campaign.py prepare
    python -B tools/diagnostics/safety/quick_campaign.py snapshot
    python -B tools/diagnostics/safety/quick_campaign.py run --tag candidate1

Preparation and snapshots run no episodes. The run command must only be used
after the old queues are stopped and the new candidate is ready. Every episode
attempt, including failures/retries, first consumes an immutable numbered token.
All tags share one campaign directory and its maximum 100 tokens. A normal paired
run is 24 cases x off/v4/v5 = 72 attempts. Existing outcomes are never substituted for
these new matched runs. No full-suite population claim follows from this sample.
Create results/safety_dev/quick_v5_budget100/STOP to stop before the next attempt.
The current episode is allowed to finish and commit; remove STOP before resuming.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import uuid

# Establish limits before cache deserialization imports NumPy or model code.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/tiers")]
from suite_status import committed_records, read_shared_bytes

CAMPAIGN = ROOT / "results/safety_dev/quick_v5_budget100"
INVENTORY = ROOT / "results/safety_dev/suites/inventory_v5/manifest.json"
LIMIT = 100
MODES = ("off", "v4", "v5")
PLANNED_EPISODES = 24 * len(MODES)
SALT = "quick-v5-24cases-2026-10-02-v1"
PLAN_REVISION = 3
SAMPLE_NAME = f"sample_manifest_v{PLAN_REVISION}.json"
ALLOWED_CANDIDATE_EDITS = {"src/safety_v5.py", "src/safety_recovery.py", "src/safety_feedback.py"}
OBSERVED_FLAGS = ("sideslip_rescue_evaluated", "sideslip_rescue_admitted",
                  "verified_policy_priority_enabled", "verified_policy_preserved",
                  "continuation_override_prevented", "policy_feedback_evaluated",
                  "policy_feedback_preserved")


class StopRequested(RuntimeError):
    """A cooperative stop never consumes an episode token."""


def check_stop(campaign):
    if (Path(campaign) / "STOP").exists():
        raise StopRequested("STOP marker exists; no further episode attempt reserved")


def mode_settings(mode):
    if mode not in MODES:
        raise ValueError(f"Unsupported quick comparison mode: {mode}")
    return (1, False) if mode == "off" else (int(mode[1:]), True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def family(case):
    if case["suite"] in ("frozen_b", "frozen_r"):
        return case["test_id"].rsplit("-", 1)[0]
    if case["suite"] == "frozen_a":
        return case["case"].split("-")[1]
    if case["suite"] in ("field_deployment", "field_validation"):
        return case["case"].rsplit("-", 1)[0]
    return "development_not_sampled"


def select_cases(cases):
    """Predeclared strata and fixed hash ranking; accepts no outcome information."""
    strata = []
    b_families = sorted({family(c) for c in cases if c["suite"] == "frozen_b"})
    if len(b_families) != 8:
        raise ValueError("Expected exactly eight frozen B families")
    for component, families in (("frozen_b", b_families),
            ("frozen_r", ["BAS-HO-NC", "CH-HO-RE", "BAS-CR-RE", "CH-OT-RE"]),
            ("field_validation", ["FV-NT", "FV-HO", "FV-CRP", "FV-BO"])):
        for name in families:
            strata.append((component + "/" + name,
                           [c for c in cases if c["suite"] == component and family(c) == name]))
    # Geometry-only coverage: guarantee narrow and intermediate named cases.
    # Draft v1 hash ranking selected two wide cases; no episodes used that draft.
    for name in ("A-HO-N", "A-CRS-I"):
        strata.append(("frozen_a/" + name, [c for c in cases if c["suite"] == "frozen_a" and c["case"] == name]))
    for layout in ("L1", "L2", "L3"):
        for speed in ("FIX", "VAR"):
            strata.append((f"field_deployment/{layout}/{speed}", [c for c in cases
                if c["suite"] == "field_deployment" and c["case"].split("-")[1] == layout
                and len(c["case"].split("-")) == 5 and c["case"].split("-")[3] == speed]))
    selected = []
    for name, pool in strata:
        if not pool:
            raise ValueError(f"Empty predeclared sample stratum: {name}")
        ranked = sorted(pool, key=lambda c: sha((SALT + "|" + c["suite"] + "|" + c["case"]
                                                 + "|" + c["scenario_sha256"]).encode()))
        selected.append(dict(ranked[0], sampling_stratum=name, stratum_population=len(pool),
                             family=family(ranked[0]), design_probability=None, inverse_probability_weight=None))
    if len(selected) != 24 or len({(c["suite"], c["case"]) for c in selected}) != 24:
        raise ValueError("Sample must contain exactly 24 distinct cases")
    return selected


def verify_reference_sources(baseline):
    mismatches = [name for name, digest in baseline["source_sha256"].items()
                  if name not in ALLOWED_CANDIDATE_EDITS and sha(read_shared_bytes(ROOT / name)) != digest]
    if mismatches:
        raise ValueError(f"Selected v4/generator/evaluator baseline changed: {mismatches}")


def validate_sample_identity(selected, canonical_cases):
    canonical = {(c["suite"], c["case"]): c for c in canonical_cases}
    if [(c["suite"], c["case"]) for c in selected] != [(c["suite"], c["case"]) for c in select_cases(canonical_cases)]:
        raise ValueError("Sample differs from the predeclared selection")
    for case in selected:
        original = canonical[case["suite"], case["case"]]
        for name in ("seed", "scenario_sha256", "obstacles", "test_id"):
            if name not in case or case[name] != original[name]:
                raise ValueError(f"Sample canonical {name} mismatch: {case['suite']}/{case['case']}")


def cached_sample(cases, inventory):
    """Read cached scenes only; reconstruct four robust clones, never resample."""
    import suite
    cache = ROOT / "results/safety_dev/suite_cache" / inventory["suite_details"]["scenario_cache_digest"]
    scenes, cache_hashes = {}, {}
    for component in ("frozen_b", "frozen_a", "field_deployment", "field_validation"):
        raw = read_shared_bytes(cache / f"{component}.pkl")
        saved = pickle.loads(raw)
        if saved["component"] != component or canonical_sha(saved["provenance"])[:20] != cache.name:
            raise ValueError(f"Cache provenance mismatch: {component}")
        cache_hashes[component] = sha(raw)
        payload = saved["payload"]
        if component == "frozen_b":
            built_rows = payload[0]
        elif component == "field_deployment":
            built_rows = [row["built"] for row in payload[0]]
        else:
            built_rows = payload
        scenes[component] = {built.case_id: built for built in built_rows}
    result = {}
    for case in cases:
        if case["suite"] == "frozen_r":
            base, behaviour = case["case"].rsplit("-", 1)
            built = copy.deepcopy(scenes["frozen_b"][base])
            built.target_behaviour = suite.target_model(behaviour.lower(), built.encounter_class)
            built.case_id = case["case"]
        else:
            built = scenes[case["suite"]][case["case"]]
        if built.digest() != case["scenario_sha256"]:
            raise ValueError(f"Cached canonical scene mismatch: {case['case']}")
        result[case["suite"], case["case"]] = built
    return result, cache_hashes


def features(built):
    flags = built.flags or {}
    fixed = flags.get("fixed_obstacles")
    obstacles = fixed if fixed is not None else built.obstacles
    return {"encounter_class": built.encounter_class, "geometry_mode": built.geometry_mode,
            "nominal_width_m": float(built.nominal_width), "effective_width_at_cpa_m": float(built.w_eff_at_cpa),
            "bend_deg": float(built.bend_deg), "slant_deg": float(built.slant_realised_deg),
            "canonical_obstacle_polygons": len(obstacles), "declared_obstacle_count": int(built.n_obstacles),
            "obstacle_specification": "fixed polygons" if fixed is not None or len(obstacles) else "seeded at environment reset",
            "target_motion": built.target_behaviour, "target_speed_mps": float(built.target_speed),
            "target_speed_profile": flags.get("speed_profile"), "nominal_dcpa_m": float(built.dcpa_m),
            "nominal_tcpa_s": float(built.tcpa_s), "own_spawn": built.own_spawn,
            "target_spawn": built.target_spawn, "basin_leg": flags.get("basin_leg")}


def prepare():
    baseline = json.loads(read_shared_bytes(CAMPAIGN / "frozen_baseline.json"))
    verify_reference_sources(baseline)
    raw = read_shared_bytes(INVENTORY)
    inventory = json.loads(raw)
    cases = inventory["cases"]
    if len(cases) != 2890 or len({(c["suite"], c["case"]) for c in cases}) != 2890:
        raise ValueError("Expected the original 2,890-case canonical inventory")
    selected = select_cases(cases)  # Selection occurs before any journal/outcome is read.
    built, cache_hashes = cached_sample(selected, inventory)
    for case in selected:
        case["features"] = features(built[case["suite"], case["case"]])
    previous_bytes = read_shared_bytes(CAMPAIGN / "sample_manifest_v2.json")
    if canonical_sha(selected) != canonical_sha(json.loads(previous_bytes)["cases"]):
        raise ValueError("Revision3 must preserve all24 revision2 cases and their features")
    populations = Counter((c["suite"], family(c)) for c in cases)
    sample_counts = Counter((c["suite"], c["family"]) for c in selected)
    coverage = [{"component": key[0], "family": key[1], "population_cases": count,
                 "sample_cases": sample_counts[key], "omitted": sample_counts[key] == 0,
                 "weight": "", "design_probability": ""} for key, count in sorted(populations.items())]
    manifest = {"schema": 1, "plan_revision": PLAN_REVISION, "selection_salt": SALT, "modes": list(MODES), "new_episodes_planned": PLANNED_EPISODES,
        "previous_sample_manifest_sha256": sha(previous_bytes),
        "shared_new_attempt_cap": LIMIT, "original_cases": 2890, "original_mode_episodes_planned": 8670,
        "inventory_path": str(INVENTORY), "inventory_sha256": sha(raw),
        "scenario_cache_digest": inventory["suite_details"]["scenario_cache_digest"], "cache_sha256": cache_hashes,
        "frozen_baseline_sha256": sha(read_shared_bytes(CAMPAIGN / "frozen_baseline.json")),
        "selection_source_sha256": sha(Path(__file__).read_bytes()), "cases": selected,
        "sampling_design": "Fixed predeclared strata; choose lowest SHA256 rank without using outcomes. This is a diagnostic subset, not a probability sample.",
        "limitations": ["No inverse-probability weights: design inclusion probabilities are not defined.",
            "24 cases do not estimate all 2,890-case performance; omitted families remain untested by this quick run.",
            "All field-layout and validation cases are simulated, not new real-world trials.",
            "Old partial benchmark outcomes remain separate; new off/v4/v5 triples use identical scenario and reset seed.",
            "Each failed/interrupted/repeated episode attempt consumes the shared maximum of100."]}
    planned = [dict(mode=mode, **case) for mode in ("off", "v4", "v5") for case in cases]
    write_json(CAMPAIGN / SAMPLE_NAME, manifest)
    write_csv(CAMPAIGN / f"sample_family_coverage_v{PLAN_REVISION}.csv", coverage)
    if not (CAMPAIGN / "original_8670_planned_identities.csv").exists():
        write_csv(CAMPAIGN / "original_8670_planned_identities.csv", planned)
    write_csv(CAMPAIGN / f"sample_features_v{PLAN_REVISION}.csv", [{"suite": case["suite"], "case": case["case"],
        "sampling_stratum": case["sampling_stratum"], "seed": case["seed"], **case["features"]} for case in selected])
    print(f"Prepared 24 fixed matched cases;{PLANNED_EPISODES} new runs planned;shared hard attempt cap100.", flush=True)
    snapshot()


def snapshot():
    """Preserve observed committed outcomes for every original planned identity."""
    inventory = json.loads(read_shared_bytes(INVENTORY))
    baseline = json.loads(read_shared_bytes(CAMPAIGN / "frozen_baseline.json"))["settings"]
    expected = {(mode, c["suite"], c["case"]): c for mode in ("off", "v4", "v5") for c in inventory["cases"]}
    completed, sources = {}, []
    for directory in sorted((ROOT / "results/safety_dev/suites").iterdir()):
        if not (directory / "manifest.json").exists():
            continue
        raw_manifest = read_shared_bytes(directory / "manifest.json")
        settings = json.loads(raw_manifest)["settings"]
        compatible = all(settings.get(key) == baseline.get(key) for key in
            ("checkpoint_sha256", "config_sha256", "source_sha256", "effective_constants", "effective_safety_constants", "runtime_versions"))
        if not compatible:
            sources.append({"run": directory.name, "included": False, "reason": "different frozen settings/source snapshot"})
            continue
        try:
            raw_journal = read_shared_bytes(directory / "episodes.jsonl")
        except FileNotFoundError:
            raw_journal = b""
        rows, tail = committed_records(raw_journal)
        for row in rows:
            key = (row["mode"], row["suite"], row["case"])
            if key not in expected or key in completed:
                raise ValueError(f"Unknown/duplicate original benchmark record: {key}")
            if any(row.get(name) != expected[key][name] for name in ("seed", "scenario_sha256")):
                raise ValueError(f"Original benchmark identity mismatch: {key}")
            if row["outcome"] not in ("goal", "timeout", "collision:obstacle", "collision:boundary", "collision:target"):
                raise ValueError(f"Invalid original benchmark outcome: {key}")
            completed[key] = (row, directory.name)
        sources.append({"run": directory.name, "included": True, "committed": len(rows),
                        "ignored_tail_bytes": tail, "manifest_sha256": sha(raw_manifest), "journal_sha256": sha(raw_journal)})
    mapping = []
    for key, case in expected.items():
        observed = completed.get(key)
        mapping.append(dict(mode=key[0], **case, record_status="committed" if observed else "not_recorded",
                            outcome=observed[0]["outcome"] if observed else "", source_run=observed[1] if observed else ""))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    write_csv(CAMPAIGN / f"original_completion_map_{stamp}.csv", mapping)
    write_json(CAMPAIGN / f"original_completion_map_{stamp}.json", {"planned": len(expected), "committed": len(completed),
        "not_recorded": len(expected)-len(completed), "sources": sources, "scope": "original frozen benchmark only; snapshot is not a complete benchmark result"})
    print(f"Preserved {len(completed)}/{len(expected)} original committed records; missing outcomes remain blank.", flush=True)


def reserve_attempt(directory, identity, *, limit=LIMIT):
    """Atomic numbered files bound attempts even across tags or concurrent calls."""
    if not 1 <= limit <= LIMIT:
        raise ValueError("Attempt cap must be between1 and100")
    check_stop(directory.parent)
    directory.mkdir(parents=True, exist_ok=True)
    for number in range(1, limit + 1):
        path = directory / f"{number:03d}.json"
        try:
            # A partial/empty token still consumes its slot after any failure.
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(identity, attempt=number), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return number
    raise RuntimeError(f"HARD NEW EPISODE BUDGET EXHAUSTED: {limit} attempt slots already reserved")


def audited_environment(base):
    """Observe post-step diagnostics only; return the original step result."""
    class AuditEnv(base):
        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            self.quick_counts = {name + "_steps": 0 for name in OBSERVED_FLAGS}
            self.quick_counts.update(recovery_template_counts=Counter(), realized_obstacle_count=len(self.obstacles))
            return result

        def step(self, action):
            result = super().step(action)
            last = getattr(getattr(self, "_safety_v2", None), "last", {})
            for name in OBSERVED_FLAGS:
                self.quick_counts[name + "_steps"] += int(bool(last.get(name, False)))
            template = last.get("recovery_template")
            if template is not None:
                self.quick_counts["recovery_template_counts"][str(template)] += 1
            return result
    return AuditEnv


def summarize_results(rows):
    indexed = {(r["mode"], r["suite"], r["case"]): r for r in rows}
    identities = sorted({(r["suite"], r["case"]) for r in rows})
    comparisons = {}
    for candidate, reference in (("v5", "v4"), ("v5", "off"), ("v4", "off")):
        matched = []
        for suite, case in identities:
            a, b = indexed.get((candidate, suite, case)), indexed.get((reference, suite, case))
            if a is None or b is None:
                continue
            matched.append({"suite": suite, "case": case, "candidate_outcome": a["outcome"],
                            "reference_outcome": b["outcome"],
                            "gained_goal": a["outcome"] == "goal" and b["outcome"] != "goal",
                            "lost_goal": a["outcome"] != "goal" and b["outcome"] == "goal"})
        comparisons[candidate + "_vs_" + reference] = {
            "candidate": candidate, "reference": reference, "paired_cases": len(matched),
            "candidate_unpaired_cases": sum(r["mode"] == candidate for r in rows) - len(matched),
            "reference_unpaired_cases": sum(r["mode"] == reference for r in rows) - len(matched),
            "gained_goals": sum(r["gained_goal"] for r in matched),
            "lost_goals": sum(r["lost_goal"] for r in matched), "pairs": matched}
    pairs = []
    for suite, case in identities:
        a, b = indexed.get(("v5", suite, case)), indexed.get(("v4", suite, case))
        if a is None or b is None:
            continue
        pairs.append({"suite": suite, "case": case, "v4_outcome": b["outcome"], "v5_outcome": a["outcome"],
                      "gained_goal": a["outcome"] == "goal" and b["outcome"] != "goal",
                      "lost_goal": a["outcome"] != "goal" and b["outcome"] == "goal",
                      "v5_sideslip_evaluated_steps": a.get("sideslip_rescue_evaluated_steps", 0),
                      "v5_sideslip_admitted_steps": a.get("sideslip_rescue_admitted_steps", 0),
                      "v5_verified_policy_preserved_steps": a.get("verified_policy_preserved_steps", 0),
                      "v5_continuation_override_prevented_steps": a.get("continuation_override_prevented_steps", 0),
                      "v5_recovery_template_counts": a.get("recovery_template_counts", {})})
    mechanisms = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        templates = Counter()
        for row in selected:
            templates.update(row.get("recovery_template_counts", {}))
        mechanisms[mode] = {"evaluated_steps": sum(r.get("sideslip_rescue_evaluated_steps", 0) for r in selected),
                            "admitted_steps": sum(r.get("sideslip_rescue_admitted_steps", 0) for r in selected),
                            "episodes_evaluated": sum(r.get("sideslip_rescue_evaluated_steps", 0) > 0 for r in selected),
                            "episodes_admitted": sum(r.get("sideslip_rescue_admitted_steps", 0) > 0 for r in selected),
                            "recovery_template_counts": dict(templates)}
        for name in OBSERVED_FLAGS:
            mechanisms[mode][name + "_steps"] = sum(r.get(name + "_steps", 0) for r in selected)
            mechanisms[mode][name + "_episodes"] = sum(r.get(name + "_steps", 0) > 0 for r in selected)
    return {"sample_only": True, "population_performance_estimate": False, "paired_cases": len(pairs),
            "gained_goals_vs_v4": sum(r["gained_goal"] for r in pairs), "lost_goals_vs_v4": sum(r["lost_goal"] for r in pairs),
            "gained_goals_vs_off": comparisons["v5_vs_off"]["gained_goals"],
            "lost_goals_vs_off": comparisons["v5_vs_off"]["lost_goals"], "comparisons": comparisons,
            "outcomes": dict(Counter(r["mode"] + "/" + r["outcome"] for r in rows)),
            "mechanism_counts": mechanisms, "pairs": pairs}


def run(tag, resume=False):
    if Path(tag).name != tag or tag in (".", "..") or "\\" in tag:
        raise ValueError("Tag must be a plain filename stem")
    check_stop(CAMPAIGN)
    manifest_bytes = read_shared_bytes(CAMPAIGN / SAMPLE_NAME)
    manifest = json.loads(manifest_bytes)
    if manifest["modes"] != list(MODES) or manifest["new_episodes_planned"] != PLANNED_EPISODES or len(manifest["cases"]) != 24:
        raise ValueError("Unexpected sample modes/count; the comparison is fixed at72 episodes")
    baseline = json.loads(read_shared_bytes(CAMPAIGN / "frozen_baseline.json"))
    if sha(read_shared_bytes(CAMPAIGN / "frozen_baseline.json")) != manifest["frozen_baseline_sha256"]:
        raise ValueError("Frozen baseline provenance changed")
    verify_reference_sources(baseline)
    inventory = json.loads(read_shared_bytes(INVENTORY))
    if sha(read_shared_bytes(INVENTORY)) != manifest["inventory_sha256"]:
        raise ValueError("Canonical inventory changed")
    validate_sample_identity(manifest["cases"], inventory["cases"])
    built, cache_hashes = cached_sample(manifest["cases"], inventory)
    if cache_hashes != manifest["cache_sha256"]:
        raise ValueError("Cached scene files changed")
    import torch
    torch.set_num_threads(1)
    import constants as cfg
    import curriculum
    import train_formulation as tf
    from prediction_audit import runtime_overrides
    from suite_eval import settings_snapshot
    from common import load_model, run_episode
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    model_path = Path(baseline["settings"]["checkpoint"])
    config_path = model_path.with_name("config.json")
    if config_path.exists():
        for key, value in (json.loads(read_shared_bytes(config_path)).get("constant_overrides") or {}).items():
            setattr(cfg, key, value)
    runtime_overrides()
    settings = settings_snapshot(model_path, sorted({c["suite"] for c in manifest["cases"]}), list(MODES))
    for mode in ("v2", "v3", "v4"):
        if settings["effective_safety_constants"][mode] != baseline["settings"]["effective_safety_constants"][mode]:
            raise ValueError(f"Reference {mode} settings changed")
    for key in ("checkpoint_sha256", "config_sha256", "effective_constants", "runtime_versions"):
        if settings[key] != baseline["settings"][key]:
            raise ValueError(f"Frozen reference {key} changed")
    directory = CAMPAIGN / "runs" / tag
    metadata = {"sample_manifest_sha256": sha(manifest_bytes), "settings": settings,
                "runner_sha256": sha(Path(__file__).read_bytes()), "shared_attempt_cap": LIMIT}
    if resume:
        if json.loads(read_shared_bytes(directory / "metadata.json")) != metadata:
            raise ValueError("Resume source/settings changed; use a new tag within the same shared budget")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "metadata.json", metadata)
    lock = CAMPAIGN / "active_run.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "tag": tag}, handle)
    try:
        done = {}
        expected = {(mode, c["suite"], c["case"]): c for c in manifest["cases"] for mode in MODES}
        for path in directory.glob("episode_*.json"):
            row = json.loads(read_shared_bytes(path))
            key = (row["mode"], row["suite"], row["case"])
            if key not in expected or key in done or any(row[k] != expected[key][k] for k in ("seed", "scenario_sha256")):
                raise ValueError("Invalid/duplicate saved quick result")
            if row.get("outcome") not in ("goal", "timeout", "collision:obstacle", "collision:boundary", "collision:target") or row.get("run_metadata_sha256") != canonical_sha(metadata):
                raise ValueError("Saved outcome/settings are invalid")
            token = json.loads(read_shared_bytes(CAMPAIGN / "attempts" / f"{int(row['attempt']):03d}.json"))
            if any(row.get(name) != value for name, value in token.items()):
                raise ValueError("Saved result disagrees with its consumed attempt token")
            done[key] = row
        remaining = sum(not (CAMPAIGN / "attempts" / f"{n:03d}.json").exists() for n in range(1, LIMIT + 1))
        if len(expected) - len(done) > remaining:
            raise RuntimeError(f"Insufficient shared budget for remaining paired cases: need{len(expected)-len(done)}, have{remaining}")
        model = load_model(model_path)
        AuditEnv = audited_environment(ASVLidarEnv)
        for case in manifest["cases"]:
            for mode in MODES:
                key = (mode, case["suite"], case["case"])
                if key in done:
                    continue
                identity = {"tag": tag, "mode": mode, "suite": case["suite"], "case": case["case"],
                            "seed": case["seed"], "scenario_sha256": case["scenario_sha256"],
                            "run_metadata_sha256": canonical_sha(metadata)}
                number = reserve_attempt(CAMPAIGN / "attempts", identity)
                cfg.SAFETY_VERSION, emergency_stop = mode_settings(mode)
                env = None
                try:
                    env = AuditEnv(render_mode=None, emergency_stop=emergency_stop)
                    started = time.perf_counter()
                    result = run_episode(env, built[case["suite"], case["case"]], case["seed"], "model", model, case["obstacles"])
                    row = dict(result, **identity, attempt=number, seconds=time.perf_counter()-started, **env.quick_counts)
                    write_json(directory / f"episode_{number:03d}.json", row)
                    done[key] = row
                    print(f"{tag} {len(done)}/{PLANNED_EPISODES}; budget attempt {number}/100; {mode} {case['suite']} {case['case']}: {row['outcome']}", flush=True)
                except BaseException as exc:
                    write_json(directory / f"error_{number:03d}.json", dict(identity, attempt=number, error=repr(exc)))
                    raise
                finally:
                    if env is not None:
                        env.close()
        write_json(directory / ("complete_" + uuid.uuid4().hex[:8] + ".json"), {"completed": len(done), "expected": PLANNED_EPISODES,
                   **summarize_results(list(done.values()))})
    except StopRequested:
        write_json(directory / ("stopped_" + uuid.uuid4().hex[:8] + ".json"), {
            "completed": len(done), "expected": PLANNED_EPISODES, "reason": "cooperative STOP marker",
            **summarize_results(list(done.values()))})
        print(f"Stopped before next attempt: {len(done)}/{PLANNED_EPISODES} results preserved; STOP marker retained.", flush=True)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("prepare", "snapshot", "run"))
    parser.add_argument("--tag")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        if not args.tag:
            parser.error("run requires --tag")
        run(args.tag, args.resume)
    elif args.action == "prepare":
        prepare()
    else:
        snapshot()


if __name__ == "__main__":
    main()
