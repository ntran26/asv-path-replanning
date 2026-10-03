"""Score completed saved decision snapshots; no simulator or controller imports.

Truth is used only for offline error scoring. Raw and filtered body velocities
are aligned to the same recorded pre-decision state. Run tags never overwrite.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


ROOT = Path(__file__).resolve().parents[3]
COMPONENTS = ("u_mps", "v_mps", "yaw_deg_s")
METHODS = ("raw", "filtered", "derived_prior")


def sha(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def metric(values):
    if not values:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "maximum_absolute_error": None}
    return {"n": len(values), "mae": statistics.fmean(abs(x) for x in values),
            "rmse": math.sqrt(statistics.fmean(x * x for x in values)),
            "bias": statistics.fmean(values), "maximum_absolute_error": max(abs(x) for x in values)}


def summarize(frames):
    return {method: {component: metric([r[method + "_error"][j] for r in frames
                                      if r.get(method + "_error") is not None])
                     for j, component in enumerate(COMPONENTS)} for method in METHODS}


def body_values(row):
    before, decision = row["diagnostic_before"], row["diagnostic_decision"]
    own = before["truth_scoring_only"]["own"]
    truth = [own["u_body"], own["v_body"], own["asv_w"]]
    raw = list(decision["raw_ego_u_v_yaw_rad_s"])
    raw[2] = math.degrees(raw[2])
    snap = decision["snapshot"]
    filtered = [snap["u"], snap["v"], math.degrees(snap["r"])]
    held = before["onboard"]["ego_hold_u_v_yaw_deg_s"]
    if not all(math.isfinite(x) for values in (truth, raw, filtered, held) for x in values):
        raise ValueError("Nonfinite body measurement/state")
    if max(abs(a - b) for a, b in zip(raw, held)) > 1e-10:
        raise ValueError("Captured raw ego differs from the pre-decision held measurement")
    recorded = row["pre_state"]
    if max(abs(a - b) for a, b in zip(recorded, [own["asv_x"], own["asv_y"], own["asv_h"], own["u_body"]])) > 1e-10:
        raise ValueError("Truth scoring state differs from recorded pre-decision state")
    return truth, raw, filtered


def archived_weight(run, manifest):
    """Read numeric gain from hashed archived source; do not execute that source."""
    with zipfile.ZipFile(run / "evaluated_sources.zip") as archive:
        text = archive.read("src/safety_v2.py")
        assert hashlib.sha256(text).hexdigest() == manifest["source_sha256"]["src/safety_v2.py"]
        observer = archive.read("src/safety_observer.py")
        assert hashlib.sha256(observer).hexdigest() == manifest["source_sha256"]["src/safety_observer.py"]
    for node in ast.parse(text.decode("utf-8-sig")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EGO_SMOOTHING" for t in node.targets):
            weight = float(ast.literal_eval(node.value))
            if not 0 < weight < 1:
                raise ValueError("Cannot reconstruct this observer weight")
            return weight
    raise ValueError("Archived observer weight unavailable")


def collect(root):
    episodes, excluded, all_frames = [], [], []
    seen = set()
    for run in sorted(root.iterdir()):
        manifest_path = run / "manifest.json"
        if not run.is_dir() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("snapshots", False):
            excluded.append({"run": run.name, "reason": "full snapshots not enabled"})
            continue
        weight = archived_weight(run, manifest)
        result_paths = sorted((run / "attempts").glob("*_result.json"))
        completed = set()
        for result_path in result_paths:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            attempt, mode = int(result["attempt"]), result["mode"]
            trace = run / f"traces/{attempt:03d}_{mode}.jsonl"
            completed.add(trace.name)
            trace_sha = sha(trace)
            if trace_sha in seen:
                excluded.append({"trace": str(trace), "reason": "duplicate exact trace hash"})
                continue
            seen.add(trace_sha)
            case = next(c for c in manifest["cases"] if c["case"] == result["case"])
            identity = f"{run.name}:{attempt:03d}:{mode}:{result['case']}"
            frames = []
            with trace.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle, 1):
                    row = json.loads(line)
                    if row["step"] != index:
                        raise ValueError(f"Noncontiguous trace: {trace}")
                    truth, raw, filtered = body_values(row)
                    before = row["diagnostic_before"]
                    dt = float(ast.literal_eval(manifest["constants"]["UPDATE_RATE"]))
                    if abs(before["elapsed_time"] - (index - 1) * dt) > 1e-7:
                        raise ValueError(f"Timestamp mismatch: {trace}, {index}")
                    fresh = not before["onboard"]["pose_stale"]
                    observer_on = row["filter"].get("model_ego_observer", False)
                    prior = None
                    if observer_on and index > 1:
                        # Exact inversion of archived update equation, not a separately
                        # logged prior or a new prediction. First-frame initialization
                        # has no preceding model prior and is intentionally excluded.
                        prior = ([(a - weight * b) / (1 - weight) for a, b in zip(filtered, raw)]
                                 if fresh else list(filtered))
                    item = {"episode": identity, "run": run.name, "case": result["case"],
                            "step": index, "fresh": fresh, "model_observer": observer_on,
                            "why": row["filter"].get("why", "idle"), "changed": row.get("changed", False),
                            "truth": truth, "raw": raw, "filtered": filtered, "derived_prior": prior,
                            "raw_error": [a-b for a, b in zip(raw, truth)],
                            "filtered_error": [a-b for a, b in zip(filtered, truth)],
                            "derived_prior_error": None if prior is None else [a-b for a, b in zip(prior, truth)]}
                    frames.append(item)
            if len(frames) != int(result["steps"]):
                raise ValueError(f"Completed result disagrees with trace length: {trace}")
            versions = {name: manifest["source_sha256"].get(name) for name in
                        ["src/env.py", "src/safety_observer.py", "src/safety_v2.py", "src/safety_v4.py",
                         "src/safety_v6.py", "src/classical/common.py", "bluefin/dynamics.py"]}
            episodes.append({"episode": identity, "run": run.name, "attempt": attempt, "mode": mode,
                "case": result["case"], "seed": result["seed"], "scenario_sha256": case["scenario_sha256"],
                "outcome": result["outcome"], "frames": len(frames), "fresh_frames": sum(x["fresh"] for x in frames),
                "filter_class": manifest["filter_classes"][mode],
                "constructor_options": manifest.get("constructor_options", {}).get(mode, {}),
                "trace": str(trace.relative_to(ROOT)).replace("\\", "/"), "trace_sha256": trace_sha,
                "manifest_sha256": sha(manifest_path), "result_sha256": sha(result_path),
                "source_versions": versions, "observer_measurement_weight": weight,
                "constants": {key: manifest["constants"].get(key) for key in
                              ["VESSEL_RANDOMISATION_SCALE", "EGO_SPEED_NOISE", "EGO_YAW_RATE_NOISE_DPS", "POSE_STALE_PROB"]},
                "all": summarize(frames), "fresh": summarize([x for x in frames if x["fresh"]]),
                "stale": summarize([x for x in frames if not x["fresh"]])})
            all_frames.extend(frames)
        for trace in (run / "traces").glob("*.jsonl"):
            if trace.name not in completed:
                excluded.append({"trace": str(trace.relative_to(ROOT)).replace("\\", "/"),
                                 "reason": "no committed completed result; excluded without reading"})
    return episodes, excluded, all_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "results/safety_dev/v10_iterations")
    parser.add_argument("--tag", required=True, help="New output folder name under input-root/audits")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in (".", ".."):
        parser.error("tag must be one plain folder name")
    output = args.input_root / "audits" / args.tag
    output.mkdir(parents=True, exist_ok=False)
    episodes, excluded, frames = collect(args.input_root)
    fresh = [r for r in frames if r["fresh"]]
    run_groups = {name: summarize([r for r in fresh if r["run"] == name])
                  for name in sorted({r["run"] for r in frames})}
    summary = {"scope": "Offline development trace scoring only; no new episodes or model calls.",
        "components": list(COMPONENTS), "episodes": len(episodes), "unique_cases": len({r["case"] for r in episodes}),
        "frames": len(frames), "fresh_frames": len(fresh), "stale_frames": len(frames)-len(fresh),
        "pooled_all": summarize(frames), "pooled_fresh": summarize(fresh), "pooled_stale": summarize([r for r in frames if not r["fresh"]]),
        "per_run_fresh": run_groups,
        "episode_comparisons": {component: {"filtered_rmse_better": sum(e["fresh"]["filtered"][component]["rmse"] < e["fresh"]["raw"][component]["rmse"] for e in episodes if e["fresh"]["raw"][component]["n"]),
            "compared_episodes": sum(bool(e["fresh"]["raw"][component]["n"]) for e in episodes),
            "macro_raw_mae": statistics.fmean(e["fresh"]["raw"][component]["mae"] for e in episodes if e["fresh"]["raw"][component]["n"]),
            "macro_filtered_mae": statistics.fmean(e["fresh"]["filtered"][component]["mae"] for e in episodes if e["fresh"]["raw"][component]["n"])} for component in COMPONENTS},
        "excluded": excluded,
        "prior_note": "The prior is not directly captured. For fresh decisions after initialization, derived_prior=(filtered-weight*raw)/(1-weight), using the hashed archived fixed-gain equation; stale prior=filtered. This is algebraic reconstruction, not an independent model validation.",
        "limitations": ["Repeated cases and controller versions are correlated; pooled frames and episode counts are descriptive, not independent trials or confidence intervals.",
            "Different controller trajectories produce different state/error distributions; no causal controller comparison is implied.",
            "Raw estimates can be noisier even when less biased. One-step state accuracy does not establish eight-second trajectory prediction or improved episode outcomes.",
            "Plant randomization scale records configuration; this analysis does not identify per-vessel dynamics or attribute all observer error to randomization.",
            "Truth is used only to score saved states; no runtime observer is modified or calibrated."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output / "episodes.json").write_text(json.dumps(episodes, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["episode", "run", "case", "step", "fresh", "why", "changed"] + [f"{method}_{c}" for method in ("truth", *METHODS) for c in COMPONENTS]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in frames:
            line = {key: row[key] for key in fields if key in row}
            for method in ("truth", *METHODS):
                for j, component in enumerate(COMPONENTS):
                    line[f"{method}_{component}"] = None if row[method] is None else row[method][j]
            writer.writerow(line)
    print(json.dumps({key: summary[key] for key in ["episodes", "unique_cases", "frames", "fresh_frames", "stale_frames", "pooled_fresh", "episode_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
