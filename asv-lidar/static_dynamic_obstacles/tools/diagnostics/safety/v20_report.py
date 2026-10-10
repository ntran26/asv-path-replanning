"""Validate and pair V20-phase runs; never imports a simulator.

    python -B tools/diagnostics/safety/v20_report.py RUN_DIR [RUN_DIR ...] --output NEW_DIR [--title TEXT]

Every run directory must be complete: manifest, evaluated source archive whose
hashes match the manifest, one attempt and one result per planned (case, mode),
a nonempty trace per result whose step count equals the result's, a clean
completion record (no source drift, unchanged checkpoint and selection). Runs
must share the checkpoint, config, constants and runtime environment, and the
V20 module hashes when V20 modes are present. Records are paired by case ID,
episode seed and scenario digest; a case/mode pair present in two runs is
refused rather than chosen. Rescue and loss are always relative to the fresh
OFF record of the same scene; changing contact type is never a rescue.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUTCOMES = ("goal", "collision:obstacle", "collision:boundary", "collision:target", "timeout")
SHARED_KEYS = ("checkpoint_sha256", "config_sha256", "constants")
V20_MODULES = ("src/safety_v20.py", "src/safety_v20_contingency.py", "src/safety_v20_tubes.py")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_run(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    completion = json.loads((directory / "completion.json").read_text())
    require(completion["completed_runs"] == manifest["planned_runs"], f"{directory}: incomplete run")
    require(not completion["source_drift"], f"{directory}: source drift")
    require(completion["checkpoint_unchanged"] and completion["selection_unchanged"], f"{directory}: identity changed")
    with zipfile.ZipFile(directory / "evaluated_sources.zip") as archive:
        for name, digest in manifest["source_sha256"].items():
            require(sha(archive.read(name)) == digest, f"{directory}: archive mismatch {name}")
    cases = {c["case"]: c for c in manifest["cases"]}
    results = {}
    for path in sorted((directory / "attempts").glob("[0-9][0-9][0-9]_result.json")):
        row = json.loads(path.read_text())
        attempt = json.loads((directory / "attempts" / f"{row['attempt']:03d}.json").read_text())
        require(all(attempt[k] == row[k] for k in ("case", "mode", "seed")), f"{path}: attempt mismatch")
        case = cases[row["case"]]
        require(int(case["seed"]) == int(row["seed"]) and case["scenario_sha256"] == row["scenario_sha256"],
                f"{path}: identity mismatch")
        trace = directory / "traces" / f"{row['attempt']:03d}_{row['mode']}.jsonl"
        if not trace.exists():
            trace = trace.with_name(trace.name + ".gz")
        lines = read_lines(trace)
        require(len(lines) == int(row["steps"]), f"{trace}: trace/result step mismatch")
        key = (row["case"], row["mode"])
        require(key not in results, f"{directory}: duplicate {key}")
        row["_trace"] = trace
        row["_directory"] = directory
        results[key] = row
    require(len(results) == manifest["planned_runs"], f"{directory}: result count mismatch")
    return manifest, results


def read_lines(trace: Path):
    if trace.suffix == ".gz":
        with gzip.open(trace, "rt", encoding="utf-8") as stream:
            return stream.read().splitlines()
    return trace.read_text(encoding="utf-8").splitlines()


def trace_metrics(trace: Path):
    levels, sources, seconds, oocs, certified, decisions = Counter(), Counter(), [], [], 0, 0
    first_v20_change = None
    tail = []                      # (step, level, contract violations) of the last decisions
    for line in read_lines(trace):
        rec = json.loads(line)
        f = rec["filter"] or {}
        decisions += 1
        level = f.get("v20_level")
        if level is None:
            continue
        tail = (tail + [(rec["step"], level, f.get("v20_contract_violations"))])[-3:]
        levels[level] += 1
        if f.get("v20_seconds") is not None:
            seconds.append(float(f["v20_seconds"]))
        if level.startswith("out_of_contract"):
            oocs.append(rec["step"])
        if level in ("v16_unchanged_certified", "replaced_by_sac", "replaced_by_v16_certified",
                     "replaced_by_projection", "committed_continuation", "gatekeeper_replaced"):
            certified += 1
        if first_v20_change is None and str(f.get("why", "")).startswith("v20 "):
            first_v20_change = rec["step"]
    return dict(levels=dict(levels), certified_decisions=certified, decisions=decisions,
                out_of_contract_steps=oocs, first_v20_command_change=first_v20_change,
                v20_seconds=seconds, last_levels=tail)


def commands(trace: Path):
    out = []
    for line in read_lines(trace):
        rec = json.loads(line)
        out.append((round(float(rec["rudder_command"]), 9), round(float(rec["signed_rpm_command"]), 9)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="V20 paired development report")
    args = parser.parse_args()
    require(not args.output.exists(), "Existing report directory; never overwritten")
    manifests, results = [], {}
    for directory in args.runs:
        manifest, rows = load_run(directory.resolve())
        manifests.append((directory, manifest))
        for key, row in rows.items():
            require(key not in results, f"{key} present in two runs")
            results[key] = row
    base = manifests[0][1]
    for directory, manifest in manifests[1:]:
        for key in SHARED_KEYS:
            require(manifest[key] == base[key], f"{directory}: {key} differs")
        require(manifest["runtime_environment"]["packages"] == base["runtime_environment"]["packages"],
                f"{directory}: package environment differs")
        require(manifest["runtime_environment"]["python"] == base["runtime_environment"]["python"],
                f"{directory}: interpreter differs")
    v20_hashes = {m["source_sha256"].get(name) for _, m in manifests if any(x.startswith("v20") for x in m["modes"])
                  for name in V20_MODULES}
    v20_sets = {tuple(m["source_sha256"].get(name) for name in V20_MODULES)
                for _, m in manifests if any(x.startswith("v20") for x in m["modes"])}
    require(len(v20_sets) <= 1, "V20 module sources differ between runs")
    modes = sorted({mode for _, mode in results}, key=lambda m: (m != "off", m != "v16", m))
    cases = sorted({case for case, _ in results})
    identities = {}
    for (case, mode), row in results.items():
        ident = (int(row["seed"]), row["scenario_sha256"])
        require(identities.setdefault(case, ident) == ident, f"{case}: seed/digest differs between modes")
    paired_rows, per_mode = [], {m: Counter() for m in modes}
    pair_stats = defaultdict(Counter)
    transitions = defaultdict(Counter)
    for case in cases:
        row = {"case": case, "seed": identities[case][0], "scenario_sha256": identities[case][1]}
        for mode in modes:
            r = results.get((case, mode))
            row[f"{mode}_outcome"] = None if r is None else r["outcome"]
            row[f"{mode}_steps"] = None if r is None else r["steps"]
            row[f"{mode}_interventions"] = None if r is None else r.get("safety_v2_steps")
            if r is not None:
                per_mode[mode][r["outcome"]] += 1
                per_mode[mode]["episodes"] += 1
                if mode.startswith("v20"):
                    m = trace_metrics(r["_trace"])
                    row[f"{mode}_certified_share"] = m["certified_decisions"] / max(1, m["decisions"])
                    row[f"{mode}_out_of_contract_decisions"] = len(m["out_of_contract_steps"])
                    row[f"{mode}_first_v20_command_change"] = m["first_v20_command_change"]
                    row[f"{mode}_levels"] = json.dumps(m["levels"], sort_keys=True)
                    row[f"{mode}_last_levels"] = json.dumps(m["last_levels"])
                    per_mode[mode]["out_of_contract_episodes"] += bool(m["out_of_contract_steps"])
                    per_mode[mode]["decisions"] += m["decisions"]
                    per_mode[mode]["certified_decisions"] += m["certified_decisions"]
                    for level, n in m["levels"].items():
                        per_mode[mode][f"level:{level}"] += n
                    if (case, "v16") in results:
                        a, b = commands(results[(case, "v16")]["_trace"]), commands(r["_trace"])
                        first = next((i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y),
                                     None if len(a) == len(b) else min(len(a), len(b)) + 1)
                        row[f"{mode}_first_command_difference_vs_v16"] = first
                        per_mode[mode]["identical_to_v16"] += first is None
        for ref in ("off", "v16"):
            ra = results.get((case, ref))
            for mode in modes:
                if mode == ref or (ref == "v16" and mode == "off"):
                    continue
                rb = results.get((case, mode))
                if ra is None or rb is None:
                    continue
                key = f"{mode}_vs_{ref}"
                s = pair_stats[key]
                s["pairs"] += 1
                ga, gb = ra["outcome"] == "goal", rb["outcome"] == "goal"
                s["reference_goals"] += ga
                s["candidate_goals"] += gb
                s["gained_goals"] += (not ga) and gb
                s["lost_goals"] += ga and not gb
                s["both_goal"] += ga and gb
                s["both_fail"] += (not ga) and (not gb)
                if ra["outcome"] != rb["outcome"]:
                    transitions[key][f"{ra['outcome']} -> {rb['outcome']}"] += 1
        paired_rows.append(row)
    args.output.mkdir(parents=True)
    fields = sorted({k for r in paired_rows for k in r}, key=lambda k: (k not in ("case", "seed", "scenario_sha256"), k))
    with (args.output / "paired.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_rows)
    seconds = [s for (case, mode), r in results.items() if mode.startswith("v20")
               for s in trace_metrics(r["_trace"])["v20_seconds"]]
    summary = {
        "schema": 1, "title": args.title, "runs": [str(d) for d, _ in manifests],
        "run_manifest_sha256": {str(d): sha((Path(d) / "manifest.json").read_bytes()) for d, _ in manifests},
        "cases": len(cases), "modes": modes,
        "per_mode": {m: dict(c) for m, c in per_mode.items()},
        "pairs": {k: dict(v) for k, v in pair_stats.items()},
        "transitions": {k: dict(v) for k, v in transitions.items()},
        "v20_decision_seconds": None if not seconds else {
            "p50": sorted(seconds)[len(seconds) // 2], "p95": sorted(seconds)[int(0.95 * (len(seconds) - 1))],
            "max": max(seconds), "decisions": len(seconds)},
        "runtime_environment": {k: base["runtime_environment"][k] for k in ("python", "platform")},
        "script_sha256": sha(Path(__file__).read_bytes()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    lines = [f"# {args.title}", "",
             f"{len(cases)} scenes; fresh runs paired by case ID, episode seed and scenario digest. "
             "Rescue/loss are relative to the fresh OFF record of the same scene.", "",
             "| Mode | Episodes | Goals | Obstacle | Boundary | Target | Timeout |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in modes:
        c = per_mode[m]
        lines.append(f"| {m} | {c['episodes']} | {c['goal']} | {c['collision:obstacle']} | "
                     f"{c['collision:boundary']} | {c['collision:target']} | {c['timeout']} |")
    lines += ["", "| Comparison | Pairs | Reference goals | Candidate goals | Gained | Lost |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for k, s in sorted(pair_stats.items()):
        lines.append(f"| {k} | {s['pairs']} | {s['reference_goals']} | {s['candidate_goals']} | "
                     f"{s['gained_goals']} | {s['lost_goals']} |")
    lines += ["", "Outcome transitions (reference -> candidate):", ""]
    for k, t in sorted(transitions.items()):
        lines.append(f"- {k}: " + "; ".join(f"{a} x{n}" for a, n in sorted(t.items())))
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
