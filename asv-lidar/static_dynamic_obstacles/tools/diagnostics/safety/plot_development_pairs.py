"""Plot completed development pairs from saved traces, without simulator imports.

Example:
  python -B tools/diagnostics/safety/plot_development_pairs.py --tag broad_v6_50 \
      --cases DV3-NT-CV-04 DV3-NT-CV-12

The same utility accepts later lost-goal cases. Geometry/state are recorded
simulator truth used only for offline illustration, never controller inputs.
Existing figures are not overwritten; use another --suffix for a new revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

from suite_status import read_shared_bytes

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "results/safety_dev/development_v6_budget150"
COLORS = {"v4": "#D55E00", "v6": "#0072B2"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_pair(directory, case):
    """Only finalized episode records and complete, contiguous traces qualify."""
    manifest_bytes = read_shared_bytes(directory / "manifest.json")
    manifest = json.loads(manifest_bytes)
    case_records = [c for c in manifest["configuration"]["cases"] if c["case"] == case]
    if len(case_records) != 1 or case_records[0]["suite"] != "dev_field":
        raise ValueError(f"Not a unique development case in manifest: {case}")
    expected = case_records[0]
    pair, inputs = {}, {"manifest.json": digest(manifest_bytes)}
    for path in sorted(directory.glob("episode_*.json")):
        # During an active campaign a newly created unrelated result may still
        # be incomplete. It cannot qualify as a selected completed pair.
        data = read_shared_bytes(path)
        try:
            row = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            continue
        if row.get("case") != case or row.get("mode") not in COLORS:
            continue
        mode = row["mode"]
        if mode in pair:
            raise ValueError(f"Duplicate completed case/mode: {case}/{mode}")
        if (row.get("run_manifest_sha256") != digest(manifest_bytes)
                or row.get("tag") != directory.name
                or row.get("suite") != "dev_field"
                or any(row.get(key) != expected[key] for key in ("seed", "scenario_sha256"))
                or path.name != f"episode_{row['attempt']:03d}.json"):
            raise ValueError(f"Completed result provenance mismatch: {path}")
        trace_path = directory / f"trace_{row['attempt']:03d}.jsonl"
        trace_bytes = read_shared_bytes(trace_path)
        traces = [json.loads(line) for line in trace_bytes.splitlines() if line.strip()]
        if (len(traces) != row["steps"] or not traces
                or [t["decision"] for t in traces] != list(range(1, len(traces) + 1))):
            raise ValueError(f"Incomplete/noncontiguous completed trace: {trace_path}")
        xy = np.asarray([[traces[0]["before"]["asv_x"], traces[0]["before"]["asv_y"]]]
                        + [[t["after"]["asv_x"], t["after"]["asv_y"]] for t in traces])
        if not np.isfinite(xy).all():
            raise ValueError(f"Non-finite recorded trajectory: {trace_path}")
        for previous, current in zip(traces, traces[1:]):
            if not np.allclose([previous["after"][key] for key in ("asv_x", "asv_y")],
                               [current["before"][key] for key in ("asv_x", "asv_y")],
                               atol=1e-10, rtol=0):
                raise ValueError(f"Discontinuous recorded trajectory: {trace_path}")
        pair[mode] = {"episode": row, "xy": xy,
                      "geometry": traces[0]["static_geometry_at_first_decision"]}
        inputs[path.name], inputs[trace_path.name] = digest(data), digest(trace_bytes)
    if set(pair) != set(COLORS):
        raise ValueError(f"Both v4 and v6 need finalized results/traces: {case}")
    if pair["v4"]["geometry"] != pair["v6"]["geometry"]:
        raise ValueError("Paired recorded static geometry differs")
    if not np.allclose(pair["v4"]["xy"][0], pair["v6"]["xy"][0], atol=1e-10, rtol=0):
        raise ValueError("Paired initial positions differ")
    return pair, inputs


def plot_pair(directory, case, output_directory, suffix=""):
    pair, inputs = load_pair(directory, case)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{directory.name}_{case}{'_' + suffix if suffix else ''}"
    paths = {ext: output_directory / f"{stem}.{ext}" for ext in ("png", "pdf", "json")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Figure exists; select a new --suffix: {stem}")
    a, b = (pair[mode]["episode"]["outcome"] for mode in ("v4", "v6"))
    comparison = ("rescue" if b == "goal" and a != "goal" else
                  "lost goal" if a == "goal" and b != "goal" else "comparison")
    geometry = pair["v4"]["geometry"]
    boundary = np.asarray(geometry["boundary_polygon_m"], dtype=float)
    obstacles = [np.asarray(points, dtype=float) for points in geometry["obstacle_polygons_m"]]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(6.1, 8.8))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.19, top=0.88)
        ax.add_patch(Polygon(boundary, closed=True, facecolor="#F7F9FA",
                             edgecolor="#555555", linewidth=1.3, zorder=0))
        for obstacle in obstacles:
            ax.add_patch(Polygon(obstacle, closed=True, facecolor="#B4B7BA",
                                 edgecolor="#4D5154", linewidth=0.8, zorder=2))
        handles = []
        for mode in ("v4", "v6"):
            record, points = pair[mode]["episode"], pair[mode]["xy"]
            style = "--" if mode == "v4" else "-"
            ax.plot(points[:, 0], points[:, 1], color=COLORS[mode], linestyle=style,
                    linewidth=2.0, zorder=3 if mode == "v4" else 4)
            endpoint = "*" if record["outcome"] == "goal" else "X" if record["outcome"].startswith("collision:") else "s"
            ax.scatter(*points[-1], marker=endpoint, s=125, color=COLORS[mode],
                       edgecolor="white", linewidth=0.65, zorder=6)
            handles.append(Line2D([0], [0], color=COLORS[mode], linestyle=style,
                                  marker=endpoint, linewidth=2,
                                  label=f"{mode.upper()}: {record['outcome']} ({record['steps']} steps)"))
            # Direction arrows follow recorded displacement, not vessel heading.
            for fraction in (0.30, 0.65):
                index = min(len(points)-2, int(fraction * (len(points)-1)))
                delta = points[index+1] - points[index]
                if np.linalg.norm(delta) > 1e-8:
                    direction = delta / np.linalg.norm(delta)
                    centre = points[index]
                    ax.annotate("", xy=centre+0.32*direction, xytext=centre-0.20*direction,
                                arrowprops=dict(arrowstyle="-|>", color=COLORS[mode], lw=1.4), zorder=5)
        start = pair["v4"]["xy"][0]
        ax.scatter(*start, color="#222222", marker="o", s=38, edgecolor="white", linewidth=0.6, zorder=7)
        ax.annotate("Start", start, xytext=(6, -12), textcoords="offset points", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        all_points = np.vstack([boundary] + [pair[m]["xy"] for m in COLORS])
        ax.set_xlim(float(all_points[:, 0].min())-0.7, float(all_points[:, 0].max())+0.7)
        ax.set_ylim(float(all_points[:, 1].min())-0.5, float(all_points[:, 1].max())+0.5)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, color="#DFE3E6", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        fig.suptitle(f"{case}\nIllustrative development {comparison}", y=0.975, fontsize=13)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.064),
                   frameon=False, fontsize=9, ncol=1)
        fig.text(0.5, 0.032, "Recorded simulated truth; identical static geometry and reset seed.\n"
                 "Lines show vessel centre; end markers show goal (*) or contact (X).\n"
                 "Selected example, not aggregate evidence of improvement.",
                 ha="center", va="center", fontsize=8, color="#444444")
        fig.savefig(paths["png"], dpi=220, facecolor="white")
        fig.savefig(paths["pdf"], facecolor="white", metadata={"Title": f"{case}: illustrative development {comparison}"})
        plt.close(fig)
    provenance = {
        "tag": directory.name, "case": case, "illustration_type": comparison,
        "scope": "Offline stored-trace illustration; no simulator construction/reset/step. Selected case, not aggregate proof.",
        "coordinate_frame": "Recorded simulator world coordinates, metres; equal x/y scale.",
        "seed": pair["v4"]["episode"]["seed"],
        "scenario_sha256": pair["v4"]["episode"]["scenario_sha256"],
        "episodes": {mode: {key: pair[mode]["episode"][key] for key in
                              ("attempt", "outcome", "steps", "run_manifest_sha256")} for mode in COLORS},
        "input_directory": str(directory.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": inputs,
        "plotter_sha256": digest(Path(__file__).read_bytes()),
        "output_sha256": {ext: digest(paths[ext].read_bytes()) for ext in ("png", "pdf")},
    }
    with paths["json"].open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="broad_v6_50")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in [args.tag, *args.cases]):
        raise ValueError("Tag/case names must contain only letters, digits, underscores and hyphens")
    if args.suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", args.suffix):
        raise ValueError("Invalid figure suffix")
    for case in args.cases:
        paths = plot_pair(CAMPAIGN / "runs" / args.tag, case, CAMPAIGN / "figures", args.suffix)
        print(paths["png"])


if __name__ == "__main__":
    main()
