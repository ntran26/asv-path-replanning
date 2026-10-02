"""Plot descriptive suite outcomes from suite_compare's *_outcomes.csv.

python -B tools/diagnostics/safety/plot_suite_outcomes.py results/safety_dev/final_suites_outcomes.csv --tag final_suites_groups

Use --level component for individual suites. Partial reports require an explicit
--allow-partial and carry a prominent watermark. No confidence intervals or
independence assumptions are introduced for paired robustness cases.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

from suite_compare import FULL_BENCHMARK_COUNTS, FULL_BENCHMARK_MODES, GROUPS
from suite_status import read_shared_bytes

ROOT = Path(__file__).resolve().parents[3]
FIELDS = ("goals", "collision_obstacle", "collision_boundary", "collision_target", "timeouts")
LABELS = ("Goal", "Obstacle", "Boundary", "Target", "Timeout")
COLORS = ("#009E73", "#E69F00", "#D55E00", "#CC79A7", "#7C858D")
MODE_LABELS = {"off": "Policy only", "v4": "Selected v4", "v5": "Frozen v5"}
SCOPE_LABELS = {"dev": "Development", "frozen": "Frozen suites", "field": "Simulated field sets", "all": "All supplied suites",
    "dev_field": "Field development (simulation)", "dev_legacy": "Original development", "dev_width": "Head-on width development",
    "frozen_a": "Frozen A: realised named cases", "frozen_b": "Frozen B: headline", "frozen_r": "Frozen R: robustness",
    "field_deployment": "Simulated deployment layouts", "field_validation": "Simulated field validation"}
SCOPE_ORDER = tuple(SCOPE_LABELS)


def load_rows(path, level, modes, allow_partial=False, *, data=None):
    """Validate counts rather than trusting rates rounded by another reader."""
    selected = {}
    data = read_shared_bytes(path) if data is None else data
    with io.StringIO(data.decode("utf-8-sig")) as handle:
        for source in csv.DictReader(handle):
            if source["level"] != level or source["mode"] not in modes:
                continue
            row = dict(source)
            for name in (*FIELDS, "collisions", "completed_cases", "expected_cases"):
                row[name] = int(source[name])
                if row[name] < 0:
                    raise ValueError(f"Negative count for {row['scope']}/{row['mode']}: {name}")
            n, total = row["completed_cases"], row["expected_cases"]
            if sum(row[field] for field in FIELDS) != n:
                raise ValueError(f"Outcome counts do not sum to completed cases: {row['scope']}/{row['mode']}")
            if sum(row[field] for field in FIELDS[1:4]) != row["collisions"] or n > total:
                raise ValueError(f"Inconsistent collision counts or denominator: {row['scope']}/{row['mode']}")
            if source["complete"].lower() not in ("true", "false"):
                raise ValueError("Expected complete=True or False")
            row["complete"] = source["complete"].lower() == "true"
            if row["complete"] != (n == total):
                raise ValueError(f"Inconsistent completeness flag: {row['scope']}/{row['mode']}")
            key = (row["scope"], row["mode"])
            if key in selected:
                raise ValueError(f"Duplicate scope/mode row: {key}")
            selected[key] = row
    if not selected:
        raise ValueError(f"No selected modes at level {level}")
    scopes = sorted({key[0] for key in selected}, key=lambda name:
                    (SCOPE_ORDER.index(name) if name in SCOPE_ORDER else len(SCOPE_ORDER), name))
    partial = any(not row["complete"] for row in selected.values())
    partial |= any((scope, mode) not in selected for scope in scopes for mode in modes)
    if partial and not allow_partial:
        raise ValueError("Incomplete or missing mode results; pass --allow-partial to label a progress plot")
    return scopes, selected, partial


def require_full_benchmark_rows(path, modes, data):
    """Check all component counts and their aggregate projections in one CSV snapshot."""
    if set(modes) != set(FULL_BENCHMARK_MODES):
        raise ValueError("Full benchmark requires modes off,v4,v5")
    _, components, _ = load_rows(path, "component", modes, data=data)
    required = {(component, mode): count for component, count in FULL_BENCHMARK_COUNTS.items()
                for mode in FULL_BENCHMARK_MODES}
    if {key: row["completed_cases"] for key, row in components.items()} != required:
        raise ValueError("Full benchmark plot requires all eight components and exactly 8,670 committed episodes")
    for level in ("group", "all"):
        _, aggregate_rows, _ = load_rows(path, level, modes, data=data)
        expected_scopes = set(GROUPS.values()) if level == "group" else {"all"}
        if set(aggregate_rows) != {(scope, mode) for scope in expected_scopes for mode in modes}:
            raise ValueError("Full benchmark aggregate scopes are incomplete")
        for (scope, mode), row in aggregate_rows.items():
            members = [record for (component, m), record in components.items()
                       if m == mode and (level == "all" or GROUPS[component] == scope)]
            for field in (*FIELDS, "completed_cases", "expected_cases", "collisions"):
                if row[field] != sum(record[field] for record in members):
                    raise ValueError(f"Aggregate {scope}/{mode}/{field} disagrees with component counts")


def make_figure(scopes, rows, modes, partial, level, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    height = max(3.5, 1.65 + len(scopes) * (len(modes) + 1) * 0.35)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        figure = plt.figure(figsize=(11.6, height))
        bars = figure.add_axes([0.145, 0.15, 0.43, 0.65])
        counts = figure.add_axes([0.615, 0.15, 0.365, 0.65])
        ymax = len(scopes) * (len(modes) + 1) - 0.3
        for axis in (bars, counts):
            axis.set_ylim(ymax, -0.25)
        bars.set_xlim(0, 100)
        bars.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
        bars.set_xlabel("Proportion of completed episodes (%)")
        bars.grid(axis="x", color="#E3E6E9", linewidth=0.65, zorder=0)
        bars.tick_params(axis="y", length=0, pad=8)
        for spine in ("top", "right", "left"):
            bars.spines[spine].set_visible(False)
        bars.spines["bottom"].set_color("#999999")
        counts.set_xlim(-0.5, 5.85)
        counts.axis("off")
        for x, label in enumerate((*LABELS, "n / N")):
            counts.text(x if x < 5 else 5.25, 1.025, label, transform=counts.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, color=COLORS[x] if x < 5 else "#333333",
                        weight="bold")
        positions, mode_labels = [], []
        for group_index, scope in enumerate(scopes):
            base = group_index * (len(modes) + 1)
            bars.text(0, base + 0.05, SCOPE_LABELS.get(scope, scope), weight="bold", fontsize=10, va="center")
            for mode_index, mode in enumerate(modes):
                y = base + mode_index + 1
                positions.append(y)
                mode_labels.append(MODE_LABELS.get(mode, mode))
                row = rows.get((scope, mode))
                n = row["completed_cases"] if row else 0
                if row and n:
                    left = 0.0
                    for field, color in zip(FIELDS, COLORS):
                        width = 100.0 * row[field] / n
                        bars.barh(y, width, left=left, height=0.63, color=color, edgecolor="white", linewidth=0.45, zorder=3)
                        left += width
                else:
                    bars.barh(y, 100, height=0.63, facecolor="#F1F2F3", edgecolor="#BBBBBB", hatch="///", linewidth=0.4)
                    bars.text(50, y, "No completed episodes", ha="center", va="center", color="#555555", fontsize=8)
                for x, field in enumerate(FIELDS):
                    counts.text(x, y, str(row[field]) if row else "—", ha="center", va="center")
                denominator = f"{n:,} / {row['expected_cases']:,}" if row else "0 / ?"
                counts.text(5.25, y, denominator, ha="center", va="center", fontsize=8,
                            color="#9C2F19" if row is None or not row["complete"] else "#333333")
        bars.set_yticks(positions, mode_labels)
        title = title or "Simulated safety outcomes" + (" by component" if level == "component" else "")
        figure.suptitle(title, y=0.975, fontsize=15, weight="bold")
        if partial:
            figure.text(0.5, 0.918, "INCOMPLETE — progress snapshot; paired comparisons are not final",
                        ha="center", color="#A32913", fontsize=10, weight="bold")
            figure.text(0.41, 0.47, "INCOMPLETE", ha="center", va="center", rotation=25,
                        fontsize=42, color="#A32913", alpha=0.16)
        legend_y = 0.895 if partial else 0.925
        figure.legend(handles=[Patch(facecolor=color, label=label) for label, color in zip(LABELS, COLORS)],
                      loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=5, frameon=False)
        figure.text(0.145, 0.035,
                    "Simulation episodes only; n = completed, N = declared. Episode-weighted proportions; robustness twins remain paired.\n"
                    "Declared generator shortfalls are excluded from N. No confidence intervals; see the source report for coverage.",
                    fontsize=8, color="#555555", va="bottom")
        return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outcomes", type=Path)
    parser.add_argument("--tag", required=True, help="new output prefix under results/safety_dev")
    parser.add_argument("--level", choices=("group", "component", "all"), default="group")
    parser.add_argument("--modes", default="off,v4,v5")
    parser.add_argument("--title", help="optional figure title identifying the policy or experiment")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-full-benchmark", action="store_true", help="require all 8,670 episodes and consistent component/group totals")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in (".", "..") or "\\" in args.tag:
        parser.error("--tag must be a plain filename stem")
    modes = args.modes.split(",")
    if not all(modes) or len(set(modes)) != len(modes):
        parser.error("--modes must contain distinct nonempty labels")
    if args.allow_partial and args.require_full_benchmark:
        parser.error("--require-full-benchmark cannot be combined with --allow-partial")
    directory = ROOT / "results/safety_dev"
    paths = {extension: directory / f"{args.tag}.{extension}" for extension in ("png", "pdf", "svg", "json")}
    if any(path.exists() for path in paths.values()):
        parser.error("Plot output exists; use a new --tag")
    try:
        source_data = read_shared_bytes(args.outcomes)
        if args.require_full_benchmark:
            require_full_benchmark_rows(args.outcomes, modes, source_data)
        scopes, rows, partial = load_rows(args.outcomes, args.level, modes, args.allow_partial, data=source_data)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(directory / ".matplotlib"))
    metadata = {"input": str(args.outcomes.resolve()), "input_sha256": hashlib.sha256(source_data).hexdigest(),
                "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "level": args.level, "modes": modes, "title": args.title, "partial": partial, "allow_partial": args.allow_partial,
                "require_full_benchmark": args.require_full_benchmark,
                "statistics": "episode-weighted simulation counts and proportions; no confidence intervals", "rows": list(rows.values())}
    with paths["json"].open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    figure = make_figure(scopes, rows, modes, partial, args.level, args.title)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(paths[extension], dpi=220, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(figure)
    print(f"Wrote PNG, PDF, SVG and provenance: {directory / args.tag}; incomplete={partial}")


if __name__ == "__main__":
    main()
