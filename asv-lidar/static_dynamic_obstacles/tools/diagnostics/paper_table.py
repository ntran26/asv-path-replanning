"""The paper's results tables, in the Paper 2 template (2026-10-08).

One column per method (training seeds pooled), one row per metric: a point estimate
with a stratified bootstrap 95 % interval (stratified by test-set cell, pooled over
runs, 5,000 resamples). Rate metrics (*) are means of per-episode 0/1 outcomes;
continuous metrics are interquartile means (IQM), because a mean of a binary rate is
what a rate is, while IQM resists the long tails of truncated episodes. Definitions
follow Paper 2 (`src/metrics.py`), plus target-ship metrics. Written for test set v4,
safety layer off, scoped to all episodes, the decoupled family and the coupled family.

A second table gives paired statistics against the first method on the same episodes:
McNemar (exact) on success, Wilcoxon signed-rank on RMS cross-track error and on the
minimum hull-to-hull clearance to the target, over all paired episodes and over those
both methods completed.

    python tools/diagnostics/paper_table.py
    python tools/diagnostics/paper_table.py --methods "SAC=sacs0_bl3_metrics,sacs1_bl3_metrics;PPO=ppos0_bl3_metrics"

Needs the per-episode metrics `tools/tiers/common.py` records since 2026-10-08.

No per-rule COLREGs rows (decision 2026-10-08): a compliance rate is not a fair measure
where the compliant manoeuvre would meet a static obstacle or the target does not keep
to the rules, and the paper already carries many metrics. COLREGs behaviour is shown by
the snapshot figure (`results/paper_snapshots`), whose cases are checked with
`tools/diagnostics/colregs_compliance.py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from metrics import iqm, stratified_bootstrap_ci  # noqa: E402

DEFAULT = ("SAC=sacs0_bl3_metrics;PPO=ppos0_bl3_metrics;RecurrentPPO=recurrent_ppos0_bl3_metrics;"
           "LOS-DWA=los_dwa;COLREGs-VO=colregs_vo")
SCOPES = (("All episodes", None), ("Decoupled", "frozen"), ("Coupled", "paper2"))

# (label, column or callable, kind, episode filter)
#   kind "rate": mean of 0/1; "iqm": interquartile mean
#   filter: None = every episode in scope; "target" = target present; "success" = reached the goal;
#           "obstacles" = at least one static obstacle; or a callable on the frame returning a mask
ROWS = [
    ("**Success rate** *", lambda d: d.outcome == "goal", "rate", None),
    ("Target collision rate *", lambda d: d.outcome == "collision:target", "rate", None),
    ("Obstacle collision rate *", lambda d: d.outcome == "collision:obstacle", "rate", None),
    ("Boundary collision rate *", lambda d: d.outcome == "collision:boundary", "rate", None),
    ("Timeout rate *", lambda d: d.outcome == "timeout", "rate", None),
    ("RMS cross-track error (m)", "rms_cte", "iqm", None),
    ("Completion time, successful episodes (s)", "completion_time_s", "iqm", "success"),
    ("Path length / reference, successful episodes", "path_efficiency", "iqm", "success"),
    ("Mean speed (m/s)", "mean_speed", "iqm", None),
    ("Minimum speed (m/s)", "min_speed", "iqm", None),
    ("Min obstacle clearance (m)", "min_obstacle_clearance", "iqm", "obstacles"),
    ("Min boundary clearance (m)", "min_boundary_clearance", "iqm", None),
    ("*Target ship*", None, "header", None),
    ("Ship-domain intrusion rate *", lambda d: d.domain_intrusion_max > 0.0, "rate", "target"),
    ("Deepest intrusion, episodes with one (fraction of radius)", "domain_intrusion_max", "iqm", "intruded"),
    ("Time inside the domain, episodes with an intrusion (s)", "domain_time_s", "iqm", "intruded"),
    ("Closest approach, centre to centre (m)", "min_target_range", "iqm", "target"),
    ("Min target clearance, hull to hull (m)", "min_target_clearance", "iqm", "target"),
    ("*Actuation*", None, "header", None),
    ("Rudder saturation fraction *", "rudder_saturation_fraction", "rate", None),
    ("Mean abs. rudder rate (deg/s)", "mean_abs_rudder_rate", "iqm", None),
    ("Control effort (int. sq. rudder cmd)", "control_effort", "iqm", None),
]


def load(tags):
    frames = []
    for t in tags:
        p = ROOT / "results" / "test_set" / "v4" / t / "episodes.csv"
        d = pd.read_csv(p, keep_default_na=False)
        d = d[d.safety == "off"].copy()
        d["run"] = t
        for c in ("min_obstacle_clearance", "min_target_range", "min_target_clearance", "domain_intrusion_max"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[(d["class"] == "") & d.cell.str.contains("null"), "class"] = "null"   # rows reused from v1-v3
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["has_target"] = d["class"].astype(str) != "no_target"
    d.loc[d["min_target_range"] == np.inf, "has_target"] = False
    return d


def subset(d, flt):
    if callable(flt):
        return d[flt(d)]
    if flt == "target":
        return d[d.has_target]
    if flt == "success":
        return d[d.outcome == "goal"]
    if flt == "intruded":
        return d[d.has_target & (d.domain_intrusion_max > 0.0)]
    if flt == "obstacles":
        return d[np.isfinite(d.min_obstacle_clearance)]
    return d


def cell(d, col, kind, flt, n_boot, rng):
    s = subset(d, flt)
    vals = (col(s) if callable(col) else s[col]).astype(float).to_numpy()
    strata = s["cell"].to_numpy()
    keep = np.isfinite(vals)
    vals, strata = vals[keep], strata[keep]
    if vals.size == 0:
        return "—", float("nan"), float("nan"), float("nan")
    stat = np.mean if kind == "rate" else iqm
    point = float(stat(vals))
    lo, hi = stratified_bootstrap_ci(vals, strata, stat, n_boot, rng=rng)
    fmt = "{:.3f}" if kind == "rate" or abs(point) < 10 else "{:.1f}"
    return f"{fmt.format(point)} [{fmt.format(lo)}, {fmt.format(hi)}]", point, lo, hi


def paired(a, b, label_a, label_b, scope_name):
    from scipy.stats import binomtest, wilcoxon
    a = a.set_index("test_id")
    b = b.set_index("test_id").reindex(a.index)
    sa, sb = (a.outcome == "goal"), (b.outcome == "goal")
    only_a, only_b = int((sa & ~sb).sum()), int((~sa & sb).sum())
    p_mc = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    out = []
    for scope, mask in (("all paired", np.ones(len(a), bool)), ("both succeeded", (sa & sb).to_numpy())):
        row = {"Comparison": f"{label_a} vs {label_b}", "Episodes": scope_name, "Scope": scope, "n": int(mask.sum()),
               "Success A": f"{sa.mean():.3f}", "Success B": f"{sb.mean():.3f}",
               "McNemar only A / only B": f"{only_a} / {only_b}", "McNemar p": f"{p_mc:.3g}"}
        for col, name in (("rms_cte", "RMS CTE"), ("min_target_clearance", "Target clearance")):
            x = a.loc[mask, col].astype(float).to_numpy()
            y = b.loc[mask, col].astype(float).to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if x.size > 10 and np.any(x != y):
                hl = float(np.median(x - y))                    # median paired difference
                row[f"{name} median A / B"] = f"{np.median(x):.3f} / {np.median(y):.3f}"
                row[f"{name} median diff"] = f"{hl:+.3f}"
                row[f"{name} Wilcoxon p"] = f"{wilcoxon(x, y).pvalue:.3g}"
            else:
                row[f"{name} median A / B"], row[f"{name} median diff"], row[f"{name} Wilcoxon p"] = "—", "—", "—"
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default=DEFAULT, help='"Name=tag[,tag...];Name=tag..." (seeds pooled per method)')
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "baseline_v3" / "paper_table")
    args = ap.parse_args()
    methods = []
    for part in args.methods.split(";"):
        name, tags = part.split("=", 1)
        methods.append((name.strip(), [t.strip() for t in tags.split(",") if t.strip()]))
    data = {name: load(tags) for name, tags in methods}
    rng = np.random.default_rng(12345)

    L = ["# Results on test set v4 (safety layer off)", "",
         "Point estimate with a stratified bootstrap 95 % interval (stratified by test-set cell, pooled "
         f"over runs, {args.boot:,} resamples). Rate metrics (marked *) are means of per-episode 0/1 "
         "outcomes; continuous metrics are interquartile means (IQM). Each run is represented by its "
         "best validation-set checkpoint. Definitions follow Paper 2 (`src/metrics.py`).", "",
         "| Method | Runs | Episodes per scope (all / decoupled / coupled) |", "|---|---|---|"]
    for name, tags in methods:
        d = data[name]
        L.append(f"| {name} | {', '.join(t.replace('_metrics', '') for t in tags)} | "
                 f"{len(d)} / {int((d.source == 'frozen').sum())} / {int((d.source == 'paper2').sum())} |")
    L.append("")
    csv_rows = []
    for scope_name, src in SCOPES:
        L += [f"## {scope_name}", "", "| Metric | " + " | ".join(n for n, _ in methods) + " |",
              "|---|" + "---|" * len(methods)]
        for label, col, kind, flt in ROWS:
            if kind == "header":
                L.append(f"| {label} |" + " |" * len(methods))
                continue
            cells = []
            for name, _ in methods:
                d = data[name] if src is None else data[name][data[name].source == src]
                text, point, lo, hi = cell(d, col, kind, flt, args.boot, rng)
                cells.append(text)
                csv_rows.append({"scope": scope_name, "metric": label.replace("*", "").strip(), "method": name,
                                 "kind": kind, "point": point, "ci_lo": lo, "ci_hi": hi})
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Notes: completion time and path length are over successful episodes only, because a "
          "collision truncates an episode; RMS cross-track error is over all episodes, as in Paper 2, "
          "so a collision can flatter it. Target-ship rows cover the episodes with a target. The ship "
          "domain is own ship's (3.14 m ahead, 1.57 m astern, 1.25 m abeam), entered by the target's "
          "centre; intrusion depth and time are over the episodes with an intrusion, since most have none "
          "and an IQM over all would read zero. COLREGs behaviour is shown by the snapshot figure, not by "
          "per-rule rates. "
          "The classical controllers are deterministic, so their intervals reflect episode variance only. "
          "Clearances are from the hull polygon with its 0.15 m margin. Path length "
          "below 1 reflects the goal test (within 1.25 m of the path end) and cut corners.", ""]

    ref_name, _ = methods[0]
    prows = []
    for name, _ in methods[1:]:
        for scope_name, src in SCOPES:
            a = data[ref_name] if src is None else data[ref_name][data[ref_name].source == src]
            b = data[name] if src is None else data[name][data[name].source == src]
            prows += paired(a, b, ref_name, name, scope_name)
    if prows:
        cols = list(prows[0].keys())
        L += ["## Paired statistics", "",
              f"Against {ref_name}, same episodes (first run of each method). McNemar (exact) on success; "
              "Wilcoxon signed-rank on RMS cross-track error and on the minimum hull-to-hull clearance to the "
              "target. `both succeeded` restricts to episodes both methods completed.", "",
              "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in prows:
            L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        L.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(args.out.with_suffix(".csv"), index=False)
    if prows:
        pd.DataFrame(prows).to_csv(args.out.parent / (args.out.name + "_paired.csv"), index=False)
    print("\n".join(L))
    print(f"-> {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
