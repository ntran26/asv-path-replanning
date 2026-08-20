"""Build the manuscript tables and figures from the per-episode CSVs.

    python src/make_outputs.py --all

Sub-steps can be run individually: `--tables`, `--learning-curve`,
`--trajectories`.

Outputs land in `eval_results/baselines/`:

    comparison_table.csv / .md      headline metrics per method
    paired_stats_table.csv / .md    SAC vs PPO and SAC vs LOS+APF
    figures/learning_curves.{png,svg,csv}
    figures/trajectories.{png,svg}

Reporting conventions
---------------------
* The learned methods are reported as **IQM with stratified bootstrap 95 % CIs**
  across seeds and episodes.  The classical baseline is deterministic -- one
  fixed parameter set, no sampling, no seed -- so it gets point values and the
  table says so rather than implying a spread that does not exist.
* Figures use **distinct line styles and markers, not colour alone**, per the
  reviewer comment about an existing figure.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

BASE = os.path.join("eval_results", "baselines")
FIG_DIR = os.path.join(BASE, "figures")

# Methods in report order: (label, glob of per-episode CSVs, deterministic?)
#
# PPO gets two rows on purpose.  `final @1M` is the strictly budget-matched
# checkpoint -- the same "model at 1M steps" that SAC is reported from -- and it
# is the headline number.  `best checkpoint` is the peak the run actually
# reached, selected by the same score SAC's training callback used.  Reporting
# only the first would look like a straw man; reporting only the second would
# advantage PPO over SAC asymmetrically.  Both, with the collapse explained, is
# the honest presentation.  See BASELINES_RESULTS.md.
#
# "SAC (published)" is the single manuscript checkpoint and is kept separate
# from "SAC (retrained)", which is three from-scratch seeds under the same
# protocol as PPO.  They are different objects and must not be pooled: the
# published model carries a long history of resumed, hand-staged runs that the
# retrains do not.
#
# LOS+APF is deterministic *as a controller* -- running it again reproduces the
# CSV byte for byte.  Its three runs come from three independent 250-config
# random searches under different search seeds, so its interval reflects
# tuning-procedure variance, which is the correct analogue of a training seed.
METHOD_SPECS: List[Tuple[str, str, bool]] = [
    ("SAC (published)", os.path.join(BASE, "sac_1M", "episodes.csv"), False),
    ("PPO (final @1M)", os.path.join(BASE, "ppo_fx_seed*_final", "episodes.csv"), False),
    ("LOS+APF (tuned)", os.path.join(BASE, "los_apf_s*", "episodes.csv"), False),
]

# (key, label, format, statistic)
#
# `rate` metrics are per-episode 0/1 outcomes, so the **mean** is the estimand --
# it is the success/collision rate itself.  IQM is meaningless on a binary
# variable: the middle 50 % of a mostly-successful set is all ones, so every
# method would report exactly 1.000.  IQM is used only for the continuous
# distributions, where trimming genuinely buys robustness to outliers.
TABLE_METRICS = [
    ("success", "Success rate", "{:.3f}", "rate"),
    ("obstacle_collision", "Obstacle collision rate", "{:.3f}", "rate"),
    ("border_collision", "Border collision rate", "{:.3f}", "rate"),
    ("timeout", "Timeout rate", "{:.3f}", "rate"),
    ("rudder_saturation_fraction", "Rudder saturation fraction", "{:.3f}", "rate"),
    ("rms_cte", "RMS cross-track error (m)", "{:.3f}", "dist"),
    ("min_obstacle_clearance", "Min obstacle clearance (m)", "{:.3f}", "dist"),
    ("min_border_clearance", "Min border clearance, all walls (m)", "{:.3f}", "dist"),
    ("min_lateral_border_clearance", "Min border clearance, lateral (m)", "{:.3f}", "dist"),
    ("control_effort", "Control effort (int. sq. rudder cmd)", "{:.3f}", "dist"),
    ("mean_abs_rudder_rate", "Mean abs. rudder rate (deg/s)", "{:.2f}", "dist"),
    ("path_completion_time_s", "Completion time (s)", "{:.1f}", "dist"),
]


# ---------------------------------------------------------------------------
def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        parsed = {}
        for k, v in r.items():
            try:
                parsed[k] = float(v)
            except (TypeError, ValueError):
                parsed[k] = v
        out.append(parsed)
    return out


def discover_methods() -> List[Dict[str, Any]]:
    """Resolve the method specs against what actually exists on disk."""
    found = []
    for label, pattern, deterministic in METHOD_SPECS:
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"  (skipping {label}: no files match {pattern})")
            continue
        runs = [{"path": p, "rows": load_csv(p),
                 "name": os.path.basename(os.path.dirname(p))} for p in paths]
        found.append({"label": label, "runs": runs, "deterministic": deterministic})
    return found


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def build_comparison_table() -> List[Dict[str, Any]]:
    from metrics import iqm, stratified_bootstrap_ci

    methods = discover_methods()
    table: List[Dict[str, Any]] = []

    for m in methods:
        # Pool episodes across seeds; keep seed identity for the per-seed column.
        pooled = [r for run in m["runs"] for r in run["rows"]]
        strata = np.array([r["obstacle_count"] for r in pooled])

        entry: Dict[str, Any] = {
            "method": m["label"],
            "n_runs": len(m["runs"]),
            "n_episodes_total": len(pooled),
            "runs": ", ".join(r["name"] for r in m["runs"]),
            "deterministic": m["deterministic"],
        }

        for key, _, _, stat in TABLE_METRICS:
            vals = np.array([r.get(key, np.nan) for r in pooled], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            estimator = np.mean if stat == "rate" else iqm
            entry[f"{key}__mean"] = float(np.mean(finite)) if finite.size else float("nan")
            entry[f"{key}__iqm"] = iqm(vals)
            entry[f"{key}__stat"] = stat
            entry[f"{key}__point"] = (entry[f"{key}__mean"] if stat == "rate"
                                      else entry[f"{key}__iqm"])
            if m["deterministic"] or not finite.size:
                entry[f"{key}__ci_lo"] = float("nan")
                entry[f"{key}__ci_hi"] = float("nan")
            else:
                lo, hi = stratified_bootstrap_ci(vals, strata, estimator, n_boot=5000)
                entry[f"{key}__ci_lo"] = lo
                entry[f"{key}__ci_hi"] = hi

            # Per-seed spread, for the learned methods
            with np.errstate(invalid="ignore"):
                per_run = [float(np.nanmean([r.get(key, np.nan) for r in run["rows"]]))
                           for run in m["runs"]]
            entry[f"{key}__per_run"] = ";".join(f"{v:.4f}" for v in per_run)

        table.append(entry)
    return table


def write_comparison_table(table: List[Dict[str, Any]]) -> None:
    os.makedirs(BASE, exist_ok=True)
    csv_path = os.path.join(BASE, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    lines = ["# Comparison on the frozen 500-episode evaluation set", ""]
    lines.append("Point estimate with a stratified bootstrap 95 % CI (stratified by "
                 "obstacle count, pooled over runs, 5000 resamples).")
    lines.append("")
    lines.append("**What each interval covers is not the same, and the rows are not "
                 "interchangeable.** `SAC (published)` is a single checkpoint, so its "
                 "interval reflects episode/layout variance only. The `SAC (retrained)` "
                 "and `PPO` rows pool three from-scratch training seeds, so theirs "
                 "reflect seed *and* episode variance. `LOS+APF` pools three "
                 "independent 250-configuration random searches under different search "
                 "seeds: the controller itself is deterministic -- re-running it "
                 "reproduces its CSV byte for byte -- so its interval reflects "
                 "tuning-procedure variance, which is the analogue of a training seed "
                 "for a non-learned method.")
    lines.append("")
    lines.append("`SAC (published)` and `SAC (retrained)` are different objects and are "
                 "deliberately not pooled: the published checkpoint carries a long "
                 "history of resumed, hand-staged training that the 1M-step retrains do "
                 "not.")
    lines.append("")
    lines.append("Rate metrics (marked *) are per-episode 0/1 outcomes and are reported "
                 "as **means**, which is what a success or collision rate is. The "
                 "remaining, continuous metrics are reported as **IQM** "
                 "(interquartile mean). IQM is not used for the rates because it is "
                 "degenerate on a binary variable -- the middle 50 % of a mostly-"
                 "successful set is all ones, so every method would read exactly 1.000.")
    lines.append("")

    header = "| Metric | " + " | ".join(e["method"] for e in table) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(table) + 1))

    for key, label, fmt, stat in TABLE_METRICS:
        cells = []
        for e in table:
            v = e[f"{key}__point"]
            if e["deterministic"] or not np.isfinite(e[f"{key}__ci_lo"]):
                cells.append(fmt.format(v) if np.isfinite(v) else "--")
            else:
                cells.append(f"{fmt.format(v)} [{fmt.format(e[f'{key}__ci_lo'])}, "
                             f"{fmt.format(e[f'{key}__ci_hi'])}]")
        marker = " *" if stat == "rate" else ""
        lines.append(f"| {label}{marker} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("| | " + " | ".join(e["method"] for e in table) + " |")
    lines.append("|" + "---|" * (len(table) + 1))
    lines.append("| Runs | " + " | ".join(e["runs"] for e in table) + " |")
    lines.append("| Episodes | " + " | ".join(str(e["n_episodes_total"]) for e in table) + " |")

    md_path = os.path.join(BASE, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {csv_path}")
    print(f"  wrote {md_path}")


# ---------------------------------------------------------------------------
# Paired statistics table
# ---------------------------------------------------------------------------
def build_paired_table() -> None:
    from compare import compare

    sac = os.path.join(BASE, "sac_1M", "episodes.csv")
    if not os.path.exists(sac):
        print("  (skipping paired table: SAC episodes.csv missing)")
        return

    # Everything is paired against the published SAC checkpoint, which is the
    # manuscript's baseline and the one a reviewer will have in front of them.
    # Everything is paired against the published SAC checkpoint, which is the
    # manuscript's baseline and the one a reviewer will have in front of them.
    pairs = []
    for run in sorted(glob.glob(os.path.join(BASE, "ppo_fx_seed*_final", "episodes.csv"))):
        seed = os.path.basename(os.path.dirname(run)).replace("ppo_fx_", "").replace("_final", "")
        pairs.append(("SAC published", sac, f"PPO final ({seed})", run))
    for los in sorted(glob.glob(os.path.join(BASE, "los_apf_s*", "episodes.csv"))):
        tag = os.path.basename(os.path.dirname(los))
        pairs.append(("SAC published", sac, f"LOS+APF ({tag})", los))

    rows: List[Dict[str, Any]] = []
    full: Dict[str, Any] = {}
    for name_a, path_a, name_b, path_b in pairs:
        res = compare(path_a, path_b, name_a=name_a, name_b=name_b)
        full[f"{name_a} vs {name_b}"] = res
        mc = res["mcnemar_success"]
        for scope in ("all_paired", "both_succeeded"):
            w = res["wilcoxon"][scope]["rms_cte"]
            c = res["wilcoxon"][scope]["min_obstacle_clearance"]
            rows.append({
                "comparison": f"{name_a} vs {name_b}",
                "scope": scope,
                "n_pairs": w["n_pairs"],
                "success_a": mc["success_rate_a"],
                "success_b": mc["success_rate_b"],
                "mcnemar_only_a": mc["only_a_success"],
                "mcnemar_only_b": mc["only_b_success"],
                "mcnemar_p": mc["p_value"],
                "rms_cte_median_a": w["median_a"],
                "rms_cte_median_b": w["median_b"],
                "rms_cte_hl_diff": w["hodges_lehmann_a_minus_b"],
                "rms_cte_wilcoxon_p": w["p_value"],
                "clearance_median_a": c["median_a"],
                "clearance_median_b": c["median_b"],
                "clearance_hl_diff": c["hodges_lehmann_a_minus_b"],
                "clearance_wilcoxon_p": c["p_value"],
            })

    if not rows:
        return
    csv_path = os.path.join(BASE, "paired_stats_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(BASE, "paired_stats.json"), "w") as f:
        json.dump(full, f, indent=2)

    lines = ["# Paired statistics on the frozen 500-episode evaluation set", "",
             "McNemar (exact) on success; Wilcoxon signed-rank on RMS cross-track "
             "error and on per-episode minimum obstacle clearance. "
             "`both_succeeded` restricts to episodes where both methods reached "
             "the goal -- a collision truncates the trajectory and flatters its "
             "RMS CTE, so the all-episode figure mixes tracking quality with "
             "failure timing.", "",
             "| Comparison | Scope | n | Succ A | Succ B | McNemar p | "
             "RMS CTE A | RMS CTE B | HL diff | Wilcoxon p |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['comparison']} | {r['scope']} | {r['n_pairs']} | "
            f"{r['success_a']:.3f} | {r['success_b']:.3f} | {r['mcnemar_p']:.3g} | "
            f"{r['rms_cte_median_a']:.3f} | {r['rms_cte_median_b']:.3f} | "
            f"{r['rms_cte_hl_diff']:+.3f} | {r['rms_cte_wilcoxon_p']:.3g} |")
    md_path = os.path.join(BASE, "paired_stats_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {csv_path}")
    print(f"  wrote {md_path}")


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------
SAC_EVENT_FILE = os.path.join(
    "sac_log", "asv_sac_2", "events.out.tfevents.1781347673.JL2VSV3.29056.0")


def read_sac_curve(path: str = SAC_EVENT_FILE,
                   tag: str = "eval/mean_ep_reward") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Recover the SAC evaluation curve from TensorBoard events.

    The root `eval_summary.json` that `plotting/plot_training_curves.py` reads
    now only covers 1.025M-1.150M steps -- a later fine-tuning run overwrote the
    0-1M data.  The tfevents files are the surviving record.  `asv_sac_2` is the
    published run: its eval return reaches +58 with success 1.0 at 950k, against
    -214 / 0.7 for `asv_sac_1`.
    """
    if not os.path.exists(path):
        print(f"  (SAC curve: {path} not found)")
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  (SAC curve: tensorboard not installed)")
        return None
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        print(f"  (SAC curve: tag {tag} absent)")
        return None
    events = ea.Scalars(tag)
    return (np.array([e.step for e in events], dtype=float),
            np.array([e.value for e in events], dtype=float))


def read_run_curves(prefix: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Evaluation-return curves for every seed of a retrained algorithm."""
    out = []
    for path in sorted(glob.glob(os.path.join("models", f"{prefix}_seed*", "eval_summary.json"))):
        with open(path, "r") as f:
            rows = json.load(f)
        rows = [r for r in rows if "timesteps" in r]
        rows.sort(key=lambda r: int(r["timesteps"]))
        if not rows:
            continue
        name = os.path.basename(os.path.dirname(path))
        out.append((name,
                    np.array([r["timesteps"] for r in rows], dtype=float),
                    np.array([r["mean_ep_reward"] for r in rows], dtype=float)))
    return out


def plot_learning_curves() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    sac = read_sac_curve()
    families = [
        ("PPO", read_run_curves("ppo"), "tab:blue", "--", "s"),
    ]
    families = [f for f in families if f[1]]
    if sac is None and not families:
        print("  (skipping learning curve: no data)")
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    if sac is not None:
        ax.plot(sac[0] / 1e3, sac[1], color="black", linestyle="-", marker="o",
                markersize=4, linewidth=1.8, label="SAC (published run)", zorder=4)

    for label, runs, colour, style, marker in families:
        # Interpolate every seed onto a common grid so the band is well defined.
        grid = np.unique(np.concatenate([c[1] for c in runs]))
        stack = np.vstack([np.interp(grid, steps, vals) for _, steps, vals in runs])
        ax.plot(grid / 1e3, stack.mean(axis=0), color=colour, linestyle=style,
                marker=marker, markersize=4, linewidth=1.8,
                label=f"{label} (mean of {len(runs)} seeds)", zorder=3)
        if len(runs) > 1:
            ax.fill_between(grid / 1e3, stack.min(axis=0), stack.max(axis=0),
                            color=colour, alpha=0.15, linewidth=0,
                            label=f"{label} seed range", zorder=1)
        for i, (_, steps, vals) in enumerate(runs):
            ax.plot(steps / 1e3, vals, color=colour, alpha=0.40,
                    linestyle=[":", (0, (1, 1)), (0, (3, 1, 1, 1))][i % 3],
                    linewidth=0.9, zorder=2)

    # No curriculum markers are drawn.  The published SAC run and every
    # reported PPO run used fixed RPM for the entire 1M steps -- verified from
    # the published run's own log, where eval/min_rpm == eval/max_rpm == 12.000
    # at all 19 evaluations.  An earlier version of this figure drew stage
    # boundaries at 700k/800k/900k taken from plot_training_curves.py; those
    # describe the post-1M resumed runs, not these.  See BASELINES_RESULTS.md section 4.

    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Mean evaluation episode return")
    ax.set_title("Evaluation return during training")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95, ncol=2)
    fig.tight_layout()

    for ext in ("png", "svg"):
        p = os.path.join(FIG_DIR, f"learning_curves.{ext}")
        fig.savefig(p, dpi=200)
        print(f"  wrote {p}")
    plt.close(fig)

    # The underlying numbers, so the figure can be rebuilt or re-styled.
    csv_path = os.path.join(FIG_DIR, "learning_curves.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "timesteps", "mean_eval_return"])
        if sac is not None:
            for s, v in zip(*sac):
                w.writerow(["sac_published", int(s), v])
        for _, runs, _, _, _ in families:
            for name, steps, vals in runs:
                for s, v in zip(steps, vals):
                    w.writerow([name, int(s), v])
    print(f"  wrote {csv_path}")


# ---------------------------------------------------------------------------
# Qualitative trajectories
# ---------------------------------------------------------------------------
def collect_trajectory(spec, record) -> Dict[str, Any]:
    from env import ASVLidarEnv
    from eval_layouts import reset_to_layout
    from evaluate import build_controller, _as_action

    controller = build_controller(spec)
    env = ASVLidarEnv(map_width=10.0, map_height=25.0, max_obs=5, path_mode="straight")
    obs, _ = reset_to_layout(env, record)
    if hasattr(controller, "reset"):
        controller.reset()

    info: Dict[str, Any] = {}
    for _ in range(2000):
        action = _as_action(controller.predict(obs, deterministic=True))
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    out = {
        "path": list(env.asv_path),
        "obstacles": [list(map(tuple, o)) for o in env.obstacles],
        "reference": env.path.points.tolist(),
        "reached_goal": bool(info.get("reached_goal", False)),
        "collided": bool(info.get("collided", False)),
        "timeout": bool(info.get("timeout", False)),
        "goal": (env.goal_x, env.goal_y),
        "start": (env.start_x, env.start_y),
        "map_width": env.map_width,
        "map_height": env.map_height,
    }
    env.close()
    return out


def plot_trajectories(case_ids: Sequence[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from eval_layouts import EVAL_LAYOUTS, load_layouts

    os.makedirs(FIG_DIR, exist_ok=True)
    records = {int(r["case_id"]): r for r in load_layouts(EVAL_LAYOUTS)}

    specs = []
    if os.path.exists("models/sac_model_1M.zip"):
        specs.append(("SAC", "sb3:sac:models/sac_model_1M.zip", "-", "black"))
    # The best checkpoint, not the 1M one: the collapsed final policy drives
    # into a wall on every layout, which makes for a useless qualitative panel.
    # The collapse is shown in the learning-curve figure instead.
    ppo_best = sorted(glob.glob("models/ppo_seed*/best_model.zip"))
    if ppo_best:
        specs.append(("PPO (best ckpt)", f"sb3:ppo:{ppo_best[0]}", "--", "tab:blue"))
    best_json = os.path.join(BASE, "los_apf_best.json")
    if os.path.exists(best_json):
        specs.append(("LOS+APF", f"los_apf:{best_json}", "-.", "tab:red"))
    if not specs:
        print("  (skipping trajectories: no controllers available)")
        return

    case_ids = [c for c in case_ids if c in records]
    fig, axes = plt.subplots(1, len(case_ids), figsize=(3.6 * len(case_ids), 7.4),
                             squeeze=False)

    for col, case_id in enumerate(case_ids):
        ax = axes[0][col]
        record = records[case_id]
        first = True
        for label, spec, style, colour in specs:
            traj = collect_trajectory(spec, record)
            if first:
                for obs in traj["obstacles"]:
                    ax.add_patch(MplPolygon(obs, closed=True, facecolor="0.75",
                                            edgecolor="0.35", zorder=1))
                ref = np.asarray(traj["reference"])
                ax.plot(ref[:, 0], ref[:, 1], color="0.5", linestyle=":",
                        linewidth=1.4, label="Reference path", zorder=2)
                ax.plot(*traj["start"], marker="o", color="green", markersize=8,
                        zorder=5)
                ax.plot(*traj["goal"], marker="*", color="darkgreen",
                        markersize=14, zorder=5)
                ax.set_xlim(0, traj["map_width"])
                ax.set_ylim(0, traj["map_height"])
                first = False

            p = np.asarray(traj["path"])
            outcome = ("goal" if traj["reached_goal"]
                       else "timeout" if traj["timeout"]
                       else "collision" if traj["collided"] else "--")
            ax.plot(p[:, 0], p[:, 1], linestyle=style, color=colour, linewidth=1.9,
                    label=f"{label} ({outcome})", zorder=4)
            if not traj["reached_goal"]:
                ax.plot(p[-1, 0], p[-1, 1], marker="x", color=colour,
                        markersize=9, markeredgewidth=2.2, zorder=6)

        ax.set_aspect("equal")
        ax.set_title(f"Case {case_id}  ({record['obstacle_count']} obstacles)",
                     fontsize=10)
        ax.set_xlabel("x (m)")
        if col == 0:
            ax.set_ylabel("y (m)")
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.95)
        ax.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    for ext in ("png", "svg"):
        p = os.path.join(FIG_DIR, f"trajectories.{ext}")
        fig.savefig(p, dpi=200)
        print(f"  wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--trajectories", action="store_true")
    ap.add_argument("--cases", type=int, nargs="+", default=[220, 320, 420],
                    help="evaluation case ids for the qualitative figure")
    args = ap.parse_args()

    if not any((args.all, args.tables, args.learning_curve, args.trajectories)):
        ap.error("pass --all or one of --tables/--learning-curve/--trajectories")

    if args.all or args.tables:
        print("Comparison table ...")
        table = build_comparison_table()
        if table:
            write_comparison_table(table)
        print("Paired statistics ...")
        build_paired_table()

    if args.all or args.learning_curve:
        print("Learning curves ...")
        plot_learning_curves()

    if args.all or args.trajectories:
        print("Trajectories ...")
        plot_trajectories(args.cases)


if __name__ == "__main__":
    main()
