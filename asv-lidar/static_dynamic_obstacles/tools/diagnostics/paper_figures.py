"""Results figures for the learner comparison (2026-10-08), with Paper 2's seed
representation: a filled marker or line for the seed mean, hollow markers for individual
training seeds, and the seed range as a bar or band.

A. Success rate by encounter type on the whole test set v4 (1,000 episodes, coupled and
   decoupled together), learners and classical comparators, as grouped bars
   (`fig_a_success_by_encounter`, the chosen form) and, kept as a reserve template in case
   Paper 2's line style is required, `fig_a_alt_lines`. Classical comparators are
   deterministic, so they carry no spread.
B. Validation-set learning curves (the 270-episode validation set, coupled and decoupled
   together, evaluated every 200,000 steps, safety layer off), RL learners only: seed mean,
   individual seeds and seed range, with the curriculum stage boundaries.

Seeds are discovered from the run folders (`runs/<algo>_formulation_seed<s>_bl3`) and the
test-set folders (`results/test_set/v4/<algo>s<s>_bl3_metrics`), so the figures fill in as
seeds finish. Decision (2026-10-08): no selected-policy marker, no classical intervals,
no trade-off figure (completion time and safety stay in the results table and the text).

    python tools/diagnostics/paper_figures.py [--out results/baseline_v3/figures_draft]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "results" / "test_set" / "v4"
LEARNERS = [("SAC", "sac"), ("PPO", "ppo"), ("RecurrentPPO", "recurrent_ppo"), ("TQC", "tqc")]
CLASSICAL = [("LOS-DWA", "los_dwa"), ("COLREGs-VO", "colregs_vo")]
STYLE = {
    "SAC": dict(color="#2ca02c", marker="^", ls="-."),
    "PPO": dict(color="#1f77b4", marker="s", ls="--"),
    "RecurrentPPO": dict(color="#9467bd", marker="o", ls="-"),
    "TQC": dict(color="#ff7f0e", marker="v", ls=":"),
    "LOS-DWA": dict(color="#d62728", marker="D", ls=(0, (1, 1))),
    "COLREGs-VO": dict(color="#8c564b", marker="P", ls=(0, (3, 1, 1, 1))),
}
STAGES_M = [0.16, 0.36, 0.64, 1.0, 1.5, 2.0]          # stage 2-7 starts, baseline-v3
CATS = [("none", "No\nencounter"), ("head_on", "Head-on"), ("crossing_port", "Crossing\nfrom port"),
        ("crossing_starboard", "Crossing\nfrom starboard"), ("overtaking", "Overtaking"),
        ("being_overtaken", "Being\novertaken"), ("all", "All")]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "savefig.dpi": 300})


def seeds_of(algo):
    out = []
    for s in range(5):
        run = ROOT / "runs" / f"{algo}_formulation_seed{s}_bl3"
        if run.exists():
            out.append((s, run, TEST / f"{algo}s{s}_bl3_metrics"))
    return out


def load_test(folder):
    d = pd.read_csv(folder / "episodes.csv", keep_default_na=False)
    d = d[d.safety == "off"].copy()
    d.loc[(d["class"] == "") & d.cell.str.contains("null"), "class"] = "null"
    d["success"] = (d.outcome == "goal").astype(float)
    cat = np.where(d["class"] == "crossing", "crossing_" + d.crossing_side, d["class"])
    d["category"] = np.where(np.isin(cat, ["null", "no_target"]), "none", cat)   # no target, or no risk
    return d


def load_dev(run):
    recs = json.loads((run / "eval_summary.json").read_text())
    df = pd.DataFrame([r for r in recs if r.get("supervisor", r.get("safety")) == "off"])
    return df[df.timesteps % 200_000 == 0].drop_duplicates("timesteps", keep="last").sort_values("timesteps")


def rate(d, cat):
    return (d if cat == "all" else d[d.category == cat]).success.mean()


def counts(d):
    return [len(d) if c == "all" else int((d.category == c).sum()) for c, _ in CATS]


def legend_label(name, n):
    return f"{name} ({n} seed{'s' if n > 1 else ''})"


def fig_a_bars(learners, classical, out):
    methods = [(n, f, True) for n, f in learners] + [(n, [d], False) for n, d in classical]
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    width = 0.8 / len(methods)
    for j, (name, frames, learned) in enumerate(methods):
        st = STYLE[name]
        x = np.arange(len(CATS)) - 0.4 + width * (j + 0.5)
        per_seed = np.array([[rate(d, c) for c, _ in CATS] for d in frames])
        mean = per_seed.mean(0)
        ax.bar(x, mean, width * 0.92, color=st["color"] if learned else "white", edgecolor=st["color"],
               hatch=None if learned else "////", lw=0.8,
               label=legend_label(name, len(frames)) if learned else name, zorder=2)
        if learned:
            for row in per_seed:
                ax.plot(x, row, "o", mfc="white", mec="k", ms=2.6, mew=0.6, lw=0, zorder=4)
            if len(per_seed) > 1:
                ax.vlines(x, per_seed.min(0), per_seed.max(0), color="k", lw=0.8, zorder=3)
    n = counts(learners[0][1][0])
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels([f"{lab}\n(n={k:,})" for (_, lab), k in zip(CATS, n)])
    ax.axvline(len(CATS) - 1.5, color="grey", lw=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, len(CATS) - 0.5)
    ax.set_ylabel("Success rate")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=len(methods), frameon=False,
              handlelength=1.6, columnspacing=1.2)
    ax.text(1.0, -0.33, "Hollow circles: individual training seeds; bars: seed mean; hatched: classical comparators",
            transform=ax.transAxes, ha="right", fontsize=6.5, color="grey")
    fig.tight_layout()
    save(fig, out / "fig_a_success_by_encounter")


def fig_a_lines(learners, classical, out):
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    x = np.arange(len(CATS))
    for name, frames in learners:
        st = STYLE[name]
        per_seed = np.array([[rate(d, c) for c, _ in CATS] for d in frames])
        mean = per_seed.mean(0)
        if len(per_seed) > 1:
            ax.fill_between(x[:-1], per_seed.min(0)[:-1], per_seed.max(0)[:-1], color=st["color"], alpha=0.15, lw=0)
            ax.vlines(x[-1], per_seed.min(0)[-1], per_seed.max(0)[-1], color=st["color"], lw=1)
        ax.plot(x[:-1], mean[:-1], color=st["color"], ls=st["ls"], marker=st["marker"], ms=5, lw=1.4,
                label=legend_label(name, len(frames)))
        ax.plot(x[-1], mean[-1], st["marker"], color=st["color"], ms=6)
        for row in per_seed:
            ax.plot(x, row, st["marker"], mfc="none", mec=st["color"], ms=3.5, alpha=0.7, lw=0)
    for name, d in classical:
        st = STYLE[name]
        vals = [rate(d, c) for c, _ in CATS]
        ax.plot(x[:-1], vals[:-1], color=st["color"], ls=st["ls"], marker=st["marker"], ms=4.5, lw=1.2, label=name)
        ax.plot(x[-1], vals[-1], st["marker"], color=st["color"], ms=5)
    n = counts(learners[0][1][0])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={k:,})" for (_, lab), k in zip(CATS, n)])
    ax.axvline(len(CATS) - 1.5, color="grey", lw=0.8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Success rate")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.9)
    fig.tight_layout()
    save(fig, out / "fig_a_alt_lines")


def fig_b(dev, out):
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    for x in STAGES_M:
        ax.axvline(x, color="grey", ls="--", lw=0.6)
    edges = [0.0] + STAGES_M + [3.0]
    for k in range(7):                                   # stage labels centred above each stage
        ax.text((edges[k] + edges[k + 1]) / 2, 1.015, f"S{k + 1}", fontsize=6.5, color="grey",
                ha="center", va="bottom", transform=ax.get_xaxis_transform())
    ax.text(1.0, 1.075, "S1-S7: curriculum stages", fontsize=6, color="grey", ha="right", va="bottom",
            transform=ax.transAxes)
    for name, runs in dev.items():
        st = STYLE[name]
        grid = sorted(set().union(*[set(r.timesteps) for r in runs]))
        mat = np.array([[r.set_index("timesteps").goal_rate.get(t, np.nan) for t in grid] for r in runs])
        t = np.array(grid) / 1e6
        if len(runs) > 1:
            ax.fill_between(t, np.nanmin(mat, 0), np.nanmax(mat, 0), color=st["color"], alpha=0.15, lw=0)
        for r in runs:
            ax.plot(r.timesteps / 1e6, r.goal_rate, st["marker"], mfc="none", mec=st["color"], ms=3, alpha=0.7, lw=0)
        ax.plot(t, np.nanmean(mat, 0), color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5, lw=1.4,
                label=legend_label(name, len(runs)))
    ax.set_xlabel("Training timesteps (million)")
    ax.set_ylabel("Validation-set success rate")
    ax.set_xlim(0, 3.05)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 0.02), ncol=4, frameon=True, framealpha=0.9)
    fig.tight_layout()
    save(fig, out / "fig_b_learning_curves")


def save(fig, stem):
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("->", stem.with_suffix(".png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "baseline_v3" / "figures_draft")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    learners, dev = [], {}
    for name, algo in LEARNERS:
        runs = seeds_of(algo)
        frames = [load_test(t) for s, r, t in runs if (t / "episodes.csv").exists()]
        devs = [load_dev(r) for s, r, t in runs if (r / "eval_summary.json").exists()]
        if frames:
            learners.append((name, frames))
        if devs:
            dev[name] = devs
    classical = [(name, load_test(TEST / tag)) for name, tag in CLASSICAL if (TEST / tag / "episodes.csv").exists()]
    fig_a_bars(learners, classical, args.out)
    fig_a_lines(learners, classical, args.out)
    fig_b(dev, args.out)


if __name__ == "__main__":
    main()
