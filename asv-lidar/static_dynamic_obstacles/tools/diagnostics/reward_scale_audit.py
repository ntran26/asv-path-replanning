"""Reward scale audit of the deployed policy on test set v4: Table R7 (decision S19, 2026-10-24).

Reads a test set v4 run that records the reward terms per episode (`tools/tiers/common.py`
writes `reward_int_<term>`, the weighted episode integral of each dense term, the terminal and
safety-layer rewards, and `reward_active_<term>`, the steps on which a term was non-zero) and
writes the audit table:

  * per term, in the order of the designed magnitude hierarchy (the weights: boundary 15 >
    ship domain 12.5 > obstacle 11 > COLREGs group 9 > path following 3 > progress 1.5 >
    smoothness 0.5 > existence 0.25): its weight, the mean weighted episode integral, its share
    of the episode's absolute dense reward, the share of episodes in which it was ever active,
    and its mean per-step magnitude on the steps where it was active (the realised scale, set
    against the weight);
  * the terminal rewards (+100 goal, -300 collision, 0 timeout) beside them;
  * all episodes, the decoupled and coupled families, and goal and collision episodes;
  * two checks: the per-episode terms add up to the recorded return, and the run reproduces
    the outcomes, steps and returns of the run reported in Table R1.

    python tools/diagnostics/reward_scale_audit.py
    python tools/diagnostics/reward_scale_audit.py --run sacs0_bl3_audit --reference sacs0_bl3_metrics

Writes `results/baseline_v3/reward_scale_audit.md` and `.csv`. Safety layer off.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from reward.config import RewardConfig  # noqa: E402

TEST = ROOT / "results" / "test_set" / "v4"
OUT = ROOT / "results" / "baseline_v3" / "reward_scale_audit"
# Designed magnitude hierarchy, largest weight first (`RewardConfig` asserts this order).
ORDER = ("bnd", "dom", "obs", "col", "pf", "prog", "smooth", "exist")
LABEL = {"bnd": "Boundary r_bnd", "dom": "Ship domain r_dom", "obs": "Static obstacle r_obs",
         "col": "COLREGs group r_col", "pf": "Path following r_pf", "prog": "Progress r_prog",
         "smooth": "Smoothness r_smooth", "exist": "Existence r_exist"}


def summarise(d: pd.DataFrame, weights: dict) -> pd.DataFrame:
    dense_abs = sum(d[f"reward_int_{t}"].abs() for t in ORDER)
    rows = []
    for t in ORDER:
        integral = d[f"reward_int_{t}"]
        active = d[f"reward_active_{t}"]
        per_step = (integral.abs() / active.where(active > 0)).dropna()
        rows.append({
            "term": t, "label": LABEL[t], "weight": weights[t],
            "mean_integral": float(integral.mean()),
            "share_of_dense_abs": float((integral.abs() / dense_abs.where(dense_abs > 0)).mean()),
            "episodes_active": float((active > 0).mean()),
            "mean_active_steps": float(active.mean()),
            "per_step_when_active": float(per_step.mean()) if len(per_step) else float("nan"),
        })
    rows.append({"term": "terminal", "label": "Terminal (+100 goal, -300 collision)", "weight": float("nan"),
                 "mean_integral": float(d["reward_int_terminal"].mean()), "share_of_dense_abs": float("nan"),
                 "episodes_active": float((d["reward_int_terminal"] != 0).mean()),
                 "mean_active_steps": float("nan"), "per_step_when_active": float("nan")})
    rows.append({"term": "return", "label": "Episode return", "weight": float("nan"),
                 "mean_integral": float(d["return"].mean()), "share_of_dense_abs": float("nan"),
                 "episodes_active": float("nan"), "mean_active_steps": float("nan"),
                 "per_step_when_active": float("nan")})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="sacs0_bl3_audit", help="test set v4 tag with reward_int_* columns")
    ap.add_argument("--reference", default="sacs0_bl3_metrics", help="the run reported in Table R1")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    d = pd.read_csv(TEST / args.run / "episodes.csv", keep_default_na=False)
    d = d[d.safety == "off"].copy()
    c = RewardConfig()
    weights = {t: float(getattr(c, f"w_{t}")) for t in ORDER}

    # Check 1: the recorded terms add up to the return, episode by episode.
    parts = sum(d[f"reward_int_{t}"] for t in ORDER) + d["reward_int_terminal"] + d["reward_int_intervention"]
    gap = (parts - d["return"]).abs()
    # Check 2: the same episodes as the Table R1 run (deterministic policy, same test set).
    ref = pd.read_csv(TEST / args.reference / "episodes.csv", keep_default_na=False)
    ref = ref[ref.safety == "off"].set_index("test_id")
    cur = d.set_index("test_id").reindex(ref.index)
    same_outcome = int((cur.outcome == ref.outcome).sum())
    same_steps = int((cur.steps == ref.steps).sum())
    ret_gap = (cur["return"].astype(float) - ref["return"].astype(float)).abs()

    scopes = [("All episodes", d), ("Decoupled", d[d.source == "frozen"]), ("Coupled", d[d.source == "paper2"]),
              ("Goal episodes", d[d.outcome == "goal"]), ("Collision episodes", d[d.outcome.str.startswith("collision")])]
    tables = {name: summarise(sd, weights) for name, sd in scopes if len(sd)}

    L = ["# Reward scale audit of the deployed policy (Table R7)", "",
         f"Run `{args.run}`: SAC seed 0 (the deployed policy), best validation-set checkpoint, on test set v4 "
         f"({len(d)} episodes), safety layer off. Generated by `tools/diagnostics/reward_scale_audit.py`.", "",
         "Columns: the term's weight (its largest per-step contribution); the mean weighted episode integral; "
         "the mean share of the episode's absolute dense reward; the share of episodes in which the term was "
         "ever non-zero; the mean number of active steps; and the mean per-step magnitude on the steps where "
         "the term was active (the realised scale, to set against the weight). Terms are listed in the "
         "designed hierarchy, largest weight first.", "",
         "## Checks", "",
         f"- Terms add up to the return: largest per-episode gap {gap.max():.2e} over {len(d)} episodes.",
         f"- Same episodes as the Table R1 run `{args.reference}`: outcome identical in {same_outcome} of "
         f"{len(ref)}, steps identical in {same_steps}, largest return difference {ret_gap.max():.2e}.", ""]
    csv = []
    for name, t in tables.items():
        n = len(dict(scopes)[name])
        L += [f"## {name} (n = {n})", "",
              "| Term | Weight | Mean episode integral | Share of dense abs. | Episodes active | Active steps | Per step when active |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for r in t.itertuples():
            fmt = lambda x, f: "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else format(x, f)
            L.append(f"| {r.label} | {fmt(r.weight, '.2f')} | {r.mean_integral:+.2f} | {fmt(r.share_of_dense_abs, '.3f')} | "
                     f"{fmt(r.episodes_active, '.3f')} | {fmt(r.mean_active_steps, '.1f')} | {fmt(r.per_step_when_active, '.3f')} |")
            csv.append({"scope": name, "n": n, **r._asdict()})
        realised = t[t.term.isin(ORDER)].sort_values("per_step_when_active", ascending=False).term.tolist()
        by_integral = t[t.term.isin(ORDER)].assign(a=lambda x: x.mean_integral.abs()).sort_values("a", ascending=False).term.tolist()
        L += ["", f"Order by per-step magnitude when active: {' > '.join(realised)} "
                  f"(designed: {' > '.join(ORDER)}).",
              f"Order by mean episode integral: {' > '.join(by_integral)}.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    pd.DataFrame(csv).drop(columns=["Index"], errors="ignore").to_csv(args.out.with_suffix(".csv"), index=False)
    print("\n".join(L))
    print(f"-> {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
