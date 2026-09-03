"""Paired statistics between two methods evaluated on the same layouts.

    python src/compare.py --a eval_results/baselines/sac_1M/episodes.csv \
                          --b eval_results/baselines/ppo_seed0/episodes.csv \
                          --out eval_results/baselines/compare_sac_ppo.json

Joins two per-episode CSVs on `episode_id` and reports:

* **Wilcoxon signed-rank** on `rms_cte` and on the per-episode clearances,
  with a Hodges-Lehmann median-difference estimate and a rank-biserial effect
  size.
* **McNemar** on `success`, computed exactly from the discordant pairs with a
  binomial test.  The exact form is used rather than the chi-square
  approximation because the discordant counts here can be small, which is
  exactly where the approximation misbehaves.

Why paired
----------
Layout difficulty dominates the between-episode variance: a 4-obstacle gate is
harder for every method than an empty basin. Pairing on `episode_id` removes
that shared variance, which is what makes a 2-3 seed comparison defensible
instead of dismissible.

A note on cross-track error and failed episodes
----------------------------------------------
On a collision the episode stops early, so its `rms_cte` is computed over a
truncated trajectory -- often a *flattering* number, because the vessel never
had time to drift. Comparing RMS CTE over all episodes therefore mixes tracking
quality with failure timing. Both are reported:

* `all_paired`      every episode present in both files
* `both_succeeded`  only episodes where both methods reached the goal

`both_succeeded` is the one to quote for a tracking-accuracy claim; `all_paired`
is reported alongside it so nothing looks cherry-picked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, wilcoxon

# Metrics compared with a signed-rank test.
PAIRED_METRICS = (
    "rms_cte",
    "mean_cte",
    "max_cte",
    "min_obstacle_clearance",
    "min_border_clearance",
    "min_lateral_border_clearance",
    "control_effort",
    "mean_abs_rudder_rate",
    "path_completion_time_s",
)


def load_episodes(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        parsed: Dict[str, Any] = {}
        for k, v in r.items():
            try:
                parsed[k] = float(v)
            except (TypeError, ValueError):
                parsed[k] = v
        out[int(parsed["episode_id"])] = parsed
    return out


def hodges_lehmann(diff: np.ndarray) -> float:
    """Median of pairwise Walsh averages -- the estimator Wilcoxon inverts to."""
    d = np.asarray(diff, dtype=np.float64)
    if d.size == 0:
        return float("nan")
    if d.size > 400:      # O(n^2) memory; subsample deterministically past this
        rng = np.random.default_rng(0)
        d = rng.choice(d, 400, replace=False)
    walsh = (d[:, None] + d[None, :]) / 2.0
    return float(np.median(walsh[np.triu_indices(d.size)]))


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Wilcoxon signed-rank on a - b, with effect sizes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    diff = a - b
    nonzero = int(np.count_nonzero(diff))

    result: Dict[str, Any] = {
        "n_pairs": int(a.size),
        "n_nonzero_differences": nonzero,
        "median_a": float(np.median(a)) if a.size else float("nan"),
        "median_b": float(np.median(b)) if b.size else float("nan"),
        "mean_a": float(np.mean(a)) if a.size else float("nan"),
        "mean_b": float(np.mean(b)) if b.size else float("nan"),
        "median_difference_a_minus_b": float(np.median(diff)) if diff.size else float("nan"),
        "hodges_lehmann_a_minus_b": hodges_lehmann(diff),
    }

    if a.size < 2 or nonzero == 0:
        result.update(statistic=float("nan"), p_value=float("nan"),
                      rank_biserial=float("nan"),
                      note="not enough non-tied pairs for a signed-rank test")
        return result

    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    # Rank-biserial correlation: (W+ - W-) / (W+ + W-), in [-1, 1].
    ranks = _signed_ranks(diff[diff != 0])
    w_pos = float(ranks[ranks > 0].sum())
    w_neg = float(-ranks[ranks < 0].sum())
    total = w_pos + w_neg
    result.update(
        statistic=float(stat),
        p_value=float(p),
        rank_biserial=float((w_pos - w_neg) / total) if total > 0 else float("nan"),
    )
    return result


def _signed_ranks(diff: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(np.abs(diff)) * np.sign(diff)


def mcnemar_exact(a_success: np.ndarray, b_success: np.ndarray) -> Dict[str, Any]:
    """Exact McNemar test on paired binary outcomes."""
    a = np.asarray(a_success).astype(bool)
    b = np.asarray(b_success).astype(bool)

    both = int(np.sum(a & b))
    only_a = int(np.sum(a & ~b))      # b in the standard 2x2 notation
    only_b = int(np.sum(~a & b))      # c
    neither = int(np.sum(~a & ~b))
    discordant = only_a + only_b

    out: Dict[str, Any] = {
        "n_pairs": int(a.size),
        "both_success": both,
        "only_a_success": only_a,
        "only_b_success": only_b,
        "neither_success": neither,
        "n_discordant": discordant,
        "success_rate_a": float(np.mean(a)) if a.size else float("nan"),
        "success_rate_b": float(np.mean(b)) if b.size else float("nan"),
        "success_rate_difference_a_minus_b": (
            float(np.mean(a) - np.mean(b)) if a.size else float("nan")),
    }

    if discordant == 0:
        out.update(p_value=1.0, odds_ratio=float("nan"),
                   note="no discordant pairs; methods agree on every episode")
        return out

    test = binomtest(only_a, discordant, 0.5, alternative="two-sided")
    out["p_value"] = float(test.pvalue)
    out["odds_ratio"] = (float(only_a / only_b) if only_b > 0 else float("inf"))
    # Chi-square with continuity correction, for readers who expect it.
    out["chi2_continuity_corrected"] = float(
        (abs(only_a - only_b) - 1) ** 2 / discordant) if discordant > 0 else float("nan")
    return out


def compare(path_a: str, path_b: str, *, name_a: str, name_b: str) -> Dict[str, Any]:
    ep_a = load_episodes(path_a)
    ep_b = load_episodes(path_b)

    shared = sorted(set(ep_a) & set(ep_b))
    if not shared:
        raise SystemExit("no episode_id values in common -- were these run on "
                         "the same layout set?")
    missing_a = sorted(set(ep_b) - set(ep_a))
    missing_b = sorted(set(ep_a) - set(ep_b))

    rows_a = [ep_a[i] for i in shared]
    rows_b = [ep_b[i] for i in shared]

    succ_a = np.array([r["success"] for r in rows_a])
    succ_b = np.array([r["success"] for r in rows_b])
    both_ok = (succ_a > 0.5) & (succ_b > 0.5)

    out: Dict[str, Any] = {
        "method_a": name_a,
        "method_b": name_b,
        "file_a": path_a,
        "file_b": path_b,
        "n_paired_episodes": len(shared),
        "n_only_in_a": len(missing_b),
        "n_only_in_b": len(missing_a),
        "n_both_succeeded": int(both_ok.sum()),
        "mcnemar_success": mcnemar_exact(succ_a, succ_b),
        "wilcoxon": {"all_paired": {}, "both_succeeded": {}},
    }

    for metric in PAIRED_METRICS:
        va = np.array([r.get(metric, np.nan) for r in rows_a], dtype=np.float64)
        vb = np.array([r.get(metric, np.nan) for r in rows_b], dtype=np.float64)
        out["wilcoxon"]["all_paired"][metric] = paired_wilcoxon(va, vb)
        out["wilcoxon"]["both_succeeded"][metric] = paired_wilcoxon(
            va[both_ok], vb[both_ok])

    # Also split success by obstacle count -- the reviewer question is about
    # generalisation across obstacle density.
    counts = np.array([int(r.get("obstacle_count", -1)) for r in rows_a])
    per_group = {}
    for g in sorted(set(counts.tolist())):
        m = counts == g
        per_group[f"obs_{g}"] = {
            "n": int(m.sum()),
            "success_a": float(np.mean(succ_a[m])),
            "success_b": float(np.mean(succ_b[m])),
            "mcnemar": mcnemar_exact(succ_a[m], succ_b[m]),
        }
    out["by_obstacle_count"] = per_group
    return out


def print_report(res: Dict[str, Any]) -> None:
    a, b = res["method_a"], res["method_b"]
    print(f"\n{'=' * 72}")
    print(f"{a}  vs  {b}     ({res['n_paired_episodes']} paired episodes)")
    print(f"{'=' * 72}")

    mc = res["mcnemar_success"]
    print(f"\nMcNemar on success (exact)")
    print(f"  success rate      {a}: {mc['success_rate_a']:.3f}   "
          f"{b}: {mc['success_rate_b']:.3f}   "
          f"diff: {mc['success_rate_difference_a_minus_b']:+.3f}")
    print(f"  discordant pairs  {a} only: {mc['only_a_success']}   "
          f"{b} only: {mc['only_b_success']}   (total {mc['n_discordant']})")
    print(f"  p = {mc['p_value']:.4g}")

    for scope in ("all_paired", "both_succeeded"):
        block = res["wilcoxon"][scope]
        n = block["rms_cte"]["n_pairs"]
        print(f"\nWilcoxon signed-rank -- {scope} (n = {n})")
        print(f"  {'metric':<32} {'median ' + a[:8]:>14} {'median ' + b[:8]:>14} "
              f"{'HL diff':>10} {'p':>10}")
        for metric, r in block.items():
            if not np.isfinite(r.get("p_value", np.nan)):
                continue
            print(f"  {metric:<32} {r['median_a']:>14.4f} {r['median_b']:>14.4f} "
                  f"{r['hodges_lehmann_a_minus_b']:>10.4f} {r['p_value']:>10.4g}")

    print(f"\nSuccess by obstacle count")
    print(f"  {'group':<10} {'n':>4} {a[:12]:>13} {b[:12]:>13} {'p':>10}")
    for g, r in res["by_obstacle_count"].items():
        print(f"  {g:<10} {r['n']:>4} {r['success_a']:>13.3f} "
              f"{r['success_b']:>13.3f} {r['mcnemar']['p_value']:>10.4g}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="per-episode CSV for method A")
    ap.add_argument("--b", required=True, help="per-episode CSV for method B")
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    return ap.parse_args()


def _default_name(path: str) -> str:
    return os.path.basename(os.path.dirname(path)) or os.path.basename(path)


def main() -> None:
    args = parse_args()
    res = compare(args.a, args.b,
                  name_a=args.name_a or _default_name(args.a),
                  name_b=args.name_b or _default_name(args.b))
    print_report(res)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
