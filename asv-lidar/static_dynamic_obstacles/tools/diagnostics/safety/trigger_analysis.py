"""Analyse `trigger_counterfactual.py` output: trigger precision/recall, the
counterfactual value of each fire, an oracle upper bound, and a policy-aware gate.

    python tools/diagnostics/safety/trigger_analysis.py

Definitions (per version, v4 and v7):
* an episode **fires** if the shadow filter would override at least once;
* a fire is **unnecessary** if the policy alone reaches the goal in that episode
  (the main run *is* the policy-alone continuation), else **necessary**;
* precision = P(policy alone fails | episode fires); recall = P(episode fires |
  policy alone fails); false-alarm rate = P(fires | policy alone succeeds);
* a branched fire is **helpful** if the filter-on replay from it reaches the goal
  where the policy alone fails, **harmful** if it fails where the policy alone
  succeeds.

**Oracle trigger** (not deployable): act at the first fire only in episodes the
policy alone fails.  Its gain is the number of helpful first-fire branches; it
breaks nothing.  **Always-on** (the current trigger): act from the first fire in
every firing episode; net = helpful - harmful first-fire branches.

**Gate.** Fire only when the filter would fire *and* the policy itself looks to
be in trouble, judged from onboard features at that step: SAC's critic Q(s,
pi(s)), its 2-s trend, the policy's action spread, speed, the filter's margins
and the risk monitor.  Fitted on a scenario-level split (half the cases by a
fixed hash), validated on the other half.  The gate is applied to the first
would-fire step of each episode; its offline estimate uses that step's branch.
This is an estimate only: a gate changes later decisions too, so the closed-loop
check (protocol step 5) is the measurement.

Writes `results/safety_dev/trigger_counterfactual/analysis.txt` and `gate.json`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "results" / "safety_dev" / "trigger_counterfactual"
FEATURES = ("q", "q_trend", "q_delta", "log_std", "speed", "policy_margin", "best_margin",
            "rm_urgent", "rm_persistent")


def _split(case: str) -> str:
    return "fit" if int(hashlib.sha256(case.encode()).hexdigest(), 16) % 2 == 0 else "val"


def load():
    ep = pd.read_csv(D / "episodes.csv")
    st = pd.read_csv(D / "steps.csv")
    br = pd.read_csv(D / "branches.csv") if (D / "branches.csv").exists() else pd.DataFrame()
    ep["policy_goal"] = ep.outcome == "goal"
    st = st.sort_values(["case", "step"])
    st["q_trend"] = st.q - st.groupby("case").q.shift(4).fillna(st.q)
    return ep, st, br


def trigger_table(ep, br, v):
    fires = ep[f"{v}_fires"] > 0
    ok, bad = ep.policy_goal, ~ep.policy_goal
    first = br[(br.version == v) & (br.fire_index == 1)] if len(br) else pd.DataFrame()
    helpful = int(((first.main_outcome != "goal") & (first.branch_outcome == "goal")).sum()) if len(first) else 0
    harmful = int(((first.main_outcome == "goal") & (first.branch_outcome != "goal")).sum()) if len(first) else 0
    return {
        "episodes": len(ep), "policy_fail": int(bad.sum()),
        "fires_any": int(fires.sum()),
        "precision": round(float(bad[fires].mean()), 3) if fires.any() else float("nan"),
        "recall": round(float(fires[bad].mean()), 3) if bad.any() else float("nan"),
        "false_alarm_rate": round(float(fires[ok].mean()), 3) if ok.any() else float("nan"),
        "first_fire_branches": len(first), "helpful": helpful, "harmful": harmful,
        "net_always_on": helpful - harmful, "net_oracle": helpful,
    }


def _first_fire_rows(st, ep, br, v):
    """Features at each episode's first would-fire step, with its branch outcome."""
    rows = st[st[f"{v}_fire"]].groupby("case").head(1).copy()
    rows = rows.merge(ep[["case", "policy_goal", "set"]], on="case")
    b = br[(br.version == v) & (br.fire_index == 1)][["case", "step", "branch_outcome"]]
    rows = rows.merge(b, on=["case", "step"], how="left")
    for f in ("policy_margin", "best_margin"):
        rows[f] = rows[f"{v}_{f}"].astype(float).replace([np.inf, -np.inf], np.nan)
    for f in ("rm_urgent", "rm_persistent"):
        rows[f] = rows[f].astype(float) if f in rows else 0.0
    rows["split"] = rows.case.map(_split)
    return rows


def _gate_eval(rows, thr, feature="q"):
    """Act at the first fire only if feature < thr; returns (gained, broken, acted)."""
    act = rows[feature] < thr
    gained = int((act & ~rows.policy_goal & (rows.branch_outcome == "goal")).sum())
    broken = int((act & rows.policy_goal & (rows.branch_outcome != "goal")).sum())
    return gained, broken, int(act.sum())


def fit_gate(rows):
    """Threshold on one feature, chosen on the fit half by (gained - broken)."""
    fit, val = rows[rows.split == "fit"], rows[rows.split == "val"]
    out = {}
    for feature in ("q", "q_trend"):
        cands = np.unique(np.quantile(fit[feature].dropna(), np.linspace(0.02, 1.0, 50)))
        best = max(cands, key=lambda t: (_gate_eval(fit, t, feature)[0] - _gate_eval(fit, t, feature)[1], -t))
        g_fit, b_fit, a_fit = _gate_eval(fit, best, feature)
        g_val, b_val, a_val = _gate_eval(val, best, feature)
        g_all, b_all, a_all = _gate_eval(val, np.inf, feature)
        out[feature] = {"threshold": float(best), "fit": {"gained": g_fit, "broken": b_fit, "acted": a_fit, "n": len(fit)},
                        "val": {"gained": g_val, "broken": b_val, "acted": a_val, "n": len(val)},
                        "val_always_on": {"gained": g_all, "broken": b_all, "acted": a_all}}
    return out


def main():
    ep, st, br = load()
    lines = [f"Trigger counterfactuals: {len(ep)} episodes ({ep.policy_goal.sum()} policy-alone goals), "
             f"{len(br)} branches", ""]
    for s, x in [("all", ep)] + list(ep.groupby("set")):
        lines.append(f"-- {s}")
        for v in ("v4", "v7"):
            lines.append(f"   {v}: " + json.dumps(trigger_table(x, br[br.case.isin(x.case)] if len(br) else br, v)))
    lines.append("")
    gates = {}
    for v in ("v4", "v7"):
        rows = _first_fire_rows(st, ep, br, v)
        known = rows.dropna(subset=["branch_outcome"])
        lines += [f"-- {v}: first-fire features (median), policy-alone fails vs succeeds",
                  known.groupby("policy_goal")[["q", "q_trend", "log_std", "speed", "policy_margin",
                                                "best_margin"]].median().round(3).to_string(), ""]
        if len(known) >= 20 and known.policy_goal.nunique() == 2:
            gates[v] = fit_gate(known)
            for f, g in gates[v].items():
                lines.append(f"   gate on {f} < {g['threshold']:.3f}: validation gained {g['val']['gained']}, "
                             f"broken {g['val']['broken']} (acts on {g['val']['acted']} of {g['val']['n']}); "
                             f"always-on on the same half: gained {g['val_always_on']['gained']}, "
                             f"broken {g['val_always_on']['broken']}")
            lines.append("")
    text = "\n".join(lines) + "\n"
    (D / "analysis.txt").write_text(text, encoding="utf-8")
    (D / "gate.json").write_text(json.dumps(gates, indent=1))
    print(text)


if __name__ == "__main__":
    main()
