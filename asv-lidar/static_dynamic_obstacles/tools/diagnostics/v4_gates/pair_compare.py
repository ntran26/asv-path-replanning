"""The v4.3 SAC pair: compare the two arms on the development sets (2026-10-04).

Arms (configs/finetune_v4_pair_v43.json, finetune_v4_pair_v3.json): SAC
baseline-v3 seed 0 continued from its final 3.0 M model and replay buffer by
+0.5 M steps, with the same reset seeds and the same development sets
(formulation v4.2: frozen-like 120 + field 210, near-impossible episodes
replaced); the v4.3 arm trains on v4's stage 7 with hard-state starts, the v3
arm on v3's stage 7.

Each arm's checkpoints were evaluated on the development sets during training
(`eval_episodes.csv`, safety off, episode seeds 900,000 + position).  This
compares the arms at each one's selected checkpoint (the callback's own rule:
goal - 2 x collision over all 330) and at the final 3.5 M checkpoint, paired by
episode.  Gate, fixed before the runs: the v4.3 arm minus the v3 arm at the
selected checkpoints, field set >= +5 points and frozen-like set >= -2 points.
Also reported, not gated (decision, 2026-10-04): the no-target episodes (50:
frozen-like 20, field 30) against the v3 start model -- SAC v4 must not pass
fewer than v3's baseline.

    python tools/diagnostics/v4_gates/pair_compare.py

Writes `results/v4_gates/pair_summary.txt`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import pandas as pd

ARMS = {"v4.3": ROOT / "runs" / "sac_formulation_seed0_bl3_v43pair",
        "v3": ROOT / "runs" / "sac_formulation_seed0_bl3_v3pair"}
OUT = ROOT / "results" / "v4_gates" / "pair_summary.txt"
FIELD_GATE, FROZEN_GATE = 0.05, -0.02


def load(run: Path) -> pd.DataFrame:
    d = pd.read_csv(run / "eval_episodes.csv")
    d = d[d.supervisor.astype(str) == "off"] if "supervisor" in d else d
    d["goal"] = d.outcome == "goal"
    d["coll"] = d.outcome.astype(str).str.startswith("collision")
    d["pos"] = d.groupby("timesteps").cumcount()
    d["crossing"] = d["class"].astype(str) == "crossing"
    d["no_target"] = (d["class"].astype(str) == "no_target") | d.case_id.astype(str).str.contains("-NT-")
    return d


def selected(d: pd.DataFrame) -> int:
    score = d.groupby("timesteps").apply(lambda x: x.goal.mean() - 2.0 * x.coll.mean(), include_groups=False)
    return int(score.idxmax())


def rates(x: pd.DataFrame) -> dict:
    f, z = x[x.set == "field"], x[x.set == "dev"]
    return {"all": x.goal.mean(), "field": f.goal.mean(), "frozen_like": z.goal.mean(),
            "field_crossing": f[f.crossing].goal.mean(), "collision": x.coll.mean(),
            "no_target_passed": int(x[x.no_target].goal.sum()), "no_target_n": int(x.no_target.sum())}


def main() -> int:
    arms = {name: load(run) for name, run in ARMS.items() if (run / "eval_episodes.csv").exists()}
    if len(arms) < 2:
        print("both arms need eval_episodes.csv"); return 1
    lines = ["v4.3 SAC pair: development sets of formulation v4.2 (frozen-like 120 + field 210), safety off", ""]
    start = {n: int(d.timesteps.min()) for n, d in arms.items()}
    rows = []
    for name, d in arms.items():
        for label, ts in (("start", start[name]), ("selected", selected(d)), ("final", int(d.timesteps.max()))):
            rows.append({"arm": name, "checkpoint": label, "timesteps": ts, **rates(d[d.timesteps == ts])})
    t = pd.DataFrame(rows).set_index(["arm", "checkpoint"])
    lines += [t.round(3).to_string(), ""]
    for label in ("selected", "final"):
        a, b = arms["v4.3"], arms["v3"]
        ta = t.loc[("v4.3", label), "timesteps"]
        tb = t.loc[("v3", label), "timesteps"]
        xa = a[a.timesteps == ta].set_index("pos")
        xb = b[b.timesteps == tb].set_index("pos")
        common = xa.index.intersection(xb.index)
        xa, xb = xa.loc[common], xb.loc[common]
        df = rates(xa)["field"] - rates(xb)["field"]
        dz = rates(xa)["frozen_like"] - rates(xb)["frozen_like"]
        dc = rates(xa)["field_crossing"] - rates(xb)["field_crossing"]
        gained = int((xa.goal & ~xb.goal).sum())
        lost = int((~xa.goal & xb.goal).sum())
        verdict = "PASS" if df >= FIELD_GATE and dz >= FROZEN_GATE else "FAIL"
        lines.append(f"{label}: v4.3 {ta:,} vs v3 {tb:,} -- field {df:+.1%}, frozen-like {dz:+.1%}, "
                     f"field crossings {dc:+.1%}; episodes gained {gained}, lost {lost} of {len(common)}"
                     + (f"  -> gate {verdict} (field >= +5, frozen-like >= -2 points)" if label == "selected" else ""))
    base = rates(arms["v3"][arms["v3"].timesteps == start["v3"]])
    for name, d in arms.items():
        r = rates(d[d.timesteps == selected(d)])
        flag = "ok" if r["no_target_passed"] >= base["no_target_passed"] else "BELOW the v3 baseline"
        lines.append(f"no-target, {name} selected checkpoint: {r['no_target_passed']} of {r['no_target_n']} "
                     f"(v3 start model {base['no_target_passed']} of {base['no_target_n']}) -> {flag}")
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
