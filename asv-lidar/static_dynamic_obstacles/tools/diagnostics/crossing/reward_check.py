"""Reward check on the development crossings (2026-10-04, BASELINE_V4_PLAN.md section 5e).

Question: does the reward itself prefer the response SAC chooses in the crossings
it fails -- a small turn, often with a slow-down -- over manoeuvres that succeed
from the same state?  If it does, retraining under the same reward cannot be
expected to fix the crossings.

For each development crossing of the crossing trace (same scenario and seed),
the kept SAC 3 M policy is replayed to the step at which its perception first
tracks the target, and the environment is copied there.  From that state:

* **SAC**: the policy's own continuation, to the end of the episode;
* **alternatives**: the oracle's manoeuvre library (`src/oracle_feasibility.py`)
  started 0, 1, 2 or 3 s after that moment, screened with the physics-only
  rollout; up to `MAX_ALT` of the successful ones (spread over port turns,
  starboard turns and speed-only manoeuvres) are rerun through the full
  `env.step`, so they collect exactly the reward the policy would.

Returns are discounted from the tracking step, at the learners' gamma 0.951
(effective horizon about 10 s), at 0.98, and undiscounted, and split by reward
term (`info["reward/weighted/<term>"]`; the remainder of the step total is
recorded as `terminal`).  The reward "prefers SAC's failing response" in an
episode when SAC's discounted return beats the best successful alternative.
Successes are run the same way, as the control.

    python tools/diagnostics/crossing/reward_check.py --processes 6

Results: `results/crossing_trace/reward_check_episodes.csv`, `reward_check_terms.csv`,
`reward_check_summary.txt`.  The test set is never used.
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers"), str(ROOT / "tools" / "diagnostics" / "feasibility"),
                str(ROOT / "tools" / "diagnostics" / "v4_gates")]

import numpy as np
import pandas as pd

import curriculum
import train_formulation as tf

MODEL = ROOT / "runs" / "sac_formulation_seed0_bl3" / "kept_best_3M" / "best_model.zip"
OUT = ROOT / "results" / "crossing_trace"
GAMMAS = {"g951": 0.9509900498999999, "g98": 0.98, "g1": 1.0}
STARTS_S = (0.0, 1.0, 2.0, 3.0)        # alternative manoeuvre starts after the first track
MAX_ALT = 16


def _returns(steps):
    """Discounted returns and per-term discounted sums of a list of (total, terms) per step."""
    out = {}
    for name, g in GAMMAS.items():
        w = np.power(g, np.arange(len(steps)))
        out[f"R_{name}"] = float(np.sum(w * np.array([s[0] for s in steps])))
    w = np.power(GAMMAS["g951"], np.arange(len(steps)))
    terms = {}
    for k, (_, t) in enumerate(steps):
        for name, v in t.items():
            terms[name] = terms.get(name, 0.0) + w[k] * v
    return out, terms


def _step_record(reward, info):
    terms = {k.split("/", 2)[2]: float(v) for k, v in info.items() if k.startswith("reward/weighted/")}
    terms["terminal"] = float(reward) - sum(terms.values())
    return float(reward), terms


def _roll_env(env, plan, obs=None, actor=None):
    """Roll the full env.step from its current state to the end; `plan(env, k)` gives
    (rudder, throttle), or the actor acts on `obs`.  Returns (steps, outcome)."""
    steps, k = [], 0
    while True:
        a = actor(obs) if actor is not None else np.array(plan(env, k), dtype=np.float32)
        obs, reward, term, trunc, info = env.step(a)
        steps.append(_step_record(reward, info))
        k += 1
        if term or trunc:
            out = (f"collision:{info['collision_kind']}" if info["collided"] else
                   "goal" if info["reached_goal"] else "timeout")
            return steps, out


def _job(args):
    import oracle_feasibility as of
    from common import _WORKER
    key, built, seed, k_track, goal = args
    env, model = _WORKER["env"], _WORKER["model"]
    obs, _ = env.reset(seed=seed, options={"generated": built})
    actor = tf.EpisodeActor(model)
    for _ in range(k_track):                          # SAC up to the first track
        obs, _, term, trunc, _ = env.step(actor(obs))
        if term or trunc:
            return {"key": key, "error": "ended before the first track"}, []
    snap = copy.deepcopy(env)
    sac_steps, sac_out = _roll_env(env, None, obs=obs, actor=actor)
    rows = [{"key": key, "who": "sac", "family": "sac", "start_s": 0.0, "outcome": sac_out,
             **_returns(sac_steps)[0], "steps": len(sac_steps)}]
    terms = [{"key": key, "who": "sac", **_returns(sac_steps)[1]}]
    # Alternatives: screen the library from the snapshot with the physics-only rollout.
    route = of.Route(snap)
    lib = [m for m in of.library(t0s=STARTS_S)]
    ok = []
    pre, k_done = copy.deepcopy(snap), 0
    for t0 in STARTS_S:
        end = None
        while k_done < int(round(t0 / 0.5)):
            end = of.physics_step(pre, *route.action(pre, 0.0))
            k_done += 1
            if end is not None:
                break
        if end is not None:
            break
        for (_, dpsi, tau, dur) in (m for m in lib if m[0] == t0):
            res = of._run(copy.deepcopy(pre), of.manoeuvre(route, dpsi, tau, dur))
            if res["outcome"] == "goal":
                ok.append((t0, dpsi, tau, dur, res["clearance"]))
    # Spread the reruns over families and starts, best clearance first within each.
    fam = lambda m: "port" if m[1] < 0 else "starboard" if m[1] > 0 else "speed"
    chosen, by = [], {}
    for m in sorted(ok, key=lambda m: -m[4]):
        by.setdefault((fam(m), m[0]), []).append(m)
    while len(chosen) < MAX_ALT and any(by.values()):
        for kf in list(by):
            if by[kf] and len(chosen) < MAX_ALT:
                chosen.append(by[kf].pop(0))
    for (t0, dpsi, tau, dur, clear) in chosen:
        e = copy.deepcopy(snap)
        man = of.manoeuvre(route, dpsi, tau, dur)
        k0 = int(round(t0 / 0.5))
        plan = (lambda env_, k, k0=k0, man=man: route.action(env_, 0.0) if k < k0 else man(env_, k - k0))
        st, out = _roll_env(e, plan)
        r, t = _returns(st)
        rows.append({"key": key, "who": "alt", "family": fam((t0, dpsi)), "start_s": t0, "dpsi": dpsi, "tau": tau,
                     "dur": dur, "clearance": clear, "outcome": out, **r, "steps": len(st)})
        terms.append({"key": key, "who": f"alt:{fam((t0, dpsi))}:{t0:g}:{dpsi:+g}:{tau:+g}:{dur:g}", **t})
    for r in rows:
        r["sac_goal"] = goal
    rows[0]["n_alt_success_screened"] = len(ok)
    return rows, terms


def run(processes: int):
    from common import _init_worker
    import oracle_sets
    ep = pd.read_csv(OUT / "episodes.csv")
    items = {k: b for k, b, s, m in oracle_sets.dev_items()}
    jobs = []
    for r in ep.itertuples():
        if math.isnan(r.t_detect):
            continue
        jobs.append((r.key, items[r.key], int(r.seed), int(round(r.t_detect / 0.5)), bool(r.goal)))
    print(f"[reward check] {len(jobs)} development crossings ({sum(not j[4] for j in jobs)} SAC failures)", flush=True)
    t0 = time.time()
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    with ProcessPoolExecutor(max_workers=processes, initializer=_init_worker,
                             initargs=(str(MODEL), {"EMERGENCY_STOP_ENABLED": False})) as pool:
        res = list(pool.map(_job, jobs, chunksize=1))
    rows = [x for r in res for x in (r[0] if isinstance(r[0], list) else [r[0]])]
    terms = [x for r in res for x in r[1]]
    pd.DataFrame(rows).to_csv(OUT / "reward_check_episodes.csv", index=False)
    pd.DataFrame(terms).to_csv(OUT / "reward_check_terms.csv", index=False)
    print(f"[reward check] done in {time.time() - t0:.0f} s", flush=True)


def report():
    pd.set_option("display.width", 220)
    d = pd.read_csv(OUT / "reward_check_episodes.csv")
    t = pd.read_csv(OUT / "reward_check_terms.csv")
    d = d[d.who.notna()]
    sac = d[d.who == "sac"].set_index("key")
    alt = d[(d.who == "alt") & (d.outcome == "goal")]
    lines = ["Reward check: SAC's continuation from the first track vs successful oracle manoeuvres from the same state",
             f"(development crossings; returns from the first-track step; {len(sac)} episodes)", ""]
    rows = []
    for key, s in sac.iterrows():
        a = alt[alt.key == key]
        row = {"key": key, "sac_goal": bool(s.sac_goal), "sac_outcome": s.outcome, "n_alt": len(a)}
        for g in GAMMAS:
            row[f"sac_{g}"] = s[f"R_{g}"]
            row[f"best_alt_{g}"] = a[f"R_{g}"].max() if len(a) else np.nan
            row[f"prefers_sac_{g}"] = bool(len(a) and s[f"R_{g}"] > a[f"R_{g}"].max())
        for fam in ("port", "starboard", "speed"):
            af = a[a.family == fam]
            row[f"best_{fam}_g951"] = af["R_g951"].max() if len(af) else np.nan
        rows.append(row)
    e = pd.DataFrame(rows)
    e.to_csv(OUT / "reward_check_summary.csv", index=False)
    has = e[e.n_alt > 0]
    for goal, name in ((False, "SAC failures"), (True, "SAC successes (control)")):
        x = has[has.sac_goal == goal]
        lines.append(f"-- {name}: {len(x)} with a successful alternative from the first track "
                     f"({int(((e.sac_goal == goal) & (e.n_alt == 0)).sum())} without)")
        for g in GAMMAS:
            lines.append(f"   reward prefers SAC's own continuation over the best successful alternative "
                         f"at {g}: {int(x[f'prefers_sac_{g}'].sum())} of {len(x)}")
        lines.append("   median discounted return (gamma 0.951): SAC %.1f, best alternative %.1f, best port %.1f, "
                     "best starboard %.1f, best speed-only %.1f" % (
                         x.sac_g951.median(), x.best_alt_g951.median(), x.best_port_g951.median(),
                         x.best_starboard_g951.median(), x.best_speed_g951.median()))
        lines.append("")
    # Term decomposition for failures: SAC vs the best (gamma 0.951) successful alternative.
    t["key"] = t["key"].astype(str)
    fail_keys = has[~has.sac_goal].key
    diffs = []
    for key in fail_keys:
        a = alt[alt.key == key]
        best = a.loc[a.R_g951.idxmax()]
        tag = f"alt:{best.family}:{best.start_s:g}:{best.dpsi:+g}:{best.tau:+g}:{best.dur:g}"
        ts = t[(t.key == key) & (t.who == "sac")].drop(columns=["key", "who"]).iloc[0]
        ta = t[(t.key == key) & (t.who == tag)].drop(columns=["key", "who"])
        if len(ta):
            diffs.append((ts - ta.iloc[0]).rename(key))
    if diffs:
        dd = pd.DataFrame(diffs).fillna(0.0)
        lines += ["-- SAC failures: per-term discounted return, SAC minus best successful alternative (gamma 0.951)",
                  "   (positive: the term pays SAC's failing response more)",
                  dd.median().round(2).sort_values().to_string(), "",
                  "   mean:", dd.mean().round(2).sort_values().to_string(), ""]
    text = "\n".join(lines) + "\n"
    (OUT / "reward_check_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "report"), nargs="?", default="run")
    ap.add_argument("--processes", type=int, default=6)
    a = ap.parse_args()
    if a.what == "run":
        run(a.processes)
    report()


if __name__ == "__main__":
    main()
