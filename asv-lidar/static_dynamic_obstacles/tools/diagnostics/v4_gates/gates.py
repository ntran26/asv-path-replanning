"""Validation gates G1-G3 for baseline-v4 (`planning/BASELINE_V4_PLAN.md`, 2026-10-02).

Development side only: conflict scenarios are fresh near-deployment field
layouts from training seeds (namespace "v4gates", generator seeds 395,000+), and
the frozen-like set is the formulation's development set.  The test set, the
frozen suite and the Paper 2 set are never used.

    python tools/diagnostics/v4_gates/gates.py g1    # reward ordering (scripted behaviours, v3 vs fix 1)
    python tools/diagnostics/v4_gates/gates.py g2    # fix-1 flags: recall on conflicts, false flags elsewhere
    python tools/diagnostics/v4_gates/gates.py g3    # v4 curriculum draws (v3 for reference)

Writes `results/v4_gates/g<n>_*.csv` and `g<n>_summary.txt`.

**G1.** Each conflict scenario runs five behaviours -- compliant (the COLREGs
alteration, `colregs_action(+1)`), wrong way (`-1`), slow (path follower,
propulsion at the floor while an encounter is engaged), COLREGs-VO and
LOS-DWA -- with fix 1 off (v3) and on (v4).  Returns are discounted with the
learners' gamma (0.951) from the first engaged step: what the policy weighs
when it decides.  Pass: under fix 1 the best-return behaviour reaches the goal
in >= 90 % of the scenarios some behaviour solves.

**G2.** The compliant behaviour does not read the admissibility flags, so its
trajectory is identical with fix 1 on and off and the flags compare state by
state.  Ground truth for "the compliant turn is blocked": the compliant run hits
a panel or the wall.  Pass: fix 1 flags >= 80 % of blocked conflict episodes;
it adds a flag in <= 10 % of the development episodes the compliant run passes.

**G3.** Draws N resets per stage from the v4 curriculum (and v3): field share,
encounter mix, crossing share, panels, own-start contact, space-time redraws,
reset time.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]

import numpy as np
import pandas as pd

import constants as cfg
import curriculum
import train_formulation as tf

OUT = ROOT / "results" / "v4_gates"
GAMMA = 0.9509900498999999
GAMMAS = (0.951, 0.97, 0.98, 0.99)         # G1 also reports the ordering at these discounts
SEED_BASE = 395_000
EPISODE_SEED_BASE = 930_000
BEHAVIOURS = ("compliant", "wrong_way", "slow", "colregs_vo", "los_dwa")
CONFLICT_PLAN = (("HO", "near-L2", 12), ("HO", "other", 6), ("CRS", "any", 8), ("CRP", "any", 8),
                 ("OT", "near-L1", 6))
GIVE_WAY_OR_TURN = ("head_on", "crossing", "overtaking")
_W = {}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def conflict_scenarios():
    import field_training as ft
    import scenario as scn
    codes = ["HO", "CRS", "CRP", "OT"]
    out = []
    for code, want, n in CONFLICT_PLAN:
        got, k = 0, 0
        while got < n and k < 300:
            base = SEED_BASE + codes.index(code) * 10_000 + (2_000 if want == "other" else 0) + k * 20
            k += 1
            built = ft.sample(np.random.default_rng(base), namespace="v4gates", encounter=code, varying=False,
                              generator=scn.ScenarioGenerator(stage=5, seed_namespace="training"),
                              seed_fn=lambda j, base=base: base + 200_000 + j, solvable_only=True, near=True)
            motif = str(built.flags.get("motif", ""))
            if (want.startswith("near-") and motif != want) or (want == "other" and motif == "near-L2"):
                continue
            built.case_id = f"G-{code}-{motif}-{got + 1:02d}"
            out.append({"built": built, "seed": EPISODE_SEED_BASE + len(out), "group": f"{code} {want}",
                        "case": built.case_id})
            got += 1
    return out


def dev_scenarios():
    """The formulation's development set (v2's 120): frozen-like, development namespace."""
    return [{"built": b, "seed": EPISODE_SEED_BASE + 5_000 + i, "group": f"dev {b.encounter_class}",
             "case": b.case_id} for i, b in enumerate(tf.development_set(20))]


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
def _init():
    import torch
    torch.set_num_threads(1)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)


def _flag_state(env):
    """(any engaged turning encounter, compliant turn inadmissible in one of them)."""
    engaged = blocked = False
    for ctx in env.encounter_contexts.values():
        if ctx.engaged and str(ctx.cls) in GIVE_WAY_OR_TURN and ctx.tcpa > 0.0:
            engaged = True
            blocked |= not ctx.turn_admissible
    return engaged, blocked


def _episode(job):
    from common import colregs_action, follower_action, make_controller
    built, seed, behaviour, fix, meta = job
    cfg.ADMISSIBILITY_STATIC = bool(fix)
    env = _W["env"]
    ctrl = make_controller(behaviour) if behaviour in ("colregs_vo", "los_dwa") else None
    obs, _ = env.reset(seed=seed, options={"generated": built})
    rewards, t_engage, eng_steps, flag_steps, t = [], None, 0, 0, 0
    while True:
        if behaviour == "compliant":
            a = colregs_action(env, +1)
        elif behaviour == "wrong_way":
            a = colregs_action(env, -1)
        elif behaviour == "slow":
            a = follower_action(env)
            if any(c.engaged and c.tcpa > 0.0 for c in env.encounter_contexts.values()):
                a = np.array([a[0], -1.0], dtype=np.float32)
        else:
            a = ctrl.action(env, obs)
        obs, r, term, trunc, info = env.step(a)
        rewards.append(float(r))
        engaged, blocked = _flag_state(env)
        if engaged:
            eng_steps += 1
            flag_steps += int(blocked)
            if t_engage is None:
                t_engage = t
        t += 1
        if term or trunc:
            break
    outcome = ("goal" if info.get("reached_goal") else
               f"collision:{info.get('collision_kind')}" if info.get("collided") else "timeout")
    rw = np.asarray(rewards)
    t0 = t_engage or 0
    disc = lambda x: float(np.sum(x * GAMMA ** np.arange(len(x))))
    out = {**meta, "behaviour": behaviour, "fix": bool(fix), "outcome": outcome, "steps": t,
           "return": float(rw.sum()), "disc_return_t0": disc(rw), "disc_return_engage": disc(rw[t0:]),
           "t_engage": t_engage, "engaged_steps": eng_steps, "flagged_steps": flag_steps}
    for g in GAMMAS:                         # the ordering at other discounts, from the same rollout
        out[f"disc_engage_g{g}"] = float(np.sum(rw[t0:] * g ** np.arange(len(rw) - t0)))
    return out


def _run(scens, behaviours, fixes, processes=3):
    jobs = [(s["built"], s["seed"], b, f, {"group": s["group"], "case": s["case"]})
            for s in scens for b in behaviours for f in fixes]
    with ProcessPoolExecutor(max_workers=processes, initializer=_init) as pool:
        return pd.DataFrame(list(pool.map(_episode, jobs, chunksize=1)))


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def g1(processes):
    scens = conflict_scenarios()
    d = _run(scens, BEHAVIOURS, (False, True), processes)
    d.to_csv(OUT / "g1_episodes.csv", index=False)
    lines = [f"G1 reward ordering: {len(scens)} conflict scenarios x {len(BEHAVIOURS)} behaviours x fix off/on", ""]
    d["goal"] = d.outcome == "goal"
    lines += ["-- success by behaviour and group (fix on; outcomes are the same with fix off for scripted runs)",
              d[d.fix].pivot_table(index="group", columns="behaviour", values="goal", aggfunc="mean").round(2).to_string(), ""]
    lines += ["-- mean discounted return from engagement (gamma 0.951)",
              d.pivot_table(index=["fix", "group"], columns="behaviour", values="disc_return_engage",
                            aggfunc="mean").round(1).to_string(), ""]
    rows = []
    for (case, fix), x in d.groupby(["case", "fix"]):
        if not x.goal.any():
            rows.append({"case": case, "fix": fix, "group": x.group.iloc[0], "solvable": False, "best_safe": None,
                         "best": None})
            continue
        best = x.loc[x.disc_return_engage.idxmax()]
        rows.append({"case": case, "fix": fix, "group": x.group.iloc[0], "solvable": True,
                     "best": best.behaviour, "best_safe": bool(best.goal)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "g1_ordering.csv", index=False)
    s = r[r.solvable]
    for fix in (False, True):
        x = s[s.fix == fix]
        lines.append(f"fix {'ON (v4)' if fix else 'off (v3)'}: best-return behaviour is safe in "
                     f"{x.best_safe.mean():.0%} of {len(x)} script-solvable scenarios "
                     f"({(~r[(r.fix == fix)].solvable).sum()} solved by no script)")
        lines.append("   by group: " + ", ".join(f"{g} {v:.0%}" for g, v in x.groupby("group").best_safe.mean().items()))
        lines.append("   best behaviour counts: " + str(x.best.value_counts().to_dict()))
    on = s[s.fix]
    lines += ["", f"G1 {'PASS' if len(on) and on.best_safe.mean() >= 0.90 else 'FAIL'} "
                  f"(threshold 90 % under fix 1)", "", "-- the same ordering at other discounts (fix on)"]
    for g in GAMMAS:
        col = f"disc_engage_g{g}"
        safe = []
        for case, x in d[d.fix].groupby("case"):
            if x.goal.any():
                safe.append(bool(x.loc[x[col].idxmax()].goal))
        lines.append(f"   gamma {g}: best-return behaviour safe in {np.mean(safe):.0%} of {len(safe)}")
    return lines


def g2(processes):
    scens = conflict_scenarios() + dev_scenarios()
    d = _run(scens, ("compliant",), (False, True), processes)
    d.to_csv(OUT / "g2_episodes.csv", index=False)
    on = d[d.fix].set_index("case")
    off = d[~d.fix].set_index("case")
    e = on[["group", "outcome", "engaged_steps"]].copy()
    e["flag_on"] = on.flagged_steps > 0
    e["flag_off"] = off.flagged_steps.reindex(e.index) > 0
    e["added"] = e.flag_on & ~e.flag_off
    e["blocked_truth"] = e.outcome.isin(["collision:obstacle", "collision:boundary"])
    e["conflict_set"] = ~e.group.str.startswith("dev")
    e.to_csv(OUT / "g2_flags.csv")
    lines = ["G2 fix-1 flags (compliant behaviour; identical trajectories with fix off and on)", ""]
    for name, x in (("conflict set", e[e.conflict_set]), ("development set", e[~e.conflict_set])):
        eng = x[x.engaged_steps > 0]
        bt = eng[eng.blocked_truth]
        ok = eng[eng.outcome == "goal"]
        lines += [f"-- {name}: {len(x)} episodes, {len(eng)} with an engaged turning encounter",
                  f"   compliant run blocked by statics: {len(bt)}; flagged with fix on: {bt.flag_on.mean() if len(bt) else float('nan'):.0%} "
                  f"(fix off: {bt.flag_off.mean() if len(bt) else float('nan'):.0%})",
                  f"   compliant run reaches goal: {len(ok)}; fix adds a flag in {ok.added.mean() if len(ok) else float('nan'):.0%}",
                  "   by group (flagged on / added): " + ", ".join(
                      f"{g} {v.flag_on.mean():.0%}/{v.added.mean():.0%}" for g, v in eng.groupby("group")), ""]
    conf = e[e.conflict_set & (e.engaged_steps > 0) & e.blocked_truth]
    dev_ok = e[~e.conflict_set & (e.engaged_steps > 0) & (e.outcome == "goal")]
    recall = conf.flag_on.mean() if len(conf) else 0.0
    false = dev_ok.added.mean() if len(dev_ok) else 0.0
    lines += [f"G2 {'PASS' if recall >= 0.80 and false <= 0.10 else 'FAIL'}: recall {recall:.0%} (>= 80 %), "
              f"added flags on passed development episodes {false:.0%} (<= 10 %)"]
    return lines


def _g3_stage(job):
    overlay, stage, n = job
    import importlib
    mod = importlib.import_module(overlay)
    mod.apply(cfg)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    import paper2_set as p2
    env = ASVLidarEnv(render_mode=None, scenario_stage=stage, emergency_stop=False)
    env.set_scenario_stage(stage)          # as training's curriculum callback does: installs the field mix
    rows = []
    for i in range(n):
        t0 = time.time()
        env.reset(seed=7_000_000 + stage * 10_000 + i)
        dt = time.time() - t0
        sc = env.scenario
        hull = np.asarray(env.hull_polygon(), dtype=float)
        gaps = [p2.polygon_distance(hull, np.asarray(p, dtype=float)) for p in env.obstacles]
        rows.append({"overlay": overlay, "stage": stage, "class": sc.encounter_class,
                     "field": (sc.flags or {}).get("set") == "field_training",
                     "near": str((sc.flags or {}).get("motif", "")).startswith("near-"),
                     "panels": len(env.obstacles), "start_gap": min(gaps) if gaps else np.inf,
                     "start_contact": env.collision_kind(env.hull_polygon()) is not None,
                     "reset_s": dt})
    rows[-1]["st_redraws"] = getattr(env, "st_redraws", 0)
    return rows


def g3(processes, n=200):
    jobs = [(ov, st, n) for ov in ("formulation_v3", "formulation_v4") for st in (5, 6, 7)]
    import multiprocessing as mp
    # One process per job: a process holds one overlay (v3 and v4 cannot coexist).
    with mp.get_context("spawn").Pool(min(processes, len(jobs)), maxtasksperchild=1) as pool:
        d = pd.DataFrame([r for rows in pool.map(_g3_stage, jobs, chunksize=1) for r in rows])
    d.to_csv(OUT / "g3_draws.csv", index=False)
    g = d.groupby(["overlay", "stage"])
    t = pd.DataFrame({"n": g.size(), "field": g.field.mean(), "near_of_field": g.apply(
        lambda x: x[x.field].near.mean() if x.field.any() else 0.0, include_groups=False),
        "crossing": g.apply(lambda x: (x["class"] == "crossing").mean(), include_groups=False),
        "head_on": g.apply(lambda x: (x["class"] == "head_on").mean(), include_groups=False),
        "panels3": g.apply(lambda x: (x.panels >= 3).mean(), include_groups=False),
        "start_contact": g.start_contact.sum(), "start_gap<0.3": g.apply(lambda x: (x.start_gap < 0.3).sum(), include_groups=False),
        "reset_mean_s": g.reset_s.mean(), "reset_p95_s": g.reset_s.quantile(0.95),
        "st_redraws": g.st_redraws.max()}).round(3)
    lines = [f"G3 curriculum draws ({n} resets per stage)", "", t.to_string(), ""]
    v4 = t.loc["formulation_v4"]
    ok = (v4.start_contact.sum() == 0) and (v4.reset_mean_s.max() < 1.0)
    lines.append(f"G3 {'PASS' if ok else 'CHECK'}: v4 own-start contacts {int(v4.start_contact.sum())}, "
                 f"mean reset {v4.reset_mean_s.max():.2f} s (< 1 s); field shares {list(v4.field.round(2))} "
                 f"(design 0.20/0.40/0.55)")
    return lines


def main():
    gate = sys.argv[1]
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lines = {"g1": g1, "g2": g2, "g3": g3}[gate](processes)
    lines.append(f"({time.time() - t0:.0f} s)")
    text = "\n".join(lines) + "\n"
    (OUT / f"{gate}_summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
