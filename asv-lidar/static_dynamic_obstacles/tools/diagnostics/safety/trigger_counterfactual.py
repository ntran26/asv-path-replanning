"""Trigger counterfactuals: when does the safety layer fire, and was it needed?
(`planning/SAFETY_LAYER_NOTES.md`, "NEXT DIRECTION", 2026-10-03.)

For each episode the SAC policy drives **alone** (safety off).  Safety v4 and v7
run in **shadow mode**: every step a copy of each filter decides on the policy's
action, and the policy's action is executed regardless.  The shadow filter then
advances as if it had passed the action: its perception and observer see the
real trajectory; its actuator model is rewound to the pre-decision history and
advanced with the executed command; any recovery it started is released.  The
state before the first would-fire step is therefore exact; later fires are
conditioned on a policy-alone history, which is what the trigger question is
about.

At the first would-fire step of each version, and every `BRANCH_EVERY`-th
would-fire step after it (at most `MAX_BRANCHES`), the environment and the
filter are deep-copied (about 4 ms; the copy steps identically) and the episode
is replayed from that step **with the filter acting**.  The main run is itself
the policy-alone continuation, so each fire gets two outcomes:

* necessary   -- the policy alone fails (the main episode is not a goal);
* unnecessary -- the policy alone reaches the goal;
* helpful / harmful -- the filter-on branch reaches the goal where the policy
  alone fails / fails where the policy alone succeeds.

Per step the runner logs onboard features only: each version's would-fire flag,
reason and margins, the V7 risk-monitor evidence, SAC's Q(s, pi(s)) and the
policy's action spread.  No scenario label or future information enters a
logged feature.

Sets (development evidence by designation): the v3 field development
set (150; episode seeds 900120+i, as `dev_eval.py`), and test set v2's 128 SAC
failures plus its 151 matched successful controls
(`results/safety_dev/testset_v2_offline/analysis/`), seeds as in the set.

    python tools/diagnostics/safety/trigger_counterfactual.py [--limit N] [--processes 2]

Writes `results/safety_dev/trigger_counterfactual/{steps,episodes,branches}.csv`
(appended per episode, so an interrupted run keeps what it finished; rerunning
skips finished episodes).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]

import numpy as np
import pandas as pd

import constants as cfg
import curriculum
import train_formulation as tf

OUT = ROOT / "results" / "safety_dev" / "trigger_counterfactual"
MODEL = ROOT / "runs" / "sac_formulation_seed0_bl3" / "kept_best_3M" / "best_model.zip"
VERSIONS = ("v4", "v7")
BRANCH_EVERY = 4
MAX_BRANCHES = 4
_W = {}

LAST_KEYS = ("mode", "why", "policy_margin", "best_margin", "checked_clearance", "plan_steps",
             "recovery_steps", "v6_proposed_change", "v7_policy_preserved", "v7_repair_clearance",
             "v9_current_plan_checked", "v9_override_suppressed", "v9_parent_why",
             "v9_proposed_clearance", "v9_policy_tail_clearance",
             "v9_proposed_first_violation", "v9_policy_first_violation",
             "v9_target_turn_rate_deg_s")


def _outcome(info) -> str:
    if info.get("reached_goal"):
        return "goal"
    if info.get("collided"):
        return f"collision:{info.get('collision_kind')}"
    return "timeout"


def episodes():
    """(case id, built, seed, set name) for every episode, in a fixed order."""
    import formulation_v3 as fv
    out = [(f"DV3:{b.case_id}", b, 900_000 + 120 + i, "dv3") for i, b in enumerate(fv.field_development_set())]
    with open(ROOT / "results" / "test_set" / "v2" / "set_v2.0.pkl", "rb") as fh:
        kept, _ = pickle.load(fh)
    by_id = {it["test_id"]: it for it in kept}
    an = ROOT / "results" / "safety_dev" / "testset_v2_offline" / "analysis"
    listed = set()
    for name in ("failures", "successful_controls"):
        for tid in pd.read_csv(an / f"{name}.csv").test_id:
            it = by_id[tid]
            listed.add(tid)
            out.append((f"TS2:{tid}", it["built"], int(it["episode_seed"]), f"ts2_{name}"))
    # The rest of test set v2 (SAC successes not chosen as controls): with these the
    # first-fire replays give the filters' closed-loop result on all 1,000 episodes.
    for tid, it in sorted(by_id.items()):
        if tid not in listed:
            out.append((f"TS2:{tid}", it["built"], int(it["episode_seed"]), "ts2_rest"))
    return out


def _apply_overrides(overrides):
    """`module.NAME=value` run-time switches for a variant (e.g. safety_v2.HOLD_BACK_GAIN_S=inf)."""
    import importlib
    for item in filter(None, (overrides or "").split(",")):
        target, value = item.split("=", 1)
        mod, name = target.rsplit(".", 1)
        m = importlib.import_module(mod)
        setattr(m, name, type(getattr(m, name))(float(value) if value in ("inf", "-inf") else eval(value)))


def _init(max_branches=MAX_BRANCHES, overrides="", versions=VERSIONS):
    _W["max_branches"] = int(max_branches)
    _W["versions"] = tuple(versions)
    _apply_overrides(overrides)
    import torch
    torch.set_num_threads(1)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    cfg.SAFETY_VERSION = 7          # only read by branches, whose filter object is supplied
    from common import load_model
    from env import ASVLidarEnv
    _W["model"] = load_model(str(MODEL))
    _W["env"] = ASVLidarEnv(render_mode=None, emergency_stop=False)


def _critic(obs):
    import torch
    model = _W["model"]
    o, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        mean, log_std, _ = model.policy.actor.get_action_dist_params(o)
        a = model.policy.actor(o, deterministic=True)
        q1, q2 = model.critic(o, a)
    return float(torch.min(q1, q2)), float(log_std.mean())


def _version_number(version):
    if version not in ("v4", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"):
        raise ValueError(f"Unsupported counterfactual filter: {version}")
    return int(version[1:])


def _new_filter(version):
    import importlib
    number = _version_number(version)
    module = importlib.import_module(f"safety_v{number}")
    return getattr(module, f"SafetyFilterV{number}")()


def _branch(env, filt, obs, version):
    """Replay from a saved state with the filter acting; returns (outcome, interventions, steps)."""
    cfg.SAFETY_VERSION = _version_number(version)
    env.estop_enabled = True
    env._safety_v2 = filt
    env.safety_v2_steps = 0
    actor = tf.EpisodeActor(_W["model"])
    n = 0
    while True:
        obs, _, term, trunc, info = env.step(actor(obs))
        n += 1
        if term or trunc:
            return _outcome(info), int(getattr(env, "safety_v2_steps", 0)), n


def run_case(job):
    case, built, seed, set_name = job
    env, model = _W["env"], _W["model"]
    env.estop_enabled = False
    env._safety_v2 = None
    obs, _ = env.reset(seed=seed, options={"generated": built})
    actor = tf.EpisodeActor(model)
    versions = _W.get("versions", VERSIONS)
    filters = {v: _new_filter(v) for v in versions}
    fires = {v: 0 for v in versions}
    saved = {v: [] for v in versions}
    rows, t, q_prev = [], 0, None
    t0 = time.time()
    while True:
        a = actor(obs)
        q, log_std = _critic(obs)
        row = {"case": case, "step": t, "q": q, "q_delta": (q - q_prev) if q_prev is not None else 0.0,
               "log_std": log_std, "speed": float(env.u_body), "rudder_cmd": float(a[0]), "throttle_cmd": float(a[1])}
        q_prev = q
        for v in versions:
            before = copy.deepcopy(filters[v])
            trial = copy.deepcopy(before)
            out, changed = trial.filter(env, a)
            env._v2_brake = False
            last = trial.last or {}
            row[f"{v}_fire"] = bool(changed)
            for k in LAST_KEYS:
                if k.startswith("v9_") and v != "v9":
                    continue  # Preserve existing V4/V7 trace columns.
                val = last.get(k)
                if isinstance(val, (bool, int, float, str, np.floating, np.integer)) or val is None:
                    row[f"{v}_{k}"] = val
            rm = last.get("risk_monitor")
            if v == "v7" and isinstance(rm, dict):
                for k, val in rm.items():
                    if isinstance(val, (bool, int, float, str)) or val is None:
                        row[f"rm_{k}"] = val
            if changed:
                fires[v] += 1
                if (fires[v] == 1 or (fires[v] - 1) % BRANCH_EVERY == 0) and len(saved[v]) < _W.get("max_branches", MAX_BRANCHES):
                    saved[v].append((t, fires[v], copy.deepcopy(env), copy.deepcopy(before), copy.deepcopy(obs)))
                # Shadow: the filter passes the policy's action after all.
                trial.actuators = copy.deepcopy(before.actuators)
                trial.actuators.issue(env, float(a[0]))
                trial._release()
            filters[v] = trial
        obs, _, term, trunc, info = env.step(a)
        for f in filters.values():
            hook = getattr(f, "observe_issued_command", None)
            if hook is not None:
                hook(env.rudder / 100.0, env.rpm)
        rows.append(row)
        t += 1
        if term or trunc:
            break
    main = _outcome(info)
    branches = []
    for v in versions:
        for k, nth, env_k, filt_k, obs_k in saved[v]:
            outcome, interventions, n = _branch(env_k, filt_k, obs_k, v)
            branches.append({"case": case, "set": set_name, "version": v, "step": k, "fire_index": nth,
                             "main_outcome": main, "branch_outcome": outcome,
                             "branch_interventions": interventions, "branch_steps": n})
    ep = {"case": case, "set": set_name, "seed": seed, "outcome": main, "steps": t,
          "wall_s": round(time.time() - t0, 1), **{f"{v}_fires": fires[v] for v in versions},
          **{f"{v}_first_fire": next((r["step"] for r in rows if r[f"{v}_fire"]), None) for v in versions}}
    return ep, rows, branches


def _append(path, frame):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--cases", default="", help="comma-separated case ids (smoke tests)")
    ap.add_argument("--sets", default="", help="comma-separated set names to run (default all)")
    ap.add_argument("--versions", default=",".join(VERSIONS))
    ap.add_argument("--overrides", default="", help="variant switches, e.g. safety_v2.HOLD_BACK_GAIN_S=inf")
    ap.add_argument("--tag", default="", help="output subfolder for a variant")
    ap.add_argument("--only-fired", default="", help="run only episodes where this version fired in the base run")
    ap.add_argument("--max-branches", type=int, default=MAX_BRANCHES,
                    help="replays per version and episode (1 = first fire only: the closed-loop outcome)")
    a = ap.parse_args()
    versions = tuple(a.versions.split(","))
    if not versions or len(set(versions)) != len(versions):
        ap.error("Use distinct filter versions")
    for version in versions:
        try:
            _version_number(version)
        except ValueError as error:
            ap.error(str(error))
    global OUT
    base = OUT
    if a.tag:
        OUT = OUT / a.tag
    OUT.mkdir(parents=True, exist_ok=True)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    jobs = episodes()
    if a.only_fired:
        fired = pd.read_csv(base / "episodes.csv")
        fired = set(fired[fired[f"{a.only_fired}_fires"] > 0].case)
        jobs = [j for j in jobs if j[0] in fired]
    if a.cases:
        want = set(a.cases.split(","))
        jobs = [j for j in jobs if j[0] in want]
    if a.sets:
        jobs = [j for j in jobs if j[3] in set(a.sets.split(","))]
    done = set(pd.read_csv(OUT / "episodes.csv").case) if (OUT / "episodes.csv").exists() else set()
    jobs = [j for j in jobs if j[0] not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} episodes to run ({len(done)} already done), {a.processes} processes", flush=True)
    (OUT / "config.json").write_text(json.dumps({"model": str(MODEL), "versions": versions,
                                                 "branch_every": BRANCH_EVERY, "max_branches": a.max_branches, "overrides": a.overrides, "tag": a.tag,
                                                 "started": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.processes, initializer=_init, initargs=(a.max_branches, a.overrides, versions)) as pool:
        futures = [pool.submit(run_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            ep, rows, branches = fut.result()
            _append(OUT / "episodes.csv", pd.DataFrame([ep]))
            _append(OUT / "steps.csv", pd.DataFrame(rows))
            if branches:
                _append(OUT / "branches.csv", pd.DataFrame(branches))
            print(f"[{i}/{len(jobs)}] {ep['case']} {ep['outcome']} steps {ep['steps']} "
                  f"fires v4 {ep.get('v4_fires', '-')} v7 {ep.get('v7_fires', '-')} branches {len(branches)} "
                  f"({ep['wall_s']} s; {(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
