"""Field suite: is a rule-consistent passing possible where the deployed policy does not pass so?

For a field test case (`src/field_test_cases.py`), from the same start, plant and seed:

1. **Room for the rule.**  The oracle's manoeuvre library (`oracle_feasibility`: perfect
   foresight, the true vessel model, the policy's action space, an A* route around the
   panels), started at or after the moment the onboard tracker first holds the target, every
   rollout's own and target tracks scored by the Section 3.4.3 rule conditions
   (`colregs_compliance.episode`, engagement from the tracking time).  Counted: rollouts that
   reach the goal with the oracle's 0.20 m margin, and of those the ones meeting the rule
   conditions, by family (starboard, port, speed only; for being overtaken the route held at
   cruise, which is the stand-on).
2. **Without the panels.**  The deployed policy on the same encounter and leg with the panels
   removed, on the case's 10 nominal seeds: goals, and goals meeting the rule conditions.

    python tools/diagnostics/field_suite_rule_feasibility.py medium_1 medium_2 hard_5 [--processes 3]
    python tools/diagnostics/field_suite_rule_feasibility.py medium_1:bare   # the oracle without the panels

Writes `results/field_suite/rule_feasibility.json`.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))
sys.path.insert(0, str(ROOT / "tools" / "diagnostics"))

import colregs_compliance as cc  # noqa: E402
import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import field_test_cases as ftc  # noqa: E402
import oracle_feasibility as of  # noqa: E402
import train_formulation as tf  # noqa: E402
from common import load_model, run_episode  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

OUT = ROOT / "results" / "field_suite" / "rule_feasibility.json"
MODEL = ROOT / "runs" / "sac_formulation_seed0_bl3" / "best_model.zip"
DPSI = (-70.0, -40.0, -20.0, 0.0, 20.0, 40.0, 70.0)
TAU = (-1.0, -0.5, 0.0)
DURATION = (4.0, 8.0, 14.0)
START_SPAN_S, START_STEP_S = 8.0, 1.0


def state(env) -> tuple:
    t = env.targets[0] if env.targets else None
    return (env.asv_x, env.asv_y, env.asv_h, env.u_body,
            t.x if t else np.nan, t.y if t else np.nan, t.heading if t else np.nan)


def rollout(env0, route, t0: float, plan) -> Dict:
    env = copy.deepcopy(env0)
    states, k = [state(env)], 0
    while k < int(round(t0 / cfg.UPDATE_RATE)):
        end = of.physics_step(env, *route.action(env, 0.0))
        k += 1
        states.append(state(env))
        if end is not None:
            return {"outcome": end, "clearance": 0.0, "states": states, "path": env.path.points}
    kk, clear = 0, math.inf
    while True:
        r, tau = plan(env, kk)
        end = of.physics_step(env, r, tau)
        kk += 1
        states.append(state(env))
        if end is not None and end != "goal":
            return {"outcome": end, "clearance": 0.0, "states": states, "path": env.path.points}
        clear = min(clear, of.clearance(env))
        if end == "goal":
            return {"outcome": "goal", "clearance": clear, "states": states, "path": env.path.points}


def rules(built, res: Dict, t_track: float) -> Dict:
    s = np.asarray(res["states"], dtype=float)
    k_track = int(math.ceil(t_track / cfg.UPDATE_RATE)) if math.isfinite(t_track) else len(s)
    tr = {"own_x": s[:, 0], "own_y": s[:, 1], "own_h": s[:, 2] % 360.0, "own_u": s[:, 3],
          "tgt_x": s[:, 4], "tgt_y": s[:, 5], "tgt_h": s[:, 6] % 360.0,
          "engaged": np.arange(len(s)) >= k_track, "path": np.asarray(res["path"], dtype=float)}
    side = (built.flags or {}).get("side", "")
    return cc.episode({"class": built.encounter_class, "crossing_side": side, "outcome": res["outcome"],
                       "variant": "FIX", "test_id": built.case_id}, tr)


def family(dpsi: float, tau: float) -> str:
    return "starboard" if dpsi > 0 else "port" if dpsi < 0 else ("stand_on" if tau == 0.0 else "speed")


def assess_case(name: str) -> Dict:
    import torch
    torch.set_num_threads(1)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    name, _, variant = name.partition(":")
    built, seed = ftc.TestCase().scenario(name)
    if variant == "bare":               # the oracle on the same encounter with the panels removed
        built = copy.deepcopy(built)
        built.flags = {k: v for k, v in (built.flags or {}).items() if k != "fixed_obstacles"}
        built.n_obstacles, built.obstacles = 0, ()
    env = ASVLidarEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0, vessel_randomisation=None)
    env.reset(seed=seed, options={"generated": copy.deepcopy(built)})
    env.estop_enabled = False
    route = of.Route(env)
    t_track, track_range = of.track_time(env, route)
    t_first = of._snap_time(t_track)
    starts = [t_first + i * START_STEP_S for i in range(int(START_SPAN_S / START_STEP_S) + 1)]
    counts: Dict[str, Dict[str, int]] = {}
    examples: Dict[str, str] = {}
    for t0 in starts:
        for dpsi in DPSI:
            for tau in TAU:
                for dur in DURATION:
                    if dpsi == 0.0 and tau == 0.0 and dur != DURATION[0]:
                        continue
                    res = rollout(env, route, t0, of.manoeuvre(route, dpsi, tau, dur))
                    fam = family(dpsi, tau)
                    c = counts.setdefault(fam, {"rollouts": 0, "goal_margin": 0, "goal_margin_rule": 0})
                    c["rollouts"] += 1
                    if res["outcome"] == "goal" and res["clearance"] >= of.MARGIN_M:
                        c["goal_margin"] += 1
                        if rules(built, res, t_track).get("rule_compliant") == 1.0:
                            c["goal_margin_rule"] += 1
                            examples.setdefault(fam, f"t0={t0:g} s dpsi={dpsi:+g} tau={tau:+g} D={dur:g}")
    # the deployed policy without the panels
    model = load_model(MODEL)
    bare = copy.deepcopy(built)
    bare.flags = {k: v for k, v in (bare.flags or {}).items() if k != "fixed_obstacles"}
    bare.n_obstacles, bare.obstacles = 0, ()
    policy = {"with_panels": {"goals": 0, "goals_rule": 0}, "without_panels": {"goals": 0, "goals_rule": 0}}
    for label, b in (("with_panels", built), ("without_panels", bare)):
        for j in range(10):
            r = run_episode(env, copy.deepcopy(b), seed + 10_000 * j, "model", model, keep_traj=True)
            if r["outcome"] == "goal":
                policy[label]["goals"] += 1
                tr = r["_traj"]
                out = cc.episode({"class": b.encounter_class, "crossing_side": (b.flags or {}).get("side", ""),
                                  "outcome": r["outcome"], "variant": "FIX", "test_id": name}, tr)
                policy[label]["goals_rule"] += int(out.get("rule_compliant") == 1.0)
                policy[label].setdefault("first_manoeuvre", []).append(out.get("first_manoeuvre"))
                policy[label].setdefault("target_side_at_cpa", []).append(out.get("target_side_at_cpa"))
    return {"case": name + (":" + variant if variant else ""), "encounter": built.encounter_class, "side": (built.flags or {}).get("side", ""),
            "layout": ftc.CASES[name]["layout"], "t_track_s": t_track, "track_range_m": track_range,
            "oracle_starts_s": [round(t, 1) for t in starts], "oracle": counts, "oracle_rule_examples": examples,
            "policy": policy}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    with Pool(min(args.processes, len(args.cases))) as pool:
        out = pool.map(assess_case, args.cases)
    previous = json.loads(OUT.read_text()) if OUT.exists() else []
    merged = {r["case"]: r for r in previous}
    merged.update({r["case"]: r for r in out})
    OUT.write_text(json.dumps(list(merged.values()), indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
