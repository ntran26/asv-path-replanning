"""Re-pick one field case under the from-rest protocol (FIELD_SESSIONS_PLAN.md section 7.1).

On the water both ships start from rest and the target is released at an offset from the own
ship. Under that protocol `medium_3` (L1, crossing from port) reached the goal on only 4 of 10
seeds. This re-runs the suite's selection for one cell with the protocol built in:

1. candidates: the cell's kept candidates of the original search (`candidates*.csv`, 10/10 with
   both ships moving at t = 0), then fresh draws from an unused seed block (generator seeds
   980,000 + k, episode seeds 985,000 + k), each first required to reach the goal on all 10
   seeds as verified (so its designed encounter is defined);
2. the release offset: as `tools/diagnostics/field_suite_both_from_rest.py`, the offset that best
   reproduces the designed encounter (closest range and own along-track position at the closest
   approach) on 3 seeds, searched at 2 s steps over -20 to +10 s, then at 1 s around the best;
3. from rest at that offset: the goal on the 10 nominal seeds (identified plant); 10 robust runs
   (plant from the training randomisation, start heading +/-5 deg, lateral +/-0.20 m, release
   +/-1 s); the Section 3.4.3 rule conditions on the first nominal run;
4. selection, the suite's rule with the protocol: kept if 10/10 from rest; then the most robust
   from rest, then the rule conditions met, then the smallest drawn DCPA.

    python tools/tiers/field_suite_repick.py --case medium_3 --processes 4 [--max-draws 400] [--wanted 3]
    python tools/tiers/field_suite_repick.py --case medium_3 --write    # the chosen case into src/field_test_cases.py

Output: results/field_suite/repick_<case>/ (candidates.csv, chosen.json). `--write` replaces that
case's entry in `CASES` (its fields as the suite records them, both ships moving at t = 0) and adds
nothing else to the module.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))
sys.path.insert(0, str(ROOT / "tools" / "diagnostics"))

import constant_temp as ct  # noqa: E402
import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import field_suite as fs  # noqa: E402
import field_test_cases as ftc  # noqa: E402
import paper2_set as p2  # noqa: E402
import train_formulation as tf  # noqa: E402
from common import load_model, run_episode  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

OUT_ROOT = ROOT / "results" / "field_suite"
K_TAU = 5.29                    # s m/s, as field_suite_both_from_rest.py
NEW_GEN_SEED = 980_000
NEW_EP_SEED = 985_000
COARSE = [float(d) for d in range(-20, 11, 2)]


class RestEnv(ASVLidarEnv):
    """Both ships from rest; the target held at its mark and released `offset` s after the own
    ship (offset < 0: the own ship held, propulsion off, for -offset s after the target's release),
    accelerating as v tanh(t / tau), tau = K / v; the target's wall stop held off for its first metre."""
    offset: float = 0.0

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        options["initial_speed"] = 0.0
        obs, info = super().reset(seed=seed, options=options)
        self._t = 0.0
        self._tg = self.targets[0] if self.targets else None
        self._hold = max(0.0, -float(self.offset))
        self._rel = max(0.0, float(self.offset))
        self._box = None
        if self._tg is not None:
            self._v = float(self._tg.speed)
            self._tg.speed = 0.0
            self._box, self._start = self._tg.stop_box, (self._tg.x, self._tg.y)
            self._tg.stop_box = None
        return obs, info

    def step(self, action):
        tg, t = self._tg, self._t
        if tg is not None:
            if self._box is not None and math.hypot(tg.x - self._start[0], tg.y - self._start[1]) >= 1.0:
                tg.stop_box, self._box = self._box, None
            if t >= self._rel - 1e-9:
                tg.speed = self._v * math.tanh((t - self._rel + 0.5 * cfg.UPDATE_RATE) / (K_TAU / max(self._v, 1e-6)))
        if t < self._hold - 1e-9:
            action = np.array([0.0, -1.0], dtype=np.float32)
        out = super().step(action)
        self._t += float(cfg.UPDATE_RATE)
        return out


class RestPerturbedEnv(RestEnv):
    perturb: Dict = {}

    def _apply_initial_conditions(self, options: dict) -> None:
        super()._apply_initial_conditions({**options, **self.perturb})


_W: Dict = {}


def _init() -> None:
    import torch
    torch.set_num_threads(1)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    _W["model"] = load_model(fs.MODEL)
    _W["check"] = ASVLidarEnv(render_mode=None, emergency_stop=False)
    _W["cruise"] = ASVLidarEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0,
                               vessel_randomisation=None)
    _W["rest"] = RestEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0, vessel_randomisation=None)
    _W["robust"] = RestPerturbedEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0,
                                    vessel_randomisation=cfg.VESSEL_RANDOMISATION_SCALE)


def geometry(row: Dict, built) -> Optional[tuple]:
    tr = row.get("_traj")
    if tr is None or "tgt_x" not in tr:
        return None
    d = np.hypot(tr["tgt_x"] - tr["own_x"], tr["tgt_y"] - tr["own_y"])
    k = int(np.argmin(d))
    h = math.radians(float(built.own_heading))
    ox, oy = float(built.own_spawn[0]), float(built.own_spawn[1])
    s = (float(tr["own_x"][k]) - ox) * math.sin(h) + (float(tr["own_y"][k]) - oy) * math.cos(h)
    return float(d[k]), s


def evaluate(job: Dict) -> Dict:
    """One candidate: designed encounter, offset, from-rest nominal and robust, rule conditions."""
    model = _W["model"]
    built = ftc.build(job["name"], job["layout"], job["code"], int(job["generator_seed"]))
    e = int(job["episode_seed"])
    row = {k: job[k] for k in ("name", "layout", "code", "generator_seed", "episode_seed", "source")}
    lo, hi = ct.FIELD_TARGET_SPEED_RANGE
    if built is None or not lo <= float(built.target_speed) <= hi:
        return {**row, "stage": "no scenario"}
    if job["source"] == "new":
        if not fs.start_clear(_W["check"], built, e):
            return {**row, "stage": "start not clear"}
        if not p2.field_feasible(fs.nominal_check(_W["check"], copy.deepcopy(built), e)):
            return {**row, "stage": "not field feasible"}
    row.update(digest=built.digest(), target_x=float(built.target_spawn[0]), target_y=float(built.target_spawn[1]),
               target_heading=float(built.target_heading) % 360.0, target_speed=float(built.target_speed),
               dcpa_m=float(built.dcpa_m), tcpa_s=float(built.tcpa_s))
    # 1. as verified (both moving at t = 0): 10/10 required; the designed encounter on 3 seeds
    refs = []
    for j in range(fs.NOMINAL_SEEDS):
        r = run_episode(_W["cruise"], copy.deepcopy(built), e + 10_000 * j, "model", model, keep_traj=(j < 3))
        if r["outcome"] != "goal":
            return {**row, "stage": "cruise", "cruise_goals": j, "cruise_failure": r["outcome"]}
        if j < 3:
            refs.append(geometry(r, built))
    row.update(cruise_goals=fs.NOMINAL_SEEDS, cruise_min_range=float(np.median([g[0] for g in refs])))
    r_star, s_star = float(np.median([g[0] for g in refs])), float(np.median([g[1] for g in refs]))

    # 2. the release offset by geometry
    def score(d: float) -> float:
        RestEnv.offset = d
        sc = []
        for j in range(3):
            r = run_episode(_W["rest"], copy.deepcopy(built), e + 10_000 * j, "model", model, keep_traj=True)
            g = geometry(r, built)
            sc.append(1e3 if g is None else abs(g[0] - r_star) + abs(g[1] - s_star))
        return float(np.median(sc))

    scores = {d: score(d) for d in COARSE}
    best = min(scores, key=lambda d: (scores[d], abs(d)))
    for d in (best - 1.0, best + 1.0):
        if -20.0 <= d <= 10.0 and d not in scores:
            scores[d] = score(d)
    offset = min(scores, key=lambda d: (scores[d], abs(d)))
    row.update(offset_s=offset, match_m=scores[offset], designed_range_m=r_star, designed_s_m=s_star)

    # 3. from rest: nominal, rule conditions, robust
    RestEnv.offset = offset
    goals, first, fails = 0, None, []
    ranges = []
    for j in range(fs.NOMINAL_SEEDS):
        r = run_episode(_W["rest"], copy.deepcopy(built), e + 10_000 * j, "model", model, keep_traj=(j == 0))
        if j == 0:
            first = r
        goals += r["outcome"] == "goal"
        ranges.append(r["min_target_range"])
        if r["outcome"] != "goal":
            fails.append(r["outcome"])
    rule = fs.rule_check(first, built)
    row.update(rest_goals=goals, rest_failures=";".join(fails), rest_min_range_median=float(np.median(ranges)),
               **{f"rule_{k}": v for k, v in rule.items()})
    rng = np.random.default_rng(e + 50_000)
    robust, rfails = 0, []
    for j in range(fs.ROBUST_RUNS):
        dh = float(rng.uniform(-fs.HEADING_JITTER_DEG, fs.HEADING_JITTER_DEG))
        dl = float(rng.uniform(-fs.LATERAL_JITTER_M, fs.LATERAL_JITTER_M))
        dt = float(rng.uniform(-fs.LAUNCH_JITTER_S, fs.LAUNCH_JITTER_S))
        RestPerturbedEnv.offset = offset + dt
        RestPerturbedEnv.perturb = {"initial_heading_error_deg": dh, "initial_lateral_offset_m": dl}
        try:
            r = run_episode(_W["robust"], copy.deepcopy(built), e + 20_000 + j, "model", model)
            outcome = r["outcome"]
        except ValueError:
            outcome = "start_in_contact"
        robust += outcome == "goal"
        if outcome != "goal":
            rfails.append(outcome)
    row.update(rest_robust_goals=robust, rest_robust_failures=";".join(rfails), stage="evaluated")
    return row


def candidates(case: str, max_draws: int) -> List[Dict]:
    cell = next(c for c in fs.cells() if c["name"] == case)
    parts = [pd.read_csv(OUT_ROOT / "candidates.csv")]
    if (OUT_ROOT / "candidates_ext.csv").exists():
        parts.append(pd.read_csv(OUT_ROOT / "candidates_ext.csv"))
    df = pd.concat(parts, ignore_index=True)
    kept = df[(df.name == case) & (df.layout == cell["layout"]) & (df.nominal_goals == fs.NOMINAL_SEEDS)]
    jobs = [{"name": case, "layout": cell["layout"], "code": cell["code"], "generator_seed": int(r.generator_seed),
             "episode_seed": int(r.episode_seed), "source": "original"} for r in kept.itertuples()]
    jobs += [{"name": case, "layout": cell["layout"], "code": cell["code"], "generator_seed": NEW_GEN_SEED + k,
              "episode_seed": NEW_EP_SEED + k, "source": "new"} for k in range(max_draws)]
    return jobs


def search(case: str, processes: int, max_draws: int, wanted: int) -> None:
    out = OUT_ROOT / f"repick_{case}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = candidates(case, max_draws)
    rows, ok, t0 = [], 0, time.time()
    with get_context("spawn").Pool(processes, initializer=_init) as pool:
        for row in pool.imap(evaluate, jobs, chunksize=1):
            rows.append(row)
            if row.get("stage") == "evaluated":
                ok += row["rest_goals"] == fs.NOMINAL_SEEDS
                print(f"{row['source']} {row['generator_seed']}: offset {row['offset_s']:+.0f} s (match "
                      f"{row['match_m']:.2f} m), from rest {row['rest_goals']}/10, robust {row['rest_robust_goals']}/10, "
                      f"rule {row.get('rule_rule_compliant')} ({time.time() - t0:.0f} s)", flush=True)
            pd.DataFrame(rows).to_csv(out / "candidates.csv", index=False)
            if ok >= wanted and sum(r["source"] == "original" for r in rows) >= sum(j["source"] == "original" for j in jobs):
                pool.terminate()
                break
    print(f"{len(rows)} candidates looked at, {ok} reach the goal on 10/10 from rest ({time.time() - t0:.0f} s)")


def choose(case: str) -> pd.Series:
    df = pd.read_csv(OUT_ROOT / f"repick_{case}" / "candidates.csv")
    k = df[(df.stage == "evaluated") & (df.rest_goals == fs.NOMINAL_SEEDS)].copy()
    if k.empty:
        raise SystemExit("no candidate reaches the goal on 10/10 from rest")
    k["rule_ok"] = k["rule_rule_compliant"].fillna(0.0).astype(float)
    k["abs_dcpa"] = k["dcpa_m"].abs()
    k = k.sort_values(["rest_robust_goals", "rule_ok", "abs_dcpa"], ascending=[False, False, True])
    return k.iloc[0]


def write(case: str) -> None:
    c = choose(case)
    built = ftc.build(case, c.layout, c.code, int(c.generator_seed))
    if built.digest() != c.digest:
        raise SystemExit("rebuilt digest differs")
    # the suite's own fields, as verified (both moving at t = 0), from a fresh run of its checks
    _init()
    model = _W["model"]
    e = int(c.episode_seed)
    first = run_episode(_W["cruise"], copy.deepcopy(built), e, "model", model, keep_traj=True)
    rng = np.random.default_rng(e + 50_000)
    robust = 0
    for j in range(fs.ROBUST_RUNS):
        dh = float(rng.uniform(-fs.HEADING_JITTER_DEG, fs.HEADING_JITTER_DEG))
        dl = float(rng.uniform(-fs.LATERAL_JITTER_M, fs.LATERAL_JITTER_M))
        b, _ = fs.launched(built, float(rng.uniform(-fs.LAUNCH_JITTER_S, fs.LAUNCH_JITTER_S)))
        env = fs.PerturbedEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0,
                              vessel_randomisation=cfg.VESSEL_RANDOMISATION_SCALE)
        env.perturb = {"initial_heading_error_deg": dh, "initial_lateral_offset_m": dl}
        try:
            robust += run_episode(env, b, e + 20_000 + j, "model", model)["outcome"] == "goal"
        except ValueError:
            pass
    rule = fs.rule_check(first, built)
    entry = {
        "mode": ftc.CASES[case]["mode"], "encounter": c.code, "layout": c.layout,
        "generator_seed": int(c.generator_seed), "episode_seed": e, "digest": c.digest,
        "own_at_t0": [round(float(built.own_spawn[0]), 2), round(float(built.own_spawn[1]), 2)],
        "own_heading_deg": round(float(built.own_heading) % 360.0, 1),
        "target_start": [round(float(built.target_spawn[0]), 2), round(float(built.target_spawn[1]), 2)],
        "target_heading_deg": round(float(built.target_heading) % 360.0, 1),
        "target_speed_mps": round(float(built.target_speed), 2),
        "tcpa_s": round(float(built.tcpa_s), 1), "dcpa_m": round(float(built.dcpa_m), 2),
        "nominal": f"{int(c.cruise_goals)}/{fs.NOMINAL_SEEDS}", "robust": f"{robust}/{fs.ROBUST_RUNS}",
        "rule_conditions_met": bool(rule.get("rule_compliant") == 1.0),
        "nominal_time_s": round(float(first["completion_time_s"]), 1),
        "nominal_min_target_range_m": round(float(first["min_target_range"]), 2)}
    body = ",\n        ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in entry.items())
    block = f'    {json.dumps(case)}: {{\n        {body}}},'
    block = block.replace("true", "True").replace("false", "False")
    path = ROOT / "src" / "field_test_cases.py"
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(rf'    "{case}": \{{\n.*?\}},\n', block + "\n", text, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("case entry not found")
    path.write_text(new, encoding="utf-8", newline="\n")
    chosen = {**{k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()}, "entry": entry}
    (OUT_ROOT / f"repick_{case}" / "chosen.json").write_text(json.dumps(chosen, indent=1, default=str), encoding="utf-8")
    print(f"{case} -> generator seed {int(c.generator_seed)}: {entry}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--max-draws", type=int, default=400)
    ap.add_argument("--wanted", type=int, default=3, help="stop once this many reach 10/10 from rest")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.write:
        write(args.case)
    else:
        search(args.case, args.processes, args.max_draws, args.wanted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
