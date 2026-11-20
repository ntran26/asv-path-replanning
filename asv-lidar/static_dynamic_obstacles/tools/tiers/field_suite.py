"""Build the field test suite (`src/field_test_cases.py`): 3 modes x the 5 COLREGs encounters.

Cells: easy (open basin, straight leg, no panels), medium (layout L1), hard (layouts L2 and L3,
the better of the two kept per encounter) x HO, CRS, CRP, OT, BO.  Each cell draws candidates
with the scenario generator under the Paper 2 field rules (`paper2_set`: constant-velocity
target, one heading, speed in `FIELD_TARGET_SPEED_RANGE`, stop short of the wall only after the
encounter) and the test set's start check (own hull 0.30 m clear of every panel), from its own
seed block.  A candidate is kept only if the deployed policy reaches the goal:

1. **nominal**: on the identified plant (no randomisation), on `NOMINAL_SEEDS` sensor-noise
   seeds, every one;
2. **robust** (scored, not required): `ROBUST_RUNS` runs with the plant drawn from the training
   randomisation, the start heading and lateral position perturbed by up to
   `HEADING_JITTER_DEG` and `LATERAL_JITTER_M`, and the target launched up to `LAUNCH_JITTER_S`
   early or late, the field's set-up errors.

Per case the selection takes, among the kept candidates, the most robust, then one whose
nominal run passes the Section 3.4.3 rule conditions (`colregs_compliance.episode`), then the
smallest drawn DCPA (the encounter that most needs a manoeuvre).

    python tools/tiers/field_suite.py --search [--processes 8]    # results/field_suite/candidates.csv
    python tools/tiers/field_suite.py --write                     # select; CASES of src/field_test_cases.py
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))
sys.path.insert(0, str(ROOT / "tools" / "diagnostics"))

import constant_temp as ct  # noqa: E402
import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import field_test_cases as ftc  # noqa: E402
import paper2_set as p2  # noqa: E402
import train_formulation as tf  # noqa: E402
from common import load_model, run_episode  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

OUT = ROOT / "results" / "field_suite"
MODEL = ROOT / "runs" / "sac_formulation_seed0_bl3" / "best_model.zip"
# Seed blocks no other set uses (training 0-99,999; development 200 k; frozen 300,000-309,999;
# Paper 2 310,000-339,999; test set v4 342 k, 450 k, 496-499.5 k, 600 k; validation 350 k and
# 550 k; studies 400 k, 500 k; extension 520 k and 720 k; field twins 910 k).
SEED_BASE = 860_000
SEEDS_PER_CELL = 1_000
EPISODE_SEED_BASE = 880_000
NOMINAL_SEEDS = 10
ROBUST_RUNS = 10
HEADING_JITTER_DEG = 5.0
LATERAL_JITTER_M = 0.20
LAUNCH_JITTER_S = 1.0
START_GAP_M = 0.30
WANTED = 6              # kept candidates per cell before the search stops
MAX_TRIES = 300


def cells() -> List[Dict]:
    return [{"index": i, "mode": mode, "layout": layout, "code": code, "name": ftc.case_name(mode, code)}
            for i, (mode, layout, code) in enumerate(
                (m, lay, c) for m in ftc.MODES for lay in ftc.MODE_LAYOUTS[m] for c in ftc.ENCOUNTERS.values())]


class PerturbedEnv(ASVLidarEnv):
    """The start perturbed as the field's set-up would be (heading error, lateral offset)."""
    perturb: Dict = {}

    def _apply_initial_conditions(self, options: dict) -> None:
        super()._apply_initial_conditions({**options, **self.perturb})


def start_clear(env, built, seed: int) -> bool:
    env.reset(seed=seed, options={"generated": copy.deepcopy(built)})
    if env.collision_kind(env.hull_polygon()) is not None:
        return False
    hull = np.asarray(env.hull_polygon(), dtype=float)
    return all(p2.polygon_distance(hull, np.asarray(poly, dtype=float)) >= START_GAP_M for poly in env.obstacles)


def nominal_check(env, built, seed: int) -> Dict:
    """`paper2_set.nominal_check`, with no panel to clear in the open basin."""
    if (built.flags or {}).get("fixed_obstacles"):
        return p2.nominal_check(env, built, seed)
    from nominal import nominal_encounter
    env.reset(seed=seed, options={"generated": built})
    nom = nominal_encounter(env)
    trk = nom["target"]
    if not len(trk):
        return {"clearance": np.inf, "turned": 0.0, "t_stop": None, "cpa_t": None}
    moved = np.hypot(np.diff(trk[:, 0]), np.diff(trk[:, 1])) > 1e-9
    t_stop = None if moved.all() else float(np.argmin(moved) * nom["dt"])
    return {"clearance": np.inf, "turned": nom["turned_deg"], "t_stop": t_stop, "cpa_t": nom.get("cpa_t")}


def launched(built, dt: float):
    """The target launched `dt` s early (dt > 0) or late: its start moved along its track, if
    the hull then still starts inside the field stop box; otherwise unchanged."""
    b = copy.deepcopy(built)
    h, s = math.radians(float(b.target_heading)), float(b.target_speed) * dt
    x, y = float(b.target_spawn[0]) + s * math.sin(h), float(b.target_spawn[1]) + s * math.cos(h)
    from ship import VESSEL_LENGTH
    m = ct.FIELD_WALL_MARGIN_M + 0.5 * VESSEL_LENGTH
    if not (m <= x <= cfg.MAP_WIDTH - m and m <= y <= cfg.MAP_HEIGHT - m):
        return built, 0.0
    b.target_spawn = (x, y)
    return b, dt


_W: Dict = {}


def _init() -> None:
    import torch
    torch.set_num_threads(1)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    _W["model"] = load_model(MODEL)
    _W["check"] = ASVLidarEnv(render_mode=None, emergency_stop=False)
    _W["nominal"] = ASVLidarEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0,
                                vessel_randomisation=None)
    _W["robust"] = PerturbedEnv(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0,
                                vessel_randomisation=cfg.VESSEL_RANDOMISATION_SCALE)


def rule_check(row: Dict, built) -> Dict:
    import colregs_compliance as cc
    tr = row.get("_traj")
    if tr is None or "tgt_x" not in tr:
        return {}
    side = (built.flags or {}).get("side", "")
    out = cc.episode({"class": built.encounter_class, "crossing_side": side or "", "outcome": row["outcome"],
                      "variant": "FIX", "test_id": built.case_id}, tr)
    return {k: out.get(k) for k in ("rule_compliant", "first_manoeuvre", "target_side_at_cpa", "crossed_ahead", "cpa_m")}


def search_cell(cell: Dict) -> List[Dict]:
    """Candidates of one cell.  `cell` may carry `start` (first try), `max_tries` and
    `require_rule`: then a candidate whose first nominal run fails the rule conditions is
    recorded and not run further, and only rule-consistent kept candidates count towards
    `wanted` (the extension search, `--extend`)."""
    model = _W["model"]
    lo, hi = ct.FIELD_TARGET_SPEED_RANGE
    rows, kept = [], 0
    require_rule = bool(cell.get("require_rule", False))
    wanted = int(cell.get("wanted", WANTED))
    for k in range(int(cell.get("start", 0)), int(cell.get("max_tries", MAX_TRIES))):
        gseed = SEED_BASE + cell["index"] * SEEDS_PER_CELL + k
        eseed = EPISODE_SEED_BASE + cell["index"] * SEEDS_PER_CELL + k
        built = ftc.build(cell["name"], cell["layout"], cell["code"], gseed)
        if built is None or not lo <= float(built.target_speed) <= hi:
            continue
        if not start_clear(_W["check"], built, eseed):
            continue
        if not p2.field_feasible(nominal_check(_W["check"], copy.deepcopy(built), eseed)):
            continue
        row = {**{k2: cell[k2] for k2 in ("mode", "layout", "code", "name")}, "generator_seed": gseed,
               "episode_seed": eseed, "digest": built.digest(), "target_x": float(built.target_spawn[0]),
               "target_y": float(built.target_spawn[1]), "target_heading": float(built.target_heading) % 360.0,
               "target_speed": float(built.target_speed), "own_x": float(built.own_spawn[0]),
               "own_y": float(built.own_spawn[1]), "own_heading": float(built.own_heading) % 360.0,
               "dcpa_m": float(built.dcpa_m), "tcpa_s": float(built.tcpa_s), "ct_deg": float(built.ct_deg)}
        goals, first = 0, None
        for j in range(NOMINAL_SEEDS):
            r = run_episode(_W["nominal"], copy.deepcopy(built), eseed + 10_000 * j, "model", model,
                            keep_traj=(j == 0))
            if j == 0:
                first = r
                if require_rule and r["outcome"] == "goal" and rule_check(r, built).get("rule_compliant") != 1.0:
                    row["nominal_failure"] = "rule_conditions"
                    break
            if r["outcome"] != "goal":
                row["nominal_failure"] = r["outcome"]
                break
            goals += 1
        row["nominal_goals"], row["nominal_runs"] = goals, j + 1
        row.update(nominal_time_s=first["completion_time_s"], nominal_min_target_range=first["min_target_range"],
                   **{f"rule_{k2}": v for k2, v in rule_check(first, built).items()})
        if goals == NOMINAL_SEEDS:
            rng = np.random.default_rng(eseed + 50_000)
            robust, failures = 0, []
            for j in range(ROBUST_RUNS):
                dh = float(rng.uniform(-HEADING_JITTER_DEG, HEADING_JITTER_DEG))
                dl = float(rng.uniform(-LATERAL_JITTER_M, LATERAL_JITTER_M))
                b, dt = launched(built, float(rng.uniform(-LAUNCH_JITTER_S, LAUNCH_JITTER_S)))
                _W["robust"].perturb = {"initial_heading_error_deg": dh, "initial_lateral_offset_m": dl}
                try:
                    r = run_episode(_W["robust"], b, eseed + 20_000 + j, "model", model)
                    outcome = r["outcome"]
                except ValueError:
                    outcome = "start_in_contact"
                robust += outcome == "goal"
                if outcome != "goal":
                    failures.append(outcome)
            row.update(robust_goals=robust, robust_runs=ROBUST_RUNS, robust_failures=";".join(failures))
            kept += (not require_rule) or robust >= ROBUST_RUNS - 1
        rows.append(row)
        if kept >= wanted:
            break
    return rows


def extend(processes: int) -> None:
    """Search further, in the same seed blocks, the cells whose kept candidates include none
    that both meets the rule conditions and reaches the goal in at least 9 of the robust runs."""
    df = pd.read_csv(OUT / "candidates.csv")
    k = df[df.nominal_goals == NOMINAL_SEEDS]
    good = set(k[(k.rule_rule_compliant == 1.0) & (k.robust_goals >= ROBUST_RUNS - 1)].name)
    jobs = []
    for c in cells():
        if c["name"] in good:
            continue
        mine = df[(df.name == c["name"]) & (df.layout == c["layout"])]
        start = int((mine.generator_seed - SEED_BASE - c["index"] * SEEDS_PER_CELL).max()) + 1 if len(mine) else MAX_TRIES
        if start >= SEEDS_PER_CELL or (len(mine) == 0 and c["layout"] != "OPEN"):
            continue                    # a cell that yielded no candidate at all is not extended
        jobs.append({**c, "start": start, "max_tries": SEEDS_PER_CELL, "require_rule": True, "wanted": 2})
    print("extending:", [(j["name"], j["layout"], j["start"]) for j in jobs], flush=True)
    t0, rows = time.time(), []
    with Pool(min(processes, len(jobs)), initializer=_init) as pool:
        for got in pool.imap_unordered(search_cell, jobs):
            rows += got
            ok = [r for r in got if r.get("nominal_goals") == NOMINAL_SEEDS]
            c = got[0] if got else {}
            print(f"{c.get('name', '?')} {c.get('layout', '?')}: {len(got)} more candidates, {len(ok)} kept "
                  f"(robust {[r.get('robust_goals') for r in ok]}) ({time.time() - t0:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "candidates_ext.csv", index=False)


def search(processes: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: List[Dict] = []
    with Pool(processes, initializer=_init) as pool:
        for got in pool.imap_unordered(search_cell, cells()):
            rows += got
            kept = [r for r in got if r.get("nominal_goals") == NOMINAL_SEEDS]
            c = got[0] if got else {}
            print(f"{c.get('name', '?')} {c.get('layout', '?')}: {len(got)} candidates, {len(kept)} kept, "
                  f"best robust {max((r['robust_goals'] for r in kept), default='-')} ({time.time() - t0:.0f} s)",
                  flush=True)
    pd.DataFrame(rows).sort_values(["mode", "code", "layout", "generator_seed"]).to_csv(OUT / "candidates.csv", index=False)
    print(f"{len(rows)} candidates -> {OUT / 'candidates.csv'}")


def select(df: pd.DataFrame) -> pd.DataFrame:
    kept = df[df.nominal_goals == NOMINAL_SEEDS].copy()
    kept["rule_ok"] = kept["rule_rule_compliant"].fillna(0.0).astype(float)
    kept["abs_dcpa"] = kept["dcpa_m"].abs()
    kept = kept.sort_values(["name", "robust_goals", "rule_ok", "abs_dcpa"], ascending=[True, False, False, True])
    return kept.groupby("name", sort=False).head(1).set_index("name")


def write(df: pd.DataFrame) -> None:
    chosen = select(df)
    missing = [c["name"] for c in cells() if c["name"] not in chosen.index]
    if missing:
        raise SystemExit(f"no kept candidate for {sorted(set(missing))}")
    cases = {}
    for mode in ftc.MODES:
        for number, code in ftc.ENCOUNTERS.items():
            name = f"{mode}_{number}"
            r = chosen.loc[name]
            built = ftc.build(name, r.layout, code, int(r.generator_seed))
            if built.digest() != r.digest:
                raise SystemExit(f"{name}: rebuilt digest differs")
            cases[name] = {
                "mode": mode, "encounter": code, "layout": r.layout, "generator_seed": int(r.generator_seed),
                "episode_seed": int(r.episode_seed), "digest": r.digest,
                "own_at_t0": [round(float(built.own_spawn[0]), 2), round(float(built.own_spawn[1]), 2)],
                "own_heading_deg": round(float(built.own_heading) % 360.0, 1),
                "target_start": [round(float(built.target_spawn[0]), 2), round(float(built.target_spawn[1]), 2)],
                "target_heading_deg": round(float(built.target_heading) % 360.0, 1),
                "target_speed_mps": round(float(built.target_speed), 2),
                "tcpa_s": round(float(built.tcpa_s), 1), "dcpa_m": round(float(built.dcpa_m), 2),
                "nominal": f"{int(r.nominal_goals)}/{NOMINAL_SEEDS}",
                "robust": f"{int(r.robust_goals)}/{ROBUST_RUNS}",
                "rule_conditions_met": bool(r.rule_ok),
                "nominal_time_s": round(float(r.nominal_time_s), 1),
                "nominal_min_target_range_m": round(float(r.nominal_min_target_range), 2)}
    lines = ["CASES: Dict[str, Dict] = {"]
    for name, c in cases.items():
        body = ",\n        ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in c.items())
        lines.append(f"    {json.dumps(name)}: {{\n        {body}}},")
    lines.append("}")
    block = "\n".join(lines).replace("true", "True").replace("false", "False")
    path = ROOT / "src" / "field_test_cases.py"
    text = path.read_text(encoding="utf-8")
    new = re.sub(r"# BEGIN CASES\n.*?# END CASES", "# BEGIN CASES\n" + block + "\n# END CASES", text, flags=re.S)
    path.write_text(new, encoding="utf-8", newline="\n")     # the file's own LF endings
    chosen.reset_index().to_csv(OUT / "selection.csv", index=False)
    print(f"{len(cases)} cases -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--extend", action="store_true", help="search further where no rule-consistent case was kept")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    if args.search:
        search(args.processes)
    if args.extend:
        extend(args.processes)
    if args.write:
        parts = [pd.read_csv(OUT / "candidates.csv")]
        if (OUT / "candidates_ext.csv").exists():
            parts.append(pd.read_csv(OUT / "candidates_ext.csv"))
        write(pd.concat(parts, ignore_index=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
