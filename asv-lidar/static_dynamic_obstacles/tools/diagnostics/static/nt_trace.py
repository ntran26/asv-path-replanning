"""Why does SAC fail no-target episodes?  (2026-10-04, BASELINE_V4_PLAN.md section 5e.)

Requirement (decision, 2026-10-04): the policy must pass every no-target
episode.  This traces the kept SAC 3 M policy (safety off) on every no-target
episode of the development set (formulation v4.2, evaluation seeds) and of test
set v4 (its own seeds), and on test set v4's null episodes (a target present
but never in conflict), and compares each run with the oracle's A* route
(`oracle_feasibility.Route`, which clears these layouts):

* per step: pose, speed, commands, heading off the path, cross-track error,
  along-path position, the gap to every panel and to the wall, the nearest LiDAR
  return within 45 deg of the bow;
* per panel along the leg: whether it sits on the path, the side the route
  passes it on, the side SAC passes it on (or whether SAC never reached it);
* per episode: what was hit, the first panel passed on the side the route does
  not take, where the avoidance started (along-path distance to the first
  on-path panel when the heading first leaves the path by 10 deg after the
  opening 2 s), the peak cross-track error and the overshoot past the path
  after the first panel, the closest approach to the wall.

    python tools/diagnostics/static/nt_trace.py --processes 2

Results: `results/static_trace/` (steps.csv.gz, episodes.csv, panels.csv, summary.txt).
Diagnosis only: no development decision is taken on the test-set rows.
"""
from __future__ import annotations

import argparse
import math
import pickle
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
OUT = ROOT / "results" / "static_trace"
ON_PATH_M = 0.9                       # a panel whose nearest point is this close to the path blocks it
ONSET_DEG, SETTLE_S = 10.0, 2.0


def _wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def items():
    import oracle_sets
    out = []
    for key, b, seed, meta in oracle_sets.deveval_items():
        if meta["code"] == "NT":
            out.append((key, b, seed, {"split": "dev", "group": "NT", "set": meta["set"]}))
    with open(ROOT / "results" / "test_set" / "v4" / "set_v4.0.pkl", "rb") as fh:
        kept, _ = pickle.load(fh)
    for it in kept:
        if it["cell"].endswith("-NT") or it["cell"] == "basin-null":
            out.append((f"test:{it['test_id']}", it["built"], int(it["episode_seed"]),
                        {"split": "test", "group": "null" if it["cell"] == "basin-null" else "NT", "set": it["cell"]}))
    return out


def _path_coords(env, x, y):
    st = env.path.project(x, y, env.asv_h)
    return float(st.s_along), float(st.cross_track_error)


def _job(args):
    import oracle_feasibility as of
    from common import _WORKER
    from env import _polygon_gap
    key, built, seed, meta = args
    env, model = _WORKER["env"], _WORKER["model"]
    obs, _ = env.reset(seed=seed, options={"generated": built})
    route = of.Route(env)
    panels = []
    for j, poly in enumerate(env.obstacles):
        p = np.asarray(poly, dtype=float)
        c = p.mean(axis=0)
        s_c, e_c = _path_coords(env, *c)
        lat = [(_path_coords(env, *v)[1]) for v in p]
        on_path = min(abs(v) for v in lat) <= ON_PATH_M or (min(lat) < 0 < max(lat))
        r_side = None
        if route.points is not None:
            rp = [(_path_coords(env, *q)) for q in route.points]
            dense = []
            for (s0, e0), (s1, e1) in zip(rp[:-1], rp[1:]):
                for t in np.linspace(0, 1, 12):
                    dense.append((s0 + t * (s1 - s0), e0 + t * (e1 - e0)))
            s_arr = np.array([d[0] for d in dense])
            k = int(np.argmin(np.abs(s_arr - s_c)))
            r_side = int(np.sign(dense[k][1] - e_c)) or 1
        panels.append({"key": key, "panel": j, "s": s_c, "e": e_c, "on_path": bool(on_path), "route_side": r_side})
    actor = tf.EpisodeActor(model)
    rows, k = [], 0
    while True:
        a = actor(obs)
        obs, _, term, trunc, info = env.step(a)
        k += 1
        tx, ty = env.path.tangent(env.closest_idx)
        hull = env.hull_polygon()
        gaps = [float(_polygon_gap(hull, o)[0]) for o in env.obstacles]
        rel = np.array([_wrap180(b) for b in env.lidar.bearings]) if hasattr(env.lidar, "bearings") else None   # deg off the bow
        front = float(np.min(env.raw_ranges[np.abs(rel) <= 45.0])) if rel is not None and len(rel) == len(env.raw_ranges) else np.nan
        s, e = float(env.s_along), float(env.cross_track_error)
        rec = {"key": key, "k": k, "t": k * 0.5, "x": env.asv_x, "y": env.asv_y, "h": env.asv_h,
               "u": float(info["speed_mps"]), "rudder": float(a[0]), "throttle": float(a[1]),
               "hdev": _wrap180(env.asv_h - math.degrees(math.atan2(tx, ty))), "cte": e, "s": s,
               "wall": float(env._border_clearance(hull)), "front_lidar": front,
               "panel_gap": min(gaps) if gaps else np.inf, "nearest_panel": int(np.argmin(gaps)) if gaps else -1}
        if env.targets:
            t0 = env.targets[0]
            rec["target_range"] = float(math.hypot(t0.x - env.asv_x, t0.y - env.asv_y))
        rows.append(rec)
        if term or trunc:
            break
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    d = pd.DataFrame(rows)
    for p in panels:                       # the side SAC passed each panel on (if it got that far)
        past = d[d.s >= p["s"]]
        p["sac_side"] = (int(np.sign(past.cte.iloc[0] - p["e"])) or 1) if len(past) else None
        p["reached"] = bool(len(past))
    ep = {"key": key, "seed": seed, **meta, "outcome": outcome, "steps": k,
          "slant_deg": abs(float(built.slant_realised_deg)), "motif": str((built.flags or {}).get("motif", "")),
          "n_panels": len(panels), "n_on_path": sum(p["on_path"] for p in panels),
          "hit_panel": int(d.nearest_panel.iloc[-1]) if outcome == "collision:obstacle" else -1,
          "peak_cte": float(d.cte.abs().max()), "min_wall": float(d.wall.min()),
          "min_panel_gap": float(d.panel_gap.min()), "u_end": float(d.u.iloc[-1]),
          "mean_throttle": float(d.throttle.mean())}
    on = sorted([p for p in panels if p["on_path"]], key=lambda p: p["s"])
    first = on[0] if on else None
    settle = d[d.t >= SETTLE_S]
    onset = settle[settle.hdev.abs() >= ONSET_DEG]
    if first is not None:
        ep["first_on_path_s"] = first["s"]
        ep["onset_dist_m"] = float(first["s"] - onset.s.iloc[0]) if len(onset) else np.nan
        after = d[d.s > first["s"]]
        side = first["sac_side"] or 0
        ep["overshoot_m"] = float((-side * (after.cte - first["e"])).max()) if len(after) and side else np.nan
    mism = [p for p in sorted(panels, key=lambda p: p["s"]) if p["reached"] and p["route_side"] is not None
            and p["sac_side"] is not None and p["sac_side"] != p["route_side"]]
    ep["n_side_mismatch"] = len(mism)
    ep["first_mismatch_panel"] = mism[0]["panel"] if mism else -1
    ep["hit_was_mismatch"] = bool(ep["hit_panel"] >= 0 and any(p["panel"] == ep["hit_panel"] for p in mism))
    return ep, rows, panels


def run(processes: int):
    from common import _init_worker
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = items()
    print(f"[static trace] {len(jobs)} episodes ({sum(j[3]['group'] == 'NT' for j in jobs)} no-target, "
          f"{sum(j[3]['group'] == 'null' for j in jobs)} null)", flush=True)
    t0 = time.time()
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    with ProcessPoolExecutor(max_workers=processes, initializer=_init_worker,
                             initargs=(str(MODEL), {"EMERGENCY_STOP_ENABLED": False})) as pool:
        res = list(pool.map(_job, jobs, chunksize=1))
    pd.DataFrame([r[0] for r in res]).to_csv(OUT / "episodes.csv", index=False)
    pd.DataFrame([x for r in res for x in r[1]]).to_csv(OUT / "steps.csv.gz", index=False)
    pd.DataFrame([x for r in res for x in r[2]]).to_csv(OUT / "panels.csv", index=False)
    print(f"[static trace] done in {time.time() - t0:.0f} s", flush=True)


def report():
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    e = pd.read_csv(OUT / "episodes.csv", keep_default_na=False, na_values=[""])
    e["goal"] = e.outcome == "goal"
    e["straight"] = e.slant_deg <= 2.0
    lines = ["Static trace: SAC 3 M (kept), safety off; no-target (development and test set v4) and null (test set v4)", ""]
    lines += ["-- success", e.groupby(["split", "group", "set"]).goal.agg(["size", "mean"]).round(2).to_string(), ""]
    lines += ["-- by leg and motif (no-target, both splits)",
              e[e.group == "NT"].groupby(["straight", "motif"]).goal.agg(["size", "mean"]).round(2).to_string(), ""]
    f = e[~e.goal]
    cols = ["key", "split", "group", "set", "motif", "slant_deg", "outcome", "steps", "hit_panel", "hit_was_mismatch",
            "n_side_mismatch", "onset_dist_m", "peak_cte", "overshoot_m", "min_wall", "u_end", "mean_throttle"]
    lines += ["-- failures", f[cols].round(2).to_string(index=False), ""]
    agg = ["n_side_mismatch", "onset_dist_m", "peak_cte", "overshoot_m", "min_wall", "u_end", "mean_throttle"]
    lines += ["-- medians, failures vs successes (no-target)",
              e[e.group == "NT"].groupby("goal")[agg].median().round(2).to_string(), "",
              "-- medians, failures vs successes (null)",
              e[e.group == "null"].groupby("goal")[agg].median().round(2).to_string(), ""]
    p = pd.read_csv(OUT / "panels.csv", keep_default_na=False, na_values=[""])
    p = p.merge(e[["key", "goal", "group"]], on="key")
    pr = p[p.reached.astype(str) == "True"].dropna(subset=["route_side", "sac_side"])
    pr["mismatch"] = pr.route_side != pr.sac_side
    lines += ["-- panels passed on the side the route does not take (reached panels)",
              pr.groupby(["group", "goal", "on_path"]).mismatch.agg(["size", "mean"]).round(2).to_string(), ""]
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "report"), nargs="?", default="run")
    ap.add_argument("--processes", type=int, default=2)
    a = ap.parse_args()
    if a.what == "run":
        run(a.processes)
    report()


if __name__ == "__main__":
    main()
