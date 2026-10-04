"""Crossing trace on the development side (2026-10-03; BASELINE_V4_PLAN.md section 3).

Why does the policy fail crossings it can see coming?  For every development
crossing (field_dev and dv4x CRP/CRS, G1's crossing conflicts, the frozen-like
crossings) this rolls out the kept SAC 3 M policy (safety off) with G4's seeds
and logs every step: own pose, speed and commands, heading off the path, the
target's true state, true constant-velocity DCPA/TCPA, and what the policy is
told (tracked, class, engaged, compliant sense, perceived DCPA/TCPA, turn
admissible).  The report joins the oracle (`results/feasibility/oracle_dev.csv`,
full library for crossings): `latest_start_s`, the latest time from which some
manoeuvre still solves the episode.  Each SAC failure is put in one bin, in
this order:

* infeasible   -- the oracle finds no solution at all;
* decided before tracking -- solutions exist only for manoeuvres that start
                  before the onboard perception first tracks the target
                  (oracle `solved_after_track` False): near-impossible onboard;
* detected late -- the policy's own perception tracked the target after
                  `latest_start_s`;
* engaged late -- the encounter engaged after `latest_start_s` (the policy's
                  encounter context arrives when no manoeuvre works any more);
* wrong way    -- after tracking, a 10 deg alteration against the compliant
                  sense came first;
* slowed instead of turning -- within 4 s of tracking the speed fell below
                  `SLOW_SPEED` with less than `TURN_DEG` + 5 deg of compliant alteration;
* no action    -- neither a 10 deg compliant alteration nor a slow-down;
* late         -- the first avoidance action came after `latest_start_s`;
* insufficient -- it acted in time, yet failed (too small, or undone).

    python tools/diagnostics/crossing/crossing_trace.py run --processes 5
    python tools/diagnostics/crossing/crossing_trace.py report

Alterations and slow-downs count only from the first track on: every episode
opens with a heading transient of about 10 deg while the own ship settles on the
path (successes and failures alike), which is not a response to the target.

Results: `results/crossing_trace/` (steps.csv.gz, episodes.csv, summary.txt).
The test set is never used.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers"), str(ROOT / "tools" / "diagnostics" / "feasibility")]

import numpy as np
import pandas as pd

import curriculum
import train_formulation as tf

MODEL = ROOT / "runs" / "sac_formulation_seed0_bl3" / "kept_best_3M" / "best_model.zip"
OUT = ROOT / "results" / "crossing_trace"
TURN_DEG = 10.0                    # an alteration counts from 10 deg off the path heading
SLOW_THROTTLE = -0.3               # a slow-down counts from this throttle command
VS_DELTA = 0.05                    # m/s: the target's speed change has started
SLOW_SPEED = 0.40                  # m/s: "slowed" in the response window
RESPONSE_S = 4.0                   # the response window after the first track


def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def crossing_items():
    import oracle_sets
    return [it for it in oracle_sets.dev_items() if it[3]["code"] in ("CRP", "CRS", "CR")]


def _cv_cpa(p_rel, v_rel):
    vv = float(v_rel @ v_rel)
    if vv < 1e-9:
        return float(np.linalg.norm(p_rel)), 0.0
    tcpa = -float(p_rel @ v_rel) / vv
    return float(np.linalg.norm(p_rel + v_rel * max(tcpa, 0.0))), tcpa


def _job(args):
    from common import _WORKER
    from env import _polygon_gap
    key, built, seed, meta = args
    env, model = _WORKER["env"], _WORKER["model"]
    obs, _ = env.reset(seed=seed, options={"generated": built})
    actor = tf.EpisodeActor(model)
    rows, k = [], 0
    while True:
        a = actor(obs)
        obs, _, term, trunc, info = env.step(a)
        k += 1
        tx, ty = env.path.tangent(env.closest_idx)
        rec = {"key": key, "k": k, "t": k * 0.5, "x": env.asv_x, "y": env.asv_y, "h": env.asv_h,
               "u": float(info["speed_mps"]), "rudder": float(a[0]), "throttle": float(a[1]),
               "hdev": _wrap180(env.asv_h - math.degrees(math.atan2(tx, ty))), "cte": float(env.cross_track_error)}
        hull = env.hull_polygon()
        rec["panel_gap"] = min([_polygon_gap(hull, o)[0] for o in env.obstacles] or [np.inf])
        if env.targets:
            t = env.targets[0]
            p_rel = np.array([t.x - env.asv_x, t.y - env.asv_y])
            dcpa, tcpa = _cv_cpa(p_rel, np.asarray(t.velocity) - env._own_velocity())
            rec.update({"tx": t.x, "ty": t.y, "tspeed": float(t.speed), "theading": float(t.heading_deg),
                        "range": float(np.linalg.norm(p_rel)), "dcpa_true": dcpa, "tcpa_true": tcpa})
        ctx = next(iter(env.encounter_contexts.values()), None)
        rec["tracked"] = ctx is not None
        if ctx is not None:
            rec.update({"cls": str(ctx.cls), "engaged": bool(ctx.engaged), "gives_way": bool(ctx.gives_way),
                        "sense": int(ctx.compliant_turn_sense), "dcpa_obs": float(ctx.dcpa), "tcpa_obs": float(ctx.tcpa),
                        "turn_admissible": bool(getattr(ctx, "turn_admissible", True))})
        rows.append(rec)
        if term or trunc:
            break
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    return {"key": key, "seed": seed, **meta, "ct_deg": float(getattr(built, "ct_deg", np.nan) or np.nan),
            "outcome": outcome, "steps": k}, rows


def run(processes: int):
    from common import _init_worker
    OUT.mkdir(parents=True, exist_ok=True)
    items = crossing_items()
    print(f"[trace] {len(items)} development crossings, {MODEL.relative_to(ROOT)}", flush=True)
    t0 = time.time()
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    with ProcessPoolExecutor(max_workers=processes, initializer=_init_worker,
                             initargs=(str(MODEL), {"EMERGENCY_STOP_ENABLED": False})) as pool:
        res = list(pool.map(_job, items, chunksize=1))
    pd.DataFrame([r[0] for r in res]).to_csv(OUT / "sac_episodes_raw.csv", index=False)
    pd.DataFrame([x for r in res for x in r[1]]).to_csv(OUT / "steps.csv.gz", index=False)
    print(f"[trace] done in {time.time() - t0:.0f} s", flush=True)


def _first(s: pd.Series, mask: pd.Series):
    hit = s[mask]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def episode_metrics(ep: pd.Series, st: pd.DataFrame) -> dict:
    eng = st[st.get("engaged", pd.Series(False, index=st.index)).fillna(False).astype(bool)]
    sense = int(eng.sense.iloc[0]) if len(eng) else {"CRP": -1, "CRS": 1}.get(ep.code, 0)
    if sense == 0:                                # never engaged, frozen-like CR: the generated geometry
        sense = -1 if float(ep.get("ct_deg", 180.0)) < 180.0 else 1     # (common.run_episode's convention)
    t_det = _first(st.t, st.tracked.astype(bool))
    seen = st.t >= t_det if not math.isnan(t_det) else pd.Series(False, index=st.index)
    # Alterations are measured from the heading held when the target is first tracked.
    h0 = float(st[st.t <= t_det].hdev.iloc[-1]) if not math.isnan(t_det) and (st.t <= t_det).any() else float("nan")
    turn = _first(st.t, seen & (sense * (st.hdev - h0) >= TURN_DEG))
    wrong = _first(st.t, seen & (-sense * (st.hdev - h0) >= TURN_DEG))
    slow = _first(st.t, seen & (st.throttle <= SLOW_THROTTLE))
    win = st[seen & (st.t <= t_det + RESPONSE_S)] if not math.isnan(t_det) else st.iloc[:0]
    t_eng = float(eng.t.iloc[0]) if len(eng) else float("nan")
    at_eng = eng.iloc[0] if len(eng) else None
    v0 = float(st.tspeed.iloc[0]) if "tspeed" in st else float("nan")
    return {
        "sense": sense, "t_detect": t_det, "t_engage": t_eng,
        "dpsi_compliant_resp": float(sense * (win.hdev.iloc[-1] - h0)) if len(win) else float("nan"),
        "max_abs_dpsi_resp": float((win.hdev - h0).abs().max()) if len(win) else float("nan"),
        "u_min_resp": float(win.u.min()) if len(win) else float("nan"),
        "tcpa_true_at_engage": float(at_eng.tcpa_true) if at_eng is not None else float("nan"),
        "range_at_engage": float(at_eng.range) if at_eng is not None else float("nan"),
        "t_turn": turn, "t_wrong": wrong, "t_slow": slow, "t_action": np.nanmin([turn, slow]) if not (
            math.isnan(turn) and math.isnan(slow)) else float("nan"),
        "max_compliant_dev_deg": float((sense * (st.hdev - h0))[seen].max()) if seen.any() else float("nan"),
        "min_speed": float(st.u.min()),
        "mean_throttle_engaged": float(eng.throttle.mean()) if len(eng) else float("nan"),
        "blocked_flag_share": float((~eng.turn_admissible.astype(bool)).mean()) if len(eng) else float("nan"),
        "t_vs": _first(st.t, (st.tspeed - v0).abs() > VS_DELTA) if "tspeed" in st else float("nan"),
        "min_panel_gap": float(st.panel_gap.min()), "t_end": float(st.t.iloc[-1]),
    }


def _bin(r) -> str:
    if r.outcome == "goal":
        return "success"
    if r.n_success == 0 and r.nominal_outcome != "goal":
        return "infeasible"
    if not bool(r.solved_after_track):
        return "decided before tracking"
    L = r.latest_start_s
    if not math.isnan(r.t_detect) and r.t_detect > L:
        return "detected late"
    if not math.isnan(r.t_engage) and r.t_engage > L:
        return "engaged late"
    if not math.isnan(r.t_wrong) and (math.isnan(r.t_turn) or r.t_wrong < r.t_turn):
        return "wrong way"
    if (not math.isnan(r.u_min_resp) and r.u_min_resp < SLOW_SPEED
            and not (r.dpsi_compliant_resp >= TURN_DEG + 5.0)):
        return "slowed instead of turning"
    if math.isnan(r.t_action):
        return "no action"
    if r.t_action > L:
        return "late"
    return "insufficient"


def report():
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    eps = pd.read_csv(OUT / "sac_episodes_raw.csv")
    steps = pd.read_csv(OUT / "steps.csv.gz")
    ora = pd.read_csv(ROOT / "results" / "feasibility" / "oracle_dev.csv")
    keep = ["key", "nominal_outcome", "n_success", "n_success_margin", "best_clearance_m", "latest_start_s",
            "latest_start_margin_s", "n_success_port", "n_success_starboard", "n_success_speed", "complete",
            "t_track_s", "solved_after_track", "margin_after_track", "decision_window_s"]
    m = pd.DataFrame([{**r._asdict(), **episode_metrics(pd.Series(r._asdict()), steps[steps.key == r.key])}
                      for r in eps.itertuples(index=False)])
    m = m.merge(ora[keep], on="key", how="left")
    m["side"] = np.where(m.sense < 0, "port", "starboard")
    m["bin"] = m.apply(_bin, axis=1)
    m["goal"] = m.outcome == "goal"
    m["lateness_s"] = m.t_action - m.latest_start_s
    m.to_csv(OUT / "episodes.csv", index=False)
    feas = m[m.bin != "infeasible"]
    lines = [f"Crossing trace: SAC 3 M (kept), safety off, {len(m)} development crossings "
             f"(oracle: results/feasibility/oracle_dev.csv)", "",
             "-- success, and success on the oracle-solvable ones",
             m.groupby(["set", "side"]).agg(n=("goal", "size"), sac=("goal", "mean"),
                                            oracle_solvable=("bin", lambda b: (b != "infeasible").mean())).round(2).to_string(),
             "", "-- SAC on oracle-solvable crossings", feas.groupby("side").goal.agg(["size", "mean"]).round(2).to_string(),
             "", "-- failure bins", pd.crosstab([m.set, m.side], m.bin).to_string(), "",
             "-- bins overall", m.bin.value_counts().to_string(), "",
             "-- timing (s from the start; medians); latest_start = oracle's latest feasible manoeuvre start",
             "-- response in the 4 s after the first track (medians)",
             m.groupby(["side", "goal"])[["dpsi_compliant_resp", "max_abs_dpsi_resp", "u_min_resp"]].median().round(2).to_string(),
             f"slowed below {SLOW_SPEED} m/s within {RESPONSE_S:g} s of tracking: failures "
             f"{int(((~m.goal) & (m.u_min_resp < SLOW_SPEED)).sum())} of {int((~m.goal).sum())}, successes "
             f"{int((m.goal & (m.u_min_resp < SLOW_SPEED)).sum())} of {int(m.goal.sum())}", "",
             m.groupby(["side", "goal"])[["t_detect", "t_track_s", "decision_window_s", "t_engage", "tcpa_true_at_engage", "t_action", "t_turn", "t_slow",
                                          "latest_start_s", "lateness_s", "max_compliant_dev_deg", "min_speed",
                                          "blocked_flag_share"]].median().round(2).to_string(), "",
             "-- varying-speed targets: speed change before / after the first action (failures)",
             m[(~m.goal) & m.varying].assign(
                 vs_first=lambda x: np.where(x.t_vs < x.t_action, "VS before action", "VS after action / none")
             ).groupby(["side", "vs_first"]).size().to_string(), "",
             "-- oracle check: SAC solved an episode the oracle calls infeasible (the oracle missed a solution)",
             str(int(((m.n_success == 0) & (m.nominal_outcome != "goal") & m.goal).sum())),
             "-- SAC solved an episode decided before tracking (pre-detection behaviour, not reaction)",
             str(int((~m.solved_after_track.astype(bool) & m.goal).sum())), ""]
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "report"))
    ap.add_argument("--processes", type=int, default=5)
    a = ap.parse_args()
    if a.what == "run":
        run(a.processes)
    else:
        report()


if __name__ == "__main__":
    main()
