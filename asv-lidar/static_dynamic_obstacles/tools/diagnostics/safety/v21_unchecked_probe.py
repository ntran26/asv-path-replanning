"""Why V16 finds no passing plan: a replay probe of its unchecked decisions (V21 phase).

Replays V16 on selected validation episodes with the runner's exact episode setup and,
at every decision V16 leaves unchecked (`no escape`, `last certificate`), scores a
fixed candidate bank from V16's own decision snapshot and pre-command actuator history
at three levels:

* **checker**: V2's evaluation as V16 applies it (inflated hull, the extra gaps of
  0.05 m to static points, 0.02 m to the boundary and 0.10 m to targets, and the
  terminal run-out of up to 2 m ahead of a hull still moving at the horizon);
* **no gaps**: the inflated hull only, no extra gaps and no run-out -- the geometry the
  environment scores a collision with (`env.collision_kind`, hull + 0.15 m);
* **true hull**: the uninflated hull (what the emergency-stop detector calls contact).

The bank is the V2/V3 one: SAC's command, the 5 x 3 rudder-throttle grid and the three
full-astern rows (braking always included), each followed by the six recoveries over
the 8 s horizon, rolled out with V3's pessimistic braking model.  For each level the
probe records how many sequences pass and the latest first-violation time, and for the
checker which hazard fails first.  Offline only, it also records the true clearances
of the collision hull to the obstacles, the corridor limits and the targets.  The
probe reads; it never changes a command (the replayed outcome must equal the run's).

    python -B tools/diagnostics/safety/v21_unchecked_probe.py --cases DVD:DVD-BO-16 --out probe.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools/tiers"), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402

SELECTION = ROOT / "results/safety_dev/v21_development/selection_validation.json"
UNCHECKED = ("no escape", "last certificate")


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def polygon_distance(P, Q) -> float:
    """Distance between two convex-or-not simple polygons; 0 when they overlap."""
    import env as envmod
    if envmod._overlaps(P, Q):
        return 0.0
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    best = np.inf
    for A, B in ((P, Q), (Q, P)):
        for p in A:
            for i in range(len(B)):
                best = min(best, _seg_dist(p, B[i], B[(i + 1) % len(B)]))
    return float(best)


def truth(env) -> dict:
    """True clearances of the collision hull at the decision (offline scoring only), and
    the filter's polygon clearance evaluated at the true pose, which separates a pose
    estimate error from a polygon that differs from the environment's limits."""
    from classical import common as cc
    hull = env.hull_polygon()
    poly = np.asarray(env.boundary_polygon, dtype=float)
    pos = np.array([[[env.asv_x, env.asv_y]]])
    hdg = np.array([[np.radians(env.asv_h)]])
    poly_true_pose = float(cc.boundary_clearance(pos, hdg, poly, np.roll(poly, -1, axis=0))[0, 0])
    return {"obstacle": min((polygon_distance(hull, o) for o in env.obstacles), default=None),
            "border": float(env._border_clearance(hull)),
            "polygon_at_true_pose": poly_true_pose,
            "target": min((polygon_distance(hull, t.hull()) for t in env.targets), default=None),
            "pose": [float(env.asv_x), float(env.asv_y), float(env.asv_h)]}


def score(filt, snap, actuators, sac) -> dict:
    import safety_v2 as v2
    import safety_v3 as v3
    from classical import common as cc
    cands = np.array([tuple(sac)] + [(r, t) for r in v2.RUDDERS for t in v2.THROTTLES]
                     + [(r, np.nan) for r in v2.BRAKE_RUDDERS], dtype=float)
    recs = np.array([(r, np.nan if t is v2.BRAKE else t) for r, t in v2.RECOVERY], dtype=float)
    seqs = np.array([v3.plan_for(c, r) for c in cands for r in recs])
    ro = v3.rollout_seq(snap, copy.deepcopy(actuators), seqs)
    first_chk, _ = v2.SafetyFilterV2._evaluate(filt, snap, ro)
    cs = cc.point_clearance(ro.positions, ro.headings, snap.points, reach=cc.HALF_L + 1.0)
    cb = cc.boundary_clearance(ro.positions, ro.headings, snap.edges_a, snap.edges_b)
    ct = np.full_like(cs, np.inf)
    for track in snap.tracks:
        ct = np.minimum(ct, cc.target_gap(ro.positions, ro.headings, ro.times, track))
    gaps = {"static": cs - v2.GAP_STATIC_M, "boundary": cb - v2.GAP_BOUNDARY_M, "target": ct - v2.GAP_TARGET_M}

    def first(c):
        bad = c < 0.0
        return np.where(bad.any(axis=0), ro.times[np.argmax(bad, axis=0)], np.inf)

    nogap = np.minimum(np.minimum(cs, cb), ct)
    levels = {"checker": first_chk, "no_gaps": first(nogap), "true_hull": first(nogap + cc.HULL_MARGIN)}
    out = {}
    for name, f in levels.items():
        out[name] = {"pass": int(np.isinf(f).sum()), "latest_first_s": float(np.max(f)),
                     "sac_pass": bool(np.isinf(f[:len(recs)]).any())}
    # Which first commands and recoveries pass the checker (labels; braking marked "B").
    lab_c = ["sac"] + [f"r{r:+.1f}t{t:+.0f}" for r in v2.RUDDERS for t in v2.THROTTLES] + [f"r{r:+.1f}B" for r in v2.BRAKE_RUDDERS]
    lab_r = [f"r{r:+.1f}" + ("B" if t is v2.BRAKE else f"t{t:+.0f}") for r, t in v2.RECOVERY]
    ok = np.flatnonzero(np.isinf(first_chk))
    out["checker_passing"] = sorted({f"{lab_c[k // len(recs)]}>{lab_r[k % len(recs)]}" for k in ok})[:40]
    out["checker_passing_braking"] = int(sum(("B" in lab_c[k // len(recs)]) or ("B" in lab_r[k % len(recs)]) for k in ok))
    within = {k: first(v) for k, v in gaps.items()}
    i = int(np.argmax(first_chk))                     # the sequence that survives longest under the checker
    hz = {k: float(v[i]) for k, v in within.items()}
    out["checker_best"] = {"first_s": float(first_chk[i]), "hazard_first_s": hz,
                           "terminal_only": bool(np.isfinite(first_chk[i]) and all(np.isinf(list(hz.values()))))}
    out["now"] = {k: float(v[0].min()) if np.isfinite(v[0]).any() else None for k, v in gaps.items()}
    out["estimated_pose"] = [float(snap.x), float(snap.y), float(np.degrees(snap.heading)) % 360.0]
    out["sequences"] = int(len(seqs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="comma-separated validation case IDs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    import constants as cfg
    import curriculum
    import train_formulation as tf
    import safety_v16
    import v21_iteration as runner
    import v9_paired_pilot as pilot
    selection = {c["case"]: c for c in json.loads(SELECTION.read_text())["cases"]}
    cases = [selection[c] for c in args.cases.split(",")]
    if any(c["case"].startswith("DVD:") for c in cases):
        import formulation_v3
        formulation_v3.apply(cfg)          # as the runner: the DVD panels are drawn at reset under it
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from common import load_model, run_episode
    from env import ASVLidarEnv
    scenes, _ = runner.load_scenes(cases)
    model = load_model(pilot.MODEL)

    class ProbeV16(safety_v16.SafetyFilterV16):
        def __init__(self, env_ref):
            super().__init__()
            self.env_ref, self.decision, self.records = env_ref, 0, []

        def filter(self, env, action):
            sac = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
            true_now = truth(env)
            out, changed = super().filter(env, action)
            self.decision += 1
            why = self.last.get("why", "idle")
            if why in UNCHECKED:
                rec = {"decision": self.decision, "why": why, "changed": bool(changed),
                       "truth": true_now, **score(self, self._observer_snapshot, self._observer_actuators, sac)}
                self.records.append(rec)
            return out, changed

    class ProbeEnv(ASVLidarEnv):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            self._safety_v2 = ProbeV16(self)
            return result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as stream:
        for case in cases:
            cfg.SAFETY_VERSION = 16
            env = ProbeEnv(render_mode=None, emergency_stop=True, low_speed_start_frac=0.0)
            try:
                result = run_episode(env, copy.deepcopy(scenes[case["case"]]), int(case["seed"]), "model", model)
                line = {"case": case["case"], "outcome": result["outcome"], "steps": result["steps"],
                        "decisions": env._safety_v2.records}
                stream.write(json.dumps(line, default=float) + "\n")
                stream.flush()
                print(case["case"], result["outcome"], result["steps"], len(line["decisions"]), flush=True)
            finally:
                env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
