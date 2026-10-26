"""Where V16's remembered static points come from: an offline audit (V21 phase, mechanism 4).

Replays V16 with the runner's exact episode setup and, at every decision where V16 is
active (not idle), classifies each remembered static point of V16's own snapshot that
lies within `NEAR_M` of the hull (the points that can bind a plan):

* `obstacle`: within `TOL_M` of a true obstacle polygon;
* `wall`: within `TOL_M` of the basin or channel limits the environment scores, or
  outside the navigable polygon (returns of the physical walls);
* `target_track`: within `TRACK_M` of the target's true centre at some earlier decision
  of the episode (a target return remembered as static);
* `other`: none of these.

Truth is read for this offline classification only.  The audit never changes a command
(the replayed outcome must equal the run's).

    python -B tools/diagnostics/safety/v21_static_audit.py --cases DVD:DVD-CR-01 --out audit.jsonl
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
NEAR_M = 1.0          # points within this of the inflated hull can bind a plan
TOL_M = 0.15          # a return lies on the surface it came from, to within sensor noise
TRACK_M = 0.9         # half a hull length: on the target's own footprint


def point_polygon_distance(points, polygon):
    """Distance from each point to a polygon's boundary, (P,); 0 inside."""
    P = np.asarray(points, float).reshape(-1, 2)
    Q = np.asarray(polygon, float)
    a, b = Q, np.roll(Q, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("pkj,kj->pk", P[:, None, :] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    d = np.linalg.norm(P[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1).min(axis=1)
    inside = np.array([_inside(p, Q) for p in P])
    return np.where(inside, 0.0, d)


def classify(env, snap, track_history, removed_mask=None):
    """Counts of near-hull points by origin; with `removed_mask` (over snap.points), the
    counts of the near points that mask removes, keyed `removed_<origin>`."""
    from classical import common as cc
    pts = np.asarray(snap.points, float).reshape(-1, 2)
    if not len(pts):
        return {}
    removed = np.zeros(len(pts), dtype=bool) if removed_mask is None else np.asarray(removed_mask, bool)
    pos = np.array([[[snap.x, snap.y]]])
    hdg = np.array([[snap.heading]])
    near = np.array([cc.point_clearance(pos, hdg, p[None], reach=cc.HALF_L + NEAR_M + 1.0)[0, 0] for p in pts]) < NEAR_M
    pts, removed = pts[near], removed[near]
    if not len(pts):
        return {}
    d_obs = np.full(len(pts), np.inf)
    for o in env.obstacles:
        d_obs = np.minimum(d_obs, point_polygon_distance(pts, o))
    lo, hi = env.corridor_bounds_x()
    d_wall = np.minimum.reduce([np.abs(pts[:, 0] - lo), np.abs(pts[:, 0] - hi), np.abs(pts[:, 1]),
                                np.abs(pts[:, 1] - env.map_height)])
    poly = np.asarray(env.boundary_polygon, float)
    outside = np.array([not _inside(p, poly) for p in pts])
    hist = np.asarray(track_history, float).reshape(-1, 2)
    d_trk = (np.linalg.norm(pts[:, None, :] - hist[None], axis=-1).min(axis=1) if len(hist)
             else np.full(len(pts), np.inf))
    label = np.where(d_obs <= TOL_M, "obstacle",
                     np.where((d_wall <= TOL_M) | outside, "wall",
                              np.where(d_trk <= TRACK_M, "target_track", "other")))
    values, counts = np.unique(label, return_counts=True)
    out = dict(zip(values.tolist(), counts.tolist()))
    if removed.any():
        rv, rc = np.unique(label[removed], return_counts=True)
        out.update({f"removed_{v}": int(c) for v, c in zip(rv.tolist(), rc.tolist())})
    return out


def _inside(p, poly):
    x, y = p
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1:
            inside = not inside
    return inside


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trail", action="store_true",
                    help="also count the near points R4's track-footprint rule would remove")
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
        formulation_v3.apply(cfg)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from common import load_model, run_episode
    from env import ASVLidarEnv
    scenes, _ = runner.load_scenes(cases)
    model = load_model(pilot.MODEL)

    class AuditV16(safety_v16.SafetyFilterV16):
        def __init__(self):
            super().__init__()
            self.decision, self.records, self.history = 0, [], []
            self.trail = None
            if args.trail:
                from safety_v21_trail import TrailExclusionPerception
                self.trail = TrailExclusionPerception(None)

        def filter(self, env, action):
            self.history += [(t.x, t.y) for t in env.targets]
            out, changed = super().filter(env, action)
            self.decision += 1
            why = self.last.get("why", "idle")
            snap = self._observer_snapshot
            mask = None
            if self.trail is not None:
                # The R4 rule applied offline to V16's own snapshot (V16 itself is unchanged).
                self.trail.footprints.append([(np.asarray(t.position, float), float(t.heading)) for t in snap.tracks
                                              if float(np.hypot(*np.asarray(t.velocity, float))) > 0.15])
                pts = np.asarray(snap.points, float).reshape(-1, 2)
                mask = np.zeros(len(pts), dtype=bool)
                for decision in self.trail.footprints:
                    for centre, heading in decision:
                        rel = pts - centre
                        sn, cs = np.sin(heading), np.cos(heading)
                        mask |= ((np.abs(rel[:, 0] * sn + rel[:, 1] * cs) <= self.trail.half_l)
                                 & (np.abs(rel[:, 0] * cs - rel[:, 1] * sn) <= self.trail.half_w))
            if why != "idle":
                self.records.append({"decision": self.decision, "why": why, "changed": bool(changed),
                                     "near_points": classify(env, snap, self.history[:-len(env.targets) or None], mask)})
            return out, changed

    class AuditEnv(ASVLidarEnv):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            self._safety_v2 = AuditV16()
            return result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as stream:
        for case in cases:
            cfg.SAFETY_VERSION = 16
            env = AuditEnv(render_mode=None, emergency_stop=True, low_speed_start_frac=0.0)
            try:
                result = run_episode(env, copy.deepcopy(scenes[case["case"]]), int(case["seed"]), "model", model)
                stream.write(json.dumps({"case": case["case"], "outcome": result["outcome"], "steps": result["steps"],
                                         "decisions": env._safety_v2.records}) + "\n")
                stream.flush()
            finally:
                env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
