"""False moving tracks on static obstacles in V16's snapshot: an offline audit (V21 phase).

Replays V16 with the runner's exact episode setup and records, at every decision, each
track in V16's own snapshot with its estimated speed, its distance to the nearest true
target hull and to the nearest true obstacle polygon (truth for offline scoring only).
A track is `false_on_obstacle` when it moves faster than 0.15 m/s (the safety
perception's moving threshold), no true target lies within `TARGET_M` of it and a true
obstacle lies within `OBSTACLE_M`.  Static points within 2 m of any track are left out
of the snapshot (`classical.common.Perception`), so such a panel is represented only as
a target moving at constant velocity.  The audit also records the number of remembered
static points within 1 m of the hull and the hull's true clearance to the obstacles.

    python -B tools/diagnostics/safety/v21_track_audit.py --cases DV3:DV3-CRS-CV-17 --out tracks.jsonl
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
TARGET_M = 1.5
OBSTACLE_M = 0.8
MOVING_MPS = 0.15


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True)
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
    from v21_static_audit import point_polygon_distance
    from v21_unchecked_probe import polygon_distance
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
            self.decision, self.records = 0, []

        def filter(self, env, action):
            out, changed = super().filter(env, action)
            self.decision += 1
            snap = self._observer_snapshot
            tracks = []
            for t in snap.tracks:
                p = np.asarray(t.position, float)
                d_tgt = min((float(np.linalg.norm(p - np.array([tt.x, tt.y]))) for tt in env.targets), default=np.inf)
                d_obs = min((float(point_polygon_distance(p[None], o)[0]) for o in env.obstacles), default=np.inf)
                spd = float(np.hypot(*np.asarray(t.velocity, float)))
                tracks.append({"id": int(t.id), "speed": spd, "d_target": d_tgt, "d_obstacle": d_obs,
                               "false_on_obstacle": bool(spd > MOVING_MPS and d_tgt > TARGET_M and d_obs < OBSTACLE_M)})
            hull = env.hull_polygon()
            self.records.append({"decision": self.decision, "why": self.last.get("why", "idle"), "changed": bool(changed),
                                 "tracks": tracks,
                                 "true_obstacle_clearance": min((polygon_distance(hull, o) for o in env.obstacles), default=None)})
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
                                         "decisions": env._safety_v2.records}, default=float) + "\n")
                stream.flush()
            finally:
                env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
