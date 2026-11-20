"""Closed-loop twin of each field run (FIELD_TRIALS_PLAN.md section 7, level 2; claim C-18).

Each field run is re-simulated in the simulator that evaluated the policy:

* the scenario the bridge ran (its header's `--case`, `--layout`, `--test-case` or start and
  goal), checked against the logged path and boundary;
* the own ship placed at the logged start pose and speed (its ground truth if given, else the
  registered pose of the first decision);
* the target moved along its measured track (`ScriptedTarget`), not by a behaviour model;
* the same frozen policy, the layer off (as in every learned-behaviour result), the
  simulator's own sensor models with `--seeds` noise seeds;
* the plant: the training model (the identified July parameters, `ship.IDENTIFIED`) and,
  given `--calibrated`, the session-1 fit, with plant randomisation off in both, so the two
  twins differ only in their parameters and share their noise seeds.

The episode runs through `tools/tiers/common.run_episode`, so every metric and the true
trajectories are those of the test-set runs.  Nothing under `src/` changes; the twin
environment is a subclass.

    python twin.py <session_dir> [--seeds 10] [--calibrated params.json] [--processes 6]
Output: <session_dir>/analysis/twins/<run_id>/twins_<model>.csv and traj_<model>.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import trial_io as tio

import p3_bridge as pb  # noqa: E402  (puts the simulator's modules first; applies the propulsion stage)
import ship as shipmod  # noqa: E402
import targets as tgtmod  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

tiers = tio.load_module("tiers_common", tio.SIM_ROOT / "tools" / "tiers" / "common.py")

TWIN_SEED_BASE = 910_000      # no other set draws from this block
DEFAULT_SEEDS = 10
COMPARED_METRICS = ("min_target_range", "min_target_clearance", "min_obstacle_clearance", "min_boundary_clearance",
                    "rms_cte", "path_efficiency", "mean_speed", "min_speed", "completion_time_s",
                    "mean_abs_rudder_rate")


# ---------------------------------------------------------------------------
# The twin environment
# ---------------------------------------------------------------------------
class ScriptedTarget(tgtmod.Target):
    """A target that follows a measured track instead of a behaviour model."""

    def step(self, dt: float, *, own=None) -> None:
        self._t += float(dt)
        self.place(self._t)

    def place(self, t: float) -> None:
        x, y, h, s = self._script.at(np.array([t]))
        self.x, self.y, self.heading, self.speed = float(x[0]), float(y[0]), float(h[0]) % 360.0, float(s[0])


def scripted_target(track: tio.Track, encounter_class: str) -> ScriptedTarget:
    target = ScriptedTarget(0.0, 0.0, 0.0, 0.0, behaviour=tgtmod.T_CV, encounter_class=encounter_class,
                            confined=False)
    target._script = track
    target.place(0.0)
    return target


class TwinEnv(ASVLidarEnv):
    """`ASVLidarEnv` with a scripted target, a supplied start and fixed plant parameters."""

    def __init__(self, *, params: Optional[Dict[str, float]] = None, randomised: bool = False, **kwargs) -> None:
        super().__init__(render_mode=None, emergency_stop=False, low_speed_start_frac=0.0,
                         vessel_randomisation=(pb.cfg.VESSEL_RANDOMISATION_SCALE if randomised else None), **kwargs)
        self.twin_params = dict(params) if params is not None else None
        self.twin_target: Optional[tio.Track] = None
        self.twin_class = "null"
        self.twin_start: Optional[Tuple[float, float, float, float]] = None
        if self.twin_params is not None:
            self.model.set_params(self.twin_params)

    def _load_generated(self, built=None, _depth: int = 0) -> None:
        super()._load_generated(built, _depth)
        self.targets = [] if self.twin_target is None else [scripted_target(self.twin_target, self.twin_class)]

    def _apply_initial_conditions(self, options: dict) -> None:
        super()._apply_initial_conditions(options)
        if self.twin_start is None:
            return
        x, y, h, u = self.twin_start
        self.asv_x, self.asv_y, self.asv_h = float(x), float(y), float(h) % 360.0
        self.model._s[3, 0] = math.radians(self.asv_h)
        self.start_speed = self.u_body = max(0.0, float(u))
        self.model._s[0, 0] = self.u_body


# ---------------------------------------------------------------------------
# The bridge's scenario, rebuilt from its header
# ---------------------------------------------------------------------------
def bridge_scenario(header: Dict):
    args = header.get("args") or {}
    geo = header["geometry"]
    if args.get("scenario"):
        import setups
        built = setups.load(args["scenario"])["built"]
    elif args.get("case"):
        built, _ = pb.load_test_case(args["case"])
    elif args.get("layout"):
        built = pb.layout_scenario(args["layout"])
    elif args.get("test_case") is not None:
        sys.path.insert(0, pb.HERE)
        try:
            from test_run import TestCase
        finally:
            sys.path.pop(0)
        case = TestCase()
        sx, sy, gx, gy = case.position(test_case=int(args["test_case"]))
        built = pb.field_scenario((sx, sy), (gx, gy), panels=case.obstacles(test_case=int(args["test_case"])),
                                  case_id=f"test_run-{args['test_case']}")
    else:
        built = pb.field_scenario(geo["start"], geo["goal"], panels=geo.get("panels") or [], case_id="leg")
    return built


def twin_spec(inp: tio.TrialInputs) -> Dict:
    run = inp.run
    own = tio.own_track(run, inp.own_gt)
    x, y, h, u = (float(v[0]) for v in own.at([0.0]))
    if inp.own_gt is None:
        u = float(run.ego[0, 0])                 # the measured surge of the first decision
    cls = tio.ENCOUNTER_CLASS[inp.sheet["encounter"]][0]
    return {"run_id": run.run_id, "header": run.header, "start": (x, y, h, u), "class": cls,
            "target": None if inp.target is None else inp.target, "path": run.path, "boundary": run.boundary}


# ---------------------------------------------------------------------------
# Running the twins
# ---------------------------------------------------------------------------
_W: Dict = {}


def _init(model_path: str) -> None:
    import torch
    torch.set_num_threads(1)
    _W["model"] = tiers.load_model(model_path)
    run_cfg = Path(model_path).parent / "config.json"
    if run_cfg.exists():
        for name, value in (json.loads(run_cfg.read_text()).get("constant_overrides") or {}).items():
            setattr(pb.cfg, name, value)
    _W["envs"] = {}


def _env(name: str, params: Optional[Dict], randomised: bool) -> TwinEnv:
    key = (name, randomised)
    if key not in _W["envs"]:
        _W["envs"][key] = TwinEnv(params=params, randomised=randomised)
    return _W["envs"][key]


def _twin_job(job) -> Dict:
    spec, name, params, seed, randomised = job
    env = _env(name, params, randomised)
    env.twin_target, env.twin_class, env.twin_start = spec["target"], spec["class"], spec["start"]
    built = bridge_scenario(spec["header"])
    row = tiers.run_episode(env, built, seed, "model", _W["model"], keep_traj=True)
    path = np.asarray(env.path.points, dtype=float)
    if path.shape != spec["path"].shape or np.max(np.abs(path - spec["path"])) > 1e-4:
        raise ValueError(f"{spec['run_id']}: the rebuilt scenario's path differs from the logged one")
    row.update(run_id=spec["run_id"], twin_model=name, seed=seed)
    return row


def load_params(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    params = data.get("params", data)
    missing = [k for k in shipmod.IDENTIFIED if k not in params]
    if missing:
        raise ValueError(f"{path}: parameters missing: {missing}")
    return {k: float(params[k]) for k in shipmod.IDENTIFIED}


def run_twins(specs: List[Dict], *, models: Dict[str, Optional[Dict]], seeds: List[int], model_path: str,
              processes: int = 1, randomised: bool = False) -> List[Dict]:
    jobs = [(spec, name, params, seed, randomised) for spec in specs for name, params in models.items()
            for seed in seeds]
    if processes <= 1:
        _init(model_path)
        return [_twin_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=processes, initializer=_init, initargs=(model_path,)) as pool:
        return list(pool.map(_twin_job, jobs, chunksize=1))


def save(rows: List[Dict], out_dir: Path) -> None:
    import pandas as pd
    by: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        by.setdefault((r["run_id"], r["twin_model"]), []).append(r)
    for (run_id, name), group in by.items():
        folder = out_dir / run_id
        folder.mkdir(parents=True, exist_ok=True)
        trajs = {r["seed"]: r.pop("_traj") for r in group if "_traj" in r}
        pd.DataFrame(group).to_csv(folder / f"twins_{name}.csv", index=False)
        np.savez_compressed(folder / f"traj_{name}.npz",
                            **{f"{seed}|{k}": v for seed, tr in trajs.items() for k, v in tr.items()})


def compare(field: Dict, twins: List[Dict]) -> Dict:
    """Field against one model's twins: outcome agreement and each metric's percentile."""
    goals = [t["outcome"] == "goal" for t in twins]
    twin_goal_share = float(np.mean(goals)) if goals else float("nan")
    out = {"twin_runs": len(twins), "twin_goal_share": twin_goal_share,
           "twin_success": bool(twin_goal_share >= 0.5) if goals else None,
           "field_success": field["outcome"] == "goal"}
    out["outcome_agrees"] = (out["twin_success"] == out["field_success"]) if goals else None
    for m in COMPARED_METRICS:
        f = field.get(m, float("nan"))
        v = np.array([t.get(m, np.nan) for t in twins], dtype=float)
        v = v[np.isfinite(v)]
        if not np.isfinite(f) or len(v) == 0:
            out[f"{m}_pct"] = out[f"{m}_minus_median"] = float("nan")
            continue
        out[f"{m}_pct"] = float(100.0 * (np.mean(v < f) + 0.5 * np.mean(v == f)))
        out[f"{m}_minus_median"] = float(f - np.median(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--calibrated", help="parameter JSON of the session-1 fit (keys of ship.IDENTIFIED)")
    ap.add_argument("--processes", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--runs", help="comma-separated run ids (default: every valid run)")
    ap.add_argument("--model", default=pb.DEFAULT_MODEL)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    inputs = [i for i in tio.load_session(args.session_dir) if i.sheet["valid"]]
    if args.runs:
        wanted = set(args.runs.split(","))
        inputs = [i for i in inputs if i.run.run_id in wanted]
    if pb.sha256_file(args.model) != pb.EXPECTED_MODEL_SHA256:
        raise SystemExit("the policy file is not the deployed policy")
    models = {"training": None}
    if args.calibrated:
        models["calibrated"] = load_params(args.calibrated)
    seeds = [TWIN_SEED_BASE + i for i in range(args.seeds)]
    rows = run_twins([twin_spec(i) for i in inputs], models=models, seeds=seeds, model_path=args.model,
                     processes=args.processes)
    out = args.out or (args.session_dir / "analysis" / "twins")
    save(rows, out)
    print(f"{len(rows)} twin episodes for {len(inputs)} runs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
