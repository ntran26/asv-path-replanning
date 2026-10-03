"""Gate G4: the baseline-v4 pilot (`planning/BASELINE_V4_PLAN.md`, 2026-10-03).

PPO continued from the baseline-v3 PPO run's own 1.5 M checkpoint (the start of
stage 6, where static panels first meet the encounter) for 0.5 M steps under the
draft v4 overlay (`formulation_v4`: fix 1 on, denser field layouts, crossings
weighted up).  The **control** is that run's own 2.0 M checkpoint -- the same
seed and start, trained 1.5 -> 2.0 M under v3, already paid for.

    python tools/diagnostics/v4_gates/pilot.py train A                # v4, gamma 0.951
    python tools/diagnostics/v4_gates/pilot.py train B --gamma 0.98   # v4, longer horizon
    python tools/diagnostics/v4_gates/pilot.py eval A B               # arms and control, paired

Arm B changes the discount when it starts; its value function was fitted for
0.951, so the arm begins at a disadvantage (a caveat when reading it).

Evaluation (development side only; the test set is never used), the same
scenarios and episode seeds for every model:
* conflict set: the v3 field development set's HO/CRP/CRS (78) plus G1's 40
  near-deployment conflicts;
* field development set, all 150;
* frozen-like: the formulation's development set (120).
Pass: conflict set >= +15 points over the control; frozen-like >= -2 points.

Runs go to `runs/pilot_v4_<arm>/`; results to `results/v4_gates/g4_*`.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers"), str(Path(__file__).parent)]

import numpy as np
import pandas as pd

import constants as cfg
import curriculum
import train_formulation as tf

RUN = ROOT / "runs" / "ppo_formulation_seed0_bl3"
START, END, SCHEDULE_TOTAL = 1_500_000, 2_000_000, 2_500_000
OVERLAY = "formulation_v4"
OUT = ROOT / "results" / "v4_gates"


def _make_env(rank: int, seed: int, stage: int, overlay: str):
    def _init():
        import constants as c
        import curriculum as cu
        importlib.import_module(overlay).apply(c)
        cu.apply_stage(tf.PROPULSION_STAGE)
        from env import ASVLidarEnv
        env = ASVLidarEnv(render_mode=None, scenario_stage=stage, vessel_randomisation=c.VESSEL_RANDOMISATION_SCALE,
                          emergency_stop=False, low_speed_start_frac=0.15)       # baseline-v3 run_args
        env.reset(seed=seed + rank)
        return env
    return _init


def train(arm: str, gamma: float, steps: int = END - START):
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    tf._no_efficiency_mode()
    mod = importlib.import_module(OVERLAY)
    mod.apply(cfg)
    tf.STAGE_SCHEDULE = cfg.CURRICULUM_STAGE_FRACTIONS
    out = ROOT / "runs" / f"pilot_v4_{arm}"
    out.mkdir(parents=True, exist_ok=True)
    stage = tf.stage_at(START / SCHEDULE_TOTAL)
    vec = SubprocVecEnv([_make_env(i, 1_700_000, stage, OVERLAY) for i in range(10)])
    vec = tf.RetryingVecMonitor(vec, filename=str(out / "monitor.csv"),
                                info_keywords=("reached_goal", "collided", "scenario_class"))
    vec = VecNormalize.load(str(RUN / f"ppo_vecnormalize_{START}_steps.pkl"), vec)
    vec.training, vec.norm_reward = True, True
    model = tf.ALGORITHMS["ppo"].load(str(RUN / f"ppo_{START}_steps.zip"), env=vec, device="cpu")
    base_gamma = float(model.gamma)
    if abs(gamma - base_gamma) > 1e-9:
        model.gamma = gamma
        model.rollout_buffer.gamma = gamma
        vec.gamma = gamma
        vec.returns = np.zeros(vec.num_envs)
    config = {"pilot": arm, "overlay": OVERLAY, "overlay_snapshot": mod.snapshot(), "start_checkpoint": str(RUN / f"ppo_{START}_steps.zip"),
              "steps": [START, END], "gamma": gamma, "gamma_checkpoint": base_gamma, "algo": "ppo",
              "constant_overrides": dict(mod.CONSTANT_OVERRIDES), "started": time.strftime("%Y-%m-%d %H:%M")}
    (out / "config.json").write_text(json.dumps(config, indent=1, default=str))
    callbacks = CallbackList([
        tf.ScenarioStageCallback(SCHEDULE_TOTAL, str(out / "curriculum.json")),
        CheckpointCallback(save_freq=250_000 // 10, save_path=str(out), name_prefix="ppo", save_vecnormalize=True),
    ])
    print(f"[PILOT {arm}] {OVERLAY} from {START:,} to {END:,}, gamma {gamma} (checkpoint {base_gamma:.4f}), "
          f"stage {stage}", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=callbacks, reset_num_timesteps=False, progress_bar=False)
    model.save(str(out / "final_model.zip"))
    vec.save(str(out / "final_vecnormalize.pkl"))
    config["wall_clock_s"] = time.time() - t0
    (out / "config.json").write_text(json.dumps(config, indent=1, default=str))
    print(f"[PILOT {arm}] done in {(time.time() - t0) / 3600:.1f} h", flush=True)


def _eval_sets():
    import formulation_v3 as fv
    import gates
    field = fv.field_development_set()
    sets = []
    for i, b in enumerate(field):
        code = b.case_id.split("-")[1]
        sets.append((b, 960_000 + i, {"set": "field_dev", "code": code, "conflict": code in ("HO", "CRP", "CRS")}))
    for j, s in enumerate(gates.conflict_scenarios()):
        sets.append((s["built"], 961_000 + j, {"set": "g1_conflict", "code": s["group"], "conflict": True}))
    for k, b in enumerate(tf.development_set(20)):
        sets.append((b, 962_000 + k, {"set": "dev_v2", "code": b.encounter_class, "conflict": False}))
    return sets


def evaluate(arms, processes: int, limit: int = 0):
    from common import run_pool
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    sets = _eval_sets()
    if limit:
        sets = sets[:limit] + sets[150:150 + limit] + sets[-limit:]
    models = {"control": RUN / f"ppo_{END}_steps.zip"}
    models.update({arm: ROOT / "runs" / f"pilot_v4_{arm}" / "final_model.zip" for arm in arms})
    rows = []
    for name, path in models.items():
        jobs = [(b, seed, "model", None, {**meta, "model": name}) for b, seed, meta in sets]
        # run_pool's workers apply the run's config.json constant_overrides (fix 1 for the arms).
        rows += run_pool(jobs, model_path=path, processes=processes, overrides={"EMERGENCY_STOP_ENABLED": False})
        print(f"[G4] evaluated {name}", flush=True)
    d = pd.DataFrame(rows)
    d["goal"] = d.outcome == "goal"
    d["case"] = d.set + ":" + d.groupby(["model", "set"]).cumcount().astype(str)
    d.to_csv(OUT / "g4_episodes.csv", index=False)
    lines = ["G4 pilot: PPO from the v3 run's 1.5 M checkpoint, 0.5 M steps; control = that run at 2.0 M", ""]
    table = pd.DataFrame({
        "conflict": d[d.conflict].groupby("model").goal.mean(),
        "field_dev": d[d.set == "field_dev"].groupby("model").goal.mean(),
        "g1_conflict": d[d.set == "g1_conflict"].groupby("model").goal.mean(),
        "dev_v2 (frozen-like)": d[d.set == "dev_v2"].groupby("model").goal.mean()}).round(3)
    lines += [table.to_string(), ""]
    lines += ["-- field development set by encounter",
              d[d.set == "field_dev"].pivot_table(index="model", columns="code", values="goal").round(2).to_string(), ""]
    ctrl = d[d.model == "control"].set_index("case").goal
    for arm in arms:
        x = d[d.model == arm].set_index("case").goal
        c_ids = d[(d.model == arm) & d.conflict].case
        f_ids = d[(d.model == arm) & (d.set == "dev_v2")].case
        dc = x[c_ids].mean() - ctrl[c_ids].mean()
        df = x[f_ids].mean() - ctrl[f_ids].mean()
        gained = int((x[c_ids] & ~ctrl[c_ids]).sum())
        lost = int((~x[c_ids] & ctrl[c_ids]).sum())
        verdict = "PASS" if dc >= 0.15 and df >= -0.02 else "FAIL"
        lines.append(f"arm {arm}: conflict {dc:+.1%} (gained {gained}, lost {lost} of {len(c_ids)}), "
                     f"frozen-like {df:+.1%} -> G4 {verdict} (needs >= +15 and >= -2 points)")
    text = "\n".join(lines) + "\n"
    (OUT / "g4_summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("train", "eval"))
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--gamma", type=float, default=0.9509900498999999)
    ap.add_argument("--processes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=END - START, help="training steps (smoke tests use fewer)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only a few episodes per set (smoke test)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "train":
        train(a.arms[0], a.gamma, a.steps)
    else:
        evaluate(a.arms, a.processes, a.limit)


if __name__ == "__main__":
    main()
