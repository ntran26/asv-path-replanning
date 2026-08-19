"""PPO baseline, trained under conditions identical to the published SAC run.

    python src/train_ppo_baseline.py --seeds 0 1 2
    python src/train_ppo_baseline.py --seeds 0 --timesteps 20000 --smoke

Everything a run produces goes under `models/ppo_seed{N}/`:

    hyperparameters.json    every hyperparameter, plus the env/curriculum config
    curriculum.json         the stage transitions actually applied, with timesteps
    monitor.csv             VecMonitor training episode returns
    eval_metrics.csv/.json  per-episode evaluation grid, same grid as SAC's
    eval_summary.csv/.json  the learning curve: mean return vs timesteps
    best_model.zip          best checkpoint by the SAC selection score
    ppo_model_*_steps.zip   periodic checkpoints
    final_model.zip

Protected-artifact safety
-------------------------
The SAC run writes `best_model.zip`, `eval_metrics.*`, `eval_summary.*` and
`train_monitor.csv` to the repository root, and checkpoints to `models/` with a
`sac_model` prefix.  Every path here is scoped to the run directory, and
`RunScopedEvalCallback` exists specifically to override the two hardcoded
`best_model.zip` saves in `train.EvalMetricsCallback`.  Nothing this script
writes can collide with a SAC artifact.

Matched to SAC, deliberately
----------------------------
* same env, observation, reward, termination (`ASVLidarEnv`, untouched)
* same 1M total environment interactions, same 8 `SubprocVecEnv` workers
* same staged propulsion curriculum, scheduled on total env steps (`curriculum.py`)
* same evaluation grid and selection score (reused from `train.py`)
* `net_arch=[256, 256]` with ReLU, matching SAC's `MultiInputPolicy` default,
  and `gamma=0.99` matching SAC

Documented differences
----------------------
* PPO is on-policy and needs `n_steps`/`n_epochs`/`clip_range`, which have no
  SAC counterpart.  These are left at SB3's continuous-control defaults.
* PPO has separate policy and value networks; SAC has an actor and twin critics.
  `[256, 256]` is applied to both `pi` and `vf`.
* This is NOT the PPO config in `train.py:build_model` ([64, 64] Tanh,
  gamma 0.999, ent_coef 0.03).  That network is 16x smaller than SAC's and
  would make the comparison a straw man.  See BASELINES_NOTES.md section 7.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import config as cfg
from curriculum import (
    CURRICULUM_SCHEDULE,
    CurriculumASVLidarEnv,
    RpmCurriculumCallback,
    stage_for_timestep,
)
from train import EvalMetricsCallback, SUMMARY_HEADER, append_csv_row

TOTAL_TIMESTEPS = 1_000_000     # matches the SAC run
NUM_ENVS = 8                    # matches the SAC run
EVAL_FREQ = 50_000              # matches the SAC run
SAVE_FREQ = 100_000
BASE_SEED = 675973              # the seed the SAC run used, per src/README.md

# SB3 PPO defaults for continuous control, except net_arch/gamma which are
# matched to SAC.  Per the brief these are only to be adjusted if training
# visibly fails; any change must be recorded here and in BASELINES_RESULTS.md.
PPO_HYPERPARAMS: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,              # matched to SAC (SB3 default is also 0.99)
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


class RunScopedEvalCallback(EvalMetricsCallback):
    """`EvalMetricsCallback` with the hardcoded `best_model.zip` saves redirected.

    The evaluation grid, the metrics and the selection score are inherited
    unchanged from `train.py`, so the PPO learning curve is produced by exactly
    the same procedure as SAC's.  Only the two save destinations differ.
    """

    def __init__(self, eval_env, *, best_model_path: str, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.best_model_path = best_model_path
        self.best_dir = os.path.dirname(best_model_path) or "."

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        ep_metrics = self._run_eval_grid()
        summary = self._summarise(ep_metrics)

        self.summary_rows.append(summary)
        append_csv_row(self.summary_csv_path, SUMMARY_HEADER,
                       [summary.get(k) for k in SUMMARY_HEADER])
        with open(self.json_path, "w") as f:
            json.dump(self.rows, f, indent=2)
        with open(self.summary_json_path, "w") as f:
            json.dump(self.summary_rows, f, indent=2)

        if summary["success_rate"] > 0.0 and summary["selection_score"] > self.best_score:
            self.best_score = summary["selection_score"]
            self.model.save(self.best_model_path)
            self.model.save(os.path.join(
                self.best_dir, f"best_model_{self.num_timesteps}.zip"))
            if self.verbose:
                print(f"New BEST model saved: score={self.best_score:.3f}, "
                      f"success={summary['success_rate']:.3f}")

        for key, value in summary.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                self.logger.record(f"eval/{key}", value)
        return True


def make_env(seed: int, rank: int):
    """Worker factory.  Mirrors `train.make_env` but builds the curriculum env."""
    def _init():
        env = CurriculumASVLidarEnv(map_width=10.0, map_height=25.0,
                                    path_mode="straight")
        env.reset(seed=seed + rank)
        return env
    return _init


def hyperparameter_record(seed: int, timesteps: int, num_envs: int,
                          schedule=CURRICULUM_SCHEDULE) -> Dict[str, Any]:
    """Everything a reviewer could ask for about how this run was configured."""
    return {
        "algorithm": "PPO",
        "sb3_policy": "MultiInputPolicy",
        "seed": int(seed),
        "total_timesteps": int(timesteps),
        "num_envs": int(num_envs),
        "vec_env": "SubprocVecEnv",
        "eval_freq": EVAL_FREQ,
        "save_freq": SAVE_FREQ,
        "device": "cpu",
        "hyperparameters": dict(PPO_HYPERPARAMS),
        "policy_kwargs": {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": "ReLU",
            "note": "matched to SAC MultiInputPolicy default net_arch [256, 256], ReLU",
        },
        "curriculum": {
            "schedule_total_env_steps": [
                {"from_timestep": b, "stage": s} for b, s in schedule
            ],
            "counter": "model.num_timesteps (total environment interactions)",
            "source": "plotting/plot_training_curves.py:96",
            "note": (
                "No curriculum scheduler exists in the repository; the SAC run "
                "advanced stages by hand-editing config.py and resuming. This "
                "schedule replicates the documented boundaries explicitly. See "
                "BASELINES_NOTES.md section 6."
            ),
        },
        "environment": {
            "class": "CurriculumASVLidarEnv (subclass of ASVLidarEnv, adds only a stage setter)",
            "map_width": 10.0,
            "map_height": 25.0,
            "path_mode": "straight",
            "max_episode_steps": cfg.MAX_EPISODE_STEPS,
            "update_rate_s": cfg.UPDATE_RATE,
            "lambda": cfg.DEFAULT_EVAL_LAMBDA,
            "obs_border_mode": cfg.OBS_BORDER_MODE,
            "cruise_rpm": cfg.CRUISE_RPM,
            "rpm_stages": {str(k): list(v) for k, v in cfg.RPM_STAGES.items()},
            "train_obs_counts": list(cfg.TRAIN_OBS_COUNTS),
            "train_obs_probs": list(cfg.TRAIN_OBS_PROBS),
            "train_scenario_modes": list(cfg.TRAIN_SCENARIO_MODES),
            "train_scenario_probs": list(cfg.TRAIN_SCENARIO_PROBS),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "stable_baselines3": __import__("stable_baselines3").__version__,
        },
    }


def train_one_seed(seed: int, timesteps: int, num_envs: int,
                   eval_freq: int, smoke: bool = False,
                   schedule=CURRICULUM_SCHEDULE) -> Dict[str, Any]:
    run_dir = os.path.join("models", f"ppo_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    record = hyperparameter_record(seed, timesteps, num_envs, schedule)
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(record, f, indent=2)

    env_seed = BASE_SEED + 100_000 * seed
    vec_env = VecMonitor(
        SubprocVecEnv([make_env(env_seed, i) for i in range(num_envs)]),
        filename=os.path.join(run_dir, "monitor.csv"),
    )
    eval_env = CurriculumASVLidarEnv(map_width=10.0, map_height=25.0,
                                     path_mode="straight")
    eval_env.reset(seed=env_seed + 10_000)

    model = PPO(
        "MultiInputPolicy", vec_env, verbose=1,
        tensorboard_log="./ppo_baseline_log/",
        seed=env_seed, device="cpu",
        policy_kwargs=dict(activation_fn=nn.ReLU,
                           net_arch=dict(pi=[256, 256], vf=[256, 256])),
        **PPO_HYPERPARAMS,
    )

    callbacks = [
        RpmCurriculumCallback(schedule=schedule,
                              log_path=os.path.join(run_dir, "curriculum.json")),
        CheckpointCallback(
            save_freq=max(SAVE_FREQ // max(num_envs, 1), 1),
            save_path=run_dir, name_prefix="ppo_model",
            save_replay_buffer=False, save_vecnormalize=False,
        ),
        RunScopedEvalCallback(
            eval_env, eval_freq=eval_freq, max_steps=2_000,
            best_model_path=os.path.join(run_dir, "best_model.zip"),
            csv_path=os.path.join(run_dir, "eval_metrics.csv"),
            json_path=os.path.join(run_dir, "eval_metrics.json"),
            summary_csv_path=os.path.join(run_dir, "eval_summary.csv"),
            summary_json_path=os.path.join(run_dir, "eval_summary.json"),
        ),
    ]

    started = time.time()
    model.learn(total_timesteps=timesteps, tb_log_name=f"ppo_seed{seed}",
                callback=CallbackList(callbacks), progress_bar=True,
                reset_num_timesteps=True)
    wall_clock = time.time() - started

    final_path = os.path.join(run_dir, "final_model.zip")
    model.save(final_path)
    vec_env.close()
    eval_env.close()

    record["wall_clock_seconds"] = wall_clock
    record["completed"] = True
    record["smoke_test"] = bool(smoke)
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n[seed {seed}] done in {wall_clock / 60:.1f} min -> {final_path}")
    return record


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    ap.add_argument("--num-envs", type=int, default=NUM_ENVS)
    ap.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    ap.add_argument("--smoke", action="store_true",
                    help="short run for wiring checks; scales the curriculum down")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    schedule = CURRICULUM_SCHEDULE
    if args.smoke:
        # Compress the schedule so every transition is exercised in a short run.
        scale = args.timesteps / TOTAL_TIMESTEPS
        schedule = tuple((int(b * scale), s) for b, s in CURRICULUM_SCHEDULE)
        print(f"SMOKE: curriculum scaled to {schedule}")

    print(f"PPO baseline: seeds={args.seeds} timesteps={args.timesteps:,} "
          f"num_envs={args.num_envs}")
    print(f"Curriculum boundaries (total env steps): {[b for b, _ in schedule]}")
    print(f"Stage at t=0: {stage_for_timestep(0, schedule)}, "
          f"at t={args.timesteps:,}: {stage_for_timestep(args.timesteps, schedule)}")

    records: List[Dict[str, Any]] = []
    for seed in args.seeds:
        print(f"\n{'=' * 70}\nSEED {seed}\n{'=' * 70}")
        records.append(train_one_seed(seed, args.timesteps, args.num_envs,
                                      args.eval_freq, smoke=args.smoke,
                                      schedule=schedule))

    total = sum(r["wall_clock_seconds"] for r in records)
    print(f"\nAll seeds complete. Total wall clock: {total / 3600:.2f} h")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
