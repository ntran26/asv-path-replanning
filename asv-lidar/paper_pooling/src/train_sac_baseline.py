"""SAC baseline, retrained from scratch across seeds under the same protocol as PPO.

    python src/train_sac_baseline.py --seeds 0 1 2
    python src/train_sac_baseline.py --seeds 99 --timesteps 20000 --smoke

Why this exists
---------------
The published SAC result is a **single** checkpoint (`models/sac_model_1M.zip`),
while the PPO baseline has three seeds.  Comparing a 1-run method against a
3-seed method is an asymmetry: PPO's interval reflects seed *and* episode
variance, SAC's reflects episode variance only.  This script removes that
asymmetry by retraining SAC from scratch on three seeds under exactly the
protocol used for PPO -- same env, same 1M total interactions, same 8 workers,
same replicated propulsion curriculum scheduled on total environment steps, same
evaluation grid and selection score.

It also **tests** a claim that `BASELINES_RESULTS.md` section 4 currently only
argues: that the stage-3 collapse is a property of the environment's reward
(the anti-stall term never fires, and `r_thrust` halves as `RPM_DELTA` widens)
rather than a weakness specific to PPO.  If retrained SAC seeds collapse at the
same boundary, that claim becomes measured.  If they do not, the claim is wrong
and the algorithms genuinely differ in robustness -- either outcome is worth
having, and neither is assumed here.

Relationship to the published model
-----------------------------------
These runs do **not** replace `models/sac_model_1M.zip`.  That checkpoint is the
manuscript's artifact and is reported separately as "SAC (published)".  The runs
here are reported as "SAC (retrained, 3 seeds)".  Nothing this script writes can
touch a protected artifact: every path is scoped to `models/sac_seed{N}/`, and
the eval callback's two hardcoded `best_model.zip` saves are redirected by
`RunScopedEvalCallback` (reused from `train_ppo_baseline.py`).

Hyperparameters are those of the published SAC run, read out of
`train.build_model`: MultiInputPolicy, lr 5e-5, batch 512, gamma 0.99, buffer
1e6, train_freq 1, gradient_steps 1, ent_coef "auto", and SB3's default SAC
net_arch [256, 256] with ReLU (policy_kwargs is not set there).
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
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import config as cfg
from curriculum import (
    CURRICULUM_SCHEDULE,
    CurriculumASVLidarEnv,
    RpmCurriculumCallback,
    stage_for_timestep,
)
from train_ppo_baseline import (
    BASE_SEED,
    EVAL_FREQ,
    NUM_ENVS,
    SAVE_FREQ,
    TOTAL_TIMESTEPS,
    RunScopedEvalCallback,
    make_env,
)

# Exactly the published SAC configuration (train.build_model), with net_arch
# stated explicitly rather than left implicit in the SB3 default.
SAC_HYPERPARAMS: Dict[str, Any] = {
    "learning_rate": 5e-5,
    "batch_size": 512,
    "gamma": 0.99,
    "buffer_size": 1_000_000,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
}


def hyperparameter_record(seed: int, timesteps: int, num_envs: int,
                          schedule=CURRICULUM_SCHEDULE) -> Dict[str, Any]:
    return {
        "algorithm": "SAC",
        "sb3_policy": "MultiInputPolicy",
        "seed": int(seed),
        "total_timesteps": int(timesteps),
        "num_envs": int(num_envs),
        "vec_env": "SubprocVecEnv",
        "eval_freq": EVAL_FREQ,
        "save_freq": SAVE_FREQ,
        "device": "cpu",
        "hyperparameters": dict(SAC_HYPERPARAMS),
        "policy_kwargs": {
            "net_arch": [256, 256],
            "activation_fn": "ReLU",
            "note": "SB3 SAC default; train.build_model does not set policy_kwargs",
        },
        "curriculum": {
            "schedule_total_env_steps": [
                {"from_timestep": b, "stage": s} for b, s in schedule
            ],
            "counter": "model.num_timesteps (total environment interactions)",
            "source": "plotting/plot_training_curves.py:96",
            "note": (
                "Same replicated schedule as the PPO baseline, so the two are "
                "directly comparable. See BASELINES_NOTES.md section 6."
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
        },
        "relationship_to_published_model": (
            "Independent retrain. Does not replace models/sac_model_1M.zip, "
            "which remains the manuscript's artifact."
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "stable_baselines3": __import__("stable_baselines3").__version__,
        },
    }


def train_one_seed(seed: int, timesteps: int, num_envs: int, eval_freq: int,
                   smoke: bool = False, schedule=CURRICULUM_SCHEDULE) -> Dict[str, Any]:
    run_dir = os.path.join("models", f"sac_seed{seed}")
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

    model = SAC(
        "MultiInputPolicy", vec_env, verbose=1,
        tensorboard_log="./sac_baseline_log/",
        seed=env_seed, device="cpu",
        **SAC_HYPERPARAMS,
    )

    callbacks = [
        RpmCurriculumCallback(schedule=schedule,
                              log_path=os.path.join(run_dir, "curriculum.json")),
        CheckpointCallback(
            save_freq=max(SAVE_FREQ // max(num_envs, 1), 1),
            save_path=run_dir, name_prefix="sac_model",
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
    model.learn(total_timesteps=timesteps, tb_log_name=f"sac_seed{seed}",
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
                    help="short run for wiring/throughput checks")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    schedule = CURRICULUM_SCHEDULE
    if args.smoke:
        scale = args.timesteps / TOTAL_TIMESTEPS
        schedule = tuple((int(b * scale), s) for b, s in CURRICULUM_SCHEDULE)
        print(f"SMOKE: curriculum scaled to {schedule}")

    print(f"SAC baseline: seeds={args.seeds} timesteps={args.timesteps:,} "
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
