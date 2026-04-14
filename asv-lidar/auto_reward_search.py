from __future__ import annotations

import argparse
import json
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from rl_env_reward_search import ASVRewardSearchEnv
from ship_model import VESSEL_LENGTH
from train_test_asv import (
    DEFAULT_BENCHMARK_CASES,
    FixedBenchmarkCallback,
    evaluate_benchmark,
)

VARIANTS: List[Dict[str, Any]] = [
    {"name": "baseline_copy", "obs_mode": "baseline", "reward_mode": "baseline"},
    {"name": "compact_threat", "obs_mode": "compact", "reward_mode": "threat_adaptive"},
    {"name": "compact_progress", "obs_mode": "compact", "reward_mode": "threat_progress"},
    {"name": "compact_turn_guided", "obs_mode": "compact", "reward_mode": "turn_guided"},
    {"name": "compact_guided_path", "obs_mode": "compact", "reward_mode": "guided_path"},
    {"name": "compact_teacher_guided", "obs_mode": "compact", "reward_mode": "teacher_guided"},
    {"name": "teacher_compact_guided", "obs_mode": "teacher_compact", "reward_mode": "teacher_guided"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-timesteps", type=int, default=40_000)
    parser.add_argument("--continue-timesteps", type=int, default=60_000)
    parser.add_argument("--max-followups", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-max-steps", type=int, default=600)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--target-success", type=float, default=0.95)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default="reward_search_runs")
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--benchmark-cases", type=int, nargs="*", default=DEFAULT_BENCHMARK_CASES)
    parser.add_argument("--train-case-pool", type=int, nargs="*", default=DEFAULT_BENCHMARK_CASES)
    parser.add_argument("--train-random", action="store_true")
    parser.add_argument("--warm-start-random-episodes", type=int, default=0)
    parser.add_argument("--random-eval-episodes", type=int, default=0)
    parser.add_argument("--random-eval-seed", type=int, default=20_000)
    parser.add_argument("--post-warmstart-timesteps", type=int, default=None)
    return parser.parse_args()


def build_search_model(env):
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=nn.Tanh)
    return PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_log/",
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
    )


def collect_teacher_dataset(
    variant: Dict[str, Any],
    cases: List[int] | None,
    seed: int,
    random_episodes: int = 0,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    observations: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []

    fixed_cases = [] if cases is None else [int(case) for case in cases]
    for case in fixed_cases:
        env = ASVRewardSearchEnv(
            render_mode=None,
            obs_mode=variant["obs_mode"],
            reward_mode=variant["reward_mode"],
        )
        env.test_case = int(case)
        obs, _ = env.reset(seed=seed + int(case))
        terminated = False
        steps = 0
        while not terminated and steps < 600:
            action = np.array([env.ref_rudder_cmd, env.ref_throttle_cmd], dtype=np.float32)
            observations.append({key: np.array(value, copy=True) for key, value in obs.items()})
            actions.append(action.copy())
            obs, _, terminated, _, _ = env.step(action)
            steps += 1
        env.close()

    for episode_idx in range(int(random_episodes)):
        env = ASVRewardSearchEnv(
            render_mode=None,
            obs_mode=variant["obs_mode"],
            reward_mode=variant["reward_mode"],
        )
        obs, _ = env.reset(seed=seed + 100_000 + episode_idx)
        terminated = False
        steps = 0
        while not terminated and steps < 600:
            action = np.array([env.ref_rudder_cmd, env.ref_throttle_cmd], dtype=np.float32)
            observations.append({key: np.array(value, copy=True) for key, value in obs.items()})
            actions.append(action.copy())
            obs, _, terminated, _, _ = env.step(action)
            steps += 1
        env.close()

    return observations, np.asarray(actions, dtype=np.float32)


def teacher_warm_start(
    model: PPO,
    variant: Dict[str, Any],
    cases: List[int] | None,
    seed: int,
    random_episodes: int = 0,
) -> Dict[str, Any]:
    observations, actions = collect_teacher_dataset(variant, cases, seed, random_episodes=random_episodes)
    fixed_cases = [] if cases is None else [int(case) for case in cases]
    if not observations:
        return {"n_samples": 0, "epochs": 0, "final_bc_loss": None, "random_episodes": int(random_episodes)}

    keys = list(observations[0].keys())
    obs_arrays = {key: np.stack([obs[key] for obs in observations]).astype(np.float32) for key in keys}
    action_tensor = torch.as_tensor(actions, device=model.policy.device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    batch_size = 128
    epochs = 20
    final_loss = None

    for _ in range(epochs):
        perm = np.random.permutation(len(observations))
        for start in range(0, len(observations), batch_size):
            idx = perm[start : start + batch_size]
            obs_t = {
                key: torch.as_tensor(value[idx], device=model.policy.device)
                for key, value in obs_arrays.items()
            }
            features = model.policy.extract_features(obs_t)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            mean_actions = model.policy.action_net(latent_pi)
            loss = F.mse_loss(mean_actions, action_tensor[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

    return {
        "n_samples": int(len(observations)),
        "epochs": epochs,
        "final_bc_loss": final_loss,
        "random_episodes": int(random_episodes),
        "fixed_cases": fixed_cases,
    }


def score_key(summary: Dict[str, Any]):
    return (
        float(summary["success_rate"]),
        float(summary["goal_rate"]),
        -float(summary["obstacle_rate"]),
        -float(summary["border_rate"]),
        -float(summary["timeout_rate"]),
        float(summary["mean_reward"]),
    )


def make_train_env(seed: int, rank: int, variant: Dict[str, Any]):
    def _init():
        env = ASVRewardSearchEnv(
            render_mode=None,
            obs_mode=variant["obs_mode"],
            reward_mode=variant["reward_mode"],
        )
        env.set_train_case_pool(variant.get("train_case_pool"))
        env.reset(seed=seed + rank)
        return env

    return _init


def evaluate_randomized(
    actor,
    variant: Dict[str, Any],
    episodes: int,
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    if episodes <= 0:
        return {}

    rewards: List[float] = []
    lengths: List[int] = []
    successes = 0
    collisions = 0
    timeouts = 0

    for episode_idx in range(int(episodes)):
        env = ASVRewardSearchEnv(
            render_mode=None,
            obs_mode=variant["obs_mode"],
            reward_mode=variant["reward_mode"],
        )
        obs, _ = env.reset(seed=seed + episode_idx)
        total_reward = 0.0
        terminated = False
        steps = 0
        while not terminated and steps < max_steps:
            action, _ = actor.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(np.asarray(action, dtype=np.float32))
            total_reward += float(reward)
            steps += 1
            terminated = bool(term or trunc)

        reached_goal = bool(env.distance_to_goal <= (VESSEL_LENGTH / 2.0))
        collided = bool(env._check_collision_geom())
        if reached_goal and not collided:
            successes += 1
        elif collided:
            collisions += 1
        else:
            timeouts += 1

        rewards.append(total_reward)
        lengths.append(steps)
        env.close()

    total = float(max(episodes, 1))
    return {
        "episodes": int(episodes),
        "success_rate": float(successes / total),
        "collision_rate": float(collisions / total),
        "timeout_rate": float(timeouts / total),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_ep_len": float(np.mean(lengths)) if lengths else 0.0,
    }


def pick_best_record(history: List[Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not history:
        return {"timesteps": fallback.get("timesteps", 0), "summary": fallback}
    return max(history, key=lambda row: score_key(row["summary"]))


def run_phase(
    args: argparse.Namespace,
    variant: Dict[str, Any],
    run_dir: Path,
    phase_idx: int,
    total_timesteps: int,
    resume_model_path: str | None = None,
) -> Dict[str, Any]:
    vec_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    env_fns = [make_train_env(args.seed + phase_idx * 1_000, i, variant) for i in range(args.num_envs)]
    vec_env = VecMonitor(
        vec_cls(env_fns),
        filename=str(run_dir / f"train_monitor_phase_{phase_idx}.csv"),
    )

    if resume_model_path:
        model = PPO.load(resume_model_path, env=vec_env)
        warm_start_info = None
        learn_timesteps = int(total_timesteps)
    else:
        model = build_search_model(vec_env)
        warm_start_info = None
        learn_timesteps = int(total_timesteps)
        if variant["reward_mode"] == "teacher_guided":
            fixed_cases = None if variant.get("train_case_pool") is None else list(variant.get("train_case_pool") or [])
            random_warm_episodes = int(args.warm_start_random_episodes)
            if variant.get("train_case_pool") is None and random_warm_episodes <= 0:
                random_warm_episodes = 128
            warm_start_info = teacher_warm_start(
                model,
                variant,
                fixed_cases,
                seed=args.seed + phase_idx * 1_000,
                random_episodes=random_warm_episodes,
            )
            model.ent_coef = 0.0
            model.lr_schedule = lambda _: 1e-6
            model.clip_range = lambda _: 0.05
            if args.post_warmstart_timesteps is None:
                post_warmstart = 0 if variant.get("train_case_pool") is None else 512
            else:
                post_warmstart = max(int(args.post_warmstart_timesteps), 0)
            learn_timesteps = min(int(total_timesteps), post_warmstart) if post_warmstart > 0 else 0

    eval_env = ASVRewardSearchEnv(
        render_mode=None,
        obs_mode=variant["obs_mode"],
        reward_mode=variant["reward_mode"],
    )
    eval_env.reset(seed=args.seed + 10_000 + phase_idx)

    best_prefix = run_dir / f"best_phase_{phase_idx}"
    benchmark_cb = FixedBenchmarkCallback(
        eval_env=eval_env,
        cases=args.benchmark_cases,
        eval_freq=args.eval_freq,
        max_steps=args.eval_max_steps,
        out_json=str(run_dir / f"benchmark_history_phase_{phase_idx}.json"),
        out_csv=str(run_dir / f"benchmark_summary_phase_{phase_idx}.csv"),
        best_model_path=str(best_prefix),
        best_json=str(run_dir / f"best_benchmark_phase_{phase_idx}.json"),
        early_stop_patience=args.early_stop_patience,
        verbose=1,
    )

    if learn_timesteps > 0:
        model.learn(
            total_timesteps=learn_timesteps,
            tb_log_name=f"reward_search_{variant['name']}_phase_{phase_idx}",
            callback=benchmark_cb,
            progress_bar=True,
        )
    else:
        benchmark_cb.model = model
        benchmark_cb.num_timesteps = int(args.eval_freq)
        initial_result = evaluate_benchmark(model, eval_env, args.benchmark_cases, args.eval_max_steps)
        initial_summary = {"timesteps": int(benchmark_cb.num_timesteps), **initial_result["summary"]}
        benchmark_cb.history.append({"timesteps": int(benchmark_cb.num_timesteps), **initial_result})
        with open(run_dir / f"benchmark_history_phase_{phase_idx}.json", "w", encoding="utf-8") as f:
            json.dump(benchmark_cb.history, f, indent=2)
        benchmark_cb._append_csv(initial_summary)
        benchmark_cb.best_metric = benchmark_cb._metric_tuple(initial_summary)
        benchmark_cb.best_summary = initial_summary
        benchmark_cb._save_best_model(initial_result, initial_summary)

    final_model_path = run_dir / f"{variant['name']}_phase_{phase_idx}.zip"
    model.save(str(final_model_path))

    final_result = evaluate_benchmark(model, eval_env, args.benchmark_cases, args.eval_max_steps)
    random_eval = evaluate_randomized(
        model,
        variant,
        episodes=int(args.random_eval_episodes),
        max_steps=args.eval_max_steps,
        seed=args.random_eval_seed + phase_idx * 1_000,
    )
    best_record = pick_best_record(benchmark_cb.history, final_result["summary"])
    best_model_path = f"{best_prefix}.zip"
    if not Path(best_model_path).exists():
        best_model_path = str(final_model_path)

    result = {
        "variant": variant["name"],
        "obs_mode": variant["obs_mode"],
        "reward_mode": variant["reward_mode"],
        "phase": phase_idx,
        "requested_timesteps": int(total_timesteps),
        "trained_timesteps": int(learn_timesteps),
        "resume_model_path": resume_model_path,
        "best_timesteps": int(best_record["timesteps"]),
        "best_summary": best_record["summary"],
        "final_summary": final_result["summary"],
        "best_model_path": best_model_path,
        "final_model_path": str(final_model_path),
        "run_dir": str(run_dir),
        "warm_start": warm_start_info,
        "random_eval": random_eval,
    }

    with open(run_dir / f"phase_{phase_idx}_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    eval_env.close()
    vec_env.close()
    return result


def main() -> None:
    multiprocessing.freeze_support()
    args = parse_args()

    selected = set(args.variants) if args.variants else None
    variants = [
        {
            **v,
            "train_case_pool": None
            if args.train_random
            else (list(args.train_case_pool) if args.train_case_pool else None),
        }
        for v in VARIANTS
        if selected is None or v["name"] in selected
    ]
    if not variants:
        raise ValueError("No variants selected")

    run_root = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    best_result: Dict[str, Any] | None = None

    for variant in variants:
        run_dir = run_root / variant["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SEARCH] Screening {variant['name']} for {args.screen_timesteps} timesteps")
        result = run_phase(args, variant, run_dir, phase_idx=1, total_timesteps=args.screen_timesteps)
        results.append(result)
        if best_result is None or score_key(result["best_summary"]) > score_key(best_result["best_summary"]):
            best_result = result
        print(json.dumps(result["best_summary"], indent=2))
        if float(result["best_summary"]["success_rate"]) >= args.target_success:
            break

    if best_result is None:
        raise RuntimeError("Search produced no results")

    variant = next(v for v in variants if v["name"] == best_result["variant"])
    followup = 0
    while float(best_result["best_summary"]["success_rate"]) < args.target_success and followup < args.max_followups:
        followup += 1
        phase_idx = 1 + followup
        print(
            f"[SEARCH] Continuing {variant['name']} phase={phase_idx} "
            f"for {args.continue_timesteps} timesteps"
        )
        best_result = run_phase(
            args,
            variant,
            Path(best_result["run_dir"]),
            phase_idx=phase_idx,
            total_timesteps=args.continue_timesteps,
            resume_model_path=best_result["best_model_path"],
        )
        results.append(best_result)
        print(json.dumps(best_result["best_summary"], indent=2))

    leaderboard = sorted(results, key=lambda item: score_key(item["best_summary"]), reverse=True)
    payload = {
        "target_success": args.target_success,
        "variants": variants,
        "results": leaderboard,
        "best_variant": leaderboard[0]["variant"],
        "best_summary": leaderboard[0]["best_summary"],
        "met_target": float(leaderboard[0]["best_summary"]["success_rate"]) >= args.target_success,
    }
    with open(run_root / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[SEARCH] Best result")
    print(json.dumps(payload["best_summary"], indent=2))


if __name__ == "__main__":
    main()
