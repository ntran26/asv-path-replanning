from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from stable_baselines3 import PPO

from auto_reward_search import evaluate_randomized
from rl_env_reward_search import ASVRewardSearchEnv
from train_test_asv import DEFAULT_BENCHMARK_CASES, evaluate_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--obs-mode", type=str, default="teacher_compact")
    parser.add_argument("--reward-mode", type=str, default="teacher_guided")
    parser.add_argument("--benchmark-cases", type=int, nargs="*", default=DEFAULT_BENCHMARK_CASES)
    parser.add_argument("--random-episodes", type=int, default=100)
    parser.add_argument("--eval-max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-case", type=int, default=None)
    parser.add_argument("--render-seed", type=int, default=40_000)
    return parser.parse_args()


def rollout_render_episode(
    model: PPO,
    obs_mode: str,
    reward_mode: str,
    max_steps: int,
    case_id: int | None,
    seed: int,
) -> Dict[str, Any]:
    env = ASVRewardSearchEnv(
        render_mode="human",
        obs_mode=obs_mode,
        reward_mode=reward_mode,
    )
    if case_id is not None:
        env.test_case = int(case_id)
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    last_info: Dict[str, Any] = {}

    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    payload = {
        "mode": "render",
        "case_id": None if case_id is None else int(case_id),
        "seed": int(seed),
        "steps": int(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward": float(total_reward),
        "distance_to_goal": float(getattr(env, "distance_to_goal", float("inf"))),
        "collided": bool(last_info.get("collided", False)),
        "reached_goal": bool(last_info.get("reached_goal", False)),
        "final_x": float(getattr(env, "asv_x", 0.0)),
        "final_y": float(getattr(env, "asv_y", 0.0)),
        "final_heading": float(getattr(env, "asv_h", 0.0)),
    }
    env.close()
    return payload


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model_path)

    variant = {
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
    }

    if args.render:
        payload = rollout_render_episode(
            model=model,
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
            max_steps=int(args.eval_max_steps),
            case_id=args.render_case,
            seed=int(args.render_seed),
        )
        print(json.dumps(payload, indent=2))
        return

    bench_env = ASVRewardSearchEnv(
        render_mode=None,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
    )
    benchmark = evaluate_benchmark(model, bench_env, list(args.benchmark_cases), args.eval_max_steps)
    bench_env.close()

    random_eval = evaluate_randomized(
        model,
        variant,
        episodes=int(args.random_episodes),
        max_steps=args.eval_max_steps,
        seed=int(args.seed),
    )

    payload = {
        "model_path": str(Path(args.model_path)),
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
        "benchmark": benchmark["summary"],
        "random_eval": random_eval,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
