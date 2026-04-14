from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model_path)

    variant = {
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
    }

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
