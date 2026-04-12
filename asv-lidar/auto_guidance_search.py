import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from rl_env import ASVLidarEnv
from train_test_asv import (
    DEFAULT_BENCHMARK_CASES,
    DEFAULT_CURRICULUM_CASES,
    DEFAULT_EVAL_MAX_STEPS,
    CurriculumCallback,
    FixedBenchmarkCallback,
    build_model,
    evaluate_benchmark,
)


VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "sector_baseline",
        "guide_mode": "sector",
        "guide_cfg": {},
    },
    {
        "name": "beam_search",
        "guide_mode": "beam",
        "guide_cfg": {},
    },
    {
        "name": "beam_aggressive",
        "guide_mode": "beam",
        "guide_cfg": {
            "beam_w_clear_threat": 1.05,
            "beam_w_goal_base": 0.55,
            "beam_w_goal_threat": -0.10,
            "beam_w_turn": 0.05,
            "beam_w_smooth": 0.05,
            "beam_w_asym": 0.28,
            "beam_max_search_deg": 90.0,
        },
    },
    {
        "name": "gap_search",
        "guide_mode": "gap",
        "guide_cfg": {
            "beam_max_search_deg": 90.0,
            "gap_open_clearance": 2.5,
            "gap_min_beams": 3,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=40_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-max-steps", type=int, default=DEFAULT_EVAL_MAX_STEPS)
    parser.add_argument("--output-dir", type=str, default="auto_runs")
    parser.add_argument("--variants", type=str, nargs="*", default=None)
    return parser.parse_args()


def make_env(seed: int, rank: int, case_pool: List[int], env_kwargs: Dict[str, Any]):
    def _init():
        env = ASVLidarEnv(render_mode=None, **env_kwargs)
        env.set_train_case_pool(case_pool)
        env.reset(seed=seed + rank)
        return env
    return _init


def pick_best_summary(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    def score_key(entry: Dict[str, Any]):
        summary = entry["summary"]
        return (
            summary["success_rate"],
            summary["goal_rate"],
            -summary["obstacle_rate"],
            -summary["border_rate"],
            summary["mean_reward"],
        )

    return max(history, key=score_key)


def run_variant(args: argparse.Namespace, variant: Dict[str, Any], run_root: Path) -> Dict[str, Any]:
    variant_name = variant["name"]
    run_dir = run_root / variant_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "guide_mode": variant["guide_mode"],
        "guide_cfg": deepcopy(variant.get("guide_cfg", {})),
    }

    stage_ends = [
        max(1, int(args.timesteps * 0.25)),
        max(1, int(args.timesteps * 0.60)),
        int(args.timesteps),
    ]
    initial_pool = DEFAULT_CURRICULUM_CASES[0]

    env_fns = [make_env(args.seed, i, initial_pool, env_kwargs) for i in range(args.num_envs)]
    vec_env = VecMonitor(DummyVecEnv(env_fns))
    model = build_model(args.algo, vec_env, args.num_envs)

    eval_env = ASVLidarEnv(render_mode=None, **env_kwargs)
    eval_env.reset(seed=args.seed + 10_000)

    benchmark_cb = FixedBenchmarkCallback(
        eval_env=eval_env,
        cases=DEFAULT_BENCHMARK_CASES,
        eval_freq=args.eval_freq,
        max_steps=args.eval_max_steps,
        out_json=str(run_dir / "benchmark_history.json"),
        out_csv=str(run_dir / "benchmark_summary.csv"),
        verbose=0,
    )
    curriculum_cb = CurriculumCallback(
        stage_end_steps=stage_ends,
        stage_cases=DEFAULT_CURRICULUM_CASES,
        verbose=0,
    )

    model.learn(
        total_timesteps=int(args.timesteps),
        tb_log_name=f"auto_{variant_name}",
        callback=CallbackList([curriculum_cb, benchmark_cb]),
        progress_bar=False,
    )

    model_path = run_dir / f"{args.algo}_{variant_name}.zip"
    model.save(str(model_path))

    final_result = evaluate_benchmark(model, eval_env, DEFAULT_BENCHMARK_CASES, args.eval_max_steps)
    if benchmark_cb.history:
        best_record = pick_best_summary(benchmark_cb.history)
        best_summary = best_record["summary"]
        best_timesteps = int(best_record["timesteps"])
    else:
        best_summary = final_result["summary"]
        best_timesteps = int(args.timesteps)

    result = {
        "variant": variant_name,
        "guide_mode": variant["guide_mode"],
        "guide_cfg": env_kwargs["guide_cfg"],
        "timesteps": int(args.timesteps),
        "best_timesteps": best_timesteps,
        "best_summary": best_summary,
        "final_summary": final_result["summary"],
        "model_path": str(model_path),
        "run_dir": str(run_dir),
    }

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    vec_env.close()
    eval_env.close()
    return result


def main() -> None:
    args = parse_args()
    run_root = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.variants) if args.variants else None
    variants = [v for v in VARIANTS if selected is None or v["name"] in selected]
    results = []

    for variant in variants:
        print(f"[AUTO] Running variant={variant['name']} timesteps={args.timesteps}")
        result = run_variant(args, variant, run_root)
        results.append(result)
        best = result["best_summary"]
        print(
            f"[AUTO] Done {variant['name']}: "
            f"best_success={best['success_rate']:.3f} "
            f"goal={best['goal_rate']:.3f} "
            f"obs={best['obstacle_rate']:.3f} "
            f"border={best['border_rate']:.3f} "
            f"@ {result['best_timesteps']}"
        )

    def score_key(item: Dict[str, Any]):
        s = item["best_summary"]
        return (
            s["success_rate"],
            s["goal_rate"],
            -s["obstacle_rate"],
            -s["border_rate"],
            s["mean_reward"],
        )

    best = max(results, key=score_key)
    with open(run_root / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_root": str(run_root),
                "results": results,
                "best_variant": best["variant"],
            },
            f,
            indent=2,
        )

    print(f"[AUTO] Best variant: {best['variant']} @ {best['best_timesteps']}")
    print(json.dumps(best["best_summary"], indent=2))


if __name__ == "__main__":
    main()
