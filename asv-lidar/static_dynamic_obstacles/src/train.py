"""Train, watch, or evaluate an SAC/PPO agent on the ASV environment.

    python src/train.py --mode train --algo sac --timesteps 1000000 --num-envs 8 --seed 675973
    python src/train.py --mode train --algo sac --timesteps 200000 --resume --model-path models/sac_model_1M.zip
    python src/train.py --mode test  --algo sac --model-path models/sac_model_1M.zip --test-case 4
    python src/train.py --mode eval  --algo sac --model-path models/sac_model_1M.zip --eval-cases 0 1 2 3

Checkpoints go to models/, TensorBoard to <algo>_log/, the episode monitor to
train_monitor.csv, and periodic evaluation to eval_metrics.* / eval_summary.*.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import rollout
from env import ASVLidarEnv
from rollout import Episode, run_episode

# Hand-authored cases used by --mode eval.
DEFAULT_EVAL_CASES = [0, 1, 2, 3, 4, 6, 7]

# Periodic evaluation during training: obstacle-count groups x episodes each.
EVAL_OBS_COUNTS = [0, 1, 2, 3, 4, 5]
EVAL_EPISODES_PER_OBS_COUNT = 10
EVAL_BASE_SEED = 675973

# Weight given to each obstacle-count group when scoring a checkpoint.
GROUP_WEIGHTS = {0: 1.0, 1: 1.2, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0}

DETAIL_HEADER = [
    "timesteps", "eval_group", "eval_episode", "num_obs", "ep_reward", "ep_len", "success", "term_reason",
    "d_start", "d_end", "progress_total", "progress_per_step",
    "mean_speed", "mean_u", "mean_v", "mean_rpm", "min_rpm", "max_rpm",
    "mean_abs_rudder", "std_rudder", "mean_abs_cte", "max_abs_cte",
    "mean_abs_course_error", "max_abs_course_error", "mean_abs_lookahead_error", "max_abs_lookahead_error",
    "min_lidar_all", "p10_front",
    "min_sector_range", "max_boundary_closeness", "min_border_clearance",
    "max_tracks", "n_targets",
    "acquisition_range", "max_coast_steps", "dropped_detections", "track_uptime",
    "reward_per_step", "collision_steps",
]

SUMMARY_HEADER = [
    "timesteps", "mean_ep_reward", "std_ep_reward", "mean_ep_len",
    "success_rate", "collision_rate", "border_rate", "obstacle_rate", "timeout_rate",
    "mean_progress_per_step", "mean_d_end", "mean_speed",
    "mean_rpm", "min_rpm", "max_rpm",
    "mean_abs_cte", "mean_abs_course_error", "mean_abs_lookahead_error",
    "min_min_lidar_all", "min_p10_front",
    "min_sector_range", "max_boundary_closeness", "min_border_clearance",
    "max_tracks", "target_rate",
    "acquisition_range", "max_coast_steps", "track_uptime",
    "selection_score",
]


def _first_finite(values) -> float:
    """First finite entry, or NaN if the target was never acquired."""
    for v in values:
        if np.isfinite(v):
            return float(v)
    return float("nan")


def _track_uptime(episode: Episode) -> float:
    """Fraction of in-range steps on which the target was actually tracked.

    1.0 means the tracker held the target for every step it was within the
    sensor horizon.  Below 1.0 is detection loss, which Study 2 sweeps.
    """
    visible = rollout.largest(episode.track("steps_target_visible"))
    tracked = rollout.largest(episode.track("steps_target_tracked"))
    return float(tracked / visible) if visible > 0 else 0.0


def episode_metrics(episode: Episode, env: ASVLidarEnv) -> Dict[str, Any]:
    """Collapse one episode's traces into the row written to eval_metrics.csv."""
    track = episode.track
    steps = episode.steps
    return {
        "ep_reward": episode.reward,
        "ep_len": steps,
        "success": episode.success,
        "term_reason": episode.reason,
        "num_obs": len(env.obstacles),
        "d_start": episode.d_start,
        "d_end": episode.d_end,
        "progress_total": episode.progress,
        "progress_per_step": episode.progress / steps if steps else 0.0,
        "mean_speed": rollout.mean(track("speed_mps")),
        "mean_u": rollout.mean(track("u_body")),
        "mean_v": rollout.mean(track("v_body")),
        "mean_rpm": rollout.mean(track("rpm")),
        "min_rpm": rollout.smallest(track("rpm")),
        "max_rpm": rollout.largest(track("rpm")),
        "mean_abs_rudder": rollout.abs_mean(track("rudder_deg")),
        "std_rudder": rollout.std(track("rudder_deg")),
        "mean_abs_cte": rollout.abs_mean(track("cross_track_error")),
        "max_abs_cte": rollout.abs_max(track("cross_track_error")),
        "mean_abs_course_error": rollout.abs_mean(track("course_error")),
        "max_abs_course_error": rollout.abs_max(track("course_error")),
        "mean_abs_lookahead_error": rollout.abs_mean(track("lookahead_course_error")),
        "max_abs_lookahead_error": rollout.abs_max(track("lookahead_course_error")),
        "min_lidar_all": rollout.smallest(track("min_lidar_all")),
        "p10_front": rollout.smallest(track("p10_front")),
        # Reward-term telemetry is 02's to define; perception telemetry is 01's.
        "min_sector_range": rollout.smallest(track("min_sector_range")),
        "max_boundary_closeness": rollout.largest(track("max_boundary_closeness")),
        "min_border_clearance": rollout.smallest(track("true_border_clearance")),
        "max_tracks": rollout.largest(track("n_tracks")),
        "n_targets": rollout.largest(track("n_targets")),
        # Perception metrics (04 §7).  `track_uptime` is the fraction of the
        # steps where the target was within sensor range that it was actually
        # tracked -- the headline detection statistic for N1.
        # NaN until the target is first acquired, so a plain max would poison it.
        "acquisition_range": _first_finite(track("acquisition_range")),
        "max_coast_steps": rollout.largest(track("max_coast_steps")),
        "dropped_detections": rollout.largest(track("dropped_detections")),
        "track_uptime": _track_uptime(episode),
        "reward_per_step": episode.reward / steps if steps else 0.0,
        "collision_steps": episode.collision_steps,
    }


def append_csv_row(path: str, header: Sequence[str], row: Sequence[Any]) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


class EvalMetricsCallback(BaseCallback):
    """Evaluates the policy on a fixed grid of obstacle counts every `eval_freq`
    steps, logs per-episode and summary metrics, and keeps the best checkpoint."""

    def __init__(self, eval_env: ASVLidarEnv, *, eval_freq: int = 50_000, max_steps: int = 2_000,
                 csv_path: str = "eval_metrics.csv", json_path: str = "eval_metrics.json",
                 summary_csv_path: str = "eval_summary.csv", summary_json_path: str = "eval_summary.json",
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.max_steps = int(max_steps)
        self.csv_path = csv_path
        self.json_path = json_path
        self.summary_csv_path = summary_csv_path
        self.summary_json_path = summary_json_path

        self.rows: List[Dict[str, Any]] = []
        self.summary_rows: List[Dict[str, Any]] = []
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        ep_metrics = self._run_eval_grid()
        summary = self._summarise(ep_metrics)

        self.summary_rows.append(summary)
        append_csv_row(self.summary_csv_path, SUMMARY_HEADER, [summary.get(k) for k in SUMMARY_HEADER])
        with open(self.json_path, "w") as f:
            json.dump(self.rows, f, indent=2)
        with open(self.summary_json_path, "w") as f:
            json.dump(self.summary_rows, f, indent=2)

        if summary["success_rate"] > 0.0 and summary["selection_score"] > self.best_score:
            self.best_score = summary["selection_score"]
            self.model.save("best_model.zip")
            self.model.save(f"best_model_{self.num_timesteps}.zip")
            if self.verbose:
                print(f"New BEST model saved: score={self.best_score:.3f}, "
                      f"success={summary['success_rate']:.3f}")

        for key, value in summary.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                self.logger.record(f"eval/{key}", value)
        return True

    def _run_eval_grid(self) -> List[Dict[str, Any]]:
        prev_case = self.eval_env.test_case
        prev_forced = self.eval_env.forced_num_obs
        self.eval_env.test_case = None

        ep_metrics: List[Dict[str, Any]] = []
        for obs_count in EVAL_OBS_COUNTS:
            self.eval_env.forced_num_obs = int(obs_count)
            for ep_i in range(EVAL_EPISODES_PER_OBS_COUNT):
                # Seeds are fixed per (obstacle count, episode) so successive
                # checkpoints are compared on identical layouts.
                seed = EVAL_BASE_SEED + 1000 * obs_count + ep_i
                episode = run_episode(self.model, self.eval_env, deterministic=True,
                                      max_steps=self.max_steps, reset_kwargs={"seed": seed})
                m = episode_metrics(episode, self.eval_env)
                m.update(eval_group=f"obs_{obs_count}", eval_episode=ep_i, requested_num_obs=obs_count)
                ep_metrics.append(m)

                append_csv_row(self.csv_path, DETAIL_HEADER,
                               [self.num_timesteps, f"obs_{obs_count}", ep_i, m["num_obs"]]
                               + [m.get(k) for k in DETAIL_HEADER[4:]])
                self.rows.append({"timesteps": int(self.num_timesteps), **m})

                if self.verbose:
                    print(f"[EVAL @{self.num_timesteps}] obs={obs_count} ep={ep_i} "
                          f"succ={m['success']} reason={m['term_reason']} "
                          f"cte={m['mean_abs_cte']:.2f} sect={m['min_sector_range']:.2f} "
                          f"R={m['ep_reward']:.1f}")

        self.eval_env.test_case = prev_case
        self.eval_env.forced_num_obs = prev_forced
        return ep_metrics

    def _summarise(self, ep_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        def avg(key: str) -> float:
            return rollout.mean([float(m.get(key, 0.0)) for m in ep_metrics])

        def worst(key: str) -> float:
            return rollout.smallest([float(m.get(key, float("inf"))) for m in ep_metrics])

        def best(key: str) -> float:
            return rollout.largest([float(m.get(key, 0.0)) for m in ep_metrics])

        def rate(subset: List[Dict[str, Any]], reason: str) -> float:
            return rollout.mean([1 if m["term_reason"] == reason else 0 for m in subset])

        success_rate = rollout.mean([m["success"] for m in ep_metrics])
        border_rate = rate(ep_metrics, "border")
        obstacle_rate = rate(ep_metrics, "obstacle")
        timeout_rate = rate(ep_metrics, "timeout")
        mean_abs_cte = avg("mean_abs_cte")
        mean_abs_course_error = avg("mean_abs_course_error")

        groups: Dict[str, float] = {}
        weighted_success = 0.0
        for obs_count in EVAL_OBS_COUNTS:
            subset = [m for m in ep_metrics if int(m.get("requested_num_obs", -1)) == obs_count]
            if not subset:
                continue
            group_success = rollout.mean([m["success"] for m in subset])
            weighted_success += GROUP_WEIGHTS[obs_count] * group_success
            groups.update({
                f"obs_{obs_count}_success_rate": group_success,
                f"obs_{obs_count}_obstacle_rate": rate(subset, "obstacle"),
                f"obs_{obs_count}_border_rate": rate(subset, "border"),
                f"obs_{obs_count}_timeout_rate": rate(subset, "timeout"),
                f"obs_{obs_count}_mean_abs_cte": rollout.mean([float(m["mean_abs_cte"]) for m in subset]),
            })
        weighted_success /= sum(GROUP_WEIGHTS.values())

        selection_score = (
            10.0 * weighted_success
            - 4.0 * obstacle_rate
            - 3.0 * border_rate
            - 2.0 * timeout_rate
            - 0.8 * mean_abs_cte
            - 0.03 * mean_abs_course_error
        )

        summary = {
            "timesteps": int(self.num_timesteps),
            "mean_ep_reward": avg("ep_reward"),
            "std_ep_reward": rollout.std([float(m["ep_reward"]) for m in ep_metrics]),
            "mean_ep_len": avg("ep_len"),
            "success_rate": success_rate,
            "collision_rate": border_rate + obstacle_rate,
            "border_rate": border_rate,
            "obstacle_rate": obstacle_rate,
            "timeout_rate": timeout_rate,
            "mean_progress_per_step": avg("progress_per_step"),
            "mean_d_end": avg("d_end"),
            "mean_speed": avg("mean_speed"),
            "mean_rpm": avg("mean_rpm"),
            "min_rpm": worst("min_rpm"),
            "max_rpm": best("max_rpm"),
            "mean_abs_cte": mean_abs_cte,
            "mean_abs_course_error": mean_abs_course_error,
            "mean_abs_lookahead_error": avg("mean_abs_lookahead_error"),
            "min_min_lidar_all": worst("min_lidar_all"),
            "min_p10_front": worst("p10_front"),
            "min_sector_range": worst("min_sector_range"),
            "max_boundary_closeness": best("max_boundary_closeness"),
            "min_border_clearance": worst("min_border_clearance"),
            "max_tracks": best("max_tracks"),
            "target_rate": avg("n_targets"),
            "acquisition_range": avg("acquisition_range"),
            "max_coast_steps": best("max_coast_steps"),
            "track_uptime": avg("track_uptime"),
            "selection_score": float(selection_score),
        }
        summary.update(groups)
        return summary


# ---------------------------------------------------------------------------
# Test-mode action guard
# ---------------------------------------------------------------------------
# Paper 2's `side_path_guard` is deliberately not carried across.  It read
# `front_clearance`, `side_clearance_diff` and `local_target_cte` -- the three
# observation fields dropped in 01 §6 -- and it hard-coded a side-choice repair
# that Paper 3 expects the policy to learn from `lidar` plus `boundary`.
# Reinstating it would be a decision for 02, not an inheritance.


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["train", "test", "eval"], default="test")
    ap.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-freq", type=int, default=50_000)
    ap.add_argument("--eval-max-steps", type=int, default=2_000)
    ap.add_argument("--save-freq", type=int, default=100_000)
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--test-case", type=int, default=None)
    ap.add_argument("--eval-cases", type=int, nargs="+", default=DEFAULT_EVAL_CASES)
    ap.add_argument("--train-map-width", type=float, default=10.0)
    ap.add_argument("--train-map-height", type=float, default=25.0)
    ap.add_argument("--eval-map-width", type=float, default=10.0)
    ap.add_argument("--eval-map-height", type=float, default=25.0)
    ap.add_argument("--train-path-mode", choices=["straight", "curve", "mixed"], default="straight")
    ap.add_argument("--eval-path-mode", choices=["straight", "curve", "mixed"], default="straight")
    ap.add_argument("--resume", action="store_true", help="Resume training from --model-path")
    ap.add_argument("--replay-buffer-path", type=str, default=None, help="Replay buffer file for SAC resume")
    return ap.parse_args()


def make_env(seed: int, rank: int, *, map_width: float, map_height: float, path_mode: str):
    def _init():
        env = ASVLidarEnv(map_width=map_width, map_height=map_height, path_mode=path_mode)
        env.reset(seed=seed + rank)
        return env
    return _init


def load_model(algo: str, path: str, env=None, tensorboard_log=None):
    cls = PPO if algo == "ppo" else SAC
    return cls.load(path, env=env, tensorboard_log=tensorboard_log, device="auto")


def build_model(algo: str, vec_env, tensorboard_log: str):
    if algo == "ppo":
        return PPO(
            "MultiInputPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log,
            learning_rate=1e-4, n_steps=1024, batch_size=256, n_epochs=10,
            gamma=0.999, gae_lambda=0.95, clip_range=0.2, ent_coef=0.03, vf_coef=0.5,
            policy_kwargs=dict(activation_fn=nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64])),
        )
    return SAC(
        "MultiInputPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log,
        learning_rate=5e-5, batch_size=512, gamma=0.99, buffer_size=1_000_000,
        train_freq=1, gradient_steps=1, ent_coef="auto",
    )


def run_training(args, algo: str, model_path: str) -> None:
    tensorboard_log = f"./{algo}_log/"
    env_fns = [
        make_env(args.seed, i, map_width=args.train_map_width,
                 map_height=args.train_map_height, path_mode=args.train_path_mode)
        for i in range(args.num_envs)
    ]
    vec_env = VecMonitor(SubprocVecEnv(env_fns), filename="train_monitor.csv")

    eval_env = ASVLidarEnv(map_width=args.eval_map_width, map_height=args.eval_map_height,
                           path_mode=args.eval_path_mode)
    eval_env.reset(seed=args.seed + 10_000)

    resuming = bool(args.resume and args.model_path and os.path.exists(args.model_path))
    if resuming:
        print(f"Resuming from {args.model_path}")
        model = load_model(algo, args.model_path, env=vec_env, tensorboard_log=tensorboard_log)
        if algo == "sac":
            if args.replay_buffer_path and os.path.exists(args.replay_buffer_path):
                print(f"Loading replay buffer from {args.replay_buffer_path}")
                model.load_replay_buffer(args.replay_buffer_path)
            else:
                print("No replay buffer loaded; SAC will resume with an empty buffer.")
    else:
        model = build_model(algo, vec_env, tensorboard_log)

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(args.save_freq // max(args.num_envs, 1), 1),
            save_path="models", name_prefix=f"{algo}_model",
            save_replay_buffer=(algo == "sac"), save_vecnormalize=False,
        ),
        EvalMetricsCallback(eval_env, eval_freq=args.eval_freq, max_steps=args.eval_max_steps),
    ])

    model.learn(total_timesteps=args.timesteps, tb_log_name=f"asv_{algo}",
                callback=callbacks, progress_bar=True, reset_num_timesteps=not resuming)
    model.save(model_path)
    print(f"Saved model -> {model_path}")
    vec_env.close()
    eval_env.close()


def run_test(args, algo: str, model_path: str) -> None:
    """Watch one episode with rendering, and dump the trajectory to asv_data.json."""
    model = load_model(algo, model_path)
    env = ASVLidarEnv(render_mode="human", map_width=args.eval_map_width, map_height=args.eval_map_height,
                      path_mode=args.eval_path_mode, test_case=args.test_case, record_video=True)

    episode = run_episode(model, env, deterministic=True, max_steps=args.eval_max_steps)
    print(f"Test episode completed. Total reward: {episode.reward:.2f}, reason: {episode.reason}")

    with open("asv_data.json", "w") as f:
        json.dump({
            "heading": env.asv_h,
            "start": [env.start_x, env.start_y],
            "goal": [env.goal_x, env.goal_y],
            "obstacles": env.obstacles,
            "path": env.path.points.tolist(),
            "asv_path": env.asv_path,
            "targets": [{"x": t.x, "y": t.y, "heading_deg": t.heading,
                         "speed": t.speed} for t in env.targets],
        }, f, indent=4)
    env.close()


def run_eval(args, algo: str, model_path: str) -> None:
    """Evaluate on the hand-authored test cases."""
    model = load_model(algo, model_path)
    env = ASVLidarEnv(map_width=args.eval_map_width, map_height=args.eval_map_height,
                      path_mode=args.eval_path_mode)

    rows = []
    for case_id in args.eval_cases:
        env.test_case = int(case_id)
        episode = run_episode(model, env, deterministic=True, max_steps=args.eval_max_steps)
        m = episode_metrics(episode, env)
        m["test_case"] = int(case_id)
        rows.append(m)
        print(f"[EVAL] case#{case_id} succ={m['success']} reason={m['term_reason']} "
              f"cte={m['mean_abs_cte']:.3f} sect={m['min_sector_range']:.2f} R={m['ep_reward']:.1f}")

    with open("eval_only_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    env.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    algo = args.algo.lower()
    model_path = args.model_path or f"{algo}_asv_model.zip"

    if args.mode == "train":
        run_training(args, algo, model_path)
    elif args.mode == "test":
        run_test(args, algo, model_path)
    else:
        run_eval(args, algo, model_path)
