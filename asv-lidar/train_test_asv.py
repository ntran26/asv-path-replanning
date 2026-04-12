import os
import csv
import json
import argparse
import multiprocessing
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from rl_env import ASVLidarEnv, RPM_MAX, RPM_MIN

"""
Train:
  python train_test_asv.py --mode train --algo ppo --timesteps 1000000

Test (render):
  python train_test_asv.py --mode test --algo ppo

Optional:
  --num-envs 8 --eval-freq 50000 --n-eval-episodes 5 --save-freq 500000
"""

DEFAULT_BENCHMARK_CASES = [0, 1, 2, 3, 4, 5]
DEFAULT_EVAL_FREQ = 50_000
DEFAULT_EVAL_MAX_STEPS = 600


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def action_to_rpm(throttle_cmd: float) -> float:
    throttle_cmd = float(np.clip(throttle_cmd, -1.0, 1.0))
    return float(RPM_MIN + (throttle_cmd + 1.0) * 0.5 * (RPM_MAX - RPM_MIN))

def action_to_rudder_deg(rudder_cmd: float) -> float:
    rudder_cmd = float(np.clip(rudder_cmd, -1.0, 1.0))
    return float(rudder_cmd * 40.0)

def lidar_front_stats(env: ASVLidarEnv) -> Dict[str, float]:
    out = {"min_lidar": float("inf"), "p10_front": float("inf")}

    if not hasattr(env, "lidar"):
        return out

    ranges = np.asarray(env.lidar.ranges, dtype=np.float32)
    angles = np.asarray(env.lidar.angles, dtype=np.float32)

    ranges[ranges <= 0.0] = np.inf
    finite = ranges[np.isfinite(ranges)]
    if finite.size:
        out["min_lidar"] = float(np.min(finite))

    front_mask = np.abs(angles) <= 45.0
    if np.any(front_mask):
        front = ranges[front_mask]
    else:
        front = ranges

    front = front[np.isfinite(front)]
    if front.size:
        out["p10_front"] = float(np.percentile(front, 10))

    return out

def infer_term_reason(env: ASVLidarEnv, terminated: bool, truncated: bool, hit_max_steps: bool) -> str:
    if hit_max_steps or truncated:
        return "timeout"

    collided = False
    if hasattr(env, "_check_collision_geom"):
        try:
            collided = bool(env._check_collision_geom())
        except Exception:
            collided = False

    if collided:
        # separate border / obstacle when possible
        try:
            hull = env._hull_polygon_world()
            xs = [p[0] for p in hull]
            ys = [p[1] for p in hull]
            if min(xs) < 0 or max(xs) > env.map_width or min(ys) < 0 or max(ys) > env.map_height:
                return "border"
        except Exception:
            pass
        return "obstacle"

    if getattr(env, "distance_to_goal", float("inf")) <= getattr(__import__("ship_model"), "VESSEL_LENGTH", 1.0):
        return "goal"

    if terminated:
        return "goal"

    return "timeout"

def rollout_episode(model, env: ASVLidarEnv, case_id: int, max_steps: int, deterministic: bool = True) -> Dict[str, Any]:
    env.test_case = case_id
    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    speeds: List[float] = []
    rpms: List[float] = []
    rudders: List[float] = []
    front_p10s: List[float] = []
    min_lidars: List[float] = []
    signed_rudders: List[float] = []
    front_clears: List[float] = []
    oa_active_flags: List[float] = []
    left_p10s: List[float] = []
    center_p10s: List[float] = []
    right_p10s: List[float] = []
    obs_left_clears: List[float] = []
    obs_center_clears: List[float] = []
    obs_right_clears: List[float] = []
    obs_left_clears_instant: List[float] = []
    obs_center_clears_instant: List[float] = []
    obs_right_clears_instant: List[float] = []
    gap_asymmetries: List[float] = []
    lidar_left_clears_m: List[float] = []
    lidar_center_clears_m: List[float] = []
    lidar_right_clears_m: List[float] = []
    lidar_left_blocked_m: List[float] = []
    lidar_center_blocked_m: List[float] = []
    lidar_right_blocked_m: List[float] = []
    lidar_left_open_fracs: List[float] = []
    lidar_center_open_fracs: List[float] = []
    lidar_right_open_fracs: List[float] = []
    r_pfs: List[float] = []
    r_oas: List[float] = []
    pf_contribs: List[float] = []
    oa_contribs: List[float] = []
    threats: List[float] = []
    goal_dist_norms: List[float] = []
    lam_values: List[float] = []
    rudder_states: List[float] = []
    rpm_states: List[float] = []
    first_oa_step = None

    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        signed_rudder = action_to_rudder_deg(float(action[0]))
        signed_rudders.append(signed_rudder)

        front_clears.append(float(info.get("front_clear", float("inf"))))
        oa_now = bool(info.get("oa_active", False))
        oa_active_flags.append(1.0 if oa_now else 0.0)

        if oa_now and first_oa_step is None:
            first_oa_step = steps

        left_p10s.append(float(info.get("left_p10", float("inf"))))
        center_p10s.append(float(info.get("center_p10", float("inf"))))
        right_p10s.append(float(info.get("right_p10", float("inf"))))
        obs_left_clears.append(float(info.get("left_clear", 0.0)))
        obs_center_clears.append(float(info.get("center_clear", 0.0)))
        obs_right_clears.append(float(info.get("right_clear", 0.0)))
        obs_left_clears_instant.append(float(info.get("left_clear_instant", 0.0)))
        obs_center_clears_instant.append(float(info.get("center_clear_instant", 0.0)))
        obs_right_clears_instant.append(float(info.get("right_clear_instant", 0.0)))
        gap_asymmetries.append(float(info.get("gap_asymmetry", 0.0)))
        lidar_left_clears_m.append(float(info.get("lidar_left_clear_m", float("inf"))))
        lidar_center_clears_m.append(float(info.get("lidar_center_clear_m", float("inf"))))
        lidar_right_clears_m.append(float(info.get("lidar_right_clear_m", float("inf"))))
        lidar_left_blocked_m.append(float(info.get("lidar_left_blocked_m", float("inf"))))
        lidar_center_blocked_m.append(float(info.get("lidar_center_blocked_m", float("inf"))))
        lidar_right_blocked_m.append(float(info.get("lidar_right_blocked_m", float("inf"))))
        lidar_left_open_fracs.append(float(info.get("lidar_left_open_fraction", 0.0)))
        lidar_center_open_fracs.append(float(info.get("lidar_center_open_fraction", 0.0)))
        lidar_right_open_fracs.append(float(info.get("lidar_right_open_fraction", 0.0)))

        r_pfs.append(float(info.get("r_pf", 0.0)))
        r_oas.append(float(info.get("r_oa", 0.0)))
        pf_contribs.append(float(info.get("reward_pf_contrib", 0.0)))
        oa_contribs.append(float(info.get("reward_oa_contrib", 0.0)))
        threats.append(float(info.get("threat", 0.0)))
        goal_dist_norms.append(float(info.get("goal_dist_norm", 0.0)))
        lam_values.append(float(info.get("lam", 0.0)))
        rudder_states.append(float(info.get("rudder_state", 0.0)))
        rpm_states.append(float(info.get("rpm_state", 0.0)))

        speeds.append(float(getattr(env, "speed_mps", 0.0)))
        rpms.append(action_to_rpm(float(action[1])))
        rudders.append(action_to_rudder_deg(float(action[0])))

        ls = lidar_front_stats(env)
        min_lidars.append(ls["min_lidar"])
        front_p10s.append(ls["p10_front"])

        if terminated or truncated:
            break

    hit_max_steps = steps >= max_steps and not (terminated or truncated)
    term_reason = infer_term_reason(env, terminated, truncated, hit_max_steps)

    return {
        "case": int(case_id),
        "ep_reward": float(total_reward),
        "ep_len": int(steps),
        "term_reason": term_reason,
        "success": int(term_reason == "goal"),
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "mean_rpm": float(np.mean(rpms)) if rpms else 0.0,
        "mean_abs_rudder": float(np.mean(np.abs(rudders))) if rudders else 0.0,
        "min_lidar": float(np.min(min_lidars)) if min_lidars else float("inf"),
        "p10_front": float(np.min(front_p10s)) if front_p10s else float("inf"),
        "d_end": float(getattr(env, "distance_to_goal", float("inf"))),
        "start": [float(env.start_x), float(env.start_y)],
        "goal": [float(env.goal_x), float(env.goal_y)],
        "mean_signed_rudder": float(np.mean(signed_rudders)) if signed_rudders else 0.0,
        "max_abs_rudder": float(np.max(np.abs(signed_rudders))) if signed_rudders else 0.0,
        "min_front_clear": float(np.min(front_clears)) if front_clears else float("inf"),
        "oa_active_frac": float(np.mean(oa_active_flags)) if oa_active_flags else 0.0,
        "first_oa_step": int(first_oa_step) if first_oa_step is not None else -1,
        "left_p10_min": float(np.min(left_p10s)) if left_p10s else float("inf"),
        "center_p10_min": float(np.min(center_p10s)) if center_p10s else float("inf"),
        "right_p10_min": float(np.min(right_p10s)) if right_p10s else float("inf"),
        "min_obs_left_clear": float(np.min(obs_left_clears)) if obs_left_clears else float("inf"),
        "min_obs_center_clear": float(np.min(obs_center_clears)) if obs_center_clears else float("inf"),
        "min_obs_right_clear": float(np.min(obs_right_clears)) if obs_right_clears else float("inf"),
        "max_obs_left_clear": float(np.max(obs_left_clears)) if obs_left_clears else 0.0,
        "max_obs_center_clear": float(np.max(obs_center_clears)) if obs_center_clears else 0.0,
        "max_obs_right_clear": float(np.max(obs_right_clears)) if obs_right_clears else 0.0,
        "min_obs_left_clear_instant": float(np.min(obs_left_clears_instant)) if obs_left_clears_instant else float("inf"),
        "min_obs_center_clear_instant": float(np.min(obs_center_clears_instant)) if obs_center_clears_instant else float("inf"),
        "min_obs_right_clear_instant": float(np.min(obs_right_clears_instant)) if obs_right_clears_instant else float("inf"),
        "max_obs_left_clear_instant": float(np.max(obs_left_clears_instant)) if obs_left_clears_instant else 0.0,
        "max_obs_center_clear_instant": float(np.max(obs_center_clears_instant)) if obs_center_clears_instant else 0.0,
        "max_obs_right_clear_instant": float(np.max(obs_right_clears_instant)) if obs_right_clears_instant else 0.0,
        "mean_abs_gap_asymmetry": float(np.mean(np.abs(gap_asymmetries))) if gap_asymmetries else 0.0,
        "min_lidar_left_clear_m": float(np.min(lidar_left_clears_m)) if lidar_left_clears_m else float("inf"),
        "min_lidar_center_clear_m": float(np.min(lidar_center_clears_m)) if lidar_center_clears_m else float("inf"),
        "min_lidar_right_clear_m": float(np.min(lidar_right_clears_m)) if lidar_right_clears_m else float("inf"),
        "min_lidar_left_blocked_m": float(np.min(lidar_left_blocked_m)) if lidar_left_blocked_m else float("inf"),
        "min_lidar_center_blocked_m": float(np.min(lidar_center_blocked_m)) if lidar_center_blocked_m else float("inf"),
        "min_lidar_right_blocked_m": float(np.min(lidar_right_blocked_m)) if lidar_right_blocked_m else float("inf"),
        "mean_lidar_left_open_fraction": float(np.mean(lidar_left_open_fracs)) if lidar_left_open_fracs else 0.0,
        "mean_lidar_center_open_fraction": float(np.mean(lidar_center_open_fracs)) if lidar_center_open_fracs else 0.0,
        "mean_lidar_right_open_fraction": float(np.mean(lidar_right_open_fracs)) if lidar_right_open_fracs else 0.0,
        "mean_r_pf": float(np.mean(r_pfs)) if r_pfs else 0.0,
        "mean_r_oa": float(np.mean(r_oas)) if r_oas else 0.0,
        "mean_pf_contrib": float(np.mean(pf_contribs)) if pf_contribs else 0.0,
        "mean_oa_contrib": float(np.mean(oa_contribs)) if oa_contribs else 0.0,
        "mean_threat": float(np.mean(threats)) if threats else 0.0,
        "mean_goal_dist_norm": float(np.mean(goal_dist_norms)) if goal_dist_norms else 0.0,
        "mean_lam": float(np.mean(lam_values)) if lam_values else 0.0,
        "mean_rudder_state": float(np.mean(rudder_states)) if rudder_states else 0.0,
        "mean_rpm_state": float(np.mean(rpm_states)) if rpm_states else 0.0,
        "final_x": float(env.asv_x),
        "final_y": float(env.asv_y),
        "final_heading": float(env.asv_h),
    }

def evaluate_benchmark(model, env: ASVLidarEnv, cases: List[int], max_steps: int) -> Dict[str, Any]:
    rows = [rollout_episode(model, env, case_id=case, max_steps=max_steps) for case in cases]

    term_reasons = [row["term_reason"] for row in rows]
    summary = {
        "n_cases": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "mean_reward": float(np.mean([row["ep_reward"] for row in rows])) if rows else 0.0,
        "mean_ep_len": float(np.mean([row["ep_len"] for row in rows])) if rows else 0.0,
        "mean_speed": float(np.mean([row["mean_speed"] for row in rows])) if rows else 0.0,
        "min_p10_front": float(np.min([row["p10_front"] for row in rows])) if rows else float("inf"),
        "min_lidar": float(np.min([row["min_lidar"] for row in rows])) if rows else float("inf"),
        "goal_rate": float(np.mean([r == "goal" for r in term_reasons])) if rows else 0.0,
        "obstacle_rate": float(np.mean([r == "obstacle" for r in term_reasons])) if rows else 0.0,
        "border_rate": float(np.mean([r == "border" for r in term_reasons])) if rows else 0.0,
        "timeout_rate": float(np.mean([r == "timeout" for r in term_reasons])) if rows else 0.0,
        "min_front_clear": float(np.min([row["min_front_clear"] for row in rows])) if rows else float("inf"),
        "mean_oa_active_frac": float(np.mean([row["oa_active_frac"] for row in rows])) if rows else 0.0,
        "mean_pf_contrib": float(np.mean([row["mean_pf_contrib"] for row in rows])) if rows else 0.0,
        "mean_oa_contrib": float(np.mean([row["mean_oa_contrib"] for row in rows])) if rows else 0.0,
        "mean_threat": float(np.mean([row["mean_threat"] for row in rows])) if rows else 0.0,
        "mean_goal_dist_norm": float(np.mean([row["mean_goal_dist_norm"] for row in rows])) if rows else 0.0,
        "mean_lam": float(np.mean([row["mean_lam"] for row in rows])) if rows else 0.0,
        "mean_rudder_state": float(np.mean([row["mean_rudder_state"] for row in rows])) if rows else 0.0,
        "mean_rpm_state": float(np.mean([row["mean_rpm_state"] for row in rows])) if rows else 0.0,
        "min_obs_left_clear": float(np.min([row["min_obs_left_clear"] for row in rows])) if rows else float("inf"),
        "min_obs_center_clear": float(np.min([row["min_obs_center_clear"] for row in rows])) if rows else float("inf"),
        "min_obs_right_clear": float(np.min([row["min_obs_right_clear"] for row in rows])) if rows else float("inf"),
        "max_obs_left_clear": float(np.max([row["max_obs_left_clear"] for row in rows])) if rows else 0.0,
        "max_obs_center_clear": float(np.max([row["max_obs_center_clear"] for row in rows])) if rows else 0.0,
        "max_obs_right_clear": float(np.max([row["max_obs_right_clear"] for row in rows])) if rows else 0.0,
        "min_obs_left_clear_instant": float(np.min([row["min_obs_left_clear_instant"] for row in rows])) if rows else float("inf"),
        "min_obs_center_clear_instant": float(np.min([row["min_obs_center_clear_instant"] for row in rows])) if rows else float("inf"),
        "min_obs_right_clear_instant": float(np.min([row["min_obs_right_clear_instant"] for row in rows])) if rows else float("inf"),
        "max_obs_left_clear_instant": float(np.max([row["max_obs_left_clear_instant"] for row in rows])) if rows else 0.0,
        "max_obs_center_clear_instant": float(np.max([row["max_obs_center_clear_instant"] for row in rows])) if rows else 0.0,
        "max_obs_right_clear_instant": float(np.max([row["max_obs_right_clear_instant"] for row in rows])) if rows else 0.0,
        "mean_abs_gap_asymmetry": float(np.mean([row["mean_abs_gap_asymmetry"] for row in rows])) if rows else 0.0,
        "min_lidar_left_clear_m": float(np.min([row["min_lidar_left_clear_m"] for row in rows])) if rows else float("inf"),
        "min_lidar_center_clear_m": float(np.min([row["min_lidar_center_clear_m"] for row in rows])) if rows else float("inf"),
        "min_lidar_right_clear_m": float(np.min([row["min_lidar_right_clear_m"] for row in rows])) if rows else float("inf"),
        "min_lidar_left_blocked_m": float(np.min([row["min_lidar_left_blocked_m"] for row in rows])) if rows else float("inf"),
        "min_lidar_center_blocked_m": float(np.min([row["min_lidar_center_blocked_m"] for row in rows])) if rows else float("inf"),
        "min_lidar_right_blocked_m": float(np.min([row["min_lidar_right_blocked_m"] for row in rows])) if rows else float("inf"),
        "mean_lidar_left_open_fraction": float(np.mean([row["mean_lidar_left_open_fraction"] for row in rows])) if rows else 0.0,
        "mean_lidar_center_open_fraction": float(np.mean([row["mean_lidar_center_open_fraction"] for row in rows])) if rows else 0.0,
        "mean_lidar_right_open_fraction": float(np.mean([row["mean_lidar_right_open_fraction"] for row in rows])) if rows else 0.0,
        "min_left_p10": float(np.min([row["left_p10_min"] for row in rows])) if rows else float("inf"),
        "min_center_p10": float(np.min([row["center_p10_min"] for row in rows])) if rows else float("inf"),
        "min_right_p10": float(np.min([row["right_p10_min"] for row in rows])) if rows else float("inf"),
    }
    return {"rows": rows, "summary": summary}

# -----------------------------------------------------------------------------
# Fixed benchmark callback
# -----------------------------------------------------------------------------
class FixedBenchmarkCallback(BaseCallback):
    def __init__(
        self,
        eval_env: ASVLidarEnv,
        cases: List[int],
        eval_freq: int = DEFAULT_EVAL_FREQ,
        max_steps: int = DEFAULT_EVAL_MAX_STEPS,
        out_json: str = "benchmark_history.json",
        out_csv: str = "benchmark_summary.csv",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.cases = list(cases)
        self.eval_freq = int(eval_freq)
        self.max_steps = int(max_steps)
        self.out_json = out_json
        self.out_csv = out_csv
        self.history: List[Dict[str, Any]] = []
        self._csv_initialized = False

    def _init_csv(self) -> None:
        if self._csv_initialized:
            return
        write_header = not os.path.exists(self.out_csv)
        with open(self.out_csv, "a", newline="") as f:
            if write_header:
                import csv
                csv.writer(f).writerow([
                    "timesteps",
                    "n_cases",
                    "success_rate",
                    "mean_reward",
                    "mean_ep_len",
                    "mean_speed",
                    "min_p10_front",
                    "min_lidar",
                    "goal_rate",
                    "obstacle_rate",
                    "border_rate",
                    "timeout_rate",
                ])
        self._csv_initialized = True

    def _append_csv(self, row: Dict[str, Any]) -> None:
        self._init_csv()
        import csv
        with open(self.out_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                row["timesteps"],
                row["n_cases"],
                row["success_rate"],
                row["mean_reward"],
                row["mean_ep_len"],
                row["mean_speed"],
                row["min_p10_front"],
                row["min_lidar"],
                row["goal_rate"],
                row["obstacle_rate"],
                row["border_rate"],
                row["timeout_rate"],
            ])

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        result = evaluate_benchmark(self.model, self.eval_env, self.cases, self.max_steps)
        summary = {"timesteps": int(self.num_timesteps), **result["summary"]}
        self.history.append({"timesteps": int(self.num_timesteps), **result})

        with open(self.out_json, "w") as f:
            json.dump(self.history, f, indent=2)
        self._append_csv(summary)

        self.logger.record("benchmark/success_rate", summary["success_rate"])
        self.logger.record("benchmark/mean_reward", summary["mean_reward"])
        self.logger.record("benchmark/obstacle_rate", summary["obstacle_rate"])
        self.logger.record("benchmark/border_rate", summary["border_rate"])
        self.logger.record("benchmark/timeout_rate", summary["timeout_rate"])
        self.logger.record("benchmark/min_p10_front", summary["min_p10_front"])
        self.logger.record("benchmark/mean_threat", summary["mean_threat"])
        self.logger.record("benchmark/mean_goal_dist_norm", summary["mean_goal_dist_norm"])
        self.logger.record("benchmark/mean_lam", summary["mean_lam"])
        self.logger.record("benchmark/mean_rudder_state", summary["mean_rudder_state"])
        self.logger.record("benchmark/mean_rpm_state", summary["mean_rpm_state"])
        self.logger.record("benchmark/min_obs_left_clear", summary["min_obs_left_clear"])
        self.logger.record("benchmark/min_obs_center_clear", summary["min_obs_center_clear"])
        self.logger.record("benchmark/min_obs_right_clear", summary["min_obs_right_clear"])
        self.logger.record("benchmark/max_obs_left_clear", summary["max_obs_left_clear"])
        self.logger.record("benchmark/max_obs_center_clear", summary["max_obs_center_clear"])
        self.logger.record("benchmark/max_obs_right_clear", summary["max_obs_right_clear"])
        self.logger.record("benchmark/min_obs_left_clear_instant", summary["min_obs_left_clear_instant"])
        self.logger.record("benchmark/min_obs_center_clear_instant", summary["min_obs_center_clear_instant"])
        self.logger.record("benchmark/min_obs_right_clear_instant", summary["min_obs_right_clear_instant"])
        self.logger.record("benchmark/max_obs_left_clear_instant", summary["max_obs_left_clear_instant"])
        self.logger.record("benchmark/max_obs_center_clear_instant", summary["max_obs_center_clear_instant"])
        self.logger.record("benchmark/max_obs_right_clear_instant", summary["max_obs_right_clear_instant"])
        self.logger.record("benchmark/mean_abs_gap_asymmetry", summary["mean_abs_gap_asymmetry"])
        self.logger.record("benchmark/min_lidar_left_clear_m", summary["min_lidar_left_clear_m"])
        self.logger.record("benchmark/min_lidar_center_clear_m", summary["min_lidar_center_clear_m"])
        self.logger.record("benchmark/min_lidar_right_clear_m", summary["min_lidar_right_clear_m"])
        self.logger.record("benchmark/min_lidar_left_blocked_m", summary["min_lidar_left_blocked_m"])
        self.logger.record("benchmark/min_lidar_center_blocked_m", summary["min_lidar_center_blocked_m"])
        self.logger.record("benchmark/min_lidar_right_blocked_m", summary["min_lidar_right_blocked_m"])
        self.logger.record("benchmark/mean_lidar_left_open_fraction", summary["mean_lidar_left_open_fraction"])
        self.logger.record("benchmark/mean_lidar_center_open_fraction", summary["mean_lidar_center_open_fraction"])
        self.logger.record("benchmark/mean_lidar_right_open_fraction", summary["mean_lidar_right_open_fraction"])

        if self.verbose:
            print(
                f"[BENCHMARK @ {self.num_timesteps}] "
                f"success={summary['success_rate']:.2f} "
                f"reward={summary['mean_reward']:.2f} "
                f"obs={summary['obstacle_rate']:.2f} "
                f"border={summary['border_rate']:.2f} "
                f"timeout={summary['timeout_rate']:.2f} "
                f"min_p10_front={summary['min_p10_front']:.2f}"
            )

        return True

# -----------------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "eval"], default="test")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--test-case", type=int, default=None)

    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--eval-max-steps", type=int, default=DEFAULT_EVAL_MAX_STEPS)
    parser.add_argument("--save-freq", type=int, default=500_000)
    parser.add_argument("--benchmark-cases", type=int, nargs="*", default=DEFAULT_BENCHMARK_CASES)

    return parser.parse_args()

def make_train_env(seed: int, rank: int):
    def _init():
        env = ASVLidarEnv(render_mode=None)
        env.reset(seed=seed + rank)
        return env
    return _init

def build_model(algo: str, env, num_envs: int):
    algo = algo.lower()
    learning_rate = 3e-4
    batch_size = 256
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.0
    vf_coef = 0.5
    n_epochs = 10
    n_steps = 2048

    if algo == "ppo":
        return PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_log/",
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
        )

    if algo == "sac":
        return SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./sac_log/",
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            buffer_size=1_000_000,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )

    raise ValueError(f"Unsupported algo: {algo}")

def load_model(algo: str, model_path: str):
    algo = algo.lower()
    if algo == "ppo":
        return PPO.load(model_path)
    if algo == "sac":
        return SAC.load(model_path)
    raise ValueError(f"Unsupported algo: {algo}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    multiprocessing.freeze_support()
    args = parse_args()

    algo = args.algo.lower()
    model_path = args.model_path or f"{algo}_asv_model.zip"

    if args.mode == "train":
        env_fns = [make_train_env(args.seed, i) for i in range(args.num_envs)]
        vec_env = VecMonitor(SubprocVecEnv(env_fns), filename="train_monitor.csv")

        model = build_model(algo, vec_env, args.num_envs)

        eval_env = ASVLidarEnv(render_mode=None)
        eval_env.reset(seed=args.seed + 10_000)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(int(args.save_freq // max(args.num_envs, 1)), 1),
            save_path="models",
            name_prefix=f"{algo}_model",
            save_replay_buffer=(algo == "sac"),
            save_vecnormalize=False,
        )

        benchmark_cb = FixedBenchmarkCallback(
            eval_env=eval_env,
            cases=args.benchmark_cases,
            eval_freq=args.eval_freq,
            max_steps=args.eval_max_steps,
            out_json="benchmark_history.json",
            out_csv="benchmark_summary.csv",
            verbose=1,
        )

        model.learn(
            total_timesteps=int(args.timesteps),
            tb_log_name=f"asv_{algo}",
            callback=CallbackList([checkpoint_cb, benchmark_cb]),
            progress_bar=True,
        )
        model.save(model_path)
        print(f"Saved model -> {model_path}")

        vec_env.close()
        eval_env.close()
        return

    if args.mode == "test":
        model = load_model(algo, model_path)
        env = ASVLidarEnv(render_mode="human")
        env.test_case = args.test_case

        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
            print(f"action={np.asarray(action).round(3).tolist()} reward={reward:.3f}")

        print(f"Test case {args.test_case} completed. Total reward: {total_reward:.2f}")

        result = {
            "test_case": int(args.test_case),
            "heading": float(env.asv_h),
            "start": [float(env.start_x), float(env.start_y)],
            "goal": [float(env.goal_x), float(env.goal_y)],
            "obstacles": env.obstacles,
            "path": env.path.tolist() if hasattr(env.path, "tolist") else env.path,
            "asv_path": env.asv_path,
        }
        with open("asv_data.json", "w") as f:
            json.dump(result, f, indent=2)

        env.close()
        return

    if args.mode == "eval":
        model = load_model(algo, model_path)
        eval_env = ASVLidarEnv(render_mode=None)
        eval_env.reset(seed=args.seed + 10_000)

        result = evaluate_benchmark(model, eval_env, args.benchmark_cases, args.eval_max_steps)

        print("Benchmark summary:")
        for k, v in result["summary"].items():
            print(f"  {k}: {v}")

        for row in result["rows"]:
            print(
                f"case={row['case']} reward={row['ep_reward']:.2f} len={row['ep_len']} "
                f"term={row['term_reason']} p10_front={row['p10_front']:.2f}"
            )

        with open("benchmark_eval.json", "w") as f:
            json.dump(result, f, indent=2)

        eval_env.close()
        return

if __name__ == "__main__":
    main()
