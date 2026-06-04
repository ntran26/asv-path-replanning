from __future__ import annotations

import argparse
import csv
import math
from typing import Dict

import numpy as np
from stable_baselines3 import SAC, PPO

from rl_env import ASVLidarEnv, UPDATE_RATE
from ship_model import MAX_RUD_ANGLE, MAX_RUD_RATE_DPS


RUDDER_SCALE = 100.0
RUDDER_SIGN = -1.0  # same as udp_live_rl.py default for real vessel


def filter_obs_for_model(obs: Dict, model):
    """Keep only the observation keys expected by the loaded old/new model."""
    spaces = model.observation_space.spaces
    out = {}
    for key, sp in spaces.items():
        if key not in obs:
            raise KeyError(f"Model expects key {key!r}, but env observation does not contain it.")
        out[key] = np.asarray(obs[key], dtype=np.float32).reshape(sp.shape)
    return out


def rudder_to_live_cmd(a0: float, sign: float = RUDDER_SIGN, scale: float = RUDDER_SCALE) -> float:
    return float(sign * scale * np.clip(float(a0), -1.0, 1.0))


def ship_model_style_live_limiter(prev_cmd: float, raw_cmd: float, dt: float) -> float:
    max_cmd_rate_per_s = RUDDER_SCALE * MAX_RUD_RATE_DPS / MAX_RUD_ANGLE  # 50 %/s
    cmd_dot = float(np.clip(raw_cmd - prev_cmd, -max_cmd_rate_per_s, +max_cmd_rate_per_s))
    return float(prev_cmd + cmd_dot * dt)


def run_case(model, case_id: int, max_steps: int):
    env = ASVLidarEnv(render_mode=None, test_case=case_id)
    obs, _ = env.reset()

    live_limited_cmd = 0.0
    rows = []

    for k in range(max_steps):
        model_obs = filter_obs_for_model(obs, model)
        action, _ = model.predict(model_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        raw_live_cmd = rudder_to_live_cmd(float(action[0]))
        live_limited_cmd = ship_model_style_live_limiter(
            live_limited_cmd,
            raw_live_cmd,
            UPDATE_RATE,
        )

        # Step the normal simulator with the original raw policy action.
        # This lets ship_model.py apply its own internal rudder actuator.
        obs, reward, terminated, truncated, info = env.step(action)

        # ship_model.py internal actual rudder angle.
        # Convert actual delta [deg] to equivalent live command percent.
        sim_actual_rudder_deg = math.degrees(float(getattr(env.model, "_delta", 0.0)))
        sim_actual_cmd_percent = sim_actual_rudder_deg / MAX_RUD_ANGLE * RUDDER_SCALE

        rows.append({
            "case": case_id,
            "step": k,
            "t": k * UPDATE_RATE,
            "a0_raw": float(action[0]),
            "raw_live_cmd": raw_live_cmd,
            "live_limited_cmd": live_limited_cmd,
            "sim_actual_rudder_deg": sim_actual_rudder_deg,
            "sim_actual_cmd_percent": sim_actual_cmd_percent,
            "error_cmd_percent": live_limited_cmd - sim_actual_cmd_percent,
            "cte": float(getattr(env, "cross_track_error", 0.0)),
            "course_error": float(getattr(env, "course_error", 0.0)),
            "yaw_rate": float(getattr(env, "asv_w", 0.0)),
            "front_clearance": float(info.get("front_clearance", np.nan)) if isinstance(info, dict) else np.nan,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "collided": bool(info.get("collided", False)) if isinstance(info, dict) else False,
            "reached_goal": bool(info.get("reached_goal", False)) if isinstance(info, dict) else False,
        })

        if terminated or truncated:
            break

    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ap.add_argument("--cases", type=int, nargs="+", default=[0, 1, 6, 7])
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--out", default="rudder_limiter_vs_sim.csv")
    args = ap.parse_args()

    model = SAC.load(args.model_path) if args.algo == "sac" else PPO.load(args.model_path)

    all_rows = []
    for case_id in args.cases:
        all_rows.extend(run_case(model, case_id, args.max_steps))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    err = np.array([r["error_cmd_percent"] for r in all_rows], dtype=float)
    rate = np.diff(np.array([r["live_limited_cmd"] for r in all_rows], dtype=float)) / UPDATE_RATE

    print(f"Saved: {args.out}")
    print(f"Mean abs limiter-vs-sim error: {np.mean(np.abs(err)):.3f} command-%")
    print(f"Max abs limiter-vs-sim error:  {np.max(np.abs(err)):.3f} command-%")
    print(f"Max live command rate:          {np.max(np.abs(rate)):.3f} command-%/s")
    print(f"Expected command rate limit:    {RUDDER_SCALE * MAX_RUD_RATE_DPS / MAX_RUD_ANGLE:.3f} command-%/s")


if __name__ == "__main__":
    main()