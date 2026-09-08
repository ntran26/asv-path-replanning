"""Quick start: drive the environment by hand or with random actions.

    python src/play.py                          # manual control, default width
    python src/play.py --mode random            # random actions
    python src/play.py --mode random --no-render --episodes 20
    python src/play.py --corridor-width 3.5 --obstacles 2
    python src/play.py --target head_on --pose-noise 0.05

Two modes:

**manual** -- arrow keys. Left/right work the helm, up/down the throttle. The
vessel starts at cruise, so it makes way immediately and you steer from there.

    LEFT / RIGHT   rudder to port / starboard   (held, like a real helm)
    UP / DOWN      throttle up / down
    SPACE          centre the rudder
    T              throttle back to cruise
    R              reset the episode
    P              pause
    ESC or Q       quit

The helm **holds** where you put it rather than springing back, because holding
a steady rate of turn is the thing you actually want to test.

**random** -- samples the action space, as an RL rollout would. Use it with
`--no-render --episodes N` as a smoke test: it validates every observation
against `observation_space` and reports anything non-finite.

Both modes check the observation contract on every step, so a shape, dtype or
range regression surfaces here rather than 200k steps into a training run.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional

import numpy as np

import constants as cfg
from env import ASVLidarEnv, TargetShip
from observation import split_target  # noqa: F401  (handy at the REPL)

# Helm rates, per control step at 10 Hz.  The ship model rate-limits the
# physical rudder anyway (20 deg/s), so these only shape the command.
RUDDER_RATE = 0.10          # full deflection in ~1.0 s of held key
THROTTLE_RATE = 0.05        # full range in ~2.0 s

KEY_HINTS = [
    "LEFT/RIGHT rudder   UP/DOWN throttle",
    "SPACE centre helm   T cruise   R reset",
    "P pause             ESC quit",
]


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------
def make_target(env: ASVLidarEnv, kind: str) -> List[TargetShip]:
    """Stage one named encounter geometry, and return the target.

    A convenience for eyeballing the classifier, not a scenario generator --
    03 owns that, with spawn DCPA and TCPA as sampled axes (02a §11.1).

    **This may reposition the own ship**, which is why it takes the whole env.
    `being_overtaken` needs clear water astern, and the default start sits 2 m
    from the end of the basin -- so the target would have to spawn outside the
    boundary, where the gate correctly discards it.
    """
    if kind == "none":
        return []

    # Speeds are expressed relative to the own ship's cruise, not as absolute
    # figures: the overtaking pair only classifies if the speed ordering is
    # right, and hardcoded values silently stop working when the thrust map or
    # `U_CRUISE` moves.
    speed = cfg.U_CRUISE
    slower = 0.35 * cfg.U_CRUISE
    # The margin has to clear `BEING_OVERTAKEN_SPEED_MARGIN` (0.15 x cruise)
    # with room to spare, or the class sits exactly on its own threshold.
    # 1.35 x matches the top of `TARGET_SPEED_RANGE`.
    overhauling = 1.35 * cfg.U_CRUISE
    ahead = min(env.path.length - 1.0, cfg.LIDAR_RANGE * 0.6)
    frac = ahead / max(env.path.length, 1e-6)
    point, tangent, normal = env.path.frame_at_frac(frac)
    course = math.degrees(math.atan2(float(tangent[0]), float(tangent[1])))
    px, py = float(point[0]), float(point[1])

    if kind == "head_on":
        return [TargetShip(px, py, (course + 180.0) % 360.0, speed)]

    if kind == "crossing_stbd":
        # Approaching from the starboard bow, crossing right to left.
        off = 5.0
        return [TargetShip(px + off, py, (course + 270.0) % 360.0, speed)]

    if kind == "crossing_port":
        off = 5.0
        return [TargetShip(px - off, py, (course + 90.0) % 360.0, speed)]

    if kind == "overtaking":
        # Slow vessel ahead on our course: we overtake it.
        near, _, _ = env.path.frame_at_frac(min(1.0, frac * 0.45))
        return [TargetShip(float(near[0]), float(near[1]), course, slower)]

    if kind == "being_overtaken":
        # Faster vessel astern on our course: it overtakes us.
        #
        # Move the own ship up the path first.  The target has to sit inside the
        # boundary polygon or the gate drops it, and it has to start far enough
        # back that it closes gradually -- the own ship accelerates from rest,
        # so a target already at speed eats a short gap before the tracker has
        # anything to work with.
        _advance_own_ship(env, 0.5)
        course = _path_course(env)
        gap = 11.0
        back_x = env.asv_x - gap * math.sin(math.radians(course))
        back_y = env.asv_y - gap * math.cos(math.radians(course))
        return [TargetShip(back_x, back_y, course, overhauling)]

    raise ValueError(f"unknown target geometry {kind!r}")


def _path_course(env: ASVLidarEnv) -> float:
    """Course of the reference path at the own ship's current station."""
    tangent = env.path.tangent(env.closest_idx)
    return math.degrees(math.atan2(float(tangent[0]), float(tangent[1])))


def _advance_own_ship(env: ASVLidarEnv, frac: float) -> None:
    """Move the own ship along its reference path, keeping it on track."""
    point, tangent, _ = env.path.frame_at_frac(frac)
    env.asv_x, env.asv_y = float(point[0]), float(point[1])
    env.asv_h = math.degrees(math.atan2(float(tangent[0]), float(tangent[1])))
    env.distance_to_goal = float(np.hypot(env.asv_x - env.goal_x,
                                          env.asv_y - env.goal_y))
    env._update_path_errors(env.asv_h)


TARGET_KINDS = ("sampled", "none", "head_on", "crossing_stbd",
                "crossing_port", "overtaking", "being_overtaken")


def build_env(args) -> ASVLidarEnv:
    env = ASVLidarEnv(
        render_mode=None if args.no_render else "human",
        corridor_width=args.corridor_width,
        pose_noise=args.pose_noise > 0.0,
        detection_dropout_p=args.detection_dropout,
        track_velocity_noise=args.velocity_noise,
        lidar_dropout_p=args.lidar_dropout,
        aft_mask_half_deg=args.aft_mask,
    )
    if args.pose_noise > 0.0:
        env._pose_noise.sigma_xy = args.pose_noise
    if args.obstacles is not None:
        env.forced_num_obs = args.obstacles
    return env


def reset_env(env: ASVLidarEnv, args, seed: Optional[int]):
    obs, info = env.reset(seed=seed)
    if args.target != "sampled":
        env.targets = make_target(env, args.target)
        env._perceive()
        obs = env._get_obs()
    return obs, info


# ---------------------------------------------------------------------------
# Observation contract check
# ---------------------------------------------------------------------------
def check_observation(env: ASVLidarEnv, obs, step: int) -> List[str]:
    """Validate one observation against the frozen contract.

    Cheap enough to run every step, and this is exactly the place a shape or
    range regression should surface.
    """
    problems = []
    space = env.observation_space
    for key, box in space.spaces.items():
        if key not in obs:
            problems.append(f"step {step}: branch {key!r} missing")
            continue
        v = obs[key]
        if v.shape != box.shape:
            problems.append(f"step {step}: {key} shape {v.shape}, expected {box.shape}")
        if v.dtype != box.dtype:
            problems.append(f"step {step}: {key} dtype {v.dtype}, expected {box.dtype}")
        if not np.all(np.isfinite(v)):
            problems.append(f"step {step}: {key} has non-finite values")
        elif np.any(v < box.low) or np.any(v > box.high):
            problems.append(f"step {step}: {key} out of range "
                            f"[{v.min():.3f}, {v.max():.3f}] vs [{box.low.min()}, {box.high.max()}]")
    return problems


def episode_line(n: int, step: int, reward: float, info: dict, env: ASVLidarEnv) -> str:
    reason = ("goal" if info.get("reached_goal") else
              info.get("collision_kind") or ("timeout" if info.get("timeout") else "reset"))
    acq = env.acquisition_range
    perception = (f"tracked {env.steps_target_tracked:3d}/{env.steps_target_visible:3d}"
                  f" | acq {acq:5.1f} m" if acq is not None else "no target acquired")
    return (f"episode {n:3d} | {step:4d} steps | {reason:8s} | R {reward:+8.1f} "
            f"| |cte| {abs(info['cross_track_error']):.2f} "
            f"| {info['encounter_class']:15s} | {perception}")


# ---------------------------------------------------------------------------
# Manual control
# ---------------------------------------------------------------------------
class Helm:
    """Held rudder and throttle, driven by the arrow keys.

    Throttle starts at 0.0, which is cruise: `RPM = CRUISE_RPM + RPM_DELTA *
    throttle`, so the vessel makes way from step one and steering is the only
    thing you have to do to get moving.
    """

    def __init__(self) -> None:
        self.rudder = 0.0
        self.throttle = 0.0

    def centre(self) -> None:
        self.rudder = 0.0

    def cruise(self) -> None:
        self.throttle = 0.0

    def update(self, keys) -> np.ndarray:
        import pygame
        if keys[pygame.K_LEFT]:
            self.rudder -= RUDDER_RATE
        if keys[pygame.K_RIGHT]:
            self.rudder += RUDDER_RATE
        if keys[pygame.K_UP]:
            self.throttle += THROTTLE_RATE
        if keys[pygame.K_DOWN]:
            self.throttle -= THROTTLE_RATE

        self.rudder = float(np.clip(self.rudder, -1.0, 1.0))
        self.throttle = float(np.clip(self.throttle, -1.0, 1.0))
        return np.array([self.rudder, self.throttle], dtype=np.float32)


def overlay_lines(env: ASVLidarEnv, helm: Optional[Helm], mode: str) -> List[str]:
    lines = [f"{mode}  |  corridor {env.corridor_width:.1f} m "
             f"({env.corridor_breadths:.0f} B)  |  targets {len(env.targets)}"]
    if helm is not None:
        rpm = cfg.CRUISE_RPM + cfg.RPM_DELTA * helm.throttle
        lines.append(f"helm {helm.rudder:+.2f}   throttle {helm.throttle:+.2f} "
                     f"({rpm:.1f} rpm)")
        lines += KEY_HINTS
    return lines


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(args) -> int:
    env = build_env(args)
    manual = args.mode == "manual" and not args.no_render

    if args.mode == "manual" and args.no_render:
        print("manual control needs a window; --no-render forces random actions",
              file=sys.stderr)

    print(f"mode={args.mode}  corridor={env.corridor_width:.1f} m "
          f"({env.corridor_breadths:.0f} B)  target={args.target}  "
          f"obs_dim={sum(int(np.prod(s.shape)) for s in env.observation_space.spaces.values())}")
    if manual:
        print("\n".join("  " + line for line in KEY_HINTS))

    pygame = None
    helm = None
    if not args.no_render:
        import pygame as _pygame
        pygame = _pygame
    if manual:
        helm = Helm()

    rng = np.random.default_rng(args.seed)
    episode = 0
    problems: List[str] = []
    paused = False
    running = True

    while running and (args.episodes == 0 or episode < args.episodes):
        seed = args.seed + episode if args.seed is not None else None
        obs, _ = reset_env(env, args, seed)
        episode += 1
        if helm is not None:
            helm.centre()
            helm.cruise()

        step = 0
        total = 0.0
        info: dict = {}
        done = False

        while not done and running:
            if pygame is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                        elif event.key == pygame.K_r:
                            done = True
                        elif event.key == pygame.K_p:
                            paused = not paused
                        elif helm is not None and event.key == pygame.K_SPACE:
                            helm.centre()
                        elif helm is not None and event.key == pygame.K_t:
                            helm.cruise()
                if not running:
                    break
                if paused:
                    if env.renderer is not None:
                        env.renderer.overlay = overlay_lines(env, helm, "PAUSED")
                        env.renderer.draw(env)
                    continue

            if helm is not None:
                action = helm.update(pygame.key.get_pressed())
            else:
                action = env.action_space.sample() if args.seed is None else \
                    rng.uniform(-1.0, 1.0, 2).astype(np.float32)

            if env.renderer is not None:
                env.renderer.overlay = overlay_lines(env, helm, args.mode)

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            total += float(reward)

            found = check_observation(env, obs, step)
            if found:
                problems.extend(found)
                for line in found:
                    print("  CONTRACT:", line, file=sys.stderr)

            done = done or terminated or truncated

        if info:
            print(episode_line(episode, step, total, info, env))

    env.close()

    if problems:
        print(f"\n{len(problems)} observation-contract violations", file=sys.stderr)
        return 1
    print("\nobservation contract held on every step")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("manual", "random"), default="manual")
    p.add_argument("--episodes", type=int, default=0,
                   help="0 = run until you quit (default)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-render", action="store_true",
                   help="headless; forces random actions")

    p.add_argument("--corridor-width", type=float, default=cfg.MAP_WIDTH,
                   help=f"metres; Study 1 sweeps {cfg.CORRIDOR_WIDTHS_M}")
    p.add_argument("--obstacles", type=int, default=None,
                   help="force the static obstacle count")
    p.add_argument("--target", choices=TARGET_KINDS, default="sampled",
                   help="place one target in a named encounter geometry")

    p.add_argument("--pose-noise", type=float, default=0.0,
                   help="localisation 1-sigma, m (Study 2)")
    p.add_argument("--detection-dropout", type=float, default=0.0,
                   help="per-detection miss probability (Study 2)")
    p.add_argument("--velocity-noise", type=float, default=0.0,
                   help="track velocity 1-sigma, m/s (Study 2)")
    p.add_argument("--lidar-dropout", type=float, default=0.0,
                   help="per-beam dropout probability (Study 2)")
    p.add_argument("--aft-mask", type=float, default=0.0,
                   help="aft occlusion half-width, deg (Study 2)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
