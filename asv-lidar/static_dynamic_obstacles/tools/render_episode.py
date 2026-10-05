"""Render one episode of a policy to an MP4 with the built-in renderer (2026-10-04).

Runs the environment with the renderer (`src/render.py`) headless -- no window
opens -- and captures the full render window every decision step: the field
view (panels, target, tracker estimate, LiDAR, boundary, path, goal) and the
telemetry panel with all seven blocks shown.  Written as H.264 / yuv420p, so it
plays in browsers and common players.  One frame per 0.5 s decision, so 2 fps
is real time.

    python tools/render_episode.py --case field_dev:DV3-CRP-CV-01 \\
        --model runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip

Cases are keys of the development-set cache (`results/feasibility/dev_sets.pkl`,
written by tools/diagnostics/feasibility/oracle_sets.py), with their seeds.
Writes `results/videos/<case>_<tag>.mp4` and a per-step trace `<case>_<tag>.csv`.

`--update-rate HZ` runs the episode at another decision rate.  `constants.py`
is loaded with only `UPDATE_RATE` replaced, before any other module imports it,
so everything derived from it stays consistent: physics substeps per decision,
the rudder command limit per step, LiDAR and tracker cadence, step-counted
timers and the episode cap (`steps_for`), and the real-time video rate.  The
file itself is not edited.  A policy trained at 2 Hz is then acting in a
different closed loop from the one it learned.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import types
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")          # headless: draw off screen
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers")]


def _constants_at(update_hz: float) -> None:
    """Install `constants` with UPDATE_RATE = 1 / update_hz, ahead of every import."""
    path = ROOT / "src" / "constants.py"
    src, n = re.subn(r"^UPDATE_RATE = [0-9.]+", f"UPDATE_RATE = {1.0 / update_hz!r}",
                     path.read_text(encoding="utf-8"), count=1, flags=re.M)
    assert n == 1, "UPDATE_RATE line not found in constants.py"
    # constants.py asserts the simulator runs at the vessel's deployed 2 Hz; this is an
    # off-formulation experiment, so the copy's guard is set to the experiment's rate.
    src, n = re.subn(r"^DEPLOYED_DECISION_DT = [0-9.]+", f"DEPLOYED_DECISION_DT = {1.0 / update_hz!r}",
                     src, count=1, flags=re.M)
    assert n == 1, "DEPLOYED_DECISION_DT line not found in constants.py"
    mod = types.ModuleType("constants")
    mod.__file__ = str(path)
    exec(compile(src, str(path), "exec"), mod.__dict__)
    sys.modules["constants"] = mod


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--update-rate", type=float, default=None)
_rate = _pre.parse_known_args()[0].update_rate
if _rate is not None:
    _constants_at(_rate)

import imageio.v2 as imageio
import numpy as np
import pygame

import constants as cfg
import curriculum
import train_formulation as tf


class _NoWait:
    """Stands in for the renderer's pygame clock, so recording runs at full speed."""

    def tick(self, *_):
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="field_dev:DV3-CRP-CV-01")
    ap.add_argument("--model", type=Path, default=ROOT / "runs" / "sac_formulation_seed0_bl3" / "kept_best_3M" / "best_model.zip")
    ap.add_argument("--tag", default="sac3M")
    ap.add_argument("--fps", type=float, default=1.0 / cfg.UPDATE_RATE, help="default: real time")
    ap.add_argument("--hold", type=int, default=4, help="extra copies of the last frame")
    ap.add_argument("--update-rate", type=float, default=None, help="decision rate in Hz (default: the formulation's 2 Hz)")
    a = ap.parse_args()
    print(f"decision rate {1.0 / cfg.UPDATE_RATE:g} Hz, physics substeps {max(1, round(cfg.UPDATE_RATE / cfg.PHYSICS_DT))}, "
          f"episode cap {cfg.MAX_EPISODE_STEPS} steps, video {a.fps:g} fps", flush=True)

    curriculum.apply_stage(tf.PROPULSION_STAGE)
    with open(ROOT / "results" / "feasibility" / "dev_sets.pkl", "rb") as fh:
        items = {key: (built, seed) for key, built, seed, _ in pickle.load(fh)}
    built, seed = items[a.case]
    from common import load_model
    from env import ASVLidarEnv
    import render
    model = load_model(a.model if a.model.is_absolute() else ROOT / a.model)
    env = ASVLidarEnv(render_mode="human", emergency_stop=False)
    frames = []

    def grab():
        r = env.renderer
        if r is None:
            return
        r.clock = _NoWait()                                # no real-time wait while recording
        r.blocks = set(render.BLOCKS)                      # the full panel: all seven blocks
        frames.append(np.transpose(pygame.surfarray.array3d(r.surface), (1, 0, 2)).copy())

    obs, _ = env.reset(seed=seed, options={"generated": built})
    if env.renderer is not None:
        env.renderer.overlay = [f"{a.tag}, safety off", f"{a.case.split(':')[-1]}  seed {seed}"]
        env.renderer.blocks = set(render.BLOCKS)
        env.render()
    grab()
    actor = tf.EpisodeActor(model)
    from env import _polygon_gap
    trace = []
    while True:
        act = actor(obs)
        obs, _, term, trunc, info = env.step(act)          # env.step draws the frame
        grab()
        hull = env.hull_polygon()
        trace.append({"t": env.step_count * cfg.UPDATE_RATE, "x": env.asv_x, "y": env.asv_y, "heading": env.asv_h,
                      "speed": float(info["speed_mps"]), "rudder": float(act[0]), "throttle": float(act[1]),
                      "target_range": min([float(np.hypot(t.x - env.asv_x, t.y - env.asv_y)) for t in env.targets] or [np.nan]),
                      "panel_gap": min([float(_polygon_gap(hull, o)[0]) for o in env.obstacles] or [np.nan])})
        if term or trunc:
            break
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    frames += [frames[-1]] * a.hold
    out = ROOT / "results" / "videos" / f"{a.case.split(':')[-1]}_{a.tag}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=a.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=8,
                     ffmpeg_params=["-crf", "18"])
    import pandas as pd
    pd.DataFrame(trace).to_csv(out.with_suffix(".csv"), index=False)       # the per-step trace beside the video
    env.close()
    print(f"{outcome} in {len(frames) - 1 - a.hold} steps; {len(frames)} frames "
          f"{frames[0].shape[1]}x{frames[0].shape[0]} at {a.fps:g} fps -> {out}")
    return 0 if outcome == "goal" else 2


if __name__ == "__main__":
    sys.exit(main())
