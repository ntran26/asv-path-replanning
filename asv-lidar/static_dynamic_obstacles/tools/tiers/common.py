"""Shared machinery for the tiered formulation tests (PROJECT_STATE F60).

    Tier 0  scripted policies     minutes   does the reward point the right way?
    Tier 1  replay a saved model  ~15 min   what does a trained policy do under today's code?
    Tier 2  short fine-tune       ~1 h      does the policy move the way the change intends?
    Tier 3  full run from scratch ~6 h      only once Tiers 0-2 read clean

Every tier replays **fixed** scenario sets from the development namespace, so a
before/after comparison changes the code and nothing else.  Episodes run in a
process pool, one environment per process.
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import scenario as scn  # noqa: E402
import train_formulation as tf  # noqa: E402

RESULTS = ROOT / "results" / "tiers"
COLREGS_PARTS = ("port", "bow", "side", "hold", "r8")
HEAD_ON_WIDTHS = (5.0, 6.0, 7.0, 8.0, 10.0)


# ---------------------------------------------------------------------------
# Fixed scenario sets
# ---------------------------------------------------------------------------
def development_set(per_class: int, classes=tf.EVAL_CLASSES) -> List:
    """The formulation run's evaluation set, optionally restricted by class."""
    return [b for b in tf.development_set(per_class) if b.encounter_class in classes]


def head_on_width_set(per_width: int = 20) -> List:
    """The C14 set: head-on at 5, 6, 7, 8 and 10 m, 20 each (F54)."""
    gen = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    out = []
    for w in HEAD_ON_WIDTHS:
        j = found = 0
        while found < per_width and j < 20 * per_width:
            b = gen.sample(scn.seed_for("development", 300_000 + int(w * 10) * 1000 + j),
                           encounter_class="head_on", width=w)
            j += 1
            if b is not None:
                out.append(b)
                found += 1
    return out


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def follower_action(env) -> np.ndarray:
    """LOS path follower at cruise; ignores targets."""
    rudder = float(np.clip(env.lookahead_course_error / 25.0 + 0.4 * env.cross_track_error, -1.0, 1.0))
    return np.array([rudder, 0.0], dtype=np.float32)


def colregs_action(env, sense_sign: int, alteration_deg: float = 30.0) -> np.ndarray:
    """Follower, plus a committed alteration while a give-way encounter is engaged.

    `sense_sign=+1` turns the way the reward says is compliant; `-1` the
    opposite way.  The alteration is a heading target relative to the heading at
    engagement, so it is one large alteration (Rule 8(b)), held until the
    encounter clears, then the follower returns to the path.
    """
    for ctx in env.encounter_contexts.values():
        if (sense_sign < 0 and ctx.engaged and str(ctx.cls) == "being_overtaken"
                and ctx.tcpa > 0.0):
            # Stand-on vessel: compliance is holding, so the "wrong" policy
            # leaves the role with the same 30 deg alteration, to starboard.
            want = float(ctx.psi_engage) + alteration_deg
            rudder = float(np.clip(_wrap180(want - env.asv_h) / 20.0, -1.0, 1.0))
            return np.array([rudder, 0.0], dtype=np.float32)
        if ctx.engaged and ctx.gives_way and ctx.compliant_turn_sense != 0 and ctx.tcpa > 0.0:
            s = sense_sign * int(ctx.compliant_turn_sense)
            want = float(ctx.psi_engage) + s * alteration_deg
            rudder = float(np.clip(_wrap180(want - env.asv_h) / 20.0, -1.0, 1.0))
            return np.array([rudder, 0.0], dtype=np.float32)
    return follower_action(env)


# The classical comparators (B8) and the CODEX reference controller: onboard
# controllers that read the environment's perception, one instance per episode.
CONTROLLERS = ("los_dwa", "encounter_vo", "reference")


def make_controller(policy: str):
    if policy == "los_dwa":
        from classical.los_dwa import LosDwaController
        return LosDwaController()
    if policy == "encounter_vo":
        from classical.encounter_vo import EncounterVOController
        return EncounterVOController()
    if policy == "reference":
        from reference_controller import ReferenceController
        return ReferenceController()
    return None


# ---------------------------------------------------------------------------
# One episode, with everything any tier reads
# ---------------------------------------------------------------------------
def run_episode(env, built, seed: int, policy: str, model=None, obstacles: Optional[int] = None) -> Dict:
    if obstacles is not None:
        env.forced_num_obs = int(obstacles)
    obs, _ = env.reset(seed=seed, options={"generated": built})
    total, col, speeds, steps, ctes = 0.0, 0.0, [], 0, []
    frames = Counter()
    integrals = Counter()
    first_engaged_cls, port_sense_frames, engaged_frames = None, 0, 0
    stops, stop_steps, r_path_max = [], [], 0.0
    min_range = float("inf")
    last_events = 0
    actor = tf.EpisodeActor(model) if policy == "model" else None
    controller = make_controller(policy)
    while True:
        if policy == "model":
            action = actor(obs)
        elif controller is not None:
            action = controller.action(env, obs)
        elif policy == "follower":
            action = follower_action(env)
        elif policy == "compliant":
            action = colregs_action(env, +1)
        elif policy == "wrong_way":
            action = colregs_action(env, -1)
        else:
            raise ValueError(policy)
        obs, reward, term, trunc, info = env.step(action)
        steps += 1
        total += float(reward)
        col += float(info["reward/weighted/col"])
        speeds.append(float(info["speed_mps"]))
        ctes.append(float(env.cross_track_error))
        r_path_max = max(r_path_max, abs(float(info["r_path_radps"])))
        for part in COLREGS_PARTS:
            value = float(info.get(f"colregs/v_{part}", 0.0))
            frames[part] += int(value > 0.0)
            integrals[part] += value
        for ctx in env.encounter_contexts.values():
            if ctx.engaged and ctx.tcpa > 0.0:
                engaged_frames += 1
                first_engaged_cls = first_engaged_cls or str(ctx.cls)
                port_sense_frames += int(ctx.compliant_turn_sense < 0 and str(ctx.cls) != "overtaking")
        if int(info["estop/events"]) > last_events:
            last_events = int(info["estop/events"])
            stops.append(env.estop.events[-1].reason)
            stop_steps.append(steps)
        for t in env.targets:
            min_range = min(min_range, math.hypot(t.x - env.asv_x, t.y - env.asv_y))
        if term or trunc:
            break
    ct = float(getattr(built, "ct_deg", 0.0))
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    return {
        "policy": policy, "class": built.encounter_class, "width": float(built.nominal_width),
        "dcpa_m": float(getattr(built, "dcpa_m", 0.0)), "ct_deg": ct,
        "crossing_side": ("port" if ct < 180.0 else "starboard") if built.encounter_class == "crossing" else "",
        "dcpa_below_floor": getattr(built, "dcpa_below_floor", None),
        "crossing_escapable": getattr(built, "crossing_escapable", None),
        "outcome": outcome, "collided": bool(info["collided"]),
        "collided_target": info["collision_kind"] == "target",
        "steps": steps, "return": total, "colregs_integral": col,
        **{f"frames_v_{p}": frames[p] for p in COLREGS_PARTS},
        **{f"integral_v_{p}": integrals[p] for p in COLREGS_PARTS},
        "mean_speed": float(np.mean(speeds)), "max_speed": float(np.max(speeds)),
        "min_target_range": min_range,
        "rms_cte": float(np.sqrt(np.mean(np.square(ctes)))),
        "estops": len(stops), "estop_reasons": " | ".join(stops),
        "estop_then_target_collision": bool(stops) and info["collision_kind"] == "target",
        "first_engaged_cls": first_engaged_cls or "never",
        "engaged_frames": engaged_frames,
        "p_port_sense": port_sense_frames / max(engaged_frames, 1),
        "ever_port_sense_non_overtaking": port_sense_frames > 0,
        "r_path_max": r_path_max,
    }


# ---------------------------------------------------------------------------
# Process pool
# ---------------------------------------------------------------------------
_WORKER = {}


def _init_worker(model_path: Optional[str], overrides: Optional[Dict] = None) -> None:
    import torch
    torch.set_num_threads(1)
    for name, value in (overrides or {}).items():
        setattr(cfg, name, value)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    from env import ASVLidarEnv
    _WORKER["env"] = ASVLidarEnv(render_mode=None)
    if overrides and "EMERGENCY_STOP_ENABLED" in overrides:
        # The constructor default is bound at import; apply the switch directly.
        _WORKER["env"].estop_enabled = bool(overrides["EMERGENCY_STOP_ENABLED"])
    if overrides and "LOW_SPEED_START_FRAC" in overrides:
        # F70: force episodes to start slow or at rest (a constructor argument too).
        _WORKER["env"].low_speed_start_frac = float(overrides["LOW_SPEED_START_FRAC"])
    _WORKER["model"] = None
    if model_path:
        _WORKER["model"] = load_model(model_path)


def load_model(model_path):
    """Load a saved policy with the learner its run recorded (any of the five)."""
    import json
    from pathlib import Path as _P
    algo = "ppo"
    config = _P(model_path).parent / "config.json"
    if config.exists():
        algo = json.loads(config.read_text()).get("algo", "ppo")
    return tf.ALGORITHMS[algo].load(str(model_path), device="cpu")


def _job(args) -> Dict:
    built, seed, policy, obstacles, extra = args
    row = run_episode(_WORKER["env"], built, seed, policy, _WORKER["model"], obstacles)
    row.update(extra)
    return row


def run_pool(jobs: List, model_path: Optional[str] = None, processes: int = None,
             overrides: Optional[Dict] = None) -> List[Dict]:
    """jobs: (built, seed, policy, obstacles, extra-columns) tuples.  `overrides`
    sets `constants` attributes in every worker, for A/B switches."""
    processes = processes or max(1, min(10, (os.cpu_count() or 2) - 2))
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    with ProcessPoolExecutor(max_workers=processes, initializer=_init_worker,
                             initargs=(str(model_path) if model_path else None, overrides)) as pool:
        return list(pool.map(_job, jobs, chunksize=1))


def width_bin(width: float) -> str:
    for edge, label in ((5.5, "5"), (6.5, "6"), (7.5, "7"), (9.0, "8")):
        if width <= edge:
            return label
    return "10"
