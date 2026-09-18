"""Explicit scene strata, mastery gates, and replayable development cases.

These are training/development cases, not a frozen scientific test set. A
failed scripted crossing escape is excluded from the normal learning pool;
this is an easier distribution, not a proof that every retained case is safe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import gymnasium as gym
import numpy as np

import constants as cfg
import corridor
import scenario as scn


STRATA = ("empty", "static", "dynamic", "combined")
FINAL_WEIGHTS = {"empty": 0.20, "static": 0.25, "dynamic": 0.40, "combined": 0.15}
LEVEL_WEIGHTS = {
    1: {"empty": 1.0},
    2: {"empty": 0.40, "static": 0.60},
    3: {"empty": 0.20, "static": 0.25, "dynamic": 0.55},
    4: FINAL_WEIGHTS,
    5: FINAL_WEIGHTS,
}
DYNAMIC_CLASSES = ("head_on", "crossing", "overtaking", "being_overtaken", "null")
# Development acceptance margins in addition to padded-hull nonintersection.
MIN_EVAL_HULL_CLEARANCE_M = 0.10
MIN_EVAL_BOUNDARY_CLEARANCE_M = 0.05


def draw_stratum(rng, level: int) -> str:
    weights = LEVEL_WEIGHTS[int(level)]
    return str(rng.choice(tuple(weights), p=tuple(weights.values())))


def classes_at(level: int):
    return ("head_on", "null") if level <= 3 else DYNAMIC_CLASSES


def sample_scene(seed: int, stratum: str, level: int, *, encounter_class=None,
                 namespace="training"):
    """Keep the requested stratum/class fixed across rejected geometry draws."""
    if stratum not in STRATA or level not in LEVEL_WEIGHTS:
        raise ValueError("Unknown scene stratum or curriculum level")
    rng = np.random.default_rng(seed)
    cls = encounter_class or ("no_target" if stratum in ("empty", "static")
                              else str(rng.choice(classes_at(level))))
    if (cls == "no_target") != (stratum in ("empty", "static")):
        raise ValueError("Encounter class and scene stratum disagree")
    generator = scn.ScenarioGenerator(stage=level, seed_namespace=namespace)
    for attempt in range(100):
        geometry_seed = scn.seed_for(namespace, seed + attempt * 997)
        built = generator.sample(geometry_seed, encounter_class=cls,
                                 case_id=f"codex-{namespace}-{stratum}-{seed}")
        if built is None:
            continue
        if getattr(built, "crossing_escapable", None) is not None and not bool(built.crossing_escapable):
            continue
        if bool(getattr(built, "dcpa_below_floor", False)):
            continue
        built.n_obstacles = (int(rng.integers(1, 2 if level <= 2 else 4))
                             if stratum in ("static", "combined") else 0)
        built.flags.update(scene_stratum=stratum, codex_curriculum_level=level,
                           scripted_escape_failures_excluded=True)
        return built
    raise RuntimeError(f"Cannot construct {stratum}/{cls} after 100 geometry draws")


class SceneTrainingEnv(gym.Wrapper):
    """Reset wrapper with independent seeded scene draws and retained skills."""

    def __init__(self, env, *, seed=0, level=1, recovery_probability=0.35):
        super().__init__(env)
        if level not in LEVEL_WEIGHTS or not 0 <= recovery_probability <= 1:
            raise ValueError("Invalid curriculum level or recovery probability")
        self.scene_rng = np.random.default_rng(seed)
        self.level = int(level)
        self.recovery_probability = float(recovery_probability)
        self.stratum = "empty"
        self.last_reset_options = {}

    def set_curriculum_level(self, level):
        if level not in LEVEL_WEIGHTS:
            raise ValueError("Curriculum level must be in 1..5")
        self.level = int(level)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.scene_rng = np.random.default_rng(seed)
        options = dict(options or {})
        if "generated" in options or "scenario" in options:
            return self.env.reset(seed=seed, options=options)
        stratum = options.pop("scene_stratum", None) or draw_stratum(self.scene_rng, self.level)
        cls = options.pop("encounter_class", None)
        # A failed placement cannot replace a difficult target class with an easy one.
        if cls is None and stratum in ("dynamic", "combined"):
            cls = str(self.scene_rng.choice(classes_at(self.level)))
        for attempt in range(30):
            episode_seed = scn.seed_for("training", int(self.scene_rng.integers(0, 2**31)))
            built = sample_scene(episode_seed, stratum, self.level, encounter_class=cls)
            reset_options = dict(options, generated=built)
            if stratum in ("empty", "static") and self.scene_rng.random() < self.recovery_probability:
                reset_options.setdefault("recovery_fraction", float(self.scene_rng.uniform(-0.10, 0.10)))
                reset_options.setdefault("recovery_heading_deg", float(self.scene_rng.uniform(-10.0, 10.0)))
            try:
                obs, info = self.env.reset(seed=episode_seed, options=reset_options)
            except ValueError as exc:
                if "recovery" in str(exc).lower():
                    continue
                raise
            wants_static = stratum in ("static", "combined")
            if bool(self.env.unwrapped.obstacles) != wants_static:
                continue
            self.stratum = stratum
            self.last_reset_options = reset_options
            return obs, dict(info, scene_stratum=stratum, curriculum_level=self.level,
                             episode_seed=episode_seed, scene_placement_retries=attempt)
        raise RuntimeError(f"No realized {stratum} scene after 30 placement attempts")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(scene_stratum=self.stratum, curriculum_level=self.level)
        return obs, reward, terminated, truncated, info


def summarize(rows):
    """Metrics are per episode; COLREG violation fraction is a reward diagnostic."""
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    result = {
        "episodes": len(rows),
        "goal_rate": float(np.mean([r["outcome"] == "goal" for r in rows])),
        "collision_rate": float(np.mean([r["outcome"].startswith("collision") for r in rows])),
        "timeout_rate": float(np.mean([r["outcome"] == "timeout" for r in rows])),
        "mean_abs_cte": float(np.mean([r["mean_abs_cte"] for r in rows])),
        "mean_speed_error": float(np.mean([r["mean_speed_error"] for r in rows])),
        "violation_fraction": float(np.mean([r["violation_fraction"] for r in rows])),
        "wrong_turn_fraction": float(np.mean([r.get("wrong_turn_fraction", 0.0) for r in rows])),
        "low_clearance_rate": float(np.mean([
            (r.get("min_hull_clearance") is not None and
             r["min_hull_clearance"] < MIN_EVAL_HULL_CLEARANCE_M) or
            (r.get("min_boundary_clearance") is not None and
             r["min_boundary_clearance"] < MIN_EVAL_BOUNDARY_CLEARANCE_M)
            for r in rows])),
        "intervention_rate": float(np.mean([r["estops"] > 0 for r in rows])),
    }
    for kind in ("boundary", "obstacle", "target"):
        result[f"collision_{kind}"] = float(np.mean([r["outcome"] == f"collision:{kind}" for r in rows]))
    return result


def checkpoint_rank(rows):
    """Safety first, then worst-stratum completion, compliance proxy and tracking."""
    overall = summarize(rows)
    per_scene = [summarize([r for r in rows if r["stratum"] == s])
                 for s in STRATA if any(r["stratum"] == s for r in rows)]
    return (-overall["collision_rate"], -overall["low_clearance_rate"], min(s["goal_rate"] for s in per_scene),
            -overall["wrong_turn_fraction"], -overall["violation_fraction"], overall["goal_rate"],
            -overall["mean_abs_cte"], -overall["mean_speed_error"])


def mastery_passes(rows, level, *, min_episodes=4):
    """Conservative development gate; no elapsed-step override.

    All retained strata need >=90% goal completion and zero collisions. Empty
    cases additionally need <=0.25m CTE and <=25% reference-speed error. Target
    cases require <=5% flagged COLREG frames. This proxy is not certification.
    """
    for stratum in LEVEL_WEIGHTS[level]:
        selected = [r for r in rows if r["stratum"] == stratum]
        if len(selected) < min_episodes:
            return False
        metrics = summarize(selected)
        if metrics["goal_rate"] < 0.9 or metrics["collision_rate"] > 0.0 or metrics["low_clearance_rate"] > 0.0:
            return False
        if stratum == "empty" and (metrics["mean_abs_cte"] > 0.25 or
                                   metrics["mean_speed_error"] > 0.25 * cfg.U_NOM):
            return False
        if stratum in ("dynamic", "combined") and (metrics["violation_fraction"] > 0.05 or
                                                    metrics["wrong_turn_fraction"] > 0.05):
            return False
    # Cover each encounter class; averages cannot hide a failed crossing class.
    if level >= 3:
        for cls in classes_at(level):
            selected = [r for r in rows if r["class"] == cls]
            if not selected or summarize(selected)["goal_rate"] < 0.9:
                return False
    return True


def case_record(built, env, episode_seed, reset_options=None):
    """Freeze actual channel and obstacle geometry, not only generator seeds."""
    channel = built.channel
    record = built.to_record()
    record["obstacles"] = [[[float(x), float(y)] for x, y in poly] for poly in env.obstacles]
    record["n_obstacles"] = len(env.obstacles)
    return {
        "scenario": record,
        "channel": {"centre": channel.centre.tolist(), "width": channel.width.tolist(),
                    "s": channel.s.tolist(), "bend_deg": channel.bend_deg,
                    "bend_requested_deg": channel.bend_requested_deg,
                    "offset_frac": channel.offset_frac, "basin": list(channel.basin)},
        "episode_seed": int(episode_seed),
        "reset_options": {k: v for k, v in (reset_options or {}).items()
                          if k in ("recovery_fraction", "recovery_heading_deg", "initial_speed")},
        "labels": {k: getattr(built, k, None) for k in
                   ("crossing_escapable", "dcpa_below_floor", "dcpa_floor_m")},
        "stratum": built.flags["scene_stratum"],
        "initial_pose": [float(env.asv_x), float(env.asv_y), float(env.asv_h)],
    }


def restore_case(record):
    built = scn.Scenario(**{k: v for k, v in record["scenario"].items()
                            if k in {f.name for f in fields(scn.Scenario)}})
    channel = dict(record["channel"])
    for key in ("centre", "width", "s"):
        channel[key] = np.asarray(channel[key], dtype=float)
    built.channel = corridor.Corridor(**channel)
    for key, value in record["labels"].items():
        setattr(built, key, value)
    options = dict(record["reset_options"], generated=built, obstacles=built.obstacles)
    return built, options


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def source_provenance():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def build_development_cases(env, per_stratum=5, level=5):
    """Balanced strata, cycling every dynamic class, with realized scene checks."""
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    records = []
    classes = classes_at(level)
    for s_index, stratum in enumerate(LEVEL_WEIGHTS[level]):
        count = max(per_stratum, len(classes)) if stratum in ("dynamic", "combined") else per_stratum
        for index in range(count):
            cls = classes[index % len(classes)] if stratum in ("dynamic", "combined") else "no_target"
            for attempt in range(100):
                seed = scn.seed_for("development", level * 1777 + s_index * 1000 + index * 37 + attempt)
                built = sample_scene(seed, stratum, level, encounter_class=cls, namespace="development")
                options = {"generated": built}
                if stratum == "empty" and index % 2:
                    options.update(recovery_fraction=0.08 * (-1 if index % 4 == 1 else 1),
                                   recovery_heading_deg=8.0 * (-1 if index % 4 == 1 else 1))
                env.reset(seed=seed, options=options)
                if bool(env.obstacles) != (stratum in ("static", "combined")):
                    continue
                records.append(case_record(built, env, seed, options))
                break
            else:
                raise RuntimeError(f"Unable to freeze realized {stratum}/{cls} case")
    return records


def write_manifest(path, cases, schema):
    value = {"manifest_version": 1, "purpose": "development; used for model selection",
             "observation_schema": schema, "source_sha256": source_provenance(),
             "cases_sha256": canonical_hash(cases), "cases": cases}
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    return value
