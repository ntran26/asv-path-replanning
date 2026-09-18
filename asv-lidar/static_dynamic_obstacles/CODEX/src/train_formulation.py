"""Isolated PPO training: explicit scenes, measured curriculum, paired replay.

Run ``python src/train_formulation.py --smoke --num-envs 1 --device cpu`` for a
1024-step training/save/load check. A smoke run is not a learned-policy result.
All outputs stay under CODEX/runs; full runs default to two million steps.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants as cfg
import curriculum
from env import ASVLidarEnv
from features_extractor import ASVFeaturesExtractor
from metrics import polygon_distance
from scene_sampling import (STRATA, FINAL_WEIGHTS, MIN_EVAL_HULL_CLEARANCE_M,
                            MIN_EVAL_BOUNDARY_CLEARANCE_M, SceneTrainingEnv, build_development_cases,
                            canonical_hash, case_record, checkpoint_rank, mastery_passes,
                            restore_case, sample_scene, source_provenance, summarize, write_manifest)

RUNS = Path(__file__).resolve().parent.parent / "runs"
PROPULSION_STAGE = 4
PPO_HYPERPARAMS = dict(learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=5,
                     gamma=cfg.discount(0.99), gae_lambda=0.95, clip_range=0.2,
                     ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03)
COLREGS_PARTS = ("port", "bow", "side", "hold", "r8")


def observation_schema(space):
    return {"version": cfg.OBSERVATION_SCHEMA_VERSION,
            "branches": {name: {"shape": list(branch.shape), "dtype": str(branch.dtype)}
                         for name, branch in space.spaces.items()}}


def validate_resume_schema(model_path, expected):
    path = Path(model_path).parent / "config.json"
    if not path.exists():
        raise ValueError("Checkpoint has no config.json observation schema; start a new CODEX policy")
    if json.loads(path.read_text(encoding="utf-8")).get("observation_schema") != expected:
        raise ValueError("Checkpoint observation schema differs; this setup requires retraining")


def output_directory(root, name):
    root = Path(root).resolve()
    if not root.is_relative_to(RUNS.resolve()):
        raise ValueError(f"Training output must stay under {RUNS}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError("Run name must be a single safe directory name")
    return root / name


def make_env(rank, seed, level, recovery_probability, supervisor):
    def initialize():
        torch.set_num_threads(1)
        curriculum.apply_stage(PROPULSION_STAGE)
        base = ASVLidarEnv(render_mode=None, emergency_stop=supervisor,
                           vessel_randomisation=cfg.VESSEL_RANDOMISATION_SCALE)
        return SceneTrainingEnv(base, seed=seed + rank * 104729,
                                level=level, recovery_probability=recovery_probability)
    return initialize


def trajectory_wrong_turn(heading, contexts, truth_latches):
    """Compare physical headings in one frame, with a separate truth latch.

    Perception chooses the obligation, but noisy engagement heading must never
    be subtracted from a true heading to judge the physical manoeuvre.
    """
    wrong = False
    for track_id, ctx in contexts.items():
        if ctx.t_engage < 0:
            continue
        key = (track_id, ctx.t_engage)
        reference = truth_latches.setdefault(key, float(heading))
        delta = (float(heading) - reference + 180.0) % 360.0 - 180.0
        wrong |= (ctx.engaged and not ctx.in_extremis and ctx.turn_admissible and
                  ctx.compliant_turn_sense != 0 and ctx.compliant_turn_sense * delta < -10.0)
    return bool(wrong)


def run_eval_episode(model, env, case, mode, *, max_steps=None):
    built, options = restore_case(case)
    env.estop_enabled = mode == "on"
    obs, _ = env.reset(seed=case["episode_seed"], options=options)
    cte, speed_error = [], []
    violation_frames = {part: 0 for part in COLREGS_PARTS}
    violations = wrong_turns = 0
    truth_latches = {}
    trajectory_wrong_turn(env.asv_h, env.observer.encounter_contexts, truth_latches)
    min_range = min_clearance = min_boundary = float("inf")
    total, steps = 0.0, 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
        cte.append(abs(float(info["cross_track_error"])))
        speed_error.append(abs(float(info["speed_mps"]) - cfg.U_NOM))
        flagged = False
        for part in COLREGS_PARTS:
            bad = info.get(f"colregs/v_{part}", 0.0) > 0.0
            violation_frames[part] += int(bad)
            flagged |= bad
        violations += int(flagged)
        # Trajectory diagnostic independent of reward values; the obligation
        # still comes from the declared encounter context, not a legal oracle.
        wrong_turns += int(trajectory_wrong_turn(env.asv_h, env.observer.encounter_contexts, truth_latches))
        hull = env.hull_polygon()
        min_boundary = min(min_boundary, float(env.true_border_clearance))
        for target in env.targets:
            min_range = min(min_range, float(np.hypot(target.x - env.asv_x, target.y - env.asv_y)))
            min_clearance = min(min_clearance, polygon_distance(hull, target.hull()))
        for obstacle in env.obstacles:
            min_clearance = min(min_clearance, polygon_distance(hull, obstacle))
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout" if truncated else "cutoff")
    return {"case_id": built.case_id, "case_sha256": canonical_hash(case),
            "episode_seed": case["episode_seed"], "stratum": case["stratum"],
            "class": built.encounter_class, "supervisor": mode,
            "outcome": outcome, "steps": steps, "return": total,
            "mean_abs_cte": float(np.mean(cte)), "max_abs_cte": float(np.max(cte)),
            "mean_speed_error": float(np.mean(speed_error)), "violation_fraction": violations / steps,
            "wrong_turn_fraction": wrong_turns / steps,
            **{f"frames_v_{p}": violation_frames[p] for p in COLREGS_PARTS},
            "estops": int(info["estop/events"]),
            "min_target_center_range": min_range if np.isfinite(min_range) else None,
            "min_hull_clearance": min_clearance if np.isfinite(min_clearance) else None,
            "min_boundary_clearance": min_boundary}


class MasteryEvalCallback(BaseCallback):
    """Paired evaluation; two consecutive passes advance the training level."""
    def __init__(self, run_dir, eval_freq, per_stratum, schema, *, level=1):
        super().__init__(0)
        self.run_dir, self.eval_freq = Path(run_dir), eval_freq
        self.per_stratum, self.schema, self.level = per_stratum, schema, level
        self.next_eval, self.last_eval, self.pass_streak = eval_freq, -1, 0
        self.best_rank, self.history, self.cases = None, [], {}
        self.transitions = [{"timesteps": 0, "level": level, "reason": "initial level"}]
        self.env = ASVLidarEnv(render_mode=None, emergency_stop=False)

    def _cases_for(self, level):
        if level not in self.cases:
            cases = build_development_cases(self.env, self.per_stratum, level)
            write_manifest(self.run_dir / f"development_level{level}.json", cases, self.schema)
            self.cases[level] = cases
        return self.cases[level]

    def _evaluate(self):
        all_rows, gate_rows = [], None
        suites = [("selection", 5)] + ([("curriculum", self.level)] if self.level != 5 else [])
        for suite, level in suites:
            for mode in ("off", "on"):
                rows = [dict(run_eval_episode(self.model, self.env, case, mode),
                             suite=suite, level=level, timesteps=int(self.num_timesteps))
                        for case in self._cases_for(level)]
                all_rows.extend(rows)
                summary = dict(summarize(rows), suite=suite, level=level,
                               supervisor=mode, timesteps=int(self.num_timesteps))
                summary["strata"] = {s: summarize([r for r in rows if r["stratum"] == s])
                                    for s in STRATA if any(r["stratum"] == s for r in rows)}
                summary["classes"] = {c: summarize([r for r in rows if r["class"] == c])
                                     for c in sorted({r["class"] for r in rows})}
                self.history.append(summary)
                if mode == "off" and level == self.level:
                    gate_rows = rows
                if suite == "selection" and mode == "off":
                    rank = checkpoint_rank(rows)
                    if self.best_rank is None or rank > self.best_rank:
                        self.best_rank = rank
                        self.model.save(self.run_dir / "best_model.zip")
                        self.training_env.save(str(self.run_dir / "best_vecnormalize.pkl"))
                print(f"[EVAL] {suite} level={level} supervisor={mode} "
                      f"goal={summary['goal_rate']:.2f} collision={summary['collision_rate']:.2f} "
                      f"violation_frames={summary['violation_fraction']:.3f}", flush=True)
        with (self.run_dir / "eval_episodes.jsonl").open("a", encoding="utf-8") as stream:
            for row in all_rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        (self.run_dir / "eval_summary.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        self.last_eval = self.num_timesteps
        passed = mastery_passes(gate_rows, self.level) if gate_rows else False
        self.pass_streak = self.pass_streak + 1 if passed else 0
        if self.pass_streak >= 2 and self.level < 5:
            self.level += 1
            self.pass_streak = 0
            self.training_env.env_method("set_curriculum_level", self.level)
            self.transitions.append({"timesteps": int(self.num_timesteps), "level": self.level,
                                     "reason": "two consecutive mastery evaluation passes"})
            print(f"[CURRICULUM] next reset uses level {self.level}", flush=True)
        (self.run_dir / "curriculum.json").write_text(json.dumps({
            "transitions": self.transitions, "current_level": self.level,
            "consecutive_passes": self.pass_streak, "last_gate_passed": passed}, indent=2), encoding="utf-8")

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_freq
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.last_eval != self.num_timesteps:
            self._evaluate()
        self.env.close()


def smoke_evaluation(model, run_dir, schema):
    """Capped one-case-per-stratum plumbing check, explicitly labelled cutoff."""
    env = ASVLidarEnv(render_mode=None, emergency_stop=False)
    cases = []
    try:
        for index, stratum in enumerate(STRATA):
            for attempt in range(30):
                seed = 208000 + index * 100 + attempt
                built = sample_scene(seed, stratum, 4, namespace="development",
                                     encounter_class="no_target" if index < 2 else "head_on")
                env.reset(seed=seed, options={"generated": built})
                if bool(env.obstacles) == (stratum in ("static", "combined")):
                    cases.append(case_record(built, env, seed))
                    break
            else:
                raise RuntimeError("Cannot construct smoke scene")
        write_manifest(run_dir / "smoke_cases.json", cases, schema)
        rows = [run_eval_episode(model, env, case, mode, max_steps=32)
                for mode in ("off", "on") for case in cases]
        (run_dir / "smoke_evaluation.json").write_text(json.dumps({
            "purpose": "training/save/load plumbing only; capped at 32 steps", "rows": rows},
            indent=2, allow_nan=False), encoding="utf-8")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=50000)
    parser.add_argument("--eval-per-stratum", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--tag", default="")
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--init-vecnormalize", type=Path)
    parser.add_argument("--start-level", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--train-supervisor", choices=("off", "on"), default="off")
    parser.add_argument("--recovery-probability", type=float, default=0.35)
    args = parser.parse_args()
    timesteps = args.timesteps if args.timesteps is not None else (1024 if args.smoke else 2000000)
    if min(timesteps, args.num_envs, args.eval_freq, args.torch_threads) <= 0:
        parser.error("Budgets, worker count, evaluation frequency and thread count must be positive")
    if args.eval_per_stratum < 4 and not args.smoke:
        parser.error("Mastery gates require at least four episodes per stratum")
    if args.init_vecnormalize and not args.init_model:
        parser.error("--init-vecnormalize requires --init-model")
    if not 0 <= args.recovery_probability <= 1:
        parser.error("Recovery probability must lie in [0,1]")
    torch.set_num_threads(args.torch_threads)
    curriculum.apply_stage(PROPULSION_STAGE)
    label = f"ppo_codex_seed{args.seed}_{args.tag or time.strftime('%Y%m%d_%H%M%S')}" + ("_smoke" if args.smoke else "")
    run_dir = output_directory(args.runs_dir, label)
    if run_dir.exists():
        parser.error(f"Run directory exists; choose a new --tag: {run_dir}")
    probe = ASVLidarEnv(render_mode=None)
    schema = observation_schema(probe.observation_space)
    probe.close()
    if args.init_model:
        validate_resume_schema(args.init_model, schema)
    run_dir.mkdir(parents=True)
    hyperparameters = dict(PPO_HYPERPARAMS)
    if args.smoke:
        hyperparameters.update(n_steps=128, batch_size=64, n_epochs=2)
    config = {"algorithm": "PPO", "observation_schema": schema, "seed": args.seed,
              "timesteps": timesteps, "num_envs": args.num_envs, "smoke_only": args.smoke,
              "hyperparameters": hyperparameters, "scene_weights_final": FINAL_WEIGHTS,
              "propulsion_stage": PROPULSION_STAGE,
              "rpm_stage_params": curriculum.stage_params(PROPULSION_STAGE),
              "vessel_randomisation": cfg.VESSEL_RANDOMISATION_SCALE,
              "start_level": args.start_level, "curriculum": "two consecutive measured mastery passes",
              "train_supervisor": args.train_supervisor, "eval_supervisor": ["off", "on"],
              "selection_rank": "safety, worst-stratum goals, trajectory turn diagnostic, COLREG proxy, goals, tracking",
              "selection_clearance_m": {"hull": MIN_EVAL_HULL_CLEARANCE_M,
                                        "boundary": MIN_EVAL_BOUNDARY_CLEARANCE_M},
              "colregs_metrics_scope": "declared encounter obligations; not legal certification",
              "recovery_probability": args.recovery_probability, "source_sha256": source_provenance(),
              "device": args.device, "torch_threads": args.torch_threads,
              "init_model": str(args.init_model) if args.init_model else None,
              "init_vecnormalize": str(args.init_vecnormalize) if args.init_vecnormalize else None}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    constructors = [make_env(i, args.seed, args.start_level, args.recovery_probability,
                             args.train_supervisor == "on") for i in range(args.num_envs)]
    vec = DummyVecEnv(constructors) if args.num_envs == 1 else SubprocVecEnv(constructors)
    vec = VecMonitor(vec, filename=str(run_dir / "monitor.csv"),
                     info_keywords=("reached_goal", "collided", "scene_stratum", "scenario_class"))
    vec = (VecNormalize.load(str(args.init_vecnormalize), vec) if args.init_vecnormalize else
           VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=hyperparameters["gamma"]))
    vec.training, vec.norm_obs, vec.norm_reward = True, False, True
    torch.set_num_threads(args.torch_threads)
    started = time.time()
    try:
        model = (PPO.load(str(args.init_model), env=vec, device=args.device, seed=args.seed,
                          **hyperparameters) if args.init_model else
                 PPO("MultiInputPolicy", vec, seed=args.seed, device=args.device, verbose=1,
                     policy_kwargs=dict(features_extractor_class=ASVFeaturesExtractor,
                                        net_arch=dict(pi=[128, 128], vf=[128, 128]),
                                        activation_fn=nn.ReLU), **hyperparameters))
        callbacks = [] if args.smoke else [
            MasteryEvalCallback(run_dir, args.eval_freq, args.eval_per_stratum, schema, level=args.start_level),
            CheckpointCallback(save_freq=max(250000 // args.num_envs, 1), save_path=str(run_dir),
                               name_prefix="ppo", save_vecnormalize=True)]
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
        model.save(run_dir / "final_model.zip")
        vec.save(str(run_dir / "final_vecnormalize.pkl"))
        reloaded = PPO.load(str(run_dir / "final_model.zip"), device=args.device)
        if reloaded.observation_space != vec.observation_space:
            raise RuntimeError("Saved policy observation space differs after reload")
        if args.smoke:
            smoke_evaluation(reloaded, run_dir, schema)
        config.update(completed=True, actual_timesteps=int(model.num_timesteps), wall_clock_s=time.time() - started)
    finally:
        vec.close()
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved isolated run to {run_dir}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
