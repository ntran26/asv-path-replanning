"""Fine-tune a baseline-v2 run from 2 M to 3 M steps with field layouts mixed in.

    python src/finetune_field.py --from runs/sac_formulation_seed0_bl2
    python src/finetune_field.py --from runs/sac_formulation_seed0_bl2 --smoke     # launch check
    python src/finetune_field.py --from runs/sac_formulation_seed0_bl2_ftfield1         --spec configs/finetune_field_v2.json                                      # v2, from v1's best

Spec options added for v2 (absent in v1, which keeps its behaviour): `start_from`
("final" -- the source's final_model.zip -- or "best", its best_model.zip with
best_vecnormalize.pkl and best replay buffer), `total_timesteps` (train to this
step count instead of `extra_timesteps` more), `encounter_weights` and
`prefetch` (passed to `env.set_field_mix`), and `entropy_boost` (SAC/TQC: raise
the entropy coefficient and target for the first `steps`, then restore).

Loads the run's 2 M `final_model.zip`, its `final_vecnormalize.pkl` and (SAC/TQC)
its 2 M replay buffer, and trains `extra_timesteps` more on stage-5 scenarios of
which `field_share` are Paper 2-style field layouts (`field_training.py`; the
spec is `configs/finetune_field_v1.json`).  Writes a new run folder,
`runs/<run>_<tag>/` -- the source run is never touched -- with the usual
checkpoints, `eval_summary.json` / `eval_episodes.csv` (development set plus the
field validation set, with a `set` column) and `best_model.zip`, selected on the
combined score.  The 2 M model is evaluated first and stays best unless beaten.

The base formulation must still be baseline-v2 (`baseline_config.check`): the
fine-tune adds scenarios, not a new reward, observation or learner.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import field_training  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize  # noqa: E402

ROOT = HERE.parent
DEFAULT_SPEC = ROOT / "configs" / "finetune_field_v1.json"


class FieldEvalCallback(tf.FormulationEvalCallback):
    """The development set plus the field validation set, scored together for
    selection and reported apart (`dev/*`, `field/*`)."""

    def __init__(self, run_dir, eval_freq, start_steps, supervisor_modes, algo, per_class,
                 extra=None, resume_from=None):
        super().__init__(run_dir, eval_freq, per_class, supervisor_modes=supervisor_modes, algo=algo,
                         resume_from=resume_from)
        self.n_dev = len(self.scenarios)
        print("[FIELD] building the field evaluation scenarios ...", flush=True)
        self.scenarios = list(self.scenarios) + list((extra or field_training.validation_set)())
        if resume_from is None:
            self.next_eval = int(start_steps)        # fine-tune: evaluate the start model first

    def _evaluate_mode(self, mode: str, select: bool) -> None:
        started = time.time()
        self.env.estop_enabled = (mode == "on")
        rows = []
        for i, built in enumerate(self.scenarios):
            r = tf.run_eval_episode(self.model, self.env, built, 900_000 + i)
            r.update(supervisor=mode, set="dev" if i < self.n_dev else "field",
                     case_id=getattr(built, "case_id", ""))
            rows.append(r)
        goal = lambda rs: sum(r["outcome"] == "goal" for r in rs) / max(len(rs), 1)
        coll = lambda rs: sum(r["outcome"].startswith("collision") for r in rs) / max(len(rs), 1)
        dev, fld = rows[:self.n_dev], rows[self.n_dev:]
        summary = {"timesteps": int(self.num_timesteps), "supervisor": mode, "episodes": len(rows),
                   "goal_rate": goal(rows), "collision_rate": coll(rows),
                   "timeout_rate": sum(r["outcome"] == "timeout" for r in rows) / len(rows),
                   "dev/goal": goal(dev), "dev/collision": coll(dev),
                   "field/goal": goal(fld), "field/collision": coll(fld),
                   "mean_return": float(np.mean([r["return"] for r in rows])),
                   "intervention_rate": float(np.mean([r["estops"] > 0 for r in rows])),
                   "eval_wall_s": round(time.time() - started, 1)}
        for code in ("NT",) + field_training.ENCOUNTER_CODES:
            sel = [r for r in fld if r["case_id"].split("-")[1:2] == [code]]
            if sel:
                summary[f"field/{code}/goal"] = goal(sel)
        self.history.append(summary)
        with open(self.run_dir / "eval_summary.json", "w") as fh:
            json.dump(self.history, fh, indent=1)
        detail = self.run_dir / "eval_episodes.csv"
        new = not detail.exists()
        with open(detail, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["timesteps", *rows[0].keys()])
            if new:
                writer.writeheader()
            for r in rows:
                writer.writerow({"timesteps": int(self.num_timesteps), **r})
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                self.logger.record(f"eval_supervisor_{mode}/{key}", value)
        score = summary["goal_rate"] - 2.0 * summary["collision_rate"]
        if select and score > self.best:
            self.best = score
            self.model.save(self.run_dir / "best_model.zip")
            self.training_env.save(str(self.run_dir / "best_vecnormalize.pkl"))
            if getattr(self.model, "replay_buffer", None) is not None:
                tf.save_replay_buffer(self.model, tf.best_replay_buffer_path(self.run_dir, self.algo))
        per_code = " ".join(f"{c} {summary[f'field/{c}/goal']:.2f}" for c in ("NT",) + field_training.ENCOUNTER_CODES
                            if f"field/{c}/goal" in summary)
        print(f"[EVAL] t={self.num_timesteps:,} supervisor {mode} goal {summary['goal_rate']:.2f} "
              f"collision {summary['collision_rate']:.2f} | dev goal {summary['dev/goal']:.2f} "
              f"| field goal {summary['field/goal']:.2f} ({per_code}) ({summary['eval_wall_s']} s)", flush=True)


class EntropyBoost(tf.BaseCallback):
    """finetune-field-v2: after 2 M steps SAC's entropy coefficient has decayed
    (0.0025 at 2.3 M), so the policy barely explores the manoeuvres the field
    layouts need.  Raise the coefficient to `initial_ent_coef` and the entropy
    target to `target_entropy` for the first `steps`, then restore the target
    (auto-tuning then brings the coefficient back down)."""

    def __init__(self, initial_ent_coef: float, target_entropy: float, steps: int):
        super().__init__(0)
        self.initial, self.target, self.steps = float(initial_ent_coef), float(target_entropy), int(steps)
        self.until, self.saved = None, None

    def _on_training_start(self) -> None:
        import math
        import torch
        m = self.model
        if getattr(m, "log_ent_coef", None) is None:
            return                                   # fixed coefficient: nothing to boost
        with torch.no_grad():
            m.log_ent_coef.fill_(math.log(self.initial))
        self.saved, m.target_entropy = float(m.target_entropy), self.target
        self.until = self.num_timesteps + self.steps
        print(f"[ENTROPY] coefficient -> {self.initial}, target {self.saved} -> {self.target} "
              f"until t={self.until:,}", flush=True)

    def _on_step(self) -> bool:
        if self.until is not None and self.num_timesteps >= self.until:
            self.model.target_entropy, self.until = self.saved, None
            print(f"[ENTROPY] t={self.num_timesteps:,}: target restored to {self.saved}", flush=True)
        return True


def _find_buffer(run_dir: Path, algo: str, steps: int):
    names = [f"{algo}_replay_buffer_{steps}_steps.pkl"]
    for folder in (tf.REPLAY_BUFFER_DIR / run_dir.name, tf.REPLAY_BUFFER_DIR / f"{run_dir.name}_kept", run_dir):
        for name in names:
            if (folder / name).exists():
                return folder / name
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="source", type=Path, required=True, help="the baseline-v2 run folder")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--smoke", action="store_true", help="4,096 steps into a _smoke folder, no evaluation")
    args = ap.parse_args()
    tf._no_efficiency_mode()

    spec = json.loads(args.spec.read_text())
    source = args.source if args.source.is_absolute() else ROOT / args.source
    previous = json.loads((source / "config.json").read_text())
    algo, seed = previous["algo"], int(previous["seed"])
    import baseline_config
    base = json.loads((ROOT / spec["base_config"]).read_text())
    problems = baseline_config.check(cfg, tf, base)      # installs baseline-v3's overlay if it has one
    if problems:
        raise SystemExit(f"code does not match {base['id']}: " + "; ".join(problems))
    overlay = "overlay" in base.get("formulation", {})
    start_from = spec.get("start_from", "final")
    if start_from == "checkpoint":
        # SAC recovery plan: an explicit checkpoint, its VecNormalize and a kept replay buffer.
        resolve = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
        start_model, start_vecnorm = resolve(spec["start_model"]), resolve(spec["start_vecnormalize"])
        buffer = resolve(spec["start_buffer"]) if algo in tf.OFF_POLICY else None
        if buffer is not None and not buffer.exists():
            raise SystemExit(f"replay buffer not found: {buffer}")
    elif start_from == "best":
        start_model, start_vecnorm = source / "best_model.zip", source / "best_vecnormalize.pkl"
        buffer = tf.best_replay_buffer_path(source, algo) if algo in tf.OFF_POLICY else None
        if buffer is not None and not buffer.exists():
            buffer = None
    else:
        start_model, start_vecnorm = source / "final_model.zip", source / "final_vecnormalize.pkl"
        buffer = (_find_buffer(source, algo, int(previous["timesteps"]))
                  if algo in tf.OFF_POLICY else None)
    if not start_model.exists():
        raise SystemExit(f"{source.name} has no {start_model.name}")
    if algo in tf.OFF_POLICY and buffer is None:
        raise SystemExit(f"no replay buffer for {source.name}'s {start_model.name}")
    import zipfile
    with zipfile.ZipFile(start_model) as z:          # the checkpoint's own step count
        start_steps = int(json.loads(z.read("data"))["num_timesteps"])

    if args.smoke:
        extra = 4_096
    elif spec.get("total_timesteps"):
        extra = int(spec["total_timesteps"]) - start_steps
    else:
        extra = int(spec["extra_timesteps"])
    base_run = previous.get("source_run", source.name)     # a fine-tune of a fine-tune keeps the base name
    run_dir = source.parent / (f"{base_run}_{spec['tag']}" + ("_smoke" if args.smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    num_envs = int(previous["num_envs"])
    config = {"finetune": spec, "spec_path": str(args.spec), "source_run": base_run,
              "started_from": f"{source.name}/{start_model.name}",
              "algo": algo, "seed": seed, "start_steps": start_steps, "extra_timesteps": extra,
              "timesteps": start_steps + extra, "num_envs": num_envs,
              "hyperparameters": previous.get("hyperparameters"),
              "train_supervisor": previous.get("train_supervisor"),
              "eval_supervisor": previous.get("eval_supervisor"),
              "low_speed_start_frac": previous.get("low_speed_start_frac", 0.0),
              "baseline_config": {"id": base["id"], "formulation_digest": base["formulation_digest"]},
              "field_training_revision": field_training.REVISION,
              "replay_buffer": str(buffer) if buffer else None,
              "switches": tf._formulation_switches(), "platform": tf._platform(),
              "observation_schema": cfg.OBSERVATION_SCHEMA_VERSION}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1, default=str))
    share = spec.get("field_share", (spec.get("stage_definition") or {}).get("overrides", {}).get("field_share"))
    print(f"[FINETUNE] {source.name} {start_steps:,} -> {start_steps + extra:,} steps, "
          f"field share {share}, into {run_dir.name}", flush=True)

    base_seed = 100_000 * (seed + 1) + int(spec["reset_seed_offset"])
    stage = int(spec["scenario_stage"])
    definition = spec.get("stage_definition")
    # A spec-defined stage is installed after the workers start (it does not exist
    # in `constants` yet), so they start one stage earlier.
    start_stage = int(definition["base_stage"]) if definition else stage
    vec = SubprocVecEnv([tf.make_env(i, base_seed, start_stage, cfg.VESSEL_RANDOMISATION_SCALE,
                                     supervisor=previous.get("train_supervisor", "on") == "on",
                                     low_speed_start_frac=float(previous.get("low_speed_start_frac", 0.0)),
                                     overlay=overlay)
                         for i in range(num_envs)])
    if definition:
        full = {**cfg.CURRICULUM_STAGES[int(definition["base_stage"])], **definition["overrides"]}
        vec.env_method("define_stage", stage, full)
        config["stage_definition_resolved"] = {k: v for k, v in full.items()}
        (run_dir / "config.json").write_text(json.dumps(config, indent=1, default=str))
        print(f"[FINETUNE] stage {stage} installed in every worker: base stage "
              f"{definition['base_stage']} + {sorted(definition['overrides'])}", flush=True)
    else:
        vec.env_method("set_field_mix", float(spec["field_share"]), spec.get("encounter_weights"),
                       int(spec.get("prefetch", 0)))
    vec = tf.RetryingVecMonitor(vec, filename=str(run_dir / "monitor.csv"),
                                info_keywords=("reached_goal", "collided", "scenario_class"))
    vec = VecNormalize.load(str(start_vecnorm), vec)
    vec.training, vec.norm_reward = True, True

    model = tf.ALGORITHMS[algo].load(str(start_model), env=vec, device="cpu",
                                     tensorboard_log=str(ROOT / "runs" / "tensorboard"))
    if buffer is not None:
        model.load_replay_buffer(str(buffer))
        print(f"[FINETUNE] replay buffer restored ({buffer.name}, {model.replay_buffer.size():,} transitions)", flush=True)
    every = max(int(spec["checkpoint_every"]) // num_envs, 1)
    callbacks = [CheckpointCallback(save_freq=every, save_path=str(run_dir), name_prefix=algo,
                                    save_vecnormalize=True)]
    if algo in tf.OFF_POLICY:
        callbacks.append(tf.ReplayBufferCheckpoint(run_dir, algo, every))
    boost = spec.get("entropy_boost")
    if boost and algo in ("sac", "tqc"):
        callbacks.append(EntropyBoost(boost["initial_ent_coef"], boost["target_entropy"], boost["steps"]))
    if not args.smoke:
        modes = ("off", "on") if previous.get("eval_supervisor") == "both" else (previous.get("eval_supervisor", "on"),)
        extra_set = None
        if overlay:
            import formulation_v3
            extra_set = formulation_v3.field_development_set
        callbacks.append(FieldEvalCallback(run_dir, int(spec["eval_freq"]), start_steps, modes, algo,
                                           int(previous.get("eval_per_class", 20)), extra=extra_set))
    started = time.time()
    model.learn(total_timesteps=extra, reset_num_timesteps=False, callback=CallbackList(callbacks),
                tb_log_name=run_dir.name, progress_bar=False)
    model.save(run_dir / "final_model.zip")
    vec.save(str(run_dir / "final_vecnormalize.pkl"))
    vec.close()
    config["wall_clock_s"] = time.time() - started
    (run_dir / "config.json").write_text(json.dumps(config, indent=1, default=str))
    print(f"done in {config['wall_clock_s'] / 3600:.2f} h -> {run_dir}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
