"""Single-algorithm test of the Paper 3 formulation: observation, reward, COLREGs.

    python src/train_formulation.py --timesteps 2000000 --seed 0
    python src/train_formulation.py --timesteps 20000 --eval-freq 10000 --smoke

**PPO, one seed, until the observation and reward are frozen** (user decision,
revision 8).  PPO rather than RecurrentPPO: the observation already carries the
tracker's velocity estimates, so the task is close to Markov; the design keeps
the headline architecture feedforward so recurrence cannot confound N1
(`constants.USE_RECURRENCE`); and on-policy PPO with vectorised workers is the
fastest route to a learning curve on a CPU.  `sb3-contrib` is not installed.

What a run is for, and what it is not
-------------------------------------
It exists to find formulation defects -- a term that dominates, a class the
agent cannot learn, an observation that carries nothing -- before the full
budget is spent.  It is **not** a result: one seed says nothing about variance.

Training distribution
---------------------
04a's scenario generator (`ScenarioGenerator`), curriculum stages 1 -> 5 on a
fraction of the budget, propulsion stage 4 throughout (stop to 2x cruise:
Rule 8(e) slowdowns need the authority), hull randomisation at
`VESSEL_RANDOMISATION_SCALE`, nominal pose and ego noise, the free-space
tracker, the emergency-stop supervisor.

Evaluation
----------
A fixed development-namespace set -- the same scenarios every time, never the
frozen suite -- at stage 5, per class, deterministic policy, nominal hull.
Reported per class: goal, collision by kind, timeout, COLREGs penalty and
violation frames, domain intrusion, emergency stops.

Everything goes under `runs/ppo_formulation_seed{N}[_{tag}]/`, or `--runs-dir`.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch.nn as nn
from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import scenario as scn  # noqa: E402
from env import ASVLidarEnv  # noqa: E402
from observation import observation_space  # noqa: E402
from features_extractor import ASVFeaturesExtractor  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"


class RetryingVecMonitor(VecMonitor):
    """VecMonitor whose CSV write survives a transient file lock.

    The first formulation run died at 1.91 M steps on `PermissionError` in
    `monitor.csv`'s flush: the run directory is inside OneDrive, whose sync
    client opens files it is uploading.  A lost monitor row costs nothing; a
    lost run costs three hours.
    """

    def step_wait(self):
        writer, self.results_writer = self.results_writer, None
        try:
            obs, rewards, dones, infos = super().step_wait()
        finally:
            self.results_writer = writer
        if writer is not None:
            for i in np.flatnonzero(dones):
                episode = infos[i].get("episode")
                if episode is None:
                    continue
                row = {k: episode[k] for k in ("r", "l", "t")}
                row.update({k: infos[i][k] for k in self.info_keywords})
                for attempt in range(5):
                    try:
                        writer.write_row(row)
                        break
                    except PermissionError:
                        time.sleep(0.2 * (attempt + 1))
        return obs, rewards, dones, infos


PROPULSION_STAGE = 4

# Scenario curriculum on the fraction of the budget.  Stages 1-2 teach the
# channel with no target; 3 introduces head-on and null; 4-5 all classes.
STAGE_SCHEDULE = cfg.CURRICULUM_STAGE_FRACTIONS    # TODO(04-3), resolved (F75)

PPO_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 512,               # per worker; x10 workers = 5,120 per update
    "batch_size": 512,
    "n_epochs": 10,
    "gamma": cfg.discount(0.99),  # 10 s horizon at 2 Hz (A12)
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # The first run's approx_kl climbed from 0.01 to 0.25 with half of every
    # batch clipped: ten epochs over one rollout overfit it.  Early-stopping the
    # epochs at a KL budget keeps the update inside the trust region.
    "target_kl": 0.03,
}

# The off-policy arms (04a §8: SAC is the protected core's method; TQC is the
# distributional comparator, 04a §8.2 rank 1).  Same discount as PPO, so the three differ in
# the learner and nothing else.  One gradient step per transition collected
# (`gradient_steps = num_envs` per vectorised step) is SAC's usual
# update-to-data ratio; `--gradient-steps` lowers it if throughput demands.
SAC_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": PPO_HYPERPARAMS["gamma"],
    "learning_starts": 10_000,
    "train_freq": 1,
    "ent_coef": "auto",
}
# TQC (Kuznetsov et al., 2020): SAC's settings with distributional critics, the
# paper's defaults -- 2 critics x 25 quantiles, the top 2 per critic dropped.
TQC_HYPERPARAMS = {"top_quantiles_to_drop_per_net": 2}
TQC_POLICY = {"n_quantiles": 25, "n_critics": 2}
# TD3 (Fujimoto et al., 2018) on the same replay settings as SAC, with the
# paper's target smoothing and delayed actor, and Gaussian exploration noise of
# 0.1 on the normalised action.
TD3_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": PPO_HYPERPARAMS["gamma"],
    "learning_starts": 10_000,
    "train_freq": 1,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
}
TD3_ACTION_NOISE_SIGMA = 0.1
# RecurrentPPO: PPO's settings with an LSTM after the features extractor, for
# actor and critic separately.  It answers the memory question (04a §8.2, rank 2)
# the context branch now partly answers by construction (A25).
RECURRENT_PPO_POLICY = {"lstm_hidden_size": 256, "n_lstm_layers": 1,
                        "enable_critic_lstm": True, "shared_lstm": False}
ALGORITHMS = {"ppo": PPO, "recurrent_ppo": RecurrentPPO, "td3": TD3,
              "sac": SAC, "tqc": TQC}
OFF_POLICY = ("td3", "sac", "tqc")


class EpisodeActor:
    """Deterministic actions for one episode, for any learner.

    RecurrentPPO's policy carries an LSTM state; calling `predict` without it
    would reset the memory every step and evaluate a different policy from the
    one trained.  `reset()` at each episode start.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.recurrent = isinstance(model, RecurrentPPO)
        self.reset()

    def reset(self) -> None:
        self.state = None
        self.start = np.ones((1,), dtype=bool)

    def __call__(self, obs):
        if not self.recurrent:
            return self.model.predict(obs, deterministic=True)[0]
        action, self.state = self.model.predict(obs, state=self.state,
                                                episode_start=self.start, deterministic=True)
        self.start = np.zeros((1,), dtype=bool)
        return action

EVAL_CLASSES = ("head_on", "crossing", "overtaking", "being_overtaken", "null", "no_target")
COLREGS_PARTS = ("port", "bow", "side", "hold", "r8")


def _platform() -> Dict:
    """Software and machine a run trained on (A26): the campaign may split between
    this machine and a cluster, and the versions are what must match."""
    import platform
    import sb3_contrib
    import stable_baselines3
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__, "numpy": np.__version__,
            "os": platform.platform(), "machine": platform.node(),
            "cpu_count": os.cpu_count()}


def _formulation_switches() -> Dict:
    """The settings a run's reward and scenario draw depend on (F86): recorded in
    `config.json` so a resume can refuse code that has changed underneath it."""
    return {"R2_SLOWDOWN_TEST": cfg.R2_SLOWDOWN_TEST,
            "CROSSING_PORT_SHARE_TRAINING": cfg.CROSSING_PORT_SHARE_TRAINING,
            "V_HOLD_GROWS": cfg.V_HOLD_GROWS,
            "V_PORT_HEADING_DEAD_DEG": cfg.V_PORT_HEADING_DEAD_DEG,
            "V_PORT_LATCHED_RHO": cfg.V_PORT_LATCHED_RHO,
            "V_PORT_HEADING_NEEDS_ADMISSIBLE": cfg.V_PORT_HEADING_NEEDS_ADMISSIBLE,
            "STAGE3_WEIGHTS": cfg.CURRICULUM_STAGES[3].get("weights"),
            "DEFAULT_GEOMETRY_MODE": cfg.DEFAULT_GEOMETRY_MODE,
            "OBSERVATION_SCHEMA_VERSION": cfg.OBSERVATION_SCHEMA_VERSION}


def stage_at(fraction: float) -> int:
    stage = STAGE_SCHEDULE[0][1]
    for start, value in STAGE_SCHEDULE:
        if fraction >= start:
            stage = value
    return stage


def make_env(rank: int, seed: int, stage: int, randomisation, torch_threads: int = 0,
             supervisor: bool = True, low_speed_start_frac: float = 0.0,
             r2_slowdown_test: str = None):
    def _init():
        if torch_threads:
            import torch
            torch.set_num_threads(torch_threads)
        if r2_slowdown_test:
            cfg.R2_SLOWDOWN_TEST = r2_slowdown_test
        curriculum.apply_stage(PROPULSION_STAGE)
        # F68: `supervisor=False` trains without the stop latch, so R_ESTOP and
        # the latch-held speed-gate suspension never act -- the stop becomes a
        # runtime layer evaluated around the policy, not part of what it learns.
        env = ASVLidarEnv(render_mode=None, scenario_stage=stage,
                          vessel_randomisation=randomisation,
                          emergency_stop=supervisor,
                          low_speed_start_frac=low_speed_start_frac)
        env.reset(seed=seed + rank)
        return env
    return _init


class ScenarioStageCallback(BaseCallback):
    def __init__(self, total: int, log_path: str, resume: bool = False):
        super().__init__(0)
        self.total = int(total)
        self.log_path = log_path
        self.stage = None
        self.transitions: List[Dict] = []
        if resume and Path(log_path).exists():
            with open(log_path) as fh:
                self.transitions = json.load(fh)

    def _apply(self) -> None:
        wanted = stage_at(self.num_timesteps / max(self.total, 1))
        if wanted != self.stage:
            self.training_env.env_method("set_scenario_stage", wanted)
            self.stage = wanted
            self.transitions.append({"timesteps": int(self.num_timesteps), "stage": wanted})
            print(f"[CURRICULUM] t={self.num_timesteps:,} -> scenario stage {wanted}", flush=True)
            with open(self.log_path, "w") as fh:
                json.dump(self.transitions, fh, indent=1)

    def _on_training_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:
        self._apply()
        return True


def development_set(per_class: int) -> List:
    """Fixed evaluation scenarios, generated once, stage 5, development namespace."""
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    out = []
    for cls in EVAL_CLASSES:
        index, found = 0, 0
        while found < per_class and index < 50 * per_class:
            built = generator.sample(scn.seed_for("development", 10_000 * (EVAL_CLASSES.index(cls) + 1) + index),
                                     encounter_class=cls)
            index += 1
            if built is not None:
                out.append(built)
                found += 1
    return out


def run_eval_episode(model, env: ASVLidarEnv, built, seed: int) -> Dict:
    obs, _ = env.reset(seed=seed, options={"generated": built})
    total = 0.0
    steps = 0
    col_sum = 0.0
    violation_frames = defaultdict(int)
    dom_frames = 0
    cte = []
    min_range = float("inf")
    speeds = []
    actor = EpisodeActor(model)
    while True:
        action = actor(obs)
        obs, reward, term, trunc, info = env.step(action)
        speeds.append(float(info["speed_mps"]))
        steps += 1
        total += float(reward)
        col_sum += float(info["reward/weighted/col"])
        for part in COLREGS_PARTS:
            violation_frames[part] += int(info.get(f"colregs/v_{part}", 0.0) > 0.0)
        dom_frames += int(info["reward/term/dom"] < 0.0)
        cte.append(abs(float(info["cross_track_error"])))
        for target in env.targets:
            min_range = min(min_range, float(np.hypot(target.x - env.asv_x, target.y - env.asv_y)))
        if term or trunc:
            break
    # Collision first, as the reward charges it: a step that both reaches the
    # goal and collides is paid -300, and labelling it a goal hid F49.
    outcome = (f"collision:{info['collision_kind']}" if info["collided"] else
               "goal" if info["reached_goal"] else "timeout")
    return {"class": built.encounter_class, "width": float(built.nominal_width),
            # The drawn geometry, so a class result can be split without a join
            # (A15's floor label; the crossing side follows from `ct_deg`).
            "dcpa_m": float(getattr(built, "dcpa_m", 0.0)),
            "ct_deg": float(getattr(built, "ct_deg", 0.0)),
            "dcpa_below_floor": getattr(built, "dcpa_below_floor", None),
            "outcome": outcome, "steps": steps, "return": total, "colregs_integral": col_sum,
            **{f"frames_v_{p}": violation_frames[p] for p in COLREGS_PARTS},
            "domain_intrusion_frames": dom_frames, "estops": int(info["estop/events"]),
            "mean_abs_cte": float(np.mean(cte)), "min_target_range": min_range,
            "mean_speed": float(np.mean(speeds)), "max_speed": float(np.max(speeds))}


class FormulationEvalCallback(BaseCallback):
    """Per-class development evaluation.

    `supervisor_modes` (F68): evaluate with the stop latch off, on, or both.  With
    both, outcomes and compliance are attributable to the policy in the "off"
    rows, and the "on" rows report the runtime layer's intervention rate.  The
    best model is chosen on the policy's own ("off") score when it is evaluated.
    """

    def __init__(self, run_dir: Path, eval_freq: int, per_class: int,
                 supervisor_modes=("on",), resume_from: Optional[int] = None):
        super().__init__(0)
        self.run_dir = Path(run_dir)
        self.eval_freq = int(eval_freq)
        self.scenarios = development_set(per_class)
        self.supervisor_modes = tuple(supervisor_modes)
        curriculum.apply_stage(PROPULSION_STAGE)
        self.env = ASVLidarEnv(render_mode=None)
        self.best = -np.inf
        self.next_eval = self.eval_freq
        self.history: List[Dict] = []
        self.resume_from = resume_from
        if resume_from is not None:
            self._restore(int(resume_from))

    def _restore(self, steps: int) -> None:
        """Resume (F86): keep what was evaluated up to the checkpoint, drop what
        the resumed run will redo, and carry the best-model score forward."""
        summary = self.run_dir / "eval_summary.json"
        if summary.exists():
            with open(summary) as fh:
                self.history = [h for h in json.load(fh) if int(h["timesteps"]) <= steps]
            with open(summary, "w") as fh:
                json.dump(self.history, fh, indent=1)
        select_on = "off" if "off" in self.supervisor_modes else self.supervisor_modes[0]
        scores = [h["goal_rate"] - 2.0 * h["collision_rate"] for h in self.history
                  if h.get("supervisor", select_on) == select_on]
        self.best = max(scores) if scores else -np.inf
        detail = self.run_dir / "eval_episodes.csv"
        if detail.exists():
            with open(detail, newline="") as fh:
                rows = list(csv.DictReader(fh))
            kept = [r for r in rows if int(float(r["timesteps"])) <= steps]
            if rows:
                with open(detail, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(kept)
        self.next_eval = (steps // self.eval_freq + 1) * self.eval_freq

    def _evaluate(self) -> None:
        select_on = "off" if "off" in self.supervisor_modes else self.supervisor_modes[0]
        for mode in self.supervisor_modes:
            self._evaluate_mode(mode, select=(mode == select_on))

    def _evaluate_mode(self, mode: str, select: bool) -> None:
        started = time.time()
        self.env.estop_enabled = (mode == "on")
        rows = [dict(run_eval_episode(self.model, self.env, built, 900_000 + i), supervisor=mode)
                for i, built in enumerate(self.scenarios)]
        n = len(rows)
        summary = {"timesteps": int(self.num_timesteps), "supervisor": mode, "episodes": n,
                   "goal_rate": sum(r["outcome"] == "goal" for r in rows) / n,
                   "collision_rate": sum(r["outcome"].startswith("collision") for r in rows) / n,
                   "timeout_rate": sum(r["outcome"] == "timeout" for r in rows) / n,
                   "mean_return": float(np.mean([r["return"] for r in rows])),
                   "mean_colregs_integral": float(np.mean([r["colregs_integral"] for r in rows])),
                   "estops_per_episode": float(np.mean([r["estops"] for r in rows])),
                   # F68: the share of episodes in which the runtime layer had to act.
                   "intervention_rate": float(np.mean([r["estops"] > 0 for r in rows])),
                   "eval_wall_s": round(time.time() - started, 1)}
        for kind in ("boundary", "obstacle", "target"):
            summary[f"collision_{kind}"] = sum(r["outcome"] == f"collision:{kind}" for r in rows) / n
        for cls in EVAL_CLASSES:
            sel = [r for r in rows if r["class"] == cls]
            if not sel:
                continue
            summary[f"{cls}/goal"] = sum(r["outcome"] == "goal" for r in sel) / len(sel)
            summary[f"{cls}/collision"] = sum(r["outcome"].startswith("collision") for r in sel) / len(sel)
            summary[f"{cls}/colregs"] = float(np.mean([r["colregs_integral"] for r in sel]))
            summary[f"{cls}/intervention"] = float(np.mean([r["estops"] > 0 for r in sel]))
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
        print(f"[EVAL] t={self.num_timesteps:,} supervisor {mode} goal {summary['goal_rate']:.2f} "
              f"collision {summary['collision_rate']:.2f} timeout {summary['timeout_rate']:.2f} "
              f"return {summary['mean_return']:.1f} colregs {summary['mean_colregs_integral']:.1f} "
              f"intervention {summary['intervention_rate']:.2f} "
              f"({summary['eval_wall_s']} s)", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_freq
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        self._evaluate()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=10)
    ap.add_argument("--eval-freq", type=int, default=200_000)
    ap.add_argument("--eval-per-class", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs-dir", type=Path, default=RUNS,
                    help="output root (default: the project's runs/; RetryingVecMonitor "
                         "handles OneDrive's file locks)")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="PyTorch threads in the learner (workers get 1); 0 keeps the default, "
                         "which oversubscribes the CPU alongside the environment workers")
    ap.add_argument("--init-model", type=Path, default=None,
                    help="Tier 2: fine-tune from this saved model instead of a fresh policy")
    ap.add_argument("--init-vecnormalize", type=Path, default=None,
                    help="reward-normalisation statistics to continue from (with --init-model)")
    ap.add_argument("--train-supervisor", choices=("on", "off"), default="on",
                    help="F68: run the stop latch in the training environments")
    ap.add_argument("--eval-supervisor", choices=("on", "off", "both"), default="on",
                    help="F68: evaluate with the stop latch off, on, or both")
    ap.add_argument("--low-speed-start-frac", type=float, default=0.0,
                    help="F68: share of training episodes starting slow or at rest")
    ap.add_argument("--r2-slowdown-test", choices=("coast", "stop"), default=None,
                    help="F70: R-2's slowing test (default: constants.R2_SLOWDOWN_TEST)")
    ap.add_argument("--checkpoint-every", type=int, default=250_000,
                    help="environment steps between checkpoints")
    ap.add_argument("--resume", type=Path, default=None,
                    help="F86: continue this run directory from its latest checkpoint "
                         "(learner, budget and switches are read from its config.json)")
    ap.add_argument("--algo", choices=tuple(ALGORITHMS), default="ppo",
                    help="learner: PPO, RecurrentPPO, TD3, SAC, TQC (the five baselines)")
    ap.add_argument("--gradient-steps", type=int, default=0,
                    help="TD3/SAC/TQC gradient steps per vectorised step (0 = num_envs)")
    ap.add_argument("--fixed-stage", type=int, default=0,
                    help="train on one scenario stage throughout instead of the curriculum")
    ap.add_argument("--config", type=Path, default=None,
                    help="F93: a frozen baseline (configs/baseline_v1.json) -- applies its run "
                         "arguments and refuses to start if the code no longer matches it")
    args = ap.parse_args()

    baseline = None
    if args.config:
        # F93: the campaign's learners differ in the learner and nothing else.  The
        # file sets every run argument; only --algo, --seed and --tag come from the CLI.
        import baseline_config
        with open(args.config) as fh:
            baseline = json.load(fh)
        if args.algo not in baseline["learners"]:
            raise SystemExit(f"{args.algo} is not a learner in {baseline['id']}")
        if args.fixed_stage or args.init_model or args.r2_slowdown_test:
            raise SystemExit("--config runs the frozen formulation; drop --fixed-stage, "
                             "--init-model and --r2-slowdown-test")
        problems = baseline_config.check(cfg, sys.modules[__name__], baseline)
        if problems:
            raise SystemExit(f"code does not match {baseline['id']}: " + "; ".join(problems))
        for key, value in baseline["run_args"].items():
            setattr(args, key, value)
        args.gradient_steps = 0              # num_envs, as recorded in the file
        if args.smoke:                       # a launch check, not a result: "_smoke" run dir
            args.timesteps, args.eval_freq, args.checkpoint_every = 4_096, 10**9, 10**9
            args.eval_per_class = 1
        print(f"[CONFIG] {baseline['id']} (formulation {baseline['formulation_digest']}): "
              f"{args.algo} seed {args.seed}", flush=True)

    global STAGE_SCHEDULE
    if args.fixed_stage:
        STAGE_SCHEDULE = ((0.0, int(args.fixed_stage)),)
    if args.torch_threads:
        import torch
        torch.set_num_threads(int(args.torch_threads))
    if args.r2_slowdown_test:
        cfg.R2_SLOWDOWN_TEST = args.r2_slowdown_test      # the evaluation process too

    resume_steps, resume_ckpt, resume_vecnorm, previous = None, None, None, None
    if args.resume:
        # F86: the run's own config decides what it is; the CLI cannot change it.
        run_dir = args.resume if args.resume.is_absolute() else Path.cwd() / args.resume
        with open(run_dir / "config.json") as fh:
            previous = json.load(fh)
        for key in ("algo", "seed", "timesteps", "num_envs", "train_supervisor",
                    "eval_supervisor", "low_speed_start_frac", "eval_freq", "eval_per_class"):
            if key in previous:
                setattr(args, key, previous[key])
        mismatched = [f"{k}: run {previous[k]!r}, code {v!r}"
                      for k, v in _formulation_switches().items()
                      if k in previous.get("switches", {}) and previous["switches"][k] != v]
        if previous.get("observation_schema") not in (None, cfg.OBSERVATION_SCHEMA_VERSION):
            mismatched.append(f"observation_schema: run {previous['observation_schema']!r}")
        if mismatched:
            raise SystemExit("cannot resume -- the code no longer matches the run: " + "; ".join(mismatched))
        if "switches" not in previous:
            print("[RESUME] this run predates recorded switches; check them by hand: "
                  f"{_formulation_switches()}", flush=True)
        prefix = f"{args.algo}_"
        steps = sorted(int(p.name[len(prefix):-len("_steps.zip")])
                       for p in run_dir.glob(f"{prefix}*_steps.zip")
                       if p.name[len(prefix):-len("_steps.zip")].isdigit())
        if not steps:
            raise SystemExit(f"no {prefix}*_steps.zip checkpoint in {run_dir}")
        resume_steps = steps[-1]
        resume_ckpt = run_dir / f"{prefix}{resume_steps}_steps.zip"
        resume_vecnorm = run_dir / f"{prefix}vecnormalize_{resume_steps}_steps.pkl"
        print(f"[RESUME] {run_dir.name} from {resume_steps:,} of {args.timesteps:,} steps", flush=True)
    else:
        run_dir = args.runs_dir / (f"{args.algo}_formulation_seed{args.seed}"
                                   + (f"_{args.tag}" if args.tag else "")
                                   + ("_smoke" if args.smoke else ""))
    if args.algo in ("ppo", "recurrent_ppo"):
        hyperparameters = dict(PPO_HYPERPARAMS)
    elif args.algo == "td3":
        hyperparameters = dict(TD3_HYPERPARAMS,
                               gradient_steps=int(args.gradient_steps or args.num_envs))
        if args.smoke:
            hyperparameters.update(learning_starts=256, buffer_size=20_000)
    else:
        hyperparameters = dict(SAC_HYPERPARAMS,
                               gradient_steps=int(args.gradient_steps or args.num_envs))
        if args.smoke:
            hyperparameters.update(learning_starts=256, buffer_size=20_000)
        if args.algo == "tqc":
            hyperparameters.update(TQC_HYPERPARAMS)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "algorithm": ALGORITHMS[args.algo].__name__, "seed": args.seed, "timesteps": args.timesteps,
        "algo": args.algo, "num_envs": args.num_envs, "hyperparameters": hyperparameters,
        "torch_threads": args.torch_threads, "fixed_stage": args.fixed_stage,
        "train_supervisor": args.train_supervisor, "eval_supervisor": args.eval_supervisor,
        "low_speed_start_frac": args.low_speed_start_frac,
        "eval_freq": args.eval_freq, "eval_per_class": args.eval_per_class,
        "r2_slowdown_test": cfg.R2_SLOWDOWN_TEST,
        # F86: the formulation switches, so a resume can refuse changed code.
        "switches": _formulation_switches(),
        "platform": _platform(),
        "baseline_config": (None if baseline is None else
                            {"id": baseline["id"], "path": str(args.config),
                             "formulation_digest": baseline["formulation_digest"]}),
        # A25: 70-value schema with the encounter-context branch.  A checkpoint
        # from an earlier schema cannot be resumed; the run records its own.
        "observation_schema": cfg.OBSERVATION_SCHEMA_VERSION,
        "observation_dim": int(sum(int(np.prod(space.shape))
                                   for space in observation_space().spaces.values())),
        "init_model": str(args.init_model) if args.init_model else None,
        "policy": {"features_extractor": "ASVFeaturesExtractor", "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                   "activation": "ReLU"},
        "reward_normalisation": "VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10)",
        "scenario_schedule": STAGE_SCHEDULE, "propulsion_stage": PROPULSION_STAGE,
        "rpm_stage_params": curriculum.stage_params(PROPULSION_STAGE),
        "constants": {"UPDATE_RATE": cfg.UPDATE_RATE, "CRUISE_RPM": cfg.CRUISE_RPM, "U_NOM": cfg.U_NOM,
                      "MAX_EPISODE_STEPS": cfg.MAX_EPISODE_STEPS,
                      "VESSEL_RANDOMISATION_SCALE": cfg.VESSEL_RANDOMISATION_SCALE,
                      "BOUNDARY_POSE_NOISE_XY": cfg.BOUNDARY_POSE_NOISE_XY,
                      "BOUNDARY_POSE_NOISE_HEADING_DEG": cfg.BOUNDARY_POSE_NOISE_HEADING_DEG,
                      "EGO_SPEED_NOISE": cfg.EGO_SPEED_NOISE, "ESTOP_TRIGGER": cfg.ESTOP_TRIGGER,
                      "R_ESTOP": cfg.R_ESTOP, "RUDDER_COMMAND_LIMIT": cfg.RUDDER_COMMAND_LIMIT,
                      "MOTION_CLASSIFIER": cfg.MOTION_CLASSIFIER,
                      "PF_OVERSPEED_TOL": cfg.PF_OVERSPEED_TOL,
                      "PF_OVERSPEED_SPAN": cfg.PF_OVERSPEED_SPAN},
    }
    if previous is not None:
        config = dict(previous)
        config.setdefault("resumes", []).append(
            {"from_steps": resume_steps, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "replay_buffer": "restored" if args.algo in OFF_POLICY and
             (run_dir / f"{args.algo}_replay_buffer_{resume_steps}_steps.pkl").exists() else "n/a"})
    with open(run_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=1, default=str)

    base_seed = 100_000 * (args.seed + 1)
    vec = SubprocVecEnv([make_env(i, base_seed, stage_at(0.0), cfg.VESSEL_RANDOMISATION_SCALE,
                                  1 if args.torch_threads else 0,
                                  supervisor=args.train_supervisor == "on",
                                  low_speed_start_frac=args.low_speed_start_frac,
                                  r2_slowdown_test=args.r2_slowdown_test)
                         for i in range(args.num_envs)])
    # SB3 appends ".monitor.csv" unless the name already ends that way.
    monitor = "monitor.csv" if resume_steps is None else f"resume_{resume_steps}.monitor.csv"
    vec = RetryingVecMonitor(vec, filename=str(run_dir / monitor),
                     info_keywords=("reached_goal", "collided", "scenario_class"))
    if resume_vecnorm is not None:
        vec = VecNormalize.load(str(resume_vecnorm), vec)
        vec.training, vec.norm_reward = True, True
    elif args.init_vecnormalize:
        vec = VecNormalize.load(str(args.init_vecnormalize), vec)
        vec.training, vec.norm_reward = True, True
    else:
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0,
                           gamma=hyperparameters["gamma"])

    if args.algo != "ppo" and args.init_model:
        raise SystemExit("--init-model is PPO-only (Tier 2)")
    if resume_ckpt is not None:
        model = ALGORITHMS[args.algo].load(str(resume_ckpt), env=vec, device="cpu",
                                           tensorboard_log=str(args.runs_dir / "tensorboard"))
        buffer = run_dir / f"{args.algo}_replay_buffer_{resume_steps}_steps.pkl"
        if args.algo in OFF_POLICY and buffer.exists():
            model.load_replay_buffer(str(buffer))
    elif args.algo in OFF_POLICY:
        extra = {}
        if args.algo == "td3":
            extra["action_noise"] = NormalActionNoise(
                mean=np.zeros(2), sigma=TD3_ACTION_NOISE_SIGMA * np.ones(2))
        policy_kwargs = dict(features_extractor_class=ASVFeaturesExtractor,
                             net_arch=dict(pi=[256, 256], qf=[256, 256]),
                             activation_fn=nn.ReLU)
        if args.algo == "tqc":
            policy_kwargs.update(TQC_POLICY)
        model = ALGORITHMS[args.algo](
            "MultiInputPolicy", vec, verbose=1, seed=args.seed, device="cpu",
            tensorboard_log=str(args.runs_dir / "tensorboard"),
            policy_kwargs=policy_kwargs, **hyperparameters, **extra)
    elif args.algo == "recurrent_ppo":
        model = RecurrentPPO(
            "MultiInputLstmPolicy", vec, verbose=1, seed=args.seed, device="cpu",
            tensorboard_log=str(args.runs_dir / "tensorboard"),
            policy_kwargs=dict(features_extractor_class=ASVFeaturesExtractor,
                               net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               activation_fn=nn.ReLU, **RECURRENT_PPO_POLICY),
            **hyperparameters)
    elif args.init_model:
        # Tier 2: continue a trained policy on the current code.  The saved
        # hyperparameters are replaced by today's, so a fine-tune and a fresh
        # run differ only in their starting weights.
        model = PPO.load(str(args.init_model), env=vec, device="cpu", seed=args.seed,
                         tensorboard_log=str(args.runs_dir / "tensorboard"),
                         custom_objects={k: v for k, v in PPO_HYPERPARAMS.items()
                                         if k in ("learning_rate", "gamma", "gae_lambda",
                                                  "ent_coef", "vf_coef", "max_grad_norm",
                                                  "target_kl", "n_epochs", "batch_size")})
    else:
        model = PPO("MultiInputPolicy", vec, verbose=1, seed=args.seed, device="cpu",
                    tensorboard_log=str(args.runs_dir / "tensorboard"),
                    policy_kwargs=dict(features_extractor_class=ASVFeaturesExtractor,
                                       net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                       activation_fn=nn.ReLU),
                    **PPO_HYPERPARAMS)

    callbacks = CallbackList([
        ScenarioStageCallback(args.timesteps, str(run_dir / "curriculum.json"),
                              resume=resume_steps is not None),
        # Off-policy checkpoints keep the replay buffer, so a resume continues
        # learning from the same data rather than from an empty buffer.
        CheckpointCallback(save_freq=max(args.checkpoint_every // args.num_envs, 1), save_path=str(run_dir),
                           name_prefix=args.algo, save_vecnormalize=True,
                           save_replay_buffer=args.algo in OFF_POLICY),
        FormulationEvalCallback(run_dir, args.eval_freq, args.eval_per_class,
                                supervisor_modes=(("off", "on") if args.eval_supervisor == "both"
                                                  else (args.eval_supervisor,)),
                                resume_from=resume_steps),
    ])

    started = time.time()
    # A resume continues the step count (the curriculum and checkpoints read it)
    # and trains only the remainder of the original budget.
    model.learn(total_timesteps=(args.timesteps if resume_steps is None
                                 else args.timesteps - resume_steps),
                reset_num_timesteps=resume_steps is None, callback=callbacks,
                tb_log_name=run_dir.name, progress_bar=False)
    model.save(run_dir / "final_model.zip")
    vec.save(str(run_dir / "final_vecnormalize.pkl"))
    vec.close()
    config["wall_clock_s"] = time.time() - started
    with open(run_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=1, default=str)
    print(f"done in {config['wall_clock_s'] / 3600:.2f} h -> {run_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
