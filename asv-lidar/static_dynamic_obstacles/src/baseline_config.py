"""The frozen baseline formulation (F93): what every learner in the campaign trains on.

`configs/baseline_v1.json` records the formulation (every constant, the
formulation switches, observation schema and curriculum), the run arguments
and each learner's hyperparameters.  `train_formulation.py --config` applies the
run arguments and refuses to start if the code no longer matches the file, so
PPO, RecurrentPPO, TD3, SAC and TQC differ in the learner and nothing else.

    python src/baseline_config.py --check              # code vs the saved file
    python src/baseline_config.py --verify-run runs/ppo_formulation_seed0_v11
    python src/baseline_config.py --write              # re-freeze (a new version)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "baseline_v1.json"
CONFIG_ID = "baseline-v1"

# The run arguments of run 11, the formulation this freezes (F92, F93).  The CLI
# defaults differ (supervisor on in training, 6 episodes per class), which is
# why the campaign reads these from the file rather than from the defaults.
RUN_ARGS = {"timesteps": 2_000_000, "num_envs": 10, "eval_freq": 200_000,
            "eval_per_class": 20, "train_supervisor": "off", "eval_supervisor": "both",
            "low_speed_start_frac": 0.15, "checkpoint_every": 250_000}
ALGOS = ["ppo", "recurrent_ppo", "td3", "sac", "tqc"]
SEEDS = [0, 1, 2, 3, 4]            # A26 (your call, 2026-09-22): 5 seeds per learner
# A26: each seed is represented by its best development-set checkpoint (the eval
# callback's goal - 2 x collision score, supervisor off); Tier B stays held out.
CHECKPOINT = "best_model.zip"


def _plain(value):
    """JSON-normalised copy, so tuples, numpy scalars and lists compare equal."""
    return json.loads(json.dumps(value, default=str, sort_keys=True))


def _constants(cfg: ModuleType) -> Dict:
    """Every upper-case setting in `constants.py`: a changed number anywhere in the
    formulation is a mismatch; a changed comment is not.  Read from a fresh load
    of the file, not the live module: `curriculum.apply_stage` rewrites the RPM
    settings in memory (propulsion stage 4, recorded separately), so a check made
    after staging would otherwise report a drift the source does not have."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_constants_as_written", cfg.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    out = {}
    for name, value in vars(fresh).items():
        if not name.isupper() or callable(value) or isinstance(value, ModuleType):
            continue
        out[name] = value
    return _plain(out)


def formulation(cfg: ModuleType, tf: ModuleType) -> Dict:
    return _plain({"switches": tf._formulation_switches(),
                   "observation_schema": cfg.OBSERVATION_SCHEMA_VERSION,
                   "curriculum_schedule": tf.STAGE_SCHEDULE,
                   "propulsion_stage": tf.PROPULSION_STAGE,
                   "constants": _constants(cfg)})


def learners(tf: ModuleType, num_envs: int) -> Dict:
    on_policy_arch = {"pi": [256, 256], "vf": [256, 256]}
    off_policy_arch = {"pi": [256, 256], "qf": [256, 256]}
    grad = {"gradient_steps": int(num_envs)}
    return _plain({
        "ppo": {"hyperparameters": tf.PPO_HYPERPARAMS, "net_arch": on_policy_arch},
        "recurrent_ppo": {"hyperparameters": tf.PPO_HYPERPARAMS, "net_arch": on_policy_arch,
                          "policy": tf.RECURRENT_PPO_POLICY},
        "td3": {"hyperparameters": dict(tf.TD3_HYPERPARAMS, **grad), "net_arch": off_policy_arch,
                "action_noise_sigma": tf.TD3_ACTION_NOISE_SIGMA},
        "sac": {"hyperparameters": dict(tf.SAC_HYPERPARAMS, **grad), "net_arch": off_policy_arch},
        "tqc": {"hyperparameters": dict(tf.SAC_HYPERPARAMS, **grad, **tf.TQC_HYPERPARAMS),
                "net_arch": off_policy_arch, "policy": tf.TQC_POLICY},
    })


def snapshot(cfg: ModuleType, tf: ModuleType) -> Dict:
    """Everything the code decides; `--check` compares exactly this."""
    return {"formulation": formulation(cfg, tf), "run_args": _plain(RUN_ARGS),
            "learners": learners(tf, RUN_ARGS["num_envs"]),
            "campaign": {"algos": ALGOS, "seeds": SEEDS, "checkpoint": CHECKPOINT,
                         "off_policy_gradient_steps_per_transition": 1.0}}


def _git(*cmd) -> str:
    try:
        return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _code_dirty() -> bool:
    """Uncommitted source changes -- not this file, and not the tracked bytecode
    that every Python run rewrites."""
    lines = _git("status", "--porcelain", "--", "src").splitlines()
    return any("__pycache__" not in line for line in lines)


def digest(block: Dict) -> str:
    return hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()[:16]


def diff(saved: Dict, now: Dict, prefix: str = "") -> List[str]:
    out = []
    for key in sorted(set(saved) | set(now)):
        a, b = saved.get(key, "<absent>"), now.get(key, "<absent>")
        if isinstance(a, dict) and isinstance(b, dict):
            out += diff(a, b, f"{prefix}{key}.")
        elif a != b:
            out.append(f"{prefix}{key}: saved {a!r}, code {b!r}")
    return out


def load() -> Dict:
    with open(CONFIG_PATH) as fh:
        return json.load(fh)


def check(cfg: ModuleType, tf: ModuleType, saved: Dict | None = None) -> List[str]:
    saved = saved if saved is not None else load()
    now = snapshot(cfg, tf)
    return diff({k: saved[k] for k in now}, now)


def verify_run(run_dir: Path, saved: Dict) -> List[str]:
    """A finished run's own record against the baseline: switches, schema,
    hyperparameters and run arguments.  Switches the run predates must be off."""
    with open(run_dir / "config.json") as fh:
        run = json.load(fh)
    problems = []
    switches = saved["formulation"]["switches"]
    recorded = run.get("switches", {})
    for key, value in switches.items():
        have = recorded.get(key, None)
        if key not in recorded and value in (False, None):
            continue                     # predates the switch; it was off by construction
        if _plain(have) != value:
            problems.append(f"switch {key}: run {have!r}, baseline {value!r}")
    if run.get("observation_schema") != saved["formulation"]["observation_schema"]:
        problems.append(f"observation_schema: run {run.get('observation_schema')!r}")
    learner = saved["learners"][run["algo"]]["hyperparameters"]
    if _plain(run.get("hyperparameters")) != learner:
        problems.append(f"hyperparameters differ: {diff(learner, _plain(run['hyperparameters']))}")
    for key, value in saved["run_args"].items():
        if key in run and _plain(run[key]) != value:
            problems.append(f"run arg {key}: run {run[key]!r}, baseline {value!r}")
    if _plain(run.get("scenario_schedule")) != saved["formulation"]["curriculum_schedule"]:
        problems.append("curriculum schedule differs")
    return problems


def write(cfg: ModuleType, tf: ModuleType) -> Dict:
    body = snapshot(cfg, tf)
    config = {
        "id": CONFIG_ID,
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": {"head": _git("rev-parse", "HEAD"),
                "dirty": _code_dirty()},
        "provenance": {
            "formulation_of": "run 11 (runs/ppo_formulation_seed{0,1}_v11)",
            "why": "PROJECT_STATE.md F92/F93: best on every development-set class and "
                   "consistent across two seeds; run 12's F91 and A31 are switched off",
            "reproduces": ["runs/ppo_formulation_seed0_v11", "runs/ppo_formulation_seed1_v11"],
            "known_limits": [
                "crossing opening direction is not side-conditioned (A32)",
                "narrow-channel head-ons where starboard has no room: 0.36 collisions (F91, A33)",
                "being overtaken by a vessel that never gives way (A30, on hold)",
            ],
        },
        "formulation_digest": digest(body["formulation"]),
        **body,
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", newline="\n") as fh:
        json.dump(config, fh, indent=1, sort_keys=False)
        fh.write("\n")
    return config


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    import constants as cfg
    import train_formulation as tf
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--verify-run", type=Path, nargs="+")
    args = ap.parse_args()
    if args.write:
        config = write(cfg, tf)
        print(f"wrote {CONFIG_PATH.relative_to(ROOT)} ({config['id']}, "
              f"formulation {config['formulation_digest']})")
        return 0
    if args.check:
        problems = check(cfg, tf)
        print("code matches " + CONFIG_ID if not problems else "\n".join(problems))
        return 1 if problems else 0
    saved, bad = load(), 0
    for run_dir in args.verify_run:
        problems = verify_run(run_dir, saved)
        bad += bool(problems)
        print(f"{run_dir.name}: " + ("matches " + CONFIG_ID if not problems
                                     else "\n  " + "\n  ".join(problems)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
