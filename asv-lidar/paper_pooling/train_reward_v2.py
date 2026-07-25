"""
train_reward_v2.py
==================

Runs the existing ``train_test_asv.py`` pipeline against the repaired-reward
environment ``rl_env_reward_v2.py`` instead of ``rl_env.py``.

``train_test_asv.py`` is NOT modified. This runner installs an import alias
(``sys.modules["rl_env"] -> rl_env_reward_v2``) before the trainer executes, so
the trainer's ``from rl_env import ...`` resolves to the v2 environment. All
CLI arguments are forwarded to the trainer unchanged.

Usage (short fine-tune from the protected 94% baseline -- the recommended path):

    python train_reward_v2.py \
        --mode train --algo sac \
        --resume --model-path models/sac_model_1M.zip \
        --timesteps 150000 --num-envs 8 --seed 675973 \
        --eval-freq 25000 --save-freq 50000

Note on --resume: train_test_asv.py loads AND saves to --model-path. To protect
the baseline, this runner refuses to let --model-path be sac_model_1M.zip on a
training run; pass --resume-from for the checkpoint to start from instead:

    python train_reward_v2.py --mode train --algo sac \
        --resume-from models/sac_model_1M.zip \
        --model-path models/sac_reward_v2.zip \
        --timesteps 150000 --num-envs 8 --seed 675973

Files the underlying trainer writes (be aware before running):
  * train_monitor.csv   -- OVERWRITTEN each training run
  * models/             -- checkpoints at --save-freq
  * sac_log/            -- TensorBoard events
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_ENV_MODULE = "rl_env_reward_v2"


def _resolve_env_module() -> str:
    """Pick the env module from --env-module, else $ASV_ENV_MODULE, else default.

    Resolved at MODULE level and mirrored into os.environ so that SubprocVecEnv
    workers (spawned with their own argv but an inherited environment) agree
    with the parent. The flag is stripped from sys.argv so the underlying
    trainer's argparse never sees it.
    """
    name = None
    if "--env-module" in sys.argv:
        i = sys.argv.index("--env-module")
        if i + 1 < len(sys.argv):
            name = sys.argv[i + 1]
            del sys.argv[i:i + 2]
    if name is None:
        for a in list(sys.argv):
            if a.startswith("--env-module="):
                name = a.split("=", 1)[1]
                sys.argv.remove(a)
                break
    if name is None:
        name = os.environ.get("ASV_ENV_MODULE", DEFAULT_ENV_MODULE)
    os.environ["ASV_ENV_MODULE"] = name
    return name


# Install the alias at MODULE level, not inside the __main__ guard.
# SubprocVecEnv workers on Windows re-import this file as "__mp_main__", so the
# alias must be established on plain import as well as on direct execution.
ENV_MODULE = _resolve_env_module()
_env_v2 = importlib.import_module(ENV_MODULE)  # noqa: E402

sys.modules["rl_env"] = _env_v2

TRAINER = os.path.join(_HERE, "train_test_asv.py")
PROTECTED = "sac_model_1M.zip"
DEFAULT_MODEL_PATH = "models/sac_reward_v2.zip"


def _argv_get(flag: str):
    """Return the value following `flag` in sys.argv, or None."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    for a in sys.argv:
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _prepare_argv() -> None:
    """Guard the protected baseline and seed a v2-specific default model path."""
    mode = _argv_get("--mode") or "test"
    model_path = _argv_get("--model-path")

    # --resume-from <ckpt>: copy the checkpoint to --model-path, then let the
    # trainer resume from that copy. Keeps the baseline read-only.
    resume_from = _argv_get("--resume-from")
    if resume_from is not None:
        i = sys.argv.index("--resume-from")
        del sys.argv[i:i + 2]
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
            sys.argv += ["--model-path", model_path]
        if not os.path.isfile(resume_from):
            sys.exit(f"[v2] --resume-from checkpoint not found: {resume_from}")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        if os.path.abspath(resume_from) != os.path.abspath(model_path):
            if os.path.exists(model_path):
                sys.exit(
                    f"[v2] refusing to overwrite existing {model_path} with a copy of "
                    f"{resume_from}. Choose a different --model-path or delete it first."
                )
            shutil.copyfile(resume_from, model_path)
            print(f"[v2] copied {resume_from} -> {model_path} (baseline left untouched)")
        if "--resume" not in sys.argv:
            sys.argv.append("--resume")

    elif model_path is None and mode == "train":
        model_path = DEFAULT_MODEL_PATH
        sys.argv += ["--model-path", model_path]
        print(f"[v2] no --model-path given, defaulting to {model_path}")

    if mode == "train" and model_path and os.path.basename(model_path) == PROTECTED:
        sys.exit(
            f"[v2] refusing to train into the protected baseline ({PROTECTED}). "
            f"Use --resume-from {model_path} --model-path {DEFAULT_MODEL_PATH}"
        )


if __name__ == "__main__":
    _prepare_argv()
    print(f"[v2] env module : {_env_v2.__name__} (OA_GAIN={_env_v2.OA_GAIN}, "
          f"floor={_env_v2.OA_DIST_FLOOR}, K_CTE_RECOVERY={_env_v2.K_CTE_RECOVERY}, "
          f"K_WRONG_SIDE_ACTION={_env_v2.K_WRONG_SIDE_ACTION})")
    if getattr(_env_v2, "BLOCK_USE_WIDE_ARC", False):
        print(f"[v2] wide-arc gate ON (BLOCK_ARC_DEG={_env_v2.BLOCK_ARC_DEG})")
    if (_argv_get("--mode") or "test") == "train":
        print("[v2] note: this run overwrites train_monitor.csv and writes to models/ and sac_log/")
    # Guard is essential: workers import this file as __mp_main__ and must NOT
    # re-enter the trainer.
    runpy.run_path(TRAINER, run_name="__main__")
