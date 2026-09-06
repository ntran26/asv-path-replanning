"""Explicit replication of the staged propulsion curriculum.

Why this file exists
--------------------
The published SAC run was trained under a speed-authority curriculum, but **no
scheduler for it exists anywhere in the code**.  `config.RPM_STAGE` is an
import-time constant and nothing assigns to it.  The curriculum was run by
hand-editing the constant and restarting training with `--resume`; the evidence
is in `plotting/plot_training_curves.py:96` (which documents the phase
boundaries), the six separate tfevents files in `sac_log/asv_sac_2/`, and the
`--resume` / `--replay-buffer-path` flags built for exactly that workflow.

For the PPO baseline to run under "identical conditions" the same schedule has
to be reproduced, and -- per the task brief -- scheduled against **total
environment interactions**, not per-env steps.  `model.num_timesteps` is exactly
that: SB3 increments it by `n_envs` per rollout step, so the boundaries land at
the same total interaction counts regardless of how many workers are used.

How it works without touching `env.py`
--------------------------------------
`ASVLidarEnv.step` reads `cfg.FIXED_RPM`, `cfg.RPM_DELTA`, `cfg.RPM_FLOOR` and
`cfg.RPM_CEIL` as module attributes **at call time**, so rebinding them on the
`config` module takes effect on the next step.  `CurriculumASVLidarEnv` is a
subclass that does the rebinding inside whichever process it lives in, driven
over `SubprocVecEnv.env_method`.

Nothing in `env.py`, `config.py` or `train.py` is modified.  The SAC training
and evaluation paths never construct this subclass and are unaffected.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from stable_baselines3.common.callbacks import BaseCallback

import constants as cfg
from env import ASVLidarEnv

# ---------------------------------------------------------------------------
# CORRECTED 2026-08-19.  Read this before using STAGED_SCHEDULE.
#
# `PUBLISHED_SCHEDULE` is what actually produced `models/sac_model_1M.zip`:
# fixed RPM for the whole 1M-step run, no propulsion curriculum at all.
#
# Evidence, from the published run's own TensorBoard log
# (`sac_log/asv_sac_2/events.out.tfevents.1781347673.*`): at every one of the 19
# evaluations from 50k to 950k, `eval/min_rpm == eval/mean_rpm ==
# eval/max_rpm == 12.000`.  Throttle was never active.  Had any stage fired at
# 700k/800k/900k, min and max would have separated immediately.
#
# The staged schedule appears only in the *resumed* runs after 1M steps
# (1.00M-1.40M, 1.05M-1.45M, ...), where throttle is active and eval RPM does
# vary.  Those resumed runs reached 0.617-0.750 eval-grid success, i.e. worse
# than the 1M checkpoint they continued from.
#
# The stage markers in `plotting/plot_training_curves.py:96` therefore do not
# describe the run that produced the published policy.  An earlier version of
# this file took them at face value and scheduled stages inside the 1M budget;
# that was wrong, and it handicapped both retrained SAC and PPO relative to the
# published setup.  See BASELINES_RESULTS.md.
#
# Note the train/eval asymmetry this implies, which is a property of the
# original work and is replicated deliberately rather than corrected: the
# published policy was TRAINED with throttle inert (RPM pinned at CRUISE_RPM)
# but is EVALUATED at RPM_STAGE = 1 with throttle live.
# ---------------------------------------------------------------------------

# Fixed RPM for the entire run.  This is the published SAC setup.
PUBLISHED_SCHEDULE: Tuple[Tuple[int, int], ...] = (
    (0, 0),
)

# The staged schedule as read from plot_training_curves.py.  Retained because
# the post-1M resumed runs did use something like it, but it is NOT the setup
# behind the published checkpoint and must not be used to replicate it.
STAGED_SCHEDULE: Tuple[Tuple[int, int], ...] = (
    (0,       0),   # "cruise": FIXED_RPM, throttle ignored, RPM held at CRUISE_RPM
    (700_000, 1),
    (800_000, 2),
    (900_000, 3),
)

# Default for every training script in this study.
CURRICULUM_SCHEDULE: Tuple[Tuple[int, int], ...] = PUBLISHED_SCHEDULE

# Stage 0 is not in cfg.RPM_STAGES -- it is the fixed-speed phase.  The delta is
# carried over from stage 1 purely so that the r_thrust term's division by
# RPM_DELTA stays well defined; with FIXED_RPM the numerator is identically zero,
# so r_thrust is 0.0 throughout the phase regardless of the value used.
STAGE_0 = (3.0, 9.0, 15.0)


def stage_params(stage: int) -> Tuple[float, float, float]:
    """(RPM_DELTA, RPM_FLOOR, RPM_CEIL) for a curriculum stage."""
    if stage == 0:
        return STAGE_0
    return cfg.RPM_STAGES[int(stage)]


def stage_for_timestep(total_steps: int, schedule=None) -> int:
    """The stage that should be active at a given total environment step count."""
    schedule = CURRICULUM_SCHEDULE if schedule is None else tuple(schedule)
    stage = schedule[0][1]
    for boundary, value in schedule:
        if total_steps >= boundary:
            stage = value
    return stage


def apply_stage(stage: int) -> Dict[str, float]:
    """Rebind the propulsion constants on the `config` module, in this process."""
    stage = int(stage)
    delta, floor, ceil = stage_params(stage)

    cfg.RPM_STAGE = stage
    cfg.FIXED_RPM = (stage == 0)
    cfg.RPM_DELTA = delta
    cfg.RPM_FLOOR = floor
    cfg.RPM_CEIL = ceil

    return {"stage": stage, "fixed_rpm": cfg.FIXED_RPM,
            "delta": delta, "floor": floor, "ceil": ceil}


class CurriculumASVLidarEnv(ASVLidarEnv):
    """`ASVLidarEnv` that can be told which propulsion stage to run at.

    Behaviourally identical to the base env at any fixed stage -- the subclass
    adds a setter and nothing else.  It overrides no dynamics, no observation,
    no reward and no termination logic.
    """

    def set_rpm_stage(self, stage: int) -> Dict[str, float]:
        """Rebind the propulsion constants for this env's process."""
        return apply_stage(stage)

    def get_rpm_stage(self) -> Dict[str, float]:
        return {"stage": int(cfg.RPM_STAGE), "fixed_rpm": bool(cfg.FIXED_RPM),
                "delta": float(cfg.RPM_DELTA), "floor": float(cfg.RPM_FLOOR),
                "ceil": float(cfg.RPM_CEIL)}


class RpmCurriculumCallback(BaseCallback):
    """Advance the propulsion stage against total environment interactions.

    Broadcasts to every worker in the vectorised training env, and applies the
    same change in the local process so that a single-process evaluation env
    stays in step with training.
    """

    def __init__(self, schedule=None, *, verbose: int = 1,
                 log_path: Optional[str] = None):
        super().__init__(verbose)
        self.schedule = CURRICULUM_SCHEDULE if schedule is None else tuple(schedule)
        self.log_path = log_path
        self.current_stage: Optional[int] = None
        self.transitions = []

    def _set_stage(self, stage: int) -> None:
        # Workers first, then this process (the eval env lives here).
        try:
            self.training_env.env_method("set_rpm_stage", int(stage))
        except AttributeError:
            # Not a vec env -- a bare env still exposes the setter.
            self.training_env.set_rpm_stage(int(stage))
        params = apply_stage(stage)

        self.current_stage = int(stage)
        self.transitions.append({"timesteps": int(self.num_timesteps), **params})
        if self.verbose:
            print(f"[CURRICULUM] t={self.num_timesteps:,} -> stage {stage} "
                  f"(fixed_rpm={params['fixed_rpm']}, delta={params['delta']}, "
                  f"floor={params['floor']}, ceil={params['ceil']})")
        if self.log_path:
            import json
            with open(self.log_path, "w") as f:
                json.dump(self.transitions, f, indent=2)

    def _on_training_start(self) -> None:
        self._set_stage(stage_for_timestep(self.num_timesteps, self.schedule))

    def _on_step(self) -> bool:
        wanted = stage_for_timestep(self.num_timesteps, self.schedule)
        if wanted != self.current_stage:
            self._set_stage(wanted)
        return True
