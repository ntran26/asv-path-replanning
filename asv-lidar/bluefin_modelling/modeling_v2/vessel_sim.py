"""`VesselSim` — the vessel model wrapped for RL training and evaluation.

The model alone is not enough for transfer: a policy also learns the *actuator
and timing environment* it is trained in. Every item below was a measured
mismatch between `udp_live_rl.py` (deployment) and the training simulator, and
each one is reproduced here so the policy meets the same plant in both places.

| parity item | deployment (measured in the July logs) | reproduced here |
|---|---|---|
| control interval | 2 Hz, dt = 0.500 s | `control_dt = 0.5` |
| command rate limit | bridge limits to 50 %/s before `$CMD` | `command_rate_pct_s = 50` |
| actuator delay | effective 0.73 s | inside the model (`rud_delay`) |
| observation staleness | the pose used by the controller is one frame old | `obs_delay_steps = 1` |
| rudder sign | `rudder_sign = -1` between action and `$CMD` | handled in `dynamics` |

The command rate limit matters more than it looks. `udp_live_rl.py` applies it
before transmitting, so the vessel never receives the bang-bang output the
policy produces. Train without it and the policy learns that full reversals are
free; deploy and they arrive as 2 s ramps. Turning it off is supported
(`command_rate_pct_s=None`) but should be a deliberate ablation, not a default.

Integration uses `sub_dt` substeps inside each control interval. The model is
timestep-consistent (see `acceptance.py`), so `sub_dt` is an accuracy setting,
not a behaviour setting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import dynamics as dyn
from ship_model_v3 import ShipModel, IDENTIFIED, sample_params

NOMINAL_RPM = 12.0
MAX_RUDDER_PCT = 100.0


@dataclass
class VesselState:
    x: float
    y: float
    heading_deg: float
    u: float
    v: float
    yaw_rate_degps: float
    speed: float
    rudder_deg: float
    t: float

    def as_dict(self) -> Dict[str, float]:
        return dict(x=self.x, y=self.y, heading_deg=self.heading_deg,
                    u=self.u, v=self.v, yaw_rate_degps=self.yaw_rate_degps,
                    speed=self.speed, rudder_deg=self.rudder_deg, t=self.t)


class VesselSim:
    """Vessel plant at the deployment control rate.

    Usage in an env step:

        sim = VesselSim()
        sim.reset(x=0.0, y=0.0, heading_deg=0.0)
        obs_state = sim.step(action)      # action in [-1, 1]
    """

    def __init__(self,
                 params: Optional[Dict[str, float]] = None,
                 control_dt: float = 0.5,
                 sub_dt: float = 0.05,
                 command_rate_pct_s: Optional[float] = 50.0,
                 obs_delay_steps: int = 1,
                 rpm: float = NOMINAL_RPM) -> None:
        self.params = dict(IDENTIFIED if params is None else params)
        self.control_dt = float(control_dt)
        self.sub_dt = float(sub_dt)
        self.command_rate_pct_s = command_rate_pct_s
        self.obs_delay_steps = int(obs_delay_steps)
        self.rpm = float(rpm)
        self._model = ShipModel(self.params)
        self.reset()

    # -- construction helpers ---------------------------------------------
    @classmethod
    def randomised(cls, rng=None, scale: float = 1.0, **kwargs) -> "VesselSim":
        """A simulator with parameters drawn for domain randomisation."""
        return cls(params=sample_params(rng, scale=scale), **kwargs)

    # -- lifecycle ---------------------------------------------------------
    def reset(self, x: float = 0.0, y: float = 0.0, heading_deg: float = 0.0,
              u: float = 0.0, rudder_pct: float = 0.0) -> VesselState:
        self._model.reset()
        self._model._s[0, 0] = float(u)
        self._model._s[3, 0] = np.deg2rad(float(heading_deg))
        self._model._s[5, 0] = float(x)
        self._model._s[6, 0] = float(y)
        self._last_cmd_pct = float(rudder_pct)
        self._t = 0.0
        st = self._state()
        self._obs_buf = deque([st] * (self.obs_delay_steps + 1),
                              maxlen=self.obs_delay_steps + 1)
        return self.observe()

    # -- stepping ----------------------------------------------------------
    def step(self, action: float, rpm: Optional[float] = None) -> VesselState:
        """Advance one control interval.

        `action` is the policy output in [-1, 1], mapped to rudder percent in
        the simulator convention (the same convention `ship_model.py` took).
        """
        cmd_pct = float(np.clip(action, -1.0, 1.0)) * MAX_RUDDER_PCT
        cmd_pct = self._apply_command_rate_limit(cmd_pct)
        self._last_cmd_pct = cmd_pct

        rpm = self.rpm if rpm is None else float(rpm)
        n_sub = max(1, int(round(self.control_dt / self.sub_dt)))
        dt = self.control_dt / n_sub
        for _ in range(n_sub):
            self._model.update(rpm, cmd_pct, dt)
        self._t += self.control_dt

        self._obs_buf.append(self._state())
        return self.observe()

    def _apply_command_rate_limit(self, cmd_pct: float) -> float:
        if self.command_rate_pct_s is None:
            return cmd_pct
        max_step = self.command_rate_pct_s * self.control_dt
        delta = np.clip(cmd_pct - self._last_cmd_pct, -max_step, max_step)
        return float(self._last_cmd_pct + delta)

    # -- state access ------------------------------------------------------
    def _state(self) -> VesselState:
        d = self._model.state_dict()
        return VesselState(x=d["x_m"], y=d["y_m"], heading_deg=d["heading_deg"],
                           u=d["u_body_mps"], v=d["v_body_mps"],
                           yaw_rate_degps=d["yaw_rate_degps"],
                           speed=d["speed_mps"], rudder_deg=d["rudder_deg"],
                           t=self._t)

    def observe(self) -> VesselState:
        """Pose as the controller sees it, one control step stale by default."""
        return self._obs_buf[0]

    def truth(self) -> VesselState:
        """Ground-truth pose, for reward shaping and logging — not for the policy."""
        return self._state()


def turning_circle(sim: VesselSim, helm: float = 1.0, duration: float = 60.0):
    """Hold helm and report the settled turn. Used by the acceptance tests."""
    sim.reset()
    # let the vessel reach cruise before applying helm
    for _ in range(int(20.0 / sim.control_dt)):
        sim.step(0.0)
    u0 = sim.truth().u
    for _ in range(int(duration / sim.control_dt)):
        sim.step(helm)
    st = sim.truth()
    r = abs(st.yaw_rate_degps)
    radius = st.u / np.deg2rad(r) if r > 1e-3 else float("inf")
    return dict(u0=u0, u=st.u, yaw_rate_degps=st.yaw_rate_degps,
                radius_m=radius, radius_L=radius / dyn.VESSEL_LENGTH,
                speed_ratio=st.u / max(u0, 1e-6))
