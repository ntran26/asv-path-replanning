"""Model-predicted ego prior with the safety filter's existing sensor weight.

The model advances the previous estimate using the issued command and the
controller's estimated rudder state. A fresh measurement corrects that prior
with the existing EGO_SMOOTHING weight, avoiding the stationary EMA's lag after
turning or braking. All inputs are onboard quantities; no simulator state is
read. The immediate nominal braking law is a state-estimation prior only and
does not change the safety predictor's separate braking envelope.
"""
from __future__ import annotations

import numpy as np

from classical import common as cc
import safety_v2 as v2
import ship


def advance_ego(snap, pre_command_actuators, actual_rudder, rpm):
    """Predict body ``[u, v, r]`` one decision after the issued command.

    ``actual_rudder`` is a normalised issued command, after any bridge limiter;
    ``rpm`` is signed issued propulsion, so zero and astern remain distinct.
    Pass actuator history from BEFORE its ``issue`` call for this command.
    Neither snapshot nor actuator history is modified.
    """
    state = np.zeros((7, 1))
    state[:4, 0] = (max(0.0, snap.u), snap.v, snap.r, snap.heading)
    state[4, 0] = pre_command_actuators.servo
    state[5:, 0] = (snap.x, snap.y)
    params = {key: np.array([value]) for key, value in cc.IDENTIFIED.items()}
    delta = np.array([-cc.MAX_RUDDER_RAD * float(actual_rudder)])
    pending = ([np.array([value]) for value in pre_command_actuators.buffer]
               if pre_command_actuators.buffer is not None
               else [delta.copy() for _ in range(cc.DELAY_STEPS)])
    rpm = float(rpm)
    thrust = np.array([max(0.0, rpm)])
    deceleration = ship.braking_thrust(rpm) / ship.M11
    for _ in range(cc.SUBSTEPS):
        pending.append(delta)
        state = cc.dyn.rk4_step(state, thrust, pending.pop(0), params, cc.PRED_DT)
        if deceleration > 0.0:
            state[0, 0] = max(0.0, state[0, 0] - deceleration * cc.PRED_DT)
    return state[:3, 0].copy()


class EgoObserver:
    """Alternating ``update`` / ``predict`` observer with no new tuned gain.

    ``update(raw, fresh=False)`` skips correction for a held sensor frame. A
    model prior still advances on every issued command. On the first call,
    even a held reading is the only available initial estimate. ``reset`` must
    be called when starting a new episode or replacing the vessel state.
    """

    def __init__(self):
        self.weight = float(v2.EGO_SMOOTHING)
        self.estimate = None
        self.prior = None

    def reset(self):
        self.estimate, self.prior = None, None

    def update(self, raw, *, fresh=True):
        raw = np.asarray(raw, dtype=float).reshape(3)
        if not np.isfinite(raw).all():
            raise ValueError("ego measurements must be finite [u, v, r]")
        if self.estimate is None:
            self.estimate = raw.copy()
        else:
            prior = self.estimate if self.prior is None else self.prior
            self.estimate = (prior + self.weight * (raw - prior)
                             if fresh else prior.copy())
        self.prior = None
        return self.estimate.copy()

    def predict(self, snap, pre_command_actuators, actual_rudder, rpm):
        if self.estimate is None:
            raise RuntimeError("call update before predicting the next ego state")
        # The snapshot must carry the estimate used by this decision's safety
        # check. No second correction or rudder-limit application belongs here.
        self.prior = advance_ego(snap, pre_command_actuators, actual_rudder, rpm)
        return self.prior.copy()
