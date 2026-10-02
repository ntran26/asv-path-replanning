"""Braking-envelope prediction for the experimental v4 safety filter.

The weak/delayed reverse assumption is conservative for stopping distance but
not for the complete trajectory: stronger reverse can remove surge and rudder
authority while yaw persists. Check both complete trajectories for every plan
that contains astern. Nonbraking plans use only the original prediction.

This two-scenario empirical envelope uses nominal onboard model parameters.
It is not a bound on every possible reverse response or a safety guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from classical import common as cc
import safety_v2 as v2
import ship


def rollout_seq(snap, actuators, sequences, *, brake_efficiency=None, brake_delay_s=None):
    """v3's sequence rollout with explicit reverse efficiency and onset delay.

    Sequences have shape ``(candidates, decisions, 2)``. NaN throttle requests
    full astern. Defaults reproduce the original weak/delayed predictor.
    """
    efficiency = v2.BRAKE_EFFICIENCY if brake_efficiency is None else float(brake_efficiency)
    delay = v2.BRAKE_DELAY_S if brake_delay_s is None else float(brake_delay_s)
    sequences = np.asarray(sequences, dtype=float)
    n, decisions = sequences.shape[:2]
    state = np.zeros((7, n))
    state[:4] = np.array([max(0.0, snap.u), snap.v, snap.r, snap.heading])[:, None]
    state[4] = actuators.servo
    state[5], state[6] = snap.x, snap.y
    params = {key: np.full(n, value) for key, value in cc.IDENTIFIED.items()}
    pending = ([np.full(n, value) for value in actuators.buffer]
               if actuators.buffer is not None else None)
    samples = decisions * cc.SUBSTEPS
    positions, headings = np.empty((samples, n, 2)), np.empty((samples, n))
    speeds = np.empty((samples, n))
    deceleration = ship.braking_thrust(v2.ASTERN_RPM, efficiency=efficiency) / ship.M11
    brake_since = np.full(n, np.inf)
    sample = 0
    for decision in range(decisions):
        control = sequences[:, decision]
        rpm = v2._rpm(control[:, 1])
        rudder = -cc.MAX_RUDDER_RAD * control[:, 0]
        braking = rpm < 0.0
        if pending is None:
            pending = [rudder.copy() for _ in range(cc.DELAY_STEPS)]
        for _ in range(cc.SUBSTEPS):
            pending.append(rudder)
            state = cc.dyn.rk4_step(state, np.maximum(rpm, 0.0), pending.pop(0), params, cc.PRED_DT)
            now = (sample + 1) * cc.PRED_DT
            brake_since = np.where(braking, np.minimum(brake_since, now), np.inf)
            active = braking & (now - brake_since >= delay)
            if active.any():
                state[0] = np.where(active, np.maximum(0.0, state[0] - deceleration * cc.PRED_DT), state[0])
            positions[sample] = state[5:7].T
            headings[sample] = state[3]
            speeds[sample] = np.hypot(state[0], state[1])
            sample += 1
    return cc.Rollout(positions, headings, speeds, cc.PRED_DT * np.arange(1, samples + 1))


@dataclass
class EnvelopeEvaluation:
    """Candidate-aligned results; nonbraking fast-branch entries are +inf."""

    first: np.ndarray
    clear: np.ndarray
    weak_first: np.ndarray
    weak_clear: np.ndarray
    fast_first: np.ndarray
    fast_clear: np.ndarray
    braking_mask: np.ndarray

    @property
    def dual_count(self):
        return int(self.braking_mask.sum())


def evaluate_sequences(snap, actuators, sequences, evaluate):
    """Evaluate both reverse scenarios, preserving each candidate's index.

    ``evaluate(snap, rollout)`` must return time of first violation (infinity
    when safe) and minimum clearance per candidate, as v2/v3 ``_evaluate`` do.
    A candidate passes only if both complete trajectories pass. The same rule
    applies to a stored continuation containing any future brake command.
    """
    sequences = np.asarray(sequences, dtype=float)
    braking = np.isnan(sequences[:, :, 1]).any(axis=1)
    weak_first, weak_clear = evaluate(snap, rollout_seq(snap, actuators, sequences))
    weak_first = np.asarray(weak_first, dtype=float)
    weak_clear = np.asarray(weak_clear, dtype=float)
    fast_first, fast_clear = np.full(len(sequences), np.inf), np.full(len(sequences), np.inf)
    if braking.any():
        fast = rollout_seq(snap, actuators, sequences[braking],
                           brake_efficiency=ship.REVERSE_THRUST_EFFICIENCY, brake_delay_s=0.0)
        fast_first[braking], fast_clear[braking] = evaluate(snap, fast)
    return EnvelopeEvaluation(np.minimum(weak_first, fast_first),
                              np.minimum(weak_clear, fast_clear),
                              weak_first, weak_clear, fast_first, fast_clear, braking)
