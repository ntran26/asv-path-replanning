"""Sampled feedback backups for the safety filter's existing predictor.

Method references:
* Eriksen et al. (2019), BC-MPC, section 3.1: predict trajectories under
  feedback course/speed control. https://arxiv.org/pdf/1907.00039
  Published version: https://doi.org/10.1002/rob.21900
* Chen et al. (2021), Backup Control Barrier Functions, sections III-IV:
  forward-integrate a feedback backup controller and assess its trajectory.
  https://arxiv.org/pdf/2104.11332

This module adapts feedback trajectory generation only. It is neither the full
BC-MPC algorithm nor a certified backup CBF: no invariant terminal set is proved.
The four sampled headings and +/-30 degree offsets are engineering choices,
not formulas or parameters taken from either paper. Existing course-controller
gains, hull dynamics, delay, weak braking model and prediction horizon are used.
"""
from __future__ import annotations

import math

import numpy as np

import constants as cfg
from classical import common as cc
import safety_v2 as v2
import ship


COURSE_NAMES = ("edge_parallel", "current_heading", "port_30", "starboard_30")
COURSE_OFFSET_DEG = 30.0


def desired_courses(snap):
    """Four compass-radian headings, using only the snapshot's known map.

    The nearest edge is chosen by centre-to-segment distance. Its direction is
    oriented closest to the path's forward heading, without commanding a return
    to the path's position. Ties retain map-edge order. Missing/degenerate map
    edges fall back to the path's forward heading for the first sample.
    """
    parallel = float(snap.base_heading)
    if snap.edges_a is not None and snap.edges_b is not None:
        starts = np.asarray(snap.edges_a, dtype=float)
        ends = np.asarray(snap.edges_b, dtype=float)
        if len(starts):
            edges = ends - starts
            length2 = np.sum(edges * edges, axis=1)
            valid = length2 > 1e-12
            if valid.any():
                starts, edges, length2 = starts[valid], edges[valid], length2[valid]
                relative = snap.position - starts
                along = np.clip(np.sum(relative * edges, axis=1) / length2, 0.0, 1.0)
                residual = relative - along[:, None] * edges
                edge = edges[int(np.argmin(np.sum(residual * residual, axis=1)))]
                heading = math.atan2(float(edge[0]), float(edge[1]))
                alternate = heading + math.pi
                parallel = (heading if abs(float(cc.wrap_pi(heading - snap.base_heading)))
                            <= abs(float(cc.wrap_pi(alternate - snap.base_heading))) else alternate)
    offset = math.radians(COURSE_OFFSET_DEG)
    return cc.wrap_pi(np.array([parallel, snap.heading, snap.heading - offset, snap.heading + offset]))


def feedback_bank(snap, act, candidates, commit_s):
    """Return ``(rollout, sequences, names)`` in candidate-major/head-minor order.

    ``sequences`` has shape ``(len(candidates), 4, D, 2)``. Flatten its first two
    axes to replay the generated commands with ``safety_v3.rollout_seq``; the
    returned rollout already has those flattened columns. ``names`` labels its
    four heading samples in order. D is the existing safety horizon in decisions.

    Each candidate (including NaN-throttle astern) is held for the rounded
    commit interval. Thereafter, every decision recomputes the existing course
    feedback from its own predicted heading and yaw rate, with cruise throttle.
    Snapshot, actuator history and candidate inputs are never modified.
    """
    return _feedback_bank(snap, act, candidates, commit_s,
                          courses=desired_courses(snap), names=COURSE_NAMES,
                          rudder_feedback=_heading_rudder)


def _heading_rudder(target, state):
    return cc.course_rudder(cc.wrap_pi(target - state[3]), state[2])


def _feedback_bank(snap, act, candidates, commit_s, *, courses, names, rudder_feedback):
    """Shared predictor for named courses and a ``(target, state)`` rudder law.

    ``state`` is the existing seven-row model state. The feedback law returns
    one normalized rudder command per flattened candidate/course column;
    commitment, propulsion and all dynamics are identical to ``feedback_bank``.
    """
    candidates = np.asarray(candidates, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("candidates must have shape (count, 2)")
    if not np.isfinite(commit_s) or commit_s < 0.0:
        raise ValueError("commit_s must be finite and nonnegative")
    decisions = int(math.ceil(v2.HORIZON_S / cfg.UPDATE_RATE))
    commit = int(round(commit_s / cfg.UPDATE_RATE))
    courses, names = np.asarray(courses, dtype=float).reshape(-1), tuple(names)
    if len(courses) != len(names):
        raise ValueError("courses and names must have the same length")
    count, headings_count = len(candidates), len(names)
    total = count * headings_count
    sequences = np.empty((count, headings_count, decisions, 2))
    flat = sequences.reshape(total, decisions, 2)
    samples = decisions * cc.SUBSTEPS
    positions = np.empty((samples, total, 2))
    headings = np.empty((samples, total))
    speeds = np.empty((samples, total))
    times = cc.PRED_DT * np.arange(1, samples + 1)
    if not total:
        return cc.Rollout(positions, headings, speeds, times), sequences, names

    target = np.tile(courses, count)
    committed = np.repeat(candidates, headings_count, axis=0)
    state = np.zeros((7, total))
    state[:4] = np.array([max(0.0, snap.u), snap.v, snap.r, snap.heading])[:, None]
    state[4] = act.servo
    state[5], state[6] = snap.x, snap.y
    params = {key: np.full(total, value) for key, value in cc.IDENTIFIED.items()}
    pending = ([np.full(total, value) for value in act.buffer]
               if act.buffer is not None else None)
    deceleration = ship.braking_thrust(v2.ASTERN_RPM, efficiency=v2.BRAKE_EFFICIENCY) / ship.M11
    brake_since = np.full(total, np.inf)
    sample = 0
    for decision in range(decisions):
        if decision < commit:
            control = committed
        else:
            rudder = rudder_feedback(target, state)
            control = np.column_stack((rudder, np.zeros(total)))
        flat[:, decision] = control
        rpm = v2._rpm(control[:, 1])
        delta = -cc.MAX_RUDDER_RAD * control[:, 0]
        braking = rpm < 0.0
        if pending is None:
            pending = [delta.copy() for _ in range(cc.DELAY_STEPS)]
        for _ in range(cc.SUBSTEPS):
            pending.append(delta)
            state = cc.dyn.rk4_step(state, np.maximum(rpm, 0.0), pending.pop(0), params, cc.PRED_DT)
            now = (sample + 1) * cc.PRED_DT
            brake_since = np.where(braking, np.minimum(brake_since, now), np.inf)
            active = braking & (now - brake_since >= v2.BRAKE_DELAY_S)
            if active.any():
                state[0] = np.where(active, np.maximum(0.0, state[0] - deceleration * cc.PRED_DT), state[0])
            positions[sample] = state[5:7].T
            headings[sample] = state[3]
            speeds[sample] = np.hypot(state[0], state[1])
            sample += 1
    return cc.Rollout(positions, headings, speeds, times), sequences, names
