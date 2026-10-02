"""Finite-candidate recovery scores inspired by predictive barrier functions.

Wabersich and Zeilinger, "Predictive control barrier functions: Enhanced safety
mechanisms for learning-based control", Eq. (9), first minimise nonnegative
constraint slacks before choosing an input close to the performance controller:
https://arxiv.org/pdf/2105.10241

Here each complete candidate sequence is scored by integrated static, boundary,
and individual-target clearance deficits. The terminal run-out contributes one
decision's deficit. This is a discrete library approximation, without their
tightening, terminal CBF, optimisation, or recovery/invariance guarantees.
Hard first-violation times and margins reproduce v2 exactly. The soft boundary
score remains negative outside the map, preventing a fast exit from appearing
safe again under the inherited unsigned edge-distance approximation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import constants as cfg
from classical import common as cc
import safety_v2 as v2


@dataclass
class RecoveryMetrics:
    first: np.ndarray
    clear: np.ndarray
    violation: np.ndarray
    immediate_violation: np.ndarray


def signed_boundary_clearance(positions, headings, starts, ends):
    """Preserve the existing inside margin; extend it negatively outside.

    Ray-crossing containment works for either polygon winding. For an outside
    centre, the deficit is distance to the nearest boundary segment plus hull
    support along that segment's normal. This penalises growing excursions;
    it is a signed extension, not exact polygon-to-rectangle penetration.
    """
    positions, headings = np.asarray(positions), np.asarray(headings)
    starts, ends = np.asarray(starts), np.asarray(ends)
    edge = ends - starts
    length2 = np.maximum(np.sum(edge * edge, axis=1), 1e-12)
    normal = np.stack([-edge[:, 1], edge[:, 0]], axis=1) / np.sqrt(length2)[:, None]
    relative = positions[..., None, :] - starts
    fraction = np.clip(np.sum(relative * edge, axis=-1) / length2, 0.0, 1.0)
    distance = np.linalg.norm(relative - fraction[..., None] * edge, axis=-1)
    forward = np.stack([np.sin(headings), np.cos(headings)], axis=-1)
    right = np.stack([np.cos(headings), -np.sin(headings)], axis=-1)
    support = cc.HALF_L * np.abs(forward @ normal.T) + cc.HALF_W * np.abs(right @ normal.T)
    x, y = positions[..., 0, None], positions[..., 1, None]
    crossing = (starts[:, 1] > y) != (ends[:, 1] > y)
    denominator = np.where(np.abs(edge[:, 1]) > 1e-12, edge[:, 1], 1.0)
    crossing_x = starts[:, 0] + (y - starts[:, 1]) * edge[:, 0] / denominator
    inside = np.count_nonzero(crossing & (x < crossing_x), axis=-1) % 2 == 1
    nearest = np.argmin(distance, axis=-1)[..., None]
    outside = -np.take_along_axis(distance + support, nearest, axis=-1)[..., 0]
    return np.where(inside, np.min(distance - support, axis=-1), outside)


def evaluate(snap, rollout):
    """Return original hard checks plus integrated per-sequence soft deficits."""
    static = cc.point_clearance(rollout.positions, rollout.headings, snap.points,
                                reach=cc.HALF_L + 1.0) - v2.GAP_STATIC_M
    boundary = cc.boundary_clearance(rollout.positions, rollout.headings,
                                     snap.edges_a, snap.edges_b) - v2.GAP_BOUNDARY_M
    signed_boundary = signed_boundary_clearance(rollout.positions, rollout.headings,
                                                snap.edges_a, snap.edges_b) - v2.GAP_BOUNDARY_M
    clearance = np.minimum(static, boundary)
    deficit = np.maximum(0.0, -static) + np.maximum(0.0, -signed_boundary)
    for track in snap.tracks:
        target = cc.target_gap(rollout.positions, rollout.headings, rollout.times, track) - v2.GAP_TARGET_M
        clearance = np.minimum(clearance, target)
        deficit += np.maximum(0.0, -target)
    bad = clearance < 0.0
    first = np.where(bad.any(axis=0), rollout.times[np.argmax(bad, axis=0)], np.inf)
    minimum = clearance.min(axis=0)
    interval = np.diff(np.concatenate(([0.0], rollout.times)))
    violation = (deficit * interval[:, None]).sum(axis=0)
    immediate = (deficit * np.minimum(interval, np.maximum(
        0.0, cfg.UPDATE_RATE - np.concatenate(([0.0], rollout.times[:-1]))))[:, None]).sum(axis=0)

    moving = rollout.speeds[-1] > v2.TERMINAL_STOP_SPEED
    if moving.any():
        run = np.minimum(v2.TERMINAL_M, v2.TERMINAL_S * rollout.speeds[-1, moving])
        distance = np.linspace(1 / 8, 1.0, 8)[:, None, None] * run[None, :, None]
        heading = rollout.headings[-1, moving]
        extended = rollout.positions[-1, moving][None] + distance * np.stack(
            [np.sin(heading), np.cos(heading)], axis=1)[None]
        extended_heading = np.broadcast_to(heading, extended.shape[:2])
        static_terminal = cc.point_clearance(extended, extended_heading, snap.points,
                                             reach=cc.HALF_L + 1.0) - v2.GAP_STATIC_M
        terminal = static_terminal
        terminal_deficit = np.maximum(0.0, -static_terminal.min(axis=0))
        if v2.TERMINAL_BOUNDARY:
            boundary_terminal = cc.boundary_clearance(extended, extended_heading,
                                                      snap.edges_a, snap.edges_b) - v2.GAP_BOUNDARY_M
            signed_terminal = signed_boundary_clearance(extended, extended_heading,
                                                        snap.edges_a, snap.edges_b) - v2.GAP_BOUNDARY_M
            terminal = np.minimum(terminal, boundary_terminal)
            terminal_deficit += np.maximum(0.0, -signed_terminal.min(axis=0))
        terminal_min = np.full(len(minimum), np.inf)
        terminal_min[moving] = terminal.min(axis=0)
        first = np.where(np.isinf(first) & (terminal_min < 0.0), rollout.times[-1], first)
        minimum = np.minimum(minimum, terminal_min)
        violation[moving] += cfg.UPDATE_RATE * terminal_deficit
    return RecoveryMetrics(first, minimum, violation, immediate)


def choose(sequences, metrics, policy_reference, *, immediate_first=False):
    """Choose a complete sequence by slack, then existing policy-action distance.

    The optional immediate-prefix priority is an explicit ablation, not enabled
    by default. No preference is given to an old plan merely because it was
    once checked, and metrics from different recovery branches are never mixed.
    """
    commands = np.asarray(sequences, dtype=float)[:, 0]
    reference = np.asarray(policy_reference, dtype=float)
    throttle = np.where(np.isnan(commands[:, 1]), -1.5, commands[:, 1])
    distance = ((commands[:, 0] - reference[0]) ** 2
                + v2.W_THROTTLE * (throttle - reference[1]) ** 2)
    keys = (np.arange(len(commands)), distance, metrics.violation)
    if immediate_first:
        keys += (metrics.immediate_violation,)
    return int(np.lexsort(keys)[0])
