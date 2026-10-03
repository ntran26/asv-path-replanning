"""Observable, action-conditioned hazard evidence; diagnostic shadow only.

Hsu, Hu and Fisac (2024), "The Safety Filter: A Unified View of Safety-Critical
Control in Autonomous Systems", separates monitoring from intervention:
https://arxiv.org/abs/2309.05837 . This module implements only that separation.
Wabersich and Zeilinger (2021), Sec. 4.1, Eqs. (5b)-(5c), motivates checking
predicted states against constraints: https://arxiv.org/abs/1812.05506 .

The proposed action is held for the EXISTING COMMIT_S horizon using the caller's
onboard predictor and actuator history. No future SAC observation is invented.
The existing hull/gap approximations are used without a terminal check. A
one-decision violation is called urgent; two consecutive fresh threatening
frames form persistent evidence. Those labels are engineering hypotheses, not
rules from either paper, calibrated probabilities, or safety certificates.
The one-second horizon can be too late for this ship's slow response. Nothing
here changes an action or authorizes bypassing a checked safety filter.

"Fresh" describes telemetry, not every static return: Snapshot.points includes
scan memory. Static persistence is a constraint-channel observation and can
span different nearest points. Track IDs are estimated associations. Missing
hazards and stale frames break persistence; stale frames never count twice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

import constants as cfg
from classical import common as cc
import safety_v2 as v2


SHADOW_ONLY = True
PERSISTENCE_FRAMES = 2  # Diagnostic design choice; unvalidated, no control effect.


def _json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return value


@dataclass(frozen=True)
class HazardEvidence:
    key: str
    kind: str
    current_clearance_m: float
    minimum_predicted_clearance_m: float
    first_violation_s: float | None
    measured_closing_rate_mps: float | None
    predicted_closing: bool
    threatening: bool
    consecutive_fresh_threats: int
    urgent: bool
    persistent: bool


@dataclass(frozen=True)
class RiskEvidence:
    fresh: bool
    horizon_s: float
    urgent: bool
    persistent: bool
    recommend_rescue: bool
    policy_action: tuple
    world_velocity_mps: tuple
    yaw_rate_rad_s: float
    hazards: tuple
    shadow_only: bool = SHADOW_ONLY

    def as_dict(self):
        """Strict JSON-compatible diagnostics; unavailable values become null."""
        return _json_finite(asdict(self))


def _clearance_channels(snap, positions, headings, times):
    """Reuse onboard geometry, with separate map edges and estimated targets."""
    if len(snap.points):
        yield "static_memory", "static_memory", cc.point_clearance(
            positions, headings, snap.points)[:, 0] - v2.GAP_STATIC_M
    if snap.edges_a is not None and snap.edges_b is not None:
        for index, (start, end) in enumerate(zip(snap.edges_a, snap.edges_b)):
            if np.linalg.norm(np.asarray(end) - start) <= 1e-12:
                continue
            yield f"boundary:{index}", "boundary", cc.boundary_clearance(
                positions, headings, np.asarray(start)[None], np.asarray(end)[None]
            )[:, 0] - v2.GAP_BOUNDARY_M
    for track in snap.tracks:
        yield f"track:{track.id}", "target", cc.target_gap(
            positions, headings, times, track)[:, 0] - v2.GAP_TARGET_M


class ObservableRiskMonitor:
    """Read-only monitor of snapshots; instantiate/reset for each episode.

    ``rollout(snap, actuators, sequences)`` must be the read-only onboard model
    callback, returning the existing cc.Rollout. It is called once for a single
    repeated-policy sequence. This object stores only previous diagnostic
    clearances and streak counts, never a controller plan or environment.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._previous = {}
        self._streaks = {}

    def update(self, snap, action, actuators, rollout, *, fresh=True):
        command = np.asarray(action, dtype=float)
        if command.shape != (2,) or not np.isfinite(command).all() or (np.abs(command) > 1).any():
            raise ValueError("Policy action must contain two finite normalized controls")
        horizon = float(v2.COMMIT_S)
        if horizon < float(cfg.UPDATE_RATE) or cfg.UPDATE_RATE <= 0:
            raise ValueError("Monitor needs at least one positive decision interval")
        decisions = int(math.ceil(horizon / cfg.UPDATE_RATE))
        sequences = np.broadcast_to(command, (1, decisions, 2)).copy()
        predicted = rollout(snap, actuators, sequences)
        all_times = np.asarray(predicted.times, dtype=float)
        if (all_times.ndim != 1 or not len(all_times) or not np.isfinite(all_times).all()
                or all_times[0] <= 0 or (np.diff(all_times) <= 0).any()
                or all_times[-1] < horizon - 1e-9):
            raise ValueError("Rollout times must increase and cover the monitor horizon")
        mask = all_times <= horizon + 1e-9
        if not mask.any() or all_times[mask][-1] < horizon - 1e-9:
            raise ValueError("Rollout times must include the monitor horizon endpoint")
        times = np.concatenate(([0.], all_times[mask]))
        positions = np.concatenate((np.asarray(snap.position)[None, None],
                                    np.asarray(predicted.positions)[mask]), axis=0)
        headings = np.concatenate(([[snap.heading]], np.asarray(predicted.headings)[mask]), axis=0)
        if (positions.shape != (len(times), 1, 2) or headings.shape != (len(times), 1)
                or not np.isfinite(positions).all() or not np.isfinite(headings).all()):
            raise ValueError("Monitor requires one finite predicted trajectory")

        hazards, previous, streaks = [], {}, {}
        for key, kind, clearances in _clearance_channels(snap, positions, headings, times):
            if key in previous:
                raise ValueError("Duplicate hazard identity in snapshot")
            current = float(clearances[0])
            future_min = float(np.min(clearances[1:]))
            if not np.isfinite(clearances).all():
                raise ValueError("Represented hazard has a nonfinite clearance")
            prior = self._previous.get(key) if fresh else None
            closing_rate = (prior - current) / cfg.UPDATE_RATE if prior is not None else None
            predicted_closing = future_min < current
            bad = np.flatnonzero(clearances < 0.)
            first = float(times[bad[0]]) if len(bad) else None
            threat = first is not None and (predicted_closing or current < 0.
                        or (closing_rate is not None and closing_rate > 0.))
            streak = self._streaks.get(key, 0) + 1 if fresh and threat else 0
            urgent = first is not None and first <= cfg.UPDATE_RATE + 1e-9
            persistent = streak >= PERSISTENCE_FRAMES
            hazards.append(HazardEvidence(key, kind, current, future_min, first,
                closing_rate, bool(predicted_closing), bool(threat), streak,
                bool(urgent), bool(persistent)))
            previous[key], streaks[key] = current, streak

        # Missing/stale observations break consecutive evidence; they do not
        # constitute observed clearance or advance a previous threat streak.
        self._previous, self._streaks = (previous, streaks) if fresh else ({}, {})
        urgent = any(h.urgent for h in hazards)
        persistent = any(h.persistent for h in hazards)
        s, c = math.sin(snap.heading), math.cos(snap.heading)
        velocity = (float(snap.u * s + snap.v * c), float(snap.u * c - snap.v * s))
        return RiskEvidence(bool(fresh), horizon, urgent, persistent,
            bool(fresh and (urgent or persistent)), tuple(map(float, command)),
            velocity, float(snap.r), tuple(hazards))
