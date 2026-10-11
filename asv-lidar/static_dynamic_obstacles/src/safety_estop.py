"""Permanent emergency stop when a collision is predicted to be certain.

Protocol and decisions: planning/SAFETY_ESTOP_PROTOCOL.md. A filter class
wrapped here (V16 or V17) decides unchanged until a decision at which it has
no plan that passed its own checker. There the detector rolls out every
candidate first command (SAC, the issued command, V2's grid and the
full-astern rows), each followed by every continuation in a fixed bank (hold
the command, the stop and turn-then-stop tails, the turn-cruise-stop tails),
from the filter's own decision snapshot and actuator history. If every
sequence is predicted to make physical contact (true hulls, no margins)
within the window, under the later of the two reverse-thrust models, the
filter commits to full astern with the rudder centred for the rest of the
episode. Only onboard estimates are used; truth, scenario identity and
outcomes are never read. The continuation bank and model are finite, so
"certain" means certain within them, not a proof.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from classical import common as cc
from reference_controller import hull_separation
import safety_v2 as v2
import safety_v16 as v16
import safety_v17 as v17
import safety_v20 as v20
import safety_v20_contingency as sc
import ship

# Decision (2026-10-10) from the development screen (planning/SAFETY_ESTOP_PROTOCOL.md
# section 6): 1.5 s with the obstacle memory included, accepting false alarms.
ESTOP_WINDOW_S = 1.5
# Boundary-and-targets trigger (protocol section 8): the window selected by the
# section 4 rule on the development screen without static points.
BOUNDARY_TARGET_OPTIONS = {"estop_window_s": 2.5, "use_static_points": False}
MAX_WINDOW_S = 6.0
TRUE_HALF_L = 0.5 * ship.VESSEL_LENGTH
TRUE_HALF_W = 0.5 * ship.VESSEL_WIDTH
CHUNK_COLUMNS = 256
# Hazards overlapping the hull at the current estimate are estimation errors.
EXCLUDE_CURRENT_OVERLAPS = False
# Remembered static LiDAR points in the contact check. The map boundary and
# target tracks are always checked.
USE_STATIC_POINTS = True


def continuation_bank(decisions: int) -> np.ndarray:
    """(tails, decisions - 1, 2): stop, turn-then-stop and turn-cruise-stop tails, truncated.

    A NaN rudder marks "hold the first command" and is filled per sequence.
    """
    stop = sc.TAILS[:, :decisions - 1]
    cruise = sc.EXTENDED_TAILS[len(sc.STOP_RUDDERS):, :decisions - 1]
    hold = np.full((1, decisions - 1, 2), np.nan)
    return np.concatenate([hold, stop, cruise], axis=0)


def sequences(first_commands, decisions: int) -> np.ndarray:
    first = np.asarray(first_commands, dtype=float).reshape(-1, 2)
    bank = continuation_bank(decisions)
    seqs = np.empty((len(first) * len(bank), decisions, 2))
    for i, command in enumerate(first):
        block = slice(i * len(bank), (i + 1) * len(bank))
        seqs[block, 0] = command
        tails = bank.copy()
        tails[0] = command                      # hold the first command throughout
        seqs[block, 1:] = tails
    return seqs


def true_point_clearance(positions, headings, points):
    """Distance from the true (uninflated) hull rectangle to the nearest point, (K, n)."""
    K, n = headings.shape
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points):
        return np.full((K, n), np.inf)
    centre = positions.reshape(-1, 2)
    lo, hi = centre.min(axis=0) - (TRUE_HALF_L + 1.0), centre.max(axis=0) + (TRUE_HALF_L + 1.0)
    points = points[np.all((points >= lo) & (points <= hi), axis=1)]
    if not len(points):
        return np.full((K, n), np.inf)
    rel = points[None, None, :, :] - positions[:, :, None, :]
    s, c = np.sin(headings)[..., None], np.cos(headings)[..., None]
    lon = np.abs(rel[..., 0] * s + rel[..., 1] * c) - TRUE_HALF_L
    lat = np.abs(rel[..., 0] * c - rel[..., 1] * s) - TRUE_HALF_W
    outside = np.hypot(np.maximum(lon, 0.0), np.maximum(lat, 0.0))
    inside = np.minimum(np.maximum(lon, lat), 0.0)
    return np.min(outside + inside, axis=-1)


def corners_outside(positions, headings, edges_a) -> np.ndarray:
    """True where any true-hull corner lies outside the map polygon, (K, n)."""
    pts = sc.corners(positions, headings, TRUE_HALF_L, TRUE_HALF_W)
    return ~sc.inside_polygon(pts, edges_a).all(axis=-1)


def current_overlap_free(snap, points):
    """Static points not inside the true hull at the current estimated pose.

    A real overlap would already be a collision, so a remembered point inside
    the hull now is an estimation error (stale memory, pose error or target
    returns kept as static points) and cannot make a contact certain.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points):
        return points
    pos = np.array([[[snap.x, snap.y]]])
    hdg = np.array([[snap.heading]])
    clear = np.array([true_point_clearance(pos, hdg, p[None])[0, 0] for p in points])
    return points[clear >= 0.0]


def first_contact_times(snap, ro, exclude_current_overlaps: bool = EXCLUDE_CURRENT_OVERLAPS,
                        use_static_points: bool = USE_STATIC_POINTS) -> np.ndarray:
    """First sample time of predicted physical contact per column; inf if none.

    With ``exclude_current_overlaps`` a hazard that already overlaps the true
    hull at the current estimated pose is treated as an estimation error and
    left out: static points inside the hull, a polygon that already excludes a
    hull corner, and target hulls that already overlap.
    """
    points = snap.points if use_static_points else np.empty((0, 2))
    points = current_overlap_free(snap, points) if exclude_current_overlaps else points
    contact = true_point_clearance(ro.positions, ro.headings, points) < 0.0
    poly = np.asarray(snap.edges_a, dtype=float)
    if poly.ndim == 2 and len(poly) >= 3:
        outside = corners_outside(ro.positions, ro.headings, poly)
        if not (exclude_current_overlaps and outside[0].any()):
            contact |= outside
    for track in snap.tracks:
        future = track.position + ro.times[:, None] * track.velocity
        gap = hull_separation(ro.positions, ro.headings, future[:, None, :], track.heading, margin=0.0)
        if exclude_current_overlaps and (gap[0] < 0.0).any():
            continue
        contact |= gap < 0.0
    hit = contact.any(axis=0)
    first = np.argmax(contact, axis=0)
    return np.where(hit, ro.times[first], np.inf)


def escape_time(snap, actuators, first_commands, window_s: float = MAX_WINDOW_S,
                exclude_current_overlaps: bool = None, use_static_points: bool = None) -> tuple:
    """T*: the longest predicted time before contact over all candidate sequences.

    Each sequence's time is the later of its two reverse-model contact times
    (sequences without astern thrust do not depend on the model). Returns
    (T*, details); T* is inf when some sequence avoids contact for the window.
    """
    decisions = int(math.ceil(window_s / 0.5))
    seqs = sequences(first_commands, decisions)
    braking = np.isnan(seqs[:, :, 1]).any(axis=1)
    best = np.full(len(seqs), -np.inf)
    for efficiency, delay in (sc.WEAK, sc.STRONG):
        columns = np.arange(len(seqs)) if efficiency == sc.WEAK[0] else np.flatnonzero(braking)
        for start in range(0, len(columns), CHUNK_COLUMNS):
            cols = columns[start:start + CHUNK_COLUMNS]
            ro = sc.rollout_states(snap, actuators, seqs[cols], efficiency, delay)
            keep = ro.times <= window_s + 1e-9
            ro = sc.StateRollout(ro.times[keep], ro.positions[keep], ro.headings[keep], ro.u[keep],
                                 ro.v[keep], ro.r[keep], ro.servo[keep])
            best[cols] = np.maximum(best[cols], first_contact_times(
                snap, ro, EXCLUDE_CURRENT_OVERLAPS if exclude_current_overlaps is None
                else bool(exclude_current_overlaps),
                USE_STATIC_POINTS if use_static_points is None else bool(use_static_points)))
    i = int(np.argmax(best))
    return float(best[i]), {"sequences": int(len(seqs)), "best_sequence_first_command": seqs[i, 0].tolist(),
                            "best_sequence_index": i}


class EmergencyStopMixin:
    """Commit to a permanent centred full-astern stop when contact is predicted certain."""

    def __init__(self, *, estop_window_s=None, use_static_points=None, **options):
        super().__init__(**options)
        self.estop_use_static_points = (USE_STATIC_POINTS if use_static_points is None
                                        else bool(use_static_points))
        window = ESTOP_WINDOW_S if estop_window_s is None else estop_window_s
        if window is None or not math.isfinite(float(window)) or not 0.0 < float(window) <= MAX_WINDOW_S:
            raise ValueError("estop_window_s must be in (0, MAX_WINDOW_S]")
        self.estop_window_s = float(window)
        self.estop_committed_decision = None
        self._estop_decisions = 0

    def _stop(self, env, before):
        self.actuators = copy.deepcopy(before)
        self.actuators.issue(env, 0.0)
        env._v2_brake = True
        env.emergency_stop_committed = True
        self.plan = None
        return np.array([0.0, -1.0], dtype=np.float32)

    def _filter(self, env, action):
        self._estop_decisions += 1
        before = copy.deepcopy(self.actuators)
        if self.estop_committed_decision is not None:
            out = self._stop(env, before)
            self.last = {"why": "emergency stop", "mode": "emergency stop",
                         "estop_committed_decision": self.estop_committed_decision}
            return out, True
        out, changed = super()._filter(env, action)
        why = self.last.get("why", v20.IDLE)
        if why == v20.IDLE or why in v20.CHECKED_BRANCHES:
            return out, changed
        snap = getattr(self, "_observer_snapshot", None)
        if snap is None or bool(getattr(env, "command_rate_limit", False)):
            self.last["estop_skipped"] = "no snapshot" if snap is None else "command limit"
            return out, changed
        started = time.perf_counter()
        sac = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        brake = bool(getattr(env, "_v2_brake", False))
        issued = np.array([float(out[0]), np.nan if brake else float(out[1])])
        grid = [(r, t) for r in v2.RUDDERS for t in v2.THROTTLES] + [(r, np.nan) for r in v2.BRAKE_RUDDERS]
        firsts = np.vstack([sac[None], issued[None], np.asarray(grid, dtype=float)])
        t_star, details = escape_time(snap, before, firsts,
                                      use_static_points=self.estop_use_static_points)
        self.last.update(estop_t_star=t_star, estop_window_s=self.estop_window_s,
                         estop_use_static_points=self.estop_use_static_points,
                         estop_seconds=time.perf_counter() - started, estop_details=details)
        if t_star <= self.estop_window_s:
            self.estop_committed_decision = self._estop_decisions
            out = self._stop(env, before)
            self.last.update(why="emergency stop", mode="emergency stop", estop_v16_why=why,
                             estop_committed_decision=self.estop_committed_decision, changed=True)
            return out, True
        return out, changed


class SafetyFilterV16E(EmergencyStopMixin, v16.SafetyFilterV16):
    """V16 with the permanent emergency stop."""


class SafetyFilterV17E(EmergencyStopMixin, v17.SafetyFilterV17):
    """V17 with the permanent emergency stop."""
