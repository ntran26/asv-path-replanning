"""Stop-terminal contingency certificate for the experimental V20 filter.

A candidate first command is *certified* when, after it is held for one
decision, at least one contingency tail (optional turn, then full astern with
the rudder centred) stops the predicted surge, and every sampled clearance
over an explicit 20 s window -- reduced by an inter-sample allowance and the
calibrated forecast-error allowances -- stays nonnegative for both
reverse-thrust models, with at least the hold horizon simulated after the
stop and the hull nearly still at the window end. Target occupancy is the
constant-velocity contract over the same window.

Rest with zero propulsion and zero rudder angle is an equilibrium of the
identified body dynamics for every parameter vector (`bluefin/dynamics`), but
it is not reached quickly: at zero surge the model's yaw and sway damping is
almost purely quadratic, so a hull braked out of a turn keeps spinning and
creeping for tens of seconds. The window therefore simulates that motion
explicitly instead of freezing a rest pose, and the certificate is a
sliding-window check rechecked every decision, not an invariant-set proof.
Excluding braking inevitable-collision states with respect to a declared
target contract follows Bouraine, Fraichard and Salhi, Autonomous Robots 32,
2012, https://doi.org/10.1007/s10514-011-9258-8 . The committed-backup
structure follows model-predictive shielding (Bastani,
https://arxiv.org/abs/1905.10691) and gatekeeper (Agrawal, Chen and Panagou,
https://arxiv.org/abs/2211.14361). These sources motivate the structure only;
the finite tail family, nominal parameters and statistical allowances here do
not inherit their theorems. See planning/SAFETY_V20_PLAN.md.

Inputs are onboard only: the filter's decision snapshot (static memory,
map polygon, track views, observer-corrected ego state) and the controller's
own pre-command actuator history. No truth, scenario identity or outcome is
read. Plans use V3's encoding: rows are (rudder, throttle) per 0.5 s decision
and a NaN throttle is full astern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as _PolygonPath

import constants as cfg
import safety_v2 as v2
import ship
from classical import common as cc

DECISIONS = 40                         # 20 s contingency window, sampled at cc.PRED_DT
HOLD_HORIZON_S = 8.0                   # minimum explicit hold after the surge reaches zero
STOP_SURGE = 0.02                      # m/s, surge regarded as stopped
TERMINAL_SPEED = 0.05                  # m/s, |(u, v)| required at the window end
TERMINAL_EXTENSION_S = 1.0             # residual-motion bound beyond the window
STOP_RUDDERS = (-1.0, 0.0, 1.0)
TURN_RUDDERS = (-1.0, -0.5, 0.5, 1.0)
TURN_THROTTLES = (0.0, 1.0)            # cruise and ceiling propulsion
TURN_DECISIONS = (1, 2, 3, 4, 6)       # 0.5 to 3 s before braking
WEAK = (v2.BRAKE_EFFICIENCY, v2.BRAKE_DELAY_S)          # 0.25, 0.75 s
STRONG = (ship.REVERSE_THRUST_EFFICIENCY, 0.0)          # simulator assumption 0.5, no delay
HULL_HALF_DIAGONAL = math.hypot(0.5 * ship.VESSEL_LENGTH, 0.5 * ship.VESSEL_WIDTH)
INFLATED_HALF_DIAGONAL = math.hypot(cc.HALF_L, cc.HALF_W)


def tails(decisions: int = DECISIONS) -> np.ndarray:
    """The 43 contingency tails, shape (43, decisions - 1, 2), from decision 1 on."""
    length = decisions - 1
    rows = []
    for rudder in STOP_RUDDERS:
        tail = np.tile((0.0, np.nan), (length, 1))
        tail[0, 0] = rudder
        rows.append(tail)
    for rudder in TURN_RUDDERS:
        for throttle in TURN_THROTTLES:
            for n in TURN_DECISIONS:
                tail = np.tile((0.0, np.nan), (length, 1))
                tail[:n] = (rudder, throttle)
                rows.append(tail)
    return np.asarray(rows, dtype=float)


TAILS = tails()


def sequences_for(first_commands: np.ndarray, tail_bank: np.ndarray = TAILS) -> np.ndarray:
    """Candidate-major sequences (m * len(tail_bank), decisions, 2)."""
    first = np.asarray(first_commands, dtype=float).reshape(-1, 2)
    m, k = len(first), len(tail_bank)
    out = np.empty((m * k, tail_bank.shape[1] + 1, 2))
    out[:, 0] = np.repeat(first, k, axis=0)
    out[:, 1:] = np.tile(tail_bank, (m, 1, 1))
    return out


def shift_sequence(sequence: np.ndarray) -> np.ndarray:
    """Remainder after one executed decision, padded with a centred brake/hold."""
    sequence = np.asarray(sequence, dtype=float)
    return np.vstack([sequence[1:], [[0.0, np.nan]]])


@dataclass
class StateRollout:
    times: np.ndarray        # (K,), starting at 0 (current state)
    positions: np.ndarray    # (K, n, 2)
    headings: np.ndarray     # (K, n)
    u: np.ndarray            # (K, n)
    v: np.ndarray
    r: np.ndarray
    servo: np.ndarray


def rollout_states(snap, actuators, sequences, efficiency: float, delay_s: float) -> StateRollout:
    """`safety_prediction.rollout_seq` returning the full state, with the current state as sample 0.

    Same identified parameters, actuator history, 0.125 s step and operator-split
    braking as the inherited predictors, so a contingency is checked with the
    model V16 uses for its own plans.
    """
    sequences = np.asarray(sequences, dtype=float)
    n, decisions = sequences.shape[:2]
    state = np.zeros((7, n))
    state[:4] = np.array([max(0.0, snap.u), snap.v, snap.r, snap.heading])[:, None]
    state[4] = actuators.servo
    state[5], state[6] = snap.x, snap.y
    params = {key: np.full(n, value) for key, value in cc.IDENTIFIED.items()}
    pending = ([np.full(n, value) for value in actuators.buffer]
               if actuators.buffer is not None else None)
    samples = decisions * cc.SUBSTEPS + 1
    pos = np.empty((samples, n, 2))
    hdg, uu, vv, rr, sv = (np.empty((samples, n)) for _ in range(5))
    pos[0], hdg[0], uu[0], vv[0], rr[0], sv[0] = state[5:7].T, state[3], state[0], state[1], state[2], state[4]
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
            active = braking & (now - brake_since >= delay_s)
            if active.any():
                state[0] = np.where(active, np.maximum(0.0, state[0] - deceleration * cc.PRED_DT), state[0])
            sample += 1
            pos[sample] = state[5:7].T
            hdg[sample], uu[sample], vv[sample], rr[sample], sv[sample] = state[3], state[0], state[1], state[2], state[4]
    return StateRollout(cc.PRED_DT * np.arange(samples), pos, hdg, uu, vv, rr, sv)


def _lookup(table: Optional[Sequence[Tuple[float, float]]], times: np.ndarray) -> np.ndarray:
    """Step lookup that uses the next tabulated horizon at or above each time."""
    if not table:
        return np.zeros_like(times, dtype=float)
    horizons = np.array([h for h, _ in table], dtype=float)
    values = np.maximum.accumulate(np.array([q for _, q in table], dtype=float))
    index = np.searchsorted(horizons, times - 1e-9, side="left")
    index = np.minimum(index, len(values) - 1)
    out = values[index]
    beyond = times > horizons[-1]
    if beyond.any() and len(values) >= 2 and horizons[-1] > horizons[-2]:
        slope = max(0.0, (values[-1] - values[-2]) / (horizons[-1] - horizons[-2]))
        out = np.where(beyond, values[-1] + slope * (times - horizons[-1]), out)
    return out


def corners(positions: np.ndarray, headings: np.ndarray, half_l: float = cc.HALF_L,
            half_w: float = cc.HALF_W) -> np.ndarray:
    """Inflated hull corners, (..., 4, 2)."""
    fwd = np.stack([np.sin(headings), np.cos(headings)], axis=-1)
    stbd = np.stack([np.cos(headings), -np.sin(headings)], axis=-1)
    signs = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=float)
    return (positions[..., None, :] + signs[:, 0, None] * half_l * fwd[..., None, :]
            + signs[:, 1, None] * half_w * stbd[..., None, :])


def inside_polygon(points: np.ndarray, edges_a: np.ndarray) -> np.ndarray:
    """Point-in-polygon for the map polygon whose vertices are `edges_a` in order."""
    shape = points.shape[:-1]
    path = _PolygonPath(np.asarray(edges_a, dtype=float))
    return path.contains_points(points.reshape(-1, 2)).reshape(shape)


def signed_corner_clearance(positions, headings, edges_a, edges_b) -> np.ndarray:
    """Minimum signed distance of the inflated hull's corners to the map polygon.

    Positive inside, negative outside; shape (K, n). Unsigned edge distance alone
    becomes positive again outside a wall, so containment is tested explicitly.
    """
    pts = corners(positions, headings)                           # (K, n, 4, 2)
    a, b = np.asarray(edges_a, float), np.asarray(edges_b, float)
    edge = b - a
    length2 = np.maximum(np.sum(edge * edge, axis=1), 1e-12)
    rel = pts[..., None, :] - a                                  # (K, n, 4, E, 2)
    t = np.clip(np.sum(rel * edge, axis=-1) / length2, 0.0, 1.0)
    dist = np.linalg.norm(rel - t[..., None] * edge, axis=-1).min(axis=-1)   # (K, n, 4)
    inside = inside_polygon(pts, a)
    return np.where(inside, dist, -dist).min(axis=-1)


@dataclass
class Certificate:
    certified: bool
    slack: float                       # best tail's minimum slack, metres
    tail_index: int                    # into TAILS, -1 when none
    sequence: Optional[np.ndarray]     # (DECISIONS, 2) first command + tail
    rest_time_s: float                 # time surge reaches zero, later of the two models
    diagnostics: Dict = field(default_factory=dict)


class ContingencyChecker:
    """Certificate evaluation from one decision snapshot and actuator history."""

    def __init__(self, snap, actuators, own_table=None, target_table=None,
                 hold_horizon_s: float = HOLD_HORIZON_S):
        if not np.isfinite(hold_horizon_s) or hold_horizon_s < 0.0:
            raise ValueError("hold horizon must be finite and nonnegative")
        self.snap, self.actuators = snap, actuators
        self.own_table, self.target_table = own_table, target_table
        self.hold_horizon_s = float(hold_horizon_s)
        poly = np.asarray(snap.edges_a, dtype=float)
        self.has_polygon = poly.ndim == 2 and len(poly) >= 3

    # -- clearances ---------------------------------------------------------
    def _slack(self, ro: StateRollout) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Per-column minimum slack over the explicit window and the stop time.

        The whole window is simulated, including the slow spin and sway creep
        that the identified model retains at zero surge (its yaw and sway
        damping is almost purely quadratic there), so no pose is frozen. A tail
        qualifies only if surge reaches zero, at least `hold_horizon_s` of the
        window remains after that, and the hull is nearly still at the end.
        """
        snap = self.snap
        times = ro.times
        speed = np.hypot(ro.u, ro.v)
        # Lipschitz bound between samples: hull points move at most speed + R*|r|.
        # Clearance on an interval is at least the smaller end value minus half
        # the interval times the bound, so each sample carries the larger bound
        # of its two neighbouring intervals (approximate within one interval).
        lip = speed + INFLATED_HALF_DIAGONAL * np.abs(ro.r)
        step = lip.copy()
        step[:-1] = np.maximum(step[:-1], lip[1:])
        step[1:] = np.maximum(step[1:], lip[:-1])
        sample_allowance = 0.5 * cc.PRED_DT * step                  # (K, n)
        # Residual motion beyond the window, added at the final sample.
        terminal = TERMINAL_EXTENSION_S * (np.abs(ro.u[-1]) + np.abs(ro.v[-1])
                                           + INFLATED_HALF_DIAGONAL * np.abs(ro.r[-1]))
        sample_allowance[-1] += terminal

        stopped = ro.u <= STOP_SURGE
        # Surge must stay stopped from its first stopped sample to the window end.
        settled = np.flip(np.logical_and.accumulate(np.flip(stopped, axis=0), axis=0), axis=0)
        reached = settled.any(axis=0)
        first = np.where(reached, np.argmax(settled, axis=0), -1)
        stop_time = np.where(reached, times[np.maximum(first, 0)], np.inf)
        qualifies = (reached & (stop_time + self.hold_horizon_s <= times[-1] + 1e-9)
                     & (speed[-1] <= TERMINAL_SPEED))
        # Calibrated own-ship error stops growing once surge is zero; later spin
        # and creep are simulated explicitly with the nominal model.
        frozen = np.minimum(times[:, None], np.where(reached, stop_time, times[-1])[None, :])
        e_own = np.maximum(0.0, _lookup(self.own_table, frozen) - ship.HULL_MARGIN)

        static = cc.point_clearance(ro.positions, ro.headings, snap.points,
                                    reach=cc.HALF_L + 1.0) - v2.GAP_STATIC_M
        static = static - e_own - sample_allowance
        parts = {"static": static.min(axis=0)}
        if self.has_polygon:
            edge = cc.boundary_clearance(ro.positions, ro.headings, snap.edges_a, snap.edges_b)
            signed = signed_corner_clearance(ro.positions, ro.headings, snap.edges_a, snap.edges_b)
            boundary = np.minimum(edge, signed) - v2.GAP_BOUNDARY_M - e_own - sample_allowance
            parts["boundary"] = boundary.min(axis=0)
        else:
            parts["boundary"] = np.full(ro.positions.shape[1], np.inf)

        target = np.full(ro.positions.shape[1], np.inf)
        if snap.tracks:
            rho_t = _lookup(self.target_table, times)[:, None]
            for track in snap.tracks:
                gap = cc.target_gap(ro.positions, ro.headings, times, track) - v2.GAP_TARGET_M
                allowance = sample_allowance + 0.5 * cc.PRED_DT * track.speed
                gap = gap - e_own - rho_t - allowance
                target = np.minimum(target, gap.min(axis=0))
        parts["target"] = target
        slack = np.minimum.reduce([parts["static"], parts["boundary"], parts["target"]])
        slack = np.where(qualifies, slack, -np.inf)
        parts["qualifies"] = qualifies.astype(float)
        return slack, stop_time, parts

    # -- public API ---------------------------------------------------------
    def evaluate(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Slack (min over both reverse models) and rest time for each sequence."""
        sequences = np.asarray(sequences, dtype=float)
        weak = rollout_states(self.snap, self.actuators, sequences, *WEAK)
        slack_w, rest_w, parts_w = self._slack(weak)
        braking = np.isnan(sequences[:, :, 1]).any(axis=1)
        slack_s = np.full(len(sequences), np.inf)
        rest_s = np.zeros(len(sequences))
        if braking.any():
            strong = rollout_states(self.snap, self.actuators, sequences[braking], *STRONG)
            s, rt, _ = self._slack(strong)
            slack_s[braking], rest_s[braking] = s, rt
        slack = np.minimum(slack_w, slack_s)
        rest = np.maximum(rest_w, rest_s)
        return slack, rest, {k: v for k, v in parts_w.items()}

    def certify(self, first_command) -> Certificate:
        first = np.asarray(first_command, dtype=float).reshape(2)
        if not np.isfinite(first[0]) or abs(first[0]) > 1.0 or np.isinf(first[1]) or (
                np.isfinite(first[1]) and abs(first[1]) > 1.0):
            raise ValueError("First command must be normalized; only throttle may be NaN")
        seqs = sequences_for(first[None])
        slack, rest, parts = self.evaluate(seqs)
        return self._best(seqs, slack, rest, parts)

    def certify_sequence(self, sequence) -> Certificate:
        seq = np.asarray(sequence, dtype=float)[None]
        slack, rest, parts = self.evaluate(seq)
        return self._best(seq, slack, rest, parts, single=True)

    @staticmethod
    def _best(seqs, slack, rest, parts, single=False) -> Certificate:
        ok = np.isfinite(slack) & (slack >= 0.0)
        finite = np.where(np.isfinite(slack), slack, -np.inf)
        if ok.any():
            # Largest slack first, then the earlier rest.
            order = np.lexsort((rest, -finite))
            i = int(next(j for j in order if ok[j]))
        else:
            i = int(np.argmax(finite))
        diag = {key: float(value[i]) for key, value in parts.items()}
        return Certificate(bool(ok[i]), float(finite[i]), -1 if single else i,
                           seqs[i].copy(), float(rest[i]), diag)


def distance_to(reference, commands) -> np.ndarray:
    """V3's action distance with a brake placed beyond the throttle floor."""
    reference = np.asarray(reference, dtype=float)
    commands = np.asarray(commands, dtype=float).reshape(-1, 2)
    thr = np.where(np.isnan(commands[:, 1]), -1.5, commands[:, 1])
    ref_thr = -1.5 if np.isnan(reference[1]) else reference[1]
    return (commands[:, 0] - reference[0]) ** 2 + v2.W_THROTTLE * (thr - ref_thr) ** 2


def projection_candidates(reference) -> np.ndarray:
    """V2's command grid plus brake rows, ordered by distance to `reference`."""
    grid = [(r, t) for r in v2.RUDDERS for t in v2.THROTTLES]
    grid += [(r, np.nan) for r in v2.BRAKE_RUDDERS]
    grid = np.asarray(grid, dtype=float)
    return grid[np.argsort(distance_to(reference, grid), kind="stable")]


def observed_free_fraction(sequence_rollout: StateRollout, column: int, sensor_origin,
                           heading: float, bearings_deg, ranges, max_range: float,
                           min_range: float) -> Dict[str, float]:
    """Share of swept hull-corner samples on currently observed free rays.

    Diagnostic only. A corner is `free` when the ray nearest its bearing
    returned beyond it, `dead_zone` inside the minimum range, otherwise
    `unobserved` (beyond a return or past maximum range).
    """
    pts = corners(sequence_rollout.positions[:, column], sequence_rollout.headings[:, column],
                  half_l=0.5 * ship.VESSEL_LENGTH, half_w=0.5 * ship.VESSEL_WIDTH).reshape(-1, 2)
    rel = pts - np.asarray(sensor_origin, dtype=float)
    rng = np.hypot(rel[:, 0], rel[:, 1])
    bearing = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1]) - heading) + 180.0) % 360.0 - 180.0
    bearings = np.asarray(bearings_deg, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    diff = np.abs((bearing[:, None] - bearings[None, :] + 180.0) % 360.0 - 180.0)
    nearest = np.argmin(diff, axis=1)
    ray = ranges[nearest]
    dead = rng < min_range
    free = (~dead) & (rng < np.minimum(ray, max_range))
    n = max(1, len(pts))
    return {"free": float(free.sum() / n), "dead_zone": float(dead.sum() / n),
            "unobserved": float((~dead & ~free).sum() / n)}
