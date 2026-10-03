"""Experimental finite ensemble of onboard target motion predictions.

Li & Jilkov (2003), "Survey of maneuvering target tracking. Part I: Dynamic
models", doi:10.1109/TAES.2003.1261132, surveys constant-turn kinematics.
Johansen, Cristofaro & Perez (2016), Secs. 3.3--3.4, motivates testing several
possible obstacle motions in marine collision avoidance:
https://torarnj.folk.ntnu.no/colregs_cams.pdf
doi:10.1016/j.ifacol.2016.10.315 .

This is a small engineering ensemble, not an IMM tracker, a COLREGS behaviour
model, or a worst-case reachable set. CV and immediate port/starboard turns
sample only three possible motions. Delayed turns, acceleration, tracking
errors and unobserved targets remain uncovered. The required turn-rate value
is an explicit, uncalibrated hypothesis; it is not a physical uncertainty bound.

Only estimated TrackView position, velocity and heading enter prediction.
IDs associate an optional supplied measured turn rate; context, simulator
targets and hidden reactive/non-compliant labels are never read. No history
is inferred here. An envelope copies these values once per decision and can
be shared by all candidate, continuation and policy-prefix checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from classical import common as cc
import safety_v2 as v2


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 2D vector")
    return result


def _times(value):
    result = np.asarray(value, dtype=float)
    if (result.ndim != 1 or not len(result) or not np.isfinite(result).all()
            or result[0] < 0. or (np.diff(result) <= 0.).any()):
        raise ValueError("Prediction times must be finite, nonnegative and strictly increasing")
    return result


def constant_turn(position, velocity, heading, times, turn_rate_rad_s):
    """Exact constant-speed turn in x-east/y-north compass coordinates.

    Positive yaw rate is clockwise/starboard. Hull and velocity rotate together,
    retaining their initial relative angle. The zero-rate limit is exactly CV;
    sinc expressions avoid cancellation near zero. A stationary centre stays
    stationary while its hull may rotate under the sampled turn hypothesis.
    """
    position, velocity = _vector(position, "Position"), _vector(velocity, "Velocity")
    times = _times(times)
    heading, rate = float(heading), float(turn_rate_rad_s)
    if not math.isfinite(heading) or not math.isfinite(rate):
        raise ValueError("Heading and turn rate must be finite")
    if rate == 0.:
        return position + times[:, None] * velocity, np.full(len(times), heading)
    angle = rate * times
    along = times * np.sinc(angle / np.pi)
    across = times * .5 * angle * np.sinc(angle / (2. * np.pi)) ** 2
    positions = position + np.column_stack((velocity[0] * along + velocity[1] * across,
                                           velocity[1] * along - velocity[0] * across))
    return positions, heading + angle


def turning_hull_gap(own_positions, own_headings, target_positions, target_headings):
    """Shared inflated-rectangle SAT formula with time-varying target heading.

    Same four separating axes and margins as reference_controller.hull_separation;
    target states are (K,2)/(K,), own states are (K,N,2)/(K,N).
    """
    forward = np.stack((np.sin(own_headings), np.cos(own_headings)), axis=-1)
    starboard = np.stack((np.cos(own_headings), -np.sin(own_headings)), axis=-1)
    tf = np.stack((np.sin(target_headings), np.cos(target_headings)), axis=-1)[:, None]
    tr = np.stack((np.cos(target_headings), -np.sin(target_headings)), axis=-1)[:, None]
    relative = target_positions[:, None] - own_positions
    dot = lambda one, two: np.sum(one * two, axis=-1)
    ff, fr = np.abs(dot(forward, tf)), np.abs(dot(forward, tr))
    sf, sr = np.abs(dot(starboard, tf)), np.abs(dot(starboard, tr))
    half_l = .5 * cc.VESSEL_LENGTH + cc.HULL_MARGIN
    half_w = .5 * cc.VESSEL_WIDTH + cc.HULL_MARGIN
    return np.maximum.reduce((
        np.abs(dot(relative, forward)) - half_l - half_l * ff - half_w * fr,
        np.abs(dot(relative, starboard)) - half_w - half_l * sf - half_w * sr,
        np.abs(dot(relative, tf)) - half_l - half_l * ff - half_w * sf,
        np.abs(dot(relative, tr)) - half_w - half_l * fr - half_w * sr,
    ))


@dataclass(frozen=True)
class _Target:
    identifier: int
    position: tuple[float, float]
    velocity: tuple[float, float]
    heading: float
    turn_rates: tuple[float, ...]


@dataclass(frozen=True)
class TargetPredictionEnvelope:
    """Immutable copied track estimates, with at most four motions per target.

    ``from_snapshot(..., turn_rate_rad_s=0.)`` is exactly the existing CV target
    check. A positive rate adds -rate and +rate. An optional measured rate from
    the caller's onboard history is clipped to that interval and added if
    distinct. No rate is estimated or persisted by this module.

    ``evaluate`` returns target-only (first violation time, minimum clearance)
    for each own-ship rollout column. Combine both arrays with the existing
    filter result using elementwise minimum; retain all static, boundary and
    terminal checks. No target terminal extrapolation is added.
    """
    _targets: tuple[_Target, ...]
    turn_rate_rad_s: float
    target_gap_m: float

    @classmethod
    def from_snapshot(cls, snap, *, turn_rate_rad_s: float,
                      measured_turn_rates: Mapping[int, float] | None = None,
                      target_gap_m: float | None = None):
        rate = float(turn_rate_rad_s)
        gap = float(v2.GAP_TARGET_M if target_gap_m is None else target_gap_m)
        if not math.isfinite(rate) or rate < 0.:
            raise ValueError("Turn-rate hypothesis must be finite and nonnegative")
        if not math.isfinite(gap) or gap < 0.:
            raise ValueError("Target gap must be finite and nonnegative")
        targets, identifiers = [], set()
        for track in snap.tracks:
            identifier = int(track.id)
            if identifier in identifiers:
                raise ValueError("Duplicate target association ID")
            identifiers.add(identifier)
            position = tuple(map(float, _vector(track.position, "Track position")))
            velocity = tuple(map(float, _vector(track.velocity, "Track velocity")))
            heading = float(track.heading)
            if not math.isfinite(heading):
                raise ValueError("Track heading must be finite")
            rates = [0.] if rate == 0. else [0., -rate, rate]
            if measured_turn_rates is not None and identifier in measured_turn_rates:
                measured = float(measured_turn_rates[identifier])
                if not math.isfinite(measured):
                    raise ValueError("Supplied measured turn rate must be finite")
                measured = float(np.clip(measured, -rate, rate))
                if measured not in rates:
                    rates.append(measured)
            targets.append(_Target(identifier, position, velocity, heading, tuple(rates)))
        return cls(tuple(targets), rate, gap)

    @property
    def targets_count(self):
        return len(self._targets)

    @property
    def hypotheses_count(self):
        return sum(len(target.turn_rates) for target in self._targets)

    def clearance(self, ro):
        """Minimum sampled target gap at each (time, candidate), in metres."""
        times = _times(ro.times)
        positions, headings = np.asarray(ro.positions), np.asarray(ro.headings)
        if (positions.ndim != 3 or positions.shape[0] != len(times)
                or positions.shape[2] != 2 or positions.shape[1] < 1
                or headings.shape != positions.shape[:2]
                or not np.isfinite(positions).all() or not np.isfinite(headings).all()):
            raise ValueError("Own rollout must contain finite (K,N,2) positions and (K,N) headings")
        minimum = np.full(headings.shape, np.inf)
        for target in self._targets:
            # Call the existing function for CV: the zero-rate mode must retain
            # its arithmetic and geometry, including the frozen hull heading.
            view = cc.TrackView(target.identifier, np.asarray(target.position),
                                np.asarray(target.velocity), target.heading)
            minimum = np.minimum(minimum, cc.target_gap(positions, headings, times, view) - self.target_gap_m)
            for rate in target.turn_rates[1:]:
                predicted, target_headings = constant_turn(target.position, target.velocity,
                                                            target.heading, times, rate)
                minimum = np.minimum(minimum, turning_hull_gap(positions, headings,
                                                               predicted, target_headings) - self.target_gap_m)
        return minimum

    def evaluate(self, ro):
        """Return (first violation time or inf, minimum gap) per candidate."""
        clearances = self.clearance(ro)
        bad = clearances < 0.
        times = np.asarray(ro.times)
        first = np.where(bad.any(axis=0), times[np.argmax(bad, axis=0)], np.inf)
        return first, clearances.min(axis=0)
