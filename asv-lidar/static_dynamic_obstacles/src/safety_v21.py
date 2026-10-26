"""Experimental V21: V16 with a braking fallback where V16 finds no escape.

V16 (through V6) admits braking candidates and braking recoveries only when a
target track is within `ENGAGE_RANGE_M` (7 m), the restriction V2's iteration 6
introduced against stopping near walls.  With no target that close, V16 checks
turning and cruising plans only, and when none passes it returns SAC's command
unchecked (`no escape`).  On the baseline-v3 validation set (V21 phase, 2026-10-25)
every plan that passes V16's own checker at those decisions brakes, either at once or
in its recovery, and V16 hands SAC the helm instead.

At a decision V16 leaves unchecked (by default `no escape` only), V21 scores the same
primitive bank with the braking candidates and braking recoveries included, from the
snapshot and pre-command actuator history V16 already used, with V16's own rollout
and evaluation.  If some plan passes, it takes V6's selection rule (the passing
candidate closest to SAC within the room floor) and commits that plan exactly as V6's
`brake` branch does; otherwise V16's decision stands.  Every other decision is
V16's, unchanged.  This relaxes no check: the fallback issues only plans that pass
the inherited checker.

**Graded buffers** (`graded_buffers=True`, refinement R2).  When no plan passes the
checker, V21 re-evaluates the same rollouts at relaxed levels, in order: the terminal
run-out off; then the extra gaps halved; then no extra gaps (the inflated hull alone,
the geometry the environment scores a collision with).  It selects at the first level
where some plan passes, by the same rule, with that level's clearances.  This relaxes
the checker only at decisions where V16 checks nothing and issues SAC's command.  The
bank admits braking when `braking_fallback` is on or a target is within 7 m, as V16's.

**Hazard model** (perception wrappers; they change what every V16 decision sees):

* `scored_boundary=True` (R3, `safety_v21_boundary.py`): the boundary is checked at the
  limits the environment scores, the basin's ends at y = 0 and the map height rather
  than the navigable polygon's end edges 0.4 m inside them;
* `track_trail=True` (R4, `safety_v21_trail.py`): remembered static points inside a
  moving track's footprint of the last 20 s are left out of the snapshot.

`braking_fallback=False, graded_buffers=False, scored_boundary=False, track_trail=False`
reproduces V16.  `fallback_branches` names the V16 branches the fallback may replace
(``("no escape", "last certificate")`` also replaces the bounded uncertified
continuation).

`SafetyFilterV21E` adds the permanent emergency stop (planning_records/
SAFETY_ESTOP_PROTOCOL.md) on top; a braking-fallback plan counts as checked there.
"""
from __future__ import annotations

import copy

import numpy as np

from classical import common as cc
import safety_estop as estop
import safety_v2 as v2
import safety_v3 as v3
import safety_v4 as v4
import safety_v16 as v16

BRAKING_FALLBACK = True
GRADED_BUFFERS = False
SCORED_BOUNDARY = False
TRACK_TRAIL = False
FALLBACK_BRANCHES = ("no escape",)
LABEL = "braking fallback"
GRADED_LABEL = "graded buffers"
# (name, scale of the extra gaps, terminal run-out) tried in order after the full checker.
GRADED_LEVELS = (("no run-out", 1.0, False), ("half gaps", 0.5, False), ("no gaps", 0.0, False))


class SafetyFilterV21(v16.SafetyFilterV16):
    def __init__(self, *, braking_fallback=None, graded_buffers=None, scored_boundary=None, track_trail=None,
                 fallback_branches=None, **v16_options):
        super().__init__(**v16_options)
        self.braking_fallback = BRAKING_FALLBACK if braking_fallback is None else bool(braking_fallback)
        self.graded_buffers = GRADED_BUFFERS if graded_buffers is None else bool(graded_buffers)
        self.scored_boundary = SCORED_BOUNDARY if scored_boundary is None else bool(scored_boundary)
        self.track_trail = TRACK_TRAIL if track_trail is None else bool(track_trail)
        # Outermost wrappers: every V16 decision, and the fallback, read the corrected snapshot.
        if self.scored_boundary:
            from safety_v21_boundary import ScoredBoundaryPerception
            self.perception = ScoredBoundaryPerception(self.perception)
        if self.track_trail:
            from safety_v21_trail import TrailExclusionPerception
            self.perception = TrailExclusionPerception(self.perception)
        branches = FALLBACK_BRANCHES if fallback_branches is None else tuple(fallback_branches)
        if not set(branches) <= {"no escape", "last certificate"}:
            raise ValueError("fallback_branches must name unchecked V16 branches")
        self.fallback_branches = branches

    def _filter(self, env, action):
        out, changed = super()._filter(env, action)
        why = self.last.get("why", "idle")
        self.last["v21_braking_fallback"] = self.braking_fallback
        self.last["v21_graded_buffers"] = self.graded_buffers
        if self.track_trail:
            self.last["v21_trail"] = dict(getattr(self.perception, "last_trail", {}))
        if self.scored_boundary:
            self.last["v21_scored_boundary"] = True
        if (self.braking_fallback or self.graded_buffers) and why in self.fallback_branches:
            result = self._fallback(env, action, why)
            if result is not None:
                return result
            self.last["v21_fallback_found"] = False
        return out, changed

    def _fallback(self, env, action, v16_why):
        snap = getattr(self, "_observer_snapshot", None)
        before = getattr(self, "_observer_actuators", None)
        if snap is None or before is None:
            return None
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        traffic = any(np.hypot(*(t.position - snap.position)) < v2.ENGAGE_RANGE_M for t in snap.tracks)
        braking = self.braking_fallback or traffic
        rows = ([tuple(action)] + [(r, t) for r in v2.RUDDERS for t in v2.THROTTLES]
                + [tuple(self._rejoin(snap))] + ([(r, np.nan) for r in v2.BRAKE_RUDDERS] if braking else []))
        candidates = np.asarray(rows, dtype=float)
        templates = v4.recovery_templates([(r, np.nan if t is v2.BRAKE else t)
                                           for r, t in v2.RECOVERY if braking or t is not v2.BRAKE])
        count, nrec = len(candidates), len(templates)
        bank = v4.sequence_bank(candidates, templates)
        rollout = self._rollout(snap, copy.deepcopy(before), bank)
        levels = ([("checker", None)] if self.braking_fallback else []) + (
            [(name, (scale, terminal)) for name, scale, terminal in GRADED_LEVELS] if self.graded_buffers else [])
        for level, setting in levels:
            first, clear = (self._evaluate(snap, rollout) if setting is None
                            else evaluate_level(snap, rollout, *setting))
            first, clear = first.reshape(count, nrec), clear.reshape(count, nrec)
            ok = np.isinf(first)
            safe = ok.any(axis=1)
            if safe.any():
                break
        else:
            return None
        margin = np.where(ok, clear, -np.inf).max(axis=1)
        best_rec = np.argmax(np.where(ok, clear, -np.inf), axis=1)
        # V6's selection: the passing candidate closest to SAC within the room floor.
        pool = np.flatnonzero(safe)
        floor = min(v2.TRIGGER_MARGIN_M, float(margin[pool].max()) - v2.ROOM_SLACK_M)
        pool = pool[margin[pool] >= floor]
        throttle = np.where(np.isnan(candidates[pool, 1]), -1.5, candidates[pool, 1])
        score = (candidates[pool, 0] - action[0]) ** 2 + v2.W_THROTTLE * (throttle - action[1]) ** 2
        chosen = int(pool[np.argmin(score)])
        rec = int(best_rec[chosen])

        # Replace V16's command as V6 commits a checked plan; the actuator history
        # restarts from before V16 issued SAC's rudder.
        self.actuators = copy.deepcopy(before)
        self.mode, self.plan = "recovery", bank[chosen * nrec + rec].copy()
        self.uncertified_steps = 0
        out = candidates[chosen].copy()
        brake = bool(np.isnan(out[1]))
        env._v2_brake = brake
        if brake:
            out[1] = -1.0
        out = out.astype(np.float32)
        self.actuators.issue(env, float(out[0]))
        changed = brake or not np.allclose(out, action)
        self.last.update({
            "mode": "recovery", "why": LABEL if level == "checker" else GRADED_LABEL,
            "v21_level": level, "v21_replaced": v16_why, "v21_fallback_found": True,
            "changed": bool(changed), "brake": brake, "chosen": out.tolist(),
            "checked_clearance": float(margin[chosen]), "selection_floor": float(floor),
            "v21_plan_brakes": bool(np.isnan(self.plan[:, 1]).any()),
            "v21_safe_candidates": int(safe.sum()), "plan_steps": len(self.plan),
            "uncertified_steps": 0, "recovery_steps": self.recovery_steps,
        })
        return out, bool(changed)


def evaluate_level(snap, ro, gap_scale: float, terminal: bool):
    """V2's evaluation with the extra gaps scaled and the terminal run-out optional."""
    c = cc.point_clearance(ro.positions, ro.headings, snap.points,
                           reach=cc.HALF_L + 1.0) - gap_scale * v2.GAP_STATIC_M
    c = np.minimum(c, cc.boundary_clearance(ro.positions, ro.headings, snap.edges_a, snap.edges_b)
                   - gap_scale * v2.GAP_BOUNDARY_M)
    for track in snap.tracks:
        c = np.minimum(c, cc.target_gap(ro.positions, ro.headings, ro.times, track) - gap_scale * v2.GAP_TARGET_M)
    bad = c < 0.0
    first = np.where(bad.any(axis=0), ro.times[np.argmax(bad, axis=0)], np.inf)
    clear = c.min(axis=0)
    if terminal:
        moving = ro.speeds[-1] > v2.TERMINAL_STOP_SPEED
        if moving.any():
            run = np.minimum(v2.TERMINAL_M, v2.TERMINAL_S * ro.speeds[-1][moving])
            d = np.linspace(1 / 8, 1.0, 8)[:, None, None] * run[None, :, None]
            hdg = ro.headings[-1][moving]
            ext = ro.positions[-1][moving][None] + d * np.stack([np.sin(hdg), np.cos(hdg)], axis=1)[None]
            hk = np.broadcast_to(hdg, ext.shape[:2])
            ce = (cc.point_clearance(ext, hk, snap.points, reach=cc.HALF_L + 1.0)
                  - gap_scale * v2.GAP_STATIC_M).min(axis=0)
            cm = np.full(len(clear), np.inf)
            cm[moving] = ce
            first = np.where(np.isinf(first) & (cm < 0.0), ro.times[-1], first)
            clear = np.minimum(clear, cm)
    return first, clear


class _CheckedLabel(SafetyFilterV21):
    """Presents a V21 fallback plan to the stop detector as the checked `brake` branch."""

    def _filter(self, env, action):
        out, changed = super()._filter(env, action)
        if self.last.get("why") in (LABEL, GRADED_LABEL):
            self.last["v21_why"] = self.last["why"]
            self.last["why"] = "brake"
        return out, changed


class SafetyFilterV21E(estop.EmergencyStopMixin, _CheckedLabel):
    """V21 with the permanent emergency stop; a V21 fallback plan counts as checked (a plan
    that passes at any level also clears the detector's uninflated-hull contact test)."""

    def _filter(self, env, action):
        out, changed = super()._filter(env, action)
        restored = self.last.pop("v21_why", None)
        if restored and self.last.get("why") == "brake":
            self.last["why"] = restored
        return out, changed


# ---------------------------------------------------------------------------
# E1: the emergency-stop detector on the scored collision geometry
# ---------------------------------------------------------------------------
# The protocol's detector (safety_estop.first_contact_times) counts contact with the
# uninflated own hull.  The environment scores a collision when the hull inflated by
# HULL_MARGIN (0.15 m) meets an obstacle, the boundary limits or a target's true hull
# (env.collision_kind).  With `estop_collision_hull=True` the detector uses that inflated
# hull, so its "certain contact" is the contact the episode scores.  With R3 the
# boundary it checks is also the scored one, since it reads the same snapshot.
ESTOP_COLLISION_HULL = False


def _rect_point_clearance(positions, headings, points, half_l, half_w):
    K, n = headings.shape
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points):
        return np.full((K, n), np.inf)
    centre = positions.reshape(-1, 2)
    lo, hi = centre.min(axis=0) - (half_l + 1.0), centre.max(axis=0) + (half_l + 1.0)
    points = points[np.all((points >= lo) & (points <= hi), axis=1)]
    if not len(points):
        return np.full((K, n), np.inf)
    rel = points[None, None, :, :] - positions[:, :, None, :]
    s, c = np.sin(headings)[..., None], np.cos(headings)[..., None]
    lon = np.abs(rel[..., 0] * s + rel[..., 1] * c) - half_l
    lat = np.abs(rel[..., 0] * c - rel[..., 1] * s) - half_w
    outside = np.hypot(np.maximum(lon, 0.0), np.maximum(lat, 0.0))
    inside = np.minimum(np.maximum(lon, lat), 0.0)
    return np.min(outside + inside, axis=-1)


def collision_contact_times(snap, ro, use_static_points: bool) -> np.ndarray:
    """First predicted scored-collision time per column (inflated own hull); inf if none."""
    import safety_v20_contingency as sc
    from reference_controller import hull_separation
    half_l, half_w = estop.TRUE_HALF_L + cc.HULL_MARGIN, estop.TRUE_HALF_W + cc.HULL_MARGIN
    points = snap.points if use_static_points else np.empty((0, 2))
    contact = _rect_point_clearance(ro.positions, ro.headings, points, half_l, half_w) < 0.0
    poly = np.asarray(snap.edges_a, dtype=float)
    if poly.ndim == 2 and len(poly) >= 3:
        pts = sc.corners(ro.positions, ro.headings, half_l, half_w)
        contact |= ~sc.inside_polygon(pts, poly).all(axis=-1)
    for track in snap.tracks:
        future = track.position + ro.times[:, None] * track.velocity
        contact |= hull_separation(ro.positions, ro.headings, future[:, None, :], track.heading,
                                   margin=cc.HULL_MARGIN) < 0.0
    hit = contact.any(axis=0)
    return np.where(hit, ro.times[np.argmax(contact, axis=0)], np.inf)


def collision_escape_time(snap, actuators, first_commands, window_s, use_static_points):
    """safety_estop.escape_time with the scored collision geometry."""
    import math
    import safety_v20_contingency as sc
    decisions = int(math.ceil(window_s / 0.5))
    seqs = estop.sequences(first_commands, decisions)
    braking = np.isnan(seqs[:, :, 1]).any(axis=1)
    best = np.full(len(seqs), -np.inf)
    for efficiency, delay in (sc.WEAK, sc.STRONG):
        columns = np.arange(len(seqs)) if efficiency == sc.WEAK[0] else np.flatnonzero(braking)
        for start in range(0, len(columns), estop.CHUNK_COLUMNS):
            cols = columns[start:start + estop.CHUNK_COLUMNS]
            ro = sc.rollout_states(snap, actuators, seqs[cols], efficiency, delay)
            keep = ro.times <= window_s + 1e-9
            ro = sc.StateRollout(ro.times[keep], ro.positions[keep], ro.headings[keep], ro.u[keep],
                                 ro.v[keep], ro.r[keep], ro.servo[keep])
            best[cols] = np.maximum(best[cols], collision_contact_times(snap, ro, use_static_points))
    i = int(np.argmax(best))
    return float(best[i]), {"sequences": int(len(seqs)), "best_sequence_first_command": seqs[i, 0].tolist(),
                            "best_sequence_index": i, "geometry": "collision hull"}


class SafetyFilterV21EC(SafetyFilterV21E):
    """V21E whose detector predicts the scored collision (E1).

    The decision logic is the protocol mixin's (safety_estop.EmergencyStopMixin._filter):
    only at a decision with no checked plan, the longest predicted time before contact
    over every first command and continuation is compared with the window; only the
    contact geometry differs."""

    def _filter(self, env, action):
        import time
        self._estop_decisions += 1
        before = copy.deepcopy(self.actuators)
        if self.estop_committed_decision is not None:
            out = self._stop(env, before)
            self.last = {"why": "emergency stop", "mode": "emergency stop",
                         "estop_committed_decision": self.estop_committed_decision}
            return out, True
        out, changed = super(estop.EmergencyStopMixin, self)._filter(env, action)   # V21, checked label
        why = self.last.get("why", "idle")
        restored = self.last.pop("v21_why", None)
        if why == "idle" or why in estop.v20.CHECKED_BRANCHES:
            if restored and why == "brake":
                self.last["why"] = restored
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
        t_star, details = collision_escape_time(snap, before, firsts, self.estop_window_s,
                                                self.estop_use_static_points)
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
