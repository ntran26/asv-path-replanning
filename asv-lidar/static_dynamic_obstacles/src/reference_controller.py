"""Perception-only predictive LOS baseline (ported from CODEX/, F71; the C3 comparator), independent of a trained policy.

This is a diagnostic comparator, not a certified COLREG controller. It uses
the known path/map, cached localiser/IMU estimates, gated scans and tracks.
Candidate rollouts use the identified manoeuvring model initialized from
measurements and command history. Simulator truth never enters action selection.
"""
from __future__ import annotations

import math
import numpy as np

import constants as cfg
from path import wrap180
from ship import HULL_MARGIN, IDENTIFIED, LIDAR_OFFSET_M, VESSEL_LENGTH, VESSEL_WIDTH, dyn


def hull_separation(own_positions, own_headings, target_positions, target_heading,
                    margin=HULL_MARGIN):
    """Conservative SAT gap in metres between inflated vessel rectangles.

    Compass headings are in radians. Positions/headings accept NumPy broadcast
    shapes; positive values certify separation, negative values mean overlap.
    A positive SAT gap is a lower bound on true Euclidean hull clearance.
    """
    forward = np.stack([np.sin(own_headings), np.cos(own_headings)], axis=-1)
    starboard = np.stack([np.cos(own_headings), -np.sin(own_headings)], axis=-1)
    tf = np.array([math.sin(target_heading), math.cos(target_heading)])
    tr = np.array([math.cos(target_heading), -math.sin(target_heading)])
    rel = np.asarray(target_positions) - np.asarray(own_positions)
    half_l, half_w = 0.5 * VESSEL_LENGTH + margin, 0.5 * VESSEL_WIDTH + margin
    return np.maximum.reduce([
        np.abs(np.sum(rel * forward, axis=-1)) - half_l - half_l * np.abs(forward @ tf) - half_w * np.abs(forward @ tr),
        np.abs(np.sum(rel * starboard, axis=-1)) - half_w - half_l * np.abs(starboard @ tf) - half_w * np.abs(starboard @ tr),
        np.abs(rel @ tf) - half_l - half_l * np.abs(forward @ tf) - half_w * np.abs(starboard @ tf),
        np.abs(rel @ tr) - half_w - half_l * np.abs(forward @ tr) - half_w * np.abs(starboard @ tr),
    ])


class ReferenceController:
    def __init__(self, horizon_s: float = 16.0):
        self.horizon_s = float(horizon_s)
        self.offset = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_diagnostics = {}
        self.obligations = {}
        self.scan_memory = []
        self.frames = 0
        self.actuator_buffer = None
        self.servo_angle = 0.0
        self.executed_rudder = 0.0

    def action(self, env, observation):
        """Return [rudder, throttle] from onboard-available measurements."""
        x, y, heading = env.estimated_pose()
        u, v, yaw = env._measured_ego()
        path_state = env.path.project(x, y, heading)
        tangent = env.path.tangent(path_state.closest_idx).astype(float)
        right = np.array([tangent[1], -tangent[0]])
        base_heading = math.atan2(tangent[0], tangent[1])
        centre = np.asarray(path_state.target)
        lateral_now = float((np.array([x, y]) - centre) @ right)
        remaining_now = float((env.path.points[-1] - [x, y]) @ tangent)
        ranges = np.asarray(getattr(env, "gated_ranges", env.lidar.ranges))
        if not env.tracks and not self.scan_memory and not np.any(ranges < cfg.LIDAR_RANGE - 1e-5):
            self.offset = 0.0
            self.last_diagnostics = {"offset_m": 0.0, "return_after_s": 0.0, "speed_fraction": 1.0, "predicted_cost": 0.0, "tracks": 0}
            return self._command(env, heading, u, v, yaw, base_heading, lateral_now, remaining_now, 1.0)
        overtaking = any(str(getattr(ctx, "cls", "")) == "overtaking" for ctx in env.encounter_contexts.values())
        speeds = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5] if overtaking else [0.0, 0.5, 0.75, 1.0]
        offsets, fractions, return_times = np.meshgrid(np.arange(-3.5, 3.51, 0.5), speeds, [6.0, 10.0, np.inf])
        offsets, fractions, return_times = offsets.ravel(), fractions.ravel(), return_times.ravel()
        n = len(offsets)
        dt = 0.5
        steps = int(self.horizon_s / dt)
        trajectory = np.empty((steps, n, 2))
        angles = np.empty((steps, n))
        state = np.zeros((7, n))
        state[0:4] = np.array([max(0.0, u), v, math.radians(yaw), math.radians(heading)])[:, None]
        state[4] = self.servo_angle
        state[5:7] = np.array([x, y])[:, None]
        params = {key: np.full(n, value) for key, value in IDENTIFIED.items()}
        delay_steps = max(0, int(round(IDENTIFIED["rud_delay"] / 0.05)))
        delayed_commands = None if self.actuator_buffer is None else [np.full(n, value) for value in self.actuator_buffer]
        predicted_rudder = np.full(n, self.executed_rudder)
        command_limited = bool(getattr(env, "command_rate_limit", False))
        finished = np.zeros(n, dtype=bool)
        active = np.ones((steps, n), dtype=bool)
        for i in range(steps):
            active[i] = ~finished
            planned_offsets = np.where(i * dt >= return_times, 0.0, offsets)
            lateral = (state[5:7].T - centre) @ right
            remaining = remaining_now - (state[5:7].T - [x, y]) @ tangent
            recovering = (remaining < 5.0) & (np.abs(lateral) > 0.65) & (np.abs(planned_offsets) < 0.75)
            lookahead = np.where(recovering, 1.0, 2.5)
            sideslip = np.clip(np.arctan2(state[1], np.maximum(0.15, state[0])), -math.radians(20), math.radians(20))
            desired = base_heading + np.arctan2(planned_offsets - lateral, lookahead) - sideslip
            error = (desired - state[3] + np.pi) % (2 * np.pi) - np.pi
            rudder = np.clip(np.degrees(error) / 35.0 - np.degrees(state[2]) / 18.0, -1, 1)
            predicted_rudder = predicted_rudder + np.clip(rudder - predicted_rudder, -0.25, 0.25) if command_limited else rudder
            rpm = cfg.CRUISE_RPM * np.where(recovering, np.minimum(0.25, fractions), fractions)
            rpm = np.full(n, cfg.CRUISE_RPM) if cfg.FIXED_RPM else np.clip(rpm, cfg.RPM_FLOOR, cfg.RPM_CEIL)
            delta = -math.radians(40) * predicted_rudder
            if delayed_commands is None:
                delayed_commands = [delta.copy() for _ in range(delay_steps)]
            for substep in range(10):
                delayed_commands.append(delta)
                delayed = delayed_commands.pop(0)
                next_state = dyn.rk4_step(state, rpm, delayed, params, 0.05)
                state[:, ~finished] = next_state[:, ~finished]
            lateral = (state[5:7].T - centre) @ right
            remaining = remaining_now - (state[5:7].T - [x, y]) @ tangent
            goal_cte = np.hypot(lateral, np.maximum(0.0, -remaining))
            finished |= (remaining <= cfg.GOAL_ALONG_DIST - 0.15) & (goal_cte <= cfg.GOAL_CTE_RADIUS - 0.05)
            trajectory[i] = state[5:7].T
            angles[i] = state[3]

        progress = (trajectory[-1] - [x, y]) @ tangent
        terminal_lateral = (trajectory[-1] - centre) @ right
        cost = 0.60 * offsets ** 2 + 0.25 * (offsets - self.offset) ** 2 - 1.6 * progress + 1.4 * terminal_lateral ** 2
        feasible = np.ones(n, dtype=bool)
        rule_allowed = np.ones(n, dtype=bool)
        in_extremis = any(bool(getattr(ctx, "in_extremis", False)) for ctx in env.encounter_contexts.values())
        remaining_by_step = remaining_now - (trajectory - [x, y]) @ tangent
        lateral_by_step = (trajectory - centre) @ right
        projected_cte = np.hypot(lateral_by_step, np.maximum(0.0, -remaining_by_step))
        premature_goal = np.where(remaining_by_step <= cfg.GOAL_ALONG_DIST,
                                  np.maximum(0.0, projected_cte - (cfg.GOAL_CTE_RADIUS - 0.05)), 0.0)
        # The environment terminates at its broad goal gate. Rejoin before
        # crossing that gate, rather than planning to rejoin after termination.
        cost += 5000 * np.max(premature_goal ** 2, axis=0)
        # A convex corridor's half-plane clearance to the oriented collision
        # rectangle, including the simulator's hull inflation.
        polygon = np.asarray(env.boundary_polygon, dtype=float)
        a, b = polygon, np.roll(polygon, -1, axis=0)
        edges = b - a
        midpoint = polygon.mean(axis=0)
        inward = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        inward /= np.maximum(np.linalg.norm(inward, axis=1)[:, None], 1e-9)
        inward *= np.where(np.sum((midpoint - a) * inward, axis=1) >= 0, 1, -1)[:, None]
        boundary_gap = np.sum((trajectory[:, :, None, :] - a) * inward, axis=-1)
        forward = np.stack([np.sin(angles), np.cos(angles)], axis=-1)
        starboard = np.stack([np.cos(angles), -np.sin(angles)], axis=-1)
        support = ((0.5 * VESSEL_LENGTH + HULL_MARGIN) * np.abs(forward @ inward.T)
                   + (0.5 * VESSEL_WIDTH + HULL_MARGIN) * np.abs(starboard @ inward.T))
        boundary_gap -= support
        # Finished predictions are held at the goal above, so end walls now
        # constrain overshooting while displaced from the reference path.
        gap = np.min(boundary_gap, axis=(0, 2))
        cost += 4000 * np.maximum(0.0, 0.10 - gap) ** 2
        feasible &= gap >= 0.10

        mask = ranges < cfg.LIDAR_RANGE - 1e-5
        bearings = np.radians(env.lidar.bearings[mask] + heading)
        origin = np.array([x, y]) + LIDAR_OFFSET_M * np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))])
        points = origin + ranges[mask, None] * np.stack([np.sin(bearings), np.cos(bearings)], axis=1)
        # Dynamic returns are predicted at their estimated velocity below.
        tracks = list(env.tracks)
        for track in tracks:
            if len(points):
                points = points[np.linalg.norm(points - track.position, axis=1) > 1.8]
        self.frames += 1
        if len(points):
            self.scan_memory.append((self.frames, points))
        self.scan_memory = [(stamp, cloud) for stamp, cloud in self.scan_memory if self.frames - stamp < 20]
        points = np.concatenate([cloud for _, cloud in self.scan_memory], axis=0) if self.scan_memory else np.empty((0, 2))
        for track in tracks:
            if len(points):
                points = points[np.linalg.norm(points - track.position, axis=1) > 2.0]
        if len(points):
            _, indices = np.unique(np.round(points / 0.25).astype(int), axis=0, return_index=True)
            points = points[indices]
            rel = points[None, None, :, :] - trajectory[:, :, None, :]
            longitudinal = rel[..., 0] * np.sin(angles)[:, :, None] + rel[..., 1] * np.cos(angles)[:, :, None]
            lateral = rel[..., 0] * np.cos(angles)[:, :, None] - rel[..., 1] * np.sin(angles)[:, :, None]
            ell = np.sqrt((longitudinal / 1.40) ** 2 + (lateral / 1.10) ** 2)
            clearance = np.min(np.where(active[:, :, None], ell, np.inf), axis=(0, 2))
            cost += 2000 * np.maximum(0, 1.0 - clearance) ** 2 + 8 * np.maximum(0, 1.4 - clearance) ** 2
            feasible &= clearance >= 1.0

        target_gaps = np.full(n, np.inf)
        for track in tracks:
            fitted_centre = getattr(track, "last_fit_centre", None)
            target_centre = track.position if fitted_centre is None else np.asarray(fitted_centre)
            future = target_centre + np.arange(1, steps + 1)[:, None] * dt * track.velocity
            rel = future[:, None, :] - trajectory
            clearance = np.min(np.where(active, np.linalg.norm(rel, axis=2), np.inf), axis=0)
            ctx = env.encounter_contexts.get(track.id)
            stand_on = ctx is not None and str(getattr(ctx, "cls", "")) == "being_overtaken"
            fitted_heading = getattr(track, "last_fit_heading_deg", None)
            target_heading = math.atan2(float(track.velocity[0]), float(track.velocity[1])) if fitted_heading is None else math.radians(fitted_heading)
            # Separating-axis gap between oriented inflated rectangles. A
            # positive max gap separates the hulls; requiring0.4m leaves room
            # for tracking/model error without pretending every vessel is a
            # 2.6m centre-radius obstacle in every direction.
            separation = hull_separation(trajectory, angles, future[:, None, :], target_heading)
            hull_gap = np.min(np.where(active, separation, np.inf), axis=0)
            target_gaps = np.minimum(target_gaps, hull_gap)
            feasible &= hull_gap >= 0.4
            cost += 8000 * np.maximum(0, 0.4 - hull_gap) ** 2 + 15 * np.maximum(0, 2.6 - clearance) ** 2
            if stand_on:
                cost += 25 * offsets ** 2
            if ctx is not None and ctx.gives_way and ctx.tcpa > 0 and ctx.dcpa < 3.0:
                sense = int(ctx.compliant_turn_sense)
                cls = str(getattr(ctx, "cls", ""))
                if sense == 0:
                    sense = {"head_on": 1, "overtaking": -1}.get(cls, 0)
                    if cls == "crossing":
                        sense = -1 if float(track.velocity @ right) > 0 else 1
                if sense:
                    self.obligations.setdefault(track.id, (cls, sense))
            obligation = self.obligations.get(track.id)
            if ctx is not None and obligation is not None:
                active_obligation = str(getattr(ctx, "state", "")) in ("engaged", "clearing")
                active_obligation |= bool(ctx.gives_way and ctx.tcpa > 0 and ctx.dcpa < 3.0)
                if active_obligation:
                    rule_allowed &= int(obligation[1]) * offsets >= 0.0
                elif str(getattr(ctx, "state", "")) == "idle" and ctx.tcpa <= 0:
                    self.obligations.pop(track.id, None)
                    obligation = None
            if obligation is not None and obligation[0] == "overtaking":
                if float((track.position - [x, y]) @ tangent) > -1.0:
                    # Commit to a complete pass instead of following forever
                    # because returning to the path dominates a short horizon.
                    cost -= 1.4 * terminal_lateral ** 2
                    cost += 30 * np.maximum(0, offsets) ** 2
                    cost -= 18 * fractions

        eligible = feasible & rule_allowed
        stand_on_active = any(str(getattr(ctx, "cls", "")) == "being_overtaken"
            and str(getattr(ctx, "state", "")) in ("engaged", "clearing")
            for ctx in env.encounter_contexts.values())
        # A safe stand-on course and cruise command take precedence over soft
        # clearance/progress costs. Keep avoidance available if holding them
        # has no feasible prediction, and expose that exception explicitly.
        stand_on_hold = eligible & (offsets == 0.0) & (fractions == 1.0)
        stand_on_override = stand_on_active and not bool(np.any(stand_on_hold))
        if stand_on_active and np.any(stand_on_hold):
            eligible = stand_on_hold
        rule_relaxed = False
        if not np.any(eligible):
            if in_extremis and np.any(feasible):
                eligible = feasible
                rule_relaxed = True
            else:
                # No certified candidate: return the least-cost permitted
                # candidate and expose fallback, never silently call it safe.
                eligible = rule_allowed
        best = int(np.argmin(np.where(eligible, cost, np.inf)))
        self.offset = float(offsets[best])
        self.last_diagnostics = {"offset_m": self.offset, "return_after_s": float(return_times[best]), "speed_fraction": float(fractions[best]), "predicted_cost": float(cost[best]), "tracks": len(tracks),
            "feasible_candidates": int(np.sum(feasible & rule_allowed)), "fallback": not bool(feasible[best]), "rule_relaxed": rule_relaxed,
            "stand_on_active": stand_on_active, "stand_on_override": stand_on_override,
            "predicted_target_hull_gap": float(target_gaps[best]) if tracks else None}
        return self._command(env, heading, u, v, yaw, base_heading, lateral_now, remaining_now, float(fractions[best]))

    def _command(self, env, heading, u, v, yaw, base_heading, lateral_now, remaining_now, speed_fraction):
        # LOS specifies ground course. Counteract measured sideslip when
        # converting it to a heading, otherwise return-to-path is too slow.
        sideslip = float(np.clip(math.degrees(math.atan2(v, max(0.15, u))), -20, 20))
        recovering_near_goal = remaining_now < 5.0 and abs(lateral_now) > 0.65
        lookahead = 1.0 if recovering_near_goal and abs(self.offset) < 0.75 else 2.5
        desired_heading = math.degrees(base_heading + math.atan2(self.offset - lateral_now, lookahead)) - sideslip
        error = wrap180(desired_heading - heading)
        rudder = float(np.clip(error / 35.0 - yaw / 18.0, -1.0, 1.0))
        rpm = cfg.CRUISE_RPM * speed_fraction
        if recovering_near_goal and abs(self.offset) < 0.75:
            rpm = min(rpm, 0.25 * cfg.CRUISE_RPM)
        throttle = float(np.clip((rpm - cfg.CRUISE_RPM) / cfg.RPM_DELTA, -1, 1))
        self.last_action = np.array([rudder, throttle], dtype=np.float32)
        command_limited = bool(getattr(env, "command_rate_limit", False))
        self.executed_rudder = self.executed_rudder + float(np.clip(rudder - self.executed_rudder, -0.25, 0.25)) if command_limited else rudder
        delta = -math.radians(40) * self.executed_rudder
        if self.actuator_buffer is None:
            self.actuator_buffer = [delta] * max(0, int(round(IDENTIFIED["rud_delay"] / 0.05)))
        for _ in range(10):
            self.actuator_buffer.append(delta)
            delayed = self.actuator_buffer.pop(0)
            self.servo_angle = float(dyn.advance_rudder(np.array([self.servo_angle]), np.array([delayed]), IDENTIFIED, 0.05)[0])
        return self.last_action.copy()
