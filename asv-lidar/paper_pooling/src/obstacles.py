"""Random obstacle layouts for training episodes.

Five layout families are sampled with the weights in `TRAIN_SCENARIO_PROBS`:

normal        one or more obstacles placed near the reference path
target_side   the path-recovery side is left passable, the wide side is not
field_repair  perturbations of a recorded layout the policy used to fail on
gate          two obstacles either side of the path, leaving a drivable gap
offpath       distractors far enough off the path to be ignored

All of them draw from the global `np.random` stream, which `ASVLidarEnv.reset`
seeds, so a seeded reset reproduces the exact layout.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

import config as cfg
from path import ReferencePath

Box = List[Tuple[float, float]]

# Placement limits shared by several generators.
MIN_START_GOAL_GAP = 2.5      # keep obstacles clear of the start/goal regions
NEAR_PATH_START_GOAL_GAP = 3.0
MAP_EDGE_MARGIN = 0.05


def make_box(cx: float, cy: float, size: float) -> Box:
    """Axis-aligned square obstacle centred on (cx, cy)."""
    h = 0.5 * float(size)
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


def boxes_overlap(a: Box, b: Box, pad: float = 0.15) -> bool:
    ax = [p[0] for p in a]
    ay = [p[1] for p in a]
    bx = [p[0] for p in b]
    by = [p[1] for p in b]
    return not (
        max(ax) + pad < min(bx)
        or max(bx) + pad < min(ax)
        or max(ay) + pad < min(by)
        or max(by) + pad < min(ay)
    )


class ObstacleSampler:
    """Draws obstacle layouts around a given reference path."""

    def __init__(self, path: ReferencePath, map_width: float, map_height: float,
                 start: Tuple[float, float], goal: Tuple[float, float]) -> None:
        self.path = path
        self.map_width = float(map_width)
        self.map_height = float(map_height)
        self.start = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))
        self.mode_used = "normal"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def sample(self, num_obs: int) -> List[Box]:
        num_obs = max(0, int(num_obs))
        if num_obs <= 0:
            self.mode_used = "none"
            return []

        probs = np.asarray(cfg.TRAIN_SCENARIO_PROBS, dtype=np.float64)
        probs = probs / np.sum(probs)
        self.mode_used = str(np.random.choice(cfg.TRAIN_SCENARIO_MODES, p=probs))

        if self.mode_used == "target_side":
            return self.target_side(num_obs)
        if self.mode_used == "gate":
            return self.gate(num_obs)
        if self.mode_used == "field_repair":
            return self.field_repair(num_obs)
        if self.mode_used == "offpath":
            layout = self._fill([], num_obs, self.one_offpath, tries=300)
            return layout if len(layout) >= num_obs else self.normal(num_obs)
        return self.normal(num_obs)

    # ------------------------------------------------------------------
    # Placement checks
    # ------------------------------------------------------------------
    def inside_map(self, obs: Box, margin: float = MAP_EDGE_MARGIN) -> bool:
        xs = [p[0] for p in obs]
        ys = [p[1] for p in obs]
        return (min(xs) >= margin and max(xs) <= self.map_width - margin
                and min(ys) >= margin and max(ys) <= self.map_height - margin)

    def _accept(self, layout: List[Box], obs: Box, pad: float = 0.25) -> bool:
        """Append `obs` to `layout` if it fits, does not clash, and clears start/goal."""
        if not self.inside_map(obs):
            return False
        if any(boxes_overlap(obs, existing, pad=pad) for existing in layout):
            return False

        cx = float(np.mean([p[0] for p in obs]))
        cy = float(np.mean([p[1] for p in obs]))
        if np.hypot(cx - self.start[0], cy - self.start[1]) < MIN_START_GOAL_GAP:
            return False
        if np.hypot(cx - self.goal[0], cy - self.goal[1]) < MIN_START_GOAL_GAP:
            return False

        layout.append(obs)
        return True

    def _fill(self, layout: List[Box], num_obs: int, make_candidate, tries: int, pad: float = 0.25) -> List[Box]:
        """Top a layout up to `num_obs` using a single-obstacle generator."""
        for _ in range(tries):
            if len(layout) >= num_obs:
                break
            candidate = make_candidate()
            if candidate:
                self._accept(layout, candidate[0], pad=pad)
        return layout

    # ------------------------------------------------------------------
    # Single-obstacle generators
    # ------------------------------------------------------------------
    def one_near_path(self) -> List[Box]:
        """One obstacle on or slightly beside the reference path."""
        s_total = self.path.length
        if s_total <= 1e-6:
            return []
        feasible = np.where(
            (self.path.s >= cfg.OBSTACLE_PATH_START_FRAC * s_total)
            & (self.path.s <= cfg.OBSTACLE_PATH_END_FRAC * s_total)
        )[0]
        if feasible.size == 0:
            return []

        for _ in range(100):
            idx = int(np.random.choice(feasible))
            if np.random.rand() < cfg.OBSTACLE_CENTER_PROB:
                lateral = 0.0
            else:
                side = -1.0 if np.random.rand() < 0.5 else 1.0
                lateral = side * float(np.random.uniform(
                    cfg.OBSTACLE_LATERAL_OFFSET_MIN, cfg.OBSTACLE_LATERAL_OFFSET_MAX))

            centre = self.path.points[idx].astype(np.float32) + lateral * self.path.left_normal(idx)
            cx, cy = float(centre[0]), float(centre[1])

            margin = 0.5 * cfg.OBSTACLE_SIZE + 0.25
            if not (margin <= cx <= self.map_width - margin and margin <= cy <= self.map_height - margin):
                continue
            if (np.hypot(cx - self.start[0], cy - self.start[1]) > NEAR_PATH_START_GOAL_GAP
                    and np.hypot(cx - self.goal[0], cy - self.goal[1]) > NEAR_PATH_START_GOAL_GAP):
                return [make_box(cx, cy, cfg.OBSTACLE_SIZE)]
        return []

    def one_offpath(self) -> List[Box]:
        """One distractor obstacle, far enough off the path to be ignorable."""
        for _ in range(100):
            frac = float(np.random.uniform(cfg.OBSTACLE_PATH_START_FRAC, cfg.OBSTACLE_PATH_END_FRAC))
            centre, tangent, normal_left = self.path.frame_at_frac(frac)
            side = -1.0 if np.random.rand() < 0.5 else 1.0
            lateral = side * float(np.random.uniform(cfg.OFFPATH_LATERAL_MIN, cfg.OFFPATH_LATERAL_MAX))
            along = float(np.random.uniform(-0.50, 0.50))
            p = centre + along * tangent + lateral * normal_left

            obs = make_box(float(p[0]), float(p[1]), cfg.OBSTACLE_SIZE)
            if self.inside_map(obs):
                return [obs]
        return []

    # ------------------------------------------------------------------
    # Layout families
    # ------------------------------------------------------------------
    def normal(self, num_obs: int) -> List[Box]:
        """Independent near-path obstacles, rejecting overlaps."""
        num_obs = max(0, int(num_obs))
        layout: List[Box] = []
        for _ in range(300):
            if len(layout) >= num_obs:
                break
            candidate = self.one_near_path()
            if not candidate:
                continue
            if any(boxes_overlap(candidate[0], existing, pad=0.25) for existing in layout):
                continue
            layout.append(candidate[0])
        return layout

    def gate(self, num_obs: int) -> List[Box]:
        """Two obstacles either side of the path, leaving a drivable gap."""
        num_obs = max(0, int(num_obs))
        if num_obs <= 0:
            return []
        if num_obs == 1:
            return self.one_near_path()

        layout: List[Box] = []
        for _ in range(80):
            frac = float(np.random.uniform(*cfg.GATE_PATH_FRAC_RANGE))
            centre, tangent, normal_left = self.path.frame_at_frac(frac)

            gap = float(np.random.uniform(*cfg.GATE_GAP_RANGE))
            extra = float(np.random.uniform(*cfg.GATE_LATERAL_EXTRA))
            lateral_mag = 0.5 * gap + 0.5 * cfg.OBSTACLE_SIZE + extra
            along_jitter = float(np.random.uniform(-cfg.GATE_CENTER_JITTER_ALONG, cfg.GATE_CENTER_JITTER_ALONG))
            lat_jitter = float(np.random.uniform(-cfg.GATE_CENTER_JITTER_LATERAL, cfg.GATE_CENTER_JITTER_LATERAL))

            base = centre + along_jitter * tangent
            left = base + (lateral_mag + lat_jitter) * normal_left
            right = base - (lateral_mag - lat_jitter) * normal_left

            trial: List[Box] = []
            left_ok = self._accept(trial, make_box(float(left[0]), float(left[1]), cfg.OBSTACLE_SIZE), pad=0.15)
            if left_ok and self._accept(trial, make_box(float(right[0]), float(right[1]), cfg.OBSTACLE_SIZE), pad=0.15):
                layout = trial
                break

        if len(layout) < min(2, num_obs):
            return self.normal(num_obs)

        # Any extra obstacles are distractors, not new centre blockers.
        for _ in range(200):
            if len(layout) >= num_obs:
                break
            candidate = self.one_offpath() if np.random.rand() < 0.60 else self.one_near_path()
            if candidate:
                self._accept(layout, candidate[0], pad=0.25)
        return layout[:num_obs]

    def field_repair(self, num_obs: int) -> List[Box]:
        """Perturbations of the recorded side-choice failure layout."""
        num_obs = max(0, int(num_obs))
        if num_obs <= 0:
            return []
        if num_obs == 1:
            return self.one_near_path()

        layout: List[Box] = []
        for base_frac, base_lat in list(zip(cfg.FIELD_REPAIR_PATH_FRACS, cfg.FIELD_REPAIR_LATERALS))[:num_obs]:
            jitter = np.random.uniform(-cfg.FIELD_REPAIR_FRAC_JITTER, cfg.FIELD_REPAIR_FRAC_JITTER)
            frac = float(np.clip(base_frac + jitter, 0.25, 0.75))
            centre, tangent, normal_left = self.path.frame_at_frac(frac)

            lateral = float(base_lat + np.random.uniform(-cfg.FIELD_REPAIR_LAT_JITTER, cfg.FIELD_REPAIR_LAT_JITTER))
            along = float(np.random.uniform(-0.35, 0.35))
            p = centre + along * tangent + lateral * normal_left
            size = float(np.random.uniform(0.85, 1.05) * cfg.OBSTACLE_SIZE)
            self._accept(layout, make_box(float(p[0]), float(p[1]), size), pad=0.20)

        for _ in range(200):
            if len(layout) >= num_obs:
                break
            candidate = self.one_near_path() if np.random.rand() < 0.5 else self.one_offpath()
            if candidate:
                self._accept(layout, candidate[0], pad=0.25)

        return layout[:num_obs] if len(layout) >= num_obs else self.normal(num_obs)

    def target_side(self, num_obs: int) -> List[Box]:
        """Leave the path-recovery side passable and block the wide side.

        This targets the failure mode where the policy commits to a wide bypass
        even though the corridor on the path side is open.
        """
        num_obs = max(0, int(num_obs))
        if num_obs <= 0:
            return []
        if num_obs == 1:
            return self._one_target_side(num_obs)

        # For a vertical path the left normal points roughly along -x, so
        # open_sign = -1 leaves a starboard corridor and +1 mirrors the layout.
        open_sign = -1.0 if np.random.rand() < cfg.TARGET_SIDE_RIGHT_PROB else +1.0
        blocked_sign = -open_sign

        frac0 = float(np.random.uniform(*cfg.TARGET_SIDE_PATH_FRAC_RANGE))
        centre0, tangent0, normal0 = self.path.frame_at_frac(frac0)

        # A near-path obstacle that nudges the vessel without closing the
        # corridor, then two farther obstacles that make the wrong side tempting.
        lateral0 = blocked_sign * float(np.random.uniform(0.05, 0.35))
        p0 = centre0 + float(np.random.uniform(-0.25, 0.25)) * tangent0 + lateral0 * normal0
        candidates = [(p0, float(np.random.uniform(0.85, 1.05) * cfg.OBSTACLE_SIZE))]

        for frac_shift, lat_range, sign in [
            (0.12, cfg.TARGET_SIDE_BLOCKED_OFFSET_RANGE, blocked_sign),
            (0.17, cfg.TARGET_SIDE_CORRIDOR_OFFSET_RANGE, open_sign),
        ]:
            frac = float(np.clip(frac0 + frac_shift + np.random.uniform(-0.035, 0.035), 0.28, 0.78))
            centre, tangent, normal_left = self.path.frame_at_frac(frac)
            lateral = sign * float(np.random.uniform(*lat_range))
            lateral += float(np.random.uniform(-cfg.TARGET_SIDE_LATERAL_JITTER, cfg.TARGET_SIDE_LATERAL_JITTER))
            along = float(np.random.uniform(-cfg.TARGET_SIDE_ALONG_JITTER, cfg.TARGET_SIDE_ALONG_JITTER))
            candidates.append((centre + along * tangent + lateral * normal_left,
                               float(np.random.uniform(0.85, 1.05) * cfg.OBSTACLE_SIZE)))

        layout: List[Box] = []
        for p, size in candidates[:min(num_obs, len(candidates))]:
            self._accept(layout, make_box(float(p[0]), float(p[1]), size), pad=0.20)

        self._fill(layout, num_obs, self.one_offpath, tries=200)
        return layout[:num_obs] if len(layout) >= num_obs else self.gate(num_obs)

    def _one_target_side(self, num_obs: int) -> List[Box]:
        """Single slightly off-centre obstacle, so the clearer side is a choice."""
        layout: List[Box] = []
        for _ in range(80):
            frac = float(np.random.uniform(*cfg.TARGET_SIDE_PATH_FRAC_RANGE))
            centre, tangent, normal_left = self.path.frame_at_frac(frac)
            side_sign = -1.0 if np.random.rand() < cfg.TARGET_SIDE_RIGHT_PROB else +1.0
            lateral = side_sign * float(np.random.uniform(0.35, 0.75))
            along = float(np.random.uniform(-cfg.TARGET_SIDE_ALONG_JITTER, cfg.TARGET_SIDE_ALONG_JITTER))
            p = centre + along * tangent + lateral * normal_left
            size = float(np.random.uniform(0.85, 1.05) * cfg.OBSTACLE_SIZE)
            if self._accept(layout, make_box(float(p[0]), float(p[1]), size), pad=0.20):
                return layout
        return self.normal(num_obs)
