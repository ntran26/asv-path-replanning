"""Field-layout scenarios for fine-tuning (2026-09-27): Paper 2-style layouts with and
without a target, drawn fresh, for the 2 M -> 3 M fine-tune (`finetune_field.py`).

The Paper 2 deployment-layout set (`paper2_set.py`) showed the baseline policies
fail when an encounter happens beside static panels (F105): training keeps
+-0.4 T_0 of own travel around the CPA clear of panels, and the deployment
layouts put 90 % of encounters within 3 m of one.  These scenarios are the
missing experience, drawn from the same *family* as the deployment layouts but
never the layouts themselves:

* **Layout.** A Paper 2 leg -- start y = 2, goal y = 22, both x in [2, 8],
  slant at most `MAX_SLANT_DEG` -- and three 1 m panels arranged as the published
  scenarios are (a gate across the leg, a panel on the leg, panels off to the
  side), each jittered, at least `PANEL_WALL_M` from the walls and
  `PANEL_GAP_M` apart, with an A* route from start to goal (F74's test).
* **Fairness.** A layout within `EXCLUDE_M` of one of the three deployment
  layouts (leg ends and all three panels matched) is redrawn, so the test set
  stays unseen.  Training draws from the training namespace; the validation set
  from its own seed block (`VALIDATION_SEED_BASE`), disjoint from both.
* **Encounter.** No target (`P_NO_TARGET`), or head-on, crossing from port or
  starboard, overtaking or being overtaken, generated on the leg with the panels
  fixed -- so **no CPA guard**: the encounter may happen beside a panel -- and the
  field rules of the deployment set: the target starts inside the basin, stops
  short of the wall, keeps 0.20-1.10 m/s, and its straight track clears every
  panel by 0.3 m.  `P_VARYING_SPEED` of targets change speed once (`T-VS`).
* **Near-deployment layouts** (baseline-v3 only, `near_share`; 2026-09-28).  The
  family above rarely lands close to a deployment layout (0.4 % of draws within
  1.5 m).  A near draw takes one of the three deployment layouts and moves every
  panel `NEAR_PANEL_SHIFT_M` and each leg end up to `NEAR_END_SHIFT_M`, keeping
  it `NEAR_DISTANCE_M` from the original (the largest panel or leg-end mismatch)
  -- similar to the field, never the tested layout itself.

Parameters are recorded in `configs/finetune_field_v1.json`; none is a
baseline-v2 constant, so the base formulation's digest is untouched.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import constant_temp as ct
import constants as cfg
import feasibility as feas
import paper2_set as p2
import scenario as scn
import targets as tgt
from corridor import nav_polygon

REVISION = "1.0"

# Layout family.
LEG_X_RANGE = (2.0, 8.0)                 # the deployment legs' span (x = 2 ... 8)
MAX_SLANT_DEG = 14.1                     # the deployment legs' slant
PANEL_FRACTION = (0.28, 0.85)            # along the leg (y about 7.6 ... 19)
PANEL_WALL_M = 0.9                       # panel centre to wall (0.4 m of water beside a 1 m panel)
PANEL_GAP_M = 0.4                        # between panels (edge to edge)
GATE_HALF_GAP = (1.3, 2.7)               # gate: each panel this far either side of the leg
SIDE_OFFSET = (1.2, 3.2)                 # a side panel's lateral offset
ONPATH_OFFSET = 0.5                      # an on-path panel's largest offset
MOTIFS = {"gate+onpath": 0.4, "slalom": 0.3, "side+onpath": 0.3}
EXCLUDE_M = 0.75                         # near-duplicate of a deployment layout
NEAR_PANEL_SHIFT_M = (0.4, 1.5)          # near draw: every panel moves this far
NEAR_END_SHIFT_M = 0.75                  # near draw: each leg end moves up to this (along x)
NEAR_DISTANCE_M = (1.0, 2.0)             # near draw: its distance from the deployment layout

# Encounters.
P_NO_TARGET = 0.20
ENCOUNTER_CODES = ("HO", "CRP", "CRS", "OT", "BO")
P_VARYING_SPEED = 0.25
OWN_START_PANEL_CLEAR_M = 1.3            # own-ship spawn to any panel centre
MAX_DRAWS = 40                           # target draws per layout before settling for no target

# Validation (fixed; seeds 350,000+, a gap no namespace uses).
VALIDATION_SEED_BASE = 350_000
VALIDATION_NO_TARGET = 30
VALIDATION_PER_ENCOUNTER = 25


def _unit(start, goal):
    d = np.asarray(goal, float) - np.asarray(start, float)
    return d / np.linalg.norm(d)


def layout_distance(start, goal, centres, lay) -> float:
    """How far a layout is from deployment layout `lay`: the largest leg-end
    offset or panel mismatch (each deployment panel to the nearest panel)."""
    ends = max(np.hypot(*np.subtract(start, lay["start"])), np.hypot(*np.subtract(goal, lay["goal"])))
    return float(max(ends, max(min(np.hypot(*np.subtract(c, q)) for q in centres) for c in lay["panels"])))


def _near_deployment_layout(start, goal, panels) -> bool:
    for lay in p2.LAYOUTS.values():
        if (np.hypot(*(np.subtract(start, lay["start"]))) > EXCLUDE_M
                or np.hypot(*(np.subtract(goal, lay["goal"]))) > EXCLUDE_M):
            continue
        if all(min(np.hypot(*(np.subtract(c, q))) for q in panels) <= EXCLUDE_M for c in lay["panels"]):
            return True
    return False


def sample_near_layout(rng) -> Optional[dict]:
    """One layout close to a deployment layout (see the module docstring), or
    None if this draw failed a check."""
    key = str(rng.choice(sorted(p2.LAYOUTS)))
    lay = p2.LAYOUTS[key]
    max_dx = 20.0 * math.tan(math.radians(MAX_SLANT_DEG))
    xs = float(np.clip(lay["start"][0] + rng.uniform(-NEAR_END_SHIFT_M, NEAR_END_SHIFT_M), *LEG_X_RANGE))
    xg = float(np.clip(lay["goal"][0] + rng.uniform(-NEAR_END_SHIFT_M, NEAR_END_SHIFT_M), *LEG_X_RANGE))
    xg = float(np.clip(xg, xs - max_dx, xs + max_dx))
    start, goal = (xs, float(cfg.BASIN_START_Y)), (xg, float(cfg.BASIN_GOAL_Y))
    centres = []
    for c in lay["panels"]:
        r, a = rng.uniform(*NEAR_PANEL_SHIFT_M), rng.uniform(0.0, 2.0 * math.pi)
        centres.append((float(c[0] + r * math.cos(a)), float(c[1] + r * math.sin(a))))
    if not NEAR_DISTANCE_M[0] <= layout_distance(start, goal, centres, lay) <= NEAR_DISTANCE_M[1]:
        return None
    return _checked(start, goal, centres, f"near-{key}")


def _checked(start, goal, centres, motif) -> Optional[dict]:
    """Walls, panel gaps, not a near-duplicate, an A* route: the layout, or None."""
    for c in centres:
        if not (PANEL_WALL_M <= c[0] <= cfg.MAP_WIDTH - PANEL_WALL_M
                and PANEL_WALL_M + 3.0 <= c[1] <= cfg.MAP_HEIGHT - PANEL_WALL_M - 2.0):
            return None
    for i in range(3):
        for j in range(i + 1, 3):
            if max(abs(centres[i][0] - centres[j][0]), abs(centres[i][1] - centres[j][1])) < p2.PANEL_SIZE_M + PANEL_GAP_M:
                return None
    if _near_deployment_layout(start, goal, centres):
        return None
    h = 0.5 * p2.PANEL_SIZE_M
    panels = [[(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)] for x, y in centres]
    if not feas.layout_feasible(start, goal, nav_polygon((cfg.MAP_WIDTH, cfg.MAP_HEIGHT)), panels):
        return None
    return {"start": start, "goal": goal, "centres": centres, "panels": panels, "motif": str(motif)}


def sample_layout(rng) -> Optional[dict]:
    """One Paper 2-style layout, or None if this draw failed a check."""
    xs = float(rng.uniform(*LEG_X_RANGE))
    max_dx = 20.0 * math.tan(math.radians(MAX_SLANT_DEG))
    xg = float(np.clip(xs + rng.uniform(-max_dx, max_dx), *LEG_X_RANGE))
    start, goal = (xs, float(cfg.BASIN_START_Y)), (xg, float(cfg.BASIN_GOAL_Y))
    t = _unit(start, goal)
    n = np.array([t[1], -t[0]])                          # starboard of the leg
    length = float(np.hypot(xg - xs, goal[1] - start[1]))
    point = lambda f, off: np.asarray(start) + f * length * t + off * n
    motif = rng.choice(list(MOTIFS), p=np.array(list(MOTIFS.values())) / sum(MOTIFS.values()))
    f = np.sort(rng.uniform(*PANEL_FRACTION, size=2))
    if motif == "gate+onpath":
        g = float(rng.uniform(*GATE_HALF_GAP))
        centres = [point(f[0], -g), point(f[0] + rng.uniform(-0.02, 0.02), g),
                   point(f[1], rng.uniform(-ONPATH_OFFSET, ONPATH_OFFSET))]
    elif motif == "slalom":
        fs = np.sort(rng.uniform(*PANEL_FRACTION, size=3))
        side = rng.choice([-1.0, 1.0])
        centres = [point(fs[k], side * (-1) ** k * rng.uniform(0.3, SIDE_OFFSET[1])) for k in range(3)]
    else:                                                # side panels flanking, one on the path
        fs = np.sort(rng.uniform(*PANEL_FRACTION, size=3))
        order = rng.permutation(3)
        centres = [point(fs[order[0]], rng.uniform(-ONPATH_OFFSET, ONPATH_OFFSET)),
                   point(fs[order[1]], -rng.uniform(*SIDE_OFFSET)),
                   point(fs[order[2]], rng.uniform(*SIDE_OFFSET))]
    return _checked(start, goal, [tuple(map(float, c)) for c in centres], motif)


def _straight_clearance(built, panels, horizon_s: float) -> float:
    """Panel clearance of a constant-heading target, stopped at the wall box (fast)."""
    box = p2.stop_box()
    x, y = map(float, built.target_spawn)
    a = math.radians(float(built.target_heading))
    step = 0.5 * float(built.target_speed)
    best = np.inf
    for _ in range(int(horizon_s / 0.5) + 1):
        hull = tgt.hull_polygon(x, y, built.target_heading)
        xs, ys = zip(*hull)
        if min(xs) < box[0] or max(xs) > box[1] or min(ys) < box[2] or max(ys) > box[3]:
            break
        best = min(best, min(p2.polygon_distance(hull, p) for p in panels))
        x, y = x + step * math.sin(a), y + step * math.cos(a)
    return best


def _inside_box(built) -> bool:
    box = p2.stop_box()
    xs, ys = zip(*tgt.hull_polygon(*map(float, built.target_spawn), built.target_heading))
    return box[0] <= min(xs) and max(xs) <= box[1] and box[2] <= min(ys) and max(ys) <= box[3]


def sample(rng, namespace: str = "training", generator: Optional[scn.ScenarioGenerator] = None,
           encounter: Optional[str] = None, varying: Optional[bool] = None,
           seed_fn=None, weights: Optional[Dict[str, float]] = None,
           varying_share: Optional[float] = None, solvable_only: bool = False,
           near_share: float = 0.0, near: Optional[bool] = None) -> scn.Scenario:
    """One field-layout scenario.  `encounter` / `varying` force the draw (the
    validation set); otherwise they are drawn -- from `weights` over "NT" and the
    encounter codes when given (finetune-field-v2's failure weighting), else no
    target at `P_NO_TARGET` and the encounters evenly.  `seed_fn(k)` gives the
    k-th generator seed (default: from `rng` inside `namespace`).  `varying_share`
    overrides `P_VARYING_SPEED`; `solvable_only` (baseline-v3) redraws any target
    scenario the space-time check (`feasibility_st.py`) finds unsolvable.
    `near_share` (baseline-v3) draws that share of layouts near a deployment
    layout; `near` forces it either way (the v3 development set).  With neither,
    the draw -- and its random stream -- is v1/v2's exactly."""
    generator = generator or scn.ScenarioGenerator(stage=5, seed_namespace=namespace)
    lo_ns, hi_ns = cfg.SEED_NAMESPACES[namespace] if namespace in cfg.SEED_NAMESPACES else (0, 99_999)
    seed_fn = seed_fn or (lambda k: int(rng.integers(lo_ns, hi_ns + 1)))
    k = 0
    while True:
        layout = None
        use_near = near if near is not None else (near_share > 0.0 and rng.uniform() < near_share)
        while layout is None:
            layout = sample_near_layout(rng) if use_near else sample_layout(rng)
        flags = {"basin_leg": [list(layout["start"]), list(layout["goal"])],
                 "fixed_obstacles": [[list(p) for p in poly] for poly in layout["panels"]],
                 "set": "field_training", "motif": layout["motif"]}
        if encounter is not None:
            code = encounter
        elif weights:
            keys = list(weights)
            w = np.array([float(weights[k]) for k in keys])
            code = str(keys[int(rng.choice(len(keys), p=w / w.sum()))])
        else:
            code = "NT" if rng.uniform() < P_NO_TARGET else str(rng.choice(ENCOUNTER_CODES))
        if code == "NT":
            built = generator.sample(seed_fn(k), encounter_class="no_target", geometry_mode="basin", flags=flags)
            k += 1
            if built is not None:
                return built
            continue
        cls, side = p2.ENCOUNTERS[code]
        extra = {"side": side} if side else {}
        for _ in range(MAX_DRAWS):
            seed = seed_fn(k)
            k += 1
            built = generator.sample(seed, encounter_class=cls, behaviour=tgt.T_CV, geometry_mode="basin",
                                     flags=dict(flags, target_stop_box=p2.stop_box(), **extra))
            if built is None:
                continue
            lo, hi = ct.FIELD_TARGET_SPEED_RANGE
            if not lo <= float(built.target_speed) <= hi or not _inside_box(built):
                continue
            if min(np.hypot(*(np.subtract(built.own_spawn, c))) for c in layout["centres"]) < OWN_START_PANEL_CLEAR_M:
                continue
            horizon = float(built.tcpa_s) + ct.FIELD_POST_CPA_S + 10.0
            if _straight_clearance(built, layout["panels"], horizon) < ct.TARGET_PANEL_CLEARANCE_M:
                continue
            share = P_VARYING_SPEED if varying_share is None else float(varying_share)
            vary = varying if varying is not None else rng.uniform() < share
            if vary:
                prof = p2.speed_profile(built, seed)          # final speed kept in the field range
                if prof is None:
                    continue
                built = copy.deepcopy(built)
                built.target_behaviour = tgt.T_VS
                built.flags = dict(built.flags, speed_profile=prof)
            if solvable_only:
                import feasibility_st
                if not feasibility_st.scenario_solvable(built, layout["panels"])["solvable"]:
                    continue
            return built
        # This layout would not take a feasible target of that kind: draw another.


def validation_set() -> List[scn.Scenario]:
    """The fixed field validation set: `VALIDATION_NO_TARGET` no-target layouts and
    `VALIDATION_PER_ENCOUNTER` constant-velocity scenarios of each encounter, each
    on its own layout, from seeds `VALIDATION_SEED_BASE`+ (disjoint from training,
    development, the frozen suite and the Paper 2 set)."""
    out = []
    plan = [("NT", VALIDATION_NO_TARGET)] + [(c, VALIDATION_PER_ENCOUNTER) for c in ENCOUNTER_CODES]
    for b, (code, n) in enumerate(plan):
        for i in range(n):
            base = VALIDATION_SEED_BASE + b * 1_000 + i * 20
            rng = np.random.default_rng(base)
            built = sample(rng, namespace="validation", encounter=code, varying=False,
                           generator=scn.ScenarioGenerator(stage=5, seed_namespace="frozen_eval"),
                           seed_fn=lambda k, base=base: base + 200_000 + k)
            built.case_id = f"FV-{code}-{i + 1:02d}"
            out.append(built)
    return out
