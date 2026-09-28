"""The Paper 2 deployment-layout set: a separate evaluation set (2026-09-27).

The three static-obstacle layouts of the published Paper 2 field trials --
Tran et al., *Drones* 10(9), 680, Fig. 8 ("Simulated and field trajectories for
the three test scenarios") -- with and without a target ship.  **Separate from
the frozen suite**: nothing here changes Tier B, and nothing in it is reported
in the paper's headline tables.

Source of the layouts.  Measured from the published Fig. 8 (the full-resolution
image, calibrated on each panel's 10 x 25 m workspace boundary, +-0.02 m) and
rounded to 0.1 m: 1 m square panels.  Scenario 1 agrees with Paper 2's
`static_obstacles/test_run.py` case 1; scenarios 2 and 3 differ from both code
copies of `test_run.py` (case 2's left panel is at x = 2.0, not 1.5; case 3's
at (6.5, 8.5) and (5.0, 17.0), not (6.0, 9.3) and (5.5, 17.0)), so the
published figure is taken as the record.  The legs are the published ones; the
slanted legs reach goal x = 8 and x = 2, 0.5 m nearer the walls than training's
basin legs (x in [2.5, 7.5]), at the same 14 deg slant.

Composition (`SET_REVISION`):

* **No target**: each layout alone, as flown -- the Paper 2 replication -- run
  on `NO_TARGET_EPISODES` episode seeds (sensor-noise realisations).
* **With a target**: five COLREGs encounters -- head-on (`HO`), crossing from
  port (`CRP`) and from starboard (`CRS`), overtaking (`OT`), being overtaken
  (`BO`) -- `PER_CELL` draws per encounter per layout, each drawn by the
  scenario generator on the layout's leg, and kept only if the target's
  nominal track clears every panel by `TARGET_PANEL_CLEARANCE_M` (targets are
  not steered around panels).
* **Fixed and varying speed**: each target scenario runs twice on the same
  episode seed -- `FIX`, the constant-velocity target of training, and `VAR`,
  the same target with one speed change on the approach (`targets.T_VS`,
  parameters in `constant_temp.TARGET_VS_*`).  Only the speed differs, so the
  pair isolates it.

**Field feasible (revision 2.0, your call, 2026-09-27).**  Every target scenario
is one a second, Bluefin-class model vessel can sail in the 10 x 25 m basin
(`constant_temp.FIELD_*`):

* the target's hull starts at least `FIELD_WALL_MARGIN_M` inside the basin walls;
* it holds **one heading** for the whole run -- a nominal track that the
  simulator's lane clamp would bend (more than `FIELD_MAX_TURN_DEG`) is redrawn,
  so an operator or autopilot only has to hold course (and, for `VAR`, change
  speed once at a stated time);
* it **stops short of the wall** (`targets.Target.stop_box`), in simulation as the
  boat would be stopped in the field, and the encounter must be over
  `FIELD_POST_CPA_S` before it does;
* its speed, and the `VAR` twin's final speed, lie in `FIELD_TARGET_SPEED_RANGE`.

`field_sheet()` gives each run's set-up: target start, heading, speed, when to
change speed and to what.

Test IDs: `P2-<layout>-<encounter>-<speed>-<nn>` (e.g. `P2-L2-CRP-VAR-07`), and
`P2-<layout>-NT-<nn>` without a target.  Replay one with
`tools/tiers/run_test.py`; run the set with `tools/tiers/paper2_suite.py`.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import constant_temp as ct
import scenario as scn
import targets as tgt
from nominal import nominal_encounter

SET_REVISION = "2.0"       # 2.0: field feasible

# The published field layouts (Paper 2, Fig. 8): leg and panel centres, metres.
LAYOUTS: Dict[str, dict] = {
    "L1": {"name": "Scenario 1 (straight)", "start": (5.0, 2.0), "goal": (5.0, 22.0),
           "panels": [(5.0, 8.0), (7.2, 15.7), (2.0, 16.5)]},
    "L2": {"name": "Scenario 2 (slanted, left to right)", "start": (3.0, 2.0), "goal": (8.0, 22.0),
           "panels": [(2.0, 8.5), (7.2, 9.3), (6.0, 17.0)]},
    "L3": {"name": "Scenario 3 (slanted, right to left)", "start": (7.0, 2.0), "goal": (2.0, 22.0),
           "panels": [(1.5, 8.5), (6.5, 8.5), (5.0, 17.0)]},
}
PANEL_SIZE_M = 1.0

# Encounter code -> (generator class, crossing side).
ENCOUNTERS: Dict[str, Tuple[str, Optional[str]]] = {
    "HO": ("head_on", None),
    "CRP": ("crossing", "port"),
    "CRS": ("crossing", "starboard"),
    "OT": ("overtaking", None),
    "BO": ("being_overtaken", None),
}
SPEEDS = ("FIX", "VAR")
PER_CELL = 20
NO_TARGET_EPISODES = 10

# Seeds.  Generator seeds from 310,000: a gap no namespace uses (training
# 0-99,999, development 200 k, frozen 300,000-309,999, studies 400 k and 500 k),
# 2,000 per cell (the field rules reject most draws).  Episode (environment)
# seeds from 470,000 + index; a VAR twin runs on its FIX twin's.
SEED_BASE = 310_000
SEEDS_PER_CELL = 2000
EPISODE_SEED_BASE = 470_000


def panel_polygons(layout: str) -> List[List[Tuple[float, float]]]:
    h = 0.5 * PANEL_SIZE_M
    return [[(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]
            for x, y in LAYOUTS[layout]["panels"]]


def cells() -> List[dict]:
    """The target cells, in seed-block order: layout x encounter."""
    return [{"layout": lay, "encounter": code, "class": cls, "side": side}
            for lay in LAYOUTS for code, (cls, side) in ENCOUNTERS.items()]


# ---------------------------------------------------------------------------
# Geometry: convex polygon distance (no geometry library in the environment)
# ---------------------------------------------------------------------------
def _separated(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for e in edges:
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return True
    return False


def _point_segment(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def polygon_distance(a, b) -> float:
    """Distance between two convex polygons (0 if they overlap)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not _separated(a, b):
        return 0.0
    best = np.inf
    for p, q in ((a, b), (b, a)):
        for v in p:
            for i in range(len(q)):
                best = min(best, _point_segment(v, q[i], q[(i + 1) % len(q)]))
    return float(best)


def stop_box() -> List[float]:
    """The box the target's hull must stay inside: the basin less the wall margin."""
    import constants as cfg
    m = ct.FIELD_WALL_MARGIN_M
    return [m, cfg.MAP_WIDTH - m, m, cfg.MAP_HEIGHT - m]


def nominal_check(env, built, episode_seed: int) -> dict:
    """The target's nominal run (own ship holding the leg at cruise): closest
    approach of its hull to a panel, how far its heading turns, when it stops at
    the wall (None if it never does), and the nominal CPA time."""
    env.reset(seed=episode_seed, options={"generated": built})
    nom = nominal_encounter(env)
    trk = nom["target"]
    if not len(trk):
        return {"clearance": np.inf, "turned": 0.0, "t_stop": None, "cpa_t": None}
    every = max(1, int(round(0.5 / nom["dt"])))
    panels = [np.asarray(p) for p in env.obstacles]
    clearance = min(polygon_distance(tgt.hull_polygon(*trk[k]), p)
                    for k in range(0, len(trk), every) for p in panels)
    moved = np.hypot(np.diff(trk[:, 0]), np.diff(trk[:, 1])) > 1e-9
    t_stop = None if moved.all() else float(np.argmin(moved) * nom["dt"])
    return {"clearance": clearance, "turned": nom["turned_deg"], "t_stop": t_stop,
            "cpa_t": nom.get("cpa_t")}


def field_feasible(check: dict) -> bool:
    """The field rules on one nominal run (see the module docstring)."""
    if check["clearance"] < ct.TARGET_PANEL_CLEARANCE_M or check["turned"] > ct.FIELD_MAX_TURN_DEG:
        return False
    if check["t_stop"] is not None and check["t_stop"] < check["cpa_t"] + ct.FIELD_POST_CPA_S:
        return False
    return True


# ---------------------------------------------------------------------------
# Building the set
# ---------------------------------------------------------------------------
def _flags(layout: str, **extra) -> dict:
    lay = LAYOUTS[layout]
    return {"basin_leg": [list(lay["start"]), list(lay["goal"])],
            "fixed_obstacles": [[list(p) for p in poly] for poly in panel_polygons(layout)],
            "set": "paper2", "layout": layout, **extra}


def speed_profile(built: scn.Scenario, seed: int) -> Optional[List[float]]:
    """The `T-VS` twin's single speed change: at a random fraction of the drawn
    TCPA, to a slower or a faster speed (even odds), ramped at the model
    vessel's acceleration, the final speed kept inside
    `FIELD_TARGET_SPEED_RANGE` (the other direction if one does not fit; None if
    neither does).  A pure function of the scenario's seed."""
    rng = np.random.default_rng(int(seed) + 17)
    t_change = float(rng.uniform(*ct.TARGET_VS_CHANGE_FRAC)) * max(float(built.tcpa_s), 1.0)
    v0 = float(built.target_speed)
    lo, hi = (v / v0 for v in ct.FIELD_TARGET_SPEED_RANGE)
    bands = [ct.TARGET_VS_SLOW_FACTOR, ct.TARGET_VS_FAST_FACTOR]
    if rng.uniform() >= 0.5:
        bands.reverse()
    u = float(rng.uniform())
    for a, b in bands:
        a, b = max(a, lo), min(b, hi)
        if a <= b:
            return [t_change, (a + u * (b - a)) * v0, float(ct.TARGET_VS_ACCEL_MPS2)]
    return None


def varying_twin(built: scn.Scenario, seed: int) -> Optional[scn.Scenario]:
    profile = speed_profile(built, seed)
    if profile is None:
        return None
    twin = copy.deepcopy(built)
    twin.target_behaviour = tgt.T_VS
    twin.flags = dict(twin.flags, speed_profile=profile)
    twin.case_id = built.case_id.replace("-FIX-", "-VAR-")
    return twin


def build(env, layouts: Optional[Sequence[str]] = None,
          encounters: Optional[Sequence[str]] = None,
          include_no_target: bool = True) -> Tuple[List[dict], List[dict]]:
    """Realise the set.  Returns (records, shortfalls).

    A record: {"built", "test_id", "layout", "encounter", "speed", "episode_seed",
    "twin", "target_clearance_m"}.  `layouts` / `encounters` realise a subset;
    each cell draws from its own seed block, so a cell comes out identical alone
    or in the full build.  `env` is an `ASVLidarEnv` on the evaluation stage
    (the clearance check steps it).
    """
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="frozen_eval")
    records, short = [], []
    base_index = 0
    for c_index, cell in enumerate(cells()):
        lay, code = cell["layout"], cell["encounter"]
        if (layouts is not None and lay not in layouts) or (encounters is not None and code not in encounters):
            base_index += PER_CELL
            continue
        got, tries = 0, 0
        while got < PER_CELL and tries < SEEDS_PER_CELL:
            seed = SEED_BASE + c_index * SEEDS_PER_CELL + tries
            tries += 1
            test_id = f"P2-{lay}-{code}-FIX-{got + 1:02d}"
            extra = {"side": cell["side"]} if cell["side"] else {}
            built = generator.sample(seed, case_id=test_id, encounter_class=cell["class"],
                                     behaviour=tgt.T_CV, geometry_mode="basin",
                                     flags=_flags(lay, target_stop_box=stop_box(), **extra))
            if built is None:
                continue
            lo, hi = ct.FIELD_TARGET_SPEED_RANGE
            if not lo <= float(built.target_speed) <= hi:
                continue
            twin = varying_twin(built, seed)
            if twin is None:
                continue
            episode_seed = EPISODE_SEED_BASE + base_index + got
            checks = [nominal_check(env, b, episode_seed) for b in (built, twin)]
            if not all(field_feasible(c) for c in checks):
                continue
            clear = min(c["clearance"] for c in checks)
            for speed, b in (("FIX", built), ("VAR", twin)):
                records.append({"built": b, "test_id": b.case_id, "layout": lay, "encounter": code,
                                "speed": speed, "episode_seed": episode_seed,
                                "twin": built.case_id if speed == "VAR" else "",
                                "target_clearance_m": round(clear, 3), "generator_seed": seed})
            got += 1
        if got < PER_CELL:
            short.append({**cell, "realised": got})
        base_index += PER_CELL
    if include_no_target:
        for l_index, lay in enumerate(LAYOUTS):
            if layouts is not None and lay not in layouts:
                continue
            seed = SEED_BASE + len(cells()) * SEEDS_PER_CELL + l_index
            built = generator.sample(seed, case_id=f"P2-{lay}-NT", encounter_class="no_target",
                                     geometry_mode="basin", flags=_flags(lay))
            if built is None:
                short.append({"layout": lay, "encounter": "NT", "realised": 0})
                continue
            for n in range(NO_TARGET_EPISODES):
                b = copy.deepcopy(built)
                b.case_id = f"P2-{lay}-NT-{n + 1:02d}"
                records.append({"built": b, "test_id": b.case_id, "layout": lay, "encounter": "NT",
                                "speed": "", "episode_seed": EPISODE_SEED_BASE + 10_000 + l_index * 100 + n,
                                "twin": "", "target_clearance_m": np.inf, "generator_seed": seed})
    return records, short


def field_sheet(records: Sequence[dict]) -> List[dict]:
    """Each run's field set-up: where the target starts, the heading to hold, its
    speed, and (VAR) when to change speed and to what.  Times are from t = 0, the
    moment the own ship passes `own_at_t0` at cruise -- the published start,
    except being overtaken, where the generator starts it further along the leg
    to leave the overtaker room astern; it runs up along the leg from the
    published start and the target is released as it passes that point."""
    rows = []
    for r in records:
        b = r["built"]
        lay = LAYOUTS[r["layout"]]
        row = {"test_id": r["test_id"], "layout": r["layout"], "encounter": r["encounter"],
               "published_start": list(lay["start"]), "own_goal": list(lay["goal"]),
               "own_at_t0": [round(float(v), 2) for v in b.own_spawn],
               "own_heading_deg": round(float(b.own_heading) % 360.0, 1),
               "panels": [list(c) for c in lay["panels"]]}
        if r["encounter"] != "NT":
            prof = (b.flags or {}).get("speed_profile")
            row.update(target_start=[round(float(v), 2) for v in b.target_spawn],
                       target_heading_deg=round(float(b.target_heading) % 360.0, 1),
                       target_speed_mps=round(float(b.target_speed), 2),
                       speed_change_at_s=round(prof[0], 1) if prof else "",
                       speed_change_to_mps=round(prof[1], 2) if prof else "",
                       stop_before_wall_m=ct.FIELD_WALL_MARGIN_M)
        rows.append(row)
    return rows


def resolve_test(ref: str) -> dict:
    """Parse a set test ID: {"layout", "encounter", "speed", "n"} (n from 1)."""
    parts = ref.strip().upper().split("-")
    if parts[0] != "P2" or len(parts) not in (4, 5) or parts[1] not in LAYOUTS:
        raise ValueError(f"not a Paper 2 set test ID: {ref}")
    if len(parts) == 4:
        if parts[2] != "NT":
            raise ValueError(f"not a Paper 2 set test ID: {ref}")
        n = int(parts[3])
        if not 1 <= n <= NO_TARGET_EPISODES:
            raise ValueError(f"{ref}: no-target runs are 01-{NO_TARGET_EPISODES:02d}")
        return {"layout": parts[1], "encounter": "NT", "speed": "", "n": n}
    lay, code, speed, n = parts[1], parts[2], parts[3], int(parts[4])
    if code not in ENCOUNTERS or speed not in SPEEDS or not 1 <= n <= PER_CELL:
        raise ValueError(f"not a Paper 2 set test ID: {ref}")
    return {"layout": lay, "encounter": code, "speed": speed, "n": n}


def find(env, ref: str) -> dict:
    """Build only the cell that holds `ref` and return its record."""
    key = resolve_test(ref)
    ref = ref.strip().upper()
    records, _ = build(env, layouts=[key["layout"]],
                       encounters=[] if key["encounter"] == "NT" else [key["encounter"]],
                       include_no_target=key["encounter"] == "NT")
    if key["encounter"] == "NT":
        records = [r for r in records if r["encounter"] == "NT"]
    match = [r for r in records if r["test_id"] == ref]
    if not match:
        raise ValueError(f"{ref} was not realised (the cell fell short)")
    return match[0]
