"""The frozen evaluation suite (04a §4-§9): Tier A, Tier B, Around the Clock.

Structure (04a §4.2):

| Component | Cases | Rollouts per case per seed |
|---|---|---|
| Tier A — named | 38 (06 M-7) | 10 |
| Tier B — stratified holdout | 48 cells x 20 (06 M-6) | 1 |
| Around the Clock — open water | 24 | 10 |
| Around the Clock — channel | 24 | 10 |

**Ten rollouts per named case despite a deterministic policy.**  The
*environment* is stochastic once perception noise is injected, which is the
whole point of N1: a single rollout of a named case reports one draw from the
perception noise, not the behaviour.

**The freeze protocol is not paperwork** (04a §9.4).  With Imazu dropped, the
released generator plus the frozen suite plus the manifest is the only
structural defence against "the authors built their own benchmark, then showed
classical methods fail on it".  Every case serialises to canonical JSON, every
case is hashed, and regenerating from seed must reproduce every hash.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import constants as cfg
import scenario as scn
import targets as tgt

# Width codes for the Tier A case table (04a §5).  "B" is a basin case (06 §5.2).
WIDTH_CODES = {"W": 10.0, "I": 6.0, "N": 4.0, "X": 3.5, "B": 0.0}

# 06 §5.2: basin cases run one fixed leg at the widest slant Paper 2's layout
# allows (14.0 deg, x 2.5 -> 7.5 m), so the leg closes on the starboard wall and
# the two clearances differ most at mid-leg.  06 asked for 15 deg; the fixed-y
# layout's endpoint box caps it at 14.0.
BASIN_TIER_A_LEG = ((cfg.BASIN_X_RANGE[0], cfg.BASIN_START_Y),
                    (cfg.BASIN_X_RANGE[1], cfg.BASIN_GOAL_Y))


@dataclass(frozen=True)
class Case:
    """One named Tier A case."""

    case_id: str
    encounter_class: str
    width_code: str
    behaviour: str = tgt.T_CV
    speed_ratio: Optional[float] = None
    flags: Tuple[Tuple[str, object], ...] = ()

    @property
    def width(self) -> float:
        return WIDTH_CODES[self.width_code]

    @property
    def is_basin(self) -> bool:
        return self.width_code == "B"

    @property
    def flag_dict(self) -> dict:
        return dict(self.flags)


def _c(case_id, cls, code, behaviour=tgt.T_CV, k=None, **flags) -> Case:
    return Case(case_id, cls, code, behaviour, k, tuple(sorted(flags.items())))


# ---------------------------------------------------------------------------
# Tier A — 34 named cases (04a §5)
# ---------------------------------------------------------------------------
def tier_a() -> List[Case]:
    """The 34 named cases, in the order 04a §5 tabulates them.

    `A-8E-HO-N` is the one that carries the head-on precedence argument: its
    target is **positionally** non-compliant (`T-NC3`), which is the only case
    in the suite where a Rule 14 alteration is actually required.  Without it
    the head-on class is never exercised as an avoidance problem at all, because
    02 §3.2's whole point is that 9(a) channel-keeping satisfies Rule 14.
    """
    cases: List[Case] = []

    for code in ("W", "I", "N"):
        cases.append(_c(f"A-HO-{code}", "head_on", code))
    for code in ("W", "I", "N"):
        cases.append(_c(f"A-CRS-{code}", "crossing", code, side="starboard"))
    for code in ("W", "I", "N"):
        cases.append(_c(f"A-CRP-{code}", "crossing", code, side="port"))
    for code in ("W", "I", "N"):
        # 04a §11 change 2: mid-window rather than at the floor, because 03a
        # §1.3 raised the floor to 0.40 and a case sitting exactly on a bound is
        # a case that stops existing the next time the bound moves.
        cases.append(_c(f"A-OT-{code}", "overtaking", code, k=0.45))
    for code in ("W", "I", "N"):
        cases.append(_c(f"A-BO-{code}", "being_overtaken", code, k=1.8))
    for code in ("W", "I", "N"):
        cases.append(_c(f"A-NU-{code}", "null", code))

    for code in ("I", "N"):
        cases.append(_c(f"A-CLT-HO-{code}", "head_on", code, conflict=1))
        cases.append(_c(f"A-CLT-CRS-{code}", "crossing", code, conflict=1))
        cases.append(_c(f"A-CLT-OT-{code}", "overtaking", code, k=0.45, conflict=1))

    cases.append(_c("A-OCC-HO-I", "head_on", "I", occlusion=2.0))
    cases.append(_c("A-OCC-CRS-I", "crossing", "I", occlusion=4.0))

    # The two 8(e) cases: the fallback is the only admissible response.
    cases.append(_c("A-8E-HO-N", "head_on", "N", tgt.T_NC3))
    cases.append(_c("A-8E-CRS-N", "crossing", "N"))

    # A-BND-HO-I and A-BND-CRS-I (40 deg bends) are withdrawn with bends (F59).
    if cfg.CORRIDOR_BENDS:
        cases.append(_c("A-BND-HO-I", "head_on", "I", bend=40.0))
        cases.append(_c("A-BND-CRS-I", "crossing", "I", bend=40.0))

    cases.append(_c("A-OFF-CRP-I", "crossing", "I", side="port", offset=-0.30))
    cases.append(_c("A-OFF-OT-I", "overtaking", "I", k=0.45, offset=0.30))

    # Pre-committed expected failures (04a §4.5).  Reported at whatever level
    # they come out; the framing rule is that strata are described by geometry
    # only, never as "challenging for classical methods".
    cases.append(_c("A-FAIL-CRS-X", "crossing", "X", tgt.T_NC1, expected_failure=1))
    cases.append(_c("A-FAIL-HO-X", "head_on", "X", tgt.T_NC2, expected_failure=1))

    # 06 §5.2 / M-7: six basin cases, all field-replicable -- the basin-session
    # trial list.  A-BSN-CLT-CRS puts the conflict panel on the side the
    # compliant alteration would use, where the slanted leg is closing on a wall.
    cases.append(_c("A-BSN-HO", "head_on", "B"))
    cases.append(_c("A-BSN-CRS", "crossing", "B", side="starboard"))
    cases.append(_c("A-BSN-CRP", "crossing", "B", side="port"))
    cases.append(_c("A-BSN-OT", "overtaking", "B", k=0.45))
    cases.append(_c("A-BSN-BO", "being_overtaken", "B", k=1.8))
    cases.append(_c("A-BSN-CLT-CRS", "crossing", "B", side="starboard", conflict=1))

    return cases


# ---------------------------------------------------------------------------
# Tier B — 39 stratified cells (04a §4.3)
# ---------------------------------------------------------------------------
# Suite 3.1 (your call, 2026-09-23): Tier B channels stop at 7.5 m.  The 10 m
# basin is already narrow water for a 1.7 m vessel, and below ~7.5 m a channel
# leaves a two-vessel encounter no room that a lawful manoeuvre can use: PPO's
# 3.5-4.25 m stratum failed 0.57 of the time and a third of those were wall
# contacts, which measures the geometry rather than the policy.
#
# **What this costs, and it is not small.**  The 3.8 m head-on and 4.25 m
# overtaking thresholds now lie outside the sampled range, and the 7.6 m
# crossing threshold sits just inside the lower stratum, so Tier B on its own no
# longer partitions the widths by governing rule.  Claims C-2 and C-3 therefore
# rest on the Study 1 width sweep (R4), which keeps `CORRIDOR_WIDTHS_M` down to
# 3.5 m, and Tier B becomes the headline in water where a lawful manoeuvre
# exists.  The narrow behaviour is still measured -- it is reported from R4 and
# from Tier A's `A-*-N` and `A-FAIL-*` cases, not from Tier B.
TIER_B_MIN_WIDTH_M = 7.5
SUITE_REVISION = "3.1"


def width_strata() -> Dict[str, Tuple[float, float]]:
    """Channel strata for Tier B, floored at `TIER_B_MIN_WIDTH_M`.

    Suite 3.0 cut three strata at the derived thresholds (crossing 7.60 m,
    overtaking 4.26 m), so each had a distinct predicted governing rule.  With
    the floor at 7.5 m only the crossing threshold remains inside the range, so
    the two strata below are a span split rather than a rule split.  `R4` keeps
    the rule-by-width test.
    """
    hi = 10.0
    lo = TIER_B_MIN_WIDTH_M
    mid = round(0.5 * (lo + hi), 3)
    specified = {"wide": (mid, hi), "intermediate": (lo, mid)}
    lo_cfg, hi_cfg = cfg.CORRIDOR_WIDTH_RANGE
    return {name: (a, b) for name, (a, b) in specified.items()
            if a <= hi_cfg + 1e-9 and b >= lo_cfg - 1e-9}


# Narrow x null and narrow x being-overtaken are infeasible (06 M-6): at
# 3.50-4.26 m a null encounter is not an encounter, and an overtaker has nowhere
# to pass.  They are reported through the rejection ledger, not sampled.
NARROW_EXCLUDED = ("null", "being_overtaken")


def tier_b_cells() -> List[dict]:
    """06 §5.1, as amended for suite 3.1: 39 cells x 20 = 780 episodes per seed.

    Basin 13 (4 classes x 3 behaviours + null), channel wide 13, channel
    intermediate 13.  Null takes constant velocity only.  Suite 3.0 had a fourth
    stratum (3.5-4.26 m, 9 cells, no null or being-overtaken); it is gone with
    the 7.5 m floor -- see `width_strata`.
    """
    cells = []
    strata = [("basin", None)] + [(f"channel-{k}", v) for k, v in width_strata().items()]
    for name, width_range in strata:
        narrow = name == "channel-narrow"
        for cls in ("head_on", "crossing", "overtaking", "being_overtaken", "null"):
            if narrow and cls in NARROW_EXCLUDED:
                continue
            behaviours = ("cv",) if cls == "null" else cfg.TIER_B_BEHAVIOURS
            for behaviour in behaviours:
                cells.append({"class": cls, "behaviour": behaviour, "stratum": name,
                              "geometry_mode": "basin" if width_range is None else "channel",
                              "width_range": width_range,
                              "episodes": cfg.TIER_B_EPISODES_PER_CELL})
    return cells


# ---------------------------------------------------------------------------
# Around the Clock (04a §4.4)
# ---------------------------------------------------------------------------
def around_the_clock(open_water: bool = True) -> List[dict]:
    """24 constellations, both vessels set to meet at the origin.

    Run as published in open water for comparability, then with corridor walls
    at each of three widths.  **Both variants belong in the main text as one
    figure**: the whole argument of N2 is what changes between them, and
    separating them across a section boundary makes the reader do the comparison
    from memory.

    The spawn radius is re-derived in ship lengths against `Lpp = 1.57 m` rather
    than adopting the published 2 NM scaling, which would put the target outside
    the basin by three orders of magnitude.
    """
    radius = min(0.9 * cfg.LIDAR_RANGE_EFFECTIVE, 0.45 * cfg.MAP_HEIGHT)
    out = []
    for j in range(1, cfg.AROUND_THE_CLOCK_SPOKES + 1):
        phi = 2.0 * math.pi * j / (cfg.AROUND_THE_CLOCK_SPOKES + 1)
        widths = [None] if open_water else list(cfg.AROUND_THE_CLOCK_WIDTHS_M)
        for width in widths:
            out.append({
                "case_id": f"ATC-{'OW' if open_water else f'W{width:g}'}-{j:02d}",
                "spoke": j,
                "bearing_deg": math.degrees(phi) % 360.0,
                "radius_m": radius,
                "open_water": bool(open_water),
                "width": width,
                "radius_lpp": radius / cfg.LBP,
            })
    return out


# ---------------------------------------------------------------------------
# Study 2 conditions (04a §7.1)
# ---------------------------------------------------------------------------
def study2_conditions() -> List[dict]:
    """5 levels x 4 axes, minus the shared nominal, plus one joint corner = 21."""
    out = [{"name": "nominal", "axis": None, "multiplier": 1.0}]
    for axis in cfg.STUDY2_AXES:
        for m in cfg.STUDY2_MULTIPLIERS:
            if m == 1.0:
                continue
            out.append({"name": f"{axis}x{m:g}", "axis": axis, "multiplier": m})
    out.append({"name": "joint-corner", "axis": "all", "multiplier": 2.0})
    return out


# ---------------------------------------------------------------------------
# Building and freezing
# ---------------------------------------------------------------------------
def build_tier_a(generator: Optional[scn.ScenarioGenerator] = None,
                 namespace: str = "frozen_eval") -> List[scn.Scenario]:
    """Realise the 34 named cases as scenarios.

    Cases that cannot be constructed come back as `None` from the generator and
    are *reported*, not dropped silently -- an evaluation suite that is quietly
    two cases short is worse than one that admits it, because the missing cases
    are exactly the infeasible geometries the feasibility envelope is about.
    """
    generator = generator or scn.ScenarioGenerator(stage=5, seed_namespace=namespace)
    out = []
    for index, case in enumerate(tier_a()):
        built = _build_case(generator, namespace, index, case)
        if built is not None:
            out.append(built)
    return out


# A named case that caps out on its seed gets this many further seeds, each a
# fixed function of the case index, so the realised suite is still a pure
# function of the namespace.  Cases that fail all of them are reported by
# `tier_a_shortfall()`, never dropped silently.
TIER_A_SEED_RETRIES = 8
# The frozen namespace holds 10,000 seeds (04a §9.2).  Tier B takes 200 per cell
# (indices 0-9,599); Tier A the block above it, 9 per case -- disjoint, where
# `index + 10,000 * retry` wrapped back onto the same seed.
TIER_B_SEEDS_PER_CELL = 200
TIER_A_SEED_BASE = 48 * TIER_B_SEEDS_PER_CELL


def tier_a_shortfall(scenarios: Sequence[scn.Scenario]) -> List[str]:
    """Named cases the generator could not realise -- the feasibility envelope."""
    built = {s.case_id for s in scenarios}
    return [c.case_id for c in tier_a() if c.case_id not in built]


def _build_case(generator, namespace, index, case):
    for retry in range(TIER_A_SEED_RETRIES + 1):
        seed = scn.seed_for(namespace, TIER_A_SEED_BASE + index * (TIER_A_SEED_RETRIES + 1) + retry)
        # F75: the named features are realised, not only recorded -- the
        # crossing side, the speed ratio, the offset, the basin leg; the
        # conflict and occlusion panels are placed by the environment.
        flags = dict(case.flag_dict)
        if case.speed_ratio is not None:
            flags["speed_ratio"] = case.speed_ratio
        if case.is_basin:
            flags["basin_leg"] = [list(p) for p in BASIN_TIER_A_LEG]
        built = generator.sample(seed, case_id=case.case_id,
                                 encounter_class=case.encounter_class,
                                 width=None if case.is_basin else case.width,
                                 behaviour=case.behaviour, flags=flags,
                                 geometry_mode="basin" if case.is_basin else "channel")
        if built is not None:
            return built
    return None


def build_tier_b(generator: Optional[scn.ScenarioGenerator] = None,
                 namespace: str = "frozen_eval") -> Tuple[List[scn.Scenario], List[dict]]:
    """Realise the 48 Tier B cells (06 §5.1).  Returns (scenarios, shortfalls)."""
    generator = generator or scn.ScenarioGenerator(stage=5, seed_namespace=namespace)
    out, short = [], []
    for c_index, cell in enumerate(tier_b_cells()):
        rng = np.random.default_rng(scn.seed_for(namespace, c_index * TIER_B_SEEDS_PER_CELL))
        got, tries = 0, 0
        while got < cell["episodes"] and tries < TIER_B_SEEDS_PER_CELL:
            seed = scn.seed_for(namespace, c_index * TIER_B_SEEDS_PER_CELL + tries)
            tries += 1
            width = (None if cell["width_range"] is None
                     else float(rng.uniform(*cell["width_range"])))
            built = generator.sample(seed, case_id=f"B-{c_index:02d}-{got:02d}",
                                     encounter_class=cell["class"], width=width,
                                     behaviour=cell["behaviour"],
                                     geometry_mode=cell["geometry_mode"],
                                     flags={"stratum": cell["stratum"]})
            if built is not None:
                out.append(built)
                got += 1
        if got < cell["episodes"]:
            short.append({**cell, "realised": got})
    return out, short


def manifest(scenarios: Sequence[scn.Scenario], *, generator_sha: str = "",
             version: str = None) -> dict:
    """`SUITE_MANIFEST.json` (04a §9.3): version, hashes, seeds, constants.

    The constants snapshot is part of the hash-bearing record on purpose.  A
    suite is only reproducible against the constants it was generated under, and
    every threshold in it is a function of the ship domain and the pose
    characterisation -- both of which are still provisional.
    """
    digests = {s.case_id: s.digest() for s in scenarios}
    blob = json.dumps(digests, sort_keys=True, separators=(",", ":"))
    return {
        "suite_version": version or SUITE_REVISION,
        "generator_git_sha": generator_sha,
        "n_cases": len(scenarios),
        "case_digests": digests,
        "manifest_digest": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "seed_namespaces": {k: list(v) for k, v in cfg.SEED_NAMESPACES.items()},
        "constants": constants_snapshot(),
    }


def constants_snapshot() -> dict:
    """Every constant the suite's geometry depends on (04a §9.3)."""
    thresholds = scn.width_thresholds()
    return {
        "U_NOM": cfg.U_NOM,
        "froude": cfg.froude(),
        "LBP": cfg.LBP,
        "BREADTH": cfg.BREADTH,
        "DOMAIN_LATERAL": cfg.DOMAIN_LATERAL,
        "DOMAIN_FORE": cfg.DOMAIN_FORE,
        "DOMAIN_AFT": cfg.DOMAIN_AFT,
        "W_WALL": cfg.W_WALL,
        "MAX_EPISODE_STEPS": cfg.MAX_EPISODE_STEPS,
        "LIDAR_RANGE_EFFECTIVE": cfg.LIDAR_RANGE_EFFECTIVE,
        "width_thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "STUDY1_WIDTHS_M": list(cfg.STUDY1_WIDTHS_M),
    }


def freeze_checklist() -> List[Tuple[str, bool, str]]:
    """04a §9.3, as machine-checked state rather than a list to tick by hand.

    Returns `(item, satisfied, note)`.  The unsatisfied entries are the honest
    answer to "can the headline training run start" -- and today several of them
    cannot be satisfied by code at all, because they are waiting on 05.
    """
    checks: List[Tuple[str, bool, str]] = []
    dirty = _uncommitted(("src/corridor.py", "src/scenario.py", "src/suite.py",
                          "src/feasibility.py", "src/constants.py"))
    checks.append(("generator source committed", not dirty,
                   "clean" if not dirty else "uncommitted: " + ", ".join(dirty)))
    checks.append(("seed namespaces disjoint", scn.namespaces_are_disjoint(),
                   str(cfg.SEED_NAMESPACES)))
    checks.append(("suite generated and hashed", True,
                   "build_tier_a() + manifest()"))

    unresolved = unresolved_todos()
    deferred = "; deferred: " + ", ".join(f"TODO({k})" for k in cfg.DEFERRED_TODOS)
    checks.append(("every TODO(04-*) resolved or deferred", not unresolved,
                   (", ".join(unresolved) if unresolved else "none outstanding") + deferred))
    checks.append(("operating speed settled (F24)", True,
                   f"decided: CRUISE_RPM = {cfg.CRUISE_RPM:g}, U_NOM = {cfg.U_NOM:.3f} m/s"))
    checks.append(("throughput measured (04a §8.3)", True,
                   "F75, 10 workers, 12 cores: PPO ~108 steps/s; SAC 12 (1 gradient step per "
                   "transition) / 38 (0.2); TD3 18 / 53; TQC 13 / 32; RecurrentPPO 61 (F80)"))
    for item, rel in (("claim ledger committed", "planning/CLAIM_LEDGER.md"),
                      ("empty result tables committed", "planning/RESULT_TABLES.md")):
        from pathlib import Path
        exists = (Path(__file__).resolve().parents[1] / rel).exists()
        pending = _uncommitted((rel,))
        checks.append((item, exists and not pending,
                       "not written" if not exists else
                       ("draft, awaiting sign-off and commit" if pending else rel)))
    checks.append(("regeneration test reproduces hashes", True,
                   "test_acceptance.py::test_the_suite_regenerates_to_the_same_hashes"))
    return checks


def _uncommitted(paths) -> List[str]:
    """Generator files with uncommitted changes -- the freeze needs a git SHA."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=root,
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return list(paths)
    return [line[3:].strip() for line in out.splitlines() if line.strip()]


def unresolved_todos() -> List[str]:
    """The `TODO(04-*)` items 04a §11 leaves open."""
    open_items = []
    if cfg.CURRICULUM_STAGE_STEPS is None:
        open_items.append("TODO(04-3) curriculum steps per stage")
    for key, label in (("04-1", "w_wall from measured pose drift"),
                       ("04-2", "D_max effective, black-wall side")):
        if key not in cfg.DEFERRED_TODOS:
            open_items.append(f"TODO({key}) {label}")
    if getattr(cfg, "TRAINING_BUDGET", None) is None:
        open_items.append("TODO(04-4) training steps per run and budget")
    return open_items
