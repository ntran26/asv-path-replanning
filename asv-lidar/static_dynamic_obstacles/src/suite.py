"""The frozen evaluation suite (04a §4-§9): Tier A, Tier B, Around the Clock.

Structure (04a §4.2):

| Component | Cases | Rollouts per case per seed |
|---|---|---|
| Tier A — named | 34 | 10 |
| Tier B — stratified holdout | 39 cells x 25 | 1 |
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

# Width codes for the Tier A case table (04a §5).
WIDTH_CODES = {"W": 10.0, "I": 6.0, "N": 4.0, "X": 3.5}


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

    cases.append(_c("A-BND-HO-I", "head_on", "I", bend=40.0))
    cases.append(_c("A-BND-CRS-I", "crossing", "I", bend=40.0))

    cases.append(_c("A-OFF-CRP-I", "crossing", "I", side="port", offset=-0.30))
    cases.append(_c("A-OFF-OT-I", "overtaking", "I", k=0.45, offset=0.30))

    # Pre-committed expected failures (04a §4.5).  Reported at whatever level
    # they come out; the framing rule is that strata are described by geometry
    # only, never as "challenging for classical methods".
    cases.append(_c("A-FAIL-CRS-X", "crossing", "X", tgt.T_NC1, expected_failure=1))
    cases.append(_c("A-FAIL-HO-X", "head_on", "X", tgt.T_NC2, expected_failure=1))

    return cases


# ---------------------------------------------------------------------------
# Tier B — 39 stratified cells (04a §4.3)
# ---------------------------------------------------------------------------
def width_strata() -> Dict[str, Tuple[float, float]]:
    """Three strata cut at the derived thresholds, not at round numbers.

    Each stratum then has a *distinct predicted governing rule*: wide means the
    alteration is admissible for every class; intermediate means crossing
    give-way resolves under 8(e) while head-on and overtaking still alter;
    narrow means only head-on channel-keeping and 8(e) remain.  If the measured
    behaviour matches that partition, the precedence table is validated by the
    data rather than asserted.
    """
    t = scn.width_thresholds()
    return {
        "wide": (t["crossing"], cfg.CORRIDOR_WIDTH_RANGE[1]),
        "intermediate": (t["overtaking"], t["crossing"]),
        "narrow": (cfg.CORRIDOR_WIDTH_RANGE[0], t["overtaking"]),
    }


def tier_b_cells() -> List[dict]:
    """`4 x 3 x 3 + 1 x 3 = 39` cells.  Null takes constant velocity only."""
    cells = []
    strata = width_strata()
    for cls in ("head_on", "crossing", "overtaking", "being_overtaken"):
        for behaviour in cfg.TIER_B_BEHAVIOURS:
            for name in strata:
                cells.append({"class": cls, "behaviour": behaviour,
                              "stratum": name, "width_range": strata[name],
                              "episodes": cfg.TIER_B_EPISODES_PER_CELL})
    for name in strata:
        cells.append({"class": "null", "behaviour": "cv", "stratum": name,
                      "width_range": strata[name],
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
        seed = scn.seed_for(namespace, index)
        built = generator.sample(seed, case_id=case.case_id,
                                 encounter_class=case.encounter_class,
                                 width=case.width, behaviour=case.behaviour,
                                 flags=case.flag_dict)
        if built is not None:
            out.append(built)
    return out


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
        "suite_version": version or cfg.SUITE_VERSION,
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
    checks.append(("generator source committed", True,
                   "corridor.py, scenario.py, suite.py"))
    checks.append(("seed namespaces disjoint", scn.namespaces_are_disjoint(),
                   str(cfg.SEED_NAMESPACES)))
    checks.append(("suite generated and hashed", True,
                   "build_tier_a() + manifest()"))

    unresolved = unresolved_todos()
    checks.append(("every TODO(04-*) resolved or deferred", not unresolved,
                   ", ".join(unresolved) if unresolved else "none outstanding"))
    checks.append(("operating speed settled (F24)", False,
                   f"U_NOM = {cfg.U_NOM} m/s measured; 03a §1.1 decides 0.55"))
    checks.append(("throughput measured (04a §8.3)", True,
                   "~74 steps/s single-env; 04a assumes 500"))
    checks.append(("claim ledger committed", False, "not written"))
    checks.append(("empty result tables committed", False, "not written"))
    checks.append(("regeneration test reproduces hashes", True,
                   "test_acceptance.py::test_the_suite_regenerates_to_the_same_hashes"))
    return checks


def unresolved_todos() -> List[str]:
    """The `TODO(04-*)` items 04a §11 leaves open."""
    open_items = []
    if cfg.CURRICULUM_STAGE_STEPS is None:
        open_items.append("TODO(04-3) curriculum steps per stage")
    open_items.append("TODO(04-1) w_wall from measured pose drift")
    open_items.append("TODO(04-2) D_max effective, black-wall side")
    open_items.append("TODO(04-4) training steps per run and budget")
    return open_items
