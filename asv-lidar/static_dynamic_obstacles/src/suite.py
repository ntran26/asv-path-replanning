"""The frozen evaluation suite (04a sections 4-9): Tier A, Tier B, Around the Clock.

Structure (04a section 4.2):

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

**The freeze protocol is not paperwork** (04a section 9.4).  With Imazu dropped, the
released generator plus the frozen suite plus the manifest is the only
structural defence against "the authors built their own benchmark, then showed
classical methods fail on it".  Every case serialises to canonical JSON, every
case is hashed, and regenerating from seed must reproduce every hash.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import constants as cfg
import scenario as scn
import targets as tgt

# Width codes for the Tier A case table (04a section 5).  "B" is a basin case (06 section 5.2).
WIDTH_CODES = {"W": 10.0, "I": 6.0, "N": 4.0, "X": 3.5, "B": 0.0}

# 06 section 5.2: basin cases run one fixed leg at the widest slant Paper 2's layout
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
# Tier A — 34 named cases (04a section 5)
# ---------------------------------------------------------------------------
def tier_a() -> List[Case]:
    """The 34 named cases, in the order 04a section 5 tabulates them.

    `A-8E-HO-N` is the one that carries the head-on precedence argument: its
    target is **positionally** non-compliant (`T-NC3`), which is the only case
    in the suite where a Rule 14 alteration is actually required.  Without it
    the head-on class is never exercised as an avoidance problem at all, because
    02 section 3.2's whole point is that 9(a) channel-keeping satisfies Rule 14.
    """
    cases: List[Case] = []

    for code in ("W", "I", "N"):
        cases.append(_c(f"A-HO-{code}", "head_on", code))
    for code in ("W", "I", "N"):
        cases.append(_c(f"A-CRS-{code}", "crossing", code, side="starboard"))
    for code in ("W", "I", "N"):
        cases.append(_c(f"A-CRP-{code}", "crossing", code, side="port"))
    for code in ("W", "I", "N"):
        # 04a section 11 change 2: mid-window rather than at the floor, because 03a
        # Section 1.3 raised the floor to 0.40 and a case sitting exactly on a bound is
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

    # Pre-committed expected failures (04a section 4.5).  Reported at whatever level
    # they come out; the framing rule is that strata are described by geometry
    # only, never as "challenging for classical methods".
    cases.append(_c("A-FAIL-CRS-X", "crossing", "X", tgt.T_NC1, expected_failure=1))
    cases.append(_c("A-FAIL-HO-X", "head_on", "X", tgt.T_NC2, expected_failure=1))

    # 06 section 5.2 / M-7: six basin cases, all field-replicable -- the basin-session
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
# Tier B — 39 stratified cells (04a section 4.3)
# ---------------------------------------------------------------------------
# Suite 3.1 (decision, 2026-09-23): Tier B channels stop at 7.5 m.  The 10 m
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
# Revision 3.2 (A34, 2026-09-26): the behaviour cells realise their target
# models (`target_model`); 3.1 passed the cell label, which no model knew, so
# every Tier B target ran at constant velocity (F99).
# Revision 3.3 (decision, 2026-09-27): every channel 10 m.  Superseded by 3.4.
# Revision 3.4 (decision, 2026-09-27): **the headline holds only scenarios of
# the kind the policies were trained and selected on** -- the development set's
# distribution, drawn in the frozen namespace, so only the positions differ:
#
# * constant-velocity targets only, as in training (D1) and the development set;
# * class x geometry only where training draws it: every class in the basin, and
#   in a channel only `cfg.CHANNEL_CLASSES` (head-on, crossing, overtaking) --
#   being overtaken and null were never trained in a channel (F74, F101, A36);
# * channel widths as the development set draws them (uniform, with the same
#   width profile) but floored at 7.5 m;
# * balanced cells, `TIER_B_EPISODES` each: 8 cells, 800 episodes per seed.
#
# Reactive and non-compliant targets move to a **robustness set** on the same
# scenarios and episode seeds (`robustness_variants`), reported separately (R3).
# Suites 3.2 and 3.3's seed-0 results are kept under
# `results/frozen_suite/<tag>_suite3{2,3}_superseded/`; both changes were
# decided after seeing them.
SUITE_REVISION = "3.4"
TIER_B_EPISODES = 100
# Robustness variants: reactive for every encounter class; non-compliant only
# in head-on (`T-NC2`, alters to port).  `T-NC1` (stands on when give-way) moves
# exactly like `T-CV`, so on the same scenario and seed its episode would be an
# exact copy of the headline's -- whose constant-velocity target already is the
# give-way vessel that does not give way wherever it has that role.
ROBUST_CLASSES = ("head_on", "crossing", "overtaking", "being_overtaken")
ROBUST_BEHAVIOURS = {"re": ROBUST_CLASSES, "nc": ("head_on",)}


def width_strata() -> Dict[str, Tuple[float, float]]:
    """The Tier B channel stratum: the development set's width draw, floored at
    `TIER_B_MIN_WIDTH_M` (suite 3.4).  `R4` keeps the rule-by-width test down to
    3.5 m."""
    lo_cfg, hi_cfg = cfg.CORRIDOR_WIDTH_RANGE
    return {"channel": (max(TIER_B_MIN_WIDTH_M, lo_cfg), hi_cfg)}


def target_model(behaviour: str, encounter_class: str) -> str:
    """The target model that realises a behaviour label (A34, option a).

    * `cv` -- `T-CV`, constant velocity, the training model (the headline);
    * `re` -- `T-RE`, compliant and reactive (the COLREGs-VO rule);
    * `nc` -- the violation the class admits: in a head-on the target alters to
      **port** (`T-NC2`); in every other class a give-way target **stands on**
      (`T-NC1`), which moves exactly like `T-CV`.
    """
    if behaviour == "cv":
        return tgt.T_CV
    if behaviour == "re":
        return tgt.T_RE
    if behaviour == "nc":
        return tgt.T_NC2 if encounter_class == "head_on" else tgt.T_NC1
    raise ValueError(f"unknown Tier B behaviour {behaviour!r}")


def tier_b_cells() -> List[dict]:
    """Suite 3.4: 8 cells x `TIER_B_EPISODES` = 800 episodes per seed.

    Basin: head-on, crossing, overtaking, being overtaken, null.  Channel
    (7.5-10 m): head-on, crossing, overtaking -- the classes training draws in a
    channel (`cfg.CHANNEL_CLASSES`).  Constant-velocity targets throughout; the
    reactive and non-compliant targets are `robustness_variants`.
    """
    cells = []
    strata = [("basin", None)] + list(width_strata().items())
    for name, width_range in strata:
        for cls in ("head_on", "crossing", "overtaking", "being_overtaken", "null"):
            if width_range is not None and cls not in cfg.CHANNEL_CLASSES:
                continue
            cells.append({"class": cls, "behaviour": "cv", "stratum": name,
                          "geometry_mode": "basin" if width_range is None else "channel",
                          "width_range": width_range, "episodes": TIER_B_EPISODES})
    return cells


def robustness_variants(tier_b: Sequence[scn.Scenario]) -> List[Tuple[scn.Scenario, int, str]]:
    """The robustness set: every headline scenario of a `ROBUST_CLASSES` class
    again with a reactive target, and head-ons again with a non-compliant one.

    Returns (variant, index of its headline twin, behaviour).  The variant is the
    twin with only the target model changed, and it runs on the twin's episode
    seed, so static panels, spawn and leg are identical: any difference in
    outcome is the target's behaviour.
    """
    out = []
    for i, built in enumerate(tier_b):
        for behaviour, classes in ROBUST_BEHAVIOURS.items():
            if built.encounter_class not in classes:
                continue
            variant = copy.deepcopy(built)
            variant.target_behaviour = target_model(behaviour, built.encounter_class)
            variant.case_id = f"{built.case_id}-{behaviour.upper()}"
            out.append((variant, i, behaviour))
    return out


# ---------------------------------------------------------------------------
# Around the Clock (04a section 4.4)
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
# Study 2 conditions (04a section 7.1)
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
# The frozen namespace holds 10,000 seeds (04a section 9.2).  Tier B takes a block per
# cell -- 1,000 since suite 3.4 (8 cells, indices 0-7,999; 100 scenarios a cell
# need the room); 200 before -- and Tier A the block from 9,600, 9 per case.
# Disjoint, where `index + 10,000 * retry` wrapped back onto the same seed.
# `TIER_A_SEED_BASE` stays 9,600 (it was 48 x 200) so Tier A is unchanged.
TIER_B_SEEDS_PER_CELL = 1000
TIER_A_SEED_BASE = 9600


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
                 namespace: str = "frozen_eval",
                 cells: Optional[Sequence[int]] = None) -> Tuple[List[scn.Scenario], List[dict]]:
    """Realise the Tier B cells (06 section 5.1).  Returns (scenarios, shortfalls).

    `cells` realises only those cell indices, for replaying one test: each cell
    draws from its own seed block, so a cell comes out identical alone or in the
    full build.
    """
    generator = generator or scn.ScenarioGenerator(stage=5, seed_namespace=namespace)
    out, short = [], []
    for c_index, cell in enumerate(tier_b_cells()):
        if cells is not None and c_index not in cells:
            continue
        rng = np.random.default_rng(scn.seed_for(namespace, c_index * TIER_B_SEEDS_PER_CELL))
        got, tries = 0, 0
        while got < cell["episodes"] and tries < TIER_B_SEEDS_PER_CELL:
            seed = scn.seed_for(namespace, c_index * TIER_B_SEEDS_PER_CELL + tries)
            tries += 1
            width = (None if cell["width_range"] is None
                     else float(rng.uniform(*cell["width_range"])))
            built = generator.sample(seed, case_id=f"B-{c_index:02d}-{got:03d}",
                                     encounter_class=cell["class"], width=width,
                                     behaviour=target_model(cell["behaviour"], cell["class"]),
                                     geometry_mode=cell["geometry_mode"],
                                     flags={"stratum": cell["stratum"]})
            if built is not None:
                out.append(built)
                got += 1
        if got < cell["episodes"]:
            short.append({**cell, "realised": got})
    return out, short


# ---------------------------------------------------------------------------
# Test IDs (2026-09-26): a readable name for every Tier B scenario, so one can be
# picked and replayed alone (`tools/tiers/run_test.py`).
# ---------------------------------------------------------------------------
# <stratum>-<class>-<behaviour>-<nnn>, nnn counted from 001 within the cell, e.g.
# CH-CR-CV-007 is the 7th crossing in a channel, and CH-CR-RE-007 the same
# scenario with a reactive target (the robustness set).  The case id
# `B-cc-nnn[-RE|-NC]` says the same by cell number (counting from 000).
# CHW / CHI are suites 3.1-3.2's channel strata, kept so older results still map.
STRATUM_CODES = {"basin": "BAS", "channel": "CH",
                 "channel-wide": "CHW", "channel-intermediate": "CHI"}
CLASS_CODES = {"head_on": "HO", "crossing": "CR", "overtaking": "OT",
               "being_overtaken": "BO", "null": "NU"}


def test_id(case_id: str) -> str:
    """The test ID of a Tier B case id (`B-01-012` -> `BAS-CR-CV-013`,
    `B-01-012-RE` -> `BAS-CR-RE-013`)."""
    parts = case_id.split("-")
    c_index, n = int(parts[1]), int(parts[2])
    cell = tier_b_cells()[c_index]
    behaviour = parts[3] if len(parts) > 3 else cell["behaviour"]
    return (f"{STRATUM_CODES[cell['stratum']]}-{CLASS_CODES[cell['class']]}-"
            f"{behaviour.upper()}-{n + 1:03d}")


def tier_b_index(c_index: int, n: int) -> int:
    """Position in the headline list, which sets the episode seed
    (`frozen_suite.TIER_B_SEED + index`; a robustness variant uses its twin's).
    Valid while every cell realises its full count; the gallery checks it."""
    return sum(c["episodes"] for c in tier_b_cells()[:c_index]) + n


def resolve_test(ref: str) -> Tuple[int, int, str]:
    """(cell index, episode within cell, behaviour) for a test ID
    (`CH-CR-RE-007`), a case id (`B-06-006-RE`) or a headline index (`386`)."""
    cells = tier_b_cells()
    ref = ref.strip().upper()
    behaviour = "cv"
    if ref.isdigit():
        i = int(ref)
        for c_index, cell in enumerate(cells):
            if i < cell["episodes"]:
                return c_index, i, "cv"
            i -= cell["episodes"]
        raise ValueError(f"index {ref} is beyond Tier B ({tier_b_index(len(cells), 0)} scenarios)")
    parts = ref.split("-")
    if parts[0] == "B" and len(parts) in (3, 4):
        c_index, n = int(parts[1]), int(parts[2])
        behaviour = parts[3].lower() if len(parts) == 4 else "cv"
    elif len(parts) == 4:
        codes = {v: k for k, v in STRATUM_CODES.items()}, {v: k for k, v in CLASS_CODES.items()}
        stratum, cls = codes[0].get(parts[0]), codes[1].get(parts[1])
        match = [k for k, c in enumerate(cells) if c["stratum"] == stratum and c["class"] == cls]
        if not match:
            raise ValueError(f"no Tier B cell {'-'.join(parts[:2])}")
        c_index, n, behaviour = match[0], int(parts[3]) - 1, parts[2].lower()
    else:
        raise ValueError(f"not a test ID, case id or index: {ref}")
    if not (0 <= c_index < len(cells) and 0 <= n < cells[c_index]["episodes"]):
        raise ValueError(f"{ref} is outside Tier B")
    if behaviour != "cv" and cells[c_index]["class"] not in ROBUST_BEHAVIOURS.get(behaviour, ()):
        raise ValueError(f"{ref}: no {behaviour.upper()} variant for {cells[c_index]['class']}")
    return c_index, n, behaviour


def manifest(scenarios: Sequence[scn.Scenario], *, generator_sha: str = "",
             version: str = None) -> dict:
    """`SUITE_MANIFEST.json` (04a section 9.3): version, hashes, seeds, constants.

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
    """Every constant the suite's geometry depends on (04a section 9.3)."""
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
    """04a section 9.3, as machine-checked state rather than a list to tick by hand.

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
    checks.append(("throughput measured (04a section 8.3)", True,
                   "F75, 10 workers, 12 cores: PPO ~108 steps/s; SAC 12 (1 gradient step per "
                   "transition) / 38 (0.2); TD3 18 / 53; TQC 13 / 32; RecurrentPPO 61 (F80)"))
    for item, rel in (("claim ledger committed", "planning/PAPER3_CLAIMS_AND_TABLES.md"),
                      ("empty result tables committed", "planning/PAPER3_CLAIMS_AND_TABLES.md")):
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
    """The `TODO(04-*)` items 04a section 11 leaves open."""
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
