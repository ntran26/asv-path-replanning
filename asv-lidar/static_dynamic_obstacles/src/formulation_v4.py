"""baseline-v4 (DRAFT, 2026-10-02): baseline-v3 plus fix 1 from the start and a
denser, earlier conflict curriculum.  `planning/BASELINE_V4_PLAN.md`.

**Draft -- not frozen, not trained.**  Validation gates G1-G4 decide whether it
is used.  It is an overlay like `formulation_v3.py`; `constants.py` is not
touched, so baseline-v2/v3 digests and runs are unaffected.

Changes from v3:

| | Change |
|---|---|
| A | `ADMISSIBILITY_STATIC` on from step 0 (fix 1): a compliant turn blocked by a perceived static obstacle is inadmissible, so the context says so and R-2 / the Rule 8 credit pay for slowing |
| B | Field layouts from stage 5 and denser: 20 % (stage 5, CPA guard 0.2), 40 % (stage 6, guard off), 55 % (stage 7); encounter weights HO .25, CRS .20, CRP .20, OT .10, BO .10, NT .15; near-deployment share 0.5 |
| C | Crossings weighted up in the generator stages 5-7 (`weights`, as A31 does for stage 3) |

| D | Start-clear rule in stages 5-7 (`start_clear`): a draw whose own hull starts in contact with, or within 0.3 m of, a panel is redrawn -- gate G3 found about 1 in 200 stage-7 draws did, in v3 too |

| E | (4.1, after G4) Straight survey lanes for 70 % of dense draws and field layouts; field targets varying speed 20 / 50 / 50 %; field weights NT .15 HO .20 CRP .175 CRS .175 OT .15 BO .15 -- test set v3: varying-speed 0.61, FS-CRP 0.45, FS-BO 0.64 |

| F | (4.2, 2026-10-04) Near-impossible development episodes replaced in place: no manoeuvre that starts at or after the first track reaches the goal with 0.2 m clearance (`src/oracle_feasibility.py`); same rule as test set v4 |

Unchanged: reward, observation, learners, stage step fractions, budget.  The discount is tested separately in pilot arm B
(gamma 0.98; gate G1: the best-return behaviour is safe in 88 % of solvable
conflict scenarios at 0.951, 100 % at 0.98).
"""
from __future__ import annotations

import copy
from typing import Dict

import formulation_v3 as v3

ID = "baseline-v4"
REVISION = "4.2-draft"                 # 4.1: straight legs, more varying-speed targets, weights; 4.2: feasible dev set

TIMESTEPS = v3.TIMESTEPS
STAGE_FRACTIONS = v3.STAGE_FRACTIONS

FIELD_WEIGHTS = {"NT": 0.15, "HO": 0.20, "CRP": 0.175, "CRS": 0.175, "OT": 0.15, "BO": 0.15}
NEAR_SHARE = 0.50
CLASS_WEIGHTS_567 = {"head_on": 0.20, "crossing": 0.30, "overtaking": 0.14,
                     "being_overtaken": 0.12, "null": 0.10, "no_target": 0.14}
STRAIGHT_SHARE = 0.70                   # 4.1: straight survey lanes for dense draws and field layouts
_FIELD = {"field_weights": FIELD_WEIGHTS, "field_near_share": NEAR_SHARE, "st_feasibility": True,
          "field_straight_share": STRAIGHT_SHARE, "dense_straight_share": STRAIGHT_SHARE, "dense_panels": 3,
          "start_clear": True}        # item D: redraw starts in contact with a panel (env._start_is_clear)

STAGE_OVERRIDES: Dict[int, dict] = {
    5: {"clutter": (0, 3), "clutter_weights": v3.CLUTTER_WEIGHTS_5, "cpa_guard": 0.20,
        "field_share": 0.20, "field_varying_share": 0.20, **_FIELD},
    6: {"clutter": (1, 3), "clutter_weights": v3.CLUTTER_WEIGHTS_67, "cpa_guard": 0.0,
        "field_share": 0.40, "field_varying_share": 0.50, **_FIELD},
    7: {"clutter": (1, 3), "clutter_weights": v3.CLUTTER_WEIGHTS_67, "cpa_guard": 0.0,
        "field_share": 0.55, "field_varying_share": 0.50, **_FIELD},
}
CONSTANT_OVERRIDES = {"ADMISSIBILITY_STATIC": True}
FIELD_PREFETCH = v3.FIELD_PREFETCH
ST_MAX_REDRAWS = v3.ST_MAX_REDRAWS

def field_development_set():
    """4.1: v3's field development set (150) plus the v4 extension
    (`dev_set_v4.extension`: 60 dense field-style episodes, 70 % straight legs,
    half the targets varying speed), so checkpoints are selected where the test
    set is hardest.  4.2: near-impossible episodes replaced in place
    (`dev_set_v4.development_sets`, `configs/dev_set_v4_replacements.json`)."""
    import dev_set_v4
    return dev_set_v4.development_sets()[1]


def development_set(per_class: int = 20):
    """4.2: the frozen-like development set (`train_formulation.development_set`)
    with its near-impossible episodes replaced in place."""
    import dev_set_v4
    return dev_set_v4.development_sets(per_class)[0]


def stages(cfg) -> Dict[int, dict]:
    """v2's stages with the v4 changes (stage 5 edited, stages 6-7 built on v2's 5)."""
    if getattr(cfg, "FORMULATION_OVERLAY", None) not in (None, ID):
        raise RuntimeError(f"another overlay ({cfg.FORMULATION_OVERLAY}) is installed in this process")
    out = copy.deepcopy(cfg.CURRICULUM_STAGES)
    base5 = copy.deepcopy(out[5])
    for k, extra in STAGE_OVERRIDES.items():
        stage = {**copy.deepcopy(out.get(k, base5)), **copy.deepcopy(extra)}
        stage["weights"] = {c: w for c, w in CLASS_WEIGHTS_567.items() if c in stage["classes"]}
        out[k] = stage
    return out


def apply(cfg) -> None:
    """Install the overlay in this process's `constants` module (idempotent)."""
    if getattr(cfg, "FORMULATION_OVERLAY", None) == ID:
        return
    cfg.CURRICULUM_STAGES = stages(cfg)
    cfg.CURRICULUM_STAGE_FRACTIONS = STAGE_FRACTIONS
    cfg.FIELD_PREFETCH = FIELD_PREFETCH
    cfg.ST_MAX_REDRAWS = ST_MAX_REDRAWS
    for name, value in CONSTANT_OVERRIDES.items():
        setattr(cfg, name, value)
    cfg.FORMULATION_OVERLAY = ID


def snapshot() -> dict:
    base = v3.snapshot()
    base.update({"id": ID, "revision": REVISION,
                 "stage_overrides": {str(k): {kk: (list(v) if isinstance(v, tuple) else v)
                                              for kk, v in d.items()} for k, d in STAGE_OVERRIDES.items()},
                 "class_weights_567": CLASS_WEIGHTS_567, "constant_overrides": CONSTANT_OVERRIDES})
    return base
