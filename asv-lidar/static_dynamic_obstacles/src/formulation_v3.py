"""baseline-v3: baseline-v2 plus field-layout curriculum stages (prepared 2026-09-28).

Prepared on your call ("create baseline-v3 in case I change my mind"), **not
trained**.  It is an *overlay* on baseline-v2: the reward, observation, vessel
model, learners and stages 1-4 are v2's unchanged; what changes is what training
sees after stage 4, and the development set that selects the checkpoint.
`constants.py` is not touched, so baseline-v2's digest -- and every v2 run and
check -- is unaffected; `apply(cfg)` installs the overlay in a process that
trains or evaluates baseline-v3 (`train_formulation.py --config
configs/baseline_v3.json` does it in the learner and every worker).

Why (F105, F106): the v2 policies fail when an encounter happens beside static
panels, because v2 keeps +-0.4 T_0 of own-ship travel around the CPA clear of
panels in every stage; a fine-tune from 2 M improved the field layouts only
slowly, as late additions are learned slowly.  v3 brings the combined squeeze
into the curriculum where the policy already has both halves of it:

| Stage | From (of 2.5 M) | What changes from v2 |
|---|---|---|
| 1-4 | 0 / 0.16 / 0.36 / 0.64 M | nothing -- v2's stages at v2's step counts |
| 5 | 1.00 M | v2's stage 5, with more 3-panel episodes (0/1/2/3 at 0.10/0.20/0.30/0.40) |
| 6 | 1.50 M | CPA guard halved (0.4 -> 0.2); **25 %** field layouts (three spread panels, Paper 2 style), constant-velocity targets; panels 1-3 elsewhere, weighted to 3 |
| 7 | 2.00 M | CPA guard **off**; **45 %** field layouts, **30 %** of their targets varying speed once (`T-VS`); panels 1-3, weighted to 3 |

**Three-panel variety** (revision 3.1).  Stages 5-7 mix three kinds of
three-panel episode: the generator's clutter (panels anywhere -- settings unlike
the field), the field family (Paper 2-style arrangements on a Paper 2 leg), and,
for `NEAR_SHARE` of field layouts, **near-deployment** layouts
(`field_training.sample_near_layout`: a deployment layout with every panel moved
0.4-1.5 m, 1-1.5 m from it overall -- similar to the field set, never the tested
layout).

**Feasibility.** Every layout keeps F74's A* route.  From stage 6, every episode
with a target must also pass a **space-time** check (`feasibility_st.py`): some
trajectory -- at the own ship's top speed, waiting allowed -- reaches the goal
before the episode ends while keeping clear of every panel and of the target at
every instant.  Draws that fail are redrawn, so no training episode is
impossible.  (The same check finds all 630 Paper 2 deployment-set cases
solvable.)

**Budget.** 2.5 M steps (v2: 2 M), so stages 1-5 keep v2's absolute step
counts and the two new stages get 0.5 M each.

**Development set** (checkpoint selection): v2's 120 episodes, plus a v3 field
development set -- 20 no-target field layouts and 26 per encounter (HO, CRP,
CRS, OT, BO: 20 constant-velocity + 6 varying-speed), 150 episodes, drawn like
stage 7 from their own seed block (360,000+), every one space-time solvable;
3 in 10 of each block are near-deployment layouts (45 in all), from seeds
training never uses, so they differ from the training layouts too.
Selection is on goal - 2 x collision over all 270.
"""
from __future__ import annotations

import copy
from typing import Dict, List

ID = "baseline-v3"
REVISION = "3.1"                        # 3.1: near-deployment layouts (training and dev)

TIMESTEPS = 2_500_000
STAGE_FRACTIONS = ((0.000, 1), (0.064, 2), (0.144, 3), (0.256, 4), (0.400, 5), (0.600, 6), (0.800, 7))

CLUTTER_WEIGHTS_5 = {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.40}
CLUTTER_WEIGHTS_67 = {1: 0.20, 2: 0.30, 3: 0.50}
FIELD_WEIGHTS = {"NT": 0.20, "HO": 0.20, "CRP": 0.15, "CRS": 0.15, "OT": 0.15, "BO": 0.15}
NEAR_SHARE = 0.30                       # of field layouts: near a deployment layout

STAGE_OVERRIDES: Dict[int, dict] = {
    5: {"clutter": (0, 3), "clutter_weights": CLUTTER_WEIGHTS_5},
    6: {"clutter": (1, 3), "clutter_weights": CLUTTER_WEIGHTS_67, "cpa_guard": 0.20,
        "field_share": 0.25, "field_weights": FIELD_WEIGHTS, "field_varying_share": 0.0,
        "field_near_share": NEAR_SHARE, "st_feasibility": True},
    7: {"clutter": (1, 3), "clutter_weights": CLUTTER_WEIGHTS_67, "cpa_guard": 0.0,
        "field_share": 0.45, "field_weights": FIELD_WEIGHTS, "field_varying_share": 0.30,
        "field_near_share": NEAR_SHARE, "st_feasibility": True},
}
FIELD_PREFETCH = 6
ST_MAX_REDRAWS = 30

# Development set (v3 field part).
DEV_SEED_BASE = 360_000
DEV_NO_TARGET = 20
DEV_PER_ENCOUNTER_CV = 20
DEV_PER_ENCOUNTER_VS = 6
DEV_NEAR_PER_TEN = 3                    # of each dev block, items i with i % 10 < 3


def stages(cfg) -> Dict[int, dict]:
    """v2's stages with the v3 changes: stage 5 edited, stages 6 and 7 new (built
    on v2's stage 5)."""
    out = copy.deepcopy(cfg.CURRICULUM_STAGES)
    base5 = copy.deepcopy(out[5])
    for k, extra in STAGE_OVERRIDES.items():
        out[k] = {**(copy.deepcopy(out.get(k, base5))), **copy.deepcopy(extra)}
    return out


def apply(cfg) -> None:
    """Install the overlay in this process's `constants` module (idempotent)."""
    if getattr(cfg, "FORMULATION_OVERLAY", None) == ID:
        return
    cfg.CURRICULUM_STAGES = stages(cfg)
    cfg.CURRICULUM_STAGE_FRACTIONS = STAGE_FRACTIONS
    cfg.FIELD_PREFETCH = FIELD_PREFETCH             # read by env.set_scenario_stage
    cfg.ST_MAX_REDRAWS = ST_MAX_REDRAWS             # read by env._load_generated
    cfg.FORMULATION_OVERLAY = ID


def snapshot() -> dict:
    """What the frozen config records for the overlay (digest-bearing)."""
    import field_training as ft
    return {"id": ID, "revision": REVISION, "timesteps": TIMESTEPS,
            "stage_fractions": [list(x) for x in STAGE_FRACTIONS],
            "stage_overrides": {str(k): {kk: (list(v) if isinstance(v, tuple) else v)
                                         for kk, v in d.items()} for k, d in STAGE_OVERRIDES.items()},
            "field_prefetch": FIELD_PREFETCH, "st_max_redraws": ST_MAX_REDRAWS,
            "dev": {"seed_base": DEV_SEED_BASE, "no_target": DEV_NO_TARGET,
                    "per_encounter_cv": DEV_PER_ENCOUNTER_CV, "per_encounter_vs": DEV_PER_ENCOUNTER_VS,
                    "near_per_ten": DEV_NEAR_PER_TEN},
            "near_layouts": {"panel_shift_m": list(ft.NEAR_PANEL_SHIFT_M), "end_shift_m": ft.NEAR_END_SHIFT_M,
                             "distance_m": list(ft.NEAR_DISTANCE_M)}}


def field_development_set() -> List:
    """The v3 field development set (150 episodes; see the module docstring)."""
    import numpy as np
    import field_training as ft
    import scenario as scn
    out = []
    plan = [("NT", DEV_NO_TARGET, False)]
    for code in ft.ENCOUNTER_CODES:
        plan += [(code, DEV_PER_ENCOUNTER_CV, False), (code, DEV_PER_ENCOUNTER_VS, True)]
    for b, (code, n, vary) in enumerate(plan):
        for i in range(n):
            base = DEV_SEED_BASE + b * 1_000 + i * 20
            built = ft.sample(np.random.default_rng(base), namespace="dev_v3", encounter=code, varying=vary,
                              generator=scn.ScenarioGenerator(stage=5, seed_namespace="development"),
                              seed_fn=lambda k, base=base: base + 200_000 + k, solvable_only=True,
                              near=(i % 10) < DEV_NEAR_PER_TEN)
            built.case_id = f"DV3-{code}-{'VS' if vary else 'CV'}-{i + 1:02d}"
            out.append(built)
    return out
