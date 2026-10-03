"""baseline-v4 development-set extension (2026-10-03; the user: "adjust the dev set
if you find necessary along with the training curriculum").

The development set selects the checkpoint that is reported.  The v4.1
curriculum and test set v3 centre on dense, straight-leg layouts (field survey
lanes) and varying-speed targets, where SAC is weakest (test v3: varying speed
0.61 vs 0.78; FS-CRP 0.45, FS-BO 0.64).  The v3 field development set (DV3,
150) has near-straight legs in only about 14 % of its layouts and 30
varying-speed episodes, so a checkpoint chosen on it can be weaker exactly
there.

This **adds** (DV3 and the frozen-like 120 stay unchanged, so v3 runs remain
comparable) 60 dense field-style episodes: no target, HO, CRP, CRS, OT and BO,
10 each; legs 70 % straight / 30 % slanted; half of the target episodes with a
varying-speed target.  Layouts come from `field_training.sample` (Paper 2-style
motifs, never near L1-L3, an A* route, the field rules, space-time solvable),
from generator seeds `EXT_SEED_BASE`+ that no other set uses (training draws
from the training namespace; DV3 uses 360,000-370,380; the hand-back harvest
380,000+; the v4 gates 395,000+; test set v3 450,000+).

    import dev_set_v4; ext = dev_set_v4.extension()      # 60 scenarios, fixed order
"""
from __future__ import annotations

from typing import List

import numpy as np

EXT_SEED_BASE = 520_000
EXT_PER_CODE = 10
EXT_STRAIGHT = 0.70
EXT_VARYING = 0.50
CODES = ("NT", "HO", "CRP", "CRS", "OT", "BO")


def extension() -> List:
    import field_training as ft
    import scenario as scn
    out = []
    for b, code in enumerate(CODES):
        for i in range(EXT_PER_CODE):
            base = EXT_SEED_BASE + b * 1_000 + i * 20
            straight = int((i + 1) * EXT_STRAIGHT) > int(i * EXT_STRAIGHT)
            vary = code != "NT" and int((i + 1) * EXT_VARYING) > int(i * EXT_VARYING)
            built = ft.sample(np.random.default_rng(base), namespace="dev_v4", encounter=code, varying=vary,
                              generator=scn.ScenarioGenerator(stage=5, seed_namespace="development"),
                              seed_fn=lambda k, base=base: base + 200_000 + k, solvable_only=True,
                              near=False, straight=straight)
            built.case_id = f"DV4X-{code}-{'VS' if vary else 'CV'}-{'S' if straight else 'L'}-{i + 1:02d}"
            out.append(built)
    return out
