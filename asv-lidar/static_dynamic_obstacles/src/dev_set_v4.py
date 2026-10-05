"""baseline-v4 development-set extension (2026-10-03; decision: adjust the dev set
where necessary, along with the training curriculum).

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


# ---------------------------------------------------------------------------
# 4.2 (2026-10-04): near-impossible development episodes replaced.
# Decision: episodes that fail the oracle test (src/oracle_feasibility.py: no
# manoeuvre that starts at or after the first track reaches the goal with 0.2 m
# clearance) leave the development set, as they leave test set v4.  Each is
# replaced in place, so it keeps its position and evaluation seed, by the next
# draw of the generator that made its set
# (tools/diagnostics/feasibility/dev_replacements.py); the accepted recipes are
# recorded in configs/dev_set_v4_replacements.json.
# ---------------------------------------------------------------------------
def dev_v2_used(cls: str, per_class: int = 20) -> int:
    """The first development-namespace index of `cls` that
    `train_formulation.development_set(per_class)` does not use."""
    import scenario as scn
    import train_formulation as tf
    generator = scn.ScenarioGenerator(stage=5, seed_namespace="development")
    index, found = 0, 0
    while found < per_class and index < 50 * per_class:
        built = generator.sample(scn.seed_for("development", 10_000 * (tf.EVAL_CLASSES.index(cls) + 1) + index),
                                 encounter_class=cls)
        index += 1
        found += built is not None
    return index


def from_recipe(recipe: dict):
    """One development scenario from a replacement recipe (None if the draw fails)."""
    import field_training as ft
    import scenario as scn
    s = recipe["set"]
    if s == "dev_v2":
        import train_formulation as tf
        cls = recipe["class"]
        generator = scn.ScenarioGenerator(stage=5, seed_namespace="development")
        built = generator.sample(scn.seed_for("development", 10_000 * (tf.EVAL_CLASSES.index(cls) + 1)
                                              + int(recipe["index"])), encounter_class=cls)
        if built is not None:
            built.case_id = f"DEV-{cls}-R{int(recipe['index']):03d}"
        return built
    code, vary, i = recipe["code"], bool(recipe["varying"]), int(recipe["i"])
    if s == "field_dev":
        import formulation_v3 as fv
        plan = [("NT", fv.DEV_NO_TARGET, False)]
        for c in ft.ENCOUNTER_CODES:
            plan += [(c, fv.DEV_PER_ENCOUNTER_CV, False), (c, fv.DEV_PER_ENCOUNTER_VS, True)]
        b = next(j for j, (c, _, v) in enumerate(plan) if c == code and v == vary)
        base = fv.DEV_SEED_BASE + b * 1_000 + i * 20
        built = ft.sample(np.random.default_rng(base), namespace="dev_v3", encounter=code, varying=vary,
                          generator=scn.ScenarioGenerator(stage=5, seed_namespace="development"),
                          seed_fn=lambda k, base=base: base + 200_000 + k, solvable_only=True,
                          near=bool(recipe["near"]))
        built.case_id = f"DV3-{code}-{'VS' if vary else 'CV'}-R{i + 1:02d}"
        return built
    straight = bool(recipe["straight"])
    base = EXT_SEED_BASE + CODES.index(code) * 1_000 + i * 20
    built = ft.sample(np.random.default_rng(base), namespace="dev_v4", encounter=code, varying=vary,
                      generator=scn.ScenarioGenerator(stage=5, seed_namespace="development"),
                      seed_fn=lambda k, base=base: base + 200_000 + k, solvable_only=True,
                      near=False, straight=straight)
    built.case_id = f"DV4X-{code}-{'VS' if vary else 'CV'}-{'S' if straight else 'L'}-R{i + 1:02d}"
    return built


def development_sets(per_class: int = 20):
    """(frozen-like, field) development sets of formulation v4.2, in evaluation
    order, with the recorded replacements applied in place.  Positions run over
    the frozen-like set first, then the field set (the v3 field development set,
    then the extension), as `FieldEvalCallback` evaluates them."""
    import json
    from pathlib import Path
    import formulation_v3 as fv
    import train_formulation as tf
    frozen_like = list(tf.development_set(per_class))
    field = list(fv.field_development_set()) + extension()
    path = Path(__file__).resolve().parents[1] / "configs" / "dev_set_v4_replacements.json"
    if path.exists():
        # Positions were recorded on the full set: 20 per class, so the field set starts
        # at 120.  A smaller frozen-like set (a smoke run's per_class=1) keeps the field
        # replacements and skips the frozen-like ones, whose positions do not exist there.
        n_full = 20 * len(tf.EVAL_CLASSES)
        for r in json.loads(path.read_text(encoding="utf-8"))["replacements"]:
            pos = int(r["position"])
            if pos < n_full:
                if per_class == 20:
                    frozen_like[pos] = from_recipe(r["recipe"])
            else:
                field[pos - n_full] = from_recipe(r["recipe"])
    return frozen_like, field
