"""A35 / S18: what judging the encounter-specific VO's passing side on the encounter costs.

Scores the encounter-specific VO on the decoupled part of the validation set (120
episodes, baseline-v3 build) three ways: the per-candidate side rule with the
comparators v1 settings, the per-candidate rule with v2's course-change weight, and
comparators v2 (the rule judged on the engaged encounter, `CLASSICAL_VO_SIDE_ON_ENCOUNTER`).
Validation set only: the frozen choice (S18) does not depend on it, and the test set
is not used.

    python tools/diagnostics/a35_side_rule_check.py

Writes `results/classical_tuning/encounter_vo/side_rule_check.txt`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "tiers"), str(ROOT / "tools" / "diagnostics")]
import constants as cfg  # noqa: E402
import formulation_v3  # noqa: E402

formulation_v3.apply(cfg)
import tune_classical as tc  # noqa: E402
from common import development_set  # noqa: E402

OUT = ROOT / "results" / "classical_tuning" / "encounter_vo" / "side_rule_check.txt"
SIDE = ("classical.encounter_vo", "SIDE_ON_ENCOUNTER", "CLASSICAL_VO_SIDE_ON_ENCOUNTER")
W_CHANGE = ("classical.encounter_vo", "W_CHANGE", "CLASSICAL_VO_W_CHANGE")
ARMS = (("per-candidate rule, W_CHANGE 0.3 (comparators v1)", False, 0.3),
        ("per-candidate rule, W_CHANGE 0.8", False, 0.8),
        ("encounter rule, W_CHANGE 0.8 (comparators v2)", True, 0.8))


def main() -> int:
    full = development_set(20)
    lines = [f"Encounter-specific VO, side rule (A35): {len(full)} decoupled validation episodes "
             "(baseline-v3 build); score = goal - 2 x collision", ""]
    for label, side, w in ARMS:
        d = tc.evaluate("encounter_vo", {SIDE: side, W_CHANGE: w}, full, 10, overlay=True)
        by = d.groupby("class").apply(lambda x: round((x.outcome == "goal").mean(), 2), include_groups=False)
        lines.append(f"{label:50s} score {tc._score(d):+.3f}  goal {(d.outcome == 'goal').mean():.3f}  "
                     f"target collision {d.collided_target.mean():.3f}  | goal by class {by.to_dict()}")
        print(lines[-1], flush=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
