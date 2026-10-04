"""Revise the baseline-v4 draft after G4 (queued 2026-10-03).

Test set v3 showed SAC weakest on varying-speed targets (0.61 vs 0.78 constant),
dense crossing-from-port (FS-CRP 0.45) and dense being-overtaken (FS-BO 0.64),
and field deployment runs straight survey lanes.  This applies, to
`src/formulation_v4.py`:
* `dense_straight_share` 0.70 (dense generator draws) and `field_straight_share`
  0.70 (field-family layouts), stages 5-7;
* field targets varying speed: 0.20 / 0.50 / 0.50 in stages 5 / 6 / 7;
* field encounter weights NT .15, HO .20, CRP .175, CRS .175, OT .15, BO .15;
* the development set gains `dev_set_v4.extension()` (60 dense field-style
  episodes, 70 % straight legs, half the targets varying speed); DV3 and the
  frozen-like 120 are unchanged;
* revision 4.0-draft -> 4.1-draft.
Run once, after the G4 pilots have trained (they read the 4.0 draft).
"""
from pathlib import Path

p = Path(__file__).resolve().parents[3] / "src" / "formulation_v4.py"
s = p.read_text(encoding="utf-8")
if 'REVISION = "4.1-draft"' in s:
    print("already revised"); raise SystemExit(0)
pairs = [
    ('REVISION = "4.0-draft"', 'REVISION = "4.1-draft"                 # 4.1: straight legs, more varying-speed targets, weights'),
    ('FIELD_WEIGHTS = {"NT": 0.15, "HO": 0.25, "CRP": 0.20, "CRS": 0.20, "OT": 0.10, "BO": 0.10}',
     'FIELD_WEIGHTS = {"NT": 0.15, "HO": 0.20, "CRP": 0.175, "CRS": 0.175, "OT": 0.15, "BO": 0.15}'),
    ('''_FIELD = {"field_weights": FIELD_WEIGHTS, "field_near_share": NEAR_SHARE, "st_feasibility": True,''',
     '''STRAIGHT_SHARE = 0.70                   # 4.1: straight survey lanes for dense draws and field layouts
_FIELD = {"field_weights": FIELD_WEIGHTS, "field_near_share": NEAR_SHARE, "st_feasibility": True,
          "field_straight_share": STRAIGHT_SHARE, "dense_straight_share": STRAIGHT_SHARE, "dense_panels": 3,'''),
    ('''        "field_share": 0.20, "field_varying_share": 0.0, **_FIELD},''',
     '''        "field_share": 0.20, "field_varying_share": 0.20, **_FIELD},'''),
    ('''        "field_share": 0.40, "field_varying_share": 0.15, **_FIELD},''',
     '''        "field_share": 0.40, "field_varying_share": 0.50, **_FIELD},'''),
    ('''        "field_share": 0.55, "field_varying_share": 0.30, **_FIELD},''',
     '''        "field_share": 0.55, "field_varying_share": 0.50, **_FIELD},'''),
    ("field_development_set = v3.field_development_set",
     '''def field_development_set():
    """4.1: v3's field development set (150, unchanged) plus the v4 extension
    (`dev_set_v4.extension`: 60 dense field-style episodes, 70 % straight legs,
    half the targets varying speed), so checkpoints are selected where test set
    v3 is hardest; the frozen-like 120 stay as they are."""
    import dev_set_v4
    return v3.field_development_set() + dev_set_v4.extension()'''),
    ('''Unchanged: reward, observation, learners, stage step fractions, budget, the''',
     '''| E | (4.1, after G4) Straight survey lanes for 70 % of dense draws and field layouts; field targets varying speed 20 / 50 / 50 %; field weights NT .15 HO .20 CRP .175 CRS .175 OT .15 BO .15 -- test set v3: varying-speed 0.61, FS-CRP 0.45, FS-BO 0.64 |

Unchanged: reward, observation, learners, stage step fractions, budget, the'''),
]
for a, b in pairs:
    assert s.count(a) == 1, (a[:60], s.count(a))
    s = s.replace(a, b)
compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("formulation_v4 revised to 4.1-draft")
