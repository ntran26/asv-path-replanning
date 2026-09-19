# Development set gallery

The 120 scenarios every formulation run is scored on (`train_formulation.development_set(20)`:
20 per class, stage 5, the `development` seed namespace). Evaluation seeds are 900 000 + index,
so the static panels shown are exactly the ones the policies met.

## Composition

| class | basin | channel |
|---|---|---|
| being_overtaken | 20 | 0 |
| crossing | 18 | 2 |
| head_on | 13 | 7 |
| no_target | 20 | 0 |
| null | 20 | 0 |
| overtaking | 13 | 7 |

- **Basin mode: 104 of 120.** Legs run from y = 2 m to y = 22 m,
  both x drawn in [2.5, 7.5] m (Paper 2's layout), inside the
  whole 10 x 25 m basin less the 0.40 m inset.
- **Slant:** mean |slant| 5.2 deg, max 12.6 deg; 43% of basin legs exceed 5 deg,
  13% exceed 10 deg. Independent x draws make large slants rare: two uniform
  draws 5 m apart at most, over a 20 m leg, cap the slant at 14.0 deg.
- **Channel mode: 16 of 120**, only head-on, crossing and overtaking (the classes whose
  rule the width decides), widths 4.2-9.8 m.
- Panels per scenario: 0.9 on average (55% have none).

## Goal rate by class (Tier 1, supervisor off)

| class | run8_final | run8_best | run7 | reference |
|---|---|---|---|---|
| being_overtaken | 0.9 | 0.85 | 0.9 | 0.95 |
| crossing | 0.55 | 0.75 | 0.3 | 0.6 |
| head_on | 0.85 | 0.95 | 0.85 | 0.95 |
| no_target | 0.95 | 1.0 | 1.0 | 0.95 |
| null | 0.9 | 0.95 | 1.0 | 1.0 |
| overtaking | 0.95 | 0.95 | 1.0 | 0.9 |

## Reading a figure

- Grey outline: basin envelope (10 x 25 m). Pale blue: navigable polygon (basin `P_nav`, or the channel).
- Darker blue strip (basin head-on only): the path band head-on traffic keeps to (Rule 9(a) with Rule 14).
- Dashed blue: reference leg; green dot: start; gold star: goal; green hull: own ship at spawn;
  green x: where the own ship would be at CPA if it held the leg at cruise.
- Red hull: target at spawn; dotted red arrow: its constant-velocity track to CPA (12 s for null);
  dashed red outline: target hull at CPA.
- Brown squares: static panels (A*-feasible layouts, F74).
- Title: index, class, geometry (basin slant or channel width), panel count, crossing side,
  drawn DCPA / TCPA / speed ratio, and the latest outcomes (run 8 final, run 8 best checkpoint,
  run 7, reference controller).

Regenerate with `python tools/diagnostics/devset_gallery.py`.
