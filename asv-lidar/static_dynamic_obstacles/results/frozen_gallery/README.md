# Frozen suite gallery (Tier B, suite 3.4)

The 800 scenarios of the default frozen suite, each under its test ID.  The static
panels are the ones the environment realises for that test's episode seed
(400,000 + index), so they are the panels a policy meets.  Shortfall: none.

## Test IDs

`<stratum>-<class>-<behaviour>-<nn>`, nn counted from 01 within the cell.

| Part | Codes |
|---|---|
| stratum | BAS basin (01-20), CH channel 10 m (01-40) |
| class | HO head-on, CR crossing, OT overtaking, BO being overtaken, NU null |
| behaviour | CV constant velocity, RE reactive, NC non-compliant (null: CV only) |

Replay one or more with a policy (seconds each):

    python tools/tiers/run_test.py --model runs/sac_formulation_seed0_bl2/best_model.zip CH-CR-RE-07
    python tools/tiers/run_test.py --policy colregs_vo BAS-HO-NC-03 --supervisor both

The case id (`B-cc-nn`, cell and episode from 00) and the index (0-799) are accepted too.

## Composition

| stratum | being_overtaken | crossing | head_on | null | overtaking |
|---|---|---|---|---|---|
| basin | 100 | 100 | 100 | 100 | 100 |
| channel | 0 | 100 | 100 | 0 | 100 |

The headline is constant-velocity only, like the development set; the robustness set
reruns the same scenarios with a reactive target (every encounter class) and, in head-ons,
a non-compliant one (`CH-CR-RE-007` is `CH-CR-CV-007` with a reactive target).

## Robustness variants as realised

Share of variants whose target heading changes by more than 1 deg against a stand-on own
ship: re 94%, nc 100% (a reactive target that is the
stand-on vessel, or already passing clear, holds its course).

## Reading a figure

- Grey outline: basin envelope (10 x 25 m). Pale blue: navigable polygon (basin `P_nav`, or the channel).
- Darker blue strip (basin head-on only): the band head-on traffic keeps to.
- Dashed blue: reference leg; green dot: start; gold star: goal; green hull: own ship at spawn.
- Green dots: the own ship holding the leg at cruise (0.558191 m/s), every 5 s.
- Red hull and line: the target at spawn and its constant-velocity track against that own
  ship (the headline), dots every 5 s (same instants as the green dots).
- Orange dashed: the same target with the reactive model (robustness set); purple dashed
  (head-on only): with the non-compliant model, which alters to port.
- Dashed outlines joined by a black line: both hulls at the nominal CPA.
- Brown squares: static panels.
- Title: test ID (case id, index); class | geometry | panels; crossing side, drawn
  DCPA / TCPA / speed ratio; nominal CPA range; the variants' turn and CPA.

Sheets (`sheets/`) show one cell each.  Regenerate with `python tools/diagnostics/frozen_gallery.py`.
