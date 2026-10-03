# BAS-NU: the original intervention is removed, but no passing backup is found at the next decision

Completed V15 case `TS2:BAS-NU-CV-070`, attempt 014, contacts the boundary after 78 decisions with one intervention. The saved SAC-only reference reaches the goal after 58 decisions. Scenario, seed, checkpoint and configuration match. This audit makes no environment or policy calls and consumes no episodes.

**Decision 10 is now preserved.** Its issued command exactly matches OFF. The parent proposes a passing +0.118179901 m turn. The exact ordinary selection best is +0.264765912 m, supplied by the passing continuation, so the existing floor is:

`min(0.15, 0.264765912 - 0.20) = 0.064765912 m`.

V15 requests that floor. The 192-plan prefix search finds 121 hard-passing plans, of which 72 meet the requested margin. Its best is +0.103391124 m, and it preserves SAC. This is the intended ordinary-floor behavior.

**The new first command difference is decision 11.** Its recorded pre-state and SAC action still exactly match OFF. SAC requests rudder +0.8456063 at 11.8408427 RPM; V15 instead issues rudder -0.9027146 at 12 RPM. There is one represented target view, raw source 84.

| Decision-11 quantity | Recorded value |
| --- | ---: |
| Parent reason | searched escape |
| Parent actual-command paired clearance | +0.150102828 m |
| Parent escape-search maximum | +0.160599714 m |
| Existing continuation clearance | +0.145362495 m |
| Exact ordinary selection floor / best / branch | unavailable: search returned before ordinary selection |
| Original one-second fixed-SAC search maximum | -0.105675784 m |
| Original fixed-SAC accepted plans | 0; hard-passing count not logged |
| Policy command followed by parent's same tail | -0.015479061 m |
| V15 extra-prefix request | skipped: `adequate_parent_clearance` |

The selected parent is only **0.000102828 m** above the existing +0.15 m trigger, but the gate is implemented as specified: an already adequate passing parent is kept. There is no ordinary selection floor to use on this early CEM escape branch.

## Does this gate exclude a passing prefix here?

The evaluated trace did not run the extra prefix search at decision 11. A separate **offline saved-state counterfactual** called the unchanged bounded search with the captured snapshot, exact retained parent plan, pre-command actuators and reconstructed prefix RNG. It finds **zero hard-passing plans out of 192**, maximum clearance **-0.007452623 m**. Thus removing the gate alone would not preserve SAC with this same candidate search. Changing its requested margin from +0.15 to zero would not make a negative-clearance candidate hard-passing.

RNG reconstruction follows recorded normal-draw batches from prefix calls at decisions 7, 9 and 10. As an independent check, replaying decision 10 reproduces its complete accepted plan, +0.103391124 m maximum, 121 hard passes, 72 accepted plans, and the exact next generator state. No parent controller was replayed.

The actual successful next 16 OFF commands also fail the current model when scored from decision 11: first violation at 1.75 s and minimum clearance -0.154722999 m. Those future commands are offline evidence only and are unavailable online.

The logged adequate-parent condition explains why the additional search was skipped, but this audit finds no passing backup being excluded by that condition at this state. The small positive-parent threshold crossing does not justify a new rule, and no outcome-separating condition is inferred from this case. Passing or failing these finite model checks is not proof of physical safety or unavoidability.

`audit.json` contains trace/source hashes, exact recorded decisions, paired identity checks, the counterfactual search diagnostics and RNG validation. `reproduce.py` repeats saved-data scoring only; all frozen controller sources remain unchanged.
