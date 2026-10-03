# V14 BAS mechanism check from completed saved traces

No new episodes, environment calls, policy calls, or controller edits were made. The isolated four-case V14 probe is complete. The combined V14 plus prefix-search campaign remains ongoing; its completed BAS case is analysed separately here.

| BAS-CR-RE-072 mode | Outcome | Decisions | Changed actions | First divergence from saved SAC-only |
| --- | --- | ---: | ---: | ---: |
| SAC-only | goal | 49 | 0 | none |
| V13 | goal | 103 | 28 | 8 |
| V14 default | goal | 49 | 1 | 26 |
| V14 plus prefix search | goal | 49 | 0 | none |

V14 default preserves the current SAC action on 48 of 49 decisions. Its commands and recorded trajectory exactly match SAC-only through decision 25, with the same pre-state and policy action at decision 26. It then selects rudder 0 and RPM 12 instead of the SAC rudder 0.5647044 and RPM 11.9670761. Its subsequent trajectory is therefore different even though it reaches the goal in the same number of decisions.

The completed combined run preserves all 49 SAC actions. Every executed rudder/RPM command, policy action, and recorded pre/post `[x, y, heading, surge]` state exactly matches the saved SAC-only episode. At decision 26, the additional 0.5-second policy-prefix search certifies a continuation with clearance +0.225394 m and removes the remaining intervention. The old SAC-only trace does not record sway, yaw, full observations, or plant actuator state; parity claims are restricted to fields actually recorded.

## The decision-8 estimate change reproduces the predicted mechanism

All versions have identical recorded pre-state and policy action at decision 8. V14 publishes the stored corrected persistent position/velocity found by the earlier saved-state replacement audit, now in the real evaluated decision. Its anchor is 0.5 s old. Relative to the current true target used only for scoring, centre error falls from 0.448428 m to 0.076963 m and velocity error from 0.156172 m/s to 0.075200 m/s. Raw IDs differ across processes; the comparison uses matched numerical estimates, while each controller associates only exact IDs within its own run.

Replaying the same saved successful next 16 SAC commands through the unchanged predictor gives:

| Decision-8 target representation | Predicted minimum clearance | First violation |
| --- | ---: | ---: |
| V13 base view | -0.225572 m | 3.375 s |
| V14 corrected persistent view | +0.214817 m | none |

The replacement clearance reproduces the earlier prediction to within 1e-12 m. This scoring uses future commands only as an offline diagnostic; they are unavailable online. No future reactive-target truth or constant-velocity truth assumption is used.

Online, V13 selects a last certificate with current clearance -0.167043 m. V14 finds a searched-policy continuation with +0.318713 m and keeps the SAC action. Thus the evidence supports removing an erroneous rejection by improving the target representation. It does not establish that simply increasing the search bank would have repaired V13's rejection of this same known-successful continuation.

## Other completed perception probes

| Case suffix | V13 | V14 | V13 / V14 changed actions |
| --- | --- | --- | ---: |
| CRP-FIX-05 | target contact, 39 decisions | target contact, 39 decisions | 11 / 11 |
| CRS-VAR-08 | obstacle contact, 57 decisions | goal, 91 decisions | 26 / 23 |
| CRP-VAR-14 | goal, 88 decisions | goal, 73 decisions | 34 / 15 |

FIX05 has exactly identical recorded commands, policy actions, and pre/post states under V13 and V14 throughout all 39 decisions. It is an unresolved failure, not a new V14 regression. The independent [FIX05 audit](../fix05_v13_final_audit/REPORT.md) therefore remains directly relevant: a newly numbered biased target view cannot be removed by exact-ID selection; the target subsequently stops while an old hypothesis coasts; and loss of the fitted hull heading at decision 34 substitutes noisy velocity direction, changing the collision rectangle by about 33.6 degrees. Its fixed-plan heading sensitivity is diagnostic evidence, not a demonstrated rescue.

This four-case development probe has one gained goal and no lost goals versus V13, but the small selected sample does not establish broad safety improvement. The combined campaign's aggregate outcomes must be reported only when its own completed records support them.

`audit.json` records trace hashes, matching scenario/seed/checkpoint/config provenance, predictor-source checks, exact field comparisons, decision diagnostics, and the reproduced margins. `reproduce.py` performs saved-data scoring only.
