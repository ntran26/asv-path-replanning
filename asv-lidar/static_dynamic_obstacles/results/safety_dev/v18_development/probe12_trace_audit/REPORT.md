# Completed V18/V19 development command audit

Created 2026-10-03T11:47:53.092937+00:00. Both runs have clean completion records: 36/36 episodes, 12 exact scene/seed triples. This analysis ran no episodes.

All three controllers reached **5/12 goals**. Both candidates have **0 gains and 0 losses** against fresh V16. V19 changes one failed case from boundary to obstacle contact; this is not a rescue.

Fresh V16 reproduces **12/12 older outcomes** and **12/12 complete exact command sequences** from motion_axis_probe5, motion_axis_remaining27 and v16_broader40_paired. Case IDs, scenario digests, seeds and checkpoint/config hashes match.

| Case | V16 | V18 | V19 | V18 active decisions | V19 active decisions | First command difference |
|---|---|---|---|---:|---:|---|
| DV3:DV3-BO-CV-04 | goal | goal | goal | 0 | 0 | None |
| DV3:DV3-CRS-CV-04 | collision:boundary | collision:boundary | collision:obstacle | 3 | 2 | v19: step 13 |
| TS2:BAS-HO-NC-059 | collision:target | collision:target | collision:target | 0 | 0 | None |
| TS2:BAS-NU-CV-070 | collision:boundary | collision:boundary | collision:boundary | 0 | 0 | None |
| TS2:CH-HO-CV-043 | goal | goal | goal | 0 | 0 | None |
| TS2:CH-HO-CV-073 | goal | goal | goal | 0 | 0 | None |
| TS2:P2-L1-CRP-FIX-05 | collision:target | collision:target | collision:target | 1 | 1 | None |
| TS2:P2-L1-CRP-VAR-12 | goal | goal | goal | 0 | 0 | None |
| TS2:P2-L1-CRS-FIX-09 | goal | goal | goal | 0 | 0 | None |
| TS2:P2-L2-CRS-FIX-19 | collision:boundary | collision:boundary | collision:boundary | 0 | 1 | None |
| TS2:P2-L3-BO-FIX-09 | collision:boundary | collision:boundary | collision:boundary | 0 | 0 | None |
| TS2:P2-L3-CRP-FIX-03 | collision:target | collision:target | collision:target | 0 | 0 | None |

V19's first changed command in DV3-CRS-CV-04 is at decision 13: V16 rudder −0.3452420533 versus V19 −0.3387790024; both command −24 RPM and report searched escape. The current-return transfer removes 9 of 58 static points at this decision. The pre-state and SAC action still match. Later trajectories diverge and end in different contact types.

V18 can correct geometry without changing the issued command. In fresh FIX05 its only activation is the terminal decision 34; the older V13 saved-state sensitivity is a different closed-loop trajectory and does not establish an earlier rescue opportunity in this V16 run.

The JSON retains every activated decision's evidence, commands and reason, first differences, historical comparisons and input hashes. Point totals are point-decision exposures, not unique physical returns. No primary test-set-v3 results were read. These selected development cases do not establish statistical safety or population success.
