# Completed primary quick V19 trace audit

Created 2026-10-03T11:55:56.976956+00:00. Both runs completed cleanly:54/54 records,18 exact scene/seed triples. This audit ran no episodes and makes no controller-tuning recommendation.

| Mode | Goals | Obstacle | Target | Boundary |
|---|---:|---:|---:|---:|
| off | 14 | 1 | 1 | 2 |
| v16 | 15 | 0 | 1 | 2 |
| v19 | 15 | 0 | 1 | 2 |

| Candidate/reference | Preserved goals | Lost goals | Rescued failures | Both failed |
|---|---:|---:|---:|---:|
| v16_vs_off | 14 | 0 | 1 | 3 |
| v19_vs_off | 14 | 0 | 1 | 3 |
| v19_vs_v16 | 15 | 0 | 0 | 3 |

V19 transfers current returns on 11 decisions across 4 cases (29 point-decision exposures). Its complete issued command sequences exactly equal V16 in 18/18 cases; 0 cases have a command difference above1e-7.

| Case | OFF | V16 | V19 | V19 active steps | V19/V16 first command divergence |
|---|---|---|---|---|---|
| TS3:BAS-BO-CV-041 | goal | goal | goal | None | None |
| TS3:BAS-BO-RE-082 | collision:boundary | collision:boundary | collision:boundary | None | None |
| TS3:BAS-CR-RE-032 | goal | goal | goal | None | None |
| TS3:BAS-HO-RE-015 | goal | goal | goal | None | None |
| TS3:BAS-NU-CV-014 | goal | goal | goal | None | None |
| TS3:CH-CR-CV-089 | goal | goal | goal | None | None |
| TS3:CH-HO-NC-092 | goal | goal | goal | 17, 18 | None |
| TS3:CH-OT-CV-089 | goal | goal | goal | None | None |
| TS3:CH-OT-RE-084 | goal | goal | goal | None | None |
| TS3:FS-BO-FIX-019 | collision:obstacle | collision:target | collision:target | None | None |
| TS3:FS-BO-VAR-018 | goal | goal | goal | None | None |
| TS3:FS-CRP-VAR-023 | collision:target | goal | goal | 5, 8, 10, 11, 12, 15 | None |
| TS3:FS-HO-FIX-009 | goal | goal | goal | None | None |
| TS3:FS-HO-VAR-011 | goal | goal | goal | 21, 22 | None |
| TS3:FS-NT-019 | goal | goal | goal | None | None |
| TS3:FS-OT-FIX-054 | goal | goal | goal | None | None |
| TS3:FS-OT-VAR-043 | goal | goal | goal | None | None |
| TS3:P2-L2-CRS-FIX-17 | collision:boundary | collision:boundary | collision:boundary | 3 | None |

The JSON records each activation and first command difference, including shared pre-state/policy-action checks, commands and filter reasons. It also preserves all input hashes. This fixed subset is an evaluation sample, not a full-population safety estimate. No controller or threshold was selected, fitted or changed from these primary outcomes.
