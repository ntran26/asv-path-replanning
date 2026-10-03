# Offline search-margin opportunity audit

Captured 2026-10-02T16:16:28.255839+00:00. No new episodes or source changes.

Among 17 failed V6 episodes (718 decisions), **13 decisions in 6 cases** meet all three logged conditions: original `any_safe=False`, escape search `accepted=False`, and `0 <= maximum_clearance < 0.15` m.

| Case | Outcome | Count | First decision | Last decision | Fallback reasons |
|---|---|---:|---:|---:|---|
| DV3-BO-CV-07 | collision:obstacle | 0 | - | - | - |
| DV3-BO-CV-08 | collision:target | 0 | - | - | - |
| DV3-BO-CV-10 | collision:obstacle | 0 | - | - | - |
| DV3-BO-CV-18 | collision:obstacle | 0 | - | - | - |
| DV3-BO-VS-05 | collision:obstacle | 0 | - | - | - |
| DV3-CRP-CV-02 | collision:target | 0 | - | - | - |
| DV3-CRP-CV-13 | collision:target | 0 | - | - | - |
| DV3-CRP-CV-16 | collision:obstacle | 4 | 46 | 56 | last certificate: 4 |
| DV3-CRP-VS-02 | collision:obstacle | 1 | 20 | 20 | no escape: 1 |
| DV3-CRS-CV-02 | collision:target | 0 | - | - | - |
| DV3-CRS-CV-04 | collision:boundary | 2 | 60 | 61 | hold back: 2 |
| DV3-CRS-VS-03 | collision:obstacle | 2 | 54 | 58 | hold back: 1, last certificate: 1 |
| DV3-HO-CV-04 | collision:boundary | 0 | - | - | - |
| DV3-HO-CV-10 | collision:target | 1 | 23 | 23 | no escape: 1 |
| DV3-HO-VS-01 | collision:boundary | 3 | 16 | 57 | last certificate: 3 |
| DV3-NT-CV-06 | collision:obstacle | 0 | - | - | - |
| DV3-NT-CV-15 | collision:obstacle | 0 | - | - | - |

Fallback totals: {"hold back": 3, "last certificate": 8, "no escape": 2}. Matches with a currently checked safe continuation: 0.

These logged nonnegative maxima below the unchanged extra search acceptance margin suggest sampled plans satisfying the hard sampled-path clearance checks were discarded by the margin gate. The first-contact value and command sequence of the maximum-clearance sample were not separately logged, so this is a next-iteration hypothesis, not a replay-verified plan or demonstrated rescue.

The frozen evaluator derives first violation from negative clearance and folds terminal violations into the same minimum clearance. Thus the recorded nonnegative maximum is evidence of a potentially hard-safe sampled alternative rejected by the extra margin, under that onboard model. The maximizer's separate first-contact result and exact command sequence were not retained; this audit cannot replay it or show that executing it would avoid the later contact. No threshold change is proposed or tested here.

The original bank flag excludes continuation, and the report records continuation status separately. Fallback branch names do not establish that the next executed command itself collides. Perception/model errors, finite-time sampling and missing recursive terminal viability remain. These failure-conditioned counts do not estimate overall success.

Exact matching decisions, clearances, fallback diagnostics, trace hashes and frozen source provenance are in search_margin_opportunities.json.
