# V16 decision branches before contact: saved-trace audit

Date: 2026-10-10. Read-only analysis of existing V16 traces for the disjoint 32- and 40-case development cohorts (72 cases, 3,801 decisions). **No episode, policy call, environment construction or controller change was made.** Simulator truth surge is used only to score when the vessel was at rest. Produced by [`audit_v20_saved_cascades.py`](../../../../tools/diagnostics/safety/audit_v20_saved_cascades.py); every input trace and paired table is hashed in [`audit.json`](audit.json); per-case rows are in [`cases.csv`](cases.csv).

## Branch labels

V16 records a `why` for each decision. Two branches issue a command whose plan the filter's own checker did **not** pass at that decision:

- `last certificate`: a stored plan whose current recheck fails is followed (at most four decisions);
- `no escape`: no plan passes, and the SAC command stands unchecked.

All other non-idle branches (`nominal`, `searched policy`, `policy prefix searched/repaired`, `policy margin dominates`, `turn`, `continue`, `brake`, `searched escape`, `handback`) issue a command whose full plan hard-passed the inherited nominal checker at that decision. `idle` means nothing was inside the engagement test. Across the 3,801 decisions there are 213 `no escape` and 171 `last certificate` decisions.

## Contacts follow unchecked branches

| Quantity | Count |
| --- | ---: |
| V16 contact episodes in the 72 cases | 22 |
| Final decision `no escape` | 18 |
| Final decision `last certificate` | 3 |
| Final decision `nominal` (DV3-BO-CV-10, contact within the first 0.5 s from an initial gap of about 0.22 m) | 1 |

Twenty-one of 22 contacts occur immediately after one or more consecutive unchecked decisions. The unchecked run before contact lasts 1-15 decisions (median 7, i.e. about 3.5 s, over those 21 episodes). For the six lost SAC successes it lasts 6 (BAS-HO-NC-059), 14 (BAS-NU-CV-070), 15 (DV3-CRS-CV-04), 9 (P2-L2-CRS-FIX-19), 11 (P2-L3-BO-FIX-09) and 14 (P2-L3-CRP-FIX-03) decisions. In BAS-NU-CV-070 the last checked decision is `idle` (decision 64); the following 14 decisions are `no escape` while the vessel approaches the end wall, where braking candidates are excluded because no target is within the traffic gate.

This is an association, not a demonstrated cause. An unchecked phase begins when the checker already finds no passing plan, so some of these states may already be inevitable-collision states. Whether a certified contingency existed at the last checked decision, and whether it would have survived the next recheck, is the question the V20 offline gates test.

## What does not separate losses from rescues

| Group | Cases | Intervened | Braked | At rest after first change (truth-scored) | Any `last certificate` | Any `no escape` | SAC + same tail hard-passes at first change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lost SAC success | 6 | 6 | 5 | 5 | 5 | 6 | 2 |
| Rescue | 19 | 19 | 13 | 11 | 14 | 10 | 8 |
| Both fail | 16 | 13 | 9 | 6 | 10 | 13 | 1 |
| Both goal | 31 | 18 | 5 | 4 | 12 | 7 | 8 |

Braking to rest and `last certificate` phases are as common in rescues as in losses, and rescues also pass through `no escape` phases (43 `no escape` and 70 `last certificate` decisions across the 19 rescues; 42 and 42 across the 31 both-goal cases). Removing braking, or forbidding rest, is therefore not supported by this evidence.

At V16's first intervention, SAC followed by the proposal's own tail hard-passes the nominal checker in 2 of 6 losses (BAS-HO-NC-059, P2-L2-CRS-FIX-19) but also in 8 of 19 rescues. A rule that preserves SAC whenever its tail hard-passes would also skip V16's first intervention in those 8 rescues; their later outcome under such a rule is unknown. This is consistent with the earlier any-feasible-policy result (3/9 versus 6/9). In that variant P2-L1-CRP-VAR-12 ends in target contact after a `nominal` decision, i.e. a checked zero-margin SAC plan whose target forecast was optimistic. The information available at the first intervention does not distinguish these groups without outcome-based tuning.

## Limits

The two cohorts are failure-enriched development selections, not population samples, and decisions within an episode are correlated. Branch labels describe the inherited nominal checker, not physical safety. The audit does not establish that any contact was avoidable.
