# Saved OFF/V4 outcomes on exact test-set-v2 matches

Offline analysis only: no environment construction, reset, step, model loading, or new episode.

The archived stopped journals contain **735 exact paired cases** from the current 1000-case definition. Pairing requires test ID, scene digest and reset seed; both archived modes share checkpoint, evaluator settings, source hashes and runtime settings.

| Mode | Goals | Obstacle | Boundary | Target | Total contacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| OFF | 684 | 14 | 5 | 32 | 51 |
| V4 | 680 | 10 | 11 | 34 | 55 |

V4 rescues **19** failed OFF episodes but loses **23** OFF goals. These are observed episode trades, not ground-truth labels for individual false overrides.

| Episode transition | Cases | Cases with recorded intervention | Intervention decisions | Brake decisions |
| --- | ---: | ---: | ---: | ---: |
| preserved_goal | 661 | 262 | 1736 | 176 |
| lost_goal | 23 | 23 | 298 | 99 |
| rescue | 19 | 19 | 367 | 51 |
| both_failed | 32 | 22 | 324 | 55 |

The journal's `safety_v2_steps` and `safety_v2_brake_steps` are recorded episode totals. A successful OFF episode shows that its complete policy trajectory succeeded; it does not show that every nominal action after V4 has diverted the boat would remain safe. Interventions in preserved-goal episodes therefore cannot all be counted as false positives. The journals do not contain predecision sensors/actions or filter margins, so they cannot support an online trigger classifier or local causal replay without additional already-recorded decision traces.

| Cell | Paired | OFF goals | V4 goals | Rescues | Lost goals |
| --- | ---: | ---: | ---: | ---: | ---: |
| basin-being_overtaken | 95 | 90 | 88 | 2 | 4 |
| basin-crossing | 100 | 84 | 83 | 6 | 7 |
| basin-head_on | 100 | 92 | 89 | 2 | 5 |
| basin-null | 75 | 72 | 71 | 0 | 1 |
| basin-overtaking | 100 | 97 | 98 | 2 | 1 |
| channel-crossing | 99 | 87 | 90 | 6 | 3 |
| channel-head_on | 98 | 94 | 93 | 1 | 2 |
| channel-overtaking | 68 | 68 | 68 | 0 | 0 |

All 760 available matching archived OFF outcomes were checked against the saved v2 SAC CSV; disagreements: 0. The latter CSV lacks its own run-level source/settings manifest, so it is a consistency check and is not substituted for a missing committed OFF result.

Coverage: `{"off_only": 25, "paired": 735, "unrecorded": 239, "v4_only": 1}`. All pairs come from the frozen-source portion. None of the simulated Paper 2 layout cases has a paired V4 result in these stopped journals. They are simulated cases, not new field trials. The 15 original L1-BO IDs were replaced by 15 regenerated IDs; the other 985 shared IDs have unchanged scene digests and reset seeds. No removed scene was matched to a replacement. Missing records, including B-06-040 OFF, remain missing.

This partial sequential sample is not a full-v2 rate estimate and reports archived V4 only. It provides no V6/V7 success claim and no evidence for a 95% result. The baseline checkpoint and configuration hashes match the saved evaluator metadata; current source equivalence is not assumed.

Artifacts: [paired cases](pairs.csv), [goal trades](goal_trades.csv), [all 1,000 coverage identities](coverage.csv), [summary](summary.json), [full input hashes and archived settings](provenance.json).
