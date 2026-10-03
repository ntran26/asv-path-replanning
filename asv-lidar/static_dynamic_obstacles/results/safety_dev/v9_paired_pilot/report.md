# Fresh SAC/V8/V9 paired development pilot

**V9 reached 16/32 goals, V8 16/32 and SAC alone 16/32.** V9 achieved the same number of goals as V8 in this fixed cohort: 6 gained and 6 lost. This is a diagnostic pilot, not an estimate for all 1,000 test cases.

The user's later request explicitly authorized fresh episodes to determine whether V9 is better or worse, superseding the earlier no-new-runs restriction for this pilot. Selection was fixed before evaluation: 32 scenarios × SAC alone/V8/V9 = **96 new episode runs**, with no historical result reuse or automatic retries. This report performs no additional runs. Five cases are DV3 and 27 are test-set v2; all are development evidence.

## Outcomes

| Mode | N | Goal | Target | Obstacle | Boundary | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| off | 32 | 16 | 8 | 6 | 2 | 0 |
| v8 | 32 | 16 | 9 | 2 | 5 | 0 |
| v9 | 32 | 16 | 8 | 6 | 2 | 0 |

All rates use 32 cases per controller; errors or missing records are rejected before reporting.

| Candidate vs reference | Matched | Gained goals | Lost goals | Net | Exact paired p (descriptive) |
|---|---:|---:|---:|---:|---:|
| v9_vs_v8 | 32 | 6 | 6 | +0 | 1 |
| v9_vs_off | 32 | 5 | 5 | +0 | 1 |
| v8_vs_off | 32 | 10 | 10 | +0 | 1 |

Exact two-sided McNemar values use twice the smaller Binomial(discordant pairs, 0.5) tail, capped at one ([Fagerland, Lydersen & Laake, 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3716987/)). They are descriptive here: outcome-enriched selection and shared scenario families do not justify a population significance claim. Collision→timeout is reported separately from collision→goal in summary.json.

V9 gained over V8: TS2:BAS-HO-NC-059, TS2:BAS-CR-RE-072, DV3:DV3-BO-CV-04, TS2:CH-CR-RE-038, TS2:CH-CR-CV-031, TS2:P2-L2-HO-FIX-19.
V9 lost versus V8: DV3:DV3-HO-CV-03, DV3:DV3-CRP-CV-04, DV3:DV3-CRP-CV-17, DV3:DV3-HO-VS-01, TS2:P2-L2-HO-FIX-05, TS2:CH-CR-CV-073.

## Dataset and selection strata

| Slice | N | SAC goals | V8 goals | V9 goals | V9/V8 gains | V9/V8 losses |
|---|---:|---:|---:|---:|---:|---:|
| dataset: dv3 | 5 | 1 | 4 | 1 | 1 | 4 |
| dataset: ts2 | 27 | 15 | 12 | 15 | 5 | 2 |
| stratum: expired_consequences | 8 | 4 | 4 | 4 | 4 | 4 |
| stratum: other_broken_boundary | 2 | 2 | 0 | 1 | 1 | 0 |
| stratum: other_broken_obstacle | 2 | 2 | 0 | 0 | 0 | 0 |
| stratum: other_broken_target | 2 | 2 | 0 | 0 | 0 | 0 |
| stratum: rescued_brake | 1 | 0 | 1 | 0 | 0 | 1 |
| stratum: rescued_continue | 2 | 0 | 2 | 1 | 0 | 1 |
| stratum: rescued_searched_escape | 1 | 0 | 1 | 1 | 0 | 0 |
| stratum: rescued_turn | 2 | 0 | 2 | 2 | 0 | 0 |
| stratum: success_active_turn | 1 | 1 | 1 | 1 | 0 | 0 |
| stratum: success_expired_repair_false | 1 | 1 | 1 | 1 | 0 | 0 |
| stratum: success_expired_repair_true | 2 | 2 | 2 | 2 | 0 | 0 |
| stratum: success_never_fire_field_layout | 1 | 1 | 1 | 1 | 0 | 0 |
| stratum: success_never_fire_frozen | 1 | 1 | 1 | 1 | 0 | 0 |
| stratum: unrescued_expired | 2 | 0 | 0 | 0 | 0 | 0 |
| stratum: unrescued_optimistic_no_fire | 3 | 0 | 0 | 0 | 0 | 0 |
| stratum: unrescued_other_active | 1 | 0 | 0 | 1 | 1 | 0 |

Per-slice collision types and all three paired comparisons are retained in summary.json. The selected cohort deliberately contains known rescues and breaks; its aggregate percentage cannot be compared directly with V8's saved 912/1,000 rate.

## Baseline reproduction

- Fresh off: 32/32 outcomes match their recorded historical outcome.
- Fresh v8: 32/32 outcomes match their recorded historical outcome.

All paired claims use the fresh outcomes regardless of historical agreement.

## Intervention mechanisms and first command divergence

| Mode | Changed-action steps | Guard checks (steps/episodes) | Suppressed proposals (steps/episodes) | Episode seconds |
|---|---:|---:|---:|---:|
| off | 0 | 0/0 | 0/0 | 81.24 |
| v8 | 393 | 0/0 | 0/0 | 1032.36 |
| v9 | 106 | 144/27 | 38/23 | 558.66 |

V8 and V9 first issue different commands in 23/32 cases. paired.csv records the first differing executed rudder/RPM command, both reasons, V9's parent reason, and its actual selected-versus-policy-tail margins and first-violation times. Steps are one-based. Command tolerances are 1e-09 normalized rudder and 1e-06 RPM. Pre-state comparison covers only recorded x/y/heading/surge; it is not a check of the entire hidden simulator state. Equal command prefixes with different trace lengths are marked explicitly. More suppressed steps or fewer changed actions alone is not evidence of improved safety. The first divergence locates where behavior separates; later interventions also differ, so it neither proves that one decision caused the final outcome nor isolates the effects of V9's two guards.

Decision-reason totals retain the runner's raw labels. In enabled V8/V9 traces, `off` can mean an idle filter with no `why` field; it does not establish that safety was disabled.

Runner wall time: 1680.88 s. Per-mode seconds sum timed episodes; they exclude some setup/archive work and reflect one serial process with rotated controller order. Outcomes change episode duration: an early collision can reduce runtime, so these totals do not isolate filter computation speed.

## Provenance

[Selection](selection.json), [frozen manifest](manifest.json), [evaluated source archive](evaluated_sources.zip), [completion](completion.json), [paired cases](paired.csv) and [report provenance](report_provenance.json). The report validates all96 tokens/results/traces, selection identity and seeds, completion flags, and every archived source hash. Source settings belong to the archived pilot; no result is credited to later controller edits. Original artifacts are read with Windows shared handles and are unchanged.
