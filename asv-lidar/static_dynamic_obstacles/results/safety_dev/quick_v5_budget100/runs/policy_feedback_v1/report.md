# Fixed 24-case safety comparison: policy_feedback_v1

**Goals: policy alone 18/24; v4 19/24; v5 19/24.** V5 achieved no additional goals over v4 in this sample.

Completed **72 fresh episodes: 24 identical scenario/seed triples × policy alone (off), selected v4, and candidate v5**. Development probes are excluded from these outcomes and remain charged to the shared attempt budget.

Shared budget: **76/100 attempts consumed**; 24 remain. This tag used 72 attempts for 72 completed results; 4 other attempts (including the earlier development probes) are excluded.

Cases were fixed from scenario-only strata and hash order before quick outcomes were observed. This mixed-suite diagnostic subset does **not** estimate performance over all 2,890 cases / 8,670 mode runs. No significance or confidence interval is claimed; robustness variants can share base scenarios. Field-layout and field-validation cases are simulations, not real-world trials.

## Outcomes

Counts use all episodes in each row as denominator; timeouts remain separate from collisions. Overall counts pool the 24 selected cases, rather than averaging suite percentages.

| Component | Mode | n | Goals | Obstacles | Boundaries | Targets | Timeouts |
|---|---|---:|---:|---:|---:|---:|---:|
| overall | off | 24 | 18 | 4 | 0 | 2 | 0 |
| overall | v4 | 24 | 19 | 2 | 1 | 2 | 0 |
| overall | v5 | 24 | 19 | 1 | 1 | 3 | 0 |
| frozen_b | off | 8 | 7 | 1 | 0 | 0 | 0 |
| frozen_b | v4 | 8 | 8 | 0 | 0 | 0 | 0 |
| frozen_b | v5 | 8 | 8 | 0 | 0 | 0 | 0 |
| frozen_r | off | 4 | 3 | 0 | 0 | 1 | 0 |
| frozen_r | v4 | 4 | 3 | 0 | 0 | 1 | 0 |
| frozen_r | v5 | 4 | 3 | 0 | 0 | 1 | 0 |
| frozen_a | off | 2 | 2 | 0 | 0 | 0 | 0 |
| frozen_a | v4 | 2 | 2 | 0 | 0 | 0 | 0 |
| frozen_a | v5 | 2 | 2 | 0 | 0 | 0 | 0 |
| field_validation | off | 4 | 3 | 1 | 0 | 0 | 0 |
| field_validation | v4 | 4 | 2 | 2 | 0 | 0 | 0 |
| field_validation | v5 | 4 | 2 | 1 | 0 | 1 | 0 |
| field_deployment | off | 6 | 3 | 2 | 0 | 1 | 0 |
| field_deployment | v4 | 6 | 4 | 0 | 1 | 1 | 0 |
| field_deployment | v5 | 6 | 4 | 0 | 1 | 1 | 0 |

## Paired changes

Each comparison has exactly 24 matched case IDs. A gained/lost goal compares goal against any non-goal outcome.

| Candidate vs reference | Goal gains | Goal losses | Collision → goal | Collision → timeout |
|---|---:|---:|---:|---:|
| v5_vs_v4 | 0 | 0 | 0 | 0 |
| v5_vs_off | 3 | 2 | 3 | 0 |
| v4_vs_off | 3 | 2 | 3 | 0 |

Changed outcome types:

| Comparison | Reference | Candidate | Cases |
|---|---|---|---:|
| v5_vs_v4 | collision:obstacle | collision:target | 1 |
| v5_vs_off | collision:obstacle | collision:boundary | 1 |
| v5_vs_off | collision:obstacle | goal | 2 |
| v5_vs_off | collision:target | goal | 1 |
| v5_vs_off | goal | collision:target | 2 |
| v4_vs_off | collision:obstacle | collision:boundary | 1 |
| v4_vs_off | collision:obstacle | goal | 2 |
| v4_vs_off | collision:target | goal | 1 |
| v4_vs_off | goal | collision:obstacle | 1 |
| v4_vs_off | goal | collision:target | 1 |

## Mechanism use

Cells show **steps / episodes with at least one such step**. `policy_feedback_preserved` counts accepted policy-feedback certification checks; it does not establish that a safety override was prevented or that the action changed. `continuation_override_prevented` is a separate diagnostic. Enabled-step counts describe configuration, not interventions.

| Diagnostic | off | v4 | v5 |
|---|---:|---:|---:|
| sideslip_rescue_evaluated | 0 / 0 | 0 / 0 | 0 / 0 |
| sideslip_rescue_admitted | 0 / 0 | 0 / 0 | 0 / 0 |
| verified_policy_priority_enabled | 0 / 0 | 0 / 0 | 1068 / 24 |
| verified_policy_preserved | 0 / 0 | 0 / 0 | 132 / 10 |
| continuation_override_prevented | 0 / 0 | 0 / 0 | 39 / 9 |
| policy_feedback_evaluated | 0 / 0 | 0 / 0 | 42 / 15 |
| policy_feedback_preserved | 0 / 0 | 0 / 0 | 6 / 3 |

Changed-action steps: off 0, v4 152, v5 121. A smaller intervention count is not evidence of a safety improvement.

Selected recovery-template step counts:

- off: `{}`
- v4: `{"continuation": 86, "v3": 892}`
- v5: `{"continuation": 62, "current_heading": 1, "edge_parallel": 2, "starboard_30": 3, "v3": 928}`

## Runtime

Recorded per-episode elapsed runtime excludes model loading and runner setup; totals are not end-to-end campaign wall time.

| Mode | Total seconds | Median seconds/episode | Maximum seconds/episode |
|---|---:|---:|---:|
| off | 60.74 | 2.38 | 7.69 |
| v4 | 171.21 | 7.36 | 16.36 |
| v5 | 176.91 | 7.16 | 15.46 |

## Provenance and limits

Policy checkpoint SHA256: `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`.

Frozen candidate switches: `PREFER_CERTIFIED_POLICY=True`; `POLICY_FEEDBACK_PRESERVATION=True`; `FEEDBACK_BACKUPS=False`; `SIDESLIP_RESCUE=False`; `ONE_DECISION_COMMIT=False`; `SOFT_RECOVERY=False`.

The report validates all 72 unique identities, original reset seeds/scenario digests, metadata and source hashes, immutable budget tokens, and completed-summary accounting. Per-episode JSON records are authoritative. Earlier partial benchmark outcomes and selected development counterfactuals are separate evidence. Omitted family coverage remains in `sample_family_coverage_v3.csv`; no inverse-probability weighting is applied.

Frozen evaluated code: [source archive](evaluated_sources.zip) and [archive manifest](evaluated_sources_manifest.json).
