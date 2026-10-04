# V4 results from the cancelled full sweep

This is an **interim cancellation snapshot captured on 2026-10-02 at 11:50:51 UTC**, not a completed benchmark. Evaluation was stopped. Process shutdown was unconfirmed at capture; records may have continued to arrive afterward. This report makes no claim about process termination. No missing episodes were replayed or imputed.

The selected v4 uses the model-based ego observer and free-space memory. Countersteer recovery, nominal-plan retention and dual-brake prediction are disabled. All comparisons below use the same SAC baseline 3 checkpoint at 3M timesteps.

| Set | Matched cases | Policy goals | V4 goals | Policy success | V4 success | Goals gained / lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original DV3 development (complete) | 150 | 111 | 123 | 74.00% | 82.00% | 19 / 7 |
| Frozen headline B (partial) | 689 | 642 | 636 | 93.18% | 92.31% | 20 / 26 |
| Frozen robustness R (partial) | 752 | 689 | 691 | 91.62% | 91.89% | 21 / 19 |
| Frozen B + R pooled (partial) | 1,441 | 1,331 | 1,327 | 92.37% | 92.09% | 41 / 45 |

V4 improves the original development set by 12 goals. On the available frozen pairs it is four goals behind the policy alone (-0.28 percentage points). The available frozen results therefore do not establish an overall improvement. These are descriptive episode counts; robustness variants share underlying scenarios and are not independent statistical replicates.

| Set and mode | Obstacle contacts | Boundary contacts | Target contacts | All collisions | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| DV3 policy | 26 | 2 | 11 | 39 | 0 |
| DV3 v4 | 9 | 7 | 11 | 27 | 0 |
| Headline policy, matched | 13 | 7 | 27 | 47 | 0 |
| Headline v4, matched | 9 | 12 | 32 | 53 | 0 |
| Robustness policy, matched | 13 | 6 | 44 | 63 | 0 |
| Robustness v4, matched | 7 | 14 | 40 | 61 | 0 |
| Frozen pooled policy, matched | 26 | 13 | 71 | 110 | 0 |
| Frozen pooled v4, matched | 16 | 26 | 72 | 114 | 0 |

On the matched frozen cases, v4 reduces obstacle contacts by 10, adds 13 boundary contacts and adds one target contact. This pattern identifies boundary handling as a remaining weakness; it does not prove a specific controller cause.

The durable journals contain 3,141 completed records: R has off 900/900, v4 752/900 and v5 0/900; B has off 799/800, v4 690/800 and v5 0/800. The raw 1,442 v4 records include one unpaired case. The policy record for B-06-040 is missing from the durable journal, although a historical CSV reconciliation note contains a row. It is excluded from paired analysis; that note is preserved as evidence and does not authorize reconstructing a committed result. The journals have no incomplete final-line bytes at this snapshot.

Only B and R are represented in this cancelled snapshot. No result for the unstarted field/development/Tier A continuations or for v5 is inferred. Sequentially truncated coverage need not represent the complete declared suites. The original DV3 development result is reported separately and is not included in the frozen pooled total.

Pairing was verified by suite component, case ID, reset seed and scenario SHA-256. The comparison tool accepted matching checkpoint, configuration, recorded source hashes, effective constants, runtime versions and thread/process settings across B and R. The original DV3 source CSV hashes were checked against their existing comparison-source manifest. Runtime process memory and successful shutdown were not audited by this report.

Generated reports: `v4_cancelled_full_sweep_comparison.csv` contains the paired summary; `v4_cancelled_full_sweep_paired.csv` contains per-case transitions; `v4_cancelled_full_sweep_outcomes.csv` contains raw completed-mode totals with coverage; `v4_cancelled_full_sweep_sources.json` records source snapshots and incomplete benchmark coverage. Rates from raw outcome totals must not be compared across modes with different completed cases.

Evidence is archived under `cancelled/v4_cancelled_full_sweep_20261002T115051Z/`, including exact journal and manifest bytes, auxiliary CSV/progress/reconciliation evidence, original DV3 source artifacts, comparison tools, this report and SHA-256 file inventory. Archive files and the ZIP are marked read-only after sealing, with a separate ZIP SHA-256 sidecar. This is a reproducible, tamper-evident snapshot rather than proof that the original running directories stopped changing.
