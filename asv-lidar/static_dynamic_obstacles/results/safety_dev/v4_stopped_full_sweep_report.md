# V4 results from the stopped full sweep

This is the **final snapshot of the stopped B/R evaluation runs**, captured on 2026-10-02 at 13:23:11 UTC. It is a partial benchmark, not a completed full sweep. The stop confirmation records both old evaluation sessions as exited and their exact evaluation PIDs absent at 13:16:47 UTC. This statement concerns those stopped runs only, not other training or evaluation processes. No missing episodes were replayed or imputed.

The selected v4 uses the model-based ego observer and free-space memory. Countersteer recovery, nominal-plan retention and dual-brake prediction are disabled. All comparisons use SAC baseline 3 at 3M timesteps (checkpoint SHA-256 `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8`).

| Set | Matched cases | Policy goals | V4 goals | Policy success | V4 success | Goals gained / lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original DV3 development (complete) | 150 | 111 | 123 | 74.00% | 82.00% | 19 / 7 |
| Frozen headline B (partial) | 791 | 740 | 737 | 93.55% | 93.17% | 23 / 26 |
| Frozen robustness R (partial) | 856 | 781 | 788 | 91.24% | 92.06% | 27 / 20 |
| Frozen B + R pooled (partial) | 1,647 | 1,521 | 1,525 | 92.35% | 92.59% | 50 / 46 |

On the original DV3 development set, v4 adds 12 goals and removes 12 collisions. On the stopped frozen intersection it adds four goals (+0.24 percentage points) and removes four collisions. The pooled net comprises 50 policy collisions converted to goals and 46 policy goals converted to collisions. Headline B is three goals worse; robustness R is seven goals better. These are descriptive paired counts only: sequential truncation need not represent the complete suites, and robustness variants share underlying scenarios rather than being independent statistical replicates. This partial result does not establish a population-level safety improvement.

| Set and mode | Obstacle contacts | Boundary contacts | Target contacts | All collisions | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| DV3 policy | 26 | 2 | 11 | 39 | 0 |
| DV3 v4 | 9 | 7 | 11 | 27 | 0 |
| Headline policy, matched | 13 | 7 | 31 | 51 | 0 |
| Headline v4, matched | 9 | 12 | 33 | 54 | 0 |
| Robustness policy, matched | 18 | 6 | 51 | 75 | 0 |
| Robustness v4, matched | 9 | 14 | 45 | 68 | 0 |
| Frozen pooled policy, matched | 31 | 13 | 82 | 126 | 0 |
| Frozen pooled v4, matched | 18 | 26 | 78 | 122 | 0 |

Contact counts are terminal collision outcomes, one category per episode. On matched frozen cases, v4 removes 13 obstacle contacts and four target contacts while adding 13 boundary contacts. The net four-collision reduction coexists with these contact-type trades and 46 lost policy successes. Outcome changes identify where performance differs; this report does not infer a controller cause or use held-out outcomes to tune a candidate.

The durable journals contain **3,347 committed records**:

| Original component | Planned per mode | Policy recorded | V4 recorded | V5 recorded | Matched policy/v4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen B | 800 | 799 | 792 | 0 | 791 |
| Frozen R | 900 | 900 | 856 | 0 | 856 |
| B + R | 1,700 | 1,699 | 1,648 | 0 | 1,647 |

The policy record for **B-06-040** is absent from the durable journal, although an older CSV reconciliation note contains a row. Its v4 result is excluded from paired analysis; no committed policy result was reconstructed from that note. The note is preserved as historical evidence; its earlier suggestion to rerun the case is superseded by the stop decision. Both final journals have zero incomplete final-line bytes.

The full original completion map records **3,347 of 8,670 planned records** and marks **5,323 not recorded**. The map covers all eight original simulated components and off/v4/v5, including unstarted identities. Only B and R have compatible committed records. No v5 result or result for the unstarted simulated field, development or Tier A continuations is inferred. The original DV3 development table above comes from the earlier development study, is reported separately, and is not counted toward this stopped full-sweep inventory. Simulated field layouts and validation scenarios would not constitute new real-world trials.

Pairing was verified by component, case ID, reset seed and scenario SHA-256. The comparison tool accepted matching checkpoint, configuration, recorded source hashes, effective constants, runtime versions and thread/process settings across B and R. The seven original DV3 source CSV hashes were checked against their existing comparison-source manifest. The full completion-map manifest/journal hashes exactly match this archived snapshot. This report verifies recorded artifacts; it does not retrospectively inspect evaluator process memory.

Generated artifacts:

- [Paired summary](v4_stopped_full_sweep_comparison.csv), [per-case transitions](v4_stopped_full_sweep_paired.csv), [raw completed-mode counts](v4_stopped_full_sweep_outcomes.csv), and [comparison provenance](v4_stopped_full_sweep_sources.json). Raw totals have unequal completed-case denominators and must not be compared as if paired.
- [Full 8,670-identity completion map](quick_v5_budget100/original_completion_map_20261002T132245Z_116a97b9.csv) and [map provenance](quick_v5_budget100/original_completion_map_20261002T132245Z_116a97b9.json).
- [Stop confirmation](cancelled/stop_confirmed.json).
- [Final evidence snapshot](cancelled/v4_stopped_full_sweep_20261002T132311Z/snapshot.json), containing exact final journal/manifest bytes and auxiliary CSV/progress/lock/reconciliation evidence. A copied `running.lock` is historical evidence, not a claim that its process remains alive.

The final archive also includes these reports, the completion map and planned inventory, original DV3 evidence, comparison scripts, and a SHA-256 file inventory. Its ZIP has a separate SHA-256 sidecar. Files are marked read-only after sealing; these are integrity checks, not immutable storage. The earlier `v4_cancelled_full_sweep_*` interim report and archive remain untouched and retain their original shutdown-unconfirmed status.
