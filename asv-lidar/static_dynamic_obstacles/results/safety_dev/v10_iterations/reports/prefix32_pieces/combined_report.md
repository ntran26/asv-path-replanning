# Policy-prefix search: explicit 32-case sum

The two completed 16-case pieces total **17/32 goals**, with 9 target, 2 obstacle and 4 boundary collisions, and 0 timeouts.

This table sums separately validated pieces with differing source inventories. All 99 shared sources, controller options, checkpoint/config and scenario identities match; the later archive additionally contains `src/dev_set_v4.py` and `src/safety_v14.py`. The cached loader bypasses the new generator. The [source-inventory audit](../prefix32_inventory_audit/audit.md) records these checks and their dynamic-import limitation. The strict cohort API was not overridden.

| Reference | Matched | Gained goals | Lost goals | Net |
|---|---:|---:|---:|---:|
| off |32|8|7|+1|
| v8 |32|3|2|+1|
| v9 |32|7|6|+1|
| v10 |32|2|3|-1|
| v11 |32|3|4|-1|

Versus default V10, gained: TS2:BAS-BO-CV-063, TS2:CH-CR-CV-007.
Versus default V10, lost: DV3:DV3-HO-VS-01, TS2:P2-L1-CRP-VAR-12, TS2:P2-L2-HO-FIX-19.

These are the same 32 outcome-enriched development scenarios, not a population sample or a formal safety guarantee. All reference outcomes are fresh recorded evaluations; no historical-only outcomes are substituted. No new episodes were run for this report.

[Separate validated pieces](report.md), [all per-case results](paired.csv), [combined counts and hashes](combined_summary.json), [table CSV](combined_table.csv).
