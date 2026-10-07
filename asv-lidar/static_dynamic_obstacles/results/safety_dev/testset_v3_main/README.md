# Primary safety benchmark: test set v3

The main evaluation target is now the complete frozen 1,000-case test set v3. [Protocol and commands](../../../planning/SAFETY_EVALUATION_PROTOCOL.md).

- [Frozen selection](selection.json): all 1,000 cases with explicit seeds and geometry digests.
- [Inventory](inventory.json): 664 decoupled cases, 336 coupled cases; 755 exact v2 geometry/seed pairs and 245 new or changed cases.
- Historical SAC baseline: **853/1,000 goals**, with 71 obstacle, 66 target and 10 boundary contacts. Its provenance is insufficient to call it a fresh current-runtime baseline.
- Next primary comparison: fresh SAC OFF versus frozen safety filter V16, matched episode by episode.

The benchmark switch itself ran no episodes. A later frozen quick check is complete:54 runs on18 cases (OFF/V16/V19). SAC reaches14/18 goals; V16 and V19 each reach15/18, rescuing one failure and preserving all14 SAC successes. V19 exactly matches V16 commands on all18 cases and is not promoted. This is18/1000 scenarios; no complete-v3 safety result is claimed. Existing v2/DV3 results remain separate.

Readiness checks passed: all 1,000 cached identities, byte-identical serialized scenes for the earlier 32-case cohort, and 99 focused loader/routing/report tests. Constants and checkpoint hashes are unchanged. [Verification](verification.json).

[Quick validated report](quick_v19_report/report.md), [command audit](quick_v19_trace_audit/REPORT.md), [subset and ranking](quick_v18_case_list.json), [development and cleanup report](../v18_development/report.md). No candidate was tuned on these primary outcomes.
