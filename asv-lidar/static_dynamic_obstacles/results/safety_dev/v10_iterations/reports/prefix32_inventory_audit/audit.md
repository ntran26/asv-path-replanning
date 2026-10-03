# Prefix32 source-inventory audit

The two 16-case selections are disjoint and exactly cover the original 32 pilot identities. Every saved seed and scenario digest matches the canonical inventory.

All 99 shared archived source files are byte-identical. The later archive adds `src/safety_v14.py` and `src/dev_set_v4.py`; the inventories are explicitly different.

No common archived source contains a literal reference to either added module or SafetyFilterV14. The added generator defines a separate 60-case DV4X extension. The archived runner loads prebuilt TS2/DV3 cache entries and validates their seeds/digests; it refuses missing caches instead of regenerating scenarios. The selected class and constructor options are unchanged.

This supports the added generator being unused in these cached-scenario runs. It is not a general dynamic-import proof. No episodes, resets, generation, or policy inference were run for this audit.

The strict cohort API remains unchanged and rejects the extra generator file. Report both completed pieces separately, then label any combined 32-case table as their explicitly disclosed sum. Source checks do not replace completion/result/trace checks.

[Exact hashes, shared-reference scan and archived cache-loader source](audit.json). Reproduce into a new directory or preserve/remove only these generated outputs before rerunning `build_audit.py`; the builder refuses overwrite.
