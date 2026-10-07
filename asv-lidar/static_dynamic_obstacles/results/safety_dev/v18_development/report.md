# Safety improvement and storage cleanup: V18/V19

**Neither new candidate improves the measured goal count over V16. Keep V16 as the reference; V18 and V19 remain experimental.** All90 new episode runs are complete, with no retries or queued evaluations. The requested100% safety and preservation of every SAC success remain unmet.

## Primary evaluation: quick test-set-v3 subset

The candidate was frozen before evaluation. Eighteen cases were selected using source/encounter groups, distinct target variants and a fixed identity hash, without consulting outcomes: nine decoupled and nine coupled cases. They share no geometry/seed pair with the earlier72-case development cohort; ten share geometry/seed with test set v2, so the subset is not wholly unseen. It is18/1000 cases, not a full-set estimate. No changes were tuned from these primary outcomes.

| Controller | Goals | Target contact | Obstacle contact | Boundary contact | Timeout |
|---|---:|---:|---:|---:|---:|
| Fresh SAC baseline3, 3M, safety OFF |14/18|1|1|2|0|
| Fresh V16 |15/18|1|0|2|0|
| V19 |15/18|1|0|2|0|

Both filters rescue `FS-CRP-VAR-023` and preserve all14 successful SAC cases. Neither resolves `BAS-BO-RE-082`, `P2-L2-CRS-FIX-17` or `FS-BO-FIX-019`. In the latter, SAC hits an obstacle and the filters hit the target; that contact-type change is not a rescue. V19 gains zero goals and loses zero goals relative to V16. Its complete issued-command sequences match V16 exactly in all18 cases, despite11 point-transfer activations across four cases (29 point-decision transfers). These finite results do not establish95?100% success or a safety guarantee.

[Validated primary report](../testset_v3_main/quick_v19_report/report.md), [paired records](../testset_v3_main/quick_v19_report/paired.csv), [frozen subset](../testset_v3_main/quick_v18_case_list.json). The original full1000-case selection and cache are unchanged. The historical853/1000 SAC result remains separate from this fresh matched check.

## Targeted development probe

Twelve cases intentionally include all six previously broken SAC successes, the missing-orientation failure, earlier geometry rescues, an active rescue and two no-intervention success controls. This enriched cohort is not a population sample. Each of V16, V18 and V19 reaches5/12 goals, with zero new goal gains or losses relative to V16.

Fresh V16 reproduces all12 previous exact command sequences and outcomes. V18's heading memory activates on four decisions across two cases but changes no executed command. In fresh FIX05 its only activation is the terminal decision34. V19 transfers current returns on four decisions across three cases. It changes the command sequence only in DV3-CRS-CV-04, starting at13; the episode changes from boundary contact to obstacle contact. The six previously broken SAC successes remain broken.

[Validated development report](validated_development_report/report.md), [command-level audit](probe12_trace_audit/REPORT.md). The SAC references here are earlier matched pilot records for eight cases and historical saved records for four, explicitly labelled in the validator. Development results are kept separate from primary-v3 results.

## Implemented methods and citations

- **V18: bounded same-source hull-heading memory.** When a fresh observed target loses its fitted hull axis, retain a recently validated axis for the existing age budget instead of letting noisy velocity direction rotate its rectangle. Identity, freshness, fit-validity and expiry gates apply; persistent views and current motion-axis corrections retain priority. Related modelling inspiration: [Granstrom, Baum and Reuter, Extended Object Tracking (2016)](https://arxiv.org/abs/1604.00970). This deterministic hold is an engineering adaptation, not their probabilistic estimator. [Plan](../../../planning/archive/safety/SAFETY_LAYER_V18_PLAN.md), [saved applicability audit](heading_memory_saved/REPORT.md).
- **V19: current-return ownership transfer.** Stop representing qualified moving-target returns as both a moving hull and stationary points. Require an existing fresh motion-admission certificate, a still-published target and exact point membership in the current scan-memory batch; retain old, ambiguous and unrelated points. Related inspiration: [Nuss et al., Dynamic Occupancy Grid Maps (2016)](https://arxiv.org/abs/1605.02406), together with Granstrom's measurement-association modelling. This is not the PHD/MIB algorithm or a proof of correct ownership. [Plan](../../../planning/archive/safety/SAFETY_LAYER_V19_PLAN.md).

Both are independent options atop V16. Neither uses simulator truth, scenario identity, future policy calls or changed safety margins. Both have explicit disabled options preserving V16's underlying perception. Native dispatch and suite options now accept18/19, with defaults unchanged.

## Rejected dynamics adaptation

Two causal saved-data experiments selected among the same30 whole physical parameter vectors using only the first ten past measured transitions, then froze before held-forward validation. One minimized repeated one-step errors; the other minimized a continuous training forecast. They reuse prediction-error estimation ideas from [Ljung (2002)](https://doi.org/10.1007/BF01211648), without claiming calibrated parameter identification or safety.

The one-step rule improved many short forecasts but worsened full-eight-second V15 heading error in12/20 cases. The continuous rule still worsened10/20 and raised mean heading and position errors, despite better results on a smaller V16 probe. Neither enters the controller. Stop this bank family without further weight/window tuning. [One-step bank audit](../v10_iterations/audits/bootstrap_model_bank_cohort/REPORT.md), [continuous bank audit](../v10_iterations/audits/bootstrap_continuous_bank_cohort/REPORT.md). These audits ran no episodes.

## Disk cleanup and verification

Transparently compressed585 completed trace files across24 runs, reclaiming **974,323,274 allocated bytes (0.907GiB)**. Every before/after SHA-256 matches. The first batch covered495 old traces; the second compressed all90 new traces. Paths, result records, source archives and scenario caches remain usable. No repository files were deleted. The small bytecode-deletion preflight stopped on a cloud reparse attribute; retaining those28KB avoids touching that file representation. Old controller versions still form the inheritance chain, and deleting their source would break current candidates.

The allocation saving is measured per compressed file; new reports/archives and unrelated disk activity also affect whole-drive free space. The new compact traces total51,962,805 logical bytes, avoiding another large full-snapshot collection.

[Initial maintenance audit](../maintenance/ntfs_compact_20261003T111517Z/README.md), [new-trace maintenance summary](../maintenance/ntfs_compact_20261003T115342Z/summary.json), [final verification](final_verification.json).

Checks passed:26 V18 tests,33 V19 tests,32 native-dispatch checks and15 maintenance checks. Independent source reviews found no implementation blocker. Strict result validators checked every attempt/result/trace, scenario identity, source archive and completion token; no filter-error fallback occurred. Constants and SAC checkpoint/config hashes are unchanged. Existing line endings were retained. No edits were made to the separate Paper2 project, and no other job was signalled or stopped. At most two single-threaded evaluation workers ran.

The remaining evidence points to coupled target-centre/orientation/motion errors and own-vessel forecast mismatch. Correcting a late heading fallback or a few duplicated points alone has not solved those failures. A more favourable predicted margin must not be treated as a successful safety improvement without matched episode gains and preserved policy successes.

[Primary command-level audit](../testset_v3_main/quick_v19_trace_audit/REPORT.md) confirms the lack of incremental V19 behavior in this subset.
