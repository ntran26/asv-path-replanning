# Explainer artifact: edit instructions (kept current)

> **Kept current.** Last updated 2026-11-05 (all blocks applied). This file holds the next edit instructions for the
> explainer artifact (https://claude.ai/artifact/S1Bm5Lf1199eXx2fC41hYF), each as a block to paste
> into the chat that holds the artifact. The blocks name planning files by path. That chat reads
> them from the synced project, so each push needs a Sync before the next block is sent. Only files
> outside `planning/` are attached. Blocks already applied move to the log at the end.

## 1. State of the artifact

The artifact was last checked on 2026-11-05, at version 1793854625-a135 (14,161 lines), rendered
locally.

- **Built:** Sections 0–10 and 12. Section 11 (field trials) is the only titled placeholder.
- **Section 10:** complete (10.1–10.14) on the results data of 2026-10-26:
  - all twelve runs, the deployed policy (SAC seed 0) and both comparators at v2;
  - 10.12, the reward scale audit;
  - 10.13, the safety layer on, presenting the reported layer only.
- **Section 9:** the reported safety layer and the evaluation design, with no result values.
- **Section 4:** at comparators v2, with no flags left.
- **Sections 0 and 1:** framed as a formulation and a deployed policy.
- **Still to come**, read from the data: the COLREGs trajectory snapshots for the deployed policy,
  and the field trials (Section 11, waiting for the field-trial plan).

## 2. Next

No block is pending. Next blocks are written when one of these changes:
- **the results files are regenerated** (refresh the data block, then re-check the interpretive
  sentences);
- **the snapshots or the field-trial plan exist** (fill 10.14's items and Section 11).

## 3. Log of applied blocks

| Date | Block | Artifact version after it | Check |
|---|---|---|---|
| 2026-10-22/23 | Section 10 structure, data and 10.1–10.3; Step 3 text replacements | — | checked 2026-10-23 at 1792735046-6b37 |
| 2026-10-23 | Section 10.4–10.8 | 1792735046-6b37 | numbers and sentences match the results files |
| 2026-10-23 | Section 10.9–10.12, takeaways, nomenclature | 1792753290-a232 | matches; three small fixes passed to the next block |
| 2026-10-24 | Width sweep and perception study removed; three small fixes | 1792804596-231b | applied in full |
| 2026-10-24 | Section 4 rewritten for the encounter-specific VO (done without a block) | 1792804596-231b | values match facts 2.3–2.4; head-on test wrong; two flags open (block A) |
| 2026-10-24 | Block A: Section 4 head-on test (\|β\| ≤ 22.5°), domain depth, reactive-target settings | 1792859529-ce15 | applied in full; Figure 4.6 caption and the opening of 4.9 also aligned |
| 2026-11-05 | Block B: results data of 2026-10-26 (all twelve runs, deployed policy, both comparators at v2, reward audit, safety layer); two-comparator wording | 1793818813-fe0e | data verbatim; every changed sentence matches the results file; console clean; no overflow at 375 px |
| 2026-11-05 | Blocks C–G: Section 4 at comparators v2; Sections 0–1 framing; 10.12 reward audit; Section 9 built; 10.13 safety layer on | 1793854625-a135 | applied in full; every number matches the results file and the facts; no flags left in Section 4; console clean |
