# Explainer artifact: edit instructions (kept current)

> **Kept current.** Last updated 2026-10-24 (after block A). This file holds the next edit instructions for the
> explainer artifact (https://claude.ai/artifact/S1Bm5Lf1199eXx2fC41hYF), each as a block to paste
> into the chat that holds the artifact. The blocks name planning files by path. That chat reads
> them from the synced project, so each push needs a Sync before the next block is sent. Only files
> outside `planning/` are attached. Blocks already applied move to the log at the end.

## 1. State of the artifact

The artifact was last checked on 2026-10-24, at version 1792859529-ce15 (13,104 lines).

- **Built:** Sections 0–8, 10 and 12. Sections 9 (safety layer and evaluation design) and 11
  (field trials) are titled placeholders.
- **Section 10:** complete. Every number is rendered from the embedded results data of 2026-10-22,
  with TQC over seeds 0–1 and the encounter-specific VO and the deployed policy shown as pending.
- **Section 4:** presents the encounter-specific VO with the comparators v1 values, and the
  reactive target's rule (COLREGs-VO) with the |α|, |β| head-on test and the target's six
  departures. The only flags left wait for comparators v2.
- **The width sweep and the perception study** are gone from every section (2026-10-24).
- **Nomenclature:** ℓ and β each have two entries now, each with its own meaning and section:
  - ℓ: Section 3's rudder lever arm and Section 4's normalized domain distance;
  - β: Section 4's bearing in the COLREGs velocity obstacle and Section 8's bearing of the null
    target.

## 2. Next

Nothing is to be sent now. Block B waits for the training session's "tqcs2_bl3 run done"
(expected about 2026-10-25); block C waits for comparators v2.

## 3. Waiting: block B, the results data refresh

When: after the training session reports "tqcs2_bl3 run done", and again after comparators v2.

Before sending:
1. Regenerate with `python tools/diagnostics/artifact_results.py --table`. The generator now keeps six decimals, which removes the 0.001 double-rounding differences in the pooled table.
2. Check the new results files.
3. Commit, push and Sync.

No attachments.

```
Replace the contents of the <script type="application/json" id="results-data"> block in the Master Artifact with the project's planning/master_artifact_results.json, verbatim. Change nothing else in the code. Then re-read every interpretive sentence in Section 10 against planning/MASTER_ARTIFACT_RESULTS.md, and correct any that no longer hold. Confirm that the status board, the pending list, the seed counts ("two of three seeds so far") and the deployed-policy line now reflect the data.
```

## 4. Waiting: block C, Section 4 at comparators v2 (draft)

When: after comparators v2 has run and `MASTER_ARTIFACT_FACTS.md` section 2.4 holds its tuned
values. This draft is finalized then.

```
Update Section 4 of the Master Artifact to comparators v2, with targeted edits only. Take the tuned values of the encounter-specific velocity obstacle from the project's planning/MASTER_ARTIFACT_FACTS.md, section 2.4, and replace every comparators v1 value in its subsection, its pipeline figure, its cost equation and example, and its comparison-table row. Remove "[TBD: comparators v2]". Describe the passing-side exemption as judged once for the encounter. Change "The values below are those of comparators v1" to state that they are the comparators v2 values tuned on the 120 decoupled validation episodes. If LOS-PID + DWA's settings changed under v2 (facts section 2.2), update them the same way. Check that no v1 value remains and that the console is clean.
```

## 5. Log of applied blocks

| Date | Block | Artifact version after it | Check |
|---|---|---|---|
| 2026-10-22/23 | Section 10 structure, data and 10.1–10.3; Step 3 text replacements | — | checked 2026-10-23 at 1792735046-6b37 |
| 2026-10-23 | Section 10.4–10.8 | 1792735046-6b37 | numbers and sentences match the results files |
| 2026-10-23 | Section 10.9–10.12, takeaways, nomenclature | 1792753290-a232 | matches; three small fixes passed to the next block |
| 2026-10-24 | Width sweep and perception study removed; three small fixes | 1792804596-231b | applied in full |
| 2026-10-24 | Section 4 rewritten for the encounter-specific VO (done without a block) | 1792804596-231b | values match facts 2.3–2.4; head-on test wrong; two flags open (block A) |
| 2026-10-24 | Block A: Section 4 head-on test (\|β\| ≤ 22.5°), domain depth, reactive-target settings | 1792859529-ce15 | applied in full; Figure 4.6 caption and the opening of 4.9 also aligned |
