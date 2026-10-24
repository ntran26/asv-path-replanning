# Explainer artifact: edit instructions (kept current)

> **Kept current.** Last updated 2026-10-24. This file holds the next edit instructions for the
> explainer artifact (https://claude.ai/artifact/S1Bm5Lf1199eXx2fC41hYF), each as a block to paste
> into the chat that holds the artifact. The blocks name planning files by path. That chat reads
> them from the synced project, so each push needs a Sync before the next block is sent. Only files
> outside `planning/` are attached. Blocks already applied move to the log at the end.

## 1. State of the artifact

The artifact was last checked on 2026-10-24, at version 1792804596-231b (13,095 lines), rendered
locally.

- **Built:** Sections 0–8, 10 and 12. Sections 9 (safety layer and evaluation design) and 11
  (field trials) are titled placeholders.
- **Section 10:** complete. Every number is rendered from the embedded results data of 2026-10-22,
  with TQC over seeds 0–1 and the encounter-specific VO and the deployed policy shown as pending.
- **Section 4:** presents the encounter-specific VO with the comparators v1 values. Three points
  are open, and block A below fixes them:
  1. The head-on test of the reactive target's rule (COLREGs-VO) is wrong in the classification
     table, the bearing-figure note and the widget's `classify()`. It tests the headings instead of
     |β| ≤ 22.5°.
  2. `[TBD: how the depth is measured]` is open; `MASTER_ARTIFACT_FACTS.md` section 2.4 item 4 has
     the measure.
  3. `[VERIFY: reactive-target settings]` is open; `MASTER_ARTIFACT_FACTS.md` section 2.3 item 9
     has the settings.
- **The width sweep and the perception study** are gone from every section (2026-10-24).

## 2. Next: block A, Section 4 (the head-on test and the two open flags)

Needs: commit ce6abcbd synced into the project. No attachments.

```
Update the Master Artifact in place with targeted edits only (no full reprint). These edits complete the Section 4 update: they correct the head-on test of the COLREGs velocity obstacle and resolve the two open flags. The project's planning/MASTER_ARTIFACT_FACTS.md was updated on 2026-10-24 from the code: section 2.3, items 4 and 9, and section 2.4, item 4. Inside JS strings, write apostrophes as ’, as elsewhere in the source.

1. The head-on test of the COLREGs velocity obstacle (facts 2.3, item 4). The rule calls an encounter head-on when |α| ≤ 22.5° and |β| ≤ 22.5°. Here β is the own ship's bearing seen from the target, measured from the target's heading, so each vessel lies within 22.5° of the other's bow. The rule does not test the headings.
   a. In classify() (the classical-methods module): replace "Math.abs(alpha) <= 22.5 && recip <= 22.5" with "Math.abs(alpha) <= 22.5 && Math.abs(phi) <= 22.5" (phi is β), and delete the now unused recip.
   b. In the classification table, Head-on row: "and headings within 22.5° of reciprocal" → "and the own ship within 22.5° of the target’s bow, |β| ≤ 22.5°", with the inequality written with M() like the α conditions. In the Overtaking row, after "own ship at least 112.5° off the target’s bow", add ", |β| ≥ 112.5°", also with M().
   c. In the note on the bearing figure: "Head-on also needs the headings" / "within 22.5° of reciprocal." → "Head-on also needs the own ship" / "within 22.5° of the target’s bow."
   d. In the paragraph "Classification with open-water roles": "from the target’s relative bearing α, measured here from −180° to +180° and positive to starboard, and from the two speeds" → "from the target’s relative bearing α, measured here from −180° to +180° and positive to starboard; from β, the own ship’s bearing seen from the target and measured from the target’s heading, in the same range; and from the two speeds". Register β in the nomenclature the way α is.
   Then check that each widget preset shows the class the corrected test gives, and that a near-reciprocal head-on still classifies as head-on.

2. The depth of an encounter-domain intrusion (facts 2.4, item 4). In step 4 of the encounter-specific velocity obstacle, replace "along the candidate’s prediction [TBD: how the depth is measured]." with:
   "at the deepest point of the candidate’s prediction: P_dom = max(0, 1 − ℓ), where ℓ = √((x/a_x)² + (y/a_y)²) is the target center’s distance from the own ship in domain radii. Here x is measured ahead and y to starboard, and a_x and a_y are the semi-axes in the table below. P_dom is 0 on the domain’s edge and 1 at the own ship’s reference point."
   Write both formulas with M(), and register ℓ, a_x and a_y with this subsection's symbols. In the cost equation's symbol table, the meaning of P_dom becomes "how deeply the target comes inside the encounter domain at the deepest point of the candidate’s prediction: 0 on its edge, 1 at the own ship’s reference point".

3. The reactive target's settings (facts 2.3, item 9). Delete the sentence "Whether the reactive targets use every setting below, their speed candidates included, is still to be checked [VERIFY: reactive-target settings]." After the paragraph "Cost and fallback", add a paragraph headed "As the reactive target runs it". Its text: every 0.5 s the target applies the rule with the own ship as its only traffic, with six departures from the settings above:
   - it changes course only: its one candidate speed is its own speed, so it never slows;
   - its preferred course is its initial course, to which it returns once the pass is clear;
   - the starboard constraint is judged once for the encounter, not for each candidate, and holds from the moment the encounter is live until the target stops closing;
   - its hard clearance is 0.75 m between hulls instead of 0.20 m (with 0.20 m it cut back close ahead of the own ship);
   - it ignores the static panels, and at a fairway edge it stops maneuvering and keeps the heading the edge gave it;
   - it turns toward the chosen course at 8°/s, while it plans with the 6°/s turn model.
   At the end of the VO widget's footnote, add: "The widget runs the rule with the settings above, not with the reactive target’s departures."

Then check:
- Section 4 keeps only the flags that wait for comparators v2 ([TBD: comparators v2] and [TBD: after comparators v2]);
- the widget still works by mouse, touch and keyboard;
- light and dark mode and 375 px still display correctly;
- the console is clean.
```

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
