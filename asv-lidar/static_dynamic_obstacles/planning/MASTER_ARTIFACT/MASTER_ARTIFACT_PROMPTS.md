# Explainer artifact: edit instructions (kept current)

> **Kept current.** Last updated 2026-10-25 (TQC seed 2 done). This file holds the next edit instructions for the
> explainer artifact (https://claude.ai/artifact/S1Bm5Lf1199eXx2fC41hYF), each as a block to paste
> into the chat that holds the artifact. The blocks name planning files by path. That chat reads
> them from the synced project, so each push needs a Sync before the next block is sent. Only files
> outside `planning/` are attached. Blocks already applied move to the log at the end.

## 1. State of the artifact

The artifact was last checked on 2026-10-24, at version 1792859529-ce15 (13,104 lines).

- **Built:** Sections 0–8, 10 and 12. Sections 9 (safety layer and evaluation design) and 11
  (field trials) are titled placeholders.
- **Section 10:** complete, but its embedded data is that of 2026-10-22 (TQC over seeds 0–1, the
  deployed policy pending). Block B brings in the twelve-run data of 2026-10-25.
- **Section 4:** presents the encounter-specific VO with the comparators v1 values, and the
  reactive target's rule (COLREGs-VO) with the |α|, |β| head-on test and the target's six
  departures. The only flags left wait for comparators v2.
- **Sections 0 and 1** still frame the paper as a comparison of four learners; block D aligns them
  with the framing decision of 2026-10-10 (a formulation and a deployed policy).
- **The width sweep and the perception study** are gone from every section (2026-10-24).
- **Nomenclature:** ℓ and β each have two entries now, each with its own meaning and section:
  - ℓ: Section 3's rudder lever arm and Section 4's normalized domain distance;
  - β: Section 4's bearing in the COLREGs velocity obstacle and Section 8's bearing of the null
    target.

## 2. Next: block B, then D, then E

### Block B: the results data refresh (send now)

When: now (results of 2026-10-25, all twelve runs), and again after comparators v2.

The 2026-10-25 data was checked before sending:
- The current artifact was rendered locally with the new data swapped in.
- Every interpretive sentence of Section 10 still holds with TQC seed 2 included.
- The status board reads "Decided: SAC seed 0", and every "two of three seeds" mention is gone.
- The console is clean.
- The pooled table matches `results/baseline_v3/paper_table.md` in every cell (the generator now
  keeps nine decimals).

Needs: the results commit synced into the project. No attachments.

```
Replace the contents of the <script type="application/json" id="results-data"> block in the Master Artifact with the project's planning/master_artifact_results.json, verbatim. Change nothing else in the code. Then re-read every interpretive sentence in Section 10 against planning/MASTER_ARTIFACT_RESULTS.md, and correct any that no longer hold. Confirm that the status board, the pending list, the seed counts ("two of three seeds so far") and the deployed-policy line now reflect the data.
```

### Block D: Sections 0 and 1, the framing (send after block B)

Needs: the same commit synced. No attachments.

```
Update Sections 0 and 1 of the Master Artifact in place, with targeted edits only (no full reprint), to the paper's framing: the paper develops one formulation and deploys one policy trained on it; it is not a learner-comparison study. The wording to follow is in the project's planning/PAPER/PAPER3_SECTIONS1-2_REVISION_BRIEF.md, section 1.1 (framing) and section 1.3 (objective, central questions, contributions 1–5). Keep the explainer's plain-language style, and follow the brief's wording for every claim. Results stay in Section 10 (planning/MASTER_ARTIFACT/MASTER_ARTIFACT_BRIEF.md, rule 7), so Sections 0 and 1 say how the deployed policy is chosen and point to Section 10 for which one it is.

1. Section 0, the "In this section" summary: "and four learners are compared on one fixed formulation" → "and one formulation is developed, four learners are trained on it, and the best is deployed".
2. Section 0, the opening paragraph: "Four deep reinforcement learning algorithms are trained and compared on one fixed problem formulation, against classical controllers." → "One problem formulation is developed and frozen. Four deep reinforcement learning algorithms are trained on it, the better of SAC and TQC on the test set becomes the deployed policy, and the other learners and two classical controllers serve as its baselines (Section 10)."
3. Section 1, the objective paragraph: rewrite it after the brief's objective paragraph (section 1.3). Develop the formulation; deploy a policy trained on it on the model-scale vessel; train the four learners on the same frozen formulation; take the better of SAC and TQC on the held-out test set as the deployed policy; the remaining learners and two classical methods serve as baselines.
4. Section 1, the central questions: replace the four with the brief's three:
   - does the policy complete the task with encounter-consistent behavior, in channels and among static obstacles too;
   - which learner should be deployed, and which learner differences hold across seeds and target behaviors;
   - how the deployed policy performs in physical basin trials.
   Update the section's summary line ("four central questions and four contributions") to match.
5. Section 1, the contributions, after the brief's contributions 1–5. Keep the existing names where the brief keeps them.
   - Contribution 1: "target tracks from onboard range sensing".
   - Contribution 2: the admissibility and stopping tests, and width thresholds stated in ship breadths. Width is not a legal definition.
   - Contribution 3: "A deployed policy, chosen and benchmarked fairly", in place of "A controlled four-learner comparison with classical references". Keep the existing citations for the encounter-specific velocity obstacle (thyri2022b) and LOS-PID with DWA (fox1997).
   - Contribution 4: an evaluation design separating task completion, safety and robustness to target behavior.
   - New contribution 5: physical encounter trials of the deployed policy (Section 11).
   Change "Within this scope, the methodological contributions are as follows." to "Within this scope, the contributions are as follows." if that sentence is present.
6. Section 1, the takeaways: "The study compares four learners on one fixed formulation and reports results by encounter type, by decoupled or coupled layout, by basin or channel and by target behavior (Section 10)." → "The study develops one formulation, deploys the better of SAC and TQC, and benchmarks it against the other learners and two classical methods by encounter type, by decoupled or coupled layout, by basin or channel and by target behavior (Section 10)."
7. Check every other section for sentences that present comparing learners as the aim, and change only those, the same way.
8. Keep "LiDAR" out of contribution names and framing sentences; it stays where the text describes the implementation (the brief, section 1.2).

Then check that the page names no deployed seed outside Section 10, that the rail and pager still work, and that the console is clean.
```

### Block E: Section 10.12, how the deployed policy uses the reward (send after block D)

The results files carry the reward scale audit (Table R7) of the deployed policy from 2026-10-25:
the JSON's `reward_audit` block, and section 17 of `MASTER_ARTIFACT_RESULTS.md`. Nothing else in
them changed since block B. Needs the commit that adds them, synced. No attachments.

```
Update the Master Artifact in place with targeted edits only (no full reprint). This adds subsection 10.12, "How the deployed policy uses the reward" (the paper's Table R7), as specified under "10.12 How the deployed policy uses the reward" in the project's planning/MASTER_ARTIFACT/MASTER_ARTIFACT_BRIEF.md. The present "Still to come" becomes 10.13.

1. Data. Replace the contents of the <script type="application/json" id="results-data"> block with the project's planning/master_artifact_results.json, verbatim. It now carries a reward_audit block; nothing else changed since the last data refresh. planning/MASTER_ARTIFACT_RESULTS.md section 17 has the same numbers as text.
2. New subsection 10.12, with H2 id 'reward', after 10.11 (Training cost) and before Still to come. Put it in results3.js, or in a new module after it, reusing the shared drawing kit.
   - An opening paragraph in the explainer's style. The weights of Section 7 cap how much each term can add or cost in one step; this audit measures how much each actually does for the deployed policy, over the same 1,000 test episodes as Table 10.1. Read the run, the policy and the two checks from reward_audit; name the deployed policy from deployment.deployed.
   - Figure "designed against realized", as the brief specifies, with a segmented control over the five scopes (the keys of reward_audit.scopes).
   - Figure "where the return comes from", as the brief specifies, under the same control.
   - The text items the brief lists. Render every number from reward_audit, and generate each sentence so it stays true if the data change: build the two orders by sorting, and compute every comparison.
   - A table view for each chart, an aria-label summarizing each, light and dark mode, 375 px, keyboard use.
   - If reward_audit is absent or null, show a short pending note in place of the figures.
3. Renumbering. Still to come becomes 10.13. Update the rail and any cross-reference that meant it.
4. Section 7: "With bounded terms the weights say exactly how the terms compare." → "With bounded terms each weight says exactly how large its term can get; how large each gets under the deployed policy is in " + XS(10, 'reward') + "."
5. Nomenclature: register "weighted episode integral" (a term's weighted reward summed over an episode) and "realized per-step magnitude" (its mean absolute value over the steps on which it is non-zero), with Section 10 as first use.

Then check:
- every number against planning/MASTER_ARTIFACT_RESULTS.md section 17;
- the new figures are numbered after Figure 10.11;
- the rail and pager still work;
- the console is clean.
```

## 3. Waiting: block C, Section 4 at comparators v2 (draft)

When: after comparators v2 has run and `MASTER_ARTIFACT_FACTS.md` section 2.4 holds its tuned
values. This draft is finalized then.

```
Update Section 4 of the Master Artifact to comparators v2, with targeted edits only. Take the tuned values of the encounter-specific velocity obstacle from the project's planning/MASTER_ARTIFACT_FACTS.md, section 2.4, and replace every comparators v1 value in its subsection, its pipeline figure, its cost equation and example, and its comparison-table row. Remove "[TBD: comparators v2]". Describe the passing-side exemption as judged once for the encounter. Change "The values below are those of comparators v1" to state that they are the comparators v2 values tuned on the 120 decoupled validation episodes. If LOS-PID + DWA's settings changed under v2 (facts section 2.2), update them the same way. Check that no v1 value remains and that the console is clean.
```

## 4. Log of applied blocks

| Date | Block | Artifact version after it | Check |
|---|---|---|---|
| 2026-10-22/23 | Section 10 structure, data and 10.1–10.3; Step 3 text replacements | — | checked 2026-10-23 at 1792735046-6b37 |
| 2026-10-23 | Section 10.4–10.8 | 1792735046-6b37 | numbers and sentences match the results files |
| 2026-10-23 | Section 10.9–10.12, takeaways, nomenclature | 1792753290-a232 | matches; three small fixes passed to the next block |
| 2026-10-24 | Width sweep and perception study removed; three small fixes | 1792804596-231b | applied in full |
| 2026-10-24 | Section 4 rewritten for the encounter-specific VO (done without a block) | 1792804596-231b | values match facts 2.3–2.4; head-on test wrong; two flags open (block A) |
| 2026-10-24 | Block A: Section 4 head-on test (\|β\| ≤ 22.5°), domain depth, reactive-target settings | 1792859529-ce15 | applied in full; Figure 4.6 caption and the opening of 4.9 also aligned |
