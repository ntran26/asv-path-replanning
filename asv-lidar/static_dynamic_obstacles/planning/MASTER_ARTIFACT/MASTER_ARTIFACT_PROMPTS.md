# Explainer artifact: edit instructions (kept current)

> **Kept current.** Last updated 2026-10-25 (comparators v2 done). This file holds the next edit instructions for the
> explainer artifact (https://claude.ai/artifact/S1Bm5Lf1199eXx2fC41hYF), each as a block to paste
> into the chat that holds the artifact. The blocks name planning files by path. That chat reads
> them from the synced project, so each push needs a Sync before the next block is sent. Only files
> outside `planning/` are attached. Blocks already applied move to the log at the end.

## 1. State of the artifact

The artifact was last checked on 2026-10-25, at version 1792859529-ce15 (13,104 lines). None of
the blocks below has been applied yet.

- **Built:** Sections 0–8, 10 and 12. Sections 9 (safety layer and evaluation design) and 11
  (field trials) are titled placeholders.
- **Section 10:** complete, but its embedded data dates from 2026-10-22:
  - TQC over seeds 0–1;
  - the deployed policy pending;
  - LOS-PID + DWA at v1;
  - the encounter-specific VO pending.

  Block B brings in the data of 2026-10-25: all twelve runs, the deployed policy (SAC seed 0),
  both comparators at v2, and the reward audit.
- **Section 4:** presents the encounter-specific VO with the comparators v1 values. Block C moves
  it to v2: course-change weight 0.8, and the passing side judged on the engaged encounter. With
  0.8, the worked cost example no longer favors the turn.
- **Sections 0 and 1** still frame the paper as a comparison of four learners (block D).
- **The reward scale audit (Table R7)** has no subsection yet (block E).

## 2. Next: blocks B, C, D and E, in that order

They need the comparators v2 commit synced (it comes after 576d4ed4). None needs attachments.

### Block B: the results data of 2026-10-25, and two comparators in Section 10

The current page was rendered locally with this data swapped in.
- The charts, the status board, the still-to-come list and every computed sentence handle the
  second comparator correctly.
- All 1,080 pooled-table cells match `results/baseline_v3/paper_table.md`.
- The console is clean.
- What remains is hand-written wording that assumes one comparator.

```
Update the Master Artifact in place with targeted edits only (no full reprint).

1. Data. Replace the contents of the <script type="application/json" id="results-data"> block with the project's planning/master_artifact_results.json, verbatim. The file now holds:
   - all twelve runs;
   - the deployed policy (SAC seed 0);
   - both classical comparators at comparators v2: LOS-PID + DWA, and the encounter-specific VO, which now has test-set data;
   - the reward scale audit that 10.12 will use.
   planning/MASTER_ARTIFACT_RESULTS.md has the same numbers as text.

2. Section 10 now has two classical comparators with data. The charts and the computed sentences already handle both. These hand-written phrases still assume one:
   a. "In this section": "what the four learners and the classical comparator achieve" → "what the four learners and the two classical comparators achieve".
   b. 10.1 callout, the comparator item. It renders as "The classical comparator. LOS-PID + DWA (v2) and Encounter-specific VO (v2) is deterministic: one run, so it has no seed spread and its result carries no seed range (Section 4.6)." With more than one comparator, make it plural: "The classical comparators. LOS-PID + DWA (v2) and the encounter-specific VO (v2) are deterministic: one run each, so neither has a seed spread or a seed range", and cite both sections, XS(4, 'dwa') and XS(4, 'esvo').
   c. 10.2: "the classical comparator, being deterministic, gets a single bar" → "the classical comparators, being deterministic, get a single bar each". In Figure 10.1's caption: "Classical comparator: a hatched bar" → "Classical comparators: a hatched bar each". Show "A dashed row is still to run." only while some comparator has no test data.
   d. The comparisons in 10.2 and 10.8, and the first takeaway, name only LOS-PID + DWA. Extend each to every comparator with data, computed the same way. On the present data:
      - every SAC, TQC and PPO seed, and two of the three RecurrentPPO seeds, succeed more often than the encounter-specific VO (v2);
      - the learned policies finish sooner and faster, and further from the reference path, than both comparators.
      The first takeaway gives both comparators' success.
   e. In running text, write the encounter-specific VO's name in lower case where it does not start a sentence ("the encounter-specific VO (v2)"). Keep the capital in labels, legends and table headers.
   f. The status board's card for the encounter-specific VO has no description line. Give it one from its status string, as the card for LOS-PID + DWA has: "Tuned on the 120 decoupled validation episodes; passing side judged on the engaged encounter."

3. Re-read every interpretive sentence in Section 10 against planning/MASTER_ARTIFACT_RESULTS.md, and correct any that no longer hold. Then confirm that:
   - the status board, the seed counts and the deployed-policy line reflect the data;
   - the still-to-come list no longer has the comparator items;
   - the console is clean.
```

### Block C: Section 4 at comparators v2

```
Update Section 4 of the Master Artifact to comparators v2, with targeted edits only (no full reprint). The values are in the project's planning/MASTER_ARTIFACT_FACTS.md: section 2 (the tuning paragraph) and section 2.4 (items 5 and 7), updated 2026-10-25. Two things changed from v1: the encounter-specific velocity obstacle's course-change weight, from 0.3 to 0.8, and its passing side, now judged on the engaged encounter. Everything else kept its v1 value.

1. Replace the paragraph that begins "The values below are those of comparators v1, the method’s untuned defaults." and ends "Its tuned values are [TBD: comparators v2]." The new paragraph says:
   - the values are those of comparators v2, tuned by coordinate descent on the 120 decoupled validation episodes (XS(4, 'fair'));
   - only the course-change weight moved, from its default 0.3 to 0.8; every other value kept its default.
2. Step 5 (the passing side). Replace "A candidate whose own closest approach is beyond 2.5 m is exempt, because it opens the pass by itself." with the v2 rule. While the formulation's encounter assessment reports the encounter engaged (XS(2, 'engage')), the side binds every candidate that closes on the target; a candidate that does not close on it, such as slowing behind a slower vessel, has no pass to keep. Add one sentence on why. Judged for each candidate on its own, a quick turn to port could open a head-on pass by itself and so escape the rule. Example: a head-on target 0.3 m to starboard of the reciprocal track at 10 m; judged per candidate the comparator turned to port, judged on the encounter it turns to starboard.
3. Step 7 (the cost):
   - In the classical-methods module, set K.J.change from 0.3 to 0.8, with the comment "encounter-specific VO, comparators v2 (facts 2.4)".
   - In the cost equation and its "In words", change the switching weight from 0.3 to 0.8.
   - The worked example. With 0.8, the 30° turn at full speed costs 30/180 + 0.8 × 30/180 = 0.300, more than holding course at half speed, 0.5 × 0.5 = 0.250. Rewrite the conclusion from the computed values (compare J1 and J2 in code). Its new content: when nothing else differs, the comparator now prefers slowing down to a fresh 30° turn, which is what the higher switching weight buys; a turn that is kept pays the switching cost only at the decision that starts it, while the course term keeps charging until the vessel is back on its line-of-sight course.
   - Keep the c_min and encounter-domain sentences, recomputing any comparison in them.
4. Check the pipeline figure and the comparison-table row for the encounter-specific VO, and replace any v1 statement with its v2 equivalent.
5. In the VO widget's subsection, delete the sentence "The same view for the encounter-specific velocity obstacle, with its domain and passing sides, is still to come [TBD: after comparators v2]." No such widget is planned.

Then check:
- no "[TBD: comparators v2]" and no v1 value remain in Section 4;
- every value matches facts 2.4;
- the VO widget is unaffected (it runs the reactive target's rule, whose weights did not change);
- the console is clean.
```

### Block D: Sections 0 and 1, the framing (send after block C)

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

### Block E: Section 10.12, how the deployed policy uses the reward (send last)

The results files carry the reward scale audit (Table R7) of the deployed policy from 2026-10-25:
the JSON's `reward_audit` block, and section 17 of `MASTER_ARTIFACT_RESULTS.md`. Block B has
already embedded this data, so step 1 below re-embeds the same file and changes nothing. No
attachments.

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

## 3. Log of applied blocks

| Date | Block | Artifact version after it | Check |
|---|---|---|---|
| 2026-10-22/23 | Section 10 structure, data and 10.1–10.3; Step 3 text replacements | — | checked 2026-10-23 at 1792735046-6b37 |
| 2026-10-23 | Section 10.4–10.8 | 1792735046-6b37 | numbers and sentences match the results files |
| 2026-10-23 | Section 10.9–10.12, takeaways, nomenclature | 1792753290-a232 | matches; three small fixes passed to the next block |
| 2026-10-24 | Width sweep and perception study removed; three small fixes | 1792804596-231b | applied in full |
| 2026-10-24 | Section 4 rewritten for the encounter-specific VO (done without a block) | 1792804596-231b | values match facts 2.3–2.4; head-on test wrong; two flags open (block A) |
| 2026-10-24 | Block A: Section 4 head-on test (\|β\| ≤ 22.5°), domain depth, reactive-target settings | 1792859529-ce15 | applied in full; Figure 4.6 caption and the opening of 4.9 also aligned |
