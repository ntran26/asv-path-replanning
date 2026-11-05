# Paper 3 draft: edit instructions (kept current)

> **Kept current.** Last updated 2026-11-04. This file holds the next edit instructions for the
> Paper 3 draft, each as a block to paste into the chat that writes the paper. The blocks name
> planning files by path; that chat reads them from the synced project, so each push needs a Sync
> before the next block is sent. Only files outside `planning/` are attached: the .docx draft,
> kept in `PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`. Blocks already applied move
> to the log at the end.

## 1. State of the draft

Current file: `Paper3_Sections1-4_with_placeholders_rev26Oct_tracked.docx` (from Downloads; to
be filed in `PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`), checked against the code
and the results on 2026-11-04.

- **Drafted:** Sections 1–4 (4.5 included), Appendices B–G (Algorithms A1–A3), Discussion 6.2–6.3.
- **Placeholders:** Sections 5, 6.1 and 7; Appendix A; Figures 4–6 (to insert).
- **Applied and checked:** everything through block F (log, entries 1–4).
- **Markers left:** [PENDING — field-trial design] (1.2.4, 1.3, 5, 5.3); [CHECK — wall-margin
  convention]; [CHECK — Figure 2 placeholder terminology]; [VERIFY] (1.2.5); two [CITE]; [TBC —
  replay logs] (5.2); [TBC — repository and DOI] and [CHECK — identification data] (Data
  Availability); comments 5–9 (comments 8 and 9: which parts of the layer run on the vessel).

## 2. Next: block G, three fixes from the 26 October log

**On hold (2026-11-04):** the draft waits until the field-trial plan is final; block G is kept
and goes out with the next round. Before that, the figures are reviewed: Figures 4 and 5 were
redrawn from the twelve runs on 2026-11-04 (outside the repository), and Figures 1–3 are not drawn.

Figure work of 2026-11-04/05, to be folded into block G once the figures are final:
- Figure 4 (`results/baseline_v3/figures_draft/fig_a_success_by_encounter`): categories "No encounter" (null and
  no target together) and "All", no n line; the placeholder text and caption follow those categories.
- Figure 5 (`fig_b_learning_curves`): the validation-set success rate, by the author's decision; the 4.1
  sentence and the placeholder say success rate, not the selection score (the best scores stay in Table 10).
- Figure 6 (`results/paper_snapshots/paper_snapshots`): panel (f) is now BAS-BO-CV-039 (basin, slanted leg, no
  panels, constant-velocity target; goal in 18.5 s; CPA 1.9 m at 9.5 s; speed 0.53-0.60 m/s from the onset of
  the encounter to the CPA; within 0.10 m of its track and 0.074 m/s of its speed until the target is within
  2.5 m, then up to 0.52 m beyond the path band; about 0.9 m/s once the target has passed). No successful
  being-overtaken episode holds within 0.15 m and 0.15 m/s through the whole pass (0 of 29), so the 4.2 text
  keeps the "until the target is within 2.5 m" wording. The caption's reactive-target sentence goes (the new
  target holds constant velocity). Panel titles read "Crossing from port" and "Crossing from starboard".
- Figure 1 (draft in the liaison scratchpad): the dashed outline is the own ship's domain, so the placeholder's
  "the target's asymmetric ship domain" becomes the own ship's. Figure 2 uses "required turn sense", which
  resolves [CHECK — Figure 2 placeholder terminology]. Figure 3 adds the ego and path branches.

Attachment: `Paper3_Sections1-4_with_placeholders_rev26Oct_tracked.docx`. Needs: the planning
folder synced at eb02b1fc or later (section 18 of `planning/MASTER_ARTIFACT_RESULTS.md`). Output:
`Paper3_Sections1-4_with_placeholders_rev04Nov_tracked.docx`.

Checked on 2026-11-04: section 18 gives 361 of 1,000 episodes with a change (0.361). The clause
"g_u = 1 while the runtime stop is latched" is in `src/reward/terms.py` (`estop_active`), set in
`src/env.py` from `self.estop`, the in-extremis stop of `src/emergency_stop.py`; the reported
layer's stop (`src/safety_estop.py`, `_stop`) sets `_v2_brake` and `emergency_stop_committed`,
not that latch, and the layer is off in training, so the clause applies to no reported run. The
reference is the standard tutorial on the method (the author checks it, as every reference). Also
verified, for the log's "not checkable" items: `planning_records/SAFETY_V21_PLAN.md` gives the
validation figures (section 15) and states that the layer-off run reproduces the test-set run of
the deployed policy in all 1,000 episodes (outcome and steps identical, returns equal to
rounding).

```
Apply the following changes to the attached Paper3_Sections1-4_with_placeholders_rev26Oct_tracked.docx as tracked changes dated 4 November 2026 (author "Claude"), keeping every earlier tracked change and comment, and save the result as Paper3_Sections1-4_with_placeholders_rev04Nov_tracked.docx. They resolve three flags of the 26 October log. Report each change in a short revision-log addendum.

G1. Table 13, last row: replace "0.36" with "0.361" (361 of the 1,000 episodes; planning/MASTER_ARTIFACT_RESULTS.md, section 18). In the Section 4.5 text, replace "and in 36 % of the episodes" with "and in 36.1 % of the episodes".

G2. Section 2.3.3, after Eq. (18): replace "Ueff is Uref except under a carve-out (Eq. 19), and gu = 1 while the runtime stop is latched." with "Ueff is Uref except under a carve-out (Eq. 19)." The clause refers to a stop latch that no reported run sets: the layer is off in training, and the stop of the layer of Section 2.3.4 does not use that latch.

G3. Section 2.3.4: cite the cross-entropy method at "a cross-entropy search", with the reference "de Boer, P.-T.; Kroese, D.P.; Mannor, S.; Rubinstein, R.Y. A tutorial on the cross-entropy method. Ann. Oper. Res. 2005, 134, 19–67." Insert it in the reference list in order of first citation, renumber every later reference and its citations, and cite it also in the "Search" row of Algorithm A3's parameter table.
```

## 3. Log of applied blocks

1. **2026-10-22, ten corrections** (3.1 learner settings; Algorithm 3 centroid and track deletion;
   3.4.1 crossing counts and the coupled composition; 3.4.3 metric list; 6.2 "entirely above";
   Algorithm A2 rudder delay; Appendix E validation-set digest; references 76 and 72). Applied in
   `Paper3_Sections1-3_with_placeholders_rev22Oct_corrected_tracked.docx` with three consistency
   edits; checked on 2026-10-24, all ten present.
2. **2026-10-24, blocks A and B, sent together** (A: the 30 no-target coupled layouts in 1.3, 3.4.1
   and Table A4; the 2.2.4 width sentence; "root-mean-square (RMS)" and the multi-axis sentence in
   3.4.3 with an Appendix G placeholder; the fixed-layout branch of Algorithm 4; the Appendix E
   digest sentence; "Stable-Baselines3 or sb3-contrib" in 3.1; the marker list. B, decision S19:
   3.4.4 and 4.4 deleted, 4.5 and 4.6 renumbered 4.4 and 4.5, the 4.4 placeholder for the deployed
   policy only, the 6.1 marker deleted, the 6.2 limitation and a 6.3 ablation sentence, the marker
   list). Applied in `Paper3_Sections1-3_with_placeholders_rev24Oct_tracked.docx`; checked on
   2026-10-25, all seventeen present, no ablation, 3.4.4, Section 4.6, Table R6 or random-policy
   reference left outside 6.2–6.3. The full text of both blocks is in commit a2eedbcf.

3. **2026-10-25, blocks C, D and E, sent together** (Section 4 introduction, 4.1–4.4 with Tables
   9–12 and Figures 4–6 placeholders, Appendix G Tables A6–A8, Algorithm A1, the 3.3 tuning text,
   and the 3.2, 3.4.3, 6.2 and 1.2.2 edits). Applied in
   `Paper3_Sections1-4_with_placeholders_rev25Oct_tracked.docx`; checked on 2026-10-26. Deviations
   from the blocks, all correct against the sources: "obstacle collisions become the main failure"
   in coupled episodes is stated for the deployed policy only (PPO and recurrent PPO collide more
   with the target there); the ship-domain intrusion and the rudder saturation fraction are named as
   rates in the table notes; "q" in Algorithm A1 is written "domain distance" and "/π" as "/180°".
   Flags raised by the log, both taken into block F: the comparator named "LOS-DWA" (3.3,
   Appendix F) and "LOS-PID + DWA" (21 times, Section 4); Algorithm A2's tuning row without the
   earlier build. The full text of blocks C, D and E is in commit 237f049b.

4. **2026-10-26, block F** (the runtime safety layer: 2.3.4 rewritten, Algorithm A3, Section 4.5
   with Table 13, and the 1.3, Algorithm 1, 2.2.4 and 6.2 edits; LOS-DWA throughout; the A2
   tuning row). Applied in `Paper3_Sections1-4_with_placeholders_rev26Oct_tracked.docx`; checked on
   2026-11-04: every item as given, Table 13 equal to `planning/MASTER_ARTIFACT_RESULTS.md`
   section 18. Comments 8 and 9 added (which parts of the layer run on the vessel). The log's
   three flags are block G; its "not checkable" items are confirmed by
   `planning_records/SAFETY_V21_PLAN.md`. The full text of block F is in commit eb02b1fc.
