# Paper 3 draft: edit instructions (kept current)

> **Kept current.** Last updated 2026-11-05. This file holds the next edit instructions for the
> Paper 3 draft, each as a block to paste into the chat that writes the paper. The blocks name
> planning files by path; that chat reads them from the synced project, so each push needs a Sync
> before the next block is sent. Only files outside `planning/` are attached: the .docx draft,
> kept in `PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`. Blocks already applied move
> to the log at the end.

## 1. State of the draft

Current file: `Paper3_Sections1-4_clean_rev05Nov.docx` in `PhD/3 Second Journal Paper -
Dynamic Obstacle/Revision/`: a single version with every tracked change accepted and Track
Changes off, checked on 2026-11-05 against `Paper3_Sections1-4_with_placeholders_rev05Nov_tracked.docx`
(log, entry 6). The next round tracks its changes against this file.

- **Drafted:** Sections 1–4 (4.5 included), Appendices B–G (Algorithms A1–A3), Discussion 6.2–6.3.
- **Placeholders:** Sections 5, 6.1 and 7; Appendix A; Figures 1–6 (to insert; Figures 4–6 are
  final, Figures 1–3 are drafts).
- **Applied and checked:** everything through block G and the clean version (log, entries 1–6).
- **References:** 1–77, first cited in order; 67 is de Boer et al. (2005).
- **Markers left:** [PENDING — field-trial design] (1.2.4, 1.3, 5, 5.3); [CHECK — wall-margin
  convention]; [CHECK — Figure 2 placeholder terminology]; [VERIFY] (1.2.5); two [CITE]; [TBC —
  replay logs] (5.2); [TBC — repository and DOI] and [CHECK — identification data] (Data
  Availability); comments 0, 1 and 4–9 (comments 8 and 9: which parts of the layer run on the
  vessel; comment 4 gives the reference history).

## 2. Next: on hold until the field-trial plan is final (2026-11-05)

No block is pending. The next round starts from the clean file and is written once the
field-trial plan is final. Queued for it:
- Figures 1–3: the placeholder texts follow the final drawings (drafts in the liaison scratchpad;
  the author returns to them). Figure 1's dashed outline is the own ship's domain, so "the
  target's asymmetric ship domain" becomes the own ship's; Figure 2 uses "required turn sense",
  which resolves [CHECK — Figure 2 placeholder terminology]; Figure 3 adds the ego and path
  branches.
- Section 6.1 (block H; writable now for the two simulation questions).
- Appendix A (needs a table of the identified parameters with their bootstrap spread).
- Sections 5 and 7, the abstract and the conclusion, after the field trials.

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

5. **2026-11-05, block G** (G1 Table 13's 0.361 and 36.1 % in 4.5; G2 the stop-latch clause after
   Eq. (18) removed; G3 de Boer et al. (2005) as reference 67, cited in 2.3.4 and Algorithm A3,
   references 68–77 renumbered; G4–G6 the texts of Figures 4–6). Applied in
   `Paper3_Sections1-4_with_placeholders_rev05Nov_tracked.docx`; checked on 2026-11-05: every item
   as given, the reference list 1–77 in order of first citation. The log flagged panel (f)'s
   "up to 0.52 m off its path" against the README's "beyond the path band": the trace of
   BAS-BO-CV-039 gives an offset from the track of 0.51 m at 4 s after the CPA (13.5 s), the end
   of the pass, so the text stands; the offset reaches 0.67 m later, on the way to the goal.
   Comment 10 recorded the renumbering. Full text of block G and its checks below.
6. **2026-11-05, clean version.** Every tracked change accepted, Track Changes off, comment 4
   updated to 77 references, comment 10 deleted: `Paper3_Sections1-4_clean_rev05Nov.docx`.
   Checked on 2026-11-05: no revision element in any part; the text equals the accepted view of
   the 5 November file (2,509 non-empty blocks, 0 differences); comments 0, 1 and 4–9 on the
   same anchors; references 1–77 in order of first citation; 30 highlighted markers, as in the
   input.

### Text of block G (sent 2026-11-05)

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
Apply the following changes to the attached Paper3_Sections1-4_with_placeholders_rev26Oct_tracked.docx as tracked changes dated 5 November 2026 (author "Claude"), keeping every earlier tracked change and comment, and save the result as Paper3_Sections1-4_with_placeholders_rev05Nov_tracked.docx. G1–G3 resolve three flags of the 26 October log; G4–G6 bring the text of Figures 4–6 in line with the final figures. Figures 1–3 are not touched. Report each change in a short revision-log addendum.

G1. Table 13, last row: replace "0.36" with "0.361" (361 of the 1,000 episodes; planning/MASTER_ARTIFACT_RESULTS.md, section 18). In the Section 4.5 text, replace "and in 36 % of the episodes" with "and in 36.1 % of the episodes".

G2. Section 2.3.3, after Eq. (18): replace "Ueff is Uref except under a carve-out (Eq. 19), and gu = 1 while the runtime stop is latched." with "Ueff is Uref except under a carve-out (Eq. 19)." The clause refers to a stop latch that no reported run sets: the layer is off in training, and the stop of the layer of Section 2.3.4 does not use that latch.

G3. Section 2.3.4: cite the cross-entropy method at "a cross-entropy search", with the reference "de Boer, P.-T.; Kroese, D.P.; Mannor, S.; Rubinstein, R.Y. A tutorial on the cross-entropy method. Ann. Oper. Res. 2005, 134, 19–67." Insert it in the reference list in order of first citation, renumber every later reference and its citations, and cite it also in the "Search" row of Algorithm A3's parameter table.

G4. Figure 4. The figure groups the null and no-target episodes as "No encounter" and adds the whole test set as "All". Replace the placeholder text with "[Figure 4 placeholder] Success by encounter type on the test set (no encounter, that is, null and no-target episodes; head-on; crossing from port; crossing from starboard; overtaking; being overtaken; and all episodes): grouped bars, learners as the three-seed mean with the individual seeds and the seed range, LOS-DWA and the encounter-specific VO hatched without an interval." Keep the caption.

G5. Figure 5 plots the validation-set success rate, not the selection score.
 (a) Replace "Figure 5 shows the validation-set score during training. Every SAC and TQC run reaches its best validation score at 2.8 × 10⁶ steps or later and scores higher at 3.0 × 10⁶ than at 2.0 × 10⁶ steps; the best scores range from +0.300 to +0.411 for SAC, from +0.267 to +0.378 for TQC, from −0.033 to +0.100 for PPO, and from −0.289 to +0.033 for recurrent PPO." with "Figure 5 shows the validation-set success rate during training. Every SAC and TQC run reaches its best validation score (goal rate − 2 × collision rate, Section 3.2) at 2.8 × 10⁶ steps or later and scores higher at 3.0 × 10⁶ than at 2.0 × 10⁶ steps; the best scores range from +0.300 to +0.411 for SAC, from +0.267 to +0.378 for TQC, from −0.033 to +0.100 for PPO, and from −0.289 to +0.033 for recurrent PPO (Table 10)."
 (b) Replace the placeholder text with "[Figure 5 placeholder] Validation-set success rate every 2 × 10⁵ steps for the four learners: three-seed mean, individual seeds and seed-range band, with the curriculum stage boundaries marked."
 (c) Replace the caption "Validation-set score of the four learners during training." with "Validation-set success rate of the four learners during training."

G6. Figure 6, panel (f), now a different episode with a constant-velocity target.
 (a) Section 4.2: replace the sentence beginning "When being overtaken (f), in the basin on a slanted leg without panels and by a reactive target (Section 3.4.2), …" and ending "… and once the target is past, it regains the path and speeds up." with "When being overtaken (f), in the basin on a slanted leg without panels and by a constant-velocity target, the own ship keeps within 0.10 m of its track and within 0.074 m/s of its speed until the target is within 2.5 m, at 0.53–0.60 m/s; as the target passes on its port side (CPA 1.9 m at 9.5 s), the own ship moves up to 0.52 m off its path, and once the target is past, it speeds up to about 0.9 m/s and reaches the goal at 18.5 s."
 (b) In the same paragraph, replace "In panel (f), the speed change of +0.14 m/s is inside that band but above the 0.10 m/s at which the speed part of the hold term grows linearly (Eq. (27)), and the turn and the slowing come once the target is close; both reflect the competition between stand-on course keeping and the ship-domain term stated in Section 6.2." with "Panel (f) stays inside the 0.10 m/s at which the speed part of the hold term grows linearly (Eq. (27)) until the target is within 2.5 m; the room it then gives is small, and some such room appears in each of the 29 successful being-overtaken episodes with a CPA of 1.2–4.5 m, and it reflects the competition between stand-on course keeping and the ship-domain term stated in Section 6.2."
 (c) In the Figure 6 caption, replace "The target of panel (f) is reactive (Section 3.4.2); the other targets keep their course and do not give way, and the target of panel (d) changes speed once." with "The targets keep their course and do not give way; the target of panel (d) changes speed once."
 (d) The figure's panel titles now read "Crossing from port" and "Crossing from starboard"; the caption already uses these terms.
```
