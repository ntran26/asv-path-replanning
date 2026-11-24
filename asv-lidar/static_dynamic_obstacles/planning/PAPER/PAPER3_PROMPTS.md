# Paper 3 draft: edit instructions (kept current)

> **Kept current.** Last updated 2026-11-23. This file holds the next edit instructions for the
> Paper 3 draft, each as a block to paste into the chat that writes the paper. The blocks name
> planning files by path; that chat reads them from the synced project, so each push needs a Sync
> before the next block is sent. Only files outside `planning/` are attached: the .docx draft,
> kept in `PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`. Blocks already applied move
> to the log at the end.

## 1. State of the draft

Current file: `Paper3_Sections1-5_rev11Nov_tracked.docx` (saved to Downloads; its place is
`PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`): block H (Sections 5.1 and 5.2) as
tracked changes dated 11 November 2026 on block I's output with its changes accepted, checked on
2026-11-23 (log, entry 8). The next round accepts these changes first.

- **Drafted:** Sections 1–4 (4.5 included), 5.1–5.2, Appendices B–G (Algorithms A1–A3),
  Discussion 6.2–6.3.
- **Placeholders:** the opening of Section 5, 5.3, 5.4, 6.1 and 7; Appendix A; Figures 1–6 (to insert; Figures 4–6 are
  final, Figures 1–3 are drafts).
- **Applied and checked:** everything through block G, the clean version, block I and block H
  (log, entries 1–8).
- **References:** 1–78, first cited in order; 62 is Paper 1 (ICMCR 2026), 68 de Boer et al.
  (2005), 69–72 SAC, PPO, LSTM and TQC.
- **Markers left:** [PENDING — field-trial design] (1.2.4, 1.3, 5, 5.3); eight [PENDING] platform
  markers in 5.1–5.2 (IMU, telemetry, pose reference point, independent stop, reversal test,
  shore laptop's figures, target ground truth, analysis definitions); [CHECK — wall-margin
  convention]; [CHECK — Figure 2 placeholder terminology]; [VERIFY] (1.2.5); two [CITE]; [TBC —
  repository and DOI] and [CHECK — identification data] (Data Availability); comments 0, 1 and
  4–7 (comment 4 gives the reference history).

## 2. Next: block K, the shore laptop in Section 5.1 (2026-11-23)

Block H's text uses the development laptop's run times and leaves the shore laptop's figures
pending. The shore laptop is chosen and measured: the personal laptop, linked to the laptop that
controls the vessel as in July (`FIELD_CALIBRATION_PLAN.md` P-10, P-11 and T-14, 2026-11-14;
`field_deployment/README_P3_BRIDGE.md`, "Run time" and the parity table). On it the command path
meets the 0.5 s interval (p99 27.2 ms on test-set streams; LiDAR line to sent command p99 92 ms on
the July logs over loopback UDP, which the plan gives as 0.03 s and 0.09 s), the parity replay
passes again (32/32, 1,497 decisions), and E1 in the loop does not meet it (p99 0.887 s, 5.0 % of
decisions above 0.5 s). The vessel-link latency is in none of these figures, and the laptop-to-laptop
link is not tested on the bench. K3 ties "the July 2026 logs" of 5.2 to the identification data
of Section 2.1.2, which do not name the campaign.

**Attachment** (one): `Paper3_Sections1-5_rev11Nov_tracked.docx`, moved from Downloads to
`PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`.

Output: `Paper3_Sections1-5_rev23Nov_tracked.docx`.

```
First accept every tracked change in the attached Paper3_Sections1-5_rev11Nov_tracked.docx. Then apply the following changes as tracked changes dated 23 November 2026 (author "Claude"), keeping comments 0, 1 and 4–7, and save the result as Paper3_Sections1-5_rev23Nov_tracked.docx. The shore laptop has now been chosen and measured, so two sentences of Section 5.1 change. Report each change in a short revision-log addendum.

K1. Section 5.1, safety-layer paragraph. Replace "An optional block with the layer active is run only if the layer decides within the 0.5 s decision interval on the shore laptop, where its 99th percentile was 0.89–2.06 s per decision on a development computer, and if the speed controller reverses [PENDING — reversal test]; otherwise, no active run is made." with "An optional block with the layer active would need the layer to decide within the 0.5 s decision interval on the shore laptop and the speed controller to reverse [PENDING — reversal test]. On the shore laptop, the layer's 99th percentile is 0.89 s per decision, with 5 % of decisions above the interval, so the first condition is not met, and no active run is made unless it is."

K2. Section 5.1, bench paragraph.
- Replace "Replaying 16 test-set episodes through the bridge in shadow and in active mode, 32 runs and 1,497 decisions, gives" with "Replaying 16 test-set episodes through the bridge in shadow and in active mode, 32 runs and 1,497 decisions, on a development computer and again on the shore laptop, gives".
- Replace "The 99th percentile of the command path is 0.04 s per decision on test-set streams and 0.09 s from the LiDAR frame to the sent command on the July logs, on a development computer [PENDING — the shore laptop's figures]." with "On the shore laptop, the 99th percentile of the command path is 0.03 s per decision on test-set streams and 0.09 s from the LiDAR frame to the sent command on the July logs replayed through the bridge's network interface; the link to the vessel is not included [PENDING — latency of the link to the vessel]."

K3. Section 5.2, second paragraph. Replace "The July 2026 logs remain the provenance of the model used in training." with "The field logs from which the training model was identified (Section 2.1.2), those of the July 2026 campaign, remain its provenance."

K4. Marker list at the end: replace the entry "[PENDING — the shore laptop's figures] (11 November 2026): Section 5.1." with "[PENDING — latency of the link to the vessel] (23 November 2026): Section 5.1."
```

Queued after block K:
- Figures 1–3: the placeholder texts follow the final drawings (drafts in the liaison scratchpad;
  the author returns to them). Figure 1's dashed outline is the own ship's domain, so "the
  target's asymmetric ship domain" becomes the own ship's; Figure 2 uses "required turn sense",
  which resolves [CHECK — Figure 2 placeholder terminology]; Figure 3 adds the ego and path
  branches.
- Section 6.1 (block J, after block I; writable now for the two simulation questions); 6.2 to state that the
  water runs use constant-velocity targets only and no channel layout.
- Section 5.3 from `FIELD_TRIALS_PLAN.md` section 4 once D-1 is fixed; 5.4 after the trials.
- Appendix A (needs a table of the identified parameters with their bootstrap spread; F4 after
  the calibration).
- Section 7, the abstract and the conclusion, after the field trials.
- Length (held 2026-11-14): the clean rev05Nov runs to over 40 single-spaced pages (about 26,300
  words of main text, half of it Section 2). Candidates that keep the content: Algorithms 1–4,
  the 2.2.4 width derivation, Table 3a and Table 10 (Table 10 is in block I) to the appendices; the repeats in 1.3, of the
  training-target behavior, of the tracker settings and of the curriculum prose against Table 8
  removed; Appendices F and G to supplementary material; the literature review compressed. One
  block after block H, once the venue is chosen. Block I (section 3) already moves Table 10 and
  Figure 5 to the appendices and halves 3.2 and 3.3; the length block starts from its output.

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
   reference left outside 6.2–6.3. The full text of both blocks is in commit a32381c7.

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
   earlier build. The full text of blocks C, D and E is in commit 690a175c.

4. **2026-10-26, block F** (the runtime safety layer: 2.3.4 rewritten, Algorithm A3, Section 4.5
   with Table 13, and the 1.3, Algorithm 1, 2.2.4 and 6.2 edits; LOS-DWA throughout; the A2
   tuning row). Applied in `Paper3_Sections1-4_with_placeholders_rev26Oct_tracked.docx`; checked on
   2026-11-04: every item as given, Table 13 equal to `planning/MASTER_ARTIFACT_RESULTS.md`
   section 18. Comments 8 and 9 added (which parts of the layer run on the vessel). The log's
   three flags are block G; its "not checkable" items are confirmed by
   `planning_records/SAFETY_V21_PLAN.md`. The full text of block F is in commit d43db7e0.

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

7. **2026-11-23, block I** (the deployed-policy framing: 1.3 first paragraph and contribution 3;
   Section 3 retitled, 3.1 rewritten around SAC with Papers 1 and 2, 3.2 and 3.3 shortened with
   their details moved to Appendices B and F; the Section 4 opening; 4.1 reordered with the
   deployed policy first; 4.3 prose; 6.1–6.2; Paper 1 added as reference 62). Output
   `Paper3_Sections1-4_rev23Nov_tracked.docx`, checked 2026-11-23:
   - Rejecting the 23 November changes gives back the 5 November clean text (773 blocks; one tab
     in a reference entry differs). Comments 0, 1 and 4–9 keep their anchors; 6 and 7 now sit on
     references 78 and 74.
   - Numbers: the only new ones are the Paper 1 and Paper 2 figures, which match the PDFs (Paper 1
     Table III: SAC 90.0–95.5 %, PPO 49.4–67.4 %, five weightings 0.5–0.9, 1,000 episodes; Paper 2
     Table 5: SAC three seeds 92.4 %, PPO 90.5 %, 500 episodes; Section 2.5.2: same observation,
     action, reward, terminations and curriculum). No existing number changed. Gone from the text:
     the VO closest approach 2.05 m (still 2.054 in Table A6) and the SAC noncompliant head-on
     seed range 0.79–0.97 (an example of a wide range in the 4.3 note).
   - The three statements hold: the SAC/TQC choice on the test set with the field trials as the
     independent evaluation (3.2, 6.2); the SAC three-seed mean and range beside seed 0 (4.1,
     Table 9); the formulation developed with PPO (3.1, 4.1, Appendix B).
   - References 1–78 first cited in order; every unchanged paragraph cites the same works after
     renumbering. Tables 10–13 are now A9 and 10–12, Figures 5–6 now A1 and 5; every
     cross-reference follows.
   - Main text, Sections 1–7: prose 20,708 to 19,735 words (−973, as the revision log gives);
     with tables, 26,209 to 24,845. Appendices 7,571 to 9,061.
   - Carry-overs, folded into block H as H6: the "other encounter types" phrase in 4.1, the
     Table 9 header, the serial comma in contribution 3, the Appendix B heading.

8. **2026-11-23, block H** (2.3.4 shadow-mode sentence; 5.1 platform, bridge, localization,
   shadow mode, bench checks, phantom tracks, logging and pre-registration; 5.2 retitled, the two
   sessions, fit and validation, the four uses, the three comparison levels; the 5.3 phrase; the
   2.1.3 channel sentence; the four carry-overs of block I as H6; the marker lists). Output
   `Paper3_Sections1-5_rev11Nov_tracked.docx` (saved to Downloads), checked 2026-11-23:
   - Rejecting the 11 November changes gives back block I's text with its changes accepted
     (776 blocks, 0 differences). Comments 8 and 9 deleted as asked; 0, 1 and 4–7 unchanged.
   - Every number in 5.1 and 5.2 matches the plans and the bridge record as committed (all five
     sources last changed by 2026-11-20 and clean in the working tree): bench parity 32/32 and
     1,497 decisions, 19 July replays and 1,085 decisions, command path 0.04 s and 0.09 s, E1
     0.89–2.06 s, phantom tracks 42 %, 61 % and 15 of 19; the sessions (about 5 h and 4 h), S1-A
     positions, surge 6–24 RPM, crash stops at 9, 12 and 15 RPM and from a turn at 12 RPM,
     turning circles and zig-zags at 12, 6 and 18 RPM as amended, three crash stops in the
     holdout. Localization by scan-to-map registration on surveyed features and the registered
     map frame match `FIELD_CALIBRATION_PLAN.md` P-7 and `FIELD_TRIALS_PLAN.md` section 2; the
     phantom-track condition matches T-15.
   - H6 carry-overs all applied. References still first cited in order 1–78.
   - Main-text prose, Sections 1–7: 19,735 to 21,017 words by the liaison count (the revision log
     gives 19,559 to 20,841 by its own count; both +1,282).
   - Outdated since 2026-11-14, corrected by block K: the shore laptop is the personal laptop and
     is measured, so the active-block condition is known to fail there and the shore laptop's
     command-path figures exist. Also K3: 5.2 names "the July 2026 logs" as the provenance
     without linking them to 2.1.2.
   - Revision-log flag, not edited: a facts file of that chat asks the paper not to name the
     identification data files; the text names the July 2026 campaign, not files, as 2.1.2 and
     5.1 already do.

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

### Text of block I (sent 2026-11-23)

Requested 2026-11-23. The paper reports the formulation and the deployed policy first. The other
learners and the classical methods are baselines, described enough to justify the choice of
learner, with one comparison table.

The block touches Sections 1.3, 3, 4.1, 4.3 and 6, Appendices B, F and G, and the references. It
leaves Section 2 and Sections 4.2, 4.4, 4.5 and 5 unchanged. Block H touches only Sections 2.1.3,
2.3.4 and 5, so the two blocks are independent: either can run first, the second on the first's
output after its changes are accepted. Section 6.1 moves to block J.

**Attachments** (four):
1. `Paper3_Sections1-4_clean_rev05Nov.docx`, or block H's output with its changes accepted.
2. `static_dynamic_obstacles/planning/PAPER3_CLAIMS_AND_TABLES.md`. Attach it until the
   2026-11-23 version is pushed and synced. It holds the restated Q2, C-11, C-12 and C-13 and the
   section "Why SAC".
3. `PhD/Publications/Autonomous_Surface_Vessels_Path_Following_and_Static_Obstacle_Collision_Avoidance_using_Deep_Reinforcement_Learning.pdf`
   (Paper 1, ICMCR 2026).
4. `PhD/Publications/Deep Reinforcement Learning Control for Path Following and Static Obstacle Avoidance for Autonomous Surface Vessels.pdf`
   (Paper 2, reference [13]).

Output: `Paper3_Sections1-4_rev23Nov_tracked.docx`, or `Sections1-5` if block H has been applied.

Checked on 2026-11-23 against the two papers:
- **Paper 1, Table III:** over 1,000 randomized static-obstacle episodes, SAC succeeds in
  90.0-95.5 % and PPO in 49.4-67.4 %, across five reward weightings (0.5-0.9). The study used raw
  LiDAR beams and rudder-only control.
- **Paper 2, Section 3.3.2:** it restates the Paper 1 figures.
- **Paper 2, Section 2.5.1:** it gives three reasons for SAC: continuous rudder and propulsion
  commands, off-policy sample efficiency over randomized layouts, and entropy-driven exploration.
- **Paper 2, Section 2.5.2 and Table 5:** a PPO baseline with the same observation, action, reward,
  terminations and curriculum, over 500 episodes. The three SAC seeds reach 92.4 % and the three
  PPO seeds 90.5 %; the deployed SAC policy alone reaches 95.0 %.
- **Paper 1 reference details:** the authors are those of [13] in the same order; the venue is the
  2026 4th International Conference on Mechatronics, Control and Robotics (ICMCR), Tokyo, Japan,
  20-22 March 2026, pages 81-87, doi 10.1109/ICMCR69541.2026.11533898. The location and dates
  come from the DOI's Crossref record; the pages match the PDF.
- **Paper 1 contradicts itself:** its abstract names λ = 0.5 as the best trade-off, but its text
  names 0.7. The block cites only the success ranges.

```
Apply the following changes to the attached Paper 3 draft as tracked changes dated 23 November 2026 (author "Claude"), keeping every existing comment, and save the result as Paper3_Sections1-4_rev23Nov_tracked.docx (Sections1-5 if Section 5 is already written). The block reframes the paper around the formulation and the deployed policy: the deployed policy is reported first, and the other learners and the classical methods are baselines, described as far as they justify the choice of learner and summarized in one comparison table. Follow the attached claim ledger (PAPER3_CLAIMS_AND_TABLES.md, Part 1: the central questions, the section "Why SAC", and rows C-1, C-11, C-12 and C-13). Change no number that is already in the draft, and add no number except the Paper 1 and Paper 2 figures given in I3. Report each change in a short revision-log addendum, and give the main-text word count (Sections 1-7) before and after.

Rules: impersonal voice; American spelling; COLREGs-aware, never compliant; validation set, coupled and decoupled as in the draft. No internal identifiers in the paper text: no claim numbers, file names or code names. Keep these three statements, which the reframing must not lose:
- The choice between SAC and TQC used the test set, and the field trials are the independent evaluation.
- The SAC three-seed mean and seed range are reported beside the deployed seed.
- The formulation was developed with PPO.

I1. Section 1.3, first paragraph.
- Replace "Four learners—PPO, recurrent PPO, soft actor–critic (SAC), and truncated quantile critics (TQC)—are trained on the same frozen formulation; the deployed policy is the better of SAC and TQC on the held-out test set, and the remaining learners and two classical methods serve as baselines on the same set." with "The deployed policy is trained with soft actor–critic (SAC), the learner of the authors' earlier studies on the same vessel [P1,13]; three further deep reinforcement learning learners, trained on the same frozen formulation under one protocol, and two classical methods serve as baselines on the same held-out test set."
- Replace the central-question sentence with: "The central questions are whether the deployed policy completes the navigation task while producing encounter-consistent behavior, including in channels and among static obstacles where the room to maneuver is limited; how it compares with the learned and classical baselines on the same episodes; and how it performs in physical basin trials."
Here and below, [P1] stands for the new reference of I10.

I2. Section 1.3, contribution 3. Replace the bullet with: "A deployed policy and its benchmark. The deployed policy is trained with SAC, the learner of the authors' earlier studies [P1,13], and is benchmarked on the held-out test set. The baselines are PPO, recurrent PPO and truncated quantile critics (TQC), trained under the same protocol (3.0 million environment steps, three training seeds, checkpoints selected on a separate validation set), and two classical methods: an encounter-specific velocity-obstacle method [31], and line-of-sight guidance with proportional–integral–derivative heading control and a dynamic-window avoidance layer [21]. All of them use the same perception pipeline. Seed variability is reported with every learner result."
In the paragraph "The remainder of the paper…", replace "Section 3 describes the learners, the training protocol, the classical comparators, and the evaluation design." with "Section 3 describes the deployed learner and the baselines, the training protocol, and the evaluation design."

I3. Section 3.
- Retitle it "Training, baselines, and evaluation design", and retitle 3.1 "Deployed learner, baselines, and policy architecture". Adjust the section's opening paragraph to match.
- Replace the first paragraph of 3.1 with a paragraph that makes these points, in this order:
  (a) The deployed policy is trained with SAC [70], an off-policy maximum-entropy actor–critic method.
  (b) SAC is the learner of the authors' earlier studies on the same vessel. It was chosen there for continuous rudder and propulsion commands, for the sample efficiency of off-policy learning over randomized layouts, and for entropy-driven exploration [13].
  (c) In the first study, with raw LiDAR beams and rudder-only control, SAC completed 90.0–95.5 % of 1,000 randomized static-obstacle episodes against 49.4–67.4 % for PPO, across five reward weightings [P1].
  (d) In the second study, against a PPO baseline trained on the same observation, action, reward and curriculum, the three SAC seeds reached 92.4 % against 90.5 % for PPO on 500 episodes [13].
  (e) Both studies concern path following among static obstacles without a target vessel, so they motivate the choice for this task rather than establish it.
  (f) Three further learners are therefore trained on the frozen formulation under the same protocol as baselines: proximal policy optimization (PPO) [68], recurrent PPO with long short-term memory (LSTM) [69], and TQC [71], a distributional variant of SAC.
  (g) Keep the existing statements, shortened: the library versions [72]; training from scratch with no warm start from [13], whose observation and reward differ; library defaults except the common discount and reward normalization, with every setting in Appendix B (the PPO rollout and Kullback–Leibler settings and the SAC/TQC warm-up go there only); no setting tuned per learner or per seed; and "The formulation was developed with PPO, but it refers to no learning algorithm."
- Keep the network paragraph, and make clear that it is the deployed policy's network, shared by the baselines.

I4. Section 3.2. Keep, in about half the length:
- the protocol: three seeds, 3.0 × 10⁶ steps on the curriculum of Section 2.3.5, 10 simulators;
- validation every 2 × 10⁵ steps with the layer off, scored as goal rate − 2 × collision rate, with each seed represented by its best checkpoint;
- the final models evaluated beside them, as decided before any final model was tested;
- "Every trained seed is reported";
- the two-step deployment rule and the sentence that the test set decides the deployed learner family as well as measuring it, with the field trials as the independent evaluation.
Move to Appendix B: the run histories (seed 0 trained to 2.5 × 10⁶ steps and continued; TQC seed 2 resumed from its 7.5 × 10⁵-step checkpoint; the extra SAC and TQC check at 2.5 × 10⁶ steps) and the retraining rule (only for a documented defect; seeds added for every learner).

I5. Section 3.3, about half the length.
Keep:
- that both comparators see only what the policy sees and act through the same actuator limits;
- the shared LOS guidance [73];
- one or two sentences each on what LOS-DWA does: a dynamic window simulated through the identified vessel model, knowing obstacles but no rules;
- and what the encounter-specific VO does: this study's interpretation of [31], not their implementation, which reads the formulation's encounter assessment and convention and treats the passing side as a hard constraint while engaged;
- that both are tuned on the decoupled part of the validation set with the learners' score and fixed before the test set;
- the sentence on Kuwata et al. [19].
Move to Appendix F, beside Algorithms A1 and A2, where they are not already there:
- the mechanics (the 9 m and 1.0 m triggers, the 56 candidates, the rejection gaps, the scoring terms, the candidate sweep and the domain penalty);
- the tuning grids and scores.

I6. Section 4, opening paragraph. Shorten it. Keep the reporting conventions in one sentence, then state the deployed policy and why in two sentences: SAC is retained by the rule of Section 3.2 (three-seed mean 0.846 against 0.830 for TQC), and seed 0 has the best validation score of the SAC seeds (+0.411). Drop the comparison of seed 0's test-set success with the other SAC seeds here; 4.1 gives it.

I7. Section 4.1, rewritten in this order.
(a) The deployed policy first. Give its success overall, decoupled and coupled; its collisions by type; its ship-domain intrusions; and its hull-to-hull clearance, all from Table 9. Give beside them the SAC three-seed mean and range (0.846, 0.827–0.865), so that the single seed is read against its family. Then the coupled episodes as its main difficulty, with obstacle collisions as the main coupled failure, and the basin and channel comparison.
(b) Against the classical comparators: keep the LOS-DWA and encounter-specific VO comparisons with their exact McNemar tests, overall, decoupled and coupled. Shorten the VO paragraph to its overall result, its boundary-collision rate and its clearance; move its per-type figures and its weakest case to the notes of Appendix G.
(c) Against the learned baselines, one paragraph:
- every off-policy seed succeeds more often than every on-policy seed (lowest 0.815 against highest 0.757);
- the gap lies in target collisions and domain intrusions, not obstacle collisions;
- this agrees with the authors' earlier SAC-over-PPO results [P1,13], obtained with earlier formulations, although the formulation was developed with PPO;
- the SAC and TQC means differ by less than their seed spread, so no ranking between them is claimed;
- the final models change no conclusion and would also select SAC (one sentence, pointing to the per-run table now in Appendix G);
- every SAC and TQC run reaches its best validation score at 2.8 × 10⁶ steps or later (one sentence, pointing to the learning curves now in Appendix B).
(d) Encounter types (Figure 4) for the deployed policy, with the other methods in one sentence.
Keep Table 9 as the single comparison table. Order its columns as deployed policy, SAC (three seeds), TQC, PPO, recurrent PPO, LOS-DWA and encounter-specific VO, and give the seed range under each learner's success.
Move Table 10 to Appendix G and Figure 5 to Appendix B, with their notes and captions. Remove the paragraph on recurrent PPO seed 1 and keep it as one sentence in the note of the moved table.

I8. Section 4.3. Report the deployed policy and the two classical comparators in the prose, and leave the other learners to Table 11. Keep:
- the noncompliant head-on result (60 of 66 against 65 of 66);
- the speed-change result (0.808 to 0.620, against LOS-DWA 0.564 to 0.553 and the encounter-specific VO 0.455 to 0.427);
- the episodes without a target;
- the note that differences below about 0.05 are not interpreted.

I9. Section 6.
- 6.1 placeholder: restate the three central questions as in I1.
- 6.2: shorten the learner sentences to one: three seeds give a limited estimate of training variability, so no ranking between SAC and TQC is drawn. Keep the sentence that the deployed learner family is chosen on the test set and that the field trials are the independent evaluation.
- 6.3: keep "Choosing the deployed learner on a second held-out set…".

I10. References. Add Paper 1 at its first citation, renumbering the later references and every citation to them: "Tran, N.H.; Nguyen, H.D.; King, P.; Tran, M. Autonomous surface vessels path following and static obstacle collision avoidance using deep reinforcement learning. In Proceedings of the 2026 4th International Conference on Mechatronics, Control and Robotics (ICMCR), Tokyo, Japan, 20–22 March 2026; pp. 81–87. https://doi.org/10.1109/ICMCR69541.2026.11533898." Renumber the tables and figures that follow the moved ones, and update every cross-reference.
```

### Text of block H (sent 2026-11-23)

Requested 2026-11-11; amended 2026-11-14 (gyro bias tied to the IMU marker; turns at the
cruise setting added to the calibration). The field-trial and calibration plans are final apart from the open
platform items (target vessel D-1, ground-truth method D-2, IMU O-2, telemetry O-1/P-1, pose
reference point O-9, shore laptop P-10/P-11, independent stop P-12), which the text marks
[PENDING]. 5.3, 5.4, 6.1 and 6.2 are out of this block except for the 5.3 phrase in H4.

**Attachments** (six). The planning files are attached as well as synced, because the synced
project lacks the 2026-11-07 registration of C-15 to C-20 (uncommitted), and `field_deployment/`
is outside `planning/`:
1. `Paper3_Sections1-4_rev23Nov_tracked.docx` (`PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`),
   the output of block I; the block accepts its changes first
2. `static_dynamic_obstacles/planning/FIELD_TRIALS_PLAN.md`
3. `static_dynamic_obstacles/planning/FIELD_CALIBRATION_PLAN.md` (the 2026-11-14 version, with the
   RPM 6 turns; uncommitted)
4. `field_deployment/README_P3_BRIDGE.md`
5. `static_dynamic_obstacles/planning/PAPER3_CLAIMS_AND_TABLES.md` (working copy)
6. `field_deployment/trial_analysis/README_TRIAL_ANALYSIS.md` (working copy, uncommitted)

Output: `Paper3_Sections1-5_rev11Nov_tracked.docx`.

Checked on 2026-11-11 against the code and the plans:
- Training applies no per-beam range noise, beam dropout, grazing no-return or aft mask:
  `LIDAR_DROPOUT_P = 0.0`, `LIDAR_NO_RETURN_GRAZING_DEG = 0.0` and `LIDAR_AFT_MASK_HALF_DEG = 0.0`
  in `src/constants.py` and `configs/baseline_v3.json`; ranges are exact ray casts with the 1 m
  dead zone. "Which training does not model" therefore holds for range noise, dropout, the
  black-wall return rate and aft self-occlusion; 2.3.1 already says so.
- The trial families (A no target, B open basin, C on the reference layouts) contain no channel
  layout, so the 5.1 channel sentence and the last sentence of 2.1.3 change (H2, H5).
- The bench figures in H2 are those of `README_P3_BRIDGE.md` (development laptop, 2026-11-05):
  parity 32/32 runs and 1,497 decisions; staleness 6/6; July logs 19 replays and 1,085 decisions;
  command path p99 37.9 ms on test-set streams and 86 ms from the LiDAR line to the sent command
  on the July logs; E1 in the loop p99 0.89 s and 2.06 s; phantom tracks on 42 % of decisions,
  61 % of them within 1 m of the map boundary or beyond, stops in 15 of 19 replays.
- Since 2026-11-14 session 1 runs turning circles at helm 50 and 100 %, both signs, and a 20/20
  zig-zag (2 repeats, both initial directions) at the cruise setting RPM 6, beside the RPM 12 sets
  (four helm settings, both signs; 20/20 and 10/10 zig-zags) and the RPM 18 pair; the crash stops
  stay at RPM 9-15 and the surge sweep starts at RPM 6. Session 1 is about 4 h 50 min ("about
  5 h"), session 2 about 4 h. The coverage groups of `README_TRIAL_ANALYSIS.md` include the RPM 6
  turns; H3(c) names the two speeds and gives no numbers.
- Gyro bias exists only if an IMU is fitted (O-2 open), so H3(a) ties it to the IMU marker.
- The "Definitions fixed here" table of `README_TRIAL_ANALYSIS.md` is not yet confirmed: H3
  describes the comparison levels and leaves the counting rules, seed counts and the twin's
  success rule to the pre-registration.

```
First accept every tracked change of 23 November 2026 in the attached Paper3_Sections1-4_rev23Nov_tracked.docx. Then apply the following changes as tracked changes dated 11 November 2026 (author "Claude"), keeping comments 0, 1 and 4–7, and save the result as Paper3_Sections1-5_rev11Nov_tracked.docx. The block writes Sections 5.1 and 5.2 from the attached field-trial plan (FIELD_TRIALS_PLAN.md, sections 2–7 and 10), calibration plan (FIELD_CALIBRATION_PLAN.md, sections 1–7 and 9), bridge record (README_P3_BRIDGE.md), claim ledger (PAPER3_CLAIMS_AND_TABLES.md, Part 1 section 2 and the F tables of Part 2) and trial-analysis record (README_TRIAL_ANALYSIS.md); where a synced copy differs, the attached copy holds. Sections 5.3 and 5.4 stay placeholders except for H4. Report each change in a short revision-log addendum, with every number in 5.1 and 5.2 traced to its file and section.

Rules for the new text: methods only, in the present tense, as a design that the sessions carry out; no field result, since none exists yet; impersonal voice; American spelling (localization, behavior, maneuver, centered), although the attached files use British spelling; COLREGs-aware, never compliant; "rule-consistent" for the track-based conditions of Section 3.4.3, although the analysis code calls them compliance; validation set, coupled and decoupled as in the draft. No internal identifiers in the paper text: no E1, D-4, P-7, S1-A, T-14, C-20 or file names; write "the runtime safety layer (Section 2.3.4)", "the pre-committed thresholds", "the dress rehearsal". The safety layer's would-be interventions are always reported apart from the policy. Open platform items are written as highlighted markers, not stated.

H1. Section 2.3.4, last paragraph. Delete "In the field trials, the same code also serves as an operator-latched stop (Section 5)." At the end of the paragraph, after "…between learned behavior and externally enforced intervention.", add: "In the field trials, the layer runs in shadow mode: it decides at every decision step and every decision is logged, but none of its commands reaches the vessel, and an operator stop is the physical safety (Section 5.1)." Delete comment 8, which this resolves.

H2. Section 5.1 (Platform and setup): replace the placeholder with method text covering, in this order:
 (a) The vessel, its two-dimensional LiDAR and the 10 × 25 m basin of the previous study [13]; the deployed policy (SAC seed 0, the best validation checkpoint) is used unchanged, and the bridge refuses any other file (it checks the checkpoint's hash and its configuration against the code).
 (b) The shore-side bridge makes one decision per LiDAR frame at 2 Hz. The observation is built by the simulator's own environment code with only its sensor models replaced by the measured scan, the estimated pose and the measured ego motion, so the map gate, the 27 pooled sectors, the seven boundary rays, clustering, tracking, the encounter classification and latch, and the 70-value observation are the code of Section 2.3, unchanged; no physics or target is simulated. The command mapping is the training one (Section 2.1.1): no rudder command limiter, propulsion from zero to twice the cruise setting, no astern from the policy.
 (c) Localization: a scan-to-map pose anchored on surveyed fixed features of the facility, with the bridge running in the registered map frame, so that the navigable polygon is the map's, as in simulation (Section 2.1.3). Markers: [PENDING — IMU on the vessel and its clock]; [PENDING — telemetry rate and pose resolution]; [PENDING — pose reference point, hull or LiDAR, settled in the first calibration session].
 (d) The safety layer in shadow mode. The layer of Section 2.3.4 runs in a separate process off the command path, so it never delays a command. It decides at every decision and each decision is logged; its brake request is cleared, and its model of its own rudder is advanced with the command actually issued, so that it judges the vessel the policy is steering; its first would-be stop is permanent, as an active stop would be, so later decisions report the stop. The learned behavior on the water is therefore the policy's own, as in Section 4, and the layer's would-be interventions are measured on the water and reported apart. The operator stop on the bridge is always armed, and a second stop does not depend on the shore laptop [PENDING — independent stop, tested on the bench]. An optional block with the layer active is run only if the layer decides within the 0.5 s decision interval on the shore laptop (its 99th percentile was 0.89–2.06 s per decision on a development computer) and the speed controller reverses [PENDING — reversal test]; otherwise no active run is made.
 (e) Bench checks, stated as bench results and not as field results: replaying 16 test-set episodes through the bridge in shadow and in active mode, 32 runs and 1,497 decisions, gives the simulator's observation, action, issued command and every safety-layer decision exactly; all 19 logs of the July 2026 campaign replay in shadow mode (1,085 decisions) without error; the command path's 99th percentile is 0.04 s per decision on test-set streams and 0.09 s from the LiDAR frame to the sent command on the July logs, on a development computer [PENDING — the shore laptop's figures].
 (f) Phantom moving tracks, as a bench finding that conditions the trials (in 5.1 or in 5.4, at the author's choice; never as a trial result): in the July logs, which hold no target and were run open loop by the previous study's policy at about twice the deployed cruise speed, 42 % of decisions carry a moving track, 61 % of them within 1 m of the map boundary or beyond it, and the shadow layer would commit a stop in 15 of the 19 replays. The trials start only once the dress rehearsal shows that the registered map frame removes them; otherwise their remaining rate is reported with its effect on the target branch.
 (g) What is logged per decision on one clock: the raw ranges, the pose and ego estimates fed to perception, the observation, the policy action, the issued command, the tracks and encounter contexts, every safety-layer decision, and operator stops; target ground truth, video and the run sheet are kept beside them [PENDING — target ground-truth method, fixed after the calibration sessions].
 (h) Pre-registration: the scenario list and order, the metric definitions, the analysis scripts and the empty result tables are committed before the first trial session; invalid runs are kept and reported as invalid.
 Replace "Channel layouts can be staged inside the basin with virtual boundaries (Section 2.1.3); which layouts are run is part of the pending trial design." with a sentence stating that the trials use the three fixed reference layouts (the previous study's basin layouts [13], with 1.0 m panels) and open basin legs, that no channel layout is run on the water, and that Section 5.3 gives the scenarios. Delete comment 9, which (d) resolves.

H3. Section 5.2: retitle it "Vessel-model calibration and sim-to-field comparison" and replace the placeholder with method text covering:
 (a) Two basin calibration sessions before the trials, the first of about 5 h and the second of about 4 h. The first identifies: a sensing baseline, stationary at surveyed poses mid-basin and near the light and the matte black walls (gyro bias if an IMU is fitted [PENDING — IMU on the vessel and its clock]; localization drift at rest, range noise, the return rate against the black wall, aft self-occlusion; training models none of these LiDAR effects, Section 2.3.1); rudder steps that separate servo rate, lag and transport delay; a surge sweep from rest at 6–24 RPM and coast-downs, for the thrust curve and drag; crash stops at full astern with the rudder centered, from straight runs at 9, 12 and 15 RPM and from a steady turn at 12 RPM; turning circles of two full turns from settled speed, at four helm settings at 12 RPM and at helm 50 and 100 % at the cruise setting of 6 RPM, each to both sides, and at helm 50 and 100 % at 18 RPM; and 20/20 and 10/10 zig-zags at 12 RPM and a 20/20 zig-zag at 6 RPM. It ends with a check against pre-committed acceptance thresholds, so that a failed block is repeated the same day. The second session validates on data the fit has not seen: repeatability without refitting; an open-loop replay holdout that includes three crash stops; and a closed-loop dress rehearsal of the deployed policy through the bridge, in the trial configuration, on the reference layouts and an open leg with no target.
 (b) The fit uses the first session only, and the validation the second only; if a block has to be refitted with second-session data, the paper says so. The July 2026 logs remain the provenance of the model used in training; they lie at about twice the cruise speed (Section 2.1.2), so the first session's surge sweep starts at the cruise setting, and turning circles and a zig-zag are also run at it, the speed of the trials.
 (c) The four uses of the calibration: model provenance (an archived identification data set and the standard maneuvering quantities, reported in Appendix A); coverage, the identified dynamics against the plant randomization the policy trained on, with the turns compared at 12 RPM and at the cruise setting, and with the field results read as out-of-distribution transfer if they lie outside it; a calibrated analysis model for the comparison in (d); and the safety layer's physical assumptions, the measured reverse response and the drift after a stop from a turn against the weak and strong reverse models of its stop test (Section 2.3.4). The training simulator stays frozen: the calibrated model is an analysis model only, and no result of Section 4 changes.
 (d) The sim-to-field comparison at three levels. Open loop: the logged command sequences replayed through the training model and the calibrated model, with heading and position error against horizon. Closed-loop twin: each run re-simulated from its logged start, with the target scripted along its measured track and the same frozen policy, in both models and over several sensor-noise seeds, compared by outcome agreement and by each metric's field value against the twins' distribution. Population: field successes per family and cell against the test set's same cells, reported as counts with the small-sample caveat. If the calibrated twin reproduces the field better than the training model, the gap is attributed to the dynamics; otherwise, to perception, localization or disturbance; either outcome is reported. Do not state the counting rules, the number of seeds or the twin's success rule: they are fixed at the pre-registration [PENDING — analysis definitions, confirmed at pre-registration].
 This resolves [TBC — logs for the replay]; delete that marker.

H4. Section 5.3: replace "run by the deployed policy with the latched stop available to the operator" with "run by the deployed policy with the operator stop armed and the runtime safety layer in shadow mode (Section 5.1)". The rest of 5.3 and its [PENDING] marker stay.

H5. Section 2.1.3, last sentence: replace "Because the boundary is virtual, every channel width used here can also be staged inside the physical basin (Section 5)." with "Because the boundary is virtual, a channel narrower than the basin could also be staged inside it; the field trials of Section 5 run no channel layout."

H6. Four carry-overs from the 23 November revision. (a) Section 4.1, last paragraph: replace "against 0.859–0.901 for the other encounter types" with "against 0.859–0.901 for the head-on, overtaking, and being-overtaken episodes", as in the 5 November text (Figure 4 also has a no-encounter group). (b) Table 9, header of the third column: "SAC (three seeds)". (c) Section 1.3, contribution 3: "PPO, recurrent PPO, and truncated quantile critics (TQC)", with the serial comma used elsewhere. (d) Appendix B heading: "Learner settings and training runs", since it now holds the run histories and Figure A1.

H7. Marker lists at the end: delete the two [CHECK] entries of 26 October on the operator-latched stop and the [TBC — logs for the model replay] entry; add the [PENDING] markers of H2 and H3 under Sections 3–7, dated 11 November 2026; extend the header line's history with "Sections 5.1 and 5.2 written and the field role of the safety layer settled 11 November 2026".
```
