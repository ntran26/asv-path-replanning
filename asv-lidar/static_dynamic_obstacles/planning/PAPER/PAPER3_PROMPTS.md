# Paper 3 draft: edit instructions (kept current)

> **Kept current.** Last updated 2026-11-14. This file holds the next edit instructions for the
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

## 2. Next: block H, Sections 5.1 and 5.2 and the field role of the safety layer (2026-11-11)

Requested 2026-11-11; amended 2026-11-14 (gyro bias tied to the IMU marker; turns at the
cruise setting added to the calibration). The field-trial and calibration plans are final apart from the open
platform items (target vessel D-1, ground-truth method D-2, IMU O-2, telemetry O-1/P-1, pose
reference point O-9, shore laptop P-10/P-11, independent stop P-12), which the text marks
[PENDING]. 5.3, 5.4, 6.1 and 6.2 are out of this block except for the 5.3 phrase in H4.

**Attachments** (six). The planning files are attached as well as synced, because the synced
project lacks the 2026-11-07 registration of C-15 to C-20 (uncommitted), and `field_deployment/`
is outside `planning/`:
1. `Paper3_Sections1-4_clean_rev05Nov.docx` (`PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`)
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
Apply the following changes to the attached Paper3_Sections1-4_clean_rev05Nov.docx as tracked changes dated 11 November 2026 (author "Claude"), keeping comments 0, 1 and 4–7, and save the result as Paper3_Sections1-5_rev11Nov_tracked.docx. The block writes Sections 5.1 and 5.2 from the attached field-trial plan (FIELD_TRIALS_PLAN.md, sections 2–7 and 10), calibration plan (FIELD_CALIBRATION_PLAN.md, sections 1–7 and 9), bridge record (README_P3_BRIDGE.md), claim ledger (PAPER3_CLAIMS_AND_TABLES.md, Part 1 section 2 and the F tables of Part 2) and trial-analysis record (README_TRIAL_ANALYSIS.md); where a synced copy differs, the attached copy holds. Sections 5.3 and 5.4 stay placeholders except for H4. Report each change in a short revision-log addendum, with every number in 5.1 and 5.2 traced to its file and section.

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

H6. Marker lists at the end: delete the two [CHECK] entries of 26 October on the operator-latched stop and the [TBC — logs for the model replay] entry; add the [PENDING] markers of H2 and H3 under Sections 3–7, dated 11 November 2026; extend the header line's history with "Sections 5.1 and 5.2 written and the field role of the safety layer settled 11 November 2026".
```

Queued after block H:
- Figures 1–3: the placeholder texts follow the final drawings (drafts in the liaison scratchpad;
  the author returns to them). Figure 1's dashed outline is the own ship's domain, so "the
  target's asymmetric ship domain" becomes the own ship's; Figure 2 uses "required turn sense",
  which resolves [CHECK — Figure 2 placeholder terminology]; Figure 3 adds the ego and path
  branches.
- Section 6.1 (block I; writable now for the two simulation questions); 6.2 to state that the
  water runs use constant-velocity targets only and no channel layout.
- Section 5.3 from `FIELD_TRIALS_PLAN.md` section 4 once D-1 is fixed; 5.4 after the trials.
- Appendix A (needs a table of the identified parameters with their bootstrap spread; F4 after
  the calibration).
- Section 7, the abstract and the conclusion, after the field trials.
- Length (held 2026-11-14): the clean rev05Nov runs to over 40 single-spaced pages (about 26,300
  words of main text, half of it Section 2). Candidates that keep the content: Algorithms 1–4,
  the 2.2.4 width derivation, Table 3a and Table 10 to the appendices; the repeats in 1.3, of the
  training-target behavior, of the tracker settings and of the curriculum prose against Table 8
  removed; Appendices F and G to supplementary material; the literature review compressed. One
  block after block H, once the venue is chosen.

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
