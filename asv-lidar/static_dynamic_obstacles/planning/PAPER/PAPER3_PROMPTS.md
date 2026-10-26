# Paper 3 draft: edit instructions (kept current)

> **Kept current.** Last updated 2026-10-26. This file holds the next edit instructions for the
> Paper 3 draft, each as a block to paste into the chat that writes the paper. The blocks name
> planning files by path; that chat reads them from the synced project, so each push needs a Sync
> before the next block is sent. Only files outside `planning/` are attached: the .docx draft,
> kept in `PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`. Blocks already applied move
> to the log at the end.

## 1. State of the draft

Current file: `Paper3_Sections1-4_with_placeholders_rev25Oct_tracked.docx`, checked against the
code and the results on 2026-10-26.

- **Drafted:** Sections 1–4.4, Appendices B–G, Discussion 6.2–6.3.
- **Placeholders:** Sections 4.5, 5, 6.1 and 7; Appendix A; Figures 4–6 (to insert); Algorithm A3.
- **Applied and checked:** the corrections of 2026-10-22 and 2026-10-24 and blocks C, D and E of
  2026-10-25 (log, entries 1–3). Every value of Tables 9–12 and A6–A8 matches the results files;
  the 4.2 snapshot facts match `results/paper_snapshots` (README, summary, trace) and the TQC seed 2
  resume matches its training log.
- **Markers left in Section 3 onward:** [PENDING — safety-on evaluation] (2.3.4, 4.5); [PENDING —
  safety-layer algorithm] (2.3.4, Algorithm A3); [PENDING — field-trial design] (5, 5.3); [TBC —
  replay logs] (5.2); [TBC — repository and DOI] and [CHECK — identification data] (Data
  Availability); comments 5–7.

## 2. Next: block F, the runtime safety layer, and two consistency fixes

Attachment: `Paper3_Sections1-4_with_placeholders_rev25Oct_tracked.docx`. Needs: the planning
folder pushed and synced after 2026-10-26 (Algorithm A3 in `planning/PAPER/PAPER3_ALGORITHMS.md`;
decision S10 in the skeleton). Output: `Paper3_Sections1-4_with_placeholders_rev26Oct_tracked.docx`.

The layer's figures are outside `planning/`, so the block quotes them. Checked on 2026-10-26
against `planning_records/SAFETY_V21_PLAN.md` section 15 (the decision and its table),
`results/safety_dev/testset_v4_main/reports/ts4_final/report.md` (outcomes; paired transitions
of `v21_e1_estop_bt` against `off`: 39 goals gained, 36 lost),
`planning_records/SAFETY_ESTOP_PROTOCOL.md` (target braking at 0.471 m/s², the stopped-episode
rule), and the code: `src/safety_v21.py` (`SafetyFilterV21EC`, `collision_escape_time`,
`collision_contact_times`), `src/safety_estop.py` (`continuation_bank`, `_stop`),
`src/safety_v20_contingency.py` (`WEAK`, `STRONG`), `src/safety_v2.py` (constants,
`_threat_in_reach`), `src/safety_trajectory_search.py`. The comparator name count (F8) and the
A2 row (F9) are from the 25 October draft.

```
Apply the following changes to the attached Paper3_Sections1-4_with_placeholders_rev25Oct_tracked.docx as tracked changes dated 26 October 2026 (author "Claude"), keeping every earlier tracked change and comment, and save the result as Paper3_Sections1-4_with_placeholders_rev26Oct_tracked.docx. Part 1 writes the runtime safety layer, now decided: Section 2.3.4, Algorithm A3 and Section 4.5. Part 2 makes two consistency fixes. Sources: this message (the layer's results and its evaluation record are not among the project files, so their figures are quoted here); planning/PAPER/PAPER3_ALGORITHMS.md (Algorithm A3, drafted 26 October 2026); planning/PAPER/PAPER3_DRAFT_SKELETON.md (decision S10). Add no other number. Report each change in a short revision-log addendum.

Rules for every sentence on the layer:
- Describe this one layer only. Mention no other configuration of it (no other stop window) and none of its development versions.
- Its purpose is to reduce the collisions of the own ship; a higher success rate is welcome but is not its purpose. A target that strikes the own ship after it has stopped is counted separately, not as an own-vessel collision.
- State that the layer is evaluated empirically and gives no guarantee; that it was chosen on the validation set, the test set having been used during its development; that it steers as well as stops; and that the count of target strikes on the stopped ship depends on the target-braking convention of the evaluation.

PART 1. The runtime safety layer

F1. Section 2.3.4. Keep its first three sentences ("Rule 8(e) is realized in two layers. … the only element that can reverse the propeller.") and the two sentences "When it acts, the overridden commands are recorded … externally enforced intervention." Replace the rest of the paragraph, from "When enabled, the layer fires if …" to "… [PENDING — safety-layer algorithm: Appendix, to be added once the layer is finalized]", with one or two paragraphs built from these facts (Algorithm A3 gives the order of the steps and every value):
  1. The layer has two parts. A predictive safety filter acts at every decision with a hazard in reach (a target track within 7 m, a static return within 7 m and not more than 1 m astern of the own ship's position, or the boundary within 1.5 m). It rolls out the policy's command and a bank of alternative first commands (a grid of five rudder and three propulsion settings, a command that rejoins the path, and, with a track within 7 m, full astern at three rudder settings), each held for 1 s and followed by recovery maneuvers to 8 s, on the identified vessel model with its rudder delay and servo lag. A plan passes if the hull, inflated by its 0.15 m margin, keeps gaps of 0.05 m to the static returns, 0.02 m to the map boundary and 0.10 m to target hulls moved at constant velocity, and a hull still moving at the horizon has a clear run ahead. The policy's command is kept whenever it passes with a clearance of at least 0.15 m; otherwise, the passing command closest to it is issued and its plan kept, and when no command of the bank passes, a cross-entropy search looks for a checked plan.
  2. A permanent emergency stop is tested only at a decision at which the filter has no checked plan. It rolls out every first command, each followed by every continuation of a fixed bank (hold the command; stop; turn, then stop; turn, cruise, then stop), over a window of 2.5 s and under a weak and a strong model of reverse thrust. Contact is the hull inflated by 0.15 m, the geometry by which a collision is scored, leaving the map boundary or meeting a target's hull on its constant-velocity track; remembered static returns are not part of this test. If every sequence makes contact within the window, the layer commits to full astern with the rudder centred for the rest of the episode. The stop is the layer's way of taking all way off; the filter also steers, so the layer is not only a stop.
  3. The layer has its own perception, which leaves the policy's observation unchanged: an observer of the own ship's motion built on the identified model, static returns remembered until fresh rays pass through them, target hypotheses admitted on observed motion and kept for a bounded time when a track is lost, and the hull of a partly seen target completed from its axis of motion. It uses onboard estimates only.
  4. Its banks, horizons and models are finite, so the layer gives no guarantee; it is evaluated empirically (Section 4.5). It is off during training, so nothing it does shapes the policy; every result for the learned policy outside Section 4.5 is obtained with it off, and its interventions are never attributed to the policy.
  Keep the sentence "In the field trials, the same code also serves as an operator-latched stop (Section 5)." and add to it the comment "[CHECK — which parts of the layer run on the vessel in the field trials, the whole layer or only the operator-latched stop; part of the field-trial design]"; add the same comment to the sentence of Section 5.1 that names the layer. Delete this subsection's markers [PENDING — safety-on evaluation …] and [PENDING — safety-layer algorithm …] and their marker-list entries, and point to Algorithm A3 instead.

F2. Algorithm A3, Appendix F. Replace the placeholder, including [PENDING — safety-layer algorithm], with Algorithm A3 of planning/PAPER/PAPER3_ALGORITHMS.md: its Require and Ensure lines, its 19 numbered lines and its parameter table, laid out like Algorithms A1 and A2 and written with the draft's symbols. Leave out the table row "Evaluation convention after a stop", which belongs to Section 4.5, and the "Notes for the text". Delete the marker's entry from the marker list.

F3. Section 4.5. Replace the placeholder, including [PENDING — safety-on evaluation: to be finalized and reported], with the items below, and delete the marker's entry from the marker list.
 (a) One paragraph on the evaluation, from these facts: the deployed policy (SAC seed 0) runs the 1,000 test episodes with the layer off and on, and nothing else changes; the run with the layer off reproduces the test-set run of Table 9 episode for episode. After a stop, every target brakes in a straight line at 0.471 m/s² until it is at rest, a convention of the evaluation; the episode ends as stopped once the own ship is still (at most 0.05 m/s and 1°/s) and every target has stopped; a target that still strikes the stopped own ship is counted apart from the own-vessel collisions, and that count depends on the convention. The layer was chosen on the validation set; the test set had been used during its development, so the test-set figures are not a fully held-out estimate for the layer.
 (b) Table 13, "Runtime safety layer on the test set (deployed policy; 1,000 episodes)", with two columns, layer off and layer on:
   Own-vessel collisions | 135 (0.135) | 70 (0.070)
   — with a static obstacle / the boundary / the target | 70 / 12 / 53 | 45 / 17 / 8
   — decoupled / coupled | 46 / 89 | 18 / 52
   Goals | 865 | 868
   Stopped | — | 52
   Target strikes on the stopped own ship | — | 9
   Timeouts | 0 | 1
   Decisions with a changed command | — | 0.089
   Episodes with a changed command | — | 0.36
   Table note: every episode has exactly one outcome (868 + 70 + 9 + 52 + 1 = 1,000); the changed decisions include every decision held after a stop.
 (c) Text, two paragraphs, from these facts:
   1. With the layer, the own ship collides in 70 of the 1,000 episodes instead of 135: target collisions fall from 53 to 8 and obstacle collisions from 70 to 45, whereas boundary collisions rise from 12 to 17; in the decoupled family 46 become 18, and in the coupled family 89 become 52.
   2. Paired by episode: of the 135 episodes that end in an own-vessel collision without the layer, 39 reach the goal with it, 32 end stopped, 3 end with the target striking the stopped ship, and 61 still end in an own-vessel collision. Of the 865 goals without the layer, 829 remain goals; 20 become stops, 9 own-vessel collisions, 6 target strikes on the stopped ship, and 1 a timeout. Goals change little (865 to 868); the layer's purpose is fewer collisions, not more goals.
   3. The layer changes the command at 8.9 % of the decisions (including every decision held after a stop) and in 36 % of the episodes.
   4. On the validation set (270 episodes), on which the layer was chosen, own-vessel collisions fall from 53 to 24 and goals rise from 217 to 222, with 23 stops; the layer changes 13.5 % of the decisions, and 4.6 % outside the episodes that end in a stop.
   5. Give no explanation of individual outcomes beyond these figures.

F4. Section 1.3, training and evaluation paragraph. Replace "it is off during training and in every result reported for the learned policy, and its interventions are not attributed to the policy." with "it is off during training and in every result reported for the learned policy except the evaluation of the layer itself in Section 4.5, and its interventions are not attributed to the policy."

F5. Algorithm 1 (Section 2.2.4), Ensure line. Replace "(the context branch of the observation, the gates of the reward, and the trigger of the runtime safety layer)" with "(the context branch of the observation and the gates of the reward)". The layer of Section 2.3.4 does not read the encounter assessment.

F6. Section 2.2.4, the paragraph on the stopping test. Replace "The policy itself cannot reverse, so the reward credits a slowdown whose completion, if required, the runtime safety layer provides." with "The policy itself cannot reverse, so the reward credits a slowdown that a full-astern stop would complete; only the runtime safety layer can command that stop (Section 2.3.4)."

F7. Section 6.2, evaluation paragraph. After the sentence on the training budget, insert: "The runtime safety layer was chosen on the validation set, but the test set had been used during its development, so its test-set figures (Section 4.5) are not a fully held-out estimate; its finite banks and models give no guarantee, and the count of target strikes on the stopped own ship depends on the convention that targets brake once the stop is committed."

PART 2. Consistency fixes

F8. Comparator name. Section 3.3 and Appendix F name the first comparator LOS-DWA. Replace every "LOS-PID + DWA" (Section 4, its tables and notes, Appendix G, and Section 6) with "LOS-DWA", so that one name is used after its definition in Section 3.3. Leave the descriptive phrases of Sections 1.2.1 and 1.3 ("line-of-sight guidance with … a dynamic-window avoidance layer") as they are.

F9. Algorithm A2, parameter row "Tuning". Replace "scored the same as the defaults on all 120 decoupled validation episodes" with "scored the same as the defaults on all 120 decoupled validation episodes, in the earlier build with three panels in 29 rather than 53 of them (Section 3.3)".

F10. Header line and marker list: bring them up to date with F1–F9.
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
