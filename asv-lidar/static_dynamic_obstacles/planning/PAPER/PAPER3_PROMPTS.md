# Paper 3 draft: edit instructions (kept current)

> **Kept current.** Last updated 2026-10-25. This file holds the next edit instructions for the
> Paper 3 draft, each as a block to paste into the chat that writes the paper. The blocks name
> planning files by path; that chat reads them from the synced project, so each push needs a Sync
> before the next block is sent. Only files outside `planning/` are attached: the .docx draft,
> kept in `PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`. Blocks already applied move
> to the log at the end.

## 1. State of the draft

Current file: `Paper3_Sections1-3_with_placeholders_rev24Oct_tracked.docx`, checked against the
code on 2026-10-25.

- **Drafted:** Sections 1–3, Appendices B–F, Discussion 6.2–6.3.
- **Placeholders:** Sections 4, 5, 6.1 and 7; Appendices A and G.
- **Applied and checked:** the ten corrections of 2026-10-22 and blocks A and B of 2026-10-24
  (log, entries 1–2). Six comments, all kept.
- **Campaign:** all twelve runs finished 2026-10-25 (commit 7ab33e90); the deployed policy is SAC
  seed 0 by the rule of Section 3.2. The reward scale audit and comparators v2 finished the same
  day; blocks D and E need them pushed.
- **Markers left in Section 3 onward:** [PENDING — comparators v2 tuning] (3.3); [PENDING — reward
  scale audit of the deployed policy] (4.4); [PENDING — safety-on evaluation] (4.5); [PENDING —
  field-trial design] (5, 5.3); [TBC — replay logs] (5.2); [TBC — repository and DOI] and [CHECK —
  identification data] (Data Availability); [PENDING — comparators v2] (Algorithm A1); [PENDING —
  safety-layer algorithm] (Algorithm A3); comments 5–7 (reference layouts; reference 76 pages;
  reference 72 publisher location).

## 2. Next: blocks C, D and E, Section 4 from the twelve runs, the reward scale audit and comparators v2

Attachment: `Paper3_Sections1-3_with_placeholders_rev24Oct_tracked.docx`. Needs: the planning folder
synced at commit 7ab33e90 or later (all twelve runs in `planning/MASTER_ARTIFACT_RESULTS.md` and
`planning/master_artifact_results.json`). Output:
`Paper3_Sections1-4_with_placeholders_rev25Oct_tracked.docx`.

The values in the block were taken from those two files on 2026-10-25 and checked against them.
The snapshot facts of C3 come from `results/paper_snapshots/README.txt` and
`paper_snapshots_summary.csv` (outside `planning/`, so quoted). The snapshot figure was made with
`runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip`, the same file (SHA-1 0800455d…) as
the deployed policy's `best_model.zip`. The resume of TQC seed 2 (C6) is in
`runs/tqc_formulation_seed2_bl3.log` ("[RESUME] … from 750,000"; "replay buffer restored"). The
note on the 3.0 M check (C2 d) follows `src/train_formulation.py`: the periodic check runs in the
step callback, before the last update; `_on_training_end` makes the end check.

```
Apply the following changes to the attached Paper3_Sections1-3_with_placeholders_rev24Oct_tracked.docx as tracked changes dated 25 October 2026 (author "Claude"), keeping every earlier tracked change and comment, and save the result as Paper3_Sections1-4_with_placeholders_rev25Oct_tracked.docx. The main part drafts Section 4 (its introduction, 4.1, 4.2 and 4.3) and Appendix G from the twelve finished runs; Sections 4.4 and 4.5 stay placeholders, and the encounter-specific VO stays pending everywhere until its comparators v2 run. Report each change in a short revision-log addendum, and list there every number written into Section 4 with its source.

Sources, in this order: this message; planning/MASTER_ARTIFACT_RESULTS.md (generated 2026-10-25 from all twelve runs, sections 2–14); planning/master_artifact_results.json, for the deployed run's own values (in "runs", the entry with learner "sac" and seed 0) and the paired statistics ("paired_vs_first", where "SAC" is seed 0, the deployed run); planning/PAPER/PAPER3_DRAFT_SKELETON.md (decisions S11, S15, S16 and S19, and its restated Section 4 plan). Take every number from these. Add no result that they do not hold, and give no causal explanation beyond those stated below.

Conventions for all of Section 4:
- Learners are three-seed values at the best validation-set checkpoint of each run. Rates are pooled over the three seeds, which equals the seed mean; continuous metrics are interquartile means (IQM) over the episodes [60]; Appendix G gives 95 % intervals from a stratified bootstrap over the test-set cells (5,000 resamples). The seed range is given for success.
- The deployed policy, SAC seed 0, is one run and is shown beside the learner pools. LOS-PID + DWA is deterministic and runs once.
- Rates as fractions with three decimals, distances in meters with two, times in seconds with one.
- The encounter-specific VO is a column marked [PENDING — comparators v2] in every table, and the text says once that its results follow its tuning (Section 3.3). At the first LOS-PID + DWA column, add the comment "[CHECK — LOS-PID + DWA is re-run under comparators v2; values expected unchanged]".
- Number the new items after the draft's: Tables 9–11, Figures 4–6, Appendix Tables A6–A8.
- Keep the terms of Sections 1–3 (COLREGs-aware, decoupled and coupled, validation set, test set, required turn sense, rule-consistent) and report no per-rule compliance rates (Section 1.3).

C1. Section 4, opening placeholder. Replace it with one paragraph that states what Section 4 reports (4.1 overall results, 4.2 encounter snapshots, 4.3 target behaviors, 4.4 reward scale audit, 4.5 runtime safety layer), that the safety layer is off everywhere except in Section 4.5, and the conventions above, and that gives the deployment outcome by the rule of Section 3.2: SAC's three-seed mean success (0.846) exceeds TQC's (0.830), so SAC is the deployed learner; within SAC, seed 0 has the best validation score (+0.411, against +0.300 for seed 1 and +0.367 for seed 2), so SAC seed 0 is the deployed policy. Its test-set success, 0.865, is also the highest of the three SAC seeds (0.827 and 0.845); its seed was chosen on the validation set and its learner on the test set (Section 3.2). Delete the sentence "The tables are fixed before the results are known."

C2. Section 4.1. Replace the placeholder with the items below.
 (a) Table 9, "Principal results on the test set (1,000 episodes; safety layer off)", with metrics as rows and methods as columns:
   Metric | Deployed policy (SAC seed 0) | SAC | TQC | PPO | Recurrent PPO | LOS-PID + DWA | Encounter-specific VO
   Success | 0.865 | 0.846 | 0.830 | 0.750 | 0.717 | 0.770 | [PENDING]
   Success, decoupled (664 episodes) | 0.931 | 0.901 | 0.891 | 0.822 | 0.799 | 0.870 |
   Success, coupled (336 episodes) | 0.735 | 0.737 | 0.709 | 0.608 | 0.555 | 0.571 |
   Target collision | 0.053 | 0.063 | 0.067 | 0.154 | 0.184 | 0.080 |
   Obstacle collision | 0.070 | 0.074 | 0.082 | 0.066 | 0.065 | 0.109 |
   Boundary collision | 0.012 | 0.017 | 0.021 | 0.030 | 0.035 | 0.038 |
   Ship-domain intrusion | 0.087 | 0.112 | 0.125 | 0.239 | 0.251 | 0.142 |
   Minimum hull-to-hull clearance to the target (m) | 1.69 | 1.62 | 1.52 | 1.30 | 1.21 | 1.33 |
   RMS cross-track error (m) | 1.42 | 1.37 | 1.27 | 0.99 | 0.90 | 0.51 |
   Completion time, successful episodes (s) | 26.2 | 25.8 | 27.3 | 30.1 | 31.0 | 36.5 |
   Table note: success and collision rates are fractions of the episodes (LOS-PID + DWA also times out in 0.003 of them; no learned policy times out); the other rows are interquartile means; the RMS cross-track error is taken over all episodes, so a collision that ends an episode early can lower it; the target-ship rows cover the episodes with a target, and a ship-domain intrusion is the target's center entering the own ship's domain (3.14 m ahead, 1.57 m astern, 1.25 m abeam); clearances are measured from the hull polygon with its 0.15 m margin; Appendix G gives every metric with its 95 % interval.
 (b) Text, three or four paragraphs, built from these facts:
   1. Learner families: the off-policy learners lead, SAC 0.846 (seed range 0.827–0.865) and TQC 0.830 (0.815–0.838), against PPO 0.750 (0.747–0.753) and recurrent PPO 0.717 (0.638–0.757). Every off-policy seed succeeds more often than every on-policy seed (lowest off-policy seed 0.815, highest on-policy seed 0.757). The SAC and TQC seed ranges overlap, so the 0.016 between their means lies within the seed spread; the deployment rule decides by the mean (Section 3.2), and the text claims no further ranking between them.
   2. The gap between the families lies mainly in target collisions (0.063 and 0.067 for SAC and TQC; 0.154 and 0.184 for PPO and recurrent PPO) and ship-domain intrusions (0.112 and 0.125; 0.239 and 0.251); obstacle collisions are similar for all four learners (0.065–0.082).
   3. The deployed policy against LOS-PID + DWA on the same 1,000 episodes: success 0.865 against 0.770; 176 episodes are completed only by the deployed policy and 81 only by LOS-PID + DWA (exact McNemar test, p = 3.0 × 10⁻⁹); decoupled 0.931 against 0.870 (75 against 35, p = 1.7 × 10⁻⁴); coupled 0.735 against 0.571 (101 against 46, p = 6.7 × 10⁻⁶). It also collides with the target less often (0.053 against 0.080), enters the target's domain less often (0.087 against 0.142) and keeps a larger hull clearance (1.69 against 1.33 m). Report paired tests for this pair only; the source pairs the other learners by their seed 0 runs, which has no meaning for a three-seed comparison.
   4. Coupled scenarios are harder for every method (deployed policy 0.931 decoupled, 0.735 coupled; LOS-PID + DWA 0.870 and 0.571), and in them obstacle collisions become the main failure (deployed policy: 0.167 obstacle and 0.077 target collisions over the 336 coupled episodes). In the decoupled family, success in basin and in channel encounters differs by at most 0.03 for every method (SAC 0.905 and 0.893; LOS-PID + DWA 0.880 and 0.854; MASTER_ARTIFACT_RESULTS section 8).
   5. The learned policies travel faster than the cruise speed of 0.558 m/s and finish sooner (SAC: mean speed 0.749 m/s, completion 25.8 s), and LOS-PID + DWA travels slower (0.495 m/s, 36.5 s), with a smaller cross-track error (0.51 m against 1.37 m for SAC). The rudder saturation fraction is 0.050 for SAC, 0.110 for TQC, 0.303 for PPO, 0.342 for recurrent PPO and 0.279 for LOS-PID + DWA (Appendix G).
   6. By encounter type (Figure 4; MASTER_ARTIFACT_RESULTS section 4), crossings are the weakest type for every method: SAC 0.741 for crossings from port and 0.743 from starboard, against 0.859–0.901 for head-on, overtaking and being overtaken; LOS-PID + DWA 0.699 and 0.630. The deployed policy completes 0.914 of the head-on, 0.735 of the port-crossing, 0.778 of the starboard-crossing, 0.877 of the overtaking and 0.943 of the being-overtaken episodes. Its weakest case is the coupled crossing from port: 19 of 51 episodes (0.373), with 16 target and 15 obstacle collisions; over the three SAC seeds this case reaches 0.490 (seed range 0.37–0.67).
 (c) Figure 4 placeholder: "[Figure 4 placeholder] Success by encounter type on the test set (head-on, crossing from port, crossing from starboard, overtaking, being overtaken, null, no target): grouped bars, learners as the three-seed mean with the individual seeds and the seed range, LOS-PID + DWA hatched without an interval; the encounter-specific VO to be added after comparators v2."
 (d) Table 10, "Test-set success by run: the best validation-set checkpoint and the final model", from MASTER_ARTIFACT_RESULTS sections 2 and 3: one row per learner and seed, with the step of the best validation check, the validation score, the test-set success with its decoupled and coupled parts, and the final model's success with its parts; below each learner's three rows, its mean with the seed range. Where the best check is the check made after the last update, write "end" for the step and "same model" for the final model. Table note: "A check at 3.0 × 10⁶ steps runs before the last policy update, so its model can differ slightly from the final model."
 (e) Text, one paragraph, from these facts: the final model changes a run's success by at most 0.039 (PPO seed 0, 0.751 to 0.712) and an off-policy run's by at most 0.008; the learner order is unchanged, and the deployment rule would pick SAC from the final models as well (means 0.841 for SAC and 0.830 for TQC); this was the selection-sensitivity check decided before any final model was tested (Section 3.2). Recurrent PPO seed 1 reaches 0.638, against 0.757 and 0.756 for seeds 0 and 2: between 1.0 × 10⁶ and 2.4 × 10⁶ steps, 258 of its 273 updates stopped at the first epoch on the KL limit and its validation score fell to −0.91 at 2.0 × 10⁶ steps before it recovered; no defect was found, so the seed is reported, not replaced (Section 3.2).
 (f) Figure 5 placeholder and one or two sentences: "[Figure 5 placeholder] Validation-set score (goal rate − 2 × collision rate) every 2 × 10⁵ steps for the four learners: three-seed mean, individual seeds and seed-range band, with the curriculum stage boundaries marked." Facts (MASTER_ARTIFACT_RESULTS section 14): every SAC and TQC run reaches its best validation score at 2.8 × 10⁶ steps or later and scores higher at 3.0 × 10⁶ than at 2.0 × 10⁶ steps; the best scores are +0.300 to +0.411 for SAC, +0.267 to +0.378 for TQC, −0.033 to +0.100 for PPO and −0.289 to +0.033 for recurrent PPO.
 (g) Delete the placeholder's mention of "the cumulative distribution of the closest approach": no such figure is made, and the closest approach is a row of Appendix G.

C3. Section 4.2. Replace the placeholder with the items below.
 (a) Figure 6 placeholder, with the caption: "Trajectories of the deployed policy (SAC seed 0) in six test episodes, one per encounter type: (a) no target, (b) head-on, (c) crossing from port, (d) crossing from starboard, (e) overtaking, and (f) being overtaken. Panels (a), (b), and (d) are coupled: (a) a generated layout without a target, and (b) and (d) two of the three fixed reference layouts, where the encounter takes place beside the panels. Panels (c), (e), and (f) are decoupled, with the panels clear of the encounter (a channel and the basin). The layouts also differ in the number of panels and the leg shape. Own-ship (blue) and target (red) hulls are drawn every 5 s; equal numbers mark the same instant, and later hulls are darker. The dotted segment marks the closest point of approach. The target of panel (f) is reactive (Section 3.4.2); the other targets keep their course and do not give way, and the target of panel (d) changes speed once."
 (b) Text, one or two paragraphs, from these facts. (a) No target, coupled layout with three panels crowding a slanted leg: passes to one side of all three and rejoins the path near the goal (26.0 s). (b) Head-on, reference layout, slanted leg, three panels: alters about 48° to starboard (first lateral departure 3.2 m to starboard), passes port to port with a CPA of 4.27 m at 13.0 s, and returns to the path. (c) Crossing from port, channel, straight leg, one panel on the path: turns about 40° to port and slows to 0.29 m/s, passes astern of the target (CPA 2.53 m at 8.5 s), then clears the panel. (d) Crossing from starboard, reference layout, slanted leg, three panels, a target that changes speed: turns about 50° to starboard and passes astern (CPA 3.02 m at 7.0 s). (e) Overtaking, basin, straight leg, one panel on the path: moves out to port of the slower target, passes it (CPA 3.04 m at 18.0 s), and regains the path. (f) Being overtaken, basin, slanted leg, no panels, a reactive target (Section 3.4.2): the own ship keeps within 0.15 m of its track and within 0.15 m/s of its speed until the target is within 2.5 m, its speed rising from 0.48 to 0.62 m/s in that time; as the target passes on its port side (CPA 2.22 m at 10.0 s), the own ship turns about 20° and slows to about 0.40 m/s, and once the target is past it regains the path and speeds up. Each encounter panel passes the track-based rule conditions of Section 3.4.3: no crossing ahead of a crossing target, the alteration to the required side, the passing side at the CPA, and, when being overtaken, course and speed held within 0.15 m and 0.15 m/s until the target is within 2.5 m. In panel (f), the speed change (+0.14 m/s) is inside that band but above the 0.10 m/s at which the speed part of the hold term grows linearly (Eq. (27)), and the turn and the slowing come once the target is close; both reflect the competition between stand-on course keeping and the ship-domain term stated in Section 6.2. Describe panel (f) in these terms, not as course and speed held throughout. The figure is a selection of successful episodes, not a sample: the deployed policy completes 865 of the 1,000 episodes, and the rates are those of Table 9.

C4. Section 4.3. Replace the placeholder with the items below.
 (a) Table 11, "Success by target behavior (test set; safety layer off)", rows as follows, with n the episodes per run and the learners as three-seed means:
   Part and target behavior | n | Deployed policy (SAC seed 0) | SAC | TQC | PPO | Recurrent PPO | LOS-PID + DWA | Encounter-specific VO
   Decoupled, constant velocity | 340 | 0.941 | 0.903 | 0.885 | 0.775 | 0.755 | 0.835 | [PENDING]
   Decoupled, reactive | 258 | 0.922 | 0.901 | 0.895 | 0.877 | 0.823 | 0.888 |
   Decoupled head-on, constant velocity | 66 | 0.985 | 0.975 | 0.934 | 0.783 | 0.818 | 0.818 |
   Decoupled head-on, reactive | 66 | 0.924 | 0.919 | 0.914 | 0.924 | 0.884 | 0.939 |
   Decoupled head-on, noncompliant | 66 | 0.909 | 0.889 | 0.904 | 0.854 | 0.934 | 0.985 |
   Coupled, constant speed | 156 | 0.808 | 0.776 | 0.724 | 0.613 | 0.528 | 0.564 |
   Coupled, one speed change | 150 | 0.620 | 0.662 | 0.653 | 0.556 | 0.524 | 0.553 |
   Coupled, no target | 30 | 0.933 | 0.911 | 0.911 | 0.844 | 0.844 | 0.700 |
   Table note: noncompliant targets occur only in head-on encounters, and the 340 and 258 decoupled episodes include the head-on rows' constant-velocity and reactive cases; the reactive target follows the velocity-obstacle rule of Kuwata et al. [19] from its own side (Section 3.4.2). Give a seed range in the text only where one is stated below.
 (b) Text, two or three paragraphs, from these facts:
   1. In the decoupled part, reactive targets leave the off-policy learners' success unchanged (SAC 0.903 with constant-velocity and 0.901 with reactive targets; TQC 0.885 and 0.895; the deployed policy 0.941 and 0.922), whereas PPO (0.775 to 0.877), recurrent PPO (0.755 to 0.823) and LOS-PID + DWA (0.835 to 0.888) succeed more often against reactive targets, which also maneuver to avoid the own ship (Section 3.4.2).
   2. The head-on rows hold 66 episodes per run, so one episode is 0.015 of a row and the seed ranges are wide (SAC against noncompliant targets: 0.79–0.97); differences below about 0.05 are not interpreted. Against the noncompliant head-on target, which turns to port, the deployed policy completes 60 of 66 episodes (0.909; five target collisions), against 65 of 66 (0.985) with a constant-velocity target. LOS-PID + DWA completes 0.985 against noncompliant and 0.818 against constant-velocity head-on targets.
   3. In the coupled part, a target that changes speed once lowers success for every method, most for the off-policy learners: the deployed policy 0.808 to 0.620, SAC 0.776 to 0.662, TQC 0.724 to 0.653, PPO 0.613 to 0.556, recurrent PPO 0.528 to 0.524, LOS-PID + DWA 0.564 to 0.553. In training, 30 % of the coupled targets of stage 7 changed speed once (Section 2.3.5). Of the deployed policy's 150 episodes with a speed change, 35 end in an obstacle collision, 16 in a target collision and 6 at the boundary.
   4. In the 30 coupled layouts without a target, the learners complete 0.844–0.933 (deployed policy 0.933) and LOS-PID + DWA 0.700.

C5. Appendix G. Replace the placeholder with Tables A6 (all 1,000 episodes), A7 (decoupled, 664) and A8 (coupled, 336): the three tables of planning/MASTER_ARTIFACT_RESULTS.md, section 13, with every row and its interval, the methods SAC, TQC, PPO, recurrent PPO and LOS-PID + DWA as columns, and a column for the encounter-specific VO marked [PENDING — comparators v2]. Give the metric names in words ("Mean absolute rudder rate (°/s)", "Control effort (integrated squared rudder command)", "Deepest domain intrusion, episodes with an intrusion (fraction of the domain radius)"). Open the appendix with one sentence pointing to Table 9 for the deployed policy's own values. Table note (once, under Table A6, and referred to by A7 and A8): point estimates with 95 % intervals in brackets; rates are means of per-episode outcomes, pooled over the three seeds; the other rows are interquartile means [60]; intervals from a stratified bootstrap over the test-set cells, pooled over runs, 5,000 resamples; LOS-PID + DWA is deterministic, so its intervals reflect episode variance only; completion time and path length cover the successful episodes, because a collision truncates an episode, and the RMS cross-track error covers all episodes; target-ship rows cover the episodes with a target, and intrusion depth and time cover the episodes with an intrusion; clearances are measured from the hull polygon with its 0.15 m margin; a path-length ratio below 1 reflects the goal test (within 1.25 m of the path end) and cut corners.

C6. Section 3.2. Replace "seeds 1 and 2 were each trained in one run to 3.0 × 10⁶ steps with the same stage timing." with "seeds 1 and 2 were each trained in one run to 3.0 × 10⁶ steps with the same stage timing, except that TQC seed 2 was interrupted once and resumed from its 7.5 × 10⁵-step checkpoint with its replay buffer."

C7. Section 3.4.3. After "…require an explicit statement of their uncertainty [60,61].", insert: "Rates are pooled over the three seeds, which equals their mean; continuous metrics are interquartile means over the episodes [60]; and Appendix G gives 95 % intervals from a stratified bootstrap over the test-set cells."

C8. Section 6.2, evaluation paragraph. After "…so small differences between learners are not taken to establish a ranking (Section 1.2.5).", insert: "The training budget was fixed at 3.0 × 10⁶ steps for every learner, and every SAC and TQC run reached its best validation score at 2.8 × 10⁶ steps or later (Section 4.1), so a longer budget might change their results."

C9. Section 1.2.2. Replace "Making this split explicit allows the relationship between supplied context, reward incentives, and executed maneuvers to be examined rather than assumed." with "Making this split explicit states which part of the decision context is engineered and which part is learned." (The paper reports no ablation, so it does not examine that relationship.)

C10. Marker list. Add "[PENDING — comparators v2]: Section 4 (the encounter-specific VO column of Tables 9, 11 and A6–A8, and Figure 4)" and "[CHECK — LOS-PID + DWA re-run under comparators v2]: Table 9". Note under the list that Figures 4–6 are placeholders for figures to insert.
```


### Block D: Section 4.4, the reward scale audit of the deployed policy

Source: `results/baseline_v3/reward_scale_audit.md` (run `sacs0_bl3_audit`, 2026-10-25; outside
`planning/`, so the values are quoted in the block). Checked on 2026-10-25: the eight dense
integrals and the terminal mean add up to the mean return (−72.12 + 46.00 = −26.12), and the
terminal mean equals (865 × 100 − 135 × 300)/1,000. The domain term is active in 84 episodes, which
over the 970 episodes with a target is the 0.087 intrusion rate of Table 9. Send block D after
block C, or append it to the same message; Table 12 follows Tables 9–11.

```
Apply the following change as tracked changes dated 25 October 2026 (author "Claude"), keeping every earlier tracked change and comment, and report it in the revision-log addendum. The values come from the reward scale audit of the deployed policy (SAC seed 0, its best validation-set checkpoint, on the 1,000 test episodes, safety layer off); they are quoted here because the result file is not among the project files. Add no other number.

D1. Section 4.4 "Reward scale audit". Replace the placeholder, including [PENDING — reward scale audit of the deployed policy], with the items below, and delete that marker's entry from the marker list.
 (a) Table 12, "Reward scale audit of the deployed policy on the test set (1,000 episodes)", terms in the order of Table 7's weights, largest first. Columns: weight (the largest per-step contribution, Table 7); mean weighted episode integral; mean share of the episode's absolute dense reward; share of episodes in which the term is ever nonzero; mean magnitude per step on the steps where the term is nonzero.
   Boundary r_bnd | 15.0 | −1.25 | 0.004 | 0.034 | 5.36
   Ship domain r_dom | 12.5 | −0.17 | 0.002 | 0.084 | 0.47
   Obstacles r_obs | 11.0 | −20.60 | 0.094 | 0.616 | 2.06
   COLREGs group r_col | 9.0 | −26.86 | 0.126 | 0.840 | 2.34
   Path following r_pf | 3.0 | −66.39 | 0.306 | 1.000 | 1.37
   Progress r_prog | 1.5 | +69.71 | 0.338 | 1.000 | 1.45
   Smoothness r_smooth | 0.5 | −14.62 | 0.071 | 1.000 | 0.30
   Existence r_exist | 0.25 | −11.94 | 0.058 | 1.000 | 0.25
   Terminal r_term (+100 goal, −300 collision) | — | +46.00 | — | 1.000 | —
   Episode return | — | −26.12 | — | — | —
   Table note: the run reproduces the test-set run of Table 9 episode for episode (outcome and length identical in all 1,000 episodes), and the terms add up to every episode's return (largest difference 1.3 × 10⁻¹³). The terminal mean reflects 865 goals and 135 collisions.
 (b) Text, two paragraphs, from these facts:
   1. Per active step, the boundary term stays the largest (5.36 of its weight of 15.0), followed by the COLREGs group (2.34 of 9.0) and the obstacle term (2.06 of 11.0); path following (1.37) and progress (1.45) come next, and smoothness and existence last. The ship-domain term realizes 0.47 of its 12.5 and is nonzero in 8.4 % of the episodes, so it falls from second to sixth: the deployed policy enters the target's domain rarely and shallowly (Table 9). The realized order therefore departs from the order of the weights in three places: the domain term, the COLREGs group above the obstacle term, and progress above path following (1.45 against 1.37); no term exceeds its weight, as the bounded design ensures (Section 2.3.3).
   2. The swap depends on the scenario: in the 336 coupled episodes, the obstacle term is nonzero in 97.3 % of the episodes and its per-step magnitude (2.44) exceeds the COLREGs group's (2.35), whereas in the 664 decoupled episodes it is nonzero in 43.5 %.
   3. Over whole episodes, path following and progress make up most of the dense reward (shares 0.306 and 0.338) and nearly cancel (−66.39 and +69.71); the COLREGs group and the obstacle term make up 0.126 and 0.094, and the boundary and domain terms together less than 0.01.
   4. The terminal reward sets the sign of the return: goal episodes return +26.16 on average and collision episodes −361.11. In the collision episodes, the COLREGs group's mean integral is −24.67, so the −300 of a collision outweighs it by an order of magnitude, as the hierarchy of Section 2.3.3 requires (a rule-consistent collision scores worse than a near miss that departs from the rules).
   5. Against the failure of the earlier formulation recalled in Section 2.3.3, where the obstacle term was about 49 times weaker than the path-following term at contact distance [13], the obstacle term here is larger than the path-following term on the steps where it acts (2.06 against 1.37 per step; 3.32 in the collision episodes).
   6. The existence term equals its weight on every step on which it applies, by construction.
```

### Block E: comparators v2 (the encounter-specific VO, Algorithm A1, LOS-PID + DWA re-run)

Needs: the planning folder pushed and synced after 2026-10-25 09:44, so that
`planning/MASTER_ARTIFACT_RESULTS.md` holds the encounter-specific VO (its section 13 column
"Encounter-VO") and `planning/PAPER/PAPER3_ALGORITHMS.md` holds Algorithm A1. Applies on top of
blocks C and D; when none of the three has been sent, send C, D and E in one message.

Checked on 2026-10-25 against `results/test_set/v4/encounter_vo_v2/summary.txt`,
`planning/MASTER_ARTIFACT_RESULTS.md` (sections 2, 4, 7–13), `planning/master_artifact_results.json`
(`comparators`, `paired_vs_first`), `results/classical_tuning/encounter_vo/` (`best.json`,
`validation.csv`, `side_rule_check.txt`), `results/classical_tuning/encounter_vo.log` (the
grid), `tools/diagnostics/tune_classical.py` (the two validation builds: 53 against 29 of 120
episodes with three panels) and `src/classical/encounter_vo.py`.

```
Apply the following changes as tracked changes dated 25 October 2026 (author "Claude"), keeping every earlier tracked change and comment, and report them in the revision-log addendum. They bring in the encounter-specific VO, now built and run on the test set (comparators v2; decisions S18 and A35 in planning/PAPER/PAPER3_DRAFT_SKELETON.md), and the re-run of LOS-PID + DWA under the same settings file. Sources: this message; planning/MASTER_ARTIFACT_RESULTS.md (generated 2026-10-25 09:44 or later: its "Encounter-specific VO (v2)" rows and the "Encounter-VO" column of section 13); planning/master_artifact_results.json ("paired_vs_first", the rows "SAC vs Encounter-VO", where SAC is seed 0, the deployed run); planning/PAPER/PAPER3_ALGORITHMS.md (Algorithm A1, drafted 2026-10-25). Wherever Section 4 or Appendix G marks the encounter-specific VO [PENDING — comparators v2], replace the marker with its values; add no other number.

E1. LOS-PID + DWA. Delete the comment "[CHECK — LOS-PID + DWA is re-run under comparators v2; values expected unchanged]" and its marker-list entry: the re-run under the comparators v2 settings reproduces every episode of the first run (outcome, length, and return), so its values stand.

E2. Section 3.3, tuning paragraph.
 (a) Replace "For LOS-DWA, the search's preferred setting scored the same as the defaults on all 120 episodes, so the defaults are kept (Algorithm A2)." with "For LOS-DWA, the search ran on an earlier build of the same 120 episodes, with the same encounters but three panels in 29 of them rather than 53; its preferred setting scored the same as the defaults on all 120, so the defaults are kept (Algorithm A2)."
 (b) Replace "The tuned values of the encounter-specific VO are [PENDING — comparators v2 tuning]." with "For the encounter-specific VO, the search covered the prediction horizon (12, 20, and 30 s), the hard gap (0.20, 0.35, and 0.50 m), the stand-on release time (8, 12, and 18 s), and the weights on course changes between decisions (0, 0.3, and 0.8) and on domain intrusion (1.5, 3, and 6); only the course-change weight moved, from 0.3 to 0.8, which raised the score on all 120 episodes from +0.350 to +0.475 (success 0.783 to 0.825). Algorithm A1 lists every value." Delete the marker's entry from the marker list.

E3. Appendix F, Algorithm A1. Replace the placeholder, including [PENDING — comparators v2], with Algorithm A1 of planning/PAPER/PAPER3_ALGORITHMS.md: its Require and Ensure lines, its 29 numbered lines and its parameter table, laid out like Algorithm A2 and written with the draft's symbols (Uref for U_nom and ψp for ψ_path, as in Algorithms 1–4 and A2). Its "Notes for the text" are not part of the box. Delete the marker's entry from the marker list. Section 3.3's description of the method already agrees with the box; leave it unchanged.

E4. Table 9, encounter-specific VO column, in the order of the rows: success 0.675; success, decoupled 0.774; success, coupled 0.479; target collision 0.116; obstacle collision 0.130; boundary collision 0.079; ship-domain intrusion 0.177; minimum hull-to-hull clearance to the target 0.96 m; RMS cross-track error 0.45 m; completion time 35.2 s. The encounter-specific VO does not time out.

E5. Section 4.1 text.
 (a) Replace the sentence saying that the encounter-specific VO's results follow its tuning with a paragraph built from these facts:
   1. The encounter-specific VO completes 0.675 of the episodes (decoupled 0.774, coupled 0.479), the lowest success in Table 9 and below LOS-PID + DWA (0.770), which knows no rules.
   2. On the same episodes, 249 are completed only by the deployed policy and 59 only by the encounter-specific VO (exact McNemar test, p = 6.3 × 10⁻²⁹); decoupled 133 against 29 (p = 4.1 × 10⁻¹⁷); coupled 116 against 30 (p = 3.9 × 10⁻¹³).
   3. It has the highest boundary-collision rate of all methods (0.079), the smallest hull-to-hull clearance to the target (0.96 m) and the smallest closest approach (2.05 m, Appendix G), with target collisions at 0.116 and ship-domain intrusions at 0.177; it keeps closest to the path (RMS cross-track error 0.45 m).
   4. By encounter type it completes 0.811 of the head-on, 0.548 of the port-crossing, 0.496 of the starboard-crossing, 0.521 of the overtaking and 0.814 of the being-overtaken episodes; its weakest case is coupled overtaking, 40 of 115 episodes (0.348), with 44 obstacle and 28 boundary collisions.
   5. Give no reason for these results beyond what Section 3.3 states about the method: Section 6.2 already states that they describe this study's interpretation of the method, tuned here.
 (b) Where Section 4.1 says that success in basin and in channel encounters differs by at most 0.03 for every method, make it "for every method except the encounter-specific VO (0.800 in the basin and 0.729 in channels)".
 (c) Where Section 4.1 says that crossings are the weakest encounter type for every method, make it "for every learner and for LOS-PID + DWA", and add that the encounter-specific VO is weakest in crossings from starboard (0.496) and in overtaking (0.521).

E6. Figure 4 placeholder. Show the encounter-specific VO hatched like LOS-PID + DWA, without an interval, and delete "the encounter-specific VO to be added after comparators v2".

E7. Table 11, encounter-specific VO column, in the order of the rows: decoupled constant velocity 0.726; decoupled reactive 0.787; decoupled head-on constant velocity 0.773; head-on reactive 0.909; head-on noncompliant 0.970; coupled constant speed 0.455; coupled one speed change 0.427; coupled no target 0.867. In the Section 4.3 text: add the encounter-specific VO (0.726 to 0.787) to the methods that succeed more often against reactive targets; give its head-on results beside those of LOS-PID + DWA (0.970 against noncompliant and 0.773 against constant-velocity targets); add it to the coupled speed-change comparison (0.455 to 0.427); and give its 0.867 in the coupled layouts without a target.

E8. Appendix G. Fill the encounter-specific VO column of Tables A6–A8 from the "Encounter-VO" column of planning/MASTER_ARTIFACT_RESULTS.md, section 13, with every row and its interval; it is deterministic, so its intervals, like LOS-PID + DWA's, reflect episode variance only.

E9. Marker list. Delete every remaining [PENDING — comparators v2] entry (Section 4 and Algorithm A1) and the [PENDING — comparators v2 tuning] entry.
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
