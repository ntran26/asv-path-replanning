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
- **Markers left in Section 3 onward:** [PENDING — comparators v2 tuning] (3.3); [PENDING — reward
  scale audit of the deployed policy] (4.4); [PENDING — safety-on evaluation] (4.5); [PENDING —
  field-trial design] (5, 5.3); [TBC — replay logs] (5.2); [TBC — repository and DOI] and [CHECK —
  identification data] (Data Availability); [PENDING — comparators v2] (Algorithm A1); [PENDING —
  safety-layer algorithm] (Algorithm A3); comments 5–7 (reference layouts; reference 76 pages;
  reference 72 publisher location).

## 2. Next: block C, Section 4 (to be written)

Written once TQC seed 2 has its test-set run (expected 2026-10-25), so the deployed learner is
known and every paragraph can name it. It drafts 4.1 and 4.3 from
`planning/MASTER_ARTIFACT_RESULTS.md` and quotes `results/baseline_v3/paper_table.md` (outside
`planning/`), with the encounter-specific VO as a placeholder until comparators v2, and keeps 4.2
(snapshots), 4.4 (reward scale audit) and 4.5 (safety layer) as placeholders. The plan is the
skeleton's restated Section 4 (`planning/PAPER/PAPER3_DRAFT_SKELETON.md`, S19): Table R1 with the
principal metrics, Table R1b with every seed and final model, Figures A and B, Table R3 by target
behavior, and Appendix G with the full per-axis table.

Items to fold into block C:
- The 4.1 placeholder still lists "the cumulative distribution of the closest approach", which no
  figure tool produces; Section 4 keeps Figures A and B only (2026-10-08), and the closest approach
  stays a table row.
- Optional, Section 1.2.2: "Making this split explicit allows the relationship between supplied
  context, reward incentives, and executed maneuvers to be examined rather than assumed." With no
  ablation, the paper states the split but does not measure it; a softer form is "Making this split
  explicit states which part of the decision context is engineered and which part is learned."

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
