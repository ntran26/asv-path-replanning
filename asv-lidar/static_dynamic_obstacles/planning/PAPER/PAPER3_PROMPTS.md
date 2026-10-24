# Paper 3 draft: edit instructions (kept current)

> **Kept current.** Last updated 2026-10-24. This file holds the next edit instructions for the
> Paper 3 draft, each as a block to paste into the chat that writes the paper. The blocks name
> planning files by path; that chat reads them from the synced project, so each push needs a Sync
> before the next block is sent. Only files outside `planning/` are attached: the .docx draft,
> kept in `PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/`. Blocks already applied move
> to the log at the end.

## 1. State of the draft

Current file: `Paper3_Sections1-3_with_placeholders_rev22Oct_corrected_tracked.docx`, checked
against the code on 2026-10-24.

- **Drafted:** Sections 1–3, Appendices B–F, Discussion 6.2–6.3.
- **Placeholders:** Sections 4, 5, 6.1 and 7; Appendix A; Section 3.4.4 (the ablation, to be
  deleted by block B).
- **Applied and checked:** the ten corrections of 2026-10-22 (log, entry 1).
- **Open in the text:** the nine items of block A and the decisions of block B.

## 2. Next: blocks A and B (send together if neither has gone yet)

Attachment for both: the current .docx. Needs: the planning folder synced (block A reads
`planning/PAPER/PAPER3_ALGORITHMS.md`, Algorithm 4 updated 2026-10-24). When both go in one message,
the output is one file, `Paper3_Sections1-3_with_placeholders_rev24Oct_tracked.docx`. When block A
has already been applied, block B applies to its output and saves as `..._rev24Oct2_tracked.docx`.

### Block A: nine corrections from the check of 2026-10-24

The five items that the 22 October log flagged without editing, each checked against the code, plus
the 30 no-target coupled layouts and the multi-axis decision.

```
Apply the following changes to the attached Paper3_Sections1-3_with_placeholders_rev22Oct_corrected_tracked.docx as tracked changes dated 24 October 2026 (author "Claude"), keeping every earlier tracked change and comment, and save the result as Paper3_Sections1-3_with_placeholders_rev24Oct_tracked.docx. Each change comes from a check against the code and the test-set definition. Report each change in a short revision-log addendum.

1. Section 1.3, training and evaluation paragraph. Replace "and 336 coupled episodes, in which the encounter takes place among three static obstacles (constant- and varying-speed targets)." with "and 336 coupled episodes, in which the encounter takes place among three static obstacles (constant- and varying-speed targets; 30 of these layouts have no target)."

2. Section 3.4.1. Replace "and its 336 coupled episodes are three-obstacle layouts with constant- and varying-speed targets, 228 drawn in the three motifs of Section 2.1.4 and 108 on three fixed reference layouts with the target drawn on their legs." with "and its 336 coupled episodes are three-obstacle layouts, 306 with a target (156 at constant speed and 150 with one speed change) and 30 without; 228 are drawn in the three motifs of Section 2.1.4 and 108 on three fixed reference layouts, with the target drawn on their legs." Keep the comment [CHECK — name the reference layouts …] anchored on "three fixed reference layouts".

3. Table A4, row "Test set v4, coupled", column Composition. Replace the text with "Three-obstacle layouts: 156 with a constant-speed target, 150 with a varying-speed target, and 30 without a target; 228 in the three motifs of Section 2.1.4 and 108 on three fixed reference layouts".

4. Section 2.2.4, last width paragraph. Replace "The head-on and overtaking encounters therefore lie mostly above their thresholds, including the 6.30 m centerline case, whereas channel crossings span the 7.60 m crossing threshold." with "The head-on and overtaking encounters therefore lie entirely above their 3.80 m and 4.26 m thresholds and mostly above the 6.30 m centerline case, whereas channel crossings span the 7.60 m crossing threshold."

5. Section 3.4.3.
   (a) Replace "the RMS cross-track error" with "the root-mean-square (RMS) cross-track error" (the abbreviation is not expanded anywhere else).
   (b) Replace the marker "[TBC — presentation of the multi-axis results: a per-axis table or a Pareto view rather than a weighted scalar]" with "Each metric is reported on its own, without a weighted combination: the results table of Section 4.1 gives the principal metrics, and Appendix G gives the full set for all test episodes and for the decoupled and coupled families."
   (c) After Appendix F, add the heading "Appendix G. Full results tables" and below it the placeholder "[Placeholder] Every metric of Section 3.4.3 for each method, for all test episodes and for the decoupled and coupled families."

6. Algorithm 4. The 108 reference-layout episodes are drawn by the same generator, with the layout's leg and panels given rather than drawn, so the box gains a fixed-layout branch (planning/PAPER/PAPER3_ALGORITHMS.md in the project files has the same change). Keep the draft's symbols.
   - Require: after "space-time solvability flag)" add "; for the test set's reference layouts, a fixed layout (a leg and three panels)".
   - Line 1: replace "if the stage has coupled layouts then with the stage's coupled share return COUPLED-LAYOUT()" with "if a fixed layout is given then return COUPLED-LAYOUT(); else if the stage has coupled layouts then with the stage's coupled share return COUPLED-LAYOUT()", keeping the comment "▷ lines 22–27".
   - Line 22: replace "COUPLED-LAYOUT(): a basin leg with start and goal x ∈ [2, 8] m (slant ≤ 14.1°)" with "COUPLED-LAYOUT(): the fixed layout's leg if one is given; otherwise a basin leg with start and goal x ∈ [2, 8] m (slant ≤ 14.1°)".
   - Line 23: replace the opening "three 1 m panels in one motif" with "the fixed layout's three panels if one is given; otherwise three 1 m panels in one motif", and keep the rest of the line.
   - Line 26: replace "reject unless SPACE-TIME-SOLVABLE()" with "if the panels were drawn then reject unless SPACE-TIME-SOLVABLE()".
   The sentences saying that every episode is drawn by Algorithm 4 (Section 2.3.5, Section 3.4.1, Appendix E) stay as they are.

7. Appendix E. Replace "The sets are frozen and identified by digest: test set v4, a8c40beb234ea7f9; the validation set is regenerated from its seed blocks by the frozen generator (formulation digest 03a9c0c22f53ee0b)." with "The test set is frozen and identified by its digest (test set v4, a8c40beb234ea7f9); the validation set is regenerated from its seed blocks by the frozen generator (formulation digest 03a9c0c22f53ee0b)."

8. Section 3.1. Replace "Each learner uses the default settings of its Stable-Baselines3 implementation" with "Each learner uses the default settings of its Stable-Baselines3 or sb3-contrib implementation".

9. Marker list at the end of the document. Delete the empty paragraph left where the [CHECK — hash] entry was removed, and delete the entry "[TBC — presentation of the multi-axis results]: Section 3.4.3."
```

### Block B: the Section 4 decisions of 2026-10-24 (skeleton S19)

No ablation (held in reserve for review); the reward scale audit for the deployed policy only; no
prior-paper policy as a comparator (the draft has none to remove).

```
Apply the following changes as tracked changes dated 24 October 2026 (author "Claude"), keeping every earlier tracked change and comment. They record three decisions on Section 4, listed in planning/PAPER/PAPER3_DRAFT_SKELETON.md, decision S19: the paper reports no ablation; the reward scale audit covers the deployed policy only; the policy of the previous study is not a comparator. Report each change in the revision-log addendum.

1. Section 3, introductory paragraph. Replace "the target behaviors, the metrics, and the ablation (Section 3.4)" with "the target behaviors, and the metrics (Section 3.4)".

2. Delete Section 3.4.4 "Ablation": its heading and its placeholder paragraph.

3. Delete Section 4.4 "Ablation": its heading and its placeholder. Renumber Section 4.5 "Reward scale audit" as 4.4 and Section 4.6 "Runtime safety layer" as 4.5, and change every "Section 4.6" to "Section 4.5" (Section 3.4.3, the opening placeholder of Section 4, and the marker list).

4. New Section 4.4 "Reward scale audit". Replace the placeholder "Table R7: the episode-integrated contribution of each reward term under a random policy and under the deployed policy, with the empirical ordering checked against the hierarchy of Section 2.3.3." with "Table R7: the episode-integrated contribution of each reward term of Table A2 under the deployed policy on the test set, with the empirical ordering checked against the magnitude hierarchy of Section 2.3.3. [PENDING — reward scale audit of the deployed policy]"

5. Section 6.1. Delete the marker "[TBC — if the ablation is run: whether the rule-related behavior is learned or supplied]".

6. Section 6.2, last paragraph. Replace "so rule-related behavior is partly supplied rather than learned, which the ablation of Section 3.4.4 is meant to separate;" with "so rule-related behavior is partly supplied rather than learned, and no ablation separates the two;".

7. Section 6.3. After the sentence ending "would separate that method from this study's interpretation of it.", insert "An ablation that hides the encounter context from the policy while keeping the COLREGs reward terms would measure how much of the rule-related behavior is learned rather than supplied."

8. Marker list at the end of the document. Delete the entry "[TBC — ablation budget]: Sections 3.4.4 and 4.4; the matching question in Section 6.1."; in the entry "[PENDING — safety-on evaluation: to be finalized and reported]: Section 4.6." change "4.6" to "4.5"; and add "[PENDING — reward scale audit of the deployed policy]: Section 4.4."
```

## 3. After that: block C, Section 4 (not yet written)

Written once TQC seed 2 has its test-set run (expected 2026-10-25), so the deployed learner is
known and every paragraph can name it. It drafts 4.1 and 4.3 from `planning/MASTER_ARTIFACT_RESULTS.md`
and `results/baseline_v3/paper_table.md` (the latter quoted, since it is outside `planning/`),
with the encounter-specific VO as a placeholder until comparators v2, and keeps 4.2 (snapshots),
4.4 (reward scale audit) and 4.5 (safety layer) as placeholders. The plan is the skeleton's
restated Section 4 (S19): Table R1 with the principal metrics, Table R1b with every seed and final
model, Figures A and B, Table R3 by target behavior, and Appendix G with the full per-axis table.

## 4. Log of applied blocks

1. **2026-10-22, ten corrections** (3.1 learner settings; Algorithm 3 centroid and track deletion;
   3.4.1 crossing counts and the coupled composition; 3.4.3 metric list; 6.2 "entirely above";
   Algorithm A2 rudder delay; Appendix E validation-set digest; references 76 and 72). Applied in
   `Paper3_Sections1-3_with_placeholders_rev22Oct_corrected_tracked.docx` with three consistency
   edits; checked on 2026-10-24, all ten present.
