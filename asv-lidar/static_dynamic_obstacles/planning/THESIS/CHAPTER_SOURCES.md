# Chapter sources (status 2026-11-21)

For each chapter of `THESIS_PLAN.md`: what to draw on, which claims it carries, and what to
attach to the chapter's drafting chat. A chapter brief is written from the repository before
each chapter is drafted. It names the exact sections, figures and tables, and lists the
attachments as files.

**Path conventions.**
- Repository paths are relative to `asv-lidar/static_dynamic_obstacles/` unless they start with
  `field_deployment/`.
- "P3" means the Paper 3 draft `Paper3_Sections1-4_clean_rev05Nov.docx`, or its successor.
  Section numbers refer to that draft.

**What is current.** The current descriptions are:
- the Paper 3 draft;
- `planning/FORMULATION_EQUATIONS.md`;
- `planning/MASTER_ARTIFACT_FACTS.md` sections 1-6;
- `planning/PAPER3_CLAIMS_AND_TABLES.md`;
- `field_deployment/README_P3_BRIDGE.md`.

`planning_records/` holds design records from earlier stages. They are used only for
development history, never for the final design.

| Ch | Draws on | Claims | Attach to the chapter chat | Waiting on |
|---|---|---|---|---|
| 1 Introduction | P3 Section 1.1; Year 3 report (scope changes only); CoC report (`PhD/Confirmation of Candidature/CoC_Report_NamTran_675973.docx`) for the original aims and RQ1-RQ3; `THESIS_PLAN.md` sections 2 and 4 | None directly | CoC report; P1 and P2 abstracts | Full text last (Feb 2027) |
| 2 Literature review | P1-P3 introductions and reference lists; `planning/PAPER/READING_LIST.md`; the existing literature draft | None | P2 and P3 reference lists; reading list; existing draft | Nothing |
| 3 Vessel, simulation and learning setup | P3 Sections 2.1, 2.3.5, 3.1-3.4 and Appendices A, B, E; `MASTER_ARTIFACT_FACTS.md` sections 1, 3, 6; `planning/CURRICULUM_AND_DATA_BRIEF.md`; `planning/FIELD_CALIBRATION_PLAN.md` (identification method); `planning/IMU_CASE_FOR_MEETING.md` | C-9 (feasibility) | P3 draft; those files | Calibrated parameters from session S1 |
| 4 Path following and static obstacle avoidance | P3 Sections 2.3.1-2.3.3 (lidar, boundary, ego and path branches; static and path reward terms; the 49-times failure of the earlier reward); P2 Algorithm 1 and Sections 2-3 (pooling origin, action, curriculum); P1 (grid observation, development only); `results/baseline_v3/seed0_summary.md` (no-target and null episodes); `results/paper_snapshots/FS-NT-010.png` | Static-only rows of `THESIS_FACTS.md` section 4 | P3 draft; P2 PDF; `FORMULATION_EQUATIONS.md`; the static evaluation output (`THESIS_PLAN.md` section 5, item 1) | Run 1 |
| 5 COLREGs-aware encounters | P3 Sections 2.1.4, 2.2 (2.2.1-2.2.4), 2.3.2-2.3.3 (target and context branches, COLREGs terms), 4.1-4.4 and Appendices C, D, G; `MASTER_ARTIFACT_RESULTS.md`; `results/baseline_v3/paper_table.md`; `results/baseline_v3/reward_scale_audit.md`; `results/baseline_v3/figures_draft/` (Figures A and B); `results/paper_snapshots/`; development record from `planning_records/BASELINE_V4_PLAN.md` | C-1, C-8, C-10 to C-14; C-2 to C-5 as withdrawn | P3 draft; `PAPER3_CLAIMS_AND_TABLES.md`; `MASTER_ARTIFACT_RESULTS.md`; the figures | Paper 3's final text |
| 6 Runtime safety layer | P3 Sections 2.3.4, 4.5 and Appendix F; `MASTER_ARTIFACT_FACTS.md` section 5; `planning/PAPER/PAPER3_ALGORITHMS.md` (Algorithm A3); `planning_records/SAFETY_LAYER_VERSIONS.md`; `planning_records/SAFETY_ESTOP_PROTOCOL.md`; `planning_records/SAFETY_EVALUATION_PROTOCOL.md`; `planning_records/SAFETY_V21_PLAN.md` section 15; `planning_records/SAFETY_LAYER_COMPLETE_SUMMARY.md` (V1-V19, consult only); run time from `README_P3_BRIDGE.md` | C-7; C-20 (method only) | P3 draft; `SAFETY_LAYER_VERSIONS.md`; `PAPER3_ALGORITHMS.md`; a condensed safety brief (written from the 137 KB record, not uploaded whole) | Nothing |
| 7 Deployment bridge | `field_deployment/README_P3_BRIDGE.md`; `field_deployment/UDP_BENCH_HANDOVER.md`; `planning/FIELD_TRIALS_PLAN.md` section 3; `planning/FIELD_CALIBRATION_PLAN.md` (P-1, P-7 to P-11, T-14, T-15); bench outputs `field_deployment/p3_bench/` (parity, shadow replay, timing) | C-15 (bench part) | `README_P3_BRIDGE.md`; `UDP_BENCH_HANDOVER.md`; a bridge brief with diagrams of the data path | The vessel-link check (first basin session) for the link figures |
| 8 Field validation and transfer | `FIELD_TRIALS_PLAN.md`; `FIELD_CALIBRATION_PLAN.md`; `src/field_test_cases.py`; `field_deployment/trial_analysis/README_TRIAL_ANALYSIS.md` (tables F1-F5); trial logs and the analysis outputs; P2 field section | C-15 to C-20 | The plans; F1-F5 outputs; P2 PDF | Sessions S1, S2, T1, T2 |
| 9 Discussion and conclusions | All chapters; `THESIS_PLAN.md` sections 2 and 4; `planning/FUTURE_WORK_IDEAS.md`; P3 Section 6 | All | Drafts of Chapters 1-8 | Everything |

## Citation of the candidate's own papers

| Component | Cite | Wording |
|---|---|---|
| Feasibility sector pooling | Meyer et al. (2020) for the idea; P2 Algorithm 1 for the form; P3 for the three changes | "the feasibility pooling of [Meyer], in the form of [P2], with three changes [P3]" |
| Path-relative frame; joint rudder and propulsion action; staged propulsion curriculum | P2, carried into P3 | "introduced in [P2]" |
| Reward (bounded terms, hierarchy), boundary branch, target and context branches, encounter classification, Rule 9 framework, safety layer, test set and feasibility check, deployment bridge | P3, or thesis-only for the bridge and the safety-layer detail | "[P3]" |
| Grid observation; first SAC-over-PPO selection | P1, in the development sections only | "the first study [P1]" |

Each component is cited once where it is introduced and once in the chapter summary. Own work
is not cited sentence by sentence.

## Figures and tables

- **From the repository.** Figures and tables are generated from the results files by the
  scripts named in `PAPER3_CLAIMS_AND_TABLES.md`, so every value traces to data.
- **New thesis figures** are generated the same way:
  - the static evaluation by obstacle count;
  - safety-layer intervention timelines;
  - the bridge data path;
  - field plots.
- **Drafts.** A draft that needs a figure marks it `[Figure: content, source]` and does not
  invent its values.
