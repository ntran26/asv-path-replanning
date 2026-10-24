# Planning records

These files were moved here from `planning/` on 2026-10-24. Since then, `planning/` holds only the
current, authoritative files, listed in `planning/00_CURRENT_STATE.md`.

What is here:

- **Design records** from the build of the formulation: `00_PAPER3_INDEX_AND_PROTOCOL.md`, the 01–06
  design and specification documents, `FORMULATION_PLAN.md`, `METHODS_BRIEF.md`, `PROJECT_BRIEF.md`,
  `PART2_BASIN_PLAN.md`, `BASELINE_V4_PLAN.md` and `Paper3_Introduction_draft4.docx`. The code cites
  their sections. Parts are superseded; where they disagree with a file in `planning/`, the
  `planning/` file is right.
- **Runtime safety-layer records** (`SAFETY_*.md`, `SAFETY_LAYER_REFERENCES.bib`). The layer is not
  finalized; `SAFETY_LAYER_COMPLETE_SUMMARY.md` and `SAFETY_V20_PLAN.md` are the latest.
- **`archive/`:** finished or superseded documents and the earlier safety-layer version plans.

**Paths.** References in the hand-written Markdown documents were updated to
`planning_records/<file>`. The frozen code (`src/`), the configuration files (`configs/`), the
diagnostic tools and the generated result records still cite `planning/<file>` in comments,
docstrings and provenance strings, because changing them would alter recorded digests or outputs.
Read any such `planning/<file>` as `planning_records/<file>`.
