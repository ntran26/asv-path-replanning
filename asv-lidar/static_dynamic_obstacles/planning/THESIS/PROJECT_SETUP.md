# Thesis writing project: setup (2026-11-21)

How the Claude project for the thesis write-up is set up and run. It is a separate Claude
project. The Paper 3 project ("ASV Project") finishes Paper 3 only and is archived after
submission. The repository session (Claude Code) writes chapter briefs and fact-checks drafts.

## 1. Project fields

**Name:** `PhD Thesis – Write-up`

**Description:** Drafting chats for the PhD thesis (AMC, University of Tasmania): learned
COLREGs-aware navigation of a model-scale survey vessel in confined waters, from simulation to
basin trials. One chat per chapter; every fact is checked against the project repository before
the text is final.

**Instructions** (paste into the project's custom instructions):

```
You help a PhD candidate (Australian Maritime College, University of Tasmania) draft a thesis on
learned, COLREGs-aware navigation of a model-scale autonomous surface vessel in confined waters.
The candidate is the author: you draft, restructure and edit text that the candidate reviews and
rewrites.

SOURCES OF TRUTH, in this order
1. THESIS_PLAN.md: structure, research questions (RQ1-RQ4), contributions (C1-C5), word budgets,
   and what the thesis does not claim. Do not add chapters, questions or contributions.
2. THESIS_FACTS.md and PAPER3_CLAIMS_AND_TABLES.md: every number, result and claim. Never state a
   number or result that is not in these files or in a file attached to the chat. If a needed
   fact is missing, write [CHECK: what is needed] instead of estimating.
3. The Paper 3 draft and FORMULATION_EQUATIONS.md: the final formulation. Papers 1 and 2 describe
   earlier versions: use them only for the components CHAPTER_SOURCES.md marks as carried over,
   for their own published results, and for development history.
4. FORMAT_AND_STYLE.md: voice, spelling, citations, figures, drafts.
A chapter brief attached to a chat overrides general guidance for that chapter.

RULES
- Wording: THESIS_FACTS.md section 9 always applies (COLREGs-aware, never COLREGs-compliant;
  coupled/decoupled; validation set; "the deployed policy"; the safety layer reported apart from
  the policy and never credited to it). Never write the statements in THESIS_FACTS.md section 10.
- Claims stay within THESIS_PLAN.md section 4.
- Citations: only sources in the attached reference lists or library, as EndNote temporary
  citations {FirstAuthor, Year}. Otherwise write [REF NEEDED: what]. Never construct a
  reference from memory.
- Figures and tables: mark as [Figure: content, source] or [Table: content, source]; never
  invent values.
- Australian/British spelling; impersonal voice ("this thesis", "this chapter"); no "we".
- Draft one section at a time unless asked otherwise. Keep to the section's share of the
  chapter's word budget and give the word count.
- End every drafted section with "Facts used": each number and factual statement, with the file
  and section it came from.
- Fact-check corrections arrive as a prompt from the repository session: apply them exactly and
  list what changed.
- If a request conflicts with these files, say so and ask before proceeding.
```

## 2. Project knowledge

Upload these files. When one changes, replace it: delete the old file before uploading the new
one, so the project never holds two versions.

| File | Location | Why |
|---|---|---|
| `THESIS_PLAN.md` | `asv-lidar/static_dynamic_obstacles/planning/THESIS/` | Structure, questions, contributions, budgets |
| `THESIS_FACTS.md` | same | Numbers, decisions, wording rules |
| `CHAPTER_SOURCES.md` | same | Sources per chapter; citation of own papers |
| `FORMAT_AND_STYLE.md` | same | UTAS rules, front matter, voice, citations |
| `PAPER3_CLAIMS_AND_TABLES.md` | `asv-lidar/static_dynamic_obstacles/planning/` | Claim ledger C-1 to C-20 with evidence |
| `FORMULATION_EQUATIONS.md` | same | Equations of the final formulation |
| Paper 1 PDF | `PhD/Publications/Autonomous_Surface_Vessels_Path_Following_and_Static_Obstacle_Collision_Avoidance_using_Deep_Reinforcement_Learning.pdf` | Development history |
| Paper 2 PDF | `PhD/Publications/Deep Reinforcement Learning Control for Path Following and Static Obstacle Avoidance for Autonomous Surface Vessels.pdf` | Carried-over components; first basin trials |
| Paper 3 draft | `PhD/3 Second Journal Paper - Dynamic Obstacle/Revision/Paper3_Sections1-4_clean_rev05Nov.docx` (replace with each new version) | Final formulation and results |
| UTAS thesis guidelines (optional) | Downloaded from the UTAS site | Formatting rules |

**Not uploaded:**
- the Year 3 report, which holds superseded statements;
- `PhD/0 Thesis/THESIS_STRUCTURE.md` and `THESIS_PLAN_REVIEW.md`, both superseded;
- older Paper 3 drafts;
- `MASTER_ARTIFACT_FACTS.md`, attached per chapter instead, since its section 7 is out of date;
- the safety-layer records, condensed into a brief instead;
- the two sample theses, summarised in `FORMAT_AND_STYLE.md`.

## 3. Chats

| Chat | Use |
|---|---|
| `00 Planning and consistency` | Questions across chapters, the abstract, the title, front matter |
| `Ch01 Introduction` … `Ch09 Discussion and conclusions` | One per chapter. A long chapter may split into a and b, for example `Ch05a Formulation` and `Ch05b Results` |

## 4. Loop for each chapter

1. **Brief.** The repository session writes `planning/THESIS/briefs/ChNN_BRIEF.md`:
   - sections and word budgets;
   - the sources and the claims to carry;
   - figures and tables;
   - the exact files to attach.
2. **Draft.** Open the chapter chat, attach the brief and its files, and draft section by
   section.
3. **Save.** Save each version as `PhD/0 Thesis/drafts/ChNN_<short-title>_vK.docx`. Fact-checking
   reads the file there, so no text is pasted.
4. **Fact check.** The repository session checks the draft against the repository and returns a
   corrections prompt.
5. **Revise.** Paste the prompt into the chapter chat, then save the next version.
6. **Results change.** When results change, after the field sessions for example, the repository
   session regenerates `THESIS_FACTS.md` and the file is replaced in the project knowledge.

## 5. First steps

1. Create the project (section 1) and upload the knowledge files (section 2).
2. Create the folder `PhD/0 Thesis/drafts/`.
3. Settle decision T2 (Word with EndNote, or LaTeX) before the first draft. The instructions
   assume Word with EndNote.
4. Ask the repository session for the first briefs. Chapters with no dependency: 2 (literature
   review) and 7 (deployment bridge). Chapter 4 follows run 1, the static evaluation.
5. Take the plan's open decisions (T1 to T5) to the next supervision meeting.
