# Format and style (status 2026-11-21)

## 1. UTAS requirements

Taken from the HDR Examinations Procedure, version 8 (Feb 2026), as excerpted in a search.
Confirm against the current PDF and the HDR Thesis Presentation and Formatting Guidelines
before final formatting.

**Word limit.**
- A PhD thesis may have at most 100,000 words.
- Chapters and footnotes count. The preamble, tables, bibliography, datasets and appendices do
  not.
- An over-length thesis needs prior approval from the Pro Vice-Chancellor (Graduate Research).

**Intention to Submit.**
- Due at least 8 weeks before submission. For a submission on 14 May 2027, that is
  19 March 2027.
- It needs an abstract of at most 2,000 characters, spaces included, and a 40-word summary.

**Examination.**
- At least two external examiners, normally over three to six months.
- Check whether an oral examination applies.

**Generative AI.** Check the current UTAS guidance on generative-AI assistance in HDR theses,
and include the disclosure it requires.

**Copyright.**
- P2 is MDPI, CC BY: reuse with citation.
- P1 is IEEE: thesis reuse is allowed with IEEE's credit line on the reused material. Check
  IEEE's current wording.

## 2. Pattern of the two AMC theses

The two theses are King (2024) and Goh (2023).

**Front matter, in this order:**
1. title page;
2. Declaration of Originality;
3. Authority of Access;
4. Statement Regarding Published Work;
5. Statement of Co-Authorship;
6. Abstract;
7. Acknowledgements;
8. Contents;
9. List of Figures;
10. List of Tables;
11. Nomenclature (symbols);
12. Abbreviations;
13. the generative-AI disclosure, if UTAS requires one.

**Statement of Co-Authorship.** It lists every co-author with their affiliation, then for each
paper:
- the full citation and the chapter it appears in;
- each contribution (underlying research, experiments or field work, analysis, supervision,
  writing).

King's statement is signed by the candidate, the primary supervisor and the centre director.

**Statement Regarding Published Work.** It names the chapters that contain published material
and states who holds the copyright. Goh adds one sentence on the unavoidable repetition between
chapters.

**Chapters.**
- Each study chapter opens with a short paragraph naming its research question and the
  publications it draws on, for example "This chapter draws on [P3] Sections 2-4…".
- Each ends with a chapter summary that links to the next chapter.
- One primary question, with research questions answered chapter by chapter (King: 1 + 3).

## 3. Language and voice

- **Spelling:** Australian/British (normalise, behaviour, manoeuvre, modelling). Reused P2 and
  P3 text, which follows MDPI's American spelling, is converted.
- **Voice:** impersonal, using "this thesis", "this chapter" and "the policy", with the passive
  where natural. "The author" is used only to separate the candidate's work from co-authors'.
  No "we".
- **Tense:** past for what was done; present for what a table, figure or result shows.
- **Definitions:** each technical term is defined once, at first use, and kept the same
  throughout. Terms follow `THESIS_FACTS.md` section 9.
- **Abbreviations:** expanded at first use in each chapter as well as in the list.
- **Claims:**
  - Every number comes from `THESIS_FACTS.md` or a results file it names.
  - Rates have three decimals in tables (0.865) and may be percentages in prose (86.5 %).
  - Every learner result carries its seed spread.
- **Limitations** are stated as design consequences, not apologies.

## 4. Citations and references

Decision T2 is open.

**If the thesis is written in Word with EndNote** (the P2 library exists):
- Drafts carry EndNote temporary citations, `{FirstAuthor, Year}`, for example
  `{Meyer, 2020}`.
- Update Citations and Bibliography turns them into the chosen style.
- An unresolved citation shows up as an EndNote error instead of a silent gap.

**If the thesis is written in LaTeX:** drafts carry BibTeX keys.

**Rules for any style:**
- Only sources that exist in the P1-P3 reference lists, the reading list or the reference
  library are cited.
- A claim that needs a source not yet in hand is marked `[REF NEEDED: what]`. A reference is
  never constructed from memory.
- Numeric style (as in King and the papers) or author-year (as in Goh): one style throughout.

## 5. Figures, tables and equations

- Figures and tables are generated from repository data (`CHAPTER_SOURCES.md`). Captions are
  self-contained.
- Numbering is by chapter: Figure 4.2, Table 5.3, Eq. (5.7).
- A draft marks a missing figure as `[Figure: content, source]`. Values are never invented.
- Equations follow `planning/FORMULATION_EQUATIONS.md`. Symbols are entered in the
  nomenclature once.

## 6. Drafts

- One file per chapter: `PhD/0 Thesis/drafts/ChNN_<short-title>_vK.docx`, or `.md` before
  typesetting. A new version is a new file; earlier versions are not edited.
- Every drafted section ends with a **Facts used** list: each number and factual statement,
  with the source file it came from. The list is removed only after the fact check.
