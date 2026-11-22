# Thesis plan review (2026-11-21)

> **Superseded for the structure by `THESIS_PLAN.md` (agreed 2026-11-21): nine chapters, with
> Paper 3 as the main source of Chapter 4 and separate chapters for the runtime safety layer and
> the deployment bridge.** This file is kept as the record of the reasoning. It is not uploaded
> to the writing project.

Compares the Year 3 completion plan (Year 3 report, 15 Nov 2026, 13 chapters) with the earlier
structure (`PhD/0 Thesis/THESIS_STRUCTURE.md`, revision 2, 8 Sep 2026, 15 chapters), checks both
against the evidence that exists now (`PAPER3_CLAIMS_AND_TABLES.md`, restated 2026-11-05), and
proposes a shorter structure on the pattern of two AMC theses from the same supervisory team.
Nothing here changes Paper 3.

## 1. The two plans side by side

| Revision 2 (Sep 2026) | Year 3 plan (Nov 2026) | Note |
|---|---|---|
| 1 Introduction | 1 | |
| 2 Literature review, seven threads | 2, same seven threads | |
| 3 Survey mission and global planning | 3, kept | Its evaluation (revision 2 Ch 13) is gone |
| 4 Problem formulation | 4 | |
| 5 Vessel model and system identification | 5 | |
| 6 Perception and observation | 6 | |
| 7 Reward and COLREGs | 7 | |
| 8 Environment, training, evaluation | 8 | |
| 9 Leg following and static obstacles | 9 | |
| 10 COLREGs with a dynamic target | 10 | Width sweep gone; ablation and "compliance by encounter type" still listed |
| 11 Robustness: perception degradation, disturbance | removed | Matches Paper 3 (2026-10-14) |
| 12 Sim-to-real and field validation | 11 | Domain-randomisation field ablation gone (C-6, 2026-11-05) |
| 13 Mission integration, coverage under avoidance, multi-vessel | removed | |
| 14 Discussion | 12 | |
| 15 Conclusions | 13 | |

With revision 2's budgets, the Year 3 plan is about 67,500 words of body text, plus about 2,000
of front matter and 5,000 of appendices.

**What the trim left inconsistent.**

1. Chapter 1 still promises six research questions and eight contributions, which are revision
   2's TRQ1-TRQ6 and C1-C8. Several of these no longer have evidence:

   | Revision 2 item | What happened to its evidence |
   |---|---|
   | TRQ3, C4: Rule 9 precedence as a function of channel width | Width sweep withdrawn 2026-10-14; the thresholds remain design reasoning only |
   | TRQ4, C5: degradation under poorer perception and disturbance | Perception study withdrawn 2026-10-14; the disturbance study was never run |
   | TRQ5: domain randomisation against improved identification | Field ablation withdrawn 2026-11-05 (C-6), because the calibration does not retrain the policy |
   | TRQ6, and the coverage part of C8 | The coverage-under-avoidance study was in the removed Chapter 13 |
   | TRQ2 second half, C2 "with the ablation isolating each" | Ablation dropped 2026-10-24 |
   | C7: algorithm ranking depends on the representation | SAC was selected in all three studies. Paper 3 shows the off-policy learners ahead at every seed (C-11), which points the other way |

2. Chapter 3 is kept without the chapter that evaluated it. It would stand as a design chapter
   with no result.
3. Chapter 10 lists two analyses that Paper 3 dropped:
   - "compliance by encounter type", replaced by trajectory snapshots checked against the
     track-based rule conditions (2026-10-08);
   - "ablation of the reward function", dropped 2026-10-24.
4. Methods are arranged by component (Chapters 4-8) and results by capability (Chapters 9-11).
   That arrangement has two costs:
   - it moves every paper's content into new places, so most of the text is rewritten rather
     than reused;
   - it needs attribution and supersession logs to keep track of where each method and result
     came from.

   It is the most expensive route from three papers to a thesis.

## 2. Reference pattern: two AMC theses

| | King (2024) | Goh (2023) |
|---|---|---|
| Topic | AUV return-home navigation beneath ice | AUV path re-planning for sonar area coverage |
| Format | Thesis by publication: four papers as Chapters 2-5, one under review | Chapter per study, built on two published conference papers; states "unavoidable repetition" |
| Chapters | 6 | 7 |
| Pages | 145 | 244 |
| Body text | about 30,000 words | about 47,000 words |
| Literature review | Inside Chapter 1 and each paper chapter | Section 1.3 of Chapter 1 |
| Questions | One primary question and three research questions, one per study chapter | One main question and seven sub-questions |
| Closing | One chapter: summary, conclusions, future work | One chapter: findings, conclusions, implications, future work |
| Citations | Numeric | Author-year |

Word counts come from text extraction and are approximate.

**Shared front matter in both theses:**
- declaration of originality;
- authority of access;
- statement regarding published work;
- statement of co-authorship with each paper's contributions;
- abstract;
- acknowledgements;
- contents and lists of figures and tables;
- symbols and abbreviations.

**UTAS rules (HDR Examinations Procedure, version 8, Feb 2026, from a search excerpt; confirm
against the current PDF):**
- **Word limit:** 100,000 words. Chapters and footnotes count; the preamble, tables,
  bibliography and appendices do not.
- **Intention to Submit:** at least 8 weeks before submission, with an abstract of at most
  2,000 characters and a 40-word summary.

Length is therefore a matter of time and focus, not of the rules.

## 3. Recommended structure: eight chapters by study

| Ch | Title | Content | Main sources | Words |
|---|---|---|---|---|
| 1 | Introduction | See below | Year 3 report; CoC | 5,000 |
| 2 | Literature review | See below | Existing draft (25 %) | 7,500 |
| 3 | Platform, vessel model and simulation environment | See below | Papers 2-3; calibration plan | 6,500 |
| 4 | Path following and static obstacle avoidance | See below | Papers 1-2 | 7,500 |
| 5 | COLREGs-aware encounters: formulation | See below | Paper 3 Section 2 | 8,000 |
| 6 | Training, evaluation design and simulation results | See below | Paper 3 Sections 3-4; results files | 9,000 |
| 7 | Field validation and simulation-to-field transfer | See below | Paper 3 Section 5; field plans; bridge records | 8,000 |
| 8 | Discussion and conclusions | See below | Thesis only | 5,500 |
| | **Total body text** | | | **57,000** |

**Chapter 1, Introduction.**
- Motivation.
- The survey mission as the frame: boustrophedon coverage gives straight legs, and the work is
  the local layer that follows each leg. This keeps the core argument of the former Chapter 3 in
  about 1,500 words.
- Scope evolution since Confirmation of Candidature:
  - sonar is out of scope;
  - the WAM-V with radar became the Bluefin with a 2D range sensor;
  - Gazebo became a custom simulator;
  - environmental disturbance (CoC RQ2) is declared out of scope, since all trials run in a
    calm basin.
- Governing question, research questions and contributions.
- Publications and the chapter map.

**Chapter 2, Literature review.** Five threads, then a gap analysis that derives the research
questions:
1. survey coverage, briefly;
2. classical guidance and collision avoidance: LOS, APF, DWA, VO, MPC;
3. deep reinforcement learning for ASV navigation and COLREGs;
4. restricted waters and Rule 9;
5. simulation-to-real transfer for learned marine control, including onboard perception.

**Chapter 3, Platform, vessel model and simulation environment.** Shared by Chapters 4-7.
- Bluefin and its sensors.
- The 3-DOF model and the actuator model.
- Identification, with the calibrated parameters in Appendix A.
- The turning circle and the minimum leg spacing it implies (the rest of the former Chapter 3).
- The simulator and its sensor model.
- Domain randomisation.

**Chapter 4, Path following and static obstacle avoidance.**
- Paper 1 as development only: the grid observation and SAC over PPO.
- Paper 2 in full:
  - sector pooling;
  - joint rudder and propulsion control;
  - the curriculum;
  - simulation results and obstacle density;
  - the classical baseline;
  - the static basin trials and the gap that motivated calibration.

**Chapter 5, COLREGs-aware encounters: formulation.**
- Task and two-vessel scope.
- The ship domain.
- The COLREGs rules stated as the reward encodes them: 8(b), (d) and (e); 9; 13-16; 17(a)(i).
- Rule 9 geometry, with the width thresholds as design reasoning.
- The observation extensions: map boundary, tracked target and encounter context.
- Reward and terminal conditions.
- The runtime safety layer (Algorithm A3).

**Chapter 6, Training, evaluation design and simulation results.**
- Learners and curriculum.
- The validation set, and the 1,000-episode test set with its feasibility check.
- The comparators: LOS-DWA and the encounter-specific VO.
- Results: C-1, C-8 and C-10 to C-14, with the safety layer (C-7) reported apart.
- Failure analysis.
- The development record: test set v1 to v4, baseline-v4.3 not adopted, and RecurrentPPO
  seed 1.

**Chapter 7, Field validation and simulation-to-field transfer.**
- The field system.
- The deployment bridge and its exact parity with the simulator.
- Timing.
- The calibration sessions.
- Trial protocol, the scenario families and the 15-case field test suite.
- Tables F1-F5 (C-15 to C-20).
- The gap across the Paper 2 and Paper 3 campaigns.

**Chapter 8, Discussion and conclusions.**
- Answers to the research questions.
- Cross-study synthesis.
- Limitations, stated as design consequences.
- Implications for confined-water survey.
- Future work: multi-vessel encounters, Rule 17(a)(ii), disturbance, the sonar payload and
  coverage replanning, and fine-tuning from field data.

**Why this order.**
- Each study chapter follows the paper it comes from. Chapter 4 reuses Papers 1-2, and Chapters
  5-7 reuse Paper 3's Sections 2-5. Supersession reduces to cross-references where Paper 3
  measures the same thing.
- New writing is about 25,000-30,000 words: Chapters 1, 2 and 8, parts of Chapter 3, linking
  text and the thesis-only detail. Under the Year 3 structure, most of about 67,000 words would
  be rewritten.
- The structure suits both a monograph and a thesis with published works, as in King's thesis.
  Decision D1 (monograph) can stand: the saving comes from ordering by study, not from the
  format label.

**Mapping from the Year 3 plan:**

| Year 3 chapter | Goes to |
|---|---|
| 1 Introduction | 1 |
| 2 Literature review | 2 |
| 3 Survey mission | 1, the frame; 3, the turning circle and leg spacing |
| 4 Problem formulation | 5, with the static-obstacle parts in 4 |
| 5 Vessel model | 3 |
| 6 Perception and observation | 3, the sensor; 4, sector pooling; 5, boundary, target and encounter context |
| 7 Reward and COLREGs | 4, the Paper 2 reward; 5, the Paper 3 reward |
| 8 Environment and protocol | 3, simulator and randomisation; 4, the Paper 2 protocol; 6, the Paper 3 protocol |
| 9 Static results | 4 |
| 10 COLREGs results | 6 |
| 11 Field | 7, with the Paper 2 static trials in 4 |
| 12-13 Discussion, conclusions | 8 |

**Minimum change if the 13-chapter structure from the review is kept:**
1. Fold Chapter 3 into Chapters 1 and 5.
2. Merge Chapters 12 and 13.
3. Rename Chapter 10 to "COLREGs-aware" and replace the dropped analyses.
4. Restate the research questions and contributions (section 4).
5. Add the safety layer and the deployment bridge to the methods.

The result is 11 chapters and about 62,000 words.

## 4. Research questions and contributions, restated to the evidence

**Governing question.** How can an autonomous surface vessel follow survey legs in confined
waters, avoiding static obstacles and another vessel with COLREGs-aware behaviour from onboard
sensing alone, and how well does a policy learned in simulation transfer to a physical vessel?

| | Question | Chapters | Evidence |
|---|---|---|---|
| RQ1 | What observation of onboard range sensing, own-vessel state and path geometry lets one learned policy follow a leg, avoid static obstacles and keep to navigable water? | 4, 5 | Papers 1-2; Paper 3 observation |
| RQ2 | Can one policy, trained on a single formulation, resolve two-vessel encounters among static obstacles in basins and channels with encounter-consistent behaviour, and how does it compare with classical comparators and other learners, with and without a runtime safety layer? | 5, 6 | C-1, C-7, C-8, C-10 to C-14 |
| RQ3 | How does the policy perform on a physical vessel, how large is the simulation-to-field gap, and is it due to the vessel's dynamics or to perception and localisation? | 4, 7 | Paper 2 trials; C-15 to C-20 |

**Contributions.**

| | Contribution | Evidence |
|---|---|---|
| C1 | A confined-water formulation: leg following, static avoidance, boundary keeping and two-vessel COLREGs-aware encounters in one policy, with Rule 9 precedence encoded geometrically | Chapters 5-6 |
| C2 | An onboard-perception observation: sector pooling (Paper 2) extended with the map boundary, the tracked target and the encounter context (Paper 3) | Chapters 4-5 |
| C3 | An evaluation design: a feasibility-checked, held-out 1,000-episode test set; checkpoint selection on the validation set; pre-registered claims; four learners with three seeds each; two classical comparators; and the safety layer reported apart from the policy | Chapter 6 |
| C4 | A simulation-to-real pipeline and field validation: identified model and randomisation, a deployment bridge that reproduces the simulator's observations and decisions exactly, static and encounter basin trials, and the gap measured at three levels | Chapters 3, 4, 7 |

The survey frame is context, not a research question.

The thesis does not claim:
- COLREGs compliance, or any per-rule compliance rate;
- behaviour in channels narrower than those tested;
- robustness to poorer perception or to disturbance;
- that the rule-related behaviour is learned rather than supplied;
- a ranking of SAC over TQC;
- coverage completeness, unless the optional study in section 8 is run.

## 5. Can be omitted

| Item | Reason | Saving |
|---|---|---|
| Chapter 3 as a chapter | No evaluation left; its argument fits in Chapter 1 and its turning-circle part in Chapter 3 | about 3,000 words |
| Two of the seven literature threads as separate threads | Restricted waters merges into COLREGs; perception merges into simulation-to-real; coverage becomes a short section | about 1,500 words |
| Separate Discussion and Conclusions | Both AMC theses close with one chapter | about 1,000 words and the duplication |
| Paper 1 results tables | Superseded in method; kept as a short development section | about 2,000 words |
| Chapter 10 "compliance by encounter type" and the reward ablation | Not run in Paper 3; snapshots and the limitation statement replace them | Avoids a new experiment |
| Revision 2 appendices F (facility survey) and G (boustrophedon planner) | No corresponding study remains | Appendix effort |
| Supersession and attribution logs | Ordering by study makes provenance follow the chapters | Process effort |

## 6. Missing from the Year 3 plan

| Item | Where |
|---|---|
| The runtime safety layer as a method: predictive filter and permanent emergency stop | Chapter 5; results in Chapters 6 and 7 |
| The deployment bridge: exact simulator parity, timing, shadow mode | Chapter 7 |
| The feasibility check behind the test set, and the history from test set v1 to v4 | Chapter 6 |
| Target behaviour variants: reactive, noncompliant head-on, single speed change | Chapter 6 |
| The development record, including negative results: baseline-v4.3, the replaced COLREGs-VO, the RecurrentPPO seed 1 collapse, the safety-layer versions | Chapter 6, or an appendix table |
| Scope evolution, with CoC RQ2 (disturbance) declared out of scope | Chapter 1 |
| A new title (see below) | Front matter |
| Statement regarding published work and statement of co-authorship for Papers 1-3 | Front matter |
| Disclosure of generative-AI assistance, as UTAS requires (check the current guidance) | Front matter |
| Copyright check: MDPI (CC BY); IEEE thesis reuse with its credit line | Front matter, Chapter 4 |
| Data and code availability: repository, test-set digest, formulation digest | Appendix |
| Intention to Submit, by 19 Mar 2027 for a 14 May 2027 submission | Timeline |

**The title.** The CoC title (sonar, coverage replanning) no longer describes the work.
Candidates, keeping "LiDAR" out of the title as for Papers 2-3:
1. *Autonomous Surface Survey in Confined Waters: Learning COLREGs-Aware Path Following and
   Collision Avoidance from Onboard Sensing*
2. *Deep Reinforcement Learning for COLREGs-Aware Navigation of an Autonomous Survey Vessel in
   Confined Waters*

## 7. Statements in the Year 3 report that must not carry into the thesis

| Year 3 report | Current record | Source |
|---|---|---|
| "COLREGs-Compliant Navigation"; "compliance by encounter type" | COLREGs-aware; snapshots checked against the track-based rule conditions | Decisions 2026-10-08; C-10 |
| Classical baselines "LOS-DWA and COLREGs-VO", 77 % and 69 % | LOS-DWA 0.770 and the encounter-specific VO 0.675 on test set v4; COLREGs-VO replaced 2026-10-20 | C-1; S18 |
| "With the safety layer, success rises to about 91 %" | Goals 865 → 868 on the test set. The layer halves own-vessel collisions (135 → 70) and is not credited to the policy. The 91 % figure came from development use of test set v2 | C-7 |
| "The final simulation step was targeted training on those situations" | The conflict curriculum (baseline-v4.3) was trained in full and not adopted: 0.833 against 0.865 | `planning_records/BASELINE_V4_PLAN.md` |
| "Ablations study on reward components" | Dropped 2026-10-24; a one-variant reserve exists only if review asks for it | S19; C-4 |
| "Six research questions and eight contributions" | Three questions and four contributions (section 4) | This review |

## 8. Writing order to May 2027

| Period | Work | Depends on |
|---|---|---|
| Now to mid-Dec 2026 | Chapter 4 from Papers 1-2; Chapter 2; the survey-frame section of Chapter 1; research questions, contributions and title agreed with supervisors | Nothing |
| Dec 2026 to Jan 2027 | Chapters 5-6 alongside the Paper 3 submission; Chapter 3 as the calibration sessions report | Paper 3 final text; calibration session 1 |
| Jan to Feb 2027 | Chapter 7, with the F1-F5 table shells written now | Trial sessions T1 and T2 |
| Feb to mid-Mar 2027 | Chapter 8, Chapter 1 in full, the abstract; Intention to Submit by 19 Mar | Everything above |
| Mar 2027 | Full draft to supervisors | |
| Mar to May 2027 | Revision, formatting, submission on 14 May | |

**Optional, only if time allows after the trials:**
- **A simulated multi-leg survey.** One run of the policy leg by leg, with the turns handled by
  a waypoint controller and coverage completeness measured under avoidance. It is the only way
  to turn the survey frame into evidence. It is a few days of simulation work and needs no
  retraining.
- **The encounter-context ablation.** One variant with three seeds, run only if Paper 3's
  reviewers ask for it.

## 9. Decisions for the supervisors

1. Eight study chapters (section 3), or the 11-chapter minimum change.
2. The research questions and contributions as restated (section 4).
3. The title.
4. Citation style (numeric or author-year) and typesetting (Word or LaTeX).
5. Whether the optional multi-leg survey is in scope.
6. How generative-AI assistance is disclosed under UTAS rules.
