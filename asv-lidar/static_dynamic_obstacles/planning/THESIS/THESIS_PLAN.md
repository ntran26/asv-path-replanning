# Thesis plan (agreed 2026-11-21)

The plan of record for the thesis. It supersedes the completion plan in the Year 3 report
(15 Nov 2026) and `PhD/0 Thesis/THESIS_STRUCTURE.md` (revision 2, 8 Sep 2026). The reasoning
behind it is in `THESIS_PLAN_REVIEW.md`. Facts, numbers and terminology are in `THESIS_FACTS.md`;
the sources for each chapter are in `CHAPTER_SOURCES.md`; format and style rules are in
`FORMAT_AND_STYLE.md`.

## 1. Principles

1. **One study per chapter.** Chapters 4-8 each carry their own method and results, as in the
   two AMC theses used for reference (King 2024, Goh 2023). Chapter 3 holds what they share.
2. **The final formulation is the main source.** Paper 3's formulation, the corrected version,
   is described throughout, including the static-obstacle part in Chapter 4. Components that
   Paper 3 carries over unchanged, or with stated changes, are cited to Paper 2. The main
   examples are the feasibility sector pooling (Paper 2, Algorithm 1), the path-relative frame
   and the joint rudder and propulsion action. Paper 1, mostly open water, appears as
   development only.
3. **Evidence first.** Every number comes from `THESIS_FACTS.md` or the files it names. No claim
   goes beyond `PAPER3_CLAIMS_AND_TABLES.md` or the thesis-level questions below.
4. **Thesis-only depth goes where the papers were compressed:**
   - the vessel model;
   - the runtime safety layer;
   - the deployment bridge;
   - the field programme;
   - the development record, including negative results.

## 2. Governing question, research questions, contributions

**Governing question.** How can an autonomous surface vessel follow survey legs in confined
waters, avoiding static obstacles and another vessel with COLREGs-aware behaviour from onboard
sensing alone, and how well does a policy learned in simulation transfer to a physical vessel?

| | Research question | Chapters |
|---|---|---|
| RQ1 | What observation and reward let one learned policy follow a leg, avoid static obstacles and keep to navigable water in confined water, from onboard range sensing, own-vessel state and path geometry? | 3, 4 |
| RQ2 | Can the same policy resolve two-vessel encounters among static obstacles in basins and channels with encounter-consistent, COLREGs-aware behaviour? How does it compare with classical comparators and with other learners trained on the same formulation? | 5 |
| RQ3 | What does an engineered runtime safety layer add to the learned policy, and at what cost? | 6 |
| RQ4 | How does the policy perform on a physical vessel? How large is the simulation-to-field gap, and is it due to the vessel's dynamics or to perception and localisation? | 7, 8 |

| | Contribution | Chapters |
|---|---|---|
| C1 | One formulation for leg following, static-obstacle avoidance, boundary keeping and two-vessel COLREGs-aware encounters in confined water, with narrow-channel (Rule 9) precedence encoded geometrically and a reward whose bounded terms enforce the intended hierarchy | 4, 5 |
| C2 | An onboard-perception observation: feasibility sector pooling (Paper 2), extended with the map-derived boundary, the tracked target and the encounter context | 4, 5 |
| C3 | An evaluation design: a feasibility-checked, held-out 1,000-episode test set; checkpoint selection on the validation set; pre-registered claims; four learners with three seeds each; two classical comparators | 3, 5 |
| C4 | A runtime safety layer: a predictive filter plus a permanent emergency stop, which reduces own-vessel collisions and is evaluated apart from the policy | 6 |
| C5 | A simulation-to-real pipeline, consisting of: an identified vessel model with domain randomisation; a deployment bridge that reproduces the simulator's observations and decisions exactly; and static and encounter basin trials with the gap measured at three levels | 3, 7, 8 |

The survey mission is the frame of the thesis, not a research question. Coverage planning
decomposes a survey area into straight legs, and the thesis studies the local layer that
follows each leg.

## 3. Chapters

| Ch | Title | Words |
|---|---|---|
| 1 | Introduction | 5,000 |
| 2 | Literature review | 7,500 |
| 3 | Vessel, simulation and learning setup | 8,000 |
| 4 | Path following and static obstacle avoidance | 6,500 |
| 5 | COLREGs-aware encounters | 10,000 |
| 6 | Runtime safety layer | 6,000 |
| 7 | Deployment bridge: from simulation to the vessel | 5,500 |
| 8 | Field validation and simulation-to-field transfer | 7,500 |
| 9 | Discussion and conclusions | 5,500 |
| | **Body text** | **61,500** |

Front matter, tables, references and appendices fall outside the UTAS word count.

**Chapter 1, Introduction.**
1. Motivation: autonomous survey in confined and restricted waters.
2. The survey mission and the two-level architecture. Coverage planning gives straight legs;
   the local layer follows them. This is why the policy is trained on straight legs.
3. Scope and its evolution since Confirmation of Candidature:
   - the sonar payload is out of scope;
   - the WAM-V with radar became the model-scale Bluefin with a 2D range sensor;
   - MOOS-IvP and Gazebo became a custom simulator;
   - environmental disturbance (CoC RQ2) is out of scope, because the trials run in a calm
     basin.
4. Governing question, research questions, contributions.
5. Publications and how they map to chapters.
6. Thesis outline.

**Chapter 2, Literature review.**
1. Survey missions and coverage path planning, briefly.
2. Classical guidance and collision avoidance for surface vessels: LOS, APF, DWA, VO, MPC and
   rule-based methods.
3. Deep reinforcement learning for ASV navigation, including COLREGs-aware learning.
4. COLREGs in restricted waters: Rule 9, narrow channels, and constrained manoeuvres.
5. Simulation-to-real transfer for learned marine control: identification, randomisation,
   onboard perception and field validation.
6. Gap analysis, leading to RQ1-RQ4.

**Chapter 3, Vessel, simulation and learning setup.**
1. The Bluefin platform and its sensors.
2. The 3-DOF manoeuvring model and the actuator model.
3. Identification: method and calibrated parameters (Appendix A).
4. The turning circle and the minimum leg spacing it implies for a survey.
5. Workspace: basin, channels and the reference layouts L1-L3.
6. The simulator: episode structure, sensor model, staleness, target vessel models.
7. Domain randomisation.
8. Learners and network.
9. Curriculum.
10. Training protocol and checkpoint selection.
11. Evaluation design:
    - the validation set;
    - test set v4 and the feasibility check;
    - target behaviours;
    - metrics and statistics;
    - the classical comparators (LOS-DWA, encounter-specific VO).

**Chapter 4, Path following and static obstacle avoidance.** Paper 3 is the main source;
unchanged components are cited to Paper 2.
1. The path-relative frame and the leg-following task.
2. The lidar branch: feasibility sector pooling (Paper 2, Algorithm 1) and the three changes in
   the final version (map-gated input, 27 non-uniform sectors, true beam spacing).
3. The boundary branch.
4. Ego and path features.
5. Static and path-following reward terms, and the bounded-term hierarchy. This covers the
   failure of the Papers 1-2 reward, where a hidden scale made the obstacle term about 49 times
   weaker than the path term at contact, and its correction.
6. Terminal conditions.
7. Results:
   - the deployed policy on the test set's no-target and null-encounter episodes;
   - path-following quality;
   - the static evaluation by obstacle count (section 5, item 1).
8. Development: the grid observation (Paper 1, open water) to the pooled sectors (Paper 2) to
   the final observation, with what changed and why.

**Chapter 5, COLREGs-aware encounters.**
1. The two-vessel scope.
2. The ship domain.
3. The rules as the reward encodes them: 8(b), (d) and (e), with 8(a) as timing pressure; 9;
   13; 14; 15 and 16 under the narrow-channel convention; 17(a)(i).
4. Encounter classification, engagement and the latched encounter.
5. The geometric framework for constrained encounter responses (Rule 9 precedence and width
   thresholds, as design reasoning).
6. Tracking, and the target and context branches.
7. The COLREGs reward terms.
8. Results:
   - the test set against the comparators (C-1);
   - basin against channel (C-8);
   - snapshots by encounter type with the rule conditions (C-10);
   - learners and seeds (C-11, C-12);
   - target behaviours (C-13);
   - the reward scale audit (C-14).
9. Failure analysis.
10. The development record: test set v1 to v4, baseline-v4.3 not adopted, COLREGs-VO replaced,
    the RecurrentPPO seed 1 collapse.

**Chapter 6, Runtime safety layer.** In more detail than Paper 3.
1. Requirements: Rule 8(e) in two layers; reduce own-vessel collisions; onboard information
   only; never credited to the policy.
2. Architecture: the position between the policy and the actuators; when it is on.
3. The predictive filter: the plan bank, the 8 s rollouts with the identified model and
   constant-velocity targets, the pass test and margins, the choice rule, hull completion.
4. The permanent emergency stop: first commands and continuations, the two reverse-thrust
   models, the 2.5 s trigger, the commitment.
5. Stopping profiles and the precomputed table.
6. Design history: the versions and what each fixed (SAFETY_LAYER_VERSIONS); the reachability
   approach that was tried and archived.
7. Evaluation protocol and scoring: what counts against the layer and what is reported apart.
8. Results on the test and validation sets, layer off against on (C-7).
9. Run time and its consequence for active use (T-14).
10. Limits: development on the test set, no guarantee, the braking convention.

**Chapter 7, Deployment bridge: from simulation to the vessel.**
1. The system: vessel-control computer and shore laptop, the UDP link, message formats, the
   2 Hz decision cycle.
2. The observation adapter: the simulator's own environment class with its sensors replaced by
   the measured scan and pose, so the same perception code runs on real data.
3. Pose synchronisation, the ego-motion estimate over the decision interval, and stale frames.
4. Command mapping and the sign conventions.
5. The safety layer in shadow mode (the worker) and in active mode; the operator stop.
6. Logging on one clock, and offline replay.
7. Verification:
   - parity with the simulator (32/32 runs, 1,497 decisions exactly equal);
   - replays of the July and February logs over loopback UDP;
   - run time on two computers.
8. Open issues before the trials: phantom moving tracks, telemetry resolution, the pose
   reference point, the map frame.

**Chapter 8, Field validation and simulation-to-field transfer.**
1. Calibration sessions S1 and S2: sensor and model checks, identification runs.
2. The trial protocol, safety and run validity.
3. Scenario families A, B and C, and the 15-case field test suite.
4. Results F1-F5: runs by scenario, encounter behaviour, the sim-to-field comparison, the
   vessel model, perception on the water (C-15 to C-20).
5. Shadow-mode safety layer on the water (C-20).
6. The gap across the Paper 2 static campaign and the Paper 3 campaign.

**Chapter 9, Discussion and conclusions.**
1. Answers to RQ1-RQ4, each tied to tables and figures.
2. Cross-study synthesis:
   - what the observation needed;
   - why off-policy learners led;
   - what the safety layer can and cannot do;
   - where the gap came from.
3. Limitations, stated as design consequences.
4. Implications for confined-water survey.
5. Future work: multi-vessel encounters; Rule 17(a)(ii); disturbance; the sonar payload and
   survey-quality replanning; fine-tuning from field data.
6. Concluding remarks.

**Appendices.**

| | Content |
|---|---|
| A | Vessel model: parameters, identification, fit statistics |
| B | Reward specification |
| C | Encounter classification thresholds |
| D | Learner hyperparameters and curriculum schedule |
| E | Evaluation sets: composition, digests, feasibility check |
| F | Classical comparators and safety-layer parameters |
| G | Full results tables with intervals |
| H | Field trial run sheet, log index and data management |
| I | Risk assessment and safety procedures |
| J | Data and code availability |

## 4. What the thesis does not claim

- COLREGs compliance, legal or per rule. The behaviour is COLREGs-aware, and no compliance rate
  is reported.
- Behaviour in channels narrower than those tested, or a width at which Rule 9 applies in law.
- Robustness to perception poorer than nominal, to wind, waves or current, or with a sensor
  other than the 2D LiDAR.
- That the rule-related behaviour is learned rather than supplied: no ablation separates them.
- A ranking of SAC over TQC, or any learner difference below about 0.05.
- That the safety layer guarantees collision avoidance or raises success.
- A field success rate comparable with the test set's.
- Multi-vessel encounters, or operation outside one calm basin.
- Coverage completeness, unless the optional study in section 5 is run.

## 5. Work still to run for the thesis

| # | Item | Status | Chapter |
|---|---|---|---|
| 1 | Static evaluation of the deployed policy by obstacle count, against LOS-DWA: evaluation only, no training | Recommended; to run in the repository | 4 |
| 2 | Field trials and calibration sessions | Planned; next basin booking | 3, 8 |
| 3 | Simulated multi-leg survey with coverage completeness under avoidance (turns by a waypoint controller) | Optional, after the trials | 1, 9 |
| 4 | Encounter-context ablation (one variant, three seeds) | Only if Paper 3's reviewers ask for it | 5 |

## 6. Writing order

| Period | Chapters | Waiting on |
|---|---|---|
| Now to mid-Dec 2026 | 2; 4 (after run 1); 7, since the bench work is done; the frame section of 1 | Nothing, apart from run 1 |
| Dec 2026 to Jan 2027 | 5 and 6, alongside the Paper 3 submission; 3 as the calibration sessions report | Paper 3's final text; session S1 |
| Jan to Feb 2027 | 8 | Sessions T1 and T2 |
| Feb to mid-Mar 2027 | 9, 1 in full, abstract and front matter. Intention to Submit by 19 Mar 2027, with an abstract of at most 2,000 characters and a 40-word summary | Everything above |
| Mar 2027 | Full draft to supervisors | |
| Mar to May 2027 | Revision, formatting, submission on 14 May 2027 | |

## 7. Open decisions

| # | Decision | Status |
|---|---|---|
| T1 | Title, from the candidates below | Open |
| T2 | Citation style (numeric or author-year) and typesetting (Word with EndNote, or LaTeX) | Open |
| T3 | Whether the optional multi-leg survey (section 5, item 3) is in scope | Open |
| T4 | How generative-AI assistance is disclosed under UTAS rules | Open; check the current UTAS guidance |
| T5 | Run 1, the static evaluation | Recommended |

**Title candidates.** "LiDAR" stays out of the title, as for Papers 2-3.
1. *Autonomous Surface Survey in Confined Waters: Learning COLREGs-Aware Path Following and
   Collision Avoidance from Onboard Sensing*
2. *Deep Reinforcement Learning for COLREGs-Aware Navigation of an Autonomous Survey Vessel in
   Confined Waters*
