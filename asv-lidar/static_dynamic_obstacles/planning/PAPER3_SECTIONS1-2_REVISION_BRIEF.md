# Paper 3, Sections 1–2: revision brief (current formulation, field trials included)

**Date:** 2026-10-14 (revised the same day with the author's decisions on the open items,
section 1.7). **Document to revise:** `Paper3_Sections1-2_Introduction_Formulation.docx`
(folder `PhD/3 Second Journal Paper - Dynamic Obstacle/`; the combined draft of 29 September
2026, written for baseline-v2).

**Purpose.** Bring Sections 1 and 2 in line with:
- the frozen formulation actually trained, **baseline-v3** (formulation digest
  `03a9c0c22f53ee0b`; the equations are baseline-v2's, and the curriculum overlay adds stages 6–7);
- the **framing decision** of 2026-10-10 (a formulation and one deployed policy, with the other
  learners and the classical methods as baselines);
- the **COLREGs reporting decision** of 2026-10-08 (snapshots, no compliance rates);
- the current **terminology**;
- the **decisions of 2026-10-14**: field trials in this paper; no channel-width sweep and no
  perception-degradation study; propulsion in RPM; target speeds reported as in the data sets;
  the 6.3 m head-on case explained; the field-trial design and the safety-on evaluation left as
  placeholders.

**Reference files** (attach with the document): `MASTER_ARTIFACT_FACTS.md` (verified values for
the vessel, comparators, perception, safety layer, curriculum and data sets),
`FORMULATION_EQUATIONS.md` (equations), `PAPER3_DRAFT_SKELETON.md` (framing, scope decisions
S1–S17, the COLREGs scope table), `PAPER3_ALGORITHMS.md` (the algorithm boxes, decision S17). **Precedence** where sources disagree: this brief, then
`MASTER_ARTIFACT_FACTS.md`, then `FORMULATION_EQUATIONS.md` (its rudder line, δc = 100 aδ, is
superseded by the facts and this brief), then the skeleton, then the draft. **Symbols:** keep the
draft's names where `FORMULATION_EQUATIONS.md` uses others — beam ranges d (file: ρ), smoothness
factor η (file: σt), bow factor ω (file: σbow), achieved alteration adone (file: At), obstacle
clearance dobs (file: dclr), class 𝒞 — the list in the draft's comment 3, which is otherwise
deleted.

---

## 0. How to edit

- Edit the document in place. Keep its style: MDPI, US spelling ("maneuver", "behavior"),
  impersonal academic voice, equation and table numbering, figure placeholders.
- Change what this brief lists, plus the consistency fixes those changes force elsewhere in
  Sections 1–2. Do not shorten the literature review. Do not add results: Sections 1–2 state
  the design, not outcomes.
- Mark every change (tracked changes, or highlight), so each edit can be reviewed.
- **Symbols:** introduce no symbol that already means something in Sections 1–2. Taken: λ (the
  geometric scale factor, Section 2.1.5), σ (the required turn sense), T (the episode length; T0,
  Ts, Teng), k (the lidar sector index), w with a subscript (the reward weights; ws, the safe
  width), m with a subscript (Eq. 24b), τ (a time variable, Eqs. 9 and 12), c with a subscript
  (cwall, ck). The one new symbol in this brief, dwall (Section 2.2.4), is unused elsewhere;
  every other new quantity is stated in words.
- **Markers:**
  - Keep every existing **[CITE]** marker that this brief does not resolve.
  - Use **[PENDING — …]** for the three items the author will supply later (the field-trial
    design, the safety-on evaluation and the safety-layer algorithm). Copy the wording given
    below verbatim.
  - Use **[CHECK — …]** only where this brief says so.
- At the end of the document, add a short list of the markers that remain.

---

## 1. Decisions that apply throughout

### 1.1 Framing: a formulation and a deployed policy

The paper develops a formulation and deploys one policy trained on it. It is **not** a
learner-comparison study.
- **Proposed method:** the frozen formulation and the deployed policy.
- **Deployment choice:** the deployed policy is the better of SAC and TQC on the held-out test
  set (test set v4), judged on the three-seed mean. The deployed seed is the one with the best
  validation score.
- **Baselines:** PPO, recurrent PPO, the non-deployed one of SAC and TQC, and two classical
  methods, all compared on the same test set.
- **Selection rules:** checkpoints are chosen on the validation set. Because the test set also
  chooses the deployed learner family, the text says so explicitly. The field trials are the
  independent evaluation of the deployed policy.
- **Wording to replace:** sentences that present "comparing four learners" as the objective or
  the main contribution.

### 1.2 Sensing: the formulation is not tied to LiDAR

- The observation is stated for onboard perception in general: pooled range returns, target
  tracks and a map boundary.
- This study implements it with a 360° two-dimensional LiDAR. Another range sensor with its
  own detection and tracking, such as radar, could supply the same observation. That is a
  design property, not a tested result, and must not be claimed as one.
- "LiDAR" stays out of contribution names and framing sentences, and stays wherever the text
  describes the implementation (sensor model, pipeline, figures).

### 1.3 COLREGs reporting: snapshots, not compliance rates

- The rule list stays: selected requirements of Rules 8, 9, 13, 14, 15/16 and 17(a)(i). Each
  rule is stated as exactly what the reward encodes (Table 2, section 4 of this brief).
- The behavior is **COLREGs-aware**, never "COLREGs-compliant". Published titles that contain
  "compliant" stay as published.
- No per-rule compliance or violation rates are reported. COLREGs behavior is shown through
  snapshots of representative encounters, each checked against track-based rule conditions.
- **The reason, to state once in Section 1.3:** in confined water with static obstacles and
  with targets that may not follow the rules, a rule-consistent maneuver is not always
  feasible. A compliance rate would count those infeasible cases as violations.
- Quantitative results cover task completion, collision type and target-ship safety:
  target-ship collision, ship-domain intrusion, closest approach and hull clearance.

### 1.4 Field trials are in this paper

- The paper gains a field-trials section after the simulation results. The roadmap becomes:
  1. Introduction
  2. Formulation
  3. Learners, training and evaluation design
  4. Simulation results
  5. Field trials
  6. Discussion
  7. Conclusion
- The trial design is not fixed yet. Wherever the text needs it, insert
  **[PENDING — field-trial design: target vessel, encounter types, layouts, number of runs]**
  rather than inventing it.
- Known design facts that may be stated:
  - The policy runs on the vessel through the shore-side bridge.
  - The observation is built from the real LiDAR scan and pose by the same perception modules
    as in simulation.
  - The same runtime safety layer code is available to the operator as a latched stop.

### 1.5 Terminology (replace throughout Sections 1–2)

| Draft wording | Replace with |
|---|---|
| development set | validation set |
| headline evaluation / held-out scenarios (as a set name) | test set (the held-out test set; "test set v4" only where a version must be named) |
| frozen-like, field-style (if present) | decoupled, coupled |
| compliant turn sense, compliant response, compliant turn, compliant pass | required turn sense, required response, required turn, required side; "rule-consistent pass" where the maneuver itself is meant |
| lawful (avoidance action, slowdown) | rule-consistent |
| COLREGs-compliant (own behavior), learned compliance | COLREGs-aware; learned COLREGs-aware behavior |
| compliance metrics, encounter violations (as reported results) | remove, or "rule-related behavior shown by snapshots" (section 1.3 above) |
| supervisor, stopping supervisor | runtime safety layer |
| two million / 2 × 10⁶ environment steps; five curriculum stages | 3.0 million (3.0 × 10⁶) environment steps; seven curriculum stages |
| LiDAR-tracked encounters (framing sentences) | encounters tracked by onboard range sensing (a 2D LiDAR in this study) |
| parallel-walled channel | variable-width channel |
| width as a "geometric experimental variable" | width as a geometric quantity (there is no width sweep, section 1.7) |

Define **decoupled** and **coupled** once, in Section 2.1.4:
- **decoupled:** the static obstacles and the encounter occur one after the other;
- **coupled:** the encounter takes place among three static obstacles.

### 1.6 Numbers that changed

| Item | Draft | Current |
|---|---|---|
| Frozen formulation | baseline-v2 (comment 3) | baseline-v3, digest `03a9c0c22f53ee0b`; equations unchanged, curriculum extended |
| Training budget | 2 × 10⁶ steps, 5 stages | 3.0 × 10⁶ steps, 7 stages; boundaries at 0.16, 0.36, 0.64, 1.0, 1.5, 2.0 M (fractions 6.4, 14.4, 25.6, 40, 60, 80 % of a 2.5 M base budget); stage 7 from 2.0 M to the end |
| Headline evaluation | 800 held-out constant-velocity scenarios per seed | test set of 1,000 episodes: 664 decoupled (basin and channel encounters; constant-velocity, reactive and noncompliant targets) + 336 coupled (three-obstacle layouts; constant- and varying-speed targets); every episode passes a feasibility check |
| Checkpoint selection | development set | validation set, 270 episodes (120 decoupled + 150 coupled), score = goal rate − 2 × collision rate, safety layer off, every 2 × 10⁵ steps |
| Classical comparators | COLREGs-VO, a narrow-channel adaptation of it, LOS + DWA | COLREGs-VO (Kuwata et al., open-water roles) and LOS guidance with heading PID and a DWA avoidance layer (obstacles only, no rules); the narrow-channel VO adaptation is **dropped** |
| Rudder command (Eq. 2) | δc = 100 aδ (% of full deflection) | δc = −40° · aδ; δ is positive to port in the identified model, so a positive aδ turns the vessel to starboard |
| Propulsion command (Eq. 2) | RPM | unchanged: RPM, nc = clip(6 + 6an, 0, 12) |
| Training targets | constant velocity | never give way; constant velocity, except that in stage 7, 30 % of the targets in coupled layouts change speed once |
| Target speeds | "0.20–0.75 m/s" | by encounter type, as in the training, validation and test sets: overall about 0.22–1.11 m/s (section 4, Section 2.1.4) |
| Width thresholds | 3.8 / 6.3 / 4.9 m, tested by a width sweep | crossing 7.60 m, head-on 3.80 m (target on its own side) or 6.30 m (target on the centerline), overtaking 4.26 m; design reasoning, not tested by a sweep (section 4, Section 2.2.4) |
| Lidar branch (Sections 2.3.1–2.3.2, Table 6) | "static returns only" | all gated returns, static obstacles and the target hull alike (pooled before segmentation and tracking); the target also enters the target branch as a track |
| Runtime layer | "evaluated both with and without it" | off in training and in every result reported for the learned policy; the safety-on evaluation is a placeholder (section 1.7) |

### 1.7 Decisions on the open items (2026-10-14)

- **Field-trial design:** pending. Use the [PENDING — field-trial design …] marker of section
  1.4.
- **Channel-width sweep and perception-degradation study:** excluded from the paper, which is
  already long.
  - Remove them from the objective, the central questions, the contributions, the scope, the
    gap paragraph and Section 2.2.4.
  - The geometric framework stays as part of the formulation: it shapes the observation and
    the reward. Its width thresholds stay as design reasoning that explains where the test
    set's channel encounters fall, not as predictions tested by a sweep.
  - The perception pipeline stays as described: the policy is trained and evaluated on tracked,
    noisy perception. Its robustness to further degradation is not studied separately.
- **Propulsion command unit:** keep RPM, as in the draft.
- **Target speeds:** report them as the training, validation and test sets realize them, by
  encounter type (Section 2.1.4 below).
- **The 6.3 m head-on threshold:** keep it and explain it alongside the 3.80 m case (Section
  2.2.4 below).
- **Safety-on evaluation:** to be finalized and then reported in the paper. Use
  **[PENDING — safety-on evaluation: to be finalized and reported]** in Section 2.3.4.
- **References:** the author checks every reference, including DOIs. Add the new entries
  without DOI markers.

---

## 2. Header line and organization

- **First line of the document** — replace with: "Paper 3 — Sections 1–2: Introduction and
  Formulation (combined draft, revised 14 October 2026 for the baseline-v3 formulation, the
  deployed-policy framing and the field trials). References in MDPI style, numbered by first
  citation. Highlighted [PENDING], [CHECK] and [CITE] markers are open items."
- **Working title** (if a title line is added): "A Deep Reinforcement Learning Formulation for
  COLREGs-Aware Path Following and Collision Avoidance of Autonomous Surface Vessels in Confined
  Water: From Simulation to Field Trials".
- **Section 1.3, last paragraph (organization)** — replace with: "The remainder of the paper is
  organized as follows. Section 2 presents the formulation: the problem and workspace, the
  COLREGs scope and the geometric framework for constrained encounter responses, and the frozen
  perception, observation, reward, runtime safety layer and curriculum. Section 3 describes the
  learners, the training protocol, the classical comparators and the evaluation design.
  Section 4 reports the simulation results on the held-out test set. Section 5 reports the field
  trials of the deployed policy. Section 6 discusses the findings and their limitations, and
  Section 7 concludes."
- **Comments:**
  - Delete comment 2 (field-section placement, now done) and comment 3 (baseline-v2 note, now
    superseded).
  - Keep comments 0 and 1.
  - Update comment 4's reference count after the new references are added.

---

## 3. Section 1 — Introduction

### 1.1 Problem statement and importance

- **Paragraph 3 (perception)**, first sentences:
  - Replace: "When these quantities are inferred from onboard light detection and ranging
    (LiDAR), their reliability depends on …"
  - With: "When these quantities are inferred from onboard range sensing, such as light
    detection and ranging (LiDAR) or radar, their reliability depends on …"
  - Keep the rest of the paragraph.
- **Paragraph 4 (the combined problem)** — append after the last sentence: "A policy trained in
  simulation is then deployed on the model-scale vessel in that basin, so that the formulation
  is evaluated both numerically and in physical encounter trials."

### 1.2.1 Classical and explicitly constrained collision avoidance

- **Last sentence of the section** — replace with: "In this study, a COLREGs-aware VO method
  following Kuwata et al. [19], which keeps the open-water role assignment, and line-of-sight
  guidance with a DWA avoidance layer, which avoids obstacles without encoding the rules, serve
  as the classical references; both receive the same perception as the learned policies."

### 1.2.2 Deep reinforcement learning for path following and COLREGs-aware navigation

- **Last paragraph, first sentence:**
  - Replace "interpreting learned compliance" with "interpreting learned COLREGs-aware
    behavior".
  - Make no other change.

### 1.2.3 Restricted and inland waterways

- **Paragraph 2, last sentence** — replace with: "The present work differs in its model-scale
  encounters perceived through onboard range sensing and tracking (a two-dimensional LiDAR in
  this study), a single learned rudder-and-propulsion policy operating on processed
  observations, an evaluation organized around maneuver admissibility in confined water, and
  physical trials of the deployed policy."
- **Paragraph 3:**
  - Replace "Width is treated as a geometric experimental variable, not as a universal legal
    criterion defining a narrow channel." with "Width is treated as a geometric quantity, not as
    a universal legal criterion defining a narrow channel."
  - Keep the rest of the paragraph.

### 1.2.4 Onboard sensing, integrated control, and physical validation

- **Last paragraph, last three sentences** — replace with: "However, the earlier static-obstacle
  experiments do not establish behavior in encounters with a moving vessel, and successful
  simulation with synthetic range data does not substitute for physical encounter trials. The
  present study therefore evaluates the deployed policy in basin trials with a moving target
  vessel (Section 5) and reports that transfer evidence separately from the numerical results.
  [PENDING — field-trial design: target vessel, encounter types, layouts, number of runs]"

### 1.2.5 Evaluation practice and the research gap

- **Paragraph 1:**
  - Replace "argues for declared, behavior-level compliance metrics rather than claims of legal
    compliance" with "argues for stating the implemented rule interpretation explicitly and for
    reporting rule-related behavior without claims of legal compliance".
  - After "… separates checkpoint selection from held-out evaluation.", insert: "Checkpoints
    are selected on a validation set. The held-out test set is used once per policy and also
    decides which learner family is deployed; this is stated rather than hidden, and the field
    trials provide the independent evaluation of the deployed policy."
- **Paragraph 2 (the gap):**
  - Replace "This paper examines a shared formulation in which a tracked LiDAR encounter and
    map-derived navigable boundaries inform a learned control policy" with "This paper examines
    a formulation in which an encounter tracked by onboard range sensing and map-derived
    navigable boundaries inform a learned control policy".
  - Replace the end of that sentence ("… and performance is measured separately under changes
    in channel width, perception quality, target behavior, and learner") with "…, and the policy
    deployed from it is evaluated on a held-out test set against other learners and classical
    methods, across target behaviors, and then in physical trials".
  - Optional, after the classical / learned / comparison clauses: "; and physical encounter
    trials of learned COLREGs-aware policies remain rare [VERIFY against Section 1.2.4]".

### 1.3 Scope, objective, and contributions

- **Objective paragraph** — replace with: "The objective is to develop a deep reinforcement
  learning formulation, driven by onboard perception, for path following, static-obstacle
  avoidance, and COLREGs-aware two-vessel encounters in confined water, and to deploy a policy
  trained on it on a model-scale vessel. Four learners—PPO, recurrent PPO, soft actor–critic
  (SAC), and truncated quantile critics (TQC)—are trained on the same frozen formulation; the
  deployed policy is the better of SAC and TQC on the held-out test set, and the remaining
  learners and two classical methods serve as baselines on the same set. The central questions
  are whether the policy completes the navigation task while producing encounter-consistent
  behavior, including in channels and among static obstacles where the room to maneuver is
  limited; which learner should be deployed, and which learner differences remain meaningful
  across training seeds and target behaviors; and how the deployed policy performs in physical
  basin trials."
- **Scope paragraph:**
  - Replace "and in parallel-walled channels" with "and in channels of variable width".
  - Replace "Target information is estimated from onboard two-dimensional LiDAR; the policy is
    not supplied with AIS or simulator ground-truth target states." with "Target information
    is estimated by onboard perception, implemented here with a two-dimensional LiDAR; the
    observation interface (pooled range returns, target tracks, and a map boundary) could
    equally be supplied by another range sensor, such as radar, through its own detection and
    tracking, a design property that is not tested here. The policy is not supplied with AIS or
    simulator ground-truth target states."
- **Regulatory-scope paragraph** — replace its last sentence ("Compliance is assessed through
  declared behavioral metrics, not legal certification.") with: "Each rule is stated as exactly
  what the reward encodes (Section 2.2.1). The resulting behavior is described as COLREGs-aware,
  not COLREGs-compliant, and is shown through snapshots of representative encounters rather than
  per-rule compliance rates: with static obstacles in confined water, and with targets that may
  not follow the rules, a rule-consistent maneuver is not always feasible, and a compliance rate
  would count such cases as violations."
- **Training and evaluation paragraph** — replace with: "Training targets never give way; they
  hold constant velocity, except that in the final curriculum stage 30 % of the targets in
  coupled layouts change speed once. Reactive and noncompliant target behaviors appear only in
  evaluation. The headline evaluation is a held-out test set of 1,000 episodes: 664 decoupled
  episodes, in which the static obstacles and the encounter occur one after the other (basin and
  channel encounters with constant-velocity, reactive, and noncompliant targets), and 336 coupled
  episodes, in which the encounter takes place among three static obstacles (constant- and
  varying-speed targets). Every test episode passes a feasibility check: some maneuver within the
  policy's action space, started once the target is first tracked, reaches the goal with
  clearance. Static-layout route checks screen out blocked geometric paths but are not
  interpreted as proof that every dynamic encounter is avoidable. An engineered runtime safety
  layer is described separately: it is off during training and in every result reported for the
  learned policy, and its interventions are not attributed to the policy. The deployed policy is
  then tested in basin trials on the model-scale vessel [PENDING — field-trial design: target
  vessel, encounter types, layouts, number of runs]."
- **"Within this scope, the methodological contributions are as follows."** — change to
  "Within this scope, the contributions are as follows."
- **Contribution 1** (keep the name "A sensor- and encounter-aware learning formulation"):
  - Replace "tracked LiDAR target information" with "target tracks from onboard range sensing".
  - Make no other change.
- **Contribution 2** — replace the whole item with: "**A geometric framework for constrained
  encounter responses.** The formulation relates the available maneuvering space to an
  encounter-consistent course alteration, holding astern, or speed reduction where predicted
  clearance supports it, and makes that judgment observable to the policy through an
  admissibility test and a stopping test. Width thresholds derived from the same geometry
  (Section 2.2.4) state, in ship breadths, where each open-water maneuver stops fitting, without
  inferring legal precedence from channel width alone."
- **Contribution 3** — replace the whole item with: "**A deployed policy, chosen and benchmarked
  fairly.** PPO, recurrent PPO, SAC, and TQC are trained on the frozen formulation under one
  protocol (3.0 million environment steps, three training seeds, checkpoints selected on a
  separate validation set). The policy for deployment is the better of SAC and TQC on the
  held-out test set, where it is benchmarked against the other learners and against a
  COLREGs-aware velocity-obstacle method [19] and line-of-sight guidance with
  proportional–integral–derivative heading control and a dynamic-window avoidance layer [21], all
  using the same perception pipeline. Seed variability is reported with every learner result."
- **Contribution 4** — replace the whole item with: "**An evaluation design separating task
  completion, safety, and robustness to target behavior.** The held-out test set, with decoupled
  and coupled scenarios and constant-velocity, reactive, and noncompliant targets, distinguishes
  task completion from collision type and target-ship safety, and COLREGs-aware behavior is
  shown through encounter snapshots."
- **New contribution 5** — add after 4: "**Physical encounter trials.** The deployed policy
  runs on the model-scale vessel in the basin, with its observation built from the real LiDAR
  scan and pose by the same perception modules as in simulation, and is evaluated in encounters
  with a moving target [PENDING — field-trial design: target vessel, encounter types, layouts,
  number of runs]. The transfer evidence is reported separately from the numerical results."
- **"Overall, …" paragraph:** keep. Optionally end with "…, rather than an assumed advantage of a
  particular algorithm, and on whether that behavior carries over to the vessel."
- **Organization paragraph:** section 2 of this brief.

---

## 4. Section 2 — Formulation

### 2 (introduction paragraph)

- Change "It is frozen before any learner comparison" to "It is frozen before any learner is
  trained (baseline-v3)".
- Change the last sentence to: "Network architectures, hyperparameters, training seeds,
  classical comparators, and data sets are described in Section 3, and the field-trial setup in
  Section 5."

### 2.1.1 Task and decision process

- **Eq. (2) and the paragraph after it** — replace with:
  - Equation: δc = −40° · aδ,  nc = clip(6 + 6an, 0, 12)   (2)
  - Text: "where δc is the commanded rudder angle, positive to port in the identified vessel
    model, so that a positive aδ turns the vessel to starboard, and nc is the propeller speed in
    revolutions per minute (RPM). The commanded angle passes through the actuator dynamics of
    Section 2.1.2. Cruise, n = 6 RPM, gives Uref = 0.558 m/s on the nominal model, so the throttle
    spans zero to twice the cruise setting; it has no astern range, and only the runtime safety
    layer can reverse (Section 2.3.4)."
  - Remove "δmax = 40°" as a separate definition, since it is now in the equation.
- **"Propulsion is a continuous action by design"** paragraph — replace "a lawful avoidance
  action" with "a rule-consistent avoidance action".

### 2.1.2 Vessel and dynamics

- **Paragraph 1** — replace the sentence "the actuator model comprises a rudder-servo rate
  limit, a first-order lag, a transport delay, and a thrust map" with: "the actuator model
  comprises a rudder transport delay of 0.73 s, a first-order servo lag with a time constant of
  0.895 s, and a servo rate limit (identified at about 2,985°/s, so effectively non-binding; the
  20°/s command limiter of the vessel's bridge is not applied in simulation), and a quadratic
  thrust map, 16.06 N at 12 RPM scaling with the square of the propeller speed (the identified
  low-speed boost is zero)."
- **Integration** — add: "Each 0.1 s physics step is integrated by fourth-order Runge–Kutta with
  sub-steps of at most 0.05 s."
- **Domain randomization** — replace "and the randomization of the dynamics parameters over
  their identified uncertainty in every training episode (randomization scale 1.0)" with: "and
  the randomization of the dynamics in every training episode: two of the 28 bootstrap solutions
  of the identification are drawn at random and blended with a weight drawn uniformly from [0, 1], and each
  parameter then receives Gaussian jitter of 5 % of its bootstrap standard deviation. Blending
  whole parameter vectors preserves the correlations between the yaw parameters, so that every
  draw is a physically consistent vessel."
- **Citations** — cite Tobin et al. (2017) [new reference] with Slawik et al. [56] for domain
  randomization.
- **Resolve** the [CHECK — parameter list …] marker. Replace it with: "[CHECK — Appendix D:
  identified parameters with their bootstrap spread]". The actuator values above are confirmed.
- **Propulsion paragraph:** keep as written (RPM).
- **Limitation paragraph** — keep. Append: "Section 5 tests the deployed policy at that
  operating speed on the vessel."
- **Table 1:**

  | Row | New value |
  |---|---|
  | Length overall | 1.725 m |
  | Draft | 0.193 m (or keep 0.19 m) |
  | Rudder command | δc = −40° aδ (δ positive to port; positive aδ turns to starboard); full deflection 40° |
  | Dynamics | add "transport delay 0.73 s, lag 0.895 s, RK4 sub-steps ≤ 0.05 s" |
  | Propulsion | unchanged: nc = clip(6 + 6an, 0, 12) RPM |
  | Domain randomization | "Per episode: blend of two of 28 bootstrap parameter sets (weight uniform on [0, 1]) with 5 % jitter" |

### 2.1.3 Workspace

- **Channel-mode paragraph** — replace its first two sentences with: "The second geometry,
  channel mode, places a straight reference path in a 25 m channel whose width, measured wall to
  wall, varies along its length by up to a factor of 1.8 (piecewise linear between two to four
  control points); the channel polygon is itself the navigable boundary. It is used only for the
  three classes whose admissible response depends on width (head-on, crossing, and overtaking),
  with probability 0.15 in curriculum stage 3 and 0.25 from stage 4 on, and nominal widths reach
  3.5–10 m from stage 5 (Section 2.3.5)."
- **Path offset** — replace "offset from the centerline by up to ±30 % of the local half-width
  with a bias to starboard (mean +12 %)" with "offset from the centerline by a fraction of the
  half-width of the narrowest section, drawn once per channel from a normal distribution with
  mean +0.12 and standard deviation 0.15 and clipped to ±0.30 (positive to starboard)".
- **"Because the boundary is virtual …"** sentence: keep. It supports the field trials. Append:
  "(Section 5)".
- **Figure 1 caption and placeholder:**
  - Panel (b): "a variable-width channel".
  - Add a panel (c): a coupled layout, with three panels and the encounter among them.
  - Caption: "(a) basin mode, decoupled; (b) channel mode; (c) basin mode, coupled layout".

### 2.1.4 Traffic and static obstacles (also defines decoupled and coupled)

- **Paragraph 1** — replace with:

  > "Each episode contains at most one target vessel, of the same class and hull outline as the
  > own ship. Training targets never give way, so the policy cannot rely on the other vessel's
  > cooperation; they hold constant velocity, except that in curriculum stage 7, 30 % of the
  > targets in coupled layouts change speed once (a single speed step with bounded acceleration).
  > Reactive and noncompliant targets are reserved for evaluation (Section 3.4).
  >
  > The target's speed is drawn by encounter type as a ratio of the own ship's cruise speed
  > Uref = 0.558 m/s, from the bands in Table 3a. The same bands generate the training, validation
  > and test sets, and the realized speeds agree across the three: target speeds span about
  > 0.22–1.11 m/s in all of them. The being-overtaken band is drawn up to 2.20 times cruise, but realized
  > ratios do not exceed about 2.0, so those targets move at 0.84–1.11 m/s, at most about twice
  > the cruise speed. Crossings realize ratios up to about 1.4 in the decoupled scenarios and up
  > to about 1.2 in the coupled layouts, whose targets are kept within 0.20–1.10 m/s."

- **New Table 3a** (after paragraph 1), caption "Target speeds by encounter type: generator band
  and realized range in the training, validation and test sets". Use the realized ranges below
  (training: a sample of 40 decoupled draws per type; validation and test: every episode):

  | Encounter type | Speed ratio U_TS/Uref (band) | Realized speed, training | Validation set | Test set |
  |---|---|---|---|---|
  | Head-on | 0.70–1.30 | 0.39–0.72 m/s | 0.39–0.72 m/s | 0.39–0.72 m/s |
  | Crossing | 0.60–1.40 | 0.34–0.67 m/s (sample) | 0.34–0.75 m/s | 0.33–0.77 m/s |
  | Own ship overtaking | 0.40–0.55 | 0.22–0.30 m/s | 0.22–0.30 m/s | 0.22–0.31 m/s |
  | Being overtaken | 1.50–2.20 (realized ≤ about 2.0) | 0.85–1.11 m/s | 0.84–1.07 m/s | 0.84–1.10 m/s |
  | Null (no conflict) | 0.85–1.15 | 0.48–0.64 m/s | 0.49–0.64 m/s | 0.48–0.64 m/s |
  | No target | — | — | — | — |

  (The training row is a sample; a full-range statement for training is the generator band.
  The validation and test rows are complete.)
- **Paragraph 2** (episode shares) — keep. Add after the crossing sentence: "Each crossing draw
  is first assigned to be unescapable with probability 0.20, and its kinematics are redrawn until
  a scripted check of holding course for 1.5 s and then either stopping or a 60° alteration in
  the required direction agrees; the label is kept with the episode."
- **Paragraph 3** (generation by class) — append: "The miss distance at the CPA is drawn
  explicitly (0–2.0 m; 0–2.5 m for crossings; at or above the contact-free floor for being
  overtaken), so that the policy also meets encounters that need no alteration. The null target
  is placed rather than solved: at 8–15 m, at a bearing of 20–160° to either side, on a course
  within ±10° of the own ship's, and accepted only if its CPA is at least 4 m or does not occur
  within the episode."
- **Paragraph 4** (static obstacles) — replace with: "Static obstacles are 1.0 m panels. In
  curriculum stages 1–5 they are placed along 25–70 % of the path with lateral offsets and kept
  clear of the stretch of own-ship travel within ±0.4 T0 of the planned CPA, where T0 is the
  drawn time to CPA, so that a panel does not decide the encounter: obstacle avoidance and
  collision avoidance occur one after the other. These are the **decoupled** scenarios. In stages
  6 and 7, 25 % and 45 % of episodes use **coupled** layouts: three panels in one of three motifs
  (a gate with an on-path panel, 40 %; a slalom, 30 %; a side panel with an on-path panel, 30 %)
  along 28–85 % of a leg (slant up to about 14°), with the encounter generated on that leg
  without the CPA guard, so that it may take place beside a panel. Elsewhere the guard is halved
  in stage 6 and removed in stage 7. Every layout admits a static route: an A∗ search [Hart,
  Nilsson and Raphael (1968), new reference] on a 0.25 m grid, with walls inflated by 0.40 m and
  panels by 0.45 m, must find a route no longer than 2.25 times the leg. In stages 6–7 every
  target episode must also pass a space-time solvability check: a forward-reachability search on
  the same grid, with the own ship as a point moving at up to 1.1 m/s and the target's
  neighborhood (1.3 m) removed at each time, must reach the goal before the episode ends. Both
  checks remove episodes that no controller could solve, but neither proves that every dynamic
  encounter is avoidable for this vessel."
  - This resolves the A∗ [CITE].

### 2.1.5 Scaling

- No change.

### 2.2.1 Rule scope

- **Paragraph 2** — replace with: "Each implemented rule is stated as exactly what the reward
  encodes, so Table 2 is a statement of the formulation's interpretation, not a claim of legal
  compliance [8]. The exclusions are stated as explicitly as the inclusions, because each of them
  bounds what the results can mean. The resulting behavior is shown in Section 4 through
  snapshots of representative encounters, each checked against track-based rule conditions,
  rather than through per-rule compliance rates (Section 1.3)."
- **Table 2** — replace these rows (wording from `PAPER3_DRAFT_SKELETON.md`, section 2.2.1):

  | Rule | Content | Treatment |
  |---|---|---|
  | 8 | Action to avoid collision | Implemented: 8(b) readily apparent action (a 20° alteration or a 30 % slowdown counts as one full action, Rule 8 term, Eq. 28); 8(d) passing at a safe distance (required passing distance dreq, Eq. 5, and the ship-domain term); 8(e) slackening speed where no alteration fits (learned, Eq. 19) and taking all way off (runtime safety layer, Section 2.3.4). 8(a) only as timing pressure: the Rule 8 penalty grows as TCPA falls below 15 s, and many confined-water encounters begin inside that horizon, so ample time is not claimed. |
  | 14 | Head-on situation | Implemented: port-to-port passing. Keeping to the starboard side of the channel (Rule 9(a)) satisfies the rule without an alteration; a starboard alteration is required only when the target is not already clear, subject to the available width (Section 2.2.4). |
  | 15 | Crossing situation | Implemented under the adopted convention: the own ship gives way to a crossing target from either side, passing astern of it — a starboard alteration for a target from starboard and a port alteration for a target from port. Stated explicitly, because a port alteration in a crossing reads as a departure from the open-water role assignment. |
  | 17 | Action by stand-on vessel | Partially: 17(a)(i) course and speed keeping when being overtaken; the hold penalty is released in extremis (DCPA < dreq, TCPA < 5 s), so the reward never pays for holding course into a collision. 17(b) is out of scope by construction (being-overtaken draws are contact-free, Section 2.1.4); 17(a)(ii) is out of scope and left to future work. |

  The other rows stay as written.

### 2.2.2 Encounter classification

- No change to the method.
- Check that "compliant" does not appear. "required turn sense" replaces it where it does.

### 2.2.3 Engagement and the latched encounter

- Replace "compliant turn sense σ" with "required turn sense σ" throughout. Rename Table 4 to
  "Required turn sense latched at engagement", and its column "Compliant response" to "Required
  response".
- **Last sentence of the σ paragraph** — replace "Under the open-water rules the own ship would
  stand on for a target crossing from port, and the classical comparators include a
  velocity-obstacle method with open-water roles [19] alongside an adaptation to the adopted
  convention, so that the effect of the convention can be measured (Section 3.3)." with "Under
  the open-water rules the own ship would stand on for a target crossing from port; the
  COLREGs-aware velocity-obstacle comparator keeps those open-water roles [19], so its behavior
  in port crossings differs from the adopted convention by design (Section 3.3)."
- **Figure 2 placeholder:** no change.

### 2.2.4 Geometric framework for constrained encounter responses

- Replace "compliant pass", "compliant turn" and "compliant maneuver" with "required pass",
  "required turn" and "rule-consistent maneuver" (Table 5 wording included).
- **Last paragraph (width and thresholds)** — replace the whole paragraph, including its
  [CHECK] marker, with:

  > "Channel width enters the framework as a geometric quantity, not as a universal legal
  > criterion defining a narrow channel. The admissibility test of Eq. (11) evaluates the room
  > online, along the actual passage; the thresholds below are fixed reference widths derived
  > from the same geometry, stated in ship breadths, that show where each open-water maneuver
  > stops fitting. They are design reasoning, not predictions tested by a width sweep. Throughout,
  > b = 1.25 m is the abeam semi-axis of the ship domain (so dreq = 2b = 2.5 m), and dwall =
  > 0.65 m is the lateral margin between each vessel's track and the wall in this reference geometry.
  >
  > *Head-on.* If the target keeps to its own side of the fairway, as Rule 9(a) requires, the two
  > vessels pass port to port with each at its station. The channel must then hold two vessels
  > abreast at the required passing distance, W ≥ dreq + 2dwall = 3.80 m (7.6 breadths), and
  > neither vessel needs to alter course: channel keeping satisfies Rule 14 (Table 5). If the
  > target instead holds the centerline, it leaves no room on either side of itself, and the own
  > ship must produce the whole required separation alone: its centerline must lie dreq to one
  > side of the target's while keeping dwall from the wall, so W/2 ≥ dreq + dwall, that is,
  > W ≥ 2dreq + 2dwall = 6.30 m (12.6 breadths). The two cases bound the head-on response. Above
  > 6.30 m, a pass at the required distance fits whatever the target does. Between 3.80 m and
  > 6.30 m, it fits only if the target keeps to its own side; otherwise the starboard alteration
  > that Rule 14 calls for is inadmissible. Slowing does not help either, because the stopping
  > test fails against a reciprocal target converging on the own ship's track (Eq. 12). Below
  > 3.80 m, no head-on pass at the required distance exists. The difference between the two
  > thresholds is the room that a target which keeps to its own side gives back.
  >
  > *Crossing.* With the own ship at the Rule 9(a) station, a quarter of the width from the
  > starboard wall, a give-way alteration needs one abeam semi-axis of starboard room, so
  > W ≥ 4(b + dwall) = 7.60 m (15.2 breadths).
  >
  > *Overtaking.* Overtaking needs the head-on width plus a margin for pose drift over the time the
  > vessels remain abreast: W ≈ 3.80 + 2 × 0.0101 m/s × 22.8 s = 4.26 m (8.5 breadths).
  >
  > In the test set, the channel encounters have nominal widths of about 6.9–10.6 m, narrowing to
  > about 5.6–6.2 m at the CPA in the narrowest cases. The head-on and overtaking encounters
  > therefore lie mostly above their thresholds, including the 6.30 m centerline case, whereas
  > channel crossings span the 7.60 m crossing threshold."

  - Add after the paragraph: **[CHECK — wall-margin convention: the reference widths measure
    dwall = 0.65 m from each vessel's track, whereas Eq. (10) measures cwall = 0.65 m from the
    hull side. Measured from the hull, each margin grows by a half-breadth (0.25 m): the head-on
    and overtaking widths grow by one breadth (to 4.30, 6.80 and 4.76 m) and the crossing width,
    where the margin enters four times, by two breadths (to 8.60 m)]**.
- **Algorithm 1 (new box, end of 2.2.4).** Insert "Algorithm 1. Encounter assessment for one
  tracked vessel at each decision step" from `PAPER3_ALGORITHMS.md`, with a lead-in sentence: "Algorithm 1
  gathers the classification, engagement and latch, and the admissibility and stopping tests of
  Sections 2.2.2–2.2.4 into the order in which they run at each decision step; its inputs are the
  perceived tracks of Section 2.3.1." Keep the box's symbols (they are the draft's) and its
  parameter table; reference it from 2.2.2 and 2.2.3 where the classifier and the latch are
  introduced.

### 2.3.1 Perception

- **Paragraph 1** — after the noise values, add: "Ranges are exact ray casts: no per-beam range
  noise or beam dropout is applied in training, pending their characterization from field data
  (Section 5)."
- **Paragraph 2 (pipeline):**
  - Replace "The remaining returns are clustered and compensated for ego-motion at the
    estimated pose." with: "The remaining returns are segmented by adaptive breakpoint detection
    [Borges and Aldon (2004), new reference] (break threshold 0.35 m plus the beam spacing at
    range; at least four points per cluster) and lifted into world coordinates at the estimated
    pose."
  - Association: "greedy global nearest neighbor within a 0.70 m gate".
  - Kalman filter: "constant-velocity Kalman filter with piecewise-constant white acceleration
    (standard deviation 0.10 m/s²) and a position measurement noise of 0.05 m (standard deviation)".
  - Resolve the [CITE] with Kalman (1960) and Bar-Shalom, Li and Kirubarajan (2001), both new
    references.
  - Static/moving test — add the hysteresis: "promoted to moving after two consecutive motion
    updates (1 s) and demoted after four quiet ones (2 s), biased toward under-detection because
    calling a panel a vessel has COLREGs consequences."
  - **Correction (what feeds the lidar branch).** The draft says the lidar branch carries static
    returns only. The frozen code pools the 27 sectors from the gated scan, **before** segmentation
    and tracking, so the pooled sectors contain every gated return: the static panels and the
    target ship's hull alike. The static/moving test decides what becomes a target track; it does
    not remove the target's returns from the sectors. (Verified on 2026-10-17: removing the target
    hull from one scan changes exactly the sectors it covers.)
    - After the gating sentence, add: "The gated scan is pooled at once into the sectors of the
      lidar branch (Section 2.3.2), so that branch contains every gated return, the target's hull
      included."
    - Replace the paragraph's last sentence, "Static returns feed the lidar branch of the
      observation, and the moving track feeds the target branch.", with: "The moving track feeds
      the target branch. A target therefore reaches the policy twice: as an obstacle at its range in
      the pooled sectors, and as a track, with its velocity, in the target branch."
- **Paragraph 3** — append: "In the field trials, the same modules run on the real scan and pose
  (Section 5)."
- **Algorithms 2 and 3 (new boxes, two separate boxes).** From `PAPER3_ALGORITHMS.md`:
  - "Algorithm 2. Feasibility sector pooling" after the gating sentence, citing Paper 2 (its
    Algorithm 1) and Meyer et al. (2020a), followed by the three changes from Paper 2 listed
    under the box: the gated input contains the target's hull; 27 non-uniform sectors assigned by
    bearing (Paper 2: 25 uniform sectors); the true 0.5° beam spacing (Paper 2: 1.205°), so pooled
    values are not comparable across the two papers.
  - "Algorithm 3. Target detection and tracking at each decision step" after the pipeline
    paragraph; the paragraph keeps its prose (segmentation, association, Kalman filter,
    static/moving test with its hysteresis) and points to the box for the order of the steps
    and the parameter values.
- **Figure 3 placeholder:** "clustering" → "segmentation". Redraw the flow as: LiDAR scan → map gate
  → (i) feasibility pooling of all gated returns → lidar branch, and (ii) segmentation →
  ego-motion compensation → association → Kalman filter → free-space static/moving test → moving
  track → target branch and the encounter module. Replace "Static returns → lidar branch" with
  "All gated returns → pooled sectors → lidar branch".

### 2.3.2 Observation

- No change to the values (70 values, six branches).
- In the context-branch paragraph and Table 6, wherever σ is described, use "required turn
  sense".
- **Lidar paragraph, first sentence** — replace "The lidar branch carries static returns only, over a
  ±135° swath; the aft 90° is left to the tracker." with: "The lidar branch pools all gated returns
  over a ±135° swath, static obstacles and the target's hull alike; the aft 90° is left to the
  tracker."
- **Target-branch paragraph** — replace "Pooled range sensing is velocity-blind: a wall and a
  closing vessel at the same range produce the same closeness. Target kinematics therefore enter
  through explicit tracking," with: "Pooled range sensing is velocity-blind: the target's hull
  registers in the lidar branch as an obstacle at its range, but a wall and a closing vessel at the
  same range produce the same closeness. Target kinematics therefore enter through explicit
  tracking,". Keep the rest of the sentence.
- **Table 6, lidar row, Contents** — replace "Static returns over ±135°:" with "All gated returns
  (static obstacles and the target hull) over ±135°:".

### 2.3.3 Reward

- No change to the equations or weights.
- **Paragraph 1:** replace "so that a compliant collision always scores worse than a
  noncompliant near miss" with "so that a rule-consistent collision always scores worse than a
  near miss that departs from the rules".
- **Carve-out paragraph:** replace "the lawful Rule 8(e) slowdown" with "the rule-consistent
  Rule 8(e) slowdown".
- **Verified-preference paragraph:** replace "a scripted 30° compliant crossing turn" with "a
  scripted 30° crossing turn in the required direction".

### 2.3.4 Runtime layer → rename to "Runtime safety layer"

Replace the trigger sentence and its [CHECK], and the evaluation sentence, with:

> "When enabled, the layer fires if an engaged give-way encounter is in extremis (DCPA < dreq and
> 0 ≤ TCPA < 5 s), the required alteration is inadmissible, and the full-astern stop of Eq. (12)
> would let the target pass clear (Dstop ≥ 1.76 m). It then commands full astern until the vessel
> has stopped (at most 0.05 m/s, within 8 s), holds zero thrust for 2–10 s, and returns control.
> The layer is off during training, so nothing it does shapes the policy, and every result
> reported for the learned policy is obtained with it off; its interventions are never
> attributed to the policy. [PENDING — safety-on evaluation: to be finalized and reported] In the
> field trials, the same code also serves as an operator-latched stop (Section 5)."

- **Appendix algorithm.** The layer gets an algorithm box in the appendix, beside the two
  classical comparators; until the layer is finalized, add only the placeholder "[PENDING —
  safety-layer algorithm: Appendix, to be added once the layer is finalized]" at the end of this
  subsection.

Keep the sentences on the overridden commands, and on the distinction between learned behavior
and external intervention.

### 2.3.5 Curriculum

- **Paragraph 1** — replace with: "Training follows seven stages over 3.0 × 10⁶ environment
  steps (Table 8). The stage boundaries are fixed timesteps at 6.4, 14.4, 25.6, 40, 60 and 80 % of
  a 2.5 × 10⁶-step base budget, and training continues in the final stage to 3.0 × 10⁶ steps.
  - The first two stages contain no target and train path following and static-obstacle
    avoidance in the basin.
  - The third introduces head-on and crossing encounters from both sides, together with null
    and no-target episodes, with times to CPA drawn from the upper part of their range; 15 % of
    the width-dependent encounter draws are placed in channels of 7–10 m.
  - The fourth adds all six episode types, raises the channel share to 25 % and narrows the
    channels to 4.5 m.
  - The fifth raises the maximum obstacle count to three, weighted toward three panels, and
    extends the channel range to 3.5–10 m.
  - The sixth and seventh add the coupled layouts of Section 2.1.4 (25 % and 45 % of episodes),
    halve and then remove the CPA guard, and in the seventh 30 % of the coupled targets change
    speed once.

  In every stage, 15 % of episodes start slow, from rest or at up to half the cruise speed.
  Reactive and noncompliant targets are evaluation-only, because a reactive opponent makes the
  training environment nonstationary and blurs the attribution of the policy's behavior to the
  formulation."
- **Table 8** — new caption "Training curriculum over 3.0 × 10⁶ environment steps (stage
  boundaries on a 2.5 × 10⁶-step base budget)", and seven rows:

  | Stage | From step | Episode types | Obstacles | Channel share and widths | Coupled layouts |
  |---|---|---|---|---|---|
  | 1 | 0 | No target | 0–1 | basin only | — |
  | 2 | 0.16 × 10⁶ | No target | 0–3 | basin only | — |
  | 3 | 0.36 × 10⁶ | Head-on, crossing (both sides), null, no target; longer times to CPA | 0–1 | 0.15; 7–10 m | — |
  | 4 | 0.64 × 10⁶ | All six types | 0–2 | 0.25; 4.5–10 m | — |
  | 5 | 1.00 × 10⁶ | All six types | 0–3 (0/1/2/3 at 0.10/0.20/0.30/0.40) | 0.25; 3.5–10 m | — |
  | 6 | 1.50 × 10⁶ | All six types | 1–3 (1/2/3 at 0.2/0.3/0.5) | 0.25; 3.5–10 m | 25 %; CPA guard halved |
  | 7 | 2.00 × 10⁶ | All six types; 30 % of coupled targets change speed once | 1–3 (1/2/3 at 0.2/0.3/0.5) | 0.25; 3.5–10 m | 45 %; no CPA guard |

- **Footnote:**
  - Keep the definition of the channel share: the fraction of head-on, crossing and overtaking
    draws placed in channels; the other episode types are always drawn in the basin, and in
    stages 6–7 the share applies to the episodes that are not coupled layouts.
  - Delete the [CHECK] marker on the stage 1–2 channel use and the obstacle counts; the stage
    1–2 basin-only use and the obstacle counts are confirmed.
- **Algorithm 4 (new box, after Table 8).** Insert "Algorithm 4. Episode generation" from
  `PAPER3_ALGORITHMS.md`, with its class-band table, and the lead-in: "Every episode, in training
  and in the validation and test sets, is drawn by Algorithm 4 from a seed in the set's own range,
  so no episode is shared between sets (Section 3)." Reference it from Section 2.1.4, where
  decoupled and coupled scenarios are defined. The test-set fairness check is not a box; it stays
  in the evaluation design (Section 3).

---

## 5. References

Add in order of first citation and renumber everything after them (MDPI style, ISO 4
abbreviations; the author completes and checks the DOIs):

- Tobin, J.; Fong, R.; Ray, A.; Schneider, J.; Zaremba, W.; Abbeel, P. Domain randomization for
  transferring deep neural networks from simulation to the real world. In *Proceedings of the
  IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2017; pp. 23–30.
  (Section 2.1.2)
- Hart, P.E.; Nilsson, N.J.; Raphael, B. A formal basis for the heuristic determination of
  minimum cost paths. *IEEE Trans. Syst. Sci. Cybern.* 1968, 4, 100–107. (Section 2.1.4;
  resolves the A∗ [CITE])
- Borges, G.A.; Aldon, M.-J. Line extraction in 2D range images for mobile robotics. *J. Intell.
  Robot. Syst.* 2004, 40, 267–297. (Section 2.3.1)
- Kalman, R.E. A new approach to linear filtering and prediction problems. *J. Basic Eng.* 1960,
  82, 35–45. (Section 2.3.1)
- Bar-Shalom, Y.; Li, X.R.; Kirubarajan, T. *Estimation with Applications to Tracking and
  Navigation*; Wiley: New York, NY, USA, 2001. (Section 2.3.1; with Kalman, resolves the tracking
  [CITE])

The two remaining [CITE] markers stay: Froude scaling of ship maneuvering models (Section 2.1.5)
and the ship-domain review (Section 2.2.2).

---

## 6. Markers that remain

| Marker | Where | What decides it |
|---|---|---|
| [PENDING — field-trial design] | 1.2.4, 1.3 (scope, contribution 5) | the author: target vessel, encounter types, layouts, number of runs |
| [PENDING — safety-on evaluation] | 2.3.4 | the author: the safety layer is being finalized, then reported |
| [PENDING — safety-layer algorithm] | 2.3.4 (box in the appendix) | the author: written once the layer is finalized |
| [CHECK — Appendix D] | 2.1.2 | identified parameters with bootstrap spread |
| [CHECK — wall-margin convention] | 2.2.4 | whether the reference widths keep the track-based 0.65 m margin or measure it from the hull (one breadth more for head-on and overtaking, two for crossing) |
| [VERIFY] | 1.2.5 gap paragraph (optional clause) | "physical encounter trials of learned COLREGs-aware policies remain rare" |
| [CITE] ×2 | 2.1.5, 2.2.2 | Froude scaling; ship-domain review |
