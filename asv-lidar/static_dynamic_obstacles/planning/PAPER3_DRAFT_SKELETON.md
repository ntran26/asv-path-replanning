**Paper 3 — Draft Skeleton**

*Repositioned to two-vessel encounters with static obstacles · Revision 4 (2026-09-28: the literature review moved into the Introduction and the paper renumbered to the six-section roadmap of Introduction draft 4; revision 3, 2026-09-27, rewrote the introduction, scope and novelty against the frozen formulation baseline-v2)*

> **Status note (2026-09-28, revision 4).** The Introduction is now written:
> **`planning/Paper3_Introduction_draft4.docx`** (draft 4, 28 Sep) is the
> authoritative text of Section 1; section 1 below summarises it and keeps the planning
> notes that sit behind it. The literature review is no longer a section of its
> own: it is **section 1.2** (five subsections), and the paper follows draft 4's
> roadmap -- **1 Introduction, 2 Formulation, 3 Learners, training and
> evaluation design, 4 Results, 5 Discussion, 6 Conclusion**. Old -> new:
>
> | Revision 3 | Revision 4 |
> |---|---|
> | Section 1.1 Motivation, section 1.2 Gap | Section 1.1 Problem statement; section 1.2.5 Evaluation practice and the research gap |
> | Section 2 Related work (2.1-2.5) | Section 1.2 Literature review (1.2.1-1.2.5) |
> | Section 1.3-1.6 Contributions, RQs, novelty, not claimed | Section 1.3 Scope, objective and contributions |
> | Section 3 Problem formulation; section 4.1-4.4, 4.6 Methodology | Section 2 Formulation (2.1 problem and workspace; 2.2 COLREGs scope and constrained encounter responses; 2.3 perception, observation, reward, vessel model and curriculum) |
> | Section 4.5 Policy, section 4.7 Training, section 5 Scenario generation and evaluation design | Section 3 Learners, training and evaluation design |
> | Section 6 Numerical studies (6.1-6.9) | Section 4 Results (4.1-4.9) |
> | Section 7 Field experiments | **parked** -- not in draft 4's roadmap (S13) |
> | Section 8 Discussion, section 9 Conclusion | Section 5 Discussion, section 6 Conclusion |
>
> **What draft 4 changes beyond the move** (carried into this revision):
>
> - **Four contributions**, not seven (section 1.3): the formulation; a geometric
>   framework for constrained encounter responses (absorbs the width sweep); the
>   controlled four-learner comparison with classical references; an evaluation
>   design separating safety, rule-related behaviour and robustness (absorbs the
>   perception study and the protocol). Old C1-C7 are mapped in section 1.3.
> - **Rule 9 is stated as an adopted operating convention**, not a legal finding:
>   operating bounds "are not assumed to establish the legal applicability of
>   Rule 9"; channel width is "a geometric experimental variable, not a universal
>   legal criterion"; compliance is measured by declared behavioural metrics, not
>   legal certification. Section 2.2.3 is renamed accordingly ("constrained encounter
>   responses"; was "Rule precedence"); its tables are unchanged.
> - **Comparators:** COLREGs-VO (Kuwata et al., 2014), "a narrow-channel
>   encounter-specific adaptation" of it (the encounter-specific VO,
>   `src/classical/encounter_vo.py`), and LOS-PID with a DWA layer, all on the same
>   perception pipeline. The prior-paper SAC and the NMPC option are not named in
>   draft 4 (section 3.3).
> - **Physical transfer is "planned"**: the design "provides a basis for the
>   planned physical-transfer assessment", and the roadmap has no field section.
>   Whether RQ4, claim C-6 and the old C7 stay in this paper is **open (S13)**;
>   until decided, old section 7 is kept, unnumbered, after section 4.
> - **Four central questions** (section 1.3): task plus encounter-consistent behaviour
>   (RQ1), response as room decreases (RQ6, new), perception degradation (RQ3),
>   learner differences across seeds (RQ5). RQ2 (the ablation) and RQ4 (field)
>   are not stated in draft 4.
>
> Carried from revision 3: framing **formulation plus a four-learner comparison**
> (PPO, RecurrentPPO, SAC, TQC) under baseline-v2 (digest `3d697858e95e5adf`);
> **2 M steps, 3 seeds**, best-on-development checkpoint (seed 0 of all four
> trained); **suite 3.4** (800 constant-velocity headline episodes per seed plus a
> robustness set); the R4 width sweep (3.5-10 m); the Paper 2 deployment-layout
> set and the field fine-tune (F104-F107), status open (S12). **Open:** A35 (the
> COLREGs-VO comparator's side rule), S12, S13.

**Working title:** Sensor-Realistic COLREGs-Compliant Path Following and Collision Avoidance for Autonomous Surface Vessels in Confined Water: A Deep Reinforcement Learning Formulation and Learner Comparison

**Alternative titles:**
- COLREGs-Compliant Collision Avoidance in Confined Water from Onboard LiDAR: One Formulation, Four Learners
- When the Channel Decides: Rule 9 Precedence in Learned COLREGs Collision Avoidance from Onboard Range Sensing \[check against draft 4's softer Rule 9 stance before use\]

**Authors:** H. N. Tran, H. Nguyen, P. King, M. Tran (order TBC)

**Target venue:** \[TBC — Ocean Engineering / IEEE JOE / Journal of Field Robotics / JMSE\]

Status legend

| **Marker** | **Meaning**                                                        |
|------------|--------------------------------------------------------------------|
| \[TBC\]    | Decision not yet made                                              |
| \[O#\]     | Blocked on open decision in 00_PAPER3_INDEX_AND_PROTOCOL.md        |
| \[RESULT\] | Awaiting experimental result — table pre-committed, values pending |
| \[VERIFY\] | Claim to check against literature before submission                |

Scope decisions carried into this revision

| **\#** | **Decision**                                                                                                                 |
|--------|------------------------------------------------------------------------------------------------------------------------------|
| S1     | Two-vessel encounters only: one own ship, **one** moving target, and zero to three static obstacles per episode.             |
| S2     | COLREGs scope: selected requirements of Rules 8, 9, 13, 14, 15/16 and 17(a)(i) (draft 4 section 1.3). **Rule 9** shapes the admissible response in narrow water through an adopted operating convention, and **Rule 8** governs action quality (timing, readiness, 8(e) slowing). |
| S3     | The own ship gives way in every crossing: an explicit **narrow-channel operating convention** motivated by the conditional non-impeding requirement of **Rule 9(b)** (A17), not the open-water Rule 15/17 role table and not a consequence of vessel length. |
| S4     | Six episode types: no target, null (a target that never meets the own ship), head-on, crossing, overtaking, being overtaken. |
| S5     | Rule 17(a)(i) course-keeping for the being-overtaken class. Rule 17(b) (last-moment action) is out of scope **by construction**: every being-overtaken draw passes at or above a contact-free floor (baseline-v2, F96). |
| S6     | The target branch is a shared-weight slot encoder, so a multi-vessel extension costs a retrain, not a redesign.                |
| S7     | **Onboard sensing only.** Target state is estimated from a 360° 2D LiDAR (motion classifier, tracker); no AIS, no ground-truth target in the policy input. The navigable limit is a map boundary ray-cast at the noisy estimated pose. Simulator ground truth is used for collision and clearance rewards in training (draft 4 section 1.3). |
| S8     | **Training targets hold constant velocity and never give way** (D1). Reactive (compliant) and non-compliant targets appear only in evaluation (the robustness set, R3). |
| S9     | **Geometry:** the 10 x 25 m basin of Paper 2's field site (straight and slanted legs, the default) and parallel-walled channels. Headline channels 7.5-10 m; the width sweep (R4) goes to 3.5 m. Every layout admits a static route (A* filter, F74) -- not a proof that every dynamic encounter is avoidable (draft 4 section 1.3). |
| S10    | **8(e) in two layers:** the learned policy slackens speed; an engineered runtime safety layer, off during training, takes all way off when a collision is imminent. Compliance is reported with it **off** (C-7); its interventions are not attributed to the policy. |
| S11    | **Formulation plus comparison:** one frozen formulation (observation, reward, curriculum, dynamics), four learners (on-policy, recurrent, off-policy, distributional), 3 seeds, 2 M steps, checkpoint selected on the development set, the frozen suite touched once per policy. |
| S12    | \[TBC\] The Paper 2 deployment-layout set and the field fine-tune (F104-F107): report as a field-readiness study, as the bridge to the field trials, or leave to the field paper. |
| S13    | \[TBC, new 2026-09-28\] **Physical transfer.** Draft 4 calls it "planned" and its roadmap has no field section. Either (a) keep basin trials in this paper -- add a field section to the roadmap and keep RQ4, C-6 and old C7; or (b) move them to the field paper -- drop RQ4 and C-6 here and state the transfer as future work. Old section 7 is parked, unnumbered, after section 4 until then. |

Abstract

*Draft skeleton — write last, but fix the shape now. Sentences marked \[RESULT\] wait for the three-seed campaign. \[Align with draft 4 before use: Rule 9 as an adopted convention, width as a geometric variable, the four contributions, and S13.\]*

Autonomous surface vessels in restricted waterways must follow a planned route, keep clear of static hazards and resolve encounters with other vessels in keeping with the collision regulations, under geometric limits that open-water methods do not face: in narrow water, Rule 9 decides which manoeuvre is admissible before Rules 13 to 17 decide which is required. Learning-based COLREGs methods almost universally take the other vessel's state from AIS or a simulation oracle and are developed around a single learning algorithm. This paper presents a deep reinforcement learning formulation for path following, static obstacle avoidance and COLREGs-compliant manoeuvring against a moving target in confined water, driven entirely by onboard sensing: the target is detected and tracked from a two-dimensional LiDAR, and the observation combines pooled range sensing, a map-derived boundary, a tracked-target branch and an encounter-context branch that makes the reward's judgement of the encounter observable. A COLREGs reward group reads the encounter as the vessel perceives it, with Rule 9 governing precedence where the open-water manoeuvre does not fit, and an engineered runtime layer separates learned speed reduction from an emergency stop. The formulation is frozen and four learners — PPO, recurrent PPO, SAC and TQC — are trained on it under an identical protocol and evaluated once on a held-out suite of \[800\] episodes per seed and a robustness set with reactive and non-compliant targets. \[RESULT\] The learners reach \[X-Y\]% success, against \[Z\]% for COLREGs-aware velocity-obstacle and dynamic-window comparators; crossing encounters remain the dominant failure for every learner. A channel-width sweep locates where each classical comparator becomes inadmissible, a perception-degradation study characterises how compliance fails as tracking degrades, and \[the transfer to a 1.73 m model vessel is evaluated in basin trials — TBC, S13\].

**Keywords:** autonomous surface vessel; COLREGs; Rule 9; narrow channel; deep reinforcement learning; learner comparison; LiDAR perception; collision avoidance

1\. Introduction

*Text: `planning/Paper3_Introduction_draft4.docx` (draft 4, 28 Sep 2026). The notes below summarise each subsection and keep the planning material behind it; edit the docx, not these notes, for wording. `planning/INTRODUCTION_DRAFT.md` (first draft, 27 Sep) is superseded.*

1.1 Problem statement and importance

Four paragraphs: (1) ASVs approaching routine operation, much of it in restricted waterways, where route following, clearance from boundaries and hazards, and traffic avoidance are coupled and confinement limits the response (Thyri & Breivik 2022; Paulig & Okhrin 2024; Waltz et al. 2025; He et al. 2025); (2) COLREGs behaviour -- Rules 8 and 9 recognise course and speed changes and conditional non-impeding duties, so the rules cannot be reduced to "turn to one side", and algorithmic encodings require interpretive choices that are a recurring error source (García Maza & Poo Argüelles 2022; Hagen et al. 2023; Sánchez-González et al. 2026); (3) perception -- target state must be estimated, LiDAR reliability depends on detection, discrimination, association and tracking (Helgesen et al. 2022), mis-tracking affects the encounter class as well as the predicted clearance, and navigable limits come from charts; (4) the combined problem studied -- path following, static-obstacle avoidance and rule-aware manoeuvring in a two-vessel encounter in laterally bounded water, motivated by a survey leg within an allocated corridor, at model scale in Paper 2's basin (Tran et al. 2026).

*Planning note kept from revision 3:* keep the survey framing to the one paragraph -- expanding it invites coverage-planning questions that are out of scope. \[Optional figure: laterally bounded corridor with reference path, one dynamic target, and static obstacles.\]

**A note on what the sensor observes** (supports section 1.1 paragraph 3 and section 2.1.1). The LiDAR does not register the basin edge; it registers the facility walls one to two metres beyond it. In a restricted waterway the navigable limit is generally not a physical structure either -- a charted depth contour, a buoyed line or a regulatory boundary -- so supplying the boundary from the chart and reserving range sensing for physical obstacles is the correct division for the application, and the basin reproduces it exactly.

1.2 Literature review

*Organised by the elements of the combined problem (draft 4 section 1.2 opening paragraph); broader surveys: Huang et al. 2020; Burmeister & Constapel 2021; Öztürk et al. 2022; Qiao et al. 2023; Wu et al. 2025b. Revision 3's section 2 threads are folded in below; references it named that draft 4 does not cite are flagged "(rev 3 only)".*

1.2.1 Classical and explicitly constrained collision avoidance

VO (Kuwata et al. 2014 with COLREGs constraints; Huang et al. 2019 generalised VO), DWA (Fox et al. 1997; COLREGs variants Guan & Wang 2023, Xu et al. 2024), artificial potential fields, and MPC (Johansen et al. 2016; Eriksen et al. 2019; Thyri & Breivik 2022; Han et al. 2024; He et al. 2025). Three limitations motivate learning: tuned weights and thresholds per vessel and geometry; guarantees resting on modelling and prediction assumptions (constant target velocity, simplified dynamics, explicit obstacle geometry -- Gonzalez-Garcia et al. 2022; local minima); and error-prone COLREGs encodings (Sánchez-González et al. 2026: 855 papers, 464 attempting the rules, none fully compliant, 29 recurring errors). Formal specification and enforcement (Krasowski & Althoff 2024; Vain et al. 2025) versus reward-encouraged compliance; Burmeister & Constapel 2021 on incomplete coverage and narrow channels. **Role in this paper:** the classical methods are the natural reference and supply the nonlearned comparators (section 3.3); the COLREGs-VO rule applied from the other vessel's side is also the reactive target model of the robustness set.

1.2.2 Deep reinforcement learning for path following and COLREGs-aware navigation

Path following (Woo et al. 2019; Zhao et al. 2021; Deraj et al. 2023); combined following and avoidance (Meyer et al. 2020a, the path-relative observation with rangefinder sensing this work descends from; Meyer et al. 2020b); multiship and benchmark studies (Zhao & Roh 2019; Sawada et al. 2021, Imazu; Chun et al. 2024); risk-based COLREGs rewards (Heiberg et al. 2022). Since 2023: spatial-temporal recurrent RL and partial observability (Waltz & Okhrin 2023; Zheng et al. 2023), reward balancing and hybrids (Lou et al. 2024; Yang et al. 2024; Sonntag et al. 2025), safe RL (Wang et al. 2024), collision grids (Teitgen et al. 2023), reward-parameterisation effects (Krautwig et al. 2025). **Position:** supplied context versus learned response -- Hart et al. 2024 (supervised risk estimate fed to the agent) and Zhang et al. 2022 (model-reference correction); this paper makes the split explicit (the encounter classifier supplies the regime, the policy the response), which is what the ablation (sections 3.4.5, 4.6) measures. Paper 2 (Tran et al. 2026) added feasibility-inspired LiDAR sector pooling, a staged curriculum and field validation for the static case.

1.2.3 Restricted and inland waterways

Burmeister & Constapel 2021 (of 48 approaches, two address Rule 9 even in part) \[VERIFY the "four mention" figure used in revision 3; draft 4 states only the two\]; Hansen et al. 2022 (Rule 9 applicability from manoeuvrability); classical confined-water planners (Thyri & Breivik 2022; He et al. 2025); canal field trials (Kim et al. 2024); learned river and restricted-water navigation (Paulig & Okhrin 2024; Hao et al. 2024). **Closest prior work:** Waltz, Paulig & Okhrin 2025 (two-level RL for inland waterways, AIS-derived scenarios, versus APF). This paper differs in model-scale LiDAR-tracked encounters, a single learned rudder-and-propulsion policy, and evaluation organised around manoeuvre admissibility as width changes. The question is how available lateral clearance changes the feasible response within a declared rule interpretation -- course alteration, reduced speed or holding astern -- with width a geometric variable, and slowing conditioned on predicted clearance (stopping in a reciprocal vessel's path may preserve the conflict). (rev 3 only: de Vries et al. 2022, urban canals.)

1.2.4 Onboard sensing, integrated control and physical validation

Sensor-driven avoidance outside RL: Han et al. 2020 (radar/LiDAR/camera fusion, field tests), Eriksen et al. 2019 and Kufoalor et al. 2020 (MPC at sea), Helgesen et al. 2022 (environment-dependent tracking), Villa et al. 2020 (LiDAR path following in harbour), Kim et al. 2022 (2D LiDAR avoidance on an ASV -- the closest platform analogue). Sensor-driven DRL: Lin et al. 2025 (distributional RL, LiDAR segmentation in the higher-fidelity simulator, cross-algorithm comparison), Wu et al. 2025a (sensor-level mapless navigation with action correction). **Open:** a policy whose encounter information comes from a tracker in training as well as evaluation, interacting with width-dependent constraints. Physical transfer: Slawik et al. 2024 (domain randomisation, basin), Wang et al. 2025 (model-scale SAC on rudder and propeller), He et al. 2025 (towing tank), Paper 2. Transfer evidence is reported separately from numerical results (S13).

1.2.5 Evaluation practice and the research gap

Automatic COLREG evaluation encodes the evaluator's interpretation (Hagen et al. 2023); declared behaviour-level metrics rather than legal-compliance claims (Sánchez-González et al. 2026). Algorithm comparisons (Larsen et al. 2021, PPO most robust; Lin et al. 2025; Slawik et al. 2024; Hart et al. 2024) and their statistical pitfalls (Agarwal et al. 2021; Patterson et al. 2024). This paper fixes observation, reward, dynamics and curriculum across learners and separates checkpoint selection from held-out evaluation -- a comparison under a specified protocol, not a claim of equally optimal tuning. **Gap (draft 4's closing paragraph):** at the intersection of sensing, constrained manoeuvring and evaluation -- classical methods handle confinement but depend on tuned rules and prediction; learned COLREGs policies mostly use idealised or noise-perturbed target states in open or loosely bounded water; learner comparisons rarely fix the formulation independently of the learner.

*Revision 3's five gaps, now argued across section 1.2 (kept as a checklist):* assumed target state (sections 1.2.2, 1.2.4); open-water bias and Rule 9 (section 1.2.3); traffic and clutter treated apart (sections 1.2.1, 1.2.3) \[VERIFY — survey the handful of papers with both\]; single-learner studies (section 1.2.5); limited physical validation (section 1.2.4) \[VERIFY — weakened since 2024; differentiate rather than assert exclusivity\].

1.3 Scope, objective and contributions

**Objective** (draft 4): develop and evaluate a sensor-realistic DRL formulation for path following, static-obstacle avoidance and COLREGs-aware two-vessel encounters in confined water, and compare PPO, recurrent PPO, SAC and TQC under that shared formulation.

**Scope** (draft 4; see S1-S13): one model-scale ASV, one moving target, up to three static obstacles, 10 x 25 m basin and parallel-walled channels; continuous rudder and propulsion at 2 Hz; target from onboard 2D LiDAR (no AIS, no ground-truth target in the policy input); map-derived boundaries at the estimated pose; detection, tracking and encounter classification as explicit stages. Selected requirements of Rules 8, 9, 13, 14, 15/16 and 17(a)(i); the narrow-channel crossing convention (S3); excluded: Rule 17(a)(ii) and 17(b), Rule 18, Rule 19, signalling, simultaneous multiple targets. Constant-velocity, non-cooperating training targets; reactive and non-compliant targets for robustness only. Headline: 800 held-out constant-velocity scenarios per seed (basin, and channels 7.5-10 m); a separate 3.5-10 m width sweep; the stopping safety layer evaluated separately and off for learned-compliance results.

Research questions -- draft 4 states four central questions (RQ1, RQ6, RQ3, RQ5):

- **RQ1 —** Can the policy complete the navigation task -- path following, static obstacle avoidance and a two-vessel encounter in confined water, from onboard range sensing -- while producing encounter-consistent behaviour?

- **RQ6 (new, draft 4) —** How does the policy's response change as maneuvering room decreases: which responses remain admissible, and when does it switch from course alteration to speed reduction or holding astern?

- **RQ3 —** How does perception degradation affect safety and compliance, and at what point does the policy fail unsafely rather than conservatively?

- **RQ5 —** Under one frozen formulation, which differences between on-policy, recurrent, off-policy and distributional learners -- in success, compliance by encounter and side, and robustness to target behaviour -- remain meaningful across training seeds?

- **RQ2 —** Does making the encounter explicit in the observation improve compliance over reward shaping alone? \[Not stated in draft 4; the ablation (C-4, three rungs) still supports its section 1.2.2 argument on supplied context versus learned response. TBC — ablation budget.\]

- **RQ4 —** Does domain randomisation over identified model uncertainty close more of the sim-to-field gap than improved nominal identification alone? \[Not in draft 4 — S13.\]

Contributions (draft 4's four; the table maps revision 3's seven onto them)

| **\#** | **Contribution** | **Revision 3** |
|--------|------------------|----------------|
| C1     | **A sensor- and encounter-aware learning formulation.** One observation and reward design combining tracked LiDAR target information, map-derived boundaries, own-ship and path states, and explicit encounter context: the classifier supplies the encounter regime, the policy learns the rudder-and-propulsion response. Encounter-dependent reward terms support route completion, clearance and selected COLREGs behaviours, without treating reward shaping as a formal safety guarantee. Frozen, digest-checked and released. | C1 |
| C2     | **A geometric framework for constrained encounter responses.** Relates available maneuvering space to an encounter-consistent course alteration, holding astern, or speed reduction where predicted clearance supports it; the channel-width sweep tests configuration-dependent admissibility thresholds and changes in manoeuvre choice, not legal precedence from width alone. | C2 + C4 |
| C3     | **A controlled four-learner comparison with classical references.** PPO, recurrent PPO, SAC and TQC on the frozen formulation, 2 M steps, 3 seeds, checkpoint selection confined to a separate development set; COLREGs-VO, its narrow-channel encounter-specific adaptation, and LOS-PID with a DWA layer as nonlearned comparators on the same perception pipeline; differences reported with seed variability, no preferred learner assumed. | C3 (+ C6's protocol) |
| C4     | **An evaluation design separating safety, rule-related behaviour and robustness.** Held-out scenarios, paired target-behaviour tests, the width sweep and the perception-degradation study (pose error, dropout, occlusion, velocity-estimation error) distinguish task completion from collision type, encounter violations and intervention dependence, and give the basis for the planned physical-transfer assessment. \[TBC S12: plus the field-stageable Paper 2 deployment-layout set.\] | C5 + C6 |
| —      | Sim-to-field transfer (system identification, randomisation over identified uncertainty, basin trials) | C7 -> S13 |

Novelty in one paragraph

*For the cover letter. Draft 4 claims the intersection (section 1.2.5); keep this consistent with it.*

To the authors' knowledge this is the first learned COLREGs collision-avoidance study that (i) perceives the other vessel only through onboard 2D LiDAR, in training as well as evaluation, while (ii) evaluating how the admissible encounter response changes with channel width under a declared narrow-channel convention, (iii) in water shared with static obstacles, and (iv) separates the contribution of the formulation from that of the learner by training four learner families on one frozen formulation and evaluating them once on a pre-registered held-out suite. \[VERIFY each clause against section 1.2 before submission; claim the intersection, not any single thread.\]

What the paper does not claim

- Multi-target encounters, Rule 19 restricted visibility, sound signals, Rule 18 responsibilities, the open-water Rule 15/17 role table, or legal compliance / certification: violation terms are declared behavioural proxies (`CLAIM_LEDGER.md` section 2).
- That operating bounds establish the legal applicability of Rule 9, or that a width is a universal legal criterion for a narrow channel (draft 4).
- Rule 17(b) last-moment action: out of scope by construction (S5).
- That every dynamic encounter is avoidable: the A* route check is static (S9); the Paper 2 set's space-time check (F107) is not applied to the headline suite.
- Co-operation from the other vessel: training targets never give way; reactive and non-compliant targets are evaluated, not trained on (S8).
- A best learner: with three seeds, headline differences of about 0.05 are at the edge of separability and crossing-level differences are not separable (F90); learners are compared on the headline, crossings reported per seed.
- Performance in untrained geometry: the headline holds the class x geometry combinations training draws (suite 3.4); channels below 7.5 m are the width sweep's, not the headline's. The Paper 2 deployment layouts, where traffic meets fixed hazards at close quarters, expose a combination the baseline formulation does not train (F105) — \[TBC S12\] how and whether it is reported.

Paper structure (draft 4's last paragraph): Section 2 the formulation (problem and workspace; COLREGs scope and the geometric framework for constrained encounter responses; the frozen perception, observation, reward and curriculum); Section 3 the learners, training protocol, classical comparators and evaluation design; Section 4 held-out results, target-behaviour tests, width sweep and perception-degradation study; Section 5 discussion and limitations; Section 6 conclusion. \[S13: add the field section here if (a).\]

2\. Formulation

*Writing plan with verified numbers, figures, equations and drafting inputs: `planning/FORMULATION_PLAN.md` (2026-09-28). Where the notes below disagree with it (its section 6), the plan is right.*

2.1 Problem and workspace

2.1.1 Vessel and workspace

Model-scale Bluefin: 64.55 kg, LOA 1.73 m, LBP 1.57 m, breadth 0.50 m, draft 0.19 m, Iz 10.45 kg·m². Three-degree-of-freedom Fossen model with explicit actuator dynamics (section 2.3.5), integrated at 0.1 s. Decision period 0.5 s (2 Hz), episode cap 180 steps (90 s); cruise 0.558 m/s at 6 RPM, RPM limited to 0-12 (no astern).

The default workspace is the 10 x 25 m basin of Paper 2's field site, with straight and slanted legs from y = 2 m to y = 22 m (basin mode, F74); parallel-walled channels of variable width carry the classes whose response the width decides (head-on, crossing, overtaking). Reference paths are deliberately not centred, so that the boundary observation branch carries information not already present in the cross-track error. Simulation matches the physical basin, with a maximum corridor width of 10 m (20 ship breadths), so that every simulated width is physically reproducible. The unconfined reference case is supplied instead by the open-water variant of the external benchmark (section 3.4.2), rather than by a wider simulated channel.

The width sweep in section 4.4 spans 10 m down to 3.5 m. The frozen predicted thresholds (`PREDICTED_THRESHOLDS_M`) are 3.8 m for a compliant port-to-port head-on (6.3 m when the target holds the centreline) and 4.9 m for overtaking, so the sweep brackets them. \[Revision 3 derived 3.66 m from a 1.18 m abeam domain; reconcile the overtaking and crossing thresholds with claim C-3 — FORMULATION_PLAN section 6.\] \[Re-verify once the ship domain is finalised from turning-circle data — the threshold moves with the domain.\]

The boundary is supplied from the map, not sensed (see the note on what the sensor observes, section 1.1).

**Why two-vessel encounters** (moved here from the discussion, as revision 3 advised). Rules 13 through 16 are formulated pairwise: each defines the obligations of one vessel with respect to a single other vessel. Multi-ship handling is an extension not specified by the regulations themselves. In restricted waters this pairwise framing is also the physically realistic one, since channel geometry sufficiently confined for Rule 9 to be operative precludes multiple simultaneous close-quarters conflicts; encounters in such waters are typically sequential rather than concurrent. The two-vessel encounter is therefore adopted as the unit of analysis, which additionally allows every reported behaviour to be reproduced in physical trials rather than validated in simulation alone. *A scope decision defended by the structure of the regulations and the geometry of the domain, not a constraint conceded after the fact.*

2.1.2 Scaling

State the model-to-full-scale relationship explicitly so that spawn TCPA, CPA thresholds and ship-domain dimensions can be read at full scale. \[TBC — Froude scaling statement.\] A reviewer will ask whether a 15 s TCPA on a 1.73 m model corresponds to anything meaningful at full scale; the answer belongs in the paper.

2.1.3 MDP formulation

State, action a = \[rudder, throttle\] ∈ \[−1,1\]², transition, reward, discount. Propulsion authority is widened relative to prior work because Rule 8 subsection (e) makes slackening speed a lawful avoidance action, and in a confined channel it is frequently the only admissible one. A policy that cannot slow down cannot comply.

2.2 COLREGs scope and constrained encounter responses

2.2.1 COLREGs scope

| **Rule**   | **Content**                                            | **Treatment**                                                                                                                                                                    |
|------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 8          | Action to avoid collision                              | Implemented — 8(a) ample time, 8(b) readily apparent, 8(e) slacken speed. Drives the timing and magnitude metrics.                                                               |
| 9          | Narrow channels                                        | Implemented as an adopted operating convention — 9(a) keep starboard, 9(b) do not impede, 9(e) overtaking. Shapes the admissible response under Rules 13–16 (section 2.2.3).          |
| 13         | Overtaking                                             | Implemented — two classes: own ship overtaking, own ship overtaken.                                                                                                              |
| 14         | Head-on                                                | Implemented — alter to starboard, subject to available channel width.                                                                                                            |
| 15         | Crossing                                               | Implemented — own ship gives way in all crossing encounters under the narrow-channel convention (S3).                                                                          |
| 16         | Give-way action                                        | Implemented — early and substantial, per Rule 8.                                                                                                                                 |
| 17         | Stand-on action                                        | Partially — 17(a)(i) passive course-keeping is retained for the being-overtaken class. Active release under 17(a)(ii) is out of scope and identified as future work.             |
| 2, 5, 6, 7 | Responsibility, lookout, safe speed, risk of collision | Acknowledged; operationalised implicitly — the collision risk index for Rule 7, the perception pipeline for Rule 5.                                                              |
| 18         | Responsibilities between vessels                       | Out of scope — own ship and target are similarly sized, so the asymmetry the rule requires does not exist. The always-give-way simplification is instead motivated by Rule 9(b). |
| 19–31      | Restricted visibility, lights, shapes, sound signals   | Out of scope — no corresponding sensing or actuation on the platform.                                                                                                            |

*State this table in the paper. Reviewers who know COLREGs read the omissions as closely as the inclusions, and naming the Rule 18 exclusion pre-empts the obvious question.*

2.2.2 Encounter classes

| **Class**       | **Governing rule** | **Own-ship obligation**                                      |
|-----------------|--------------------|--------------------------------------------------------------|
| None            | —                  | Follow path                                                  |
| Head-on         | 14                 | Alter to starboard, subject to channel width                 |
| Crossing        | 15, 16, 9(b)       | Give way regardless of which side the target approaches from |
| Overtaking      | 13, 16, 9(e)       | Keep clear of the vessel being overtaken                     |
| Being overtaken | 13, 17(a)(i)       | Hold course and speed                                        |

2.2.3 Geometric framework for constrained encounter responses (revision 3: "Rule precedence")

The organising principle: under the adopted convention, Rule 9 constrains the available space, and Rule 8 subsection (e) supplies the action when that space is unavailable. Width thresholds are an output of the sweep in section 4.4, not an input to it. *Draft 4 wording to keep: width is a geometric experimental variable, not a universal legal criterion; slowing is conditioned on predicted clearance, not a general preference for stopping when space is limited.*

| **Encounter**   | **Wide channel**                             | **Narrow channel**                                                   | **Governing**        | **Fallback**                       |
|-----------------|----------------------------------------------|----------------------------------------------------------------------|----------------------|------------------------------------|
| Head-on         | Alter to starboard                           | Hold starboard side; 9(a) compliance satisfies 14 without alteration | 14 + 9(a)            | Slacken speed, 8(e)                |
| Crossing        | Give way — alter to starboard or pass astern | Give way if room exists                                              | 15, 16 + 9(b)        | Slacken speed or stop, 8(e)        |
| Overtaking      | Pass either side                             | Pass to port of target                                               | 13 + 9(a), 9(e)\*    | Hold astern at reduced speed, 8(e) |
| Being overtaken | Hold course and speed                        | Hold course and speed, keep starboard                                | 13 + 17(a)(i) + 9(a) | —                                  |

*\* Geometric constraint only. Rule 9(e) requires sound signals and the overtaken vessel's agreement, which the platform cannot provide. State as an explicit scope limitation.*

Head-on rationale: Rule 9(a) already requires both vessels to keep to the starboard side of the fairway, so if both comply a port-to-port pass occurs without either altering course, and Rule 14 is satisfied by channel-keeping rather than by an evasive manoeuvre. An alteration is required only where the target is not where 9(a) says it should be. Overtaking side follows from the same rule: if the overtaken vessel keeps starboard, the room lies on its port side.

The threshold at which a channel becomes "narrow" for a given encounter is itself a result, produced by the width sweep in section 4.4. Define it geometrically — in ship breadths and in terms of the lateral excursion the compliant manoeuvre requires — not by reference to where any method fails.

2.3 Perception, observation, reward, vessel model and curriculum

2.3.1 Perception pipeline

Raw LiDAR: 360°, 720 beams, 0.5° angular resolution, 10 Hz. Returns falling outside the known channel polygon are gated geometrically, because the sensor is mounted above the basin wall and would otherwise register objects beyond it as obstacles or, worse, as phantom targets with plausible velocities. Remaining returns are clustered, ego-motion compensated, associated to tracks, and velocity-estimated by a constant-velocity Kalman filter. Static clusters feed the pooled sector channel; the dynamic track feeds the target branch.

*\[Figure: perception pipeline block diagram.\]*

2.3.2 Observation space

| **Branch** | **Contents**                                                                          | **Dim** |
|------------|---------------------------------------------------------------------------------------|---------|
| lidar      | Pooled sector closeness, obstacles only, forward-biased to ±135°, non-uniform sectors | 27      |
| boundary   | Virtual raycast against the channel polygon, 7 rays, pose noise injected              | 7       |
| ego        | Surge u, sway v, yaw rate r                                                           | 3       |
| path       | Cross-track error, course error, look-ahead course error                              | 3       |
| target     | Tracked target features plus presence bit                                             | 16      |
|            | Total                                                                                 | ≈56     |

\[Table out of date: baseline-v2 has **six** branches, 70 values, including the encounter-context branch (C1; `METHODS_BRIEF.md`). Restate from the frozen config.\]

Target features: distance to ship domain; sine and cosine of relative bearing; sine and cosine of heading-intersection angle; target speed; relative speed; DCPA; TCPA; collision risk index; encounter class as a five-way one-hot; presence bit.

**Design rationale to state explicitly.** Pooled range sensing is velocity-blind: a static wall and a closing vessel at the same range produce identical sector closeness. Target kinematics must therefore enter through either recurrence or explicit tracking. Explicit tracking is chosen because it additionally gives information parity with the velocity-obstacle comparators, which require the same quantities.

N_max is a configuration parameter. The target branch is built as an indexed slot so that extension to multiple targets requires retraining rather than redesign.

2.3.3 Encounter classification and collision risk

One module, two consumers: the same function feeds the observation feature and the reward gate, with hysteresis applied inside it. If the two diverged even at sector boundaries, the agent would be penalised for a role it was never shown.

Classification thresholds adapted from Waltz & Okhrin (2023), with the head-on band widened from ±5° \[TBC — value and justification\] and a fifth class added for being overtaken. Collision risk computed as the maximum of a CPA-based and a Euclidean-distance-based term, with distance measured to the ship domain rather than the hull. The Euclidean term is not optional here: two vessels on near-parallel courses in a channel have a CPA far in the past or future, so a CPA-only risk reads as low until either vessel turns slightly, at which point the situation becomes urgent instantly. Near-parallel geometry is the normal case in a corridor.

**All constants must be re-derived in ship lengths.** The source values are tuned for a 320 m KVLCC2 with decay scaled to two nautical miles, and the asymmetric three-ship-length fore-aft domain does not fit a channel of the width considered here.

A compressed asymmetric domain is adopted: 3.14 m ahead, 1.57 m astern and 1.25 m abeam (frozen in baseline-v2), so the required passing distance is d_req = 2.5 m, a lateral footprint of about 25 per cent of a 10 m corridor. \[Revision 3 said 0.75 Lpp = 1.18 m abeam; the frozen value is 1.25 m.\]

**The principle matters more than the values.** These are not defended as a scaled copy of another vessel's domain. The final values are derived from measured manoeuvring performance: advance and tactical diameter from the turning-circle tests and stopping distance from the stop test (physical transfer, P.2). The domain is then sized to this vessel's demonstrated ability to avoid, which is the argument made for confined-water domains in the classical literature. The table above is a provisional input; the final values are an output of the identification campaign.

2.3.4 Reward function

Six task terms carried from the prior redesign — exponential clearance-based avoidance, unified path following, border penalty, progress, action smoothness, existence cost — plus class-conditional COLREGs terms:

| **Term**                    | **Condition**                                                                      |
|-----------------------------|------------------------------------------------------------------------------------|
| Wrong-side passing          | Any class with a defined passing side                                              |
| Port turn in head-on        | Head-on, TCPA \> 0                                                                 |
| Bow crossing                | Crossing and overtaking classes                                                    |
| Course-keeping hold         | Being overtaken                                                                    |
| Late or insufficient action | Give-way classes (Rule 8)                                                          |
| Compliant speed reduction   | Give-way classes where course alteration is geometrically inadmissible (Rule 8(e)) |

**A tension requiring explicit treatment.** The progress term and the existence cost both penalise slowness, while Rule 8 subsection (e) compliance requires it. A class-conditional carve-out attenuates the progress penalty during a compliant slow-down, gated on an active encounter class and a collision risk index above threshold. Without the gate the obvious degenerate policy is to proceed slowly at all times, trivially avoiding conflict and never completing; the speed profile in open stretches is therefore an acceptance check on the trained policy.

**Implementation trap.** The head-on term penalises altering to port, but overtaking in a confined channel requires altering to port. Class-conditional gating handles this in principle, but it is readily miscoded as a global penalty on port alterations and should be asserted in a unit test.

**Turn direction is judged from yaw rate, not rudder angle.** Vessel dynamics delay the sign change in yaw rate by several timesteps after a rudder reversal, so rudder angle is a poor proxy for whether the vessel is actually turning — more so for an underactuated model vessel than for a full-scale ship.

**Magnitude hierarchy, enforced by design:** collision ≫ border ≫ COLREGs violation ≫ path following ≫ smoothness. A COLREGs-compliant collision is worse than a non-compliant near-miss.

**Scale audit.** Per-term episode-integrated contributions reported in Table R7 (section 4.7). This is mandatory: the obstacle-avoidance term in the prior paper was approximately 49 times weaker than the path-following term at contact distance, which the weighting notation concealed entirely until the terms were integrated and compared empirically.

2.3.5 Vessel model and domain randomisation

Three-DOF Fossen model with linear and quadratic damping, plus an explicit actuator model: servo rate limit, first-order lag, transport delay, and thrust map. The observed field behaviour — wider turns, larger oscillation, roughly double the RMS cross-track error — indicates actuator lag together with underestimated yaw damping and rudder effectiveness, rather than missing degrees of freedom.

Domain randomisation over identified parameters plus or minus their confidence intervals, together with the perception noise sources characterised in section 2.3.1. A better nominal model reduces bias but does not create robustness; the defensible claim is identification *and* randomisation within identification uncertainty.

*The identification procedure (manoeuvres, ground truth, synchronisation, validation) is parked with the physical-transfer block (P.2) pending S13; under S13(b) only the fitted model and its provenance stay here.*

2.3.6 Curriculum

Constant-velocity targets during training; reactive and non-compliant behaviours are reserved for evaluation, because training against a reactive opponent makes the environment non-stationary and destroys attribution.

| **Stage** | **Content**                                               |
|-----------|-----------------------------------------------------------|
| 1         | Static obstacles only, straight constant-width corridor   |
| 2         | Static obstacles, variable width and bends                |
| 3         | Single dynamic target, generous spawn TCPA, wide corridor |
| 4         | Single dynamic target, reduced TCPA, narrowed corridor    |
| 5         | Full difficulty range with static clutter                 |

\[Restate from baseline-v2's frozen schedule (stage fractions, basin mode) before use.\]

3\. Learners, training and evaluation design

3.1 Learners and policy architecture

One multi-input policy over the **six** observation branches (70 values), identical for every learner: PPO, RecurrentPPO, SAC and TQC, which differ in the algorithm alone. Trained from scratch — no warm start from the prior policy, whose observation and reward semantics have both changed on every channel. The prior policy is retained as a frozen zero-shot comparator, which is stronger for being genuinely independent rather than an ancestor of the new agent. \[Draft 4 does not name the prior policy as a comparator — keep or drop.\]

\[Largely settled: RecurrentPPO is the recurrent learner; the explicit tracker carries some memory already; occlusion of the target behind a static obstacle is the case that would justify more.\]

3.2 Training protocol

2 M environment steps per run, **three** seeds per learner (2026-09-24; was five), checkpoint selected on the development set (goal − 2 × collision), frozen suite touched once per policy. Runs trained on demand, one learner × seed at a time. Hyperparameters in Appendix B.

3.3 Classical comparators

| **Family** | **Method**                                                                                         |
|------------|----------------------------------------------------------------------------------------------------|
| Classical  | LOS-PID with dynamic window approach (Fox et al. 1997)                                            |
| Classical  | COLREGs-aware velocity obstacles (Kuwata et al. 2014) — its rule from the target's side is also the reactive target model; side rule open (A35) |
| Classical  | Encounter-specific VO — the same machinery under the narrow-channel convention (draft 4: "a narrow-channel encounter-specific adaptation"; revision 3 credited Thyri & Breivik 2022) |
| Classical  | \[Optional, not in draft 4\] NMPC with COLREGs constraints                                         |
| Learned    | Prior-paper SAC, unmodified, zero-shot (frozen) \[not in draft 4\]                                 |
| Learned    | PPO, RecurrentPPO, SAC, TQC on the identical formulation (the comparison itself)                   |
| Learned    | COLREGs-ablated policy (avoidance only)                                                            |

All comparators receive the same perception pipeline (draft 4). Classical baselines are run against reactive targets as well, or the comparison is not like-for-like. Tuned on the development set and pinned in `configs/comparators_v1.json`.

3.4 Evaluation design

3.4.1 Scenario generator

Scenarios are parameterised by encounter class rather than by initial condition: sample the class, then a heading-intersection angle from its valid interval, then a target speed, then a spawn TCPA, and solve backwards for the spawn position that produces that geometry. Random spawning frequently produces targets that pose no threat and wastes training samples. The null class — a target on a similar course, posing no conflict — is included deliberately, since it never arises from a purely class-conditioned spawner but does occur in practice.

The same generator produces training scenarios and the evaluation suite, with disjoint seeds.

3.4.2 Evaluation suite

One frozen tier, versioned and hashed before the first training run.

\[Tier A, the 38 named deterministic cases, is **out of this paper** (2026-09-24). It exists in the suite and can be reported later; the argument here rests on Tier B and the width sweep.\]

- **Tier B (the default frozen suite, suite 3.4) —** a held-out draw of the development set's kind of scenario: only the positions differ. 8 balanced cells × 100 = 800 episodes per seed, constant-velocity targets as in training; every encounter class in the basin, and head-on, crossing and overtaking in channels of 7.5–10 m — the class × geometry combinations training draws (being overtaken and null are trained in the basin only). **Robustness set (R3):** the same scenarios and episode seeds with a compliant reactive target (700) and, in head-ons, a non-compliant one that alters to port (200); a non-compliant stand-on target moves exactly like the constant-velocity one, which already is the give-way vessel that does not give way. Narrower water is the R4 width sweep. Static clutter is 0–3 obstacles per episode, dropped where they would decide the encounter.

- **External benchmark —** the "Around the Clock" set of 24 single-ship encounters at equally spaced target headings. \[Out of this paper (2026-09-24) unless built; see section 4.8.\]

- \[TBC S12\] **Paper 2 deployment-layout set** (F104-F107): 630 field-stageable cases on the three published layouts, all space-time solvable.

3.4.3 Difficulty definition

Geometric only: channel width in ship breadths, spawn TCPA, static clutter count. Never defined by baseline performance. One stratum is deliberately narrow enough that no method passes cleanly — a suite the proposed method succeeds on everywhere reads as constructed regardless of how it was built.

3.4.4 Metrics

**Task.** Success rate; collision rate reported separately for static obstacle, boundary and target vessel; RMS and maximum cross-track error; path length ratio; action smoothness.

**COLREGs.** Violation rate per encounter class; minimum CPA distribution reported as a CDF rather than a mean, since the tail is the safety claim; ship-domain intrusion rate and depth; time to first evasive action; magnitude of first evasive action; course-keeping stability while being overtaken; side-of-passing correctness; speed-reduction (manoeuvre-mode) share.

**Perception.** Track acquisition range; classification latency and stability; velocity estimate error; occlusion duration.

**Intervention dependence** (draft 4 C4). Safety layer intervention rate, reported apart from learned compliance.

\[TBC — presentation format for multi-axis results. A weighted scalar will be contested; prefer a per-axis table or a Pareto view.\]

3.4.5 Ablation matrix

|                          | **Encounter feature OFF** | **Encounter feature ON** |
|--------------------------|---------------------------|--------------------------|
| COLREGs reward terms OFF | Avoidance only            | Told, not rewarded       |
| COLREGs reward terms ON  | Learned from kinematics   | Full method              |

This answers the question a sceptical reviewer will actually ask: is compliance learned, or handed to the agent? (draft 4 section 1.2.2: supplied context versus learned response.) Supplementary leave-one-out ablations on individual COLREGs terms and on the boundary observation branch. \[RQ2 — not stated in draft 4; budget TBC.\]

4\. Results

*Tables pre-committed. Values pending. Writing this section before training begins is what prevents the experiment being designed after the fact. Draft 4's roadmap names 4.1, 4.3, 4.4 and 4.5; the rest are kept pending the page budget.*

4.1 Overall performance (held-out results)

Table R1 — Tier B holdout (suite 3.4: 800 constant-velocity episodes per seed, 3 seeds). \[RESULT\]

| **Method**            | **Success** | **Static coll.** | **Boundary coll.** | **Target coll.** | **RMS CTE (m)** | **Path ratio** |
|-----------------------|-------------|------------------|--------------------|------------------|-----------------|----------------|
| PPO                   |             |                  |                    |                  |                 |                |
| RecurrentPPO          |             |                  |                    |                  |                 |                |
| SAC                   |             |                  |                    |                  |                 |                |
| TQC                   |             |                  |                    |                  |                 |                |
| No COLREGs terms (ablation, budget TBC) |             |                  |                    |                  |                 |                |
| Prior SAC (frozen)    |             |                  |                    |                  |                 |                |
| Encounter-specific VO |             |                  |                    |                  |                 |                |
| COLREGs-VO            |             |                  |                    |                  |                 |                |
| LOS-PID + DWA         |             |                  |                    |                  |                 |                |

4.2 Compliance by encounter class

Table R2 — violation rate per class. \[RESULT\]

| **Method** | **Head-on** | **Crossing** | **Overtaking** | **Being overtaken** |
|------------|-------------|--------------|----------------|---------------------|
|            |             |              |                |                     |
|            |             |              |                |                     |

4.3 Performance by target behaviour

Table R3 — the robustness set: the headline's scenarios with compliant reactive and (head-on) non-compliant targets, paired with their constant-velocity twins. \[RESULT\]

4.4 Channel-width sweep

Table R4 and accompanying figure. \[RESULT\] Sweep corridor width from open-water-equivalent down to the point at which the Rule 14 starboard alteration no longer fits within the channel. Report, per width: success rate, compliance rate, minimum CPA, manoeuvre-mode share, and the governing response from the section 2.2.3 table. Identify the width at which each classical method becomes inadmissible.

| **Corridor width** | **Ship breadths** | **Compliant head-on admissible?**  |
|--------------------|-------------------|------------------------------------|
| 10 m               | 20 B              | Yes, comfortably                   |
| 8 m                | 16 B              | Yes                                |
| 6 m                | 12 B              | Yes                                |
| 5 m                | 10 B              | Marginal                           |
| 4 m                | 8 B               | Tight                              |
| 3.5 m              | 7 B               | No — below the geometric threshold |

*This is the primary evidence for contribution C2 (and RQ6), and it replaces the multi-vessel results in providing depth.*

4.5 Perception degradation study

Table R5 and accompanying figure. \[RESULT\] Degrade the tracker along four axes independently and jointly: pose drift magnitude, detection dropout rate, occlusion duration, velocity-estimate noise. Report compliance and safety as a function of each. Identify the point at which failures become unsafe rather than conservative.

*Primary evidence for contributions C1 and C4 (RQ3). Requires no additional basin time.*

4.6 Ablation

Table R6 — the 2 × 2 plus leave-one-out. \[RESULT\]

4.7 Reward scale audit

Table R7 — per-term episode-integrated contribution under a random policy and under the trained policy, with the empirical ordering checked against the intended hierarchy. \[RESULT\]

4.8 External benchmark

Table R8 — the 24-case Around the Clock set. \[RESULT\]

\[BLOCKED — Around the Clock has no scenario builder yet (C3–C6 in the build order). Until it is built this section has no content: either build it or drop the external-benchmark claim and lean on releasing the generator.\]

4.9 Figures

- Learning curves with seed spread

- Trajectory overlays for selected Tier B cases, one per encounter class

- Minimum-CPA cumulative distribution

- Success, compliance and manoeuvre mode versus channel width (section 4.4)

- Compliance versus perception degradation (section 4.5)

Physical transfer (parked — S13; revision 3 section 7, not in draft 4's roadmap)

*Under S13(a) this becomes a numbered section after Section 4 (and the roadmap in section 1.3 gains it); under S13(b) it moves to the field paper and section 5.5 lists it as future work.*

P.1 Platform and setup

Bluefin model vessel, RPLidar C1, UDP offboard control at 10 Hz. The basin is an indoor pool within a hall, with the facility walls standing one to two metres beyond the pool edge. Returns from beyond the pool boundary are removed by geometric gating against the known pool polygon rather than by a physical barrier: the facility walls carry the fixed features on which localisation depends, and occluding them would remove the only available registration reference. Gating is in any case necessary rather than merely preferable, since operators standing on the deck lie at scan height and would otherwise be tracked as dynamic targets. Localisation is run on the complete scan and the obstacle gate applied only afterwards, so that the walls serve as a registration asset while being excluded from the tracker.

Static obstacles are suspended panels, confirmed stable in the water. Apparent motion of static objects in the scan therefore arises almost entirely from ego-pose error, which affects all objects identically; the static-versus-dynamic threshold is consequently a property of localisation quality rather than of the obstacles, and is set from measured pose noise.

P.2 System identification

Identification manoeuvres: straight-line acceleration and deceleration at several throttle settings; turning circles at multiple rudder angles and speeds; 10/10 and 20/20 zig-zags; rudder step response; stop test. Zig-zag overshoot angles are the single most informative measurement for the observed symptom. Fitting by prediction-error minimisation with one zig-zag and one turning circle held out.

Scan-to-scan odometry is not adequate ground truth for identifying yaw dynamics, since error integrates without bound. Identification therefore relies on scan-to-map registration against surveyed facility geometry, which is absolute rather than incremental and so bounded in error, combined with an inertial measurement unit added to the vessel for this purpose. Raw angular rate and acceleration are logged at 100 Hz or better; fused orientation outputs are avoided because their internal filtering would corrupt the lag and damping parameters under estimation. The two sources are complementary: registration supplies drift-free absolute pose at 10 Hz while the inertial unit supplies derivatives at high rate and, through the accelerometer, the along-track information that two parallel walls cannot constrain.

Time synchronisation between the two streams is the critical detail: a constant offset appears in the fit as actuator lag and would be absorbed into the model as a physical parameter. Both streams are logged on a common clock where possible, and each run begins with a stationary period for bias estimation and a sharp yaw impulse for alignment.

Two long parallel walls lie within sensor range from mid-basin; the end walls do not. Heading and lateral position are consequently well constrained, while along-track position is constrained only where fixed features — recessed doorways, structural columns, wall-mounted hardware — fall within range. Zig-zag overshoot angles are read directly from the heading trace and are therefore the best-constrained measurements available; turning circles are partially constrained; straight-line tests require either timing between surveyed features or positioning so that an end wall remains in range.

The model is fitted to measured yaw rate and to heading, and the two fits cross-checked. Disagreement between them indicates a synchronisation or mounting error rather than a model deficiency, which makes the cross-check a useful diagnostic rather than a redundancy.

Validation without external truth is achieved by static tests at surveyed positions, giving absolute accuracy, and closed-loop runs returning to a common physical point, giving drift. \[Risk to check first: one full-length facility wall is matte black, and near-infrared reflectivity of carbon-based finishes can be very low. Plot return density against bearing in the retained logs before committing to continuous-wall registration; sparse registration against surveyed landmarks mounted on that wall is the fallback.\]

The identification dataset is archived and released. This directly discharges the reviewer concession in the prior paper, where the identification data was not retained in a form supporting independent reporting and the model could only be described as calibrated rather than identified.

P.3 Model validation

Replay of the prior field command sequences through the recalibrated model, overlaid against the recorded trajectories. The logs are already retained, so this requires no basin time and produces a validation figure at zero marginal cost. \[Useful under either S13 option: it supports section 2.3.5 without basin time.\]

P.4 Domain randomisation ablation

Table R9. \[RESULT\] Compare policies trained on the identified nominal model alone against policies trained with randomisation over identified uncertainty, both evaluated in the field. This is the evidence for RQ4. If the identified nominal model alone closes the gap, that is a reportable finding; if it does not, the gap is localisation and disturbance rather than dynamics, which is equally reportable.

P.5 Dynamic encounter trials

One target vessel. \[TBC — target platform, repeatability protocol, localisation of both vessels in a common frame, abort procedure, risk assessment.\] \[N\] repetitions per scenario. The prior work ran each scenario once and had to qualify its claims to feasibility rather than robustness; repeated trials are what change that.

P.6 Results

\[RESULT\]

5\. Discussion

5.1 Answers to the research questions

\[RESULT\] RQ1, RQ6, RQ3, RQ5 (draft 4's central questions); RQ2 if the ablation is run; RQ4 under S13(a).

5.2 Claim ledger

*Every claim in the abstract and conclusion maps to a table or figure. Complete this before writing either. The working ledger is `planning/CLAIM_LEDGER.md`.*

| **\#** | **Claim** | **Evidence** | **Status** |
|--------|-----------|--------------|------------|
| 1      |           |              |            |
| 2      |           |              |            |
| 3      |           |              |            |

5.3 Limitations

- Model scale; full-scale correspondence rests on the Froude scaling statement in section 2.1.2

- Single dynamic target; sequential rather than concurrent multi-vessel encounters

- Active stand-on release under Rule 17(a)(ii) is out of scope

- Calm water with no generated waves or current

- Three seeds is a limited estimate of training variability: a headline difference of ~0.05 is at the edge of separability and the measured 0.20 crossing spread is not separable, so crossing behaviour is reported descriptively per seed

- The encounter classifier supplies the rule regime, so compliance is partly given rather than fully learned — quantified by the ablation in section 4.6, not merely acknowledged

- Simulator ground truth drives the collision and clearance rewards in training (the policy input is perception-only; draft 4 section 1.3)

- Static route checks do not prove every dynamic encounter avoidable (S9)

- \[S13(b)\] No physical encounter trials in this paper; simulation with synthetic LiDAR does not substitute for them (draft 4 section 1.2.4)

5.4 Future work

- Extension to concurrent multi-vessel encounters via the existing N_max parameter

- Active stand-on release under Rule 17(a)(ii) against non-compliant targets

- Disturbance rejection under wave and current loading

- \[S13(b)\] Physical-transfer assessment: system identification, randomisation over identified uncertainty, basin encounter trials

*(Revision 3's section 8.3 "Why two-vessel encounters" moved to section 2.1.1.)*

6\. Conclusion

*Write last. Must contain no claim absent from the ledger in section 5.2.*

Appendices

|     | **Content**                                                                                                                        |
|-----|------------------------------------------------------------------------------------------------------------------------------------|
| A   | Vessel model: full numerical parameters, actuator dynamics, saturation limits, delays, identification procedure and fit statistics |
| B   | Learner hyperparameters (PPO, RecurrentPPO, SAC, TQC) and curriculum schedule                                                      |
| C   | Complete reward specification with coefficients and normalisation ranges                                                           |
| D   | Encounter classification thresholds and hysteresis parameters                                                                      |
| E   | Evaluation suite composition and hash                                                                                              |

**Data and code availability.** Scenario generator source and seed; frozen evaluation suite; system identification dataset; trained policy checkpoints. \[Repository DOI TBC\]

Open items

Blocking

| **Item**     | **Question**                                                                                                                                                 | **Blocks**                              |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| Precedence   | Constrained-response (Rule 9 convention) table — the single blocking deliverable                                                                             | Sections 2.2.3, 2.3.4, section 4.4, contribution C2   |
| Physical transfer | S13: keep basin trials in this paper, or move them to the field paper                                                                                   | Section 1.3 roadmap, RQ4, C-6, P.1-P.6         |
| IMU          | Is adding an inertial unit feasible? A low-cost gyroscope logging at 100 Hz would remove the yaw-rate observability constraint entirely. Assume none for now | Section 2.3.5, P.2                             |
| Reflectivity | One full-length facility wall is matte black. Verify LiDAR return rate on that side before committing to continuous-wall registration                        | Section 2.3.5, P.2                             |
| Compute      | Wall-clock estimate per run, then the final comparator list                                                                                                  | Sections 3.3, 4.1                              |

Pending, not blocking

| **Question**                                                                               | **Blocks**       |
|--------------------------------------------------------------------------------------------|------------------|
| Head-on classification band width — the source value of ±5° is narrow relative to practice | Section 2.3.3           |
| Final ship domain values from turning-circle data                                          | Sections 2.3.3, 4.4     |
| Target vessel platform and field repeatability protocol                                    | P.5              |
| Presentation format for multi-axis results                                                 | Sections 3.4.4, 4.1     |
| Prior-paper SAC and NMPC as comparators (named in revision 3, not in draft 4)              | Sections 3.3, 4.1       |

Resolved

| **Item**            | **Resolution**                                                                                                                            |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Paper structure     | Literature review in the Introduction (section 1.2); six sections per Introduction draft 4 (2026-09-28)                                         |
| Scope               | Two-vessel encounters, Rules 8, 9, 13–16 and 17(a)(i), own ship give-way throughout under the narrow-channel convention (Rule 9(b))      |
| External benchmark  | "Around the Clock" adopted in revision 2; out of this paper unless built (2026-09-24)                                                     |
| Sim-to-real         | Revision 2 retained it as RQ4; reopened by draft 4 (S13)                                                                                  |
| Corridor dimensions | Simulation matches the basin; maximum width 10 m, sweeping to 3.5 m                                                                       |
| Boundary handling   | Geometric gating against the pool polygon, not a physical barrier; facility walls retained as the localisation reference                  |
| Ground truth        | No external instrumentation; scan-to-map registration against surveyed facility geometry                                                  |
| Ship domain         | Compressed asymmetric, provisionally 2.0 / 1.0 / 0.75 Lpp ahead, astern and abeam                                                         |
| Recurrence          | RecurrentPPO is one of the four learners                                                                                                  |
