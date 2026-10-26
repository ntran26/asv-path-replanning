**Paper 3 — Draft Skeleton**

*Repositioned to two-vessel encounters with static obstacles · Revision 4 (2026-09-28: the literature review moved into the Introduction and the paper renumbered to the six-section roadmap of Introduction draft 4; revision 3, 2026-09-27, rewrote the introduction, scope and novelty against the frozen formulation baseline-v2)*

> **Status note (2026-09-28, revision 4).** The Introduction is now written:
> **`planning_records/Paper3_Introduction_draft4.docx`** (draft 4, 28 Sep) is the
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
>   controlled four-learner comparison with classical references (restated on 2026-10-10 as a
>   deployed policy chosen and benchmarked fairly, below); an evaluation
>   design separating safety, rule-related behaviour and robustness (absorbs the
>   perception study and the protocol). Old C1-C7 are mapped in section 1.3.
> - **Rule 9 is stated as an adopted operating convention**, not a legal finding:
>   operating bounds "are not assumed to establish the legal applicability of
>   Rule 9"; channel width is "a geometric experimental variable, not a universal
>   legal criterion"; compliance is measured by declared behavioural metrics, not
>   legal certification. Section 2.2.3 is renamed accordingly ("constrained encounter
>   responses"; was "Rule precedence"); its tables are unchanged.
> - **Comparators (S18, 2026-10-20):** the encounter-specific VO (after Thyri and Breivik,
>   2022; `src/classical/encounter_vo.py`), which replaces COLREGs-VO (Kuwata et al., 2014), and
>   LOS-PID with a DWA layer, all on the same perception pipeline. The prior-paper SAC and the NMPC option are not named in
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
> Carried from revision 3 (its framing superseded on 2026-10-10, below): **formulation plus a four-learner comparison**
> (PPO, RecurrentPPO, SAC, TQC) under baseline-v2 (digest `3d697858e95e5adf`);
> **2 M steps, 3 seeds**, best-on-development checkpoint (seed 0 of all four
> trained); **suite 3.4** (800 constant-velocity headline episodes per seed plus a
> robustness set); the R4 width sweep (3.5-10 m); the Paper 2 deployment-layout
> set and the field fine-tune (F104-F107), status open (S12). **Open:** S12. Decided: S13
> (2026-10-14) and A35 (2026-10-19: the side rule judged on the encounter, as published;
> applied to the encounter-specific VO under S18; comparators v2, built after TQC seeds 1-2).

> **Framing (decision, 2026-10-10).** The paper develops a formulation and deploys one policy trained
> on it; it is not a learner-comparison study. The other learners make the evaluation fair (Paper 2's
> first draft compared SAC with no other learner). **Proposed method:** the frozen formulation and the
> deployed policy. **Baselines:** the other learners on the identical formulation (PPO, RecurrentPPO,
> and whichever of SAC and TQC is not deployed; three seeds each, checkpoints selected on the
> validation set) and the classical references LOS-PID + DWA and the encounter-specific VO (S18). **Deployment choice:**
> the better of SAC and TQC on test set v4 (the larger and more balanced set), on its three-seed
> mean, with the classical references compared on the same set; the deployed seed is the one with
> the best validation score. The test set therefore also chooses the learner family; the paper states
> this, and the field trials (S13) are the independent evaluation of the deployed policy.
> **Sensing:** the formulation is stated for onboard perception in general (pooled range returns,
> target tracks and a map boundary) and is not tied to LiDAR, following the supervisors' guidance
> for Paper 2: this study implements it with a 2D LiDAR, and another range sensor such as radar
> could supply the same observation through its own detection and tracking. That portability is a
> design property, not a tested result. LiDAR stays out of the title and the contribution names.

**Working title:** A Deep Reinforcement Learning Formulation for COLREGs-Aware Path Following and Collision Avoidance of Autonomous Surface Vessels in Confined Water

**Alternative titles:**
- \[S13(a), with the field trials\] A Deep Reinforcement Learning Formulation for COLREGs-Aware Path Following and Collision Avoidance of Autonomous Surface Vessels in Confined Water: From Simulation to Field Trials
- COLREGs-Aware Path Following and Collision Avoidance in Confined Water from Onboard Perception: Formulation and Deployment of a Deep Reinforcement Learning Policy
- When the Channel Decides: Rule 9 Precedence in Learned COLREGs Collision Avoidance from Onboard Range Sensing \[not usable: there is no width sweep (S14)\]

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
| S2     | COLREGs scope: selected requirements of Rules 8, 9, 13, 14, 15/16 and 17(a)(i) (draft 4 section 1.3), each stated as exactly what the reward encodes (section 2.2.1; decision 2026-10-08). **Rule 9** shapes the admissible response in narrow water through an adopted operating convention, and **Rule 8** governs action quality (8(b) readily apparent action, 8(d) passing at a safe distance, 8(e) slowing; 8(a) as timing pressure only). The behaviour is described as COLREGs-aware, not COLREGs-compliant. |
| S3     | The own ship gives way in every crossing: an explicit **narrow-channel operating convention** motivated by the conditional non-impeding requirement of **Rule 9(b)** (A17), not the open-water Rule 15/17 role table and not a consequence of vessel length. |
| S4     | Six episode types: no target, null (a target that never meets the own ship), head-on, crossing, overtaking, being overtaken. |
| S5     | Rule 17(a)(i) course-keeping for the being-overtaken class. Rule 17(b) (last-moment action) is out of scope **by construction**: every being-overtaken draw passes at or above a contact-free floor (baseline-v2, F96). |
| S6     | The target branch is a shared-weight slot encoder, so a multi-vessel extension costs a retrain, not a redesign.                |
| S7     | **Onboard sensing only.** The observation is built from pooled range returns, target tracks and a map boundary, a sensor-agnostic interface: this study supplies it with a 360° 2D LiDAR (motion classifier, tracker), and another range sensor such as radar could supply it through its own detection and tracking (not tested here). No AIS, no ground-truth target in the policy input. The navigable limit is a map boundary ray-cast at the noisy estimated pose. Simulator ground truth is used for collision and clearance rewards in training (draft 4 section 1.3). |
| S8     | **Training targets hold constant velocity and never give way** (D1). Reactive (compliant) and non-compliant targets appear only in evaluation (the robustness set, R3). |
| S9     | **Geometry:** the 10 x 25 m basin of Paper 2's field site (straight and slanted legs, the default) and parallel-walled channels. Headline channels 7.5-10 m; the width sweep (R4) went to 3.5 m and is dropped (S14). Every layout admits a static route (A* filter, F74) -- not a proof that every dynamic encounter is avoidable (draft 4 section 1.3). |
| S10    | **8(e) in two layers:** the learned policy slackens speed; an engineered runtime safety layer, off in training, replaces commands it predicts unsafe and takes all way off when contact is predicted with no checked plan. **Decided 2026-10-25:** the layer is a predictive filter plus a permanent emergency stop (Algorithm A3); the paper reports this layer only (no other stop window, no development version), and its purpose is fewer own-vessel collisions, not a higher success rate. Learned behavior is reported with it off; Section 4.5 reports it on, on the test set (own-vessel collisions 135 → 70) and the validation set. Its interventions are never attributed to the policy. |
| S11    | **Formulation, deployed policy and baselines (2026-10-10):** one frozen formulation (observation, reward, curriculum, dynamics); four learners trained on it (on-policy, recurrent, off-policy, distributional), 3 seeds each, 3.0 M steps, checkpoints selected on the validation set; the deployed policy is the better of SAC and TQC on test set v4 (three-seed mean; deployed seed by validation score), with the other learners and the classical references compared on the same set. |
| S12    | \[TBC\] The Paper 2 deployment-layout set and the field fine-tune (F104-F107): report as a field-readiness study, as the bridge to the field trials, or leave to the field paper. |
| S13    | **Decided (2026-10-14): (a), field trials in this paper.** After the simulation study the deployed policy is tested on the model-scale vessel in the basin. Roadmap: 1 Introduction, 2 Formulation, 3 Learners, training and evaluation design, 4 Simulation results, 5 Field trials, 6 Discussion, 7 Conclusion; title with ": From Simulation to Field Trials"; a fifth contribution (physical encounter trials). The trial design (target vessel, encounter types, layouts, number of runs) is still open. Sections 1-2 revision instructions: `planning/PAPER/PAPER3_SECTIONS1-2_REVISION_BRIEF.md`. (Was: (a) keep basin trials here, or (b) move them to the field paper.) |
| S14    | **Decided (2026-10-14).** (1) The channel-width sweep (R4, RQ6) and the perception-degradation study (R5, RQ3) are **excluded**: the paper is already long. C2 stays as a formulation component (the admissibility and stopping tests in the observation and reward); its width thresholds (crossing 7.60 m; head-on 3.80 m with the target on its own side, 6.30 m with the target on the centreline; overtaking 4.26 m) are design reasoning, not predictions tested by a sweep. C4 becomes the test-set evaluation design (task completion, collision type, target-ship safety, robustness to target behaviour). (2) Propulsion is stated in RPM. (3) Target speeds are reported by encounter type as the training, validation and test sets realise them (about 0.22-1.11 m/s overall). (4) The field-trial design and the safety-on evaluation are placeholders until finalised. |
| S15    | **Checkpoint-selection sensitivity (decided 2026-10-15, before any final-model test result).** The headline result of every learner and seed stays its best validation-set checkpoint. Each run's final 3.0 M model is also run on test set v4 (tag `<id>_final`) and reported beside it, so the paper states how much checkpoint selection matters. Prompted by RecurrentPPO seed 1, whose best checkpoint (1.6 M, -0.29) is within about one standard error (about 0.09 on the 270-episode validation set) of its recovered late checkpoints; changing one run's checkpoint after seeing its results would be post-hoc selection. |
| S16    | **Seed retention (decided 2026-10-19).** Every trained seed is reported; no seed is re-trained or replaced because of its result. A run is re-trained only for a documented defect: a code or configuration mismatch with baseline-v3, or an interrupted or corrupted run. RecurrentPPO seed 1 has none: its configuration matches seed 2 except the seed, it trained in one uninterrupted 3.0 M run on the frozen baseline-v3 code, and the launcher (`train_asv.py`) was ruled out earlier (identical configuration, exact replays). Its collapse is a property of the recurrent on-policy baseline under this protocol: between 1.0 and 2.4 M the PPO update stopped at the first epoch on the KL limit in 258 of 273 updates (target_kl 0.03, stopping at 1.5 x target), the policy standard deviation stayed at 0.65-0.66, and the validation-set score fell to -0.91 at 2.0 M before recovering after 2.4 M. Seed 2, identical except for the seed, shows none of this: over the same span 25 of 273 updates stopped at the first epoch (the rest after one to nine epochs) and the standard deviation fell from 0.58 to 0.50. Test set v4: seed 1 0.638 at its best checkpoint (final model 0.669), against 0.757 for seed 0 and 0.756 for seed 2. The paper reports all three seeds and states the collapse and its cause; re-running seed 1 would at best reproduce it, and replacing it with another seed would be selection on the outcome. Further seeds, if ever added, are added for every learner. |
| S17    | **Algorithm boxes (decided 2026-10-19).** Main text: Algorithm 1, encounter assessment (classification, engagement and latch, admissibility and stopping tests; end of Section 2.2.4); Algorithm 2, feasibility sector pooling (Paper 2's algorithm with the Paper 3 changes; Section 2.3.1); Algorithm 3, target detection and tracking (Section 2.3.1); Algorithm 4, episode generation, one generator for the training, validation and test sets (Section 2.3.5). Pooling and tracking stay separate boxes. The test-set fairness check is described in prose, not boxed. Appendix: the two classical comparators (the encounter-specific VO, S18, to be drafted from comparators v2; LOS-DWA, drafted 2026-10-19) and the runtime safety layer, the last **[PENDING]** until the layer is finalized. Drafts written from the frozen code: `planning/PAPER/PAPER3_ALGORITHMS.md`. |
| S18    | **Classical COLREGs comparator: the encounter-specific VO (decided 2026-10-20; replaces COLREGs-VO).** The reported velocity-obstacle comparator is the encounter-specific VO (`src/classical/encounter_vo.py`), an interpretation of Thyri and Breivik (2022), a method designed for confined waters: passing-side constraints per encounter class, a ship domain scaled per class whose intrusion is penalised rather than forbidden (so a narrow channel can admit a small intrusion where no clear pass exists), scan points and the map boundary as velocity obstacles, and candidates swept along the hull's turn model. It reads the same encounter context as the learned policy (Algorithm 1: latched class, required turn sense, engagement, in extremis) and follows the same narrow-channel convention (give way to a crossing target from either side and pass astern of it), so the learned-versus-classical comparison measures the method rather than a difference in rule sets. COLREGs-VO (Kuwata et al. 2014, open-water roles) is not reported as a comparator, and its test set v4 run (2026-10-08, success 0.687) is set aside; it remains the rule of the test set's reactive target (`T-RE`), a frozen part of the test set. The per-candidate passing-side exemption found in A35 exists here too (`_wrong_side`): comparators v2 judges the passing side on the encounter, latched while it is engaged. **Built 2026-10-25** (test set v4 success 0.675; Algorithm A1 drafted from it). Built after TQC seeds 1–2 (the tuning would compete with training for CPU): the side-rule change; an encounter-VO grid in `tools/diagnostics/tune_classical.py`; coordinate descent on 60 decoupled validation episodes, winners re-scored on all 120; `configs/comparators_v2.json` (LOS-DWA unchanged); one test set v4 run; Algorithm A1 redrafted from v2. Consequence for the claims: with no open-water baseline, the effect of the narrow-channel convention is not measured by a comparator; it is argued from Rule 9(b) and the reference widths (Section 2.2.4) and shown by the snapshot figure. |
| S19    | **Section 4 scope (decided 2026-10-24).** (1) **The ablation is dropped.** Its planned metric, the violation rate, is not reported (2026-10-08), and any variant needs `src/` changes and a digest of its own: the observation-side switch `encounter_feature_enabled` (`src/reward/config.py`) is read nowhere, and training cannot set `colregs_terms_enabled` from the configuration. Held in reserve, run only if review asks for it: one variant on the deployed learner (SAC expected), with the encounter context hidden from the observation and the COLREGs terms kept, at three seeds (about 139 h ideal for SAC, 170 h for TQC, one run at a time). Section 6.2 states the supplied-regime limitation, and 6.3 names the ablation as future work. (2) **The reward scale audit (R7) covers the deployed policy only**, on the test set, run after TQC seed 2 and before comparators v2 and the safety-layer work; the evaluation output gains the integrals of the seven task terms, beside the COLREGs-group and per-term integrals it already records (a change to the tools, not to `src/`). (3) **The prior-paper SAC is removed** from every table and plan: its formulation and problem differ too far from this paper's (25 uniform sectors with no target, boundary or context branch; another simulator). (4) **Multi-axis results** are per-axis tables (section 3.4.4). |

Abstract

*Draft skeleton — write last, but fix the shape now. Sentences marked \[RESULT\] wait for the three-seed campaign. \[Align with draft 4 before use: Rule 9 as an adopted convention, width as a geometric variable, the four contributions, and S13.\]*

Autonomous surface vessels in restricted waterways must follow a planned route, keep clear of static hazards and resolve encounters with other vessels in keeping with the collision regulations, under geometric limits that open-water methods do not face: in narrow water, Rule 9 decides which manoeuvre is admissible before Rules 13 to 17 decide which is required. Learning-based COLREGs methods almost universally take the other vessel's state from AIS or a simulation oracle, and are seldom evaluated against other learning algorithms on the same problem. This paper presents a deep reinforcement learning formulation for path following, static obstacle avoidance and COLREGs-aware manoeuvring against a moving target in confined water, driven entirely by onboard perception: the observation combines pooled range sensing, a map-derived boundary, a tracked-target branch and an encounter-context branch that makes the reward's judgement of the encounter observable, an interface that any range sensor with detection and tracking can supply (a two-dimensional LiDAR here). A COLREGs reward group reads the encounter as the vessel perceives it, with Rule 9 governing precedence where the open-water manoeuvre does not fit, and an engineered runtime layer separates learned speed reduction from an emergency stop. The formulation is frozen; four learners (PPO, recurrent PPO, SAC and TQC) are trained on it under an identical protocol, and the policy for deployment is the best of them on a 1,000-episode held-out test set of decoupled and coupled scenarios with constant-velocity, reactive and noncompliant targets, compared on the same set with an encounter-specific velocity-obstacle method and a dynamic-window method (S18). \[RESULT\] The deployed policy reaches \[X\]% success, against \[Y-Z\]% for the other learners and \[W\]% for the classical methods; crossing encounters remain the dominant failure \[check\]. The deployed policy is then tested in basin trials on the 1.73 m model vessel \[PENDING — field-trial design\] (S13(a)); there is no width sweep and no perception-degradation study (S14).

**Keywords:** autonomous surface vessel; COLREGs; Rule 9; narrow channel; deep reinforcement learning; problem formulation; onboard perception; collision avoidance

1\. Introduction

*Text: `planning_records/Paper3_Introduction_draft4.docx` (draft 4, 28 Sep 2026). The notes below summarise each subsection and keep the planning material behind it; edit the docx, not these notes, for wording. `planning/INTRODUCTION_DRAFT.md` (first draft, 27 Sep) is superseded.*

1.1 Problem statement and importance

Four paragraphs: (1) ASVs approaching routine operation, much of it in restricted waterways, where route following, clearance from boundaries and hazards, and traffic avoidance are coupled and confinement limits the response (Thyri & Breivik 2022; Paulig & Okhrin 2024; Waltz et al. 2025; He et al. 2025); (2) COLREGs behaviour -- Rules 8 and 9 recognise course and speed changes and conditional non-impeding duties, so the rules cannot be reduced to "turn to one side", and algorithmic encodings require interpretive choices that are a recurring error source (García Maza & Poo Argüelles 2022; Hagen et al. 2023; Sánchez-González et al. 2026); (3) perception -- target state must be estimated, LiDAR reliability depends on detection, discrimination, association and tracking (Helgesen et al. 2022), mis-tracking affects the encounter class as well as the predicted clearance, and navigable limits come from charts; (4) the combined problem studied -- path following, static-obstacle avoidance and rule-aware manoeuvring in a two-vessel encounter in laterally bounded water, motivated by a survey leg within an allocated corridor, at model scale in Paper 2's basin (Tran et al. 2026).

*Planning note kept from revision 3:* keep the survey framing to the one paragraph -- expanding it invites coverage-planning questions that are out of scope. \[Optional figure: laterally bounded corridor with reference path, one dynamic target, and static obstacles.\]

**A note on what the sensor observes** (supports section 1.1 paragraph 3 and section 2.1.1). The LiDAR does not register the basin edge; it registers the facility walls one to two metres beyond it. In a restricted waterway the navigable limit is generally not a physical structure either -- a charted depth contour, a buoyed line or a regulatory boundary -- so supplying the boundary from the chart and reserving range sensing for physical obstacles is the correct division for the application, and the basin reproduces it exactly.

1.2 Literature review

*Organised by the elements of the combined problem (draft 4 section 1.2 opening paragraph); broader surveys: Huang et al. 2020; Burmeister & Constapel 2021; Öztürk et al. 2022; Qiao et al. 2023; Wu et al. 2025b. Revision 3's section 2 threads are folded in below; references it named that draft 4 does not cite are flagged "(rev 3 only)".*

1.2.1 Classical and explicitly constrained collision avoidance

VO (Kuwata et al. 2014 with COLREGs constraints; Huang et al. 2019 generalised VO; Thyri & Breivik 2022 encounter-specific VO for confined waters), DWA (Fox et al. 1997; COLREGs variants Guan & Wang 2023, Xu et al. 2024), artificial potential fields, and MPC (Johansen et al. 2016; Eriksen et al. 2019; Thyri & Breivik 2022; Han et al. 2024; He et al. 2025). Three limitations motivate learning: tuned weights and thresholds per vessel and geometry; guarantees resting on modelling and prediction assumptions (constant target velocity, simplified dynamics, explicit obstacle geometry -- Gonzalez-Garcia et al. 2022; local minima); and error-prone COLREGs encodings (Sánchez-González et al. 2026: 855 papers, 464 attempting the rules, none fully compliant, 29 recurring errors). Formal specification and enforcement (Krasowski & Althoff 2024; Vain et al. 2025) versus reward-encouraged compliance; Burmeister & Constapel 2021 on incomplete coverage and narrow channels. **Role in this paper:** the classical methods are the natural reference and supply the nonlearned comparators (section 3.3); the COLREGs-VO rule applied from the other vessel's side is also the reactive target model of the robustness set.

1.2.2 Deep reinforcement learning for path following and COLREGs-aware navigation

Path following (Woo et al. 2019; Zhao et al. 2021; Deraj et al. 2023); combined following and avoidance (Meyer et al. 2020a, the path-relative observation with rangefinder sensing this work descends from; Meyer et al. 2020b); multiship and benchmark studies (Zhao & Roh 2019; Sawada et al. 2021, Imazu; Chun et al. 2024); risk-based COLREGs rewards (Heiberg et al. 2022). Since 2023: spatial-temporal recurrent RL and partial observability (Waltz & Okhrin 2023; Zheng et al. 2023), reward balancing and hybrids (Lou et al. 2024; Yang et al. 2024; Sonntag et al. 2025), safe RL (Wang et al. 2024), collision grids (Teitgen et al. 2023), reward-parameterisation effects (Krautwig et al. 2025). **Position:** supplied context versus learned response -- Hart et al. 2024 (supervised risk estimate fed to the agent) and Zhang et al. 2022 (model-reference correction); this paper makes the split explicit (the encounter classifier supplies the regime, the policy the response), which the paper states as a limitation rather than measuring (the ablation is dropped, S19). Paper 2 (Tran et al. 2026) added feasibility-inspired LiDAR sector pooling, a staged curriculum and field validation for the static case.

1.2.3 Restricted and inland waterways

Burmeister & Constapel 2021 (of 48 approaches, two address Rule 9 even in part) \[VERIFY the "four mention" figure used in revision 3; draft 4 states only the two\]; Hansen et al. 2022 (Rule 9 applicability from manoeuvrability); classical confined-water planners (Thyri & Breivik 2022; He et al. 2025); canal field trials (Kim et al. 2024); learned river and restricted-water navigation (Paulig & Okhrin 2024; Hao et al. 2024). **Closest prior work:** Waltz, Paulig & Okhrin 2025 (two-level RL for inland waterways, AIS-derived scenarios, versus APF). This paper differs in model-scale LiDAR-tracked encounters, a single learned rudder-and-propulsion policy, and evaluation organised around manoeuvre admissibility as width changes. The question is how available lateral clearance changes the feasible response within a declared rule interpretation -- course alteration, reduced speed or holding astern -- with width a geometric variable, and slowing conditioned on predicted clearance (stopping in a reciprocal vessel's path may preserve the conflict). (rev 3 only: de Vries et al. 2022, urban canals.)

1.2.4 Onboard sensing, integrated control and physical validation

Sensor-driven avoidance outside RL: Han et al. 2020 (radar/LiDAR/camera fusion, field tests), Eriksen et al. 2019 and Kufoalor et al. 2020 (MPC at sea), Helgesen et al. 2022 (environment-dependent tracking), Villa et al. 2020 (LiDAR path following in harbour), Kim et al. 2022 (2D LiDAR avoidance on an ASV -- the closest platform analogue). Sensor-driven DRL: Lin et al. 2025 (distributional RL, LiDAR segmentation in the higher-fidelity simulator, cross-algorithm comparison), Wu et al. 2025a (sensor-level mapless navigation with action correction). **Open:** a policy whose encounter information comes from a tracker in training as well as evaluation, interacting with width-dependent constraints. Physical transfer: Slawik et al. 2024 (domain randomisation, basin), Wang et al. 2025 (model-scale SAC on rudder and propeller), He et al. 2025 (towing tank), Paper 2. Transfer evidence is reported separately from numerical results (S13).

1.2.5 Evaluation practice and the research gap

Automatic COLREG evaluation encodes the evaluator's interpretation (Hagen et al. 2023); declared behaviour-level metrics rather than legal-compliance claims (Sánchez-González et al. 2026). Algorithm comparisons (Larsen et al. 2021, PPO most robust; Lin et al. 2025; Slawik et al. 2024; Hart et al. 2024) and their statistical pitfalls (Agarwal et al. 2021; Patterson et al. 2024). This paper fixes observation, reward, dynamics and curriculum across learners and separates checkpoint selection from held-out evaluation -- a comparison under a specified protocol, not a claim of equally optimal tuning. **Gap (draft 4's closing paragraph):** at the intersection of sensing, constrained manoeuvring and evaluation -- classical methods handle confinement but depend on tuned rules and prediction; learned COLREGs policies mostly use idealised or noise-perturbed target states in open or loosely bounded water; learner comparisons rarely fix the formulation independently of the learner.

*Revision 3's five gaps, now argued across section 1.2 (kept as a checklist):* assumed target state (sections 1.2.2, 1.2.4); open-water bias and Rule 9 (section 1.2.3); traffic and clutter treated apart (sections 1.2.1, 1.2.3) \[VERIFY — survey the handful of papers with both\]; single-learner studies (section 1.2.5); limited physical validation (section 1.2.4) \[VERIFY — weakened since 2024; differentiate rather than assert exclusivity\].

1.3 Scope, objective and contributions

**Objective** (revised 2026-10-10): develop a DRL formulation, driven by onboard perception, for path following, static-obstacle avoidance and COLREGs-aware two-vessel encounters in confined water, and deploy the best policy trained on it, chosen on the test set against the other learners under the same formulation (PPO, recurrent PPO, SAC, TQC) and against classical references.

**Scope** (draft 4; see S1-S13): one model-scale ASV, one moving target, up to three static obstacles, 10 x 25 m basin and parallel-walled channels; continuous rudder and propulsion at 2 Hz; target tracks from onboard perception, a 2D LiDAR in this study (no AIS, no ground-truth target in the policy input); map-derived boundaries at the estimated pose; detection, tracking and encounter classification as explicit stages. Selected requirements of Rules 8, 9, 13, 14, 15/16 and 17(a)(i); the narrow-channel crossing convention (S3); excluded: Rule 17(a)(ii) and 17(b), Rule 18, Rule 19, signalling, simultaneous multiple targets. Non-cooperating training targets that hold their course (30 % of stage-7 coupled targets change speed once); reactive and noncompliant targets in the test set and the robustness set. Headline: test set v4, 1,000 held-out episodes (664 decoupled, 336 coupled); no width sweep (S14); the runtime safety layer evaluated separately and off for learned-behaviour results.

Research questions -- draft 4 states four central questions (RQ1, RQ6, RQ3, RQ5):

- **RQ1 —** Can the policy complete the navigation task -- path following, static obstacle avoidance and a two-vessel encounter in confined water, from onboard range sensing -- while producing encounter-consistent behaviour?

- **RQ6 (new, draft 4) —** How does the policy's response change as maneuvering room decreases: which responses remain admissible, and when does it switch from course alteration to speed reduction or holding astern? \[Dropped with the width sweep (S14).\]

- **RQ3 —** How does perception degradation affect safety and compliance, and at what point does the policy fail unsafely rather than conservatively? \[Dropped with the perception study (S14).\]

- **RQ5 —** Under one frozen formulation, which learner's policy should be deployed, and which differences between on-policy, recurrent, off-policy and distributional learners -- in success, behaviour by encounter type and robustness to target behaviour -- remain meaningful across training seeds?

- **RQ2 —** Does making the encounter explicit in the observation improve compliance over reward shaping alone? \[Dropped with the ablation (S19, 2026-10-24); the supplied-context argument of section 1.2.2 stays as positioning and as a stated limitation.\]

- **RQ4 —** Does domain randomisation over identified model uncertainty close more of the sim-to-field gap than improved nominal identification alone? \[Not in draft 4 — S13.\]

Contributions (draft 4's four; the table maps revision 3's seven onto them)

| **\#** | **Contribution** | **Revision 3** |
|--------|------------------|----------------|
| C1     | **A sensor- and encounter-aware learning formulation.** One observation and reward design combining tracked target information from onboard range sensing (implemented with a 2D LiDAR; the interface admits other range sensors such as radar), map-derived boundaries, own-ship and path states, and explicit encounter context: the classifier supplies the encounter regime, the policy learns the rudder-and-propulsion response. Encounter-dependent reward terms support route completion, clearance and selected COLREGs behaviours, without treating reward shaping as a formal safety guarantee. Frozen, digest-checked and released. | C1 |
| C2     | **A geometric framework for constrained encounter responses.** Relates the available maneuvering space to an encounter-consistent course alteration, holding astern, or speed reduction where predicted clearance supports it, and makes that judgment observable to the policy through an admissibility test and a stopping test; width thresholds derived from the same geometry state, in ship breadths, where each open-water maneuver stops fitting, without inferring legal precedence from width alone (revision brief, contribution 2; no width sweep, S14). | C2 + C4 |
| C3     | **A deployed policy, chosen and benchmarked fairly.** Four learners (PPO, recurrent PPO, SAC and TQC) trained on the frozen formulation (3.0 M steps, 3 seeds, checkpoints selected on a separate validation set); the policy for field deployment is the better of SAC and TQC on the held-out test set, benchmarked there against the other learners and against the encounter-specific VO (after Thyri and Breivik 2022) and LOS-PID with a DWA layer on the same perception pipeline, with seed variability reported. The learned baselines close the gap in Paper 2's first draft, which evaluated SAC alone. | C3 (+ C6's protocol) |
| C4     | **An evaluation design separating task completion, safety and robustness to target behavior.** The held-out test set, with decoupled and coupled scenarios and constant-velocity, reactive and noncompliant targets, distinguishes task completion from collision type and target-ship safety, and COLREGs-aware behavior is shown through encounter snapshots (revision brief, contribution 4; S14). | C5 + C6 |
| C5     | **Physical encounter trials** (S13(a)). The deployed policy runs on the model-scale vessel in the basin, its observation built from the real LiDAR scan and pose by the same perception modules as in simulation, in encounters with a moving target \[PENDING — field-trial design\]; the transfer evidence is reported apart from the numerical results (revision brief, contribution 5). | C7 |

Novelty in one paragraph

*For the cover letter. Draft 4 claims the intersection (section 1.2.5); keep this consistent with it.*

To the authors' knowledge this is the first learned COLREGs collision-avoidance study that (i) perceives the other vessel only through onboard perception (a 2D LiDAR here, behind a sensor-agnostic observation), in training as well as evaluation, while (ii) evaluating how the admissible encounter response changes with channel width under a declared narrow-channel convention, (iii) in water shared with static obstacles, and (iv) deploys a policy chosen against other learner families trained on the identical frozen formulation and against classical references, on a held-out test set. \[VERIFY each clause against section 1.2 before submission; claim the intersection, not any single thread.\]

What the paper does not claim

- Multi-target encounters, Rule 19 restricted visibility, sound signals, Rule 18 responsibilities, the open-water Rule 15/17 role table, or legal compliance / certification: violation terms are declared behavioural proxies (`PAPER3_CLAIMS_AND_TABLES.md` section 2).
- That operating bounds establish the legal applicability of Rule 9, or that a width is a universal legal criterion for a narrow channel (draft 4).
- Rule 17(b) last-moment action: out of scope by construction (S5).
- That every dynamic encounter is avoidable: the A* route check is static (S9); the Paper 2 set's space-time check (F107) is not applied to the headline suite.
- Co-operation from the other vessel: training targets never give way; reactive and non-compliant targets are evaluated, not trained on (S8).
- A best learner: with three seeds, headline differences of about 0.05 are at the edge of separability and crossing-level differences are not separable (F90); learners are compared on the headline, crossings reported per seed.
- Performance in untrained geometry: the headline holds the class x geometry combinations training draws (suite 3.4); channels narrower than the test set's (nominal 6.9–10.6 m, about 5.6–6.2 m at the CPA in the narrowest cases) are not evaluated, since there is no width sweep (S14). The Paper 2 deployment layouts, where traffic meets fixed hazards at close quarters, expose a combination the baseline formulation does not train (F105) — \[TBC S12\] how and whether it is reported.

Paper structure (draft 4's last paragraph): Section 2 the formulation (problem and workspace; COLREGs scope and the geometric framework for constrained encounter responses; the frozen perception, observation, reward and curriculum); Section 3 the learners, training protocol, classical comparators and evaluation design; Section 4 the simulation results on the held-out test set; Section 5 the field trials of the deployed policy (S13(a)); Section 6 discussion and limitations; Section 7 conclusion (revision brief, section 2).

2\. Formulation

*Writing plan with verified numbers, figures, equations and drafting inputs: `planning_records/FORMULATION_PLAN.md` (2026-09-28). Where the notes below disagree with it (its section 6), the plan is right.*

2.1 Problem and workspace

2.1.1 Vessel and workspace

Model-scale Bluefin: 64.55 kg, LOA 1.73 m, LBP 1.57 m, breadth 0.50 m, draft 0.19 m, Iz 10.45 kg·m². Three-degree-of-freedom Fossen model with explicit actuator dynamics (section 2.3.5), integrated at 0.1 s. Decision period 0.5 s (2 Hz), episode cap 180 steps (90 s); cruise 0.558 m/s at 6 RPM, RPM limited to 0-12 (no astern).

The default workspace is the 10 x 25 m basin of Paper 2's field site, with straight and slanted legs from y = 2 m to y = 22 m (basin mode, F74); parallel-walled channels of variable width carry the classes whose response the width decides (head-on, crossing, overtaking). Reference paths are deliberately not centred, so that the boundary observation branch carries information not already present in the cross-track error. Simulation matches the physical basin, with a maximum corridor width of 10 m (20 ship breadths), so that every simulated width is physically reproducible. The unconfined reference case is supplied instead by the open-water variant of the external benchmark (section 3.4.2), rather than by a wider simulated channel.

There is no width sweep (S14). The reference widths of the draft's Section 2.2.4 (crossing 7.60 m; head-on 3.80 m, or 6.30 m with the target on the centerline; overtaking 4.26 m) are design reasoning that places the test set's channel encounters, not predictions tested by a sweep.

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
| 8          | Action to avoid collision                              | Implemented — 8(b) readily apparent (a 20° alteration or a 30 % slowdown counts as one full action), 8(d) passing at a safe distance (the required separation d_req = 2.5 m and the ship-domain term), 8(e) slacken speed where no alteration fits. 8(a) only as timing pressure: the Rule 8 penalty grows as TCPA falls below 15 s, and many confined-water encounters begin inside that horizon, so ample time is not claimed. |
| 9          | Narrow channels                                        | Implemented as an adopted operating convention — 9(a) keep starboard, 9(b) do not impede, 9(e) overtaking. Shapes the admissible response under Rules 13–16 (section 2.2.3).          |
| 13         | Overtaking                                             | Implemented — two classes: own ship overtaking, own ship overtaken.                                                                                                              |
| 14         | Head-on                                                | Implemented — port-to-port passing. Keeping to the starboard side of the channel (9(a)) satisfies the rule without an alteration; a starboard alteration is required only when the target is not already clear, subject to available channel width. |
| 15         | Crossing                                               | Implemented — own ship gives way in all crossing encounters under the narrow-channel convention (S3), passing astern of the target: a starboard alteration for a target from starboard, a **port** alteration for a target from port (A17). Stated explicitly, since a port alteration in a crossing reads as a violation under the open-water role table. |
| 16         | Give-way action                                        | Implemented — early and substantial, per Rule 8.                                                                                                                                 |
| 17         | Stand-on action                                        | Partially — 17(a)(i) passive course-keeping (course and speed held) for the being-overtaken class. The hold penalty is released in extremis (DCPA < d_req, TCPA < 5 s), so the reward never pays for holding course into a collision; action is then left to the collision and ship-domain terms. Rule 17(b) action itself is not claimed: no evaluation draw requires it (S5, baseline-v2). Active release under 17(a)(ii) is out of scope and identified as future work. |
| 2, 5, 6, 7 | Responsibility, lookout, safe speed, risk of collision | Acknowledged; operationalised implicitly — the collision risk index for Rule 7, the perception pipeline for Rule 5.                                                              |
| 18         | Responsibilities between vessels                       | Out of scope — own ship and target are similarly sized, so the asymmetry the rule requires does not exist. The always-give-way simplification is instead motivated by Rule 9(b). |
| 19–31      | Restricted visibility, lights, shapes, sound signals   | Out of scope — no corresponding sensing or actuation on the platform.                                                                                                            |

*State this table in the paper. Reviewers who know COLREGs read the omissions as closely as the inclusions, and naming the Rule 18 exclusion pre-empts the obvious question.*

2.2.2 Encounter classes

| **Class**       | **Governing rule** | **Own-ship obligation**                                      |
|-----------------|--------------------|--------------------------------------------------------------|
| None            | —                  | Follow path                                                  |
| Head-on         | 14, 9(a)           | Pass port to port; alter to starboard only when the target is not already clear, subject to channel width |
| Crossing        | 15, 16, 9(b)       | Give way regardless of which side the target approaches from, passing astern: starboard alteration from starboard, port alteration from port (A17) |
| Overtaking      | 13, 16, 9(a), 9(e) | Keep clear of the vessel being overtaken, passing to port of it (geometric reading of 9(e) only); hold astern at reduced speed where the pass does not fit |
| Being overtaken | 13, 17(a)(i)       | Hold course and speed; the hold penalty is released in extremis |

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

The width at which a channel becomes "narrow" for a given encounter is a reference width derived from the geometry (the draft's Section 2.2.4), not a result: there is no width sweep (S14). It is defined geometrically — in ship breadths and in terms of the lateral excursion the compliant manoeuvre requires — not by reference to where any method fails.

2.3 Perception, observation, reward, vessel model and curriculum

2.3.1 Perception pipeline

Raw LiDAR: 360°, 720 beams, 0.5° angular resolution, 10 Hz. Returns falling outside the known channel polygon are gated geometrically, because the sensor is mounted above the basin wall and would otherwise register objects beyond it as obstacles or, worse, as phantom targets with plausible velocities. Remaining returns are clustered, ego-motion compensated, associated to tracks, and velocity-estimated by a constant-velocity Kalman filter. The pooled sector channel is built from all gated returns, the target hull included (corrected 2026-10-17); the dynamic track feeds the target branch.

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

One multi-input policy over the **six** observation branches (70 values), identical for every learner: PPO, RecurrentPPO, SAC and TQC, which differ in the algorithm alone. Trained from scratch — no warm start from the prior policy, whose observation and reward semantics have both changed on every channel. The prior policy is not a comparator either (S19, 2026-10-24): its formulation and problem differ too far from this paper's.

\[Largely settled: RecurrentPPO is the recurrent learner; the explicit tracker carries some memory already; occlusion of the target behind a static obstacle is the case that would justify more.\]

3.2 Training protocol

3.0 M environment steps per run (stage boundaries at 0.16-2.0 M on the 2.5 M schedule, stage 7 to the end), **three** seeds per learner (2026-09-24; was five), checkpoint selected on the validation set (goal − 2 × collision); test set v4 touched once per policy, and the deployed learner chosen on it (S11). Runs trained on demand, one learner × seed at a time. Hyperparameters in Appendix B.

3.3 Baselines

| **Family** | **Method**                                                                                         |
|------------|----------------------------------------------------------------------------------------------------|
| Classical  | LOS-PID with dynamic window approach (Fox et al. 1997)                                            |
| Classical  | Encounter-specific velocity obstacles (after Thyri and Breivik 2022; S18) — same encounter context and narrow-channel convention as the formulation; side rule judged on the encounter (A35, comparators v2). COLREGs-VO (Kuwata et al. 2014) is not a comparator; its rule from the target's side remains the reactive target model |
| Learned    | PPO, RecurrentPPO, SAC, TQC on the identical formulation: the deployed policy (the better of SAC and TQC on the test set) and the learned baselines |

All comparators receive the same perception pipeline (draft 4). Classical baselines are run against reactive targets as well, or the comparison is not like-for-like. Tuned on the validation set and pinned in `configs/comparators_v1.json` (comparators v2: `configs/comparators_v2.json`). The paper has these two classical comparators only: no NMPC, no prior-paper SAC and no ablated policy (S19).

3.4 Evaluation design

3.4.1 Scenario generator

Scenarios are parameterised by encounter class rather than by initial condition: sample the class, then a heading-intersection angle from its valid interval, then a target speed, then a spawn TCPA, and solve backwards for the spawn position that produces that geometry. Random spawning frequently produces targets that pose no threat and wastes training samples. The null class — a target on a similar course, posing no conflict — is included deliberately, since it never arises from a purely class-conditioned spawner but does occur in practice.

The same generator produces training scenarios and the evaluation suite, with disjoint seeds.

3.4.2 Evaluation suite

*Superseded: the evaluation is test set v4 (1,000 episodes: 664 decoupled, 336 coupled; the draft's Section 3.4.1 and Appendix E), and there is no width sweep (S14). The notes below describe suite 3.4, which test set v4 absorbed, and are kept for its history.*

One frozen tier, versioned and hashed before the first training run.

\[Tier A, the 38 named deterministic cases, is **out of this paper** (2026-09-24). It exists in the suite and can be reported later; the argument here rests on Tier B and the width sweep.\]

- **Tier B (the default frozen suite, suite 3.4) —** a held-out draw of the validation set's kind of scenario: only the positions differ. 8 balanced cells × 100 = 800 episodes per seed, constant-velocity targets as in training; every encounter class in the basin, and head-on, crossing and overtaking in channels of 7.5–10 m — the class × geometry combinations training draws (being overtaken and null are trained in the basin only). **Robustness set (R3):** the same scenarios and episode seeds with a compliant reactive target (700) and, in head-ons, a non-compliant one that alters to port (200); a non-compliant stand-on target moves exactly like the constant-velocity one, which already is the give-way vessel that does not give way. Narrower water is the R4 width sweep. Static clutter is 0–3 obstacles per episode, dropped where they would decide the encounter.

- **External benchmark —** the "Around the Clock" set of 24 single-ship encounters at equally spaced target headings. \[Out of this paper (2026-09-24) unless built; see section 4.8.\]

- \[TBC S12\] **Paper 2 deployment-layout set** (F104-F107): 630 field-stageable cases on the three published layouts, all space-time solvable.

3.4.3 Difficulty definition

Geometric only: channel width in ship breadths, spawn TCPA, static clutter count. Never defined by baseline performance. One stratum is deliberately narrow enough that no method passes cleanly — a suite the proposed method succeeds on everywhere reads as constructed regardless of how it was built.

3.4.4 Metrics

**Task** (as computed by `src/metrics.py`; the draft's Section 3.4.3, corrected 2026-10-22). Success rate; collision rate by type (static obstacle, boundary, target vessel); RMS cross-track error; completion time and path-length ratio of successful episodes; mean and minimum speed; minimum clearances to the panels and the boundary. **Actuation:** rudder saturation fraction, mean rudder rate, control effort. The maximum cross-track error is not computed.

**COLREGs.** Shown by trajectory snapshots of each encounter type (section 4.2), not by per-rule rates (decision 2026-10-08). Target-ship safety is quantified: target collision rate, ship-domain intrusion rate and depth, closest approach (centre to centre) and minimum hull-to-hull clearance.

**Perception.** No perception metric is computed by any evaluation, and the perception study is out (S14): none is reported.

**Intervention dependence** (draft 4 C4). Safety layer intervention rate, reported apart from learned compliance.

**Decided (2026-10-24): per-axis tables, no Pareto view and no weighted scalar.** The aim is a table a reader can take in: the main-text results table carries a short set of metrics (success, collisions by type, target-ship safety, path keeping), and the full per-axis table of `results/baseline_v3/paper_table.md` (speed, actuation, intrusion depth and time, clearances) goes to an appendix.

3.4.5 Ablation matrix (dropped, S19; kept as the reserve design)

|                          | **Encounter feature OFF** | **Encounter feature ON** |
|--------------------------|---------------------------|--------------------------|
| COLREGs reward terms OFF | Avoidance only            | Told, not rewarded       |
| COLREGs reward terms ON  | Learned from kinematics   | Full method              |

This answers the question a sceptical reviewer will actually ask: is compliance learned, or handed to the agent? (draft 4 section 1.2.2: supplied context versus learned response.) Supplementary leave-one-out ablations on individual COLREGs terms and on the boundary observation branch. **Dropped (S19, 2026-10-24):** the paper reports no ablation. If review asks for one, the reserve is the single cell "learned from kinematics" (encounter context hidden, COLREGs terms kept) on the deployed learner at three seeds; it needs the observation-side switch wired in `src/` and a digest of its own, and a rule-related measure that the paper otherwise does not report.

4\. Simulation results

*Restated 2026-10-24 to the draft's Section 4 and decisions S14 and S19: the width sweep, the perception study, the ablation, the external benchmark and the prior-paper SAC are out. Every number comes from `MASTER_ARTIFACT_RESULTS.md` and `results/baseline_v3/paper_table.md`; the safety layer is off everywhere except 4.5.*

**Presentation (S19).** Per-axis tables, no Pareto view and no weighted score. Rates are means of per-episode outcomes pooled over the seeds (equal to the seed mean, since every seed runs the same 1,000 episodes); continuous metrics are interquartile means; 95 % intervals by stratified bootstrap over the test-set cells. Success is also given by seed, with the seed range. The main text carries the principal metrics; Appendix G carries the full per-axis table for all episodes and for the decoupled and coupled families.

4.1 Overall performance on the test set

Table R1 — principal metrics on all 1,000 episodes, metrics as rows and methods as columns: SAC, TQC, PPO, RecurrentPPO (the deployed one of SAC and TQC marked), LOS-PID + DWA, encounter-specific VO. Rows: success (all, decoupled, coupled); target, obstacle and boundary collision rates; ship-domain intrusion rate; minimum hull-to-hull clearance to the target; RMS cross-track error; completion time of successful episodes. **All columns final (2026-10-25):** twelve runs; LOS-PID + DWA re-run under comparators v2 reproduces v1 in all 1,000 episodes; the encounter-specific VO (comparators v2) 0.675.

Table R1b — success by run: every seed at its best validation-set checkpoint and as the final 3.0 M model (S15), with each learner's seed mean and range; RecurrentPPO seed 1's collapse stated with its cause (S16).

Figure A — success by encounter type on the whole test set: grouped bars, seed mean with the individual seeds and the seed range; classical comparators hatched, without intervals (`tools/diagnostics/paper_figures.py`). Figure B — validation-set score every 2 × 10⁵ steps for the four learners: seed mean, seeds and seed-range band, curriculum stages marked. No selected-policy marker and no trade-off figure (2026-10-08).

4.2 COLREGs-aware behavior by encounter type

Figure R2 — trajectory snapshots of the deployed policy: one successful episode per encounter type (head-on, crossing from starboard, crossing from port, overtaking, being overtaken) and a no-target layout, spread across decoupled and coupled layouts, obstacle counts and leg shapes (`tools/paper_snapshots.py`, `results/paper_snapshots`). Each case is verified with the track-based rule checks of `tools/diagnostics/colregs_compliance.py`. A selection, not a sample; no per-rule rates (2026-10-08). **The figure exists** (`results/paper_snapshots`, 2026-10-06): it was made with SAC seed 0's best checkpoint, the same model file as the deployed policy (decided 2026-10-25). Panel (f), being overtaken by a reactive target (BAS-BO-RE-016), keeps its track and speed within the check's 0.15 m and 0.15 m/s only until the target is within 2.5 m, then turns about 20° and slows to about 0.40 m/s at the CPA; a steadier being-overtaken case could replace it. The draft caption's "targets keep their course" does not hold for that reactive target.

4.3 Performance by target behavior

Table R3 — success by target behavior for every method: the decoupled part with constant-velocity, reactive and noncompliant targets, and its head-on cells, where all three occur; the coupled part with constant-speed and varying-speed targets (`MASTER_ARTIFACT_RESULTS.md`, sections 10–12). \[RESULT\]

4.4 Reward scale audit

Table R7 — the episode-integrated contribution of each reward term (the seven task terms, the COLREGs group with its five terms, and the terminal rewards) under the deployed policy on the test set, with the empirical ordering checked against the magnitude hierarchy of Section 2.3.3. Deployed policy only (S19). The test-set output records per episode the return and the integrals of the COLREGs group and its five terms, but not the seven task terms (`DENSE_TERMS`, `src/reward/terms.py`); adding them is a change to the evaluation tools. Run after TQC seed 2, before comparators v2 and the safety-layer work. **Done 2026-10-25:** `results/baseline_v3/reward_scale_audit.md` (run `sacs0_bl3_audit`; the evaluation output now records `reward_int_<term>` and `reward_active_<term>`).

4.5 Runtime safety layer

Table 13 — the deployed policy on the test set with the layer off and on (S10, decided 2026-10-25; Algorithm A3): own-vessel collisions by type and family, goals, stops, target strikes on the stopped ship (scored apart, under the target-braking convention), timeouts, and the share of decisions and episodes with a changed command; the paired transitions; the validation-set figures, on which the layer was chosen. Statements kept: an empirical evaluation, not a guarantee; the test set had been used in the layer's development; the layer steers as well as stops. Numbers: `planning/PAPER/PAPER3_ALGORITHMS.md`, Algorithm A3 notes; record `planning_records/SAFETY_V21_PLAN.md` section 15; report `results/safety_dev/testset_v4_main/reports/ts4_final/`.

Physical transfer (parked — S13; revision 3 section 7, not in draft 4's roadmap)

*Under S13(a) this becomes a numbered section after Section 4 (and the roadmap in section 1.3 gains it); under S13(b) it moves to the field paper and section 5.5 lists it as future work.*

P.1 Platform and setup

Bluefin model vessel, RPLidar C1, UDP offboard control; the policy decides at 2 Hz, the rate the deployed bridge delivers (median command period 0.5 s over 1,085 logged frames), which is also the simulator's decision rate. The basin is an indoor pool within a hall, with the facility walls standing one to two metres beyond the pool edge. Returns from beyond the pool boundary are removed by geometric gating against the known pool polygon rather than by a physical barrier: the facility walls carry the fixed features on which localisation depends, and occluding them would remove the only available registration reference. Gating is in any case necessary rather than merely preferable, since operators standing on the deck lie at scan height and would otherwise be tracked as dynamic targets. Localisation is run on the complete scan and the obstacle gate applied only afterwards, so that the walls serve as a registration asset while being excluded from the tracker.

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

\[RESULT\] RQ1, RQ6, RQ3, RQ5 (draft 4's central questions); RQ4 under S13(a); RQ2 is dropped with the ablation (S19).

5.2 Claim ledger

*Every claim in the abstract and conclusion maps to a table or figure. Complete this before writing either. The working ledger is `planning/PAPER3_CLAIMS_AND_TABLES.md`.*

| **\#** | **Claim** | **Evidence** | **Status** |
|--------|-----------|--------------|------------|
| 1      |           |              |            |
| 2      |           |              |            |
| 3      |           |              |            |

5.3 Limitations

- Model scale; full-scale correspondence rests on the Froude scaling statement in section 2.1.2

- Single dynamic target; sequential rather than concurrent multi-vessel encounters

- Active stand-on release under Rule 17(a)(ii) is out of scope

- The deployed learner is chosen on the test set, so its test-set figures include that choice (between two learners, on 1,000 episodes × 3 seeds, with checkpoints selected on the validation set); the field trials are its independent evaluation \[S13\]

- Only LiDAR supplies the onboard perception here; the observation interface is sensor-agnostic, but another range sensor such as radar is not tested

- Stand-on course-keeping competes with the ship-domain term: the domain term (weight 12.5 per decision) outweighs the whole COLREGs group (capped at 9), so letting an overtaking target into the domain astern costs more than departing from Rule 17(a)(i). The learned policies change speed while being overtaken (seed 0: SAC speeds up, PPO slows), shown by the track-based diagnostics, not reported as a rate

- Calm water with no generated waves or current

- Three seeds is a limited estimate of training variability: a headline difference of ~0.05 is at the edge of separability and the measured 0.20 crossing spread is not separable, so crossing behaviour is reported descriptively per seed. One of the three RecurrentPPO seeds collapsed mid-training (KL-limited updates; S16) and is reported, not replaced

- The encounter classifier supplies the rule regime, so rule-related behavior is partly supplied rather than learned; no ablation separates the two (S19), and the ablation is named as future work

- Simulator ground truth drives the collision and clearance rewards in training (the policy input is perception-only; draft 4 section 1.3)

- Static route checks do not prove every dynamic encounter avoidable (S9)

- \[S13(b)\] No physical encounter trials in this paper; simulation with synthetic LiDAR does not substitute for them (draft 4 section 1.2.4)

5.4 Future work

- Extension to concurrent multi-vessel encounters via the existing N_max parameter

- Active stand-on release under Rule 17(a)(ii) against non-compliant targets

- Disturbance rejection under wave and current loading

- Closing the sim-to-field gap with logged field runs: fine-tuning the deployed policy in a field-calibrated simulator, or offline RL from the field logs (`planning/FUTURE_WORK_IDEAS.md`, idea 1; a candidate conference paper)

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
| Ship domain         | Compressed asymmetric, 3.14 / 1.57 / 1.25 m ahead, astern and abeam (2.0 / 1.0 / 0.796 Lpp), d_req = 2.5 m (frozen in baseline-v2)         |
| Recurrence          | RecurrentPPO is one of the four learners                                                                                                  |
| Multi-axis results  | Per-axis tables (2026-10-24): a short main-text table, the full per-axis table in an appendix; no Pareto view, no weighted scalar         |
| Ablation, prior SAC, NMPC | Dropped (S19, 2026-10-24): no ablation (a one-variant reserve on the deployed learner), no prior-paper SAC, no NMPC |
