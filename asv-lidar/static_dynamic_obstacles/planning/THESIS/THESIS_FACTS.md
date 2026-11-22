# Thesis facts (status 2026-11-21)

The numbers, decisions and wording that the thesis may state, each with the file that records
it. Anything not here, or not in a file named here, is checked against the repository before it
appears in a draft. Regenerated when results change, first after the field sessions. Older
copies are discarded, not kept beside this one.

Repository paths are relative to `asv-lidar/static_dynamic_obstacles/` unless they start with
`field_deployment/`.

## 1. Publications

| | Reference | Status | Role in the thesis |
|---|---|---|---|
| P1 | Tran et al., "Autonomous surface vessels path following and static obstacle collision avoidance using deep reinforcement learning", ICMCR 2026, doi 10.1109/ICMCR69541.2026.11533898 | Published | Development only: the first study, mostly open water, grid observation, SAC chosen over PPO |
| P2 | Tran, N.H.; Nguyen, H.D.; King, P.; Tran, M., "Deep reinforcement learning control for path following and static obstacle avoidance for autonomous surface vessels", *Drones* 2026, 10, 680, doi 10.3390/drones10090680 | Published | Cited for the components Paper 3 keeps: feasibility sector pooling (its Algorithm 1), the path-relative frame, the joint rudder and propulsion action; also the first basin trials |
| P3 | "COLREGs-aware collision avoidance in confined waters using deep reinforcement learning" (working title) | Sections 1-4 drafted (`Paper3_Sections1-4_clean_rev05Nov.docx`); submission planned Dec 2026 to Jan 2027; journal open | Main source for the formulation, the results and the field programme |

**Paper 2 results.** These are published figures, made with the earlier formulation:
- **Simulation:** SAC reaches the goal in 95.0 % of 500 randomised episodes with zero to four
  obstacles, and 92.4 % on average over three seeds. A tuned line-of-sight controller with
  potential-field avoidance reaches 92.2 % and PPO 90.5 %. SAC's RMS cross-track error is about
  30 % lower than both baselines, with a smaller obstacle clearance.
- **Field:** in the 25 m × 10 m indoor basin the policy ran once in each of three layouts and
  avoided every obstacle, with minimum clearances of 0.50-1.17 m. RMS cross-track error was
  1.14 m in the field against 0.69 m in simulation.

## 2. Platform

| Item | Value | Source |
|---|---|---|
| Vessel | Bluefin model-scale ASV: length 1.725 m, beam 0.50 m, draft 0.193 m, mass 64.55 kg; one propeller and a stern rudder (±40°) | `planning/MASTER_ARTIFACT_FACTS.md`, section 1 |
| Range sensor | 360° 2D LiDAR at the bow, 0.8625 m ahead of the reference point; 720 beams at 0.5° | Same; `field_deployment/README_P3_BRIDGE.md` |
| Model | 3-DOF manoeuvring model in the form of Fossen, with MMG-style hydrodynamic and rudder terms; identified parameters | `MASTER_ARTIFACT_FACTS.md`, section 1; Paper 3, Appendix A |
| Target vessel | Same class and hull outline as the own ship | `MASTER_ARTIFACT_FACTS.md`, section 1 |
| Decision interval | 0.5 s (2 Hz) | Paper 3, Section 2.1.1 |
| Cruise speed of the deployed policy | about 0.56 m/s | `README_P3_BRIDGE.md` |
| Field site | Indoor basin, 25 m × 10 m, calm water | Paper 2 |

## 3. The final formulation (baseline-v3, digest 03a9c0c22f53ee0b)

**Observation: 70 values.**
- **lidar (27):** the map-gated scan over ±135°, feasibility-pooled into 27 non-uniform sectors
  (15 × 6° within ±45°, 8 × 11.25° abeam, 4 × 22.5° on the quarters). Safe width 0.80 m; beam
  spacing 0.5°; closeness = clip(1 − d/16 m, 0, 1).
  - The branch holds every gated return, the target's hull included.
  - It changes three things from Paper 2's version (25 uniform sectors, beam spacing 1.205°):
    map-gated input, non-uniform sectors and the true beam spacing. Pooled values are not
    comparable between the two studies.
- **boundary (7):** rays at −90° to +90° in 30° steps, against the map polygon at the estimated
  pose, up to 16 m.
- **Ego and path:** estimated surge, sway and yaw rate; cross-track error, positive to
  starboard (the reverse of Paper 2's sign), normalised by the distance to the boundary;
  course and look-ahead course errors.
- **Previous action.**
- **Target branch:** the single tracked vessel, with a presence bit.
- **Context branch:** encounter class, engagement state and the latched encounter.
- **Dropped from Paper 2:** front clearance, the side-clearance difference and the local target
  cross-track error, so that the passing side is learned rather than supplied.

Source: Paper 3 Section 2.3; `planning/FORMULATION_EQUATIONS.md`.

**Perception pipeline.** Five stages:
1. map gate;
2. pooling, which forms the lidar branch;
3. adaptive-breakpoint segmentation;
4. nearest-neighbour association with a 0.70 m gate (confirm after 2 hits, drop after more
   than 3 misses), then a constant-velocity Kalman filter;
5. a static or moving verdict from free-space evidence over 2 s.

A speed test was rejected for the last stage: it promoted panels to vessels in 28-33 % of
frames during development. Source: Paper 3 Section 2.3.1.

**Reward.**
- **Terminal and existence terms:** +100 at the goal, −300 at a collision, 0 at a timeout;
  existence −1 per step.
- **Bounded dense terms:** every dense term is bounded before weighting, so each weight is the
  term's largest per-step contribution.
- **Hierarchy:** collision ≫ boundary ≫ (ship domain, obstacle clearance) ≫ COLREGs ≫ path
  following ≫ smoothness. A rule-consistent collision always scores worse than a near miss that
  departs from the rules.
- **The Papers 1-2 failure:** a scale hidden inside the obstacle term made it about 49 times
  weaker than the path term at contact distance. The final formulation carries neither that
  reward nor that observation over.

Source: Paper 3 Section 2.3.3 and Appendix C.

**COLREGs scope, stated as the reward encodes it.**
- **Rule 8:** 8(b), 8(d) and 8(e), with 8(a) as timing pressure only.
- **Rule 9:** narrow channels.
- **Rule 13:** overtaking.
- **Rule 14:** port-to-port passing, with an alteration only when the target is not already
  clear.
- **Rules 15 and 16:** under the narrow-channel convention, the own ship gives way to a
  crossing target from either side by passing astern, altering to port for a target from port.
- **Rule 17(a)(i):** with the hold penalty released in extremis.

Source: `planning/PAPER3_CLAIMS_AND_TABLES.md`, decision record of 2026-10-08.

**Learners and training.**
- **Learners:** PPO, RecurrentPPO, SAC and TQC, with Stable-Baselines3 2.3.2.
- **Training:** 3.0 M steps; 10 parallel simulators; discount 0.951 per step; a seven-stage
  curriculum.
- **Checkpoint selection:** on the validation set every 200,000 steps, scored as goal rate − 2 ×
  collision rate.
- **Seeds:** three per learner, with no per-learner tuning.

Source: `MASTER_ARTIFACT_FACTS.md`, section 3.

**Data sets.**

| Set | Composition | Source |
|---|---|---|
| Validation set | 270 episodes: 120 decoupled, 150 coupled | `src/dev_set_v4.py` |
| Test set v4 | 1,000 episodes, digest a8c40beb234ea7f9: 664 decoupled, 336 coupled | `results/test_set/v4/` |
| Test set v4 by class | crossing 301, head-on 243, overtaking 211, being overtaken 140, null encounter 75, no target 30 | `results/test_set/v4/definition.csv` |

The test set merges the frozen suite and the reference-layout set, trimmed of near-duplicates.
Episodes that no manoeuvre can solve once the target is first tracked are replaced, using the
oracle feasibility check in `src/oracle_feasibility.py`.

**Scenario families.**
- **Decoupled:** static obstacles are kept clear of the predicted closest point of approach, so
  obstacle avoidance and collision avoidance happen one after the other.
- **Coupled:** three obstacles in a structured arrangement (a gate, a slalom, or a lateral
  obstacle with an on-path obstacle), with the encounter generated around them. This covers the
  generated FS- family and the reference layouts L1-L3.

**Classical comparators.**
- LOS-PID with DWA, written "LOS-DWA".
- The encounter-specific velocity obstacle, after Thyri and Breivik (2022). It shares the
  formulation's encounter assessment and narrow-channel convention, and replaced COLREGs-VO on
  2026-10-20.

## 4. Simulation results

All rows are with the safety layer off, at each run's best validation-set checkpoint. Source:
`PAPER3_CLAIMS_AND_TABLES.md`; `planning/MASTER_ARTIFACT_RESULTS.md`.

| Method | Test set v4 success | Decoupled | Coupled |
|---|---|---|---|
| SAC seed 0, the deployed policy | 0.865 (Wilson 95 %: 0.84-0.88) | 0.931 | 0.735 |
| SAC, three-seed mean | 0.846 (seeds 0.827-0.865) | | |
| TQC, three-seed mean | 0.830 (0.815-0.838) | | |
| PPO, three-seed mean | 0.750 (0.747-0.753) | | |
| RecurrentPPO, three-seed mean | 0.717 (0.638-0.757) | | |
| LOS-DWA | 0.770 | 0.870 | 0.571 |
| Encounter-specific VO | 0.675 | 0.774 | 0.479 |

| Claim | Result |
|---|---|
| C-1 | Exact McNemar test against LOS-DWA: 176 episodes completed by the policy only, 81 by the comparator only, p = 3.0 × 10⁻⁹. Against the VO: 249 against 59, p = 6.3 × 10⁻²⁹ |
| C-8 | Deployed policy, decoupled, over the classes both strata hold: basin 0.906 (254 episodes), channel 0.938 (240) |
| C-10 | Six successful episodes shown as snapshots, each passing the track-based rule conditions; stated as a selection. When being overtaken, the own ship gives a little room near the CPA |
| C-11 | Off-policy learners lead at every seed (lowest off-policy seed 0.815, highest on-policy 0.757). Target collisions 0.063 and 0.067 against 0.154 and 0.184; ship-domain intrusions 0.112 and 0.125 against 0.239 and 0.251; obstacle collisions 0.065-0.082 for all four |
| C-12 | SAC deployed by the registered rule, as the better three-seed mean of SAC and TQC, with seed 0 chosen by validation score (+0.411). No SAC-over-TQC ranking is claimed |
| C-13 | Deployed policy: reactive targets, decoupled, 0.941 → 0.922. One speed change, coupled, 0.808 → 0.620. Noncompliant head-on: 60 of 66 completed |
| C-14 | No reward term exceeds its weight. The COLREGs integral in collision episodes is −24.67, against −300 for the collision |
| Static-only | Deployed policy: no-target episodes 28 of 30, null-encounter episodes 0.95 (n = 75) (`results/baseline_v3/seed0_summary.md`) |
| Final models | The final 3.0 M models change single runs by at most 0.039 and leave the learner order unchanged |

## 5. Runtime safety layer (C-7)

**Configuration.** It is called "E1" in the bridge and the records. It combines a predictive
filter (`src/safety_v16.py`, on `src/safety_v2.py`) with a permanent emergency stop
(`src/safety_v21.py`, `SafetyFilterV21EC(estop_window_s=2.5, use_static_points=False,
braking_fallback=False)`).

**Predictive filter.**
- It forward-simulates for 8 s with the identified model, holding targets at constant velocity
  from their tracks.
- It checks the policy's command and a bank of alternatives, each followed by recovery
  manoeuvres.
- A plan passes when the hull keeps a 0.15 m margin, plus small extra gaps. The policy's
  command is issued if it passes; otherwise the closest passing command is issued.

**Emergency stop.** It runs only when the filter has no checked plan. If every first-command
and continuation sequence leads to contact or a boundary crossing within 2.5 s, under both
reverse-thrust models, the vessel commits to full astern with the rudder centred.

**Results.**

| | Test set | Validation set |
|---|---|---|
| Own-vessel collisions, layer off → on | 135 → 70 (target 53 → 8, obstacle 70 → 45, boundary 12 → 17) | 53 → 24 |
| Goals | 865 → 868 | |
| Stops | 52 | |
| Target strikes on the stopped own ship, reported apart | 9 | |
| Decisions changed by the layer | 8.9 % | 13.5 % (4.6 % outside the 23 episodes ending in a stop) |
| Episodes changed by the layer | 36.1 % | |

The registered intervention-rate prediction of at most 5 % was not met.

**Rules for reporting it.**
- The layer is off in training and in every validation-set evaluation used for selection.
- Its interventions are never credited to the policy.
- Its purpose is fewer own-vessel collisions; a higher success rate is not its purpose.
- **Stated limits:** the test set was used during its development; it gives no guarantee; the
  strike count depends on the braking convention.

Sources: `MASTER_ARTIFACT_FACTS.md`, section 5; `PAPER3_CLAIMS_AND_TABLES.md`, C-7;
`planning_records/SAFETY_LAYER_VERSIONS.md`.

## 6. Deployment bridge

Source: `field_deployment/README_P3_BRIDGE.md`. It is current; `MASTER_ARTIFACT_FACTS.md`
section 7 is out of date for the bridge.

**Design.**
- **Policy:** the deployed policy file is refused unless its SHA-256 (993db156…c2bdc8) and
  configuration match.
- **Decision rate:** one decision per LiDAR frame, at 2 Hz.
- **Observation adapter:** the simulator's environment class, with its sensor models replaced by
  the measured scan, pose and ego motion. The same perception code and observation builder run
  on real data, with no physics and no simulated targets.
- **Ego motion:** estimated from pose differences over the decision interval, 0.5 s, at either
  telemetry rate (since 2026-11-20).
- **Safety layer:** shadow mode by default, in a separate worker off the command path. Active
  mode is optional. The operator stop is always armed.
- **Link:** UDP between the shore laptop and the vessel-control computer, with the July 2026
  ports and command line kept.

**Verification on the bench.**

| Check | Result |
|---|---|
| Parity on 16 test-set episodes, in shadow and active modes | 32 of 32 runs, 1,497 decisions, with observation, action, command and every safety-layer decision exactly equal to the simulator's. Recorded safety-layer traces 16 of 16. Repeated on a second computer with the same result |
| July 2026 logs | 19 logs, 1,085 decisions, replayed in process. Three logs over loopback UDP, equal to the in-process run |
| February 2026 logs | Four logs, 418, 162, 156 and 443 decisions |
| Command path, LiDAR line to command | p99 about 0.09 s, against a 0.5 s limit (T-14 met) |
| Safety layer in the loop | p99 0.89 s on test-set streams, so active mode does not run at 2 Hz on the bench computers. Shadow mode is unaffected |
| Telemetry rate | July logs 2 Hz and February logs 10 Hz, set by the vessel's onboard program, not by the receiver |

**Open before the trials:**
- phantom moving tracks from basin-wall returns (T-15), resolved by map registration (P-7);
- telemetry resolution;
- the pose reference point;
- the map frame.

## 7. Field programme (Paper 3), with results pending

**Sessions.**

| Session | Content |
|---|---|
| S1 | Calibration, 290 min |
| S2 | Calibration and dress rehearsal, 210 min |
| T1 | Trials, 24 runs |
| T2 | Trials, 11 runs plus up to 7 |

Source: `planning/FIELD_CALIBRATION_PLAN.md`; `planning/FIELD_TRIALS_PLAN.md`.

**Scenario families.**
- **A:** no target.
- **B:** open-basin encounters.
- **C:** coupled encounters on the reference layouts.

**Field test suite.** 15 cases: easy, medium and hard, each with the five encounter types
(head-on, crossing from starboard, crossing from port, overtaking, being overtaken). Easy cases
are in the open basin, medium on L1, hard on L2 and L3. They are named `easy_1` to `hard_5`, are
defined in `src/field_test_cases.py`, and load in simulation and in the bridge.

**Pre-registered field claims C-15 to C-20** (registered 2026-11-07):
- C-15: the policy runs unchanged. The bench part is supported.
- C-16: encounter-consistent manoeuvres in family B.
- C-17: family C against its simulated twins.
- C-18: the gap at three levels, with attribution.
- C-19: identified dynamics inside the training randomisation.
- C-20: the shadow safety layer on the water.

**Reporting.**
- Field results are reported as n/N per scenario, not as a rate comparable with the test set.
- Operator stops are counted apart.
- Invalid runs are kept and reported as invalid.

## 8. Development record (thesis-only material)

| Item | Record |
|---|---|
| Test set versions | v1 to v4. v2 replaced 15 impossible L1 being-overtaken starts. v4 replaced the episodes no controller can solve once the target is first tracked (oracle feasibility check). Earlier figures on v1: SAC 0.857, LOS-DWA 0.804, COLREGs-VO 0.703 |
| Targeted training, baseline-v4.3 | Blocked-turn flags from the start, a denser conflict curriculum and crossing emphasis. Trained in full and not adopted: 0.833 against 0.865 (`planning_records/BASELINE_V4_PLAN.md`) |
| COLREGs-VO | Replaced by the encounter-specific VO on 2026-10-20 |
| RecurrentPPO seed 1 | Collapsed in mid-training, between 1.0 and 2.4 M steps: 258 of 273 updates stopped at the first epoch on the KL limit. Reported, not replaced; seeds 0 and 2 show none of it |
| TQC seed 2 | Resumed from its 750k checkpoint after the disk filled (a documented interruption) |
| Withdrawn studies | Width sweep and perception degradation (2026-10-14); encounter-feature ablation (2026-10-24); domain-randomisation field ablation (2026-11-05) |
| Development learner | PPO was used to iterate the formulation, which references no algorithm |

## 9. Wording rules

| Use | Never |
|---|---|
| COLREGs-aware | COLREGs-compliant, compliance rate |
| coupled / decoupled scenarios | field-style, frozen-like |
| validation set | development set, dev set |
| reference layouts L1-L3 | field deployment test set |
| deployed policy (SAC seed 0); other learners and classical methods as baselines | a learner-comparison study |
| onboard perception (pooled range returns, target tracks, a map boundary) | LiDAR in the title or a contribution name |
| the safety layer reduces own-vessel collisions | the safety layer raises success or guarantees safety |

**Also:**
- Strata are described by geometry only, never as "challenging for classical methods".
- Every learner result carries its seed spread. Differences below about 0.05 are not
  interpreted.
- The learner family was chosen on the test set, and the text says so. The field trials are the
  independent evaluation of the deployed policy.

## 10. Statements that must not appear

These come from earlier documents, including the Year 3 report:
- "success rises to about 91 % with the safety layer" (a development figure on test set v2);
- "COLREGs-VO, 69 %" as the comparator;
- "the final simulation step was targeted training";
- "compliance by encounter type";
- a reward ablation;
- the Papers 1-2 reward or observation described as the final one;
- six research questions or eight contributions (the agreed plan has four and five).
