**Paper 3 — Draft Skeleton**

*Repositioned to two-vessel encounters with static obstacles · Revision 2*

**Working title:** Sensor-Realistic COLREGs-Compliant Path Following and Collision Avoidance for Autonomous Surface Vessels in Narrow Waterways using Deep Reinforcement Learning

**Alternative title:** COLREGs-Compliant Collision Avoidance in Confined Waters from Onboard LiDAR: A Deep Reinforcement Learning Approach with Field Validation

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
| S1     | Two-vessel encounters only. One dynamic target plus up to three static obstacles.                                            |
| S2     | COLREGs scope: Rules 13–16, with Rule 9 governing precedence and Rule 8 governing action quality.                            |
| S3     | Own ship gives way in all crossing encounters, justified by Rule 9(b) rather than Rule 18.                                   |
| S4     | Five encounter classes: none, head-on, crossing, overtaking, being overtaken.                                                |
| S5     | Rule 17(a)(i) passive course-keeping retained for the being-overtaken class. Active release under 17(a)(ii) is out of scope. |
| S6     | Architecture keeps N_max as a configuration parameter so multi-vessel extension costs a retrain, not a redesign.             |

Abstract

*Draft skeleton — write last, but fix the shape now.*

Autonomous surface vessels operating in restricted waterways must avoid traffic while remaining compliant with the international collision regulations, under geometric constraints that open-water methods do not face. Existing learning-based approaches almost universally assume that the state of the target vessel is available, whether from AIS or from a simulation oracle. This paper presents a deep reinforcement learning agent that performs path following, static obstacle avoidance, and COLREGs-compliant manoeuvring against a dynamic target in a narrow channel, using target state estimated entirely from an onboard two-dimensional LiDAR. The observation combines pooled range sensing for static obstacles, a map-derived channel boundary, and a tracked-target branch carrying explicit encounter-class and collision-risk features. Compliance with Rules 13 to 16 is enforced through class-conditional reward terms gated on a deterministic encounter classifier shared between observation and reward, with Rule 9 governing precedence where channel width makes the open-water manoeuvre inadmissible. The agent is trained with Soft Actor-Critic under a staged curriculum and evaluated on a frozen suite of \[N\] deterministic named cases and \[N\] stratified randomised cases. \[RESULT\] The trained policy achieves \[X\]% success and \[Y\]% COLREGs violation rate against \[baseline\]. A channel-width sweep establishes where classical methods become inadmissible, a perception-degradation study characterises tolerance to tracking error, and sim-to-field transfer is validated on a 1.73 m model vessel across \[N\] basin trials.

**Keywords:** autonomous surface vessel; COLREGs; deep reinforcement learning; collision avoidance; narrow channel; LiDAR perception; sim-to-real

1\. Introduction

1.1 Motivation

Restricted waterways — port approaches, canals, rivers, and the individual survey sectors of a larger area-coverage mission — concentrate traffic and constrain manoeuvring at the same time. Autonomy in these environments must satisfy the collision regulations under geometric limits that the open-water literature does not model.

A concrete instance motivates the geometry used throughout this paper. In an area-coverage survey, a global planner decomposes the survey region into parallel legs; the vessel must remain within the lateral bounds of the current leg to preserve sensor coverage, and cannot shortcut between legs. Each leg therefore presents the vessel with a laterally bounded corridor along which it must hold a reference path while resolving any encounter that arises. Port approach channels and canal transits impose the same structure for different reasons. This paper addresses the corridor problem in general rather than any single application of it.

*\[Optional figure: laterally bounded corridor with reference path, one dynamic target, and static obstacles. Keep the survey framing to this paragraph — expanding it invites coverage-planning questions that are out of scope.\]*

1.2 Gap

Three gaps, to be argued in Section 2:

- **Assumed target state.** Learning-based COLREGs methods overwhelmingly take target position, course and speed as given. Estimating them from an onboard sensor introduces detection, association, occlusion and velocity-estimation error that the policy must tolerate, and this degradation is rarely characterised.

- **Open-water bias.** Rule 9 and the confined-water case remain under-addressed. \[VERIFY — Burmeister & Constapel (2021): of 48 papers surveyed, four mention Rule 9 and two address it.\]

- **Limited physical validation.** Most learned COLREGs policies are evaluated in simulation only. Where field results exist they are typically demonstrations rather than systematic transfer studies. \[VERIFY — this claim has weakened since 2024; differentiate rather than assert exclusivity.\]

1.3 Contributions

| **\#** | **Contribution**                                                                                                                                                                                                                     |
|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C1     | A COLREGs-compliant path-following and collision-avoidance policy whose target-ship state is derived entirely from onboard 2D LiDAR, with the resulting perception noise characterised from field logs and injected during training. |
| C2     | An explicit Rule 9 precedence framework specifying, per encounter class and channel width, which rule governs and what action is admissible when the open-water manoeuvre does not fit.                                              |
| C3     | A channel-width sweep identifying the geometric regime in which classical reactive methods become inadmissible and quantifying learned-policy behaviour across the transition.                                                       |
| C4     | A perception-degradation study quantifying how COLREGs compliance responds to tracking error, detection dropout and occlusion.                                                                                                       |
| C5     | A complete sim-to-real pipeline — system identification, domain randomisation over identified uncertainty, and field validation — with the identification dataset, scenario generator and frozen evaluation suite released.          |

1.4 Research questions

- **RQ1 —** Can a single end-to-end policy perform path following, static obstacle avoidance and COLREGs-compliant manoeuvring in a laterally bounded channel using only onboard range sensing?

- **RQ2 —** Does supplying the encounter class explicitly as an observation feature improve compliance over reward shaping alone, and by how much?

- **RQ3 —** How does compliance degrade as perception quality degrades, and at what point does the policy fail unsafely rather than conservatively?

- **RQ4 —** Does domain randomisation over identified model uncertainty close more of the sim-to-field gap than improved nominal identification alone?

1.5 Paper structure

\[Standard.\]

2\. Related work

*Four threads. Keep each to a paragraph; the contribution is the intersection, not any single thread.*

2.1 DRL for ASV path following and collision avoidance

Meyer et al. (2020, Taming) established the path-relative observation with rangefinder sensing that this work descends from. Heiberg et al. (2022) introduced risk-index-based COLREGs rewards. Woo & Kim (2020) provide the standard DRL-USV reference. Our own prior work \[cite Paper 2\] added feasibility-inspired LiDAR sector pooling, a staged curriculum, and field validation for the static case.

2.2 COLREGs in learned policies

Waltz & Okhrin (2023) for encounter classification, CPA and collision-risk formulation; Zhao & Roh (2019) and Chun et al. (2021) for multi-ship handling; Sawada et al. (2021) for the Imazu benchmark tradition. Position: these assume target state is directly available and operate in open water.

2.3 Restricted and confined waters

Hansen et al. (2022) on Rule 9 manoeuvrability assessment; de Vries et al. (2022) on urban canals; Waltz, Paulig & Okhrin (2025) on inland waterways — the closest prior work. Villa, Aaltonen & Koskinen on LiDAR-based path following in harbour conditions. Position: none combine the confined-water constraint with COLREGs compliance driven by onboard-sensor target state and validated physically.

2.4 Sensor-based detection, tracking and field-validated avoidance

Han et al. (2020) demonstrate LiDAR and radar track fusion feeding COLREGs-compliant manoeuvres, verified in field experiments across a range of environmental conditions. Kim et al. (2022) embed 2D LiDAR obstacle detection in a physical catamaran-type ASV, verified in both simulation and experiment — the closest platform analogue to this work. A 2025 memory-based SAC approach validates COLREGs-compliant avoidance with practical experiments as well as simulation. Position: these establish that field-validated sensor-driven avoidance is achievable; this work adds the confined-water constraint and a systematic characterisation of how compliance degrades with perception quality.

2.5 Classical COLREGs collision avoidance

Kuwata et al. (2014) for COLREGs-aware velocity obstacles; Thyri & Breivik (2022) for encounter-specific velocity obstacles, confined-water domains and COLREGs-constrained MPC; Gonzalez-Garcia et al. (2022) for NMPC path following with LiDAR-based avoidance and physical experiments. These provide the comparators in Section 5.5, and the encounter-specific VO additionally serves as the reactive target model.

3\. Problem formulation

3.1 Vessel and workspace

Model-scale Bluefin: 64.55 kg, LOA 1.73 m, LBP 1.57 m, breadth 0.50 m, draft 0.19 m, Iz 10.45 kg·m². Three-degree-of-freedom Fossen model with explicit actuator dynamics (Section 4.6). Control period 0.1 s (10 Hz), episode cap \[700\] steps.

The workspace is a laterally bounded corridor with variable width, bends, and reference paths that are deliberately not centred, so that the boundary observation branch carries information not already present in the cross-track error. Simulation matches the physical basin, with a maximum corridor width of 10 m (20 ship breadths), so that every simulated width is physically reproducible. The unconfined reference case is supplied instead by the open-water variant of the external benchmark (Section 5.2), rather than by a wider simulated channel.

The width sweep in Section 6.4 spans 10 m down to 3.5 m. With the ship domain of Section 4.3, the minimum width admitting a compliant port-to-port head-on is approximately 3.66 m (7.3 breadths): 2.36 m centre-to-centre lateral separation for non-overlapping domains, plus roughly 0.65 m wall clearance each side. The sweep therefore brackets the transition between 4 m and 3.5 m. \[Re-verify once the ship domain is finalised from turning-circle data — the threshold moves with the domain.\]

**A note on what the sensor observes.** The LiDAR does not register the basin edge at all; it registers the facility walls one to two metres beyond it. This is not a peculiarity to be worked around. In a restricted waterway the navigable limit is generally not a physical structure either, being defined by a charted depth contour, a buoyed line, or a regulatory boundary, none of which a range sensor can detect. Supplying the boundary from the chart while reserving range sensing for physical obstacles is therefore the architecturally correct division for the application, and the basin reproduces that situation exactly.

3.2 Scaling

State the model-to-full-scale relationship explicitly so that spawn TCPA, CPA thresholds and ship-domain dimensions can be read at full scale. \[TBC — Froude scaling statement.\] A reviewer will ask whether a 15 s TCPA on a 1.73 m model corresponds to anything meaningful at full scale; the answer belongs in the paper.

3.3 COLREGs scope

| **Rule**   | **Content**                                            | **Treatment**                                                                                                                                                                    |
|------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 8          | Action to avoid collision                              | Implemented — 8(a) ample time, 8(b) readily apparent, 8(e) slacken speed. Drives the timing and magnitude metrics.                                                               |
| 9          | Narrow channels                                        | Implemented — 9(a) keep starboard, 9(b) do not impede, 9(e) overtaking. Governs precedence over Rules 13–16.                                                                     |
| 13         | Overtaking                                             | Implemented — two classes: own ship overtaking, own ship overtaken.                                                                                                              |
| 14         | Head-on                                                | Implemented — alter to starboard, subject to available channel width.                                                                                                            |
| 15         | Crossing                                               | Implemented — own ship gives way in all crossing encounters under Rule 9(b).                                                                                                     |
| 16         | Give-way action                                        | Implemented — early and substantial, per Rule 8.                                                                                                                                 |
| 17         | Stand-on action                                        | Partially — 17(a)(i) passive course-keeping is retained for the being-overtaken class. Active release under 17(a)(ii) is out of scope and identified as future work.             |
| 2, 5, 6, 7 | Responsibility, lookout, safe speed, risk of collision | Acknowledged; operationalised implicitly — the collision risk index for Rule 7, the perception pipeline for Rule 5.                                                              |
| 18         | Responsibilities between vessels                       | Out of scope — own ship and target are similarly sized, so the asymmetry the rule requires does not exist. The always-give-way simplification is instead justified by Rule 9(b). |
| 19–31      | Restricted visibility, lights, shapes, sound signals   | Out of scope — no corresponding sensing or actuation on the platform.                                                                                                            |

*State this table in the paper. Reviewers who know COLREGs read the omissions as closely as the inclusions, and naming the Rule 18 exclusion pre-empts the obvious question.*

3.4 Encounter classes

| **Class**       | **Governing rule** | **Own-ship obligation**                                      |
|-----------------|--------------------|--------------------------------------------------------------|
| None            | —                  | Follow path                                                  |
| Head-on         | 14                 | Alter to starboard, subject to channel width                 |
| Crossing        | 15, 16, 9(b)       | Give way regardless of which side the target approaches from |
| Overtaking      | 13, 16, 9(e)       | Keep clear of the vessel being overtaken                     |
| Being overtaken | 13, 17(a)(i)       | Hold course and speed                                        |

3.5 Rule precedence

The organising principle: Rule 9 constrains the available space, and Rule 8 subsection (e) supplies the action when that space is unavailable. Width thresholds are an output of the sweep in Section 6.4, not an input to it.

| **Encounter**   | **Wide channel**                             | **Narrow channel**                                                   | **Governing**        | **Fallback**                       |
|-----------------|----------------------------------------------|----------------------------------------------------------------------|----------------------|------------------------------------|
| Head-on         | Alter to starboard                           | Hold starboard side; 9(a) compliance satisfies 14 without alteration | 14 + 9(a)            | Slacken speed, 8(e)                |
| Crossing        | Give way — alter to starboard or pass astern | Give way if room exists                                              | 15, 16 + 9(b)        | Slacken speed or stop, 8(e)        |
| Overtaking      | Pass either side                             | Pass to port of target                                               | 13 + 9(a), 9(e)\*    | Hold astern at reduced speed, 8(e) |
| Being overtaken | Hold course and speed                        | Hold course and speed, keep starboard                                | 13 + 17(a)(i) + 9(a) | —                                  |

*\* Geometric constraint only. Rule 9(e) requires sound signals and the overtaken vessel's agreement, which the platform cannot provide. State as an explicit scope limitation.*

Head-on rationale: Rule 9(a) already requires both vessels to keep to the starboard side of the fairway, so if both comply a port-to-port pass occurs without either altering course, and Rule 14 is satisfied by channel-keeping rather than by an evasive manoeuvre. An alteration is required only where the target is not where 9(a) says it should be. Overtaking side follows from the same rule: if the overtaken vessel keeps starboard, the room lies on its port side.

The threshold at which a channel becomes "narrow" for a given encounter is itself a result, produced by the width sweep in Section 6.4. Define it geometrically — in ship breadths and in terms of the lateral excursion the compliant manoeuvre requires — not by reference to where any method fails.

3.6 MDP formulation

State, action a = \[rudder, throttle\] ∈ \[−1,1\]², transition, reward, discount. Propulsion authority is widened relative to prior work because Rule 8 subsection (e) makes slackening speed a lawful avoidance action, and in a confined channel it is frequently the only admissible one. A policy that cannot slow down cannot comply.

4\. Methodology

4.1 Perception pipeline

Raw LiDAR: 360°, 720 beams, 0.5° angular resolution, 10 Hz. Returns falling outside the known channel polygon are gated geometrically, because the sensor is mounted above the basin wall and would otherwise register objects beyond it as obstacles or, worse, as phantom targets with plausible velocities. Remaining returns are clustered, ego-motion compensated, associated to tracks, and velocity-estimated by a constant-velocity Kalman filter. Static clusters feed the pooled sector channel; the dynamic track feeds the target branch.

*\[Figure: perception pipeline block diagram.\]*

4.2 Observation space

| **Branch** | **Contents**                                                                          | **Dim** |
|------------|---------------------------------------------------------------------------------------|---------|
| lidar      | Pooled sector closeness, obstacles only, forward-biased to ±135°, non-uniform sectors | 27      |
| boundary   | Virtual raycast against the channel polygon, 7 rays, pose noise injected              | 7       |
| ego        | Surge u, sway v, yaw rate r                                                           | 3       |
| path       | Cross-track error, course error, look-ahead course error                              | 3       |
| target     | Tracked target features plus presence bit                                             | 16      |
|            | Total                                                                                 | ≈56     |

Target features: distance to ship domain; sine and cosine of relative bearing; sine and cosine of heading-intersection angle; target speed; relative speed; DCPA; TCPA; collision risk index; encounter class as a five-way one-hot; presence bit.

**Design rationale to state explicitly.** Pooled range sensing is velocity-blind: a static wall and a closing vessel at the same range produce identical sector closeness. Target kinematics must therefore enter through either recurrence or explicit tracking. Explicit tracking is chosen because it additionally gives information parity with the velocity-obstacle comparators, which require the same quantities.

N_max is a configuration parameter. The target branch is built as an indexed slot so that extension to multiple targets requires retraining rather than redesign.

4.3 Encounter classification and collision risk

One module, two consumers: the same function feeds the observation feature and the reward gate, with hysteresis applied inside it. If the two diverged even at sector boundaries, the agent would be penalised for a role it was never shown.

Classification thresholds adapted from Waltz & Okhrin (2023), with the head-on band widened from ±5° \[TBC — value and justification\] and a fifth class added for being overtaken. Collision risk computed as the maximum of a CPA-based and a Euclidean-distance-based term, with distance measured to the ship domain rather than the hull. The Euclidean term is not optional here: two vessels on near-parallel courses in a channel have a CPA far in the past or future, so a CPA-only risk reads as low until either vessel turns slightly, at which point the situation becomes urgent instantly. Near-parallel geometry is the normal case in a corridor.

**All constants must be re-derived in ship lengths.** The source values are tuned for a 320 m KVLCC2 with decay scaled to two nautical miles, and the asymmetric three-ship-length fore-aft domain does not fit a channel of the width considered here.

A compressed asymmetric domain is adopted, provisionally 2.0 Lpp ahead, 1.0 Lpp astern and 0.75 Lpp abeam — 3.14 m, 1.57 m and 1.18 m respectively — giving a lateral footprint of 2.36 m, approximately 24 per cent of a 10 m corridor.

**The principle matters more than the values.** These are not defended as a scaled copy of another vessel's domain. The final values are derived from measured manoeuvring performance: advance and tactical diameter from the turning-circle tests and stopping distance from the stop test, both in Section 7.2. The domain is then sized to this vessel's demonstrated ability to avoid, which is the argument made for confined-water domains in the classical literature. The table above is a provisional input; the final values are an output of the identification campaign.

4.4 Reward function

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

**Scale audit.** Per-term episode-integrated contributions reported in Table R7. This is mandatory: the obstacle-avoidance term in the prior paper was approximately 49 times weaker than the path-following term at contact distance, which the weighting notation concealed entirely until the terms were integrated and compared empirically.

4.5 Policy architecture

Soft Actor-Critic with a multi-input policy over the five observation branches. Trained from scratch — no warm start from the prior policy, whose observation and reward semantics have both changed on every channel. The prior policy is retained as a frozen zero-shot comparator, which is stronger for being genuinely independent rather than an ancestor of the new agent.

\[TBC — whether any recurrence is added. The explicit tracker carries some memory already; occlusion of the target behind a static obstacle is the case that would justify more. Quantify the occlusion frequency in the scenario distribution before adding it.\]

4.6 Vessel model and system identification

Three-DOF Fossen model with linear and quadratic damping, plus an explicit actuator model: servo rate limit, first-order lag, transport delay, and thrust map. The observed field behaviour — wider turns, larger oscillation, roughly double the RMS cross-track error — indicates actuator lag together with underestimated yaw damping and rudder effectiveness, rather than missing degrees of freedom.

Identification manoeuvres: straight-line acceleration and deceleration at several throttle settings; turning circles at multiple rudder angles and speeds; 10/10 and 20/20 zig-zags; rudder step response; stop test. Zig-zag overshoot angles are the single most informative measurement for the observed symptom. Fitting by prediction-error minimisation with one zig-zag and one turning circle held out.

Scan-to-scan odometry is not adequate ground truth for identifying yaw dynamics, since error integrates without bound. Identification therefore relies on scan-to-map registration against surveyed facility geometry, which is absolute rather than incremental and so bounded in error, combined with an inertial measurement unit added to the vessel for this purpose. Raw angular rate and acceleration are logged at 100 Hz or better; fused orientation outputs are avoided because their internal filtering would corrupt the lag and damping parameters under estimation. The two sources are complementary: registration supplies drift-free absolute pose at 10 Hz while the inertial unit supplies derivatives at high rate and, through the accelerometer, the along-track information that two parallel walls cannot constrain.

Time synchronisation between the two streams is the critical detail: a constant offset appears in the fit as actuator lag and would be absorbed into the model as a physical parameter. Both streams are logged on a common clock where possible, and each run begins with a stationary period for bias estimation and a sharp yaw impulse for alignment.

Two long parallel walls lie within sensor range from mid-basin; the end walls do not. Heading and lateral position are consequently well constrained, while along-track position is constrained only where fixed features — recessed doorways, structural columns, wall-mounted hardware — fall within range. Zig-zag overshoot angles are read directly from the heading trace and are therefore the best-constrained measurements available; turning circles are partially constrained; straight-line tests require either timing between surveyed features or positioning so that an end wall remains in range.

The model is fitted to measured yaw rate and to heading, and the two fits cross-checked. Disagreement between them indicates a synchronisation or mounting error rather than a model deficiency, which makes the cross-check a useful diagnostic rather than a redundancy.

Validation without external truth is achieved by static tests at surveyed positions, giving absolute accuracy, and closed-loop runs returning to a common physical point, giving drift. \[Risk to check first: one full-length facility wall is matte black, and near-infrared reflectivity of carbon-based finishes can be very low. Plot return density against bearing in the retained logs before committing to continuous-wall registration; sparse registration against surveyed landmarks mounted on that wall is the fallback.\]

Domain randomisation over identified parameters plus or minus their confidence intervals, together with the perception noise sources characterised in Section 4.1. A better nominal model reduces bias but does not create robustness; the defensible claim is identification *and* randomisation within identification uncertainty.

4.7 Training

Constant-velocity targets during training; reactive and non-compliant behaviours are reserved for evaluation, because training against a reactive opponent makes the environment non-stationary and destroys attribution.

| **Stage** | **Content**                                               |
|-----------|-----------------------------------------------------------|
| 1         | Static obstacles only, straight constant-width corridor   |
| 2         | Static obstacles, variable width and bends                |
| 3         | Single dynamic target, generous spawn TCPA, wide corridor |
| 4         | Single dynamic target, reduced TCPA, narrowed corridor    |
| 5         | Full difficulty range with static clutter                 |

Five seeds per reported configuration. Hyperparameters in Appendix B.

5\. Scenario generation and evaluation design

5.1 Generator

Scenarios are parameterised by encounter class rather than by initial condition: sample the class, then a heading-intersection angle from its valid interval, then a target speed, then a spawn TCPA, and solve backwards for the spawn position that produces that geometry. Random spawning frequently produces targets that pose no threat and wastes training samples. The null class — a target on a similar course, posing no conflict — is included deliberately, since it never arises from a purely class-conditioned spawner but does occur in practice.

The same generator produces training scenarios and the evaluation suite, with disjoint seeds.

5.2 Evaluation suite

Two tiers, frozen, versioned and hashed before the first training run.

- **Tier A —** \[30–40\] deterministic named cases: one per encounter class and channel-width condition, plus static-clutter variants. These carry the trajectory figures.

- **Tier B —** stratified randomised holdout. Encounter class (5) × target behaviour (3: constant velocity, compliant reactive, non-compliant) × channel width (3) × static clutter (0–3). Approximately \[800–900\] cases.

- **External benchmark —** the "Around the Clock" set of 24 single-ship encounters at equally spaced target headings. The reduction to two-vessel scope makes this an exact fit rather than an adaptation, and it sweeps every classification boundary systematically, including the astern sector. Adopting it is the principal defence against the criticism that the benchmark was constructed by the authors.

5.3 Difficulty definition

Geometric only: channel width in ship breadths, spawn TCPA, static clutter count. Never defined by baseline performance. One stratum is deliberately narrow enough that no method passes cleanly — a suite the proposed method succeeds on everywhere reads as constructed regardless of how it was built.

5.4 Metrics

**Task.** Success rate; collision rate reported separately for static obstacle, boundary and target vessel; RMS and maximum cross-track error; path length ratio; action smoothness.

**COLREGs.** Violation rate per encounter class; minimum CPA distribution reported as a CDF rather than a mean, since the tail is the safety claim; ship-domain intrusion rate and depth; time to first evasive action; magnitude of first evasive action; course-keeping stability while being overtaken; side-of-passing correctness.

**Perception.** Track acquisition range; classification latency and stability; velocity estimate error; occlusion duration.

\[TBC — presentation format for multi-axis results. A weighted scalar will be contested; prefer a per-axis table or a Pareto view.\]

5.5 Comparators

| **Family** | **Method**                                                                                         |
|------------|----------------------------------------------------------------------------------------------------|
| Classical  | LOS-PID with dynamic window approach                                                               |
| Classical  | COLREGs-aware velocity obstacles (Kuwata et al.)                                                   |
| Classical  | Encounter-specific velocity obstacles (Thyri & Breivik) — also serves as the reactive target model |
| Classical  | \[Optional\] NMPC with COLREGs constraints                                                         |
| Learned    | Prior-paper SAC, unmodified, zero-shot (frozen)                                                    |
| Learned    | PPO, RecurrentPPO, TQC retrained on the same environment                                           |
| Learned    | COLREGs-ablated policy (avoidance only)                                                            |

Classical baselines are run against reactive targets as well, or the comparison is not like-for-like. \[TBC — final list after the compute estimate.\]

5.6 Ablation matrix

|                          | **Encounter feature OFF** | **Encounter feature ON** |
|--------------------------|---------------------------|--------------------------|
| COLREGs reward terms OFF | Avoidance only            | Told, not rewarded       |
| COLREGs reward terms ON  | Learned from kinematics   | Full method              |

This answers the question a sceptical reviewer will actually ask: is compliance learned, or handed to the agent? Design the training campaign around it rather than retrofitting. Supplementary leave-one-out ablations on individual COLREGs terms and on the boundary observation branch.

6\. Numerical studies

*Tables pre-committed. Values pending. Writing this section before training begins is what prevents the experiment being designed after the fact.*

6.1 Overall performance

Table R1 — Tier B holdout. \[RESULT\]

| **Method**            | **Success** | **Static coll.** | **Boundary coll.** | **Target coll.** | **RMS CTE (m)** | **Path ratio** |
|-----------------------|-------------|------------------|--------------------|------------------|-----------------|----------------|
| Proposed (SAC, full)  |             |                  |                    |                  |                 |                |
| SAC, no COLREGs terms |             |                  |                    |                  |                 |                |
| Prior SAC (frozen)    |             |                  |                    |                  |                 |                |
| Encounter-specific VO |             |                  |                    |                  |                 |                |
| COLREGs-VO            |             |                  |                    |                  |                 |                |
| LOS-PID + DWA         |             |                  |                    |                  |                 |                |

6.2 Compliance by encounter class

Table R2 — violation rate per class. \[RESULT\]

| **Method** | **Head-on** | **Crossing** | **Overtaking** | **Being overtaken** |
|------------|-------------|--------------|----------------|---------------------|
|            |             |              |                |                     |
|            |             |              |                |                     |

6.3 Performance by target behaviour

Table R3 — constant velocity / compliant reactive / non-compliant. \[RESULT\]

6.4 Channel-width sweep

Table R4 and accompanying figure. \[RESULT\] Sweep corridor width from open-water-equivalent down to the point at which the Rule 14 starboard alteration no longer fits within the channel. Report, per width: success rate, compliance rate, minimum CPA, and the governing rule from the precedence table. Identify the width at which each classical method becomes inadmissible.

| **Corridor width** | **Ship breadths** | **Compliant head-on admissible?**  |
|--------------------|-------------------|------------------------------------|
| 10 m               | 20 B              | Yes, comfortably                   |
| 8 m                | 16 B              | Yes                                |
| 6 m                | 12 B              | Yes                                |
| 5 m                | 10 B              | Marginal                           |
| 4 m                | 8 B               | Tight                              |
| 3.5 m              | 7 B               | No — below the geometric threshold |

*This is the primary evidence for contributions C2 and C3, and it replaces the multi-vessel results in providing depth.*

6.5 Perception degradation study

Table R5 and accompanying figure. \[RESULT\] Degrade the tracker along four axes independently and jointly: pose drift magnitude, detection dropout rate, occlusion duration, velocity-estimate noise. Report compliance and safety as a function of each. Identify the point at which failures become unsafe rather than conservative.

*Primary evidence for contribution C1 and C4. Requires no additional basin time.*

6.6 Ablation

Table R6 — the 2 × 2 plus leave-one-out. \[RESULT\]

6.7 Reward scale audit

Table R7 — per-term episode-integrated contribution under a random policy and under the trained policy, with the empirical ordering checked against the intended hierarchy. \[RESULT\]

6.8 Named cases and external benchmark

Table R8 — Tier A cases and the 24-case Around the Clock set. \[RESULT\]

6.9 Figures

- Learning curves with seed spread

- Trajectory overlays for selected Tier A cases, one per encounter class

- Minimum-CPA cumulative distribution

- Success and compliance versus channel width (Section 6.4)

- Compliance versus perception degradation (Section 6.5)

7\. Field experiments

7.1 Platform and setup

Bluefin model vessel, RPLidar C1, UDP offboard control at 10 Hz. The basin is an indoor pool within a hall, with the facility walls standing one to two metres beyond the pool edge. Returns from beyond the pool boundary are removed by geometric gating against the known pool polygon rather than by a physical barrier: the facility walls carry the fixed features on which localisation depends, and occluding them would remove the only available registration reference. Gating is in any case necessary rather than merely preferable, since operators standing on the deck lie at scan height and would otherwise be tracked as dynamic targets. Localisation is run on the complete scan and the obstacle gate applied only afterwards, so that the walls serve as a registration asset while being excluded from the tracker.

Static obstacles are suspended panels, confirmed stable in the water. Apparent motion of static objects in the scan therefore arises almost entirely from ego-pose error, which affects all objects identically; the static-versus-dynamic threshold is consequently a property of localisation quality rather than of the obstacles, and is set from measured pose noise.

7.2 System identification

Manoeuvre set per Section 4.6. The identification dataset is archived and released. This directly discharges the reviewer concession in the prior paper, where the identification data was not retained in a form supporting independent reporting and the model could only be described as calibrated rather than identified.

7.3 Model validation

Replay of the prior field command sequences through the recalibrated model, overlaid against the recorded trajectories. The logs are already retained, so this requires no basin time and produces a validation figure at zero marginal cost.

7.4 Domain randomisation ablation

Table R9. \[RESULT\] Compare policies trained on the identified nominal model alone against policies trained with randomisation over identified uncertainty, both evaluated in the field. This is the evidence for RQ4. If the identified nominal model alone closes the gap, that is a reportable finding; if it does not, the gap is localisation and disturbance rather than dynamics, which is equally reportable.

7.5 Dynamic encounter trials

One target vessel. \[TBC — target platform, repeatability protocol, localisation of both vessels in a common frame, abort procedure, risk assessment.\] \[N\] repetitions per scenario. The prior work ran each scenario once and had to qualify its claims to feasibility rather than robustness; repeated trials are what change that.

7.6 Results

\[RESULT\]

8\. Discussion

8.1 Answers to the research questions

\[RESULT\]

8.2 Claim ledger

*Every claim in the abstract and conclusion maps to a table or figure. Complete this before writing either.*

| **\#** | **Claim** | **Evidence** | **Status** |
|--------|-----------|--------------|------------|
| 1      |           |              |            |
| 2      |           |              |            |
| 3      |           |              |            |

8.3 Why two-vessel encounters

*Draft text — adapt for the paper.*

Rules 13 through 16 are formulated pairwise: each defines the obligations of one vessel with respect to a single other vessel. Multi-ship handling is an extension not specified by the regulations themselves. In restricted waters this pairwise framing is also the physically realistic one, since channel geometry sufficiently confined for Rule 9 to be operative precludes multiple simultaneous close-quarters conflicts; encounters in such waters are typically sequential rather than concurrent. We therefore adopt the two-vessel encounter as the unit of analysis, which additionally allows every reported behaviour to be reproduced in physical trials rather than validated in simulation alone.

Place this in the problem formulation rather than the limitations section. It is a scope decision defended by the structure of the regulations and the geometry of the domain, not a constraint conceded after the fact.

8.4 Limitations

- Model scale; full-scale correspondence rests on the Froude scaling statement in Section 3.2

- Single dynamic target; sequential rather than concurrent multi-vessel encounters

- Active stand-on release under Rule 17(a)(ii) is out of scope

- Calm water with no generated waves or current

- Five seeds is a limited estimate of training variability

- The encounter classifier supplies the rule regime, so compliance is partly given rather than fully learned — quantified by the ablation in Section 6.6, not merely acknowledged

8.5 Future work

- Extension to concurrent multi-vessel encounters via the existing N_max parameter

- Active stand-on release under Rule 17(a)(ii) against non-compliant targets

- Disturbance rejection under wave and current loading

9\. Conclusion

*Write last. Must contain no claim absent from the ledger in Section 8.2.*

Appendices

|     | **Content**                                                                                                                        |
|-----|------------------------------------------------------------------------------------------------------------------------------------|
| A   | Vessel model: full numerical parameters, actuator dynamics, saturation limits, delays, identification procedure and fit statistics |
| B   | SAC hyperparameters and curriculum schedule                                                                                        |
| C   | Complete reward specification with coefficients and normalisation ranges                                                           |
| D   | Encounter classification thresholds and hysteresis parameters                                                                      |
| E   | Evaluation suite composition and hash                                                                                              |
| F   | Tier A case list                                                                                                                   |

**Data and code availability.** Scenario generator source and seed; frozen evaluation suite; system identification dataset; trained policy checkpoints. \[Repository DOI TBC\]

Open items

Blocking

| **Item**     | **Question**                                                                                                                                                 | **Blocks**                        |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| Precedence   | Rule 9 versus Rules 13–16 precedence table — the single blocking deliverable                                                                                 | §3.5, §4.4, §6.4, contribution C2 |
| IMU          | Is adding an inertial unit feasible? A low-cost gyroscope logging at 100 Hz would remove the yaw-rate observability constraint entirely. Assume none for now | §4.6, §7.2                        |
| Reflectivity | One full-length facility wall is matte black. Verify LiDAR return rate on that side before committing to continuous-wall registration                        | §4.6, §7.2                        |
| Compute      | Wall-clock estimate per run, then the final comparator list                                                                                                  | §5.5, §6.1                        |

Pending, not blocking

| **Question**                                                                               | **Blocks** |
|--------------------------------------------------------------------------------------------|------------|
| Head-on classification band width — the source value of ±5° is narrow relative to practice | §4.3       |
| Whether recurrence is added to handle target occlusion                                     | §4.5       |
| Final ship domain values from turning-circle data                                          | §4.3, §6.4 |
| Target vessel platform and field repeatability protocol                                    | §7.5       |
| Presentation format for multi-axis results                                                 | §5.4, §6.1 |

Resolved in this revision

| **Item**            | **Resolution**                                                                                                                            |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Scope               | Two-vessel encounters, Rules 13–16, Rule 9 precedence, own ship give-way throughout under Rule 9(b)                                       |
| External benchmark  | "Around the Clock" adopted; the two-vessel scope makes it an exact fit, and its open-water variant supplies the unconfined reference case |
| Sim-to-real         | Retained in this paper as RQ4, delivered as a domain randomisation ablation evaluated in the field                                        |
| Corridor dimensions | Simulation matches the basin; maximum width 10 m, sweeping to 3.5 m                                                                       |
| Boundary handling   | Geometric gating against the pool polygon, not a physical barrier; facility walls retained as the localisation reference                  |
| Ground truth        | No external instrumentation; scan-to-map registration against surveyed facility geometry                                                  |
| Ship domain         | Compressed asymmetric, provisionally 2.0 / 1.0 / 0.75 Lpp ahead, astern and abeam                                                         |
