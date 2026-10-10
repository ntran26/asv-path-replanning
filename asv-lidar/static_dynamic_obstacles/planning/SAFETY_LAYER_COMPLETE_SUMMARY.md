# Safety controller development: complete research record, V1-V19

**Evidence cutoff: 2026-10-03. Prepared as the starting point for researching and designing the next solution.** The per-version plans (V2-V19) and the chronological debugging notes are archived in [archive/safety/](archive/safety/) (2026-10-04); this file consolidates them.

**Subsequent research continuation:** [Reachability stage-1 plan](SAFETY_REACHABILITY_STAGE1_PLAN.md) and [saved-data results](../results/safety_dev/reachability_stage1/README.md) document the new offline conditional interval prototype, 101 passing tests, remaining interval overexpansion and a falsified smooth target-turn bound. No new controller or mission certificate was added, no episodes were run, and the V1-V19 performance history below remains unchanged.

This document is self-contained: it describes the task, simulator, controller architecture, every numbered version, measured successes and failures, rejected approaches, diagnostic evidence, and the gap between empirical improvement and a safety guarantee. Repository paths below are relative to `asv-lidar/static_dynamic_obstacles/` unless stated otherwise. External references are listed with method mappings in the bibliography. The preparation of this document ran **zero new episodes** and changed no controller.

Navigation: [current result](#1-present-conclusion-and-the-objective), [project setup](#2-project-mission-and-physical-setup), [evidence definitions](#3-how-to-interpret-the-evidence), [architecture](#4-architecture-and-version-lineage), [V1-V5](#5-version-history-v1-v5), [V6-V9](#6-version-history-v6-v9), [V10-V19](#7-version-history-v10-v19), [trigger studies](#8-failure-pattern-analysis-and-trigger-experiments), [diagnosed cases](#9-detailed-mechanism-cases-to-carry-into-the-next-research-session), [rejected model work](#10-model-audits-and-rejected-approaches-that-should-not-be-repeated-blindly), [remaining limitations](#11-why-it-still-does-not-work-reliably), [guarantee requirements](#12-what-could-justify-a-safety-guarantee-and-what-can-be-claimed-now), [constraints and files](#13-operational-constraints-evaluation-tools-and-artifact-map), [citations](#14-method-bibliography-and-exact-scope-of-attribution), [next-phase brief](#15-starting-brief-for-the-next-phase).

## 1. Present conclusion and the objective

The fixed learning policy is **SAC baseline 3, seed 0, kept-best checkpoint at 3 million training timesteps**. The safety layer should prevent contact with static obstacles, the target vessel, and navigable boundaries, while retaining the policy's ability to finish missions. The objective is 100% safety and preservation of the episodes SAC already solves.

**That objective has not been achieved. V16 is the strongest currently supported reference candidate; V17, V18 and V19 have not demonstrated an incremental net improvement. No version is a certified collision-avoidance controller.** “Reference” here is an experimental choice, not a change to the global runtime default or a deployment approval.

The clearest recent evidence is:

| Cohort | SAC OFF | V16 | Later comparison | Interpretation |
| --- | ---: | ---: | --- | --- |
| 32 deliberately difficult development cases | 16/32 goals | 25/32 | V15: 23/32 | V16 rescues 11 SAC failures but loses 2 SAC successes. |
| Separate 40-case development challenge | 21/40 | 25/40 | Historical V8 reference only | V16 rescues 8 SAC failures but loses 4 SAC successes. Boundary contacts rise from 1 to 6. |
| 12 targeted development regressions/controls | 10/12 in saved SAC references | 5/12 | V18: 5/12; V19: 5/12 | Intentionally includes all six V16-broken SAC successes from the two cohorts above. No new rescue. |
| Frozen 18-case primary test-set-v3 subset | Fresh 14/18 | Fresh 15/18 | V19: 15/18 | Both retain all 14 SAC successes and rescue the same one failure; V19 issues exactly the same commands as V16. Three collisions remain. |

These are different selections and must not be combined into one success percentage. **Only 18 of the 1,000 primary test-set-v3 cases have been evaluated with V16/V19.** Older full-set V7/V8 results belong to test set v2, which was subsequently used for development. The historical v3 SAC result is 853/1,000, with weaker provenance than the new matched runs.

The central limitation is a combination of prediction error and the structure of the filter. Its finite search asks whether a modeled backup exists after an action. It does not know the future closed-loop SAC trajectory, nor does it prove that a backup remains feasible forever. It can reject actual successful SAC maneuvers, accept unsafe forecasts, or reach a state where none of its sampled plans passes. More intervention and less intervention have both caused regressions.

## 2. Project, mission and physical setup

### 2.1 What is being controlled

The main project is a Gymnasium marine path-following environment. An own ship follows a reference path toward a goal through constrained water containing static panels/obstacles and normally one target vessel. Encounters include head-on, crossing, overtaking, being overtaken, null encounters, and no-target cases. The broader research also evaluates declared encounter-consistent behavior; the safety work described here focuses on physical contact and mission completion.

The model-scale Bluefin ASV has length 1.725 m and beam 0.50 m. The plant has surge, sway and yaw dynamics, a rudder servo and command delay. The simulator makes decisions every **0.5 s**, checks contact at **0.1 s**, and internally integrates vessel dynamics with RK4 substeps no larger than **0.05 s**. It normally caps an episode at 180 decisions (90 s). The predictive filter normally integrates at **0.125 s** and looks ahead **8 s**. The nominal rudder delay is approximately 0.73 s, discretized to 0.75 s in the predictor. The plant parameters are randomized in the evaluated scenarios, whereas the ordinary filter predicts with the nominal identified parameters. A bridge command-rate limiter is supported but disabled in the reported reference setup; internal servo dynamics are still modeled.

The basin is approximately 10 x 25 m; other cases use narrower virtual navigable channels. A mapped navigable boundary is a constraint even where it is not a physical wall. Consequently a reported “wall” or “boundary” contact can be a violation of that mapped navigable polygon. Facility walls and mapped boundaries are distinct inputs.

The SAC action is normalized rudder and propulsion in `[-1, 1]^2`. In the evaluated propulsion stage, its forward command is `clip(6 + 6 * throttle, 0, 12)` in simulator **rpm-units**; nominal cruise at 6 units is about 0.558 m/s. These units are a command scale, **not measured shaft RPM** (`bluefin/REPORT.md`). The policy cannot command astern. The safety layer can request full astern, **-24 rpm-units / S2=-100**, through a separate flag and environment hook. Braking reduces surge to zero; it does not provide a fully modeled maneuvering astern escape. Loss of propeller wash can also reduce steering authority while sway/yaw persist.

The identified model comes from forward-running field logs; reverse thrust was not identified from those logs. The simulator's reverse assumption is 0.5 times the forward thrust law with no reversal onset delay. The retained ordinary filter prediction uses a weaker 0.25 factor plus a 0.75 s onset delay. V4 added an **optional** check of braking plans under both weak/delayed and faster/stronger reverse assumptions. That option was tested but is disabled in the selected V4 and inherited reference configuration. Even when enabled, it is an empirical two-model check, not a bound on all possible reverse responses.

### 2.2 What the policy and filter can observe

The policy uses a 70-value structured observation: 27 pooled LiDAR values, 7 map-boundary rays, 3 ego-motion values, 3 path values, 16 target values and 14 context values. LiDAR has a 16 m maximum range and approximately 1 m near-sensor dead zone. An optional aft sensor mask exists but is disabled by default; separately, the compressed static-sector observation spans +/-135 degrees, with the tracker retaining the aft region. The observation pipeline includes noisy pose/motion, clustering, tracking, static/dynamic classification and encounter context. A partial hull can produce a biased cluster center and an unreliable hull heading.

The safety layer uses the onboard information behind this observation: estimated own pose and motion, LiDAR returns/history, the known map boundary, and track estimates. Later wrappers can use raw cluster/motion evidence and persistent hypotheses from the same sensor pipeline; they do not use hidden target behavior, true plant parameters or future policy actions. Thus “same information source as the policy” does **not** mean the filter reads only the compressed 70-vector.

Simulator collision/termination uses true geometric state. It is deliberately separate from what the controller senses. Full-snapshot diagnostics may record truth for error scoring or run an explicitly labeled oracle comparison; those quantities are not supplied to the deployed candidate. Never mistake an oracle forecast, a future recorded SAC command sequence, or a cloned simulator branch for an online certificate.

Targets are not uniformly constant velocity. The evaluation sets include constant-velocity (CV), varying-speed (VS/VAR), compliant reactive (RE), and non-compliant (NC) behavior. Older formulation prose saying that all targets are CV describes an earlier training scope; baseline-v3 adds varying-speed coupled curriculum cases. Most safety predictions still assume CV unless an explicitly experimental target hypothesis option is enabled.

### 2.3 Frozen policy and source identity

| Artifact | Path / identity |
| --- | --- |
| Policy | `runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip` |
| Policy SHA-256 | `993db1568929639903547a70087e5e913954318b9111f5a28413423a27c2bdc8` |
| Policy config | `runs/sac_formulation_seed0_bl3/kept_best_3M/config.json` |
| Config SHA-256 | `a4df64a97f641de0b1c660b04f73ccc8401a29f7fa48711c2a6ee694dc98ed44` |
| Baseline formulation | `configs/baseline_v3.json`, digest `03a9c0c22f53ee0b` |
| Unchanged `src/constants.py` SHA-256 | `3d71c76d3abfef1ac86e93afcd8fc0ea686b16b8a8930aa13050ed564c68f23f` |

The checkpoint config has no `constant_overrides`. SAC was trained with the supervisor off. Interventions may therefore put it into low-speed, displaced or wall-adjacent states rarely visited during training. This is a plausible source of poor hand-back behavior, not proof that retraining alone would solve the failures. The checkpoint has not been retrained during this safety work.

## 3. How to interpret the evidence

### 3.1 Terms and required metrics

- **Goal / success:** reaches the episode's goal termination. A timeout is not a success, even if no collision occurs.
- **Contact:** terminal obstacle, target or boundary collision. Changing the contact type does not count as a rescue.
- **Rescue:** same scene/seed is a failure with SAC OFF and a goal with the filter.
- **Lost/broken SAC success:** SAC OFF reaches the goal but the filter does not.
- **Preservation:** number of SAC-success cases still reaching the goal, divided by the number of SAC-success cases in that same selection.
- **Hard-passing plan:** no sampled violation and nonnegative minimum clearance under the current checker. This is a numerical/model statement, not a physical safety proof.
- **Trigger margin:** an extra acceptance/preference margin, usually 0.15 m, beyond the geometric clearance conditions. A hard-passing plan can still lose to a higher-margin plan.
- **Decision / frame:** normally one 0.5 s controller decision. Thousands of frames are correlated observations, not thousands of independent episodes.
- **Episode run:** one controller on one scene/seed. Three controllers on 32 scenarios are 96 runs and 32 distinct scenarios.
- **Last certificate:** a historical mode name for a stored plan. A currently failed recheck followed for a bounded number of steps is explicitly **uncertified**, despite that name.

### 3.2 Datasets and exposure

| Name used in this document | Meaning and status |
| --- | --- |
| DV3 / development-150 | The 150-case coupled validation set used for the early 111/150 SAC reference and later V7/V8 comparisons. DV3 is a validation-set version, not safety V3. |
| Old frozen / field sweep | Original broad comparison inventory: 2,890 scenarios x OFF/V4/V5 = 8,670 planned records. Stopped early. Details below. |
| TS2 | Test set v2, 1,000 saved cases. Designated for development on its failures; it is now development-exposed. |
| 32-case cohort | 27 TS2 + 5 DV3 cases, deliberately enriched for failures, rescues, lost SAC successes and six successful controls. Same identities used in V9-V16 comparisons. |
| 40-case challenge | 30 TS2 + 10 DV3, disjoint from the 32, frozen before expanded-candidate results but still an enriched development selection. |
| 12-case probe | Selected from the preceding development history, intentionally retaining all six V16 regressions. |
| TS3 | Current primary benchmark: 1,000 cases, 664 decoupled and 336 coupled. 755 exact geometry/seed pairs overlap TS2; 245 are new or changed. It is not wholly unseen. |
| Primary quick-18 | 9 decoupled + 9 coupled TS3 cases selected by metadata and a fixed identity hash before results; no overlap with the earlier 72 development scene/seed pairs. Ten overlap TS2 generally. |

Scene identity is the namespace, test ID, episode seed and scenario digest. A shared textual ID is insufficient. In particular, `TS2:` and `TS3:` must remain distinct. Do not compare percentages across different selections as if they were one learning curve.

Later strict result reports validate attempts, completed result JSON, traces, source archives, checkpoint/config hashes, effective controller classes/options, scene identities and completion tokens. Older CSVs have less complete provenance. Some historical numbers refer to different runtimes or baselines; discrepancies are explicitly identified below rather than silently reconciled by subtraction.

## 4. Architecture and version lineage

The safety hook runs at the top of `ASVLidarEnv.step()` before rudder/propulsion commands are issued. The environment then applies bridge command limiting when enabled and brake-to-zero conversion; an observer callback records the command actually issued. V1 uses `src/emergency_stop.py`; predictive versions use `src/safety_v<version>.py`. Native dispatch and suite options now accept versions through 19. `constants.py` was not edited to select a new default.

Core shared modules:

| Module | Role |
| --- | --- |
| `src/classical/common.py` | Perceived snapshots, static memory, actuator history, nominal rollout, rectangle/point/map/target clearance. Shared with classical comparators. |
| `src/safety_v2.py` | Candidate/recovery bank, reverse model, geometric checker, common margins and finite run-out condition. |
| `src/safety_v3.py` | Stored multi-step backup and recovery state machine. |
| `src/safety_prediction.py` | Weak/delayed and fast/strong braking trajectory checks. |
| `src/safety_observer.py` | Model prediction plus fixed-weight measured ego-motion correction. |
| `src/safety_track_persistence.py` | Independently admitted moving-target hypotheses and bounded persistence. |
| `src/safety_consistent_tracks.py` | Partial-extent center reconstruction and exact-source ownership choices. |
| `src/safety_policy_prefix.py` | Additional finite search with only the current 0.5 s SAC command fixed. |
| `src/safety_motion_axis.py` | V16 partial-hull completion using motion direction and known dimensions. |
| `src/safety_yaw_observer.py` | V17 fresh measured-yaw ablation. |
| `src/safety_heading_memory.py` | V18 bounded same-source hull-heading memory. |
| `src/safety_source_points.py` | V19 exact current-return ownership transfer. |

Logical inheritance is `V2 -> V3 -> V4`; V5 and V6 branch separately from V4; then `V6 -> V7 -> V8 -> V9 -> V10 -> V11`. V12 and V13 are separate V11 branches; `V13 -> V14 -> V15 -> V16`. **V17, V18 and V19 are separate V16 branches**, not a cumulative V17+V18+V19 stack. V10 reuses V9 utilities but deliberately calls V8's proposal logic instead of V9's blanket veto.

The ordinary hard checker subtracts additional gaps of 0.05 m to static returns, 0.02 m to the boundary and 0.10 m to targets, on top of hull geometry/margin conventions. The trigger margin is 0.15 m; relative room slack is 0.20 m. Static points are not a complete obstacle-surface model. Target prediction normally translates a fixed-orientation hull with its estimated CV motion.

At the horizon, a vessel moving faster than 0.10 m/s receives an extra straight-ahead static-clearance test over `min(2 m, 3 s * final speed)`, sampled at eight points. Terminal boundary extension is disabled in the retained configuration. This finite run-out is not a robust invariant terminal set and does not establish indefinite target/boundary safety.

Most versions initially commit a candidate for 1 s, although only 0.5 s is executed before replanning. Later prefix methods reduce that hypothetical policy commitment. Search banks and stored continuations are re-evaluated under the current snapshot; model changes, target maneuvers and sensor updates can invalidate an earlier plan.

The numbered versions below describe the tested ideas and their limitations. External papers supply inspiration or related architecture; the sampled engineering implementations do **not** automatically inherit the papers' assumptions or guarantees.

## 5. Version history: V1-V5

### V1: encounter-rule emergency-stop latch

**Implementation:** `src/emergency_stop.py`; early evidence is preserved in `planning/archive/safety/SAFETY_LAYER_V2_PLAN.md`.

V1 used an `IDLE -> BRAKING -> HOLDING -> policy` latch. A give-way encounter had to be judged in extremis, lack an admissible compliant maneuver, and have a predicted stop that cleared. It changed propulsion only; SAC continued steering. Minimum/maximum hold durations and a braking timeout governed release. COLREG Rules 8(e) and 17(b) were the stated rationale; this was a hand-built rule supervisor, not a barrier-function certificate. The selected COLREG context in the project is not a legal-compliance proof. See reference R1.

**Measured problem:** on an older **baseline-v2** SAC / frozen-suite-3.4 comparison, reported goal rate fell from 0.892 to 0.876, while target-contact rate rose from 0.064 to 0.066. This is not the present baseline-3/3M cohort. A separate paired table in the plan reports 46 stopped SAC episodes, 25 lost successes and 4 gains, with 95 of 98 target contacts occurring without a stop. The scopes of those two historical tables are not established as identical; do not derive common exact counts from their rates.

**Resolved by later work:** moved away from a rule-only stop trigger to physical rollout checks and turning alternatives. **Unresolved in V1:** false alarms in overtakes that would pass, no useful trigger for many crossings/head-ons, and low-speed states unfamiliar to SAC. It could not comprehensively monitor static obstacles or boundaries.

### V2: sampled predictive filter with recoveries

**Implementation:** `src/safety_v2.py`, `src/classical/common.py`; plan `planning/archive/safety/SAFETY_LAYER_V2_PLAN.md`; tests `tests/test_safety_v2.py`.

For each candidate, V2 simulated a committed command followed by a recovery maneuver. It tested the policy action, a rudder/throttle grid, braking candidates and sometimes the previous override. Recoveries included left/right powered turns, low-propulsion continuation and braking. It selected a passing action close to SAC, with more penalty for changing throttle because thrust supplies steerage. This is related to sampled forward simulation in the Dynamic Window Approach (R2), but is not an implementation of that original algorithm. Testing a learning input followed by a backup is also related to model-predictive shielding and predictive safety filtering (R3-R4); these are architectural relationships, not claims that V2 implemented their proofs.

The initial horizon was 6 s; iteration 2 used 8 s, braking and hold-back logic. Later revisions introduced bounded hysteresis, ego exponential smoothing, turning-room margins and a speed-scaled terminal run-out. A failed bank did not prove unavoidable collision: it either left SAC in charge or chose an alternative that delayed the predicted first contact sufficiently. An early use of V1's stop latch held the vessel in a target's path; braking was changed to a single decision, re-evaluated at the next decision.

**Early separate smoke test:** 100 L1/L2 fixed-speed cases: SAC 58 goals versus initial V2 57; obstacle contacts 26 -> 15, boundary contacts 6 -> 18, target contacts 10 -> 10. Fewer obstacle hits did not mean better mission completion.

**Matched DV3, 150 cases:**

| Configuration | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAC OFF | 111 | 26 | 2 | 11 | 0 |
| V2 iteration 2, historical best | **115** | 17 | 2 | 13 | 3 |
| Iteration 3: run-out, smoothing, 0.5 m trigger, turn preference | 107 | 10 | 9 | 10 | 14 |
| Iteration 5: 0.15 m trigger, speed-scaled run-out | 105 | 15 | 13 | 10 | 7 |
| Iteration 6: braking restricted to traffic | 105 | 19 | 17 | 9 | 0 |
| Iteration 6 without ego smoothing | 94 | 29 | 17 | 10 | 0 |
| Iteration 6 without terminal boundary run-out, later configuration | 109 | 18 | 14 | 9 | 0 |

Best iteration 2 rescues **16** SAC failures and loses **12** SAC successes: net +4, not 16 net additional goals. Its source/configuration differs from the later state of `safety_v2.py`; use the saved result/configuration evidence when reproducing historical iterations. Counts are recorded in the V2/V3 plans, `results/safety_v2_dev_*.csv` and `results/safety_dev/final_fullset_summary.csv`.

**Resolved:** some one-step margin collapses when approaching obstacles; some stop/deadlock behavior near walls; indefinite stop-latch use. **Unresolved:** boundary/obstacle/target tradeoffs, false interventions, no retained guaranteed escape and no calibrated error bounds. The weaker reverse model is not necessarily conservative for the whole trajectory: faster braking can remove steering authority while yaw continues.

### V3: committed backup and recovery/hand-back mode

**Implementation:** `src/safety_v3.py`; plan `planning/archive/safety/SAFETY_LAYER_V3_PLAN.md`; tests `tests/test_safety_v3.py`.

V3 stored the full selected plan, shifted it one decision each step, rechecked it against the new state and added a continuation preference. Recovery initially used the shared line-of-sight path-rejoin reference (Fossen, Breivik and Skjetne, R23), and hand-back required adequate policy clearance, steerage speed, alignment and a minimum recovery duration. The design was inspired by model-predictive shielding, backup controllers and predictive safety filters (R3-R5). The initial plan also mentioned a gatekeeper concept through a secondary literature summary; no specific gatekeeper theorem was implemented or verified.

Two implementation corrections mattered: unrestricted stale-plan following, observed for 13-140 steps, was capped at **2 s / four decisions**; and the recovery reference reverted to SAC because the nominal path itself can run through obstacles. “Last certificate” remained a bounded fallback after a failed recheck, not a current certificate.

**DV3 results:** initial path rejoin 93/150 goals; bounded plan expiry plus path rejoin 95/150; bounded expiry plus policy reference **108/150** (18 obstacle, 14 boundary, 10 target, no timeouts). The 108-goal configuration rescues 17 SAC failures and loses 20 successes. It is worse than V2 iteration 2's 115 goals and 2 boundary contacts.

Rejected ablations included throttle weight 0.5 (101 goals), no room slack (108), 16 s horizon with a 4 s recovery turn (98), and 16 s horizon with an 8 s turn (82). A longer horizon did not cure the boundary problem. All 14 original V3 boundary contacts ultimately reported `no escape`.

**Resolved:** stale stored-plan runaway and recovery steering back through an obstacle. **Unresolved:** model pessimism/optimism, hand-back distribution shift, bounded uncertified continuation, and many boundary failures. The old plan asserts that no invariant terminal set exists because the basin is narrower than a steady turning circle. That is an engineering argument, **not a mathematical impossibility proof**; the justified statement is that no suitable robust invariant terminal set was constructed or verified.

### V4: ego-motion observer and evidence-based static-memory clearing

**Implementation:** `src/safety_v4.py`, `src/safety_observer.py`, `src/safety_perception.py`, optional `src/safety_prediction.py`; plan `planning/archive/safety/SAFETY_LAYER_V4_PLAN.md`.

V4 followed the diagnostic order set in the original notes: compare forecasts with actual motion, use truth-only oracles to separate perception from prediction, classify failures, then test changes.

The selected configuration adds two methods. A command-driven nonlinear model observer advances the previous body velocity/yaw state and applies the existing 0.3 measurement correction. It uses the actual issued rudder and signed RPM after environment conversion, and avoids re-assimilating a stale measurement. This is retrospectively related to model-plus-output-error observers (Luenberger, R6); it is not a Kalman filter or a convergence proof. Static-memory clearing removes an old point only when a fresh finite ray gives contradicting free-space evidence and current returns do not support occupancy. No-return, stale, occluded, aft-masked and dead-zone regions do not count as free. This is retrospectively related to ray-based occupancy updates (Hornung et al., R7), without using an octree or probabilistic log-odds map.

Optional countersteering recovery, retaining checked nominal plans and checking both reverse-response models were implemented/tested but are disabled in selected V4. The selected switches are `MODEL_EGO_OBSERVER=True`, `FREE_SPACE_MEMORY=True`, with `COUNTERSTEER_RECOVERY`, `RETAIN_NOMINAL_PLAN` and `DUAL_BRAKE_PREDICTION` false.

| DV3 configuration, N=150 | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial countersteer + plan retention | 108 | 23 | 10 | 9 | 0 |
| Memory + dual reverse response | 118 | 13 | 10 | 9 | 0 |
| Observer + memory + dual reverse response | 118 | 11 | 10 | 11 | 0 |
| Observer only | 110 | 16 | 12 | 12 | 0 |
| **Selected observer + memory** | **123** | **9** | **7** | **11** | **0** |

Selected V4 rescues **19** SAC failures and loses **7** successes; against V2 iteration 2 it gains 16/loses 8, and against V3 it gains 24/loses 9. It rescues 12 of V3's 14 wall-failure cases but introduces six other wall outcomes. Of V4's seven boundary contacts, six end in `no escape` and one in `last certificate`.

On the 14 V3 wall-failure cases, a true-ego-plus-target oracle produced 8 goals, full true geometry 11, memory-only 5, dual-brake-only 4 and observer-only 9. These are **failure-selected diagnostic interventions**, not full-set controller results or deployable options. A 938-decision replay reduced mean absolute surge error 0.0537 -> 0.0169 m/s, sway 0.0281 -> 0.0152 m/s and yaw 2.665 -> 0.849 deg/s relative to the older estimate. Raw measured yaw was slightly better still, at 0.803 deg/s. Later cohorts expose different observer tradeoffs; this early improvement does not establish universal yaw superiority.

**Broader test limitation:** the stopped comparison below shows only a small net gain on its own matched cases. **Resolved:** useful reductions in ghost static occupancy and old ego-estimate error. **Unresolved:** false overrides, uncertain target extent/motion, boundary regressions, nominal-model mismatch and lack of a verified terminal backup.

Verification at this stage included 69 passing focused tests. An existing V3 dead-ahead test also failed with the original environment source and was explicitly deselected, not silently repaired. Local timing on 552 decisions gave filter median/p95/max 0.077/0.122/0.299 s; whole-step p95/p99/max 0.445/0.523/0.581 s. Those host-specific measurements do not guarantee a 0.5 s hardware deadline.

### The stopped 8,670-record sweep: final accounting

Evaluation beyond 150 validation episodes was started and later stopped. The original inventory contained DV3 150 + legacy development 120 + Tier-1 head-on 100 + frozen B 800 + frozen R 900 + frozen A 35 + field deployment layouts 630 + coupled validation layouts 155 = **2,890 scenarios**. OFF, V4 and proposed V5 made **8,670 planned controller/scenario records**. “Field” means simulated coupled layouts here, not new physical trials.

The authoritative final snapshot is `results/safety_dev/v4_stopped_full_sweep_report.md`: **3,347 committed records, 5,323 absent**. Only B and R ran; **zero V5 records** were completed in this old sweep. B has OFF 799 and V4 792 records, giving 791 pairs; R has OFF 900 and V4 856, giving 856 pairs. OFF `B-06-040` is absent from the durable journal and was excluded rather than reconstructed.

| Final matched scope | SAC goals | V4 goals | Rescues / losses |
| --- | ---: | ---: | ---: |
| B, 791 pairs | 740 | 737 | 23 / 26 |
| R, 856 pairs | 781 | 788 | 27 / 20 |
| **Combined, 1,647 pairs** | **1,521 (92.35%)** | **1,525 (92.59%)** | **50 / 46** |

Combined OFF/V4 contact counts are obstacle 31/18, boundary 13/26, target 82/78; total 126/122, no timeouts. V4 removes 13 obstacle contacts but doubles boundary contacts, producing only four additional goals. Sequential truncation and correlated robustness variants prevent treating this as a representative independent population sample.

An earlier cancellation snapshot reports 3,141 records, 1,441 pairs and V4 1,327 versus OFF 1,331 goals. It was an interim snapshot and is superseded by the final counts above. The earlier DV3 evaluation and subsequent quick campaign do not fill missing records in this ledger. **Do not restart this queue or claim that 8,670 episodes completed.**

### V5: feedback backups, soft recovery and policy-preservation experiments

**Implementation:** `src/safety_v5.py` and its helper modules; plan `planning/archive/safety/SAFETY_LAYER_V5_PLAN.md`; method bibliography `planning/SAFETY_LAYER_REFERENCES.bib`.

V5 explored several separately switchable changes rather than one universally enabled method:

1. **One-decision commitment:** fix SAC for 0.5 s instead of 1 s before backup, motivated by one learning action followed by backup in model-predictive shielding (R3).
2. **Soft infeasibility ranking:** when every candidate fails, rank integrated positive clearance deficits and then action deviation, including outside-map cost. Related to predictive control-barrier recovery (R8), but the sampled deficit is not a proved predictive barrier function.
3. **Feedback course backups:** recompute rudder along predicted trajectories for edge-parallel/current-course/offset alternatives. Maritime branching-course MPC and backup-controller integration motivate this (R9, R5); the four directions and +/-30 degree offsets are engineering choices.
4. **Sideslip compensation:** use `heading = desired course - atan2(v, u)` within eligible feedback rescue. The kinematic relationship comes from Fossen, Pettersen and Galeazzi (R10); the full adaptive LOS law and stability proof were not implemented.
5. **Verified-policy priority:** if SAC already belongs to the same accepted feasible pool, let it win before the continuation bonus. This approximates the first-input deviation objective of predictive safety filtering (R4), not a general no-regression guarantee.
6. **Policy-only feedback preservation:** test the exact current SAC command plus four feedback backup tails when ordinary policy backups are poor and an alternative exists. It retains the 1 s commitment and 0.15 m acceptance margin; it does not roll SAC forward on imagined observations.

Initial DV3 ablations on 150 cases were: one-decision-only 111 goals (18 obstacle/9 boundary/12 target); soft-only 121 (12/7/9 plus one timeout), with 3 gains/5 losses against V4; broad feedback 121 (13/6/10); infeasibility-only feedback **123** (10/6/11). The latter has **zero gained or lost goals** relative to V4: one boundary failure becomes an obstacle failure. It was the first experimental V5 preset, but no V5 episodes reached the stopped large sweep.

The revised quick preset enables policy priority and policy feedback, retains V4's observer/memory, and disables one-decision, soft, broad-feedback and sideslip options. Four development probes (two failed cases under two options) produced no rescue. Then 24 preselected decoupled/coupled scenarios were evaluated with OFF/V4/V5: **72 new runs**, plus the four probes = **76 of the then-authorized 100-run budget**. The 24 cases were a stratified quick check representing the original inventory's important categories, not a statistical substitute for 8,670 records.

| Quick-24 controller | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| OFF | 18 | 4 | 0 | 2 | 0 |
| V4 | 19 | 2 | 1 | 2 | 0 |
| V5 revised preset | 19 | 1 | 1 | 3 | 0 |

V5 gains/loses 3/2 against SAC and 0/0 against V4. Changed-action steps fall from 152 to 121, but one obstacle collision becomes a target collision. Fewer interventions are not an outcome improvement. There were 42 feedback checks over 15 episodes, six accepted over three; activity categories overlap. V5 was **not promoted**. The 72 comparison runs took about 409 episode-seconds, excluding probes and model setup; 99 focused tests passed. Source: `results/safety_dev/quick_v5_budget100/runs/policy_feedback_v1/report.md`.

## 6. Version history: V6-V9

### V6: evidence-qualified provisional targets and trajectory search

**Implementation:** `src/safety_v6.py`, provisional-track/search/geometry/calibration helpers; plan `planning/archive/safety/SAFETY_LAYER_V6_PLAN.md`; report `results/safety_dev/development_v6_budget150/report.md`.

V6 branches from selected V4 and does not implicitly enable V5. Its accepted preset adds (a) safety-only provisional views of moving clusters before ordinary dynamic-track publication, requiring existing finite-ray motion evidence, and (b) a bounded cross-entropy-method (CEM) trajectory search when the rigid recovery bank is insufficient. The search expands the maneuver family but remains a finite sampler. CEM inspiration is Zheng et al. (R11), without their Gaussian-process or barrier-function construction. Motion evidence follows the free-space consistency idea of Yoon et al. (R12). Optional hull-fit/center methods relate to Zhang et al. and extended-object modeling (R13-R14).

Additional ablations tried calibrated braking, fitted target center/return masking and historical target hypotheses. Brake calibration used measured development braking pulses and prediction-error estimation (Ljung, R15), not hidden plant parameters. The fitted effective response at -24 rpm-units is about 0.465206 m/s^2, equivalent to efficiency 0.493899 with zero effective delay. Delay and thrust are not separately resolved by these sparse short records; this is not a general physical reverse model. Calibration and the fitted-hull/history options remain disabled in the accepted V6 preset. V6's original CEM samples three rounds of 64 plans plus supplied seeds, with eight elites; two searches can therefore use 384 new samples plus seeds in one decision. This differs from the later prefix helper's strict 192-plan total cap.

| V6 development experiment | Goals and outcome |
| --- | --- |
| Provisional tracks alone, 4 cases | 2/4, equal to fresh V4: one rescue and one regression. A static panel's changing visible centroid was mistaken for motion. |
| Provisional tracks with existing motion evidence, same 4 | 3/4; retained that rescue and removed that inspected regression. |
| CEM search alone, 4 cases | 3/4 versus V4 2/4. |
| Search + motion-qualified provisional tracks, 6 cases | 4/6. |
| Add historical target hypotheses, same 6 | 3/6; loses CRP-CV-12. Rejected. |
| Calibrated braking + fitted hull geometry, 8 cases | 1/8 versus historical V4 2/8. Components were not isolated; prediction fit did not establish better control. |

The larger frozen development selection contained **all 27 historical V4 failures plus 23 V4 successes**:

| Controller, N=50 | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh V4 | 23 | 9 | 7 | 11 | 0 |
| V6 accepted preset | **33** | **9** | **3** | **5** | **0** |

V6 rescues 13 V4 failures and loses three V4 successes: HO-VS-01, CRP-CV-16 and CRS-VS-03. This is substantial improvement on a failure-enriched subset, not a 66% full-set estimate. Its 17 observed failures also bound that **frozen candidate's** possible result on the original 150 cases at at most 133/150, even if every untested case succeeds. This is a finite-set arithmetic bound, not a fundamental controller or physical limit.

The then-authorized 150-attempt campaign consumed **149 attempts: 144 policy episode runs plus five legacy verification simulations**. A proposed final probe stopped before reserving an attempt because `scenario.py` changed at source preflight. One legacy V3 simulation test failed and did not instantiate V6. These distinctions matter when interpreting the budget and tests.

**Resolved:** some delayed/missing published target views and limitations of a rigid primitive library; one inspected static-panel false-motion regression. **Unresolved:** 17/50 failures, three sacrificed V4 successes, prediction bias, initial blind-zone risk and missing proof of future recoverability. Improving a brake fit or adding target history alone did not improve outcomes.

### V7: check SAC's current action with the proposed backup tail

**Implementation:** `src/safety_v7.py`, `src/safety_risk_monitor.py`; plan `planning/archive/safety/SAFETY_LAYER_V7_PLAN.md`.

Before a V6 override, V7 substitutes the current SAC command for **only the first 0.5 s decision** of the proposed backup, retaining/padding the rest to the full eight-second horizon. It evaluates from pre-command actuator history and preserves SAC only if the complete sequence satisfies the existing hard checks and 0.15 m margin. It retains that checked plan. This is a local feasible-first-input adaptation of predictive safety filtering (R4), not a prediction of future SAC actions.

A separate observable-risk monitor records per-hazard clearance, predicted contact, closing trends and persistence. It is **shadow-only**: its recommendation never authorizes an unchecked bypass or changes the action. Monitor/intervention separation relates to Hsu, Hu and Fisac (R16). Initially V7 was implemented with saved-data analysis and synthetic tests only (96 focused checks); later branch experiments supplied its episode evidence. Old “unmeasured” prose in the source/plan is superseded by those results.

Full saved outcome reconstruction: **904/1,000 TS2 goals**, 58 SAC failures rescued and 26 SAC successes lost; **128/150 DV3 goals**, 21 rescues and four losses. V4 on those references is 880/1,000 and 123/150 respectively. Details of how these full outcomes were assembled appear below.

**Resolved:** avoids some unnecessary overrides caused by a longer fixed policy commitment or continuity preference. **Unresolved:** checks only one backup tail, requires a hand-set margin, retains unchecked fallback branches and cannot classify whether future closed-loop SAC succeeds. A local checked action can still lead to failure after subsequent decisions.

### V8: remove the harmful hold-back fallback

**Implementation:** `src/safety_v8.py`; plan `planning/archive/safety/SAFETY_LAYER_V8_PLAN.md`; saved audit `results/safety_dev/v8_followup_offline/`.

V7's hold-back first-fire branch had **three rescues but eight lost successes** in the analyzed branch study. That branch selected a presently failing candidate because it delayed predicted contact by enough. V8 disables it: if no acceptable alternative exists, SAC stands unless the separate stored-continuation fallback applies. This is an empirical ablation of the inherited sampled filter, not a new formal method. Its architectural citations remain R3-R4/R11; the reason for removing this particular branch is the project evidence.

A later saved-only fix changed the implementation from temporarily mutating a shared constant to an instance method: `_hold_back_gain_s()` returns infinity in V8. This preserves serial no-hold behavior while avoiding shared-state interference between concurrent/nested filters. It does not remove `last certificate`, which can still execute a currently failed stored continuation for up to four decisions.

| Full historical outcome reconstruction | SAC OFF | V4 | V7 | V8 |
| --- | ---: | ---: | ---: | ---: |
| TS2, 1,000 cases | **872** | 880 | 904 | **912** |
| DV3, 150 cases | **111** | 123 | **128** | 127 |

TS2 rescue/loss pairs versus SAC are V4 49/41, V7 58/26, V8 **57/17**. DV3 pairs are V4 19/7, V7 21/4, V8 **20/4**. V8's TS2 failures are 31 target, 27 obstacle and 30 boundary contacts: 71 unrescued SAC failures plus 17 newly broken successes. Twenty-two of its 57 rescues began below 0.15 m, so a retrospectively stricter uniform intervention margin would discard real rescues.

**Provenance:** full V7/V8 results are validated closed-loop reconstructions from a common SAC prefix and recorded first-fire branches, not 1,150 separate new standalone filtered episodes. The V8 audit assembles 526 first-fire branches, 601 cases where base V7 never fired and 23 where the no-hold variant never fired. `results/safety_dev/v8_followup_offline/audit/summary.json` and `all_1150.csv` retain identity/outcome provenance. The source TS2 OFF CSV/summary gives **872 goals**, 61 target, 55 obstacle and 12 boundary contacts; the TS2 definition/manifest digest is `6df8076223414a88`. A remembered 895-goal figure has no supporting artifact in this audit and must not replace that baseline.

**Resolved:** better net TS2 outcomes and removal of one empirically harmful unchecked maneuver selection; shared-constant mutation eliminated. **Unresolved:** lost SAC successes, 88 TS2 contacts, lower DV3 goals than V7, failed stored-plan use, and three target collisions with no intervention despite positive final predicted margins. `last certificate` first-fire analysis shows 0 rescues/3 losses in TS2 but 4 rescues/1 loss in DV3: blanket removal has a real tradeoff.

### V9: currently recheck overrides and compare policy clearance

**Implementation:** `src/safety_v9.py`, optional `src/safety_target_prediction.py`; plan `planning/archive/safety/SAFETY_LAYER_V9_PLAN.md`.

Two independent guards were added. First, recheck the actual proposed action plus complete backup using the current snapshot and pre-command actuator history, suppressing an override if it fails. Second, test SAC followed by the same tail, preserving SAC if it passes and its minimum clearance is at least that of the proposal. The second rule is related to feasible first-input filtering (R4). Suppressing a failed override does **not** certify the returned policy action.

An optional target ensemble samples CV plus immediate port/starboard constant turns (Li and Jilkov, R17; Johansen et al., R18). Default turn rate is **zero**, so this is disabled in the reported defaults. It is not an IMM tracker or a reachable set; delayed turns, accelerations and missing targets are outside its sampled hypotheses, and the chosen turn rate is uncalibrated. No formal guarantee is claimed. Adding a union of motion hypotheses can increase conservatism and cannot remove a false rejection already caused by the retained CV hypothesis.

The initial follow-up used only saved data and code tests (134 focused plus three supplemental checks). Those older logs lacked complete snapshots/plans, so they could not retrospectively measure V9. Once fresh episodes were authorized, a separate pilot ran **96 episodes = 32 scenarios x OFF/V8/V9**, one process/one Torch thread, about 28 minutes, with no retries.

| Fresh pilot, N=32 | Goals | Rescues of 16 SAC failures | Lost of 16 SAC successes | Changed-action steps |
| --- | ---: | ---: | ---: | ---: |
| SAC OFF | 16 | - | - | 0 |
| V8 | 16 | 10 | 10 | 393 |
| V9 | 16 | 5 | 5 | 106 |

V9 gains six and loses six goals against V8. Eleven switches first diverge when it rejects `last certificate` (five gains, six losses); the remaining gain first uses policy-margin dominance in P2-L2-HO-FIX-19. These identify first divergence, not isolated causal effects of each guard over the entire episode. Selected TS2 results are V8 12/27 versus V9 15/27; selected DV3 results V8 4/5 versus V9 1/5. All six successful control cases remain goals. All 64 fresh OFF/V8 outcomes reproduce the historical references; 4,208 decision records and all 32 triples are preserved.

**Resolved:** fewer unchecked overrides and fewer lost SAC successes on this selection. **Unresolved:** equal net goal count, five fewer SAC failures rescued than V8 (six V8 rescues lost, offset by one new rescue), three never-triggered target contacts, and unverified policy fallback. V9 was not promoted. Source: `results/safety_dev/v9_paired_pilot/{report.md,paired.csv,verification.json}`.

## 7. Version history: V10-V19

### 7.1 Common 32-case comparison

The V10-V17 campaign completed **399 new runs over 19 experiment tags**, covering 72 distinct development scenarios with no retries. It contains the 32-case progression, a disjoint 40-case challenge, and repeated diagnostic/ablation runs. The earlier V9 pilot's 96 runs are a separate ledger. Source: `results/safety_dev/v10_iterations/report.md` and `campaign_status.json`.

| Controller | Goals / 32 | Target | Obstacle | Boundary | SAC failures rescued / 16 | SAC successes preserved / 16 | SAC successes lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF | 16 | 8 | 6 | 2 | - | 16 | 0 |
| V8 | 16 | 9 | 2 | 5 | 10 | 6 | 10 |
| V9 | 16 | 8 | 6 | 2 | 5 | 11 | 5 |
| V10 | 18 | 9 | 2 | 3 | 11 | 7 | 9 |
| V11 | 18 | 7 | 3 | 4 | 12 | 6 | 10 |
| V14 | 22 | 5 | 2 | 3 | 11 | 11 | 5 |
| V15 | 23 | 6 | 2 | 1 | 11 | 12 | 4 |
| **V16** | **25** | **4** | **2** | **1** | **11** | **14** | **2** |

All rows have zero timeouts. **V12, V13 and V17 have only smaller probe results and no full 32-case result.** Source/configuration differences between numbered candidates are archived rather than assumed away. All six successful control cases remain successful with V16.

### V10: retain useful V8 fallbacks and preserve only a passing SAC backup

**Implementation:** `src/safety_v10.py`; history `planning/archive/safety/SAFETY_LAYER_V11_PLAN.md`.

V10 uses V8's action proposal and V9's paired checker. It preserves SAC if SAC+the same full backup tail passes while the proposal fails, or both pass and SAC has at least as much clearance. If both fail, it retains V8's action, stored plan and bounded fallback counters. Unlike V9, a failed proposed backup alone is not treated as sufficient reason to return SAC. The first-input preservation motivation is R4; the same-tail comparison and relative-margin rule are project heuristics.

It reaches **18/32**, rescuing 11 SAC failures but breaking nine successes. A guard ablation with only V9's blanket veto disabled reproduces V8's 16/32. An optional `prefer_any_feasible_policy=True` accepts any hard-passing SAC backup regardless of the override's larger margin: this reaches **15/32**, with two gains and five losses versus default V10. It remains disabled. **Resolved:** recovers rescues lost by V9's blanket veto. **Unresolved:** unchecked fallback when both plans fail, large policy-success losses, and the possibility that a locally feasible SAC command is poor over an episode.

### V11: independently observed target motion and bounded persistence

**Implementation:** `src/safety_v11.py`, `src/safety_track_persistence.py`; history `planning/archive/safety/SAFETY_LAYER_V11_PLAN.md`.

V11 admits a moving hull after at least three actually observed suitable clusters, full-length extent, coherent translation of both fitted endpoints and acceptable regression residuals. It anchors measured center/velocity and coasts for at most eight seconds, even if the raw tracker deletes its ID. Finite-ray contradiction removes a hypothesis; a missing/no-return/stale observation is not proof of empty space. Synthetic persistent views use separate negative IDs. At this stage static points are retained, even if target-related, which later motivates V19.

Sources of inspiration are dynamic occupancy/persistence (Nuss et al., R19), free-space motion evidence (Yoon et al., R12) and fitted geometry (Zhang et al., R13). This is a deterministic wrapper, not their full estimators or a probabilistic association guarantee. It can retain stale target motion or duplicate a base view.

V11 reaches **18/32**: 12 rescues and ten lost SAC successes. Versus V10 it gains two goals and loses two. The three historical V8/V9 never-intervened target cases were P2-L1-CRP-FIX-05, P2-L1-CRS-VAR-08 and P2-L1-CRP-VAR-14. V11 rescues the latter two while FIX-05 remains a target collision. It addresses missing target views, but false/conflicting geometry offsets those gains elsewhere. Later V16 still succeeds on CRP-VAR-14 but loses CRS-VAR-08 to an obstacle; these are not three unchanged never-intervened failures in every version.

A separate reusable **current-policy-prefix search** was also developed: fix only the current 0.5 s SAC command, search the remaining full eight-second backup with up to 192 plans, three CEM rounds and eight elites, and apply the existing hard checker. Inspiration: R4 and R11. Failure leaves the parent action; finite-search failure is not an infeasibility proof. This option is not simply synonymous with default V11. Isolated on V10 it reaches **17/32**, two gains and three losses versus V10. Its two slices have an explicitly audited archive-inventory difference (an unused validation-set module), so they were not presented as a strict source-identical cohort.

### V12: prefer the existing base view for the same source

**Implementation:** `src/safety_v12.py`, `src/safety_track_fallback.py`; independent V11 branch.

If a persistent view has an exact matching source ID already represented by a base track, V12 gives the base view priority, retaining persistence as fallback. This tests whether duplicate/conflicting extrapolation is the problem. It is a project association/ownership rule, related to motion persistence and extended-object association (R19, R14), not a new tracking theorem.

**Only a four-case probe was run:** three goals and one target contact. This cannot be compared with a 32-case score. It reduces duplicate-source conservatism in some cases but relies on an occasionally biased base geometry. It was not established as a full-cohort improvement.

### V13: partial-hull center continuity and fixed source ownership

**Implementation:** `src/safety_v13.py`, `src/safety_consistent_tracks.py`; separate V11 branch, not a V12 successor.

For a partially observed axis with known hull size, V13 projects a CV center prior into the interval compatible with the observed extent: `[observed maximum - size/2, observed minimum + size/2]`. A fully observed axis uses its midpoint. This separates object extent/kinematics from the visible return centroid, as motivated by extended-object tracking (R14). Ownership is chosen at anchor birth: an existing base owner keeps priority; an independently admitted owner uses its persistent view. Association uses exact source identity, not arbitrary cross-ID merging.

**Four-case probe:** two goals, one obstacle contact and one target contact. Disabling both center/ownership options reproduces V11. This experiment did not fix the stopping/missing-heading target case. A correct size prior alone cannot recover a wrong orientation, a bad motion prior, or a changing target behavior. V13 has no 32-case result.

### V14: prefer the corrected persistent estimate for a matching source

**Implementation:** `src/safety_v14.py`; builds on V13.

V14 prefers the corrected persistent estimate for valid exact source IDs regardless of initial owner. It retains unrelated base views and all static points, without cross-ID association. This tests which of the available estimates should constrain a single object, drawing on the same extent/state and association distinction (R14/R19).

It reaches **22/32**, with 11 rescues and five lost SAC successes; versus V11 it gains five goals and loses one. It improves partial-hull/duplicate-view handling but can prefer an old CV anchor over new evidence of turning/stopping. Broadly enabling current-policy-prefix search on V14 reaches **21/32**, below the default 22; broader policy preservation was therefore not selected wholesale.

### V15: search for a SAC-first backup only in selected branches

**Implementation:** `src/safety_v15.py`, `src/safety_policy_prefix.py`; inherited V6 adds diagnostic selection-floor fields without changing its selection logic.

V15 applies the shorter SAC-prefix search only when the paired parent currently fails, using minimum margin zero, or when an ordinary passing parent is below the existing 0.15 m trigger margin. In the latter case it uses the parent's exact selection floor, clamped nonnegative: `max(0, min(0.15, best clearance - ROOM_SLACK_M))`. An already adequate parent or invalid/missing branch diagnostics skips the extra search. There is no new tuned numerical threshold. The inspiration is minimizing first-input deviation subject to a feasible backup (R4), with CEM (R11); the branch gates themselves are engineering decisions.

It reaches **23/32** with 11 rescues and four lost SAC successes. Versus V14 it gains three and loses two, so it is not regression-free. It preserves BAS-NU's earlier decision 10 but still fails the episode after a new override at decision 11. **Resolved:** some cases where a useful current SAC action was rejected by a rigid or overly long prefix. **Unresolved:** model false rejections, high-margin-parent preference and future loss of backup feasibility. Thirty-five focused V15 checks and separate prefix-hook checks were used; overlapping test totals must not be summed.

### V16: motion-axis hull completion for eligible partial base tracks

**Implementation:** `src/safety_v16.py`, `src/safety_motion_axis.py`; plan `planning/archive/safety/SAFETY_LAYER_V16_PLAN.md`; 24 targeted tests.

V16 corrects eligible fresh nonnegative-ID base views using measured velocity direction and known hull dimensions. It requires a unique confirmed source, current pose, no missed detection, finite estimates, enough current/history points, speed above the existing 0.15 m/s threshold, and motion evidence. The observed beam must agree with known beam within the existing 0.15 m tolerance, while length is partial; the observer must lie outside the observed longitudinal span, and dead-zone clipping must be absent. At least three actual observations in a two-second window must support forward endpoint translation above 0.25 m with residual/velocity consistency. No serial-number gaps are invented as elapsed observations.

For qualifying visible bow/stern geometry it completes known length **away from the visible sensor face**, while centering the beam. It preserves track velocities, context, ordering, all static points and negative persistent views. It does not coast a correction through missing evidence. Disabled `motion_axis_geometry=False` restores V15 perception. This is an engineering partial-extent correction inspired by R14 and R13; course need not equal hull heading, and occlusion can mimic a visible end face.

The matched mechanism and outcome are strongest in **CH-HO-CV-073** and **DV3-BO-CV-04**: the old target center/axis rejected SAC's known successful future; onboard-only completion improved that estimate, and fresh episodes then succeeded. These two new goals give **25/32**, +2/-0 versus V15, with four target, two obstacle and one boundary contact. It retains 14/16 SAC successes and all six successful controls. The two remaining lost SAC successes are **BAS-HO-NC-059 (target)** and **BAS-NU-CV-070 (boundary)**.

The separate 40-case challenge gives fresh OFF **21/40** (8 target, 10 obstacle, 1 boundary) versus V16 **25/40** (5 target, 4 obstacle, 6 boundary). V16 rescues 8/19 failures and preserves 17/21 successes, including all ten selected successful controls. It retains 7/11 cases previously broken by V8, retains 8/11 rescue controls and rescues 0/8 cases both SAC and V8 failed. Fresh SAC reproduces all 40 historical outcomes. A historical V8 comparison gives +7/-3; no new V8 episode is implied.

The four lost SAC successes in that challenge are **DV3-CRS-CV-04, P2-L2-CRS-FIX-19, P2-L3-BO-FIX-09** (boundary) and **P2-L3-CRP-FIX-03** (target). These plus the previous two form the six-regression inventory. Do not claim all regression causes are fully diagnosed: the latest deep mechanisms are strongest for BAS-HO, BAS-NU and DV3-CRS-CV-04.

The existing any-feasible-policy option was retested on nine targeted cases: **3/9 versus default V16 6/9**, zero gains and three losses (DV3-HO-CV-03, P2-L1-CRP-VAR-12, P2-L2-HO-FIX-19). Both BAS regressions remain. Simply lowering the priority of safety intervention is not supported as a general fix.

### V17: fresh measured yaw instead of model-corrected yaw

**Implementation:** `src/safety_v17.py`, `src/safety_yaw_observer.py`; plan `planning/archive/safety/SAFETY_LAYER_V17_PLAN.md`; 23 tests. Independent V16 branch.

Saved observer evidence showed u/v filtering helps but filtered yaw can be worse than the raw gyro. V17 retains the ordinary observer validation, u/v correction, reset and stale-frame prior, but uses the raw measured yaw rate on fresh frames. This is a gain-one measurement ablation related to the observer correction structure of Luenberger (R6), not fitted dynamics, a Kalman redesign, or a proved gain choice. Later u/v can change through coupled dynamics. Disabling it reproduces the parent observer.

**Nine-case probe only:** 6/9 goals, one target and two boundary contacts, equal to V16's goal count. It gains DV3-HO-VS-01 and loses P2-L2-HO-FIX-19, leaving both BAS regressions. It is not promoted. Better instantaneous yaw estimation is not sufficient evidence of better multi-step collision forecasts or closed-loop behavior.

### V18: bounded memory of a validated hull heading

**Implementation:** `src/safety_v18.py`, `src/safety_heading_memory.py`; plan `planning/archive/safety/SAFETY_LAYER_V18_PLAN.md`; 26 tests. Independent V16 branch; it does not include V17.

V18 retains the last validated fitted hull axis for the same source when a fresh observed target loses its fitted heading and would otherwise fall back to noisy velocity course. Only fresh matched confirmed new-serial observations update/use it. Expiry is the existing `TRACK_MAX_MISSES` budget, three decisions / 1.5 s. Missing IDs, detectable identity reset, duplicates, expiry or a finite but rejected fit clear/reject memory. Persistent synthetic views and a current V16 geometry correction have priority. Positions, velocities, static points and context are unchanged. The inspiration is separating object kinematics and extent from individual observations (R14); this deterministic hold can still be wrong for a turning target.

The motivating saved FIX05 recheck changed a nominally passing margin into a failure after correcting a missing-heading fallback; details below. That is a model correction, not a demonstrated rescue.

**Fresh 12-case development probe:** V16 and V18 both **5/12** (three target, four boundary contacts). V18 activates at four decisions in two cases but changes **zero issued commands**. In the fresh V16-derived FIX05 path its only activation is at terminal decision 34, too late to help. No V16 goal is gained/lost, and all six SAC regressions remain. It is not promoted and was **not** evaluated in the primary quick-18 comparison.

### V19: transfer exact current returns already owned by a moving target

**Implementation:** `src/safety_v19.py`, `src/safety_source_points.py`; plan `planning/archive/safety/SAFETY_LAYER_V19_PLAN.md`; 33 tests. Independent V16 branch, not combined with V18.

Some targets admitted by the safety wrapper still have current returns in static memory because ordinary dynamic classification did not publish that object. V19 removes only exact current point rows already owned by a qualified freshly updated moving hypothesis, avoiding simultaneous treatment as moving hull and stationary cloud. It requires fresh unique raw ID/scan evidence, existing full-extent and coherent endpoint-translation gates, a measured admission/refresh, a corresponding final age-zero hypothesis, and exact float64 point membership in the current base-memory batch. Points present in any older batch, shared with another cluster, ambiguous or unrelated are retained. There is no radius/grid/tolerance deletion; underlying memory, tracker and policy observation remain unchanged.

Dynamic occupancy and measurement ownership motivate this adaptation (Nuss et al., R19; extended-object association, R14). A mixed wall/target cluster could still give incorrect ownership. The exact gates limit scope but do not certify every removed point as dynamic. Disabling `source_owned_points` restores V16 perception.

**Development probe:** V19 **5/12** (three target, one obstacle, three boundary), matching V16's goals. It activates four times across three cases and changes the command sequence only in DV3-CRS-CV-04, starting at decision 13. A boundary collision becomes an obstacle collision: no rescue. All six lost SAC successes remain. Fresh V16 reproduces all 12 older exact command sequences/outcomes. SAC's 10/12 reference comprises eight earlier matched pilot records and four explicitly historical records; no fresh OFF runs were added to this probe.

**Primary quick-18:** a pre-recorded advancement rule allowed a tied development candidate with no V16 goal loss to enter the small frozen primary comparison; V19 won the recorded tie rule. All **54 runs = 18 x OFF/V16/V19** completed. OFF reaches 14 goals (one target, one obstacle, two boundary); V16 and V19 each 15 (one target, two boundary). Both rescue **TS3:FS-CRP-VAR-023**, retain all 14 SAC successes and fail **TS3:BAS-BO-RE-082**, **TS3:P2-L2-CRS-FIX-17** and **TS3:FS-BO-FIX-019**. The last changes from OFF obstacle contact to filtered target contact, not a rescue.

V19 activates 11 times over four primary cases, transferring 29 point-decision instances, but its **complete issued command sequences are identical to V16 in all 18 cases**. It has no incremental measured benefit and is not promoted. Selection was metadata-only, fixed before outcomes; no candidate change was tuned on these primary results.

The V18/V19 continuation completed **90 new runs**: 36 development (12 x V16/V18/V19) plus 54 primary. No retries or queued evaluation remain. Source: `results/safety_dev/v18_development/report.md`, `validated_development_report/`, `probe12_trace_audit/`, and `results/safety_dev/testset_v3_main/{quick_v19_report,quick_v19_trace_audit}/`.

## 8. Failure-pattern analysis and trigger experiments

### 8.1 What the 1,000 saved TS2 episodes reveal

Development targeted failed TS2 episodes while retaining successful controls. The initial stage was saved-data-only, producing `results/safety_dev/testset_v2_offline/report.md` and `analysis/`.

There are **128 SAC failures: 61 target, 55 obstacle, 12 boundary**. Crossings contribute 76 failures, including 46 of the 61 target contacts. Deployment-layout cases contribute 76 failures among 239; frozen-source cases contribute 52 among 761. All 15 L2 head-on cases fail (12 obstacle, three boundary). These are descriptive concentrations, not causal online trigger rules.

The exact archived OFF/V4 overlap initially available in TS2 was only **735 frozen-source pairs**: SAC 684 goals, V4 680, 19 rescues and 23 lost successes. Those 23 losses comprise 16 target, four obstacle and three boundary contacts. Another 262 preserved-success episodes contain V4 interventions; an intervention in a successful episode is not automatically harmful. Later full V4/V7 branch reconstructions have different coverage and must not replace these initial 735-pair facts.

Successful controls were retrieved using standardized static geometry within the same cell and variant: **151 unique successful cases in 226 pairs for 113 failures**. The remaining 15 failures (L2 head-on) have no successful same-cell control. This adapts the repository's geometric selection method, not propensity matching or a causal estimator.

The original saved episode rows did not contain pre-decision sensor/action histories. Final outcome, minimum range over the entire episode and average speed cannot be used as online inputs without future leakage. Consequently this analysis supplied failure groups and controls, not a validated success/failure classifier.

Historical baseline clarification: the older test-set file `results/test_set/sacs0_bl3/episodes.csv` has **857** goals. Test set v2 replaced 15 initially colliding L1-BO cases; all 15 replacements succeed, giving 872, with the other 985 rows unchanged. This **test-set version 1** is unrelated to **safety-controller V1**. Neither saved baseline supports 895 goals.

### 8.2 First-fire counterfactual branches and a clairvoyant trigger

Phase 1 used **429 cases**: all 150 DV3, all 128 TS2 failures and the 151 successful TS2 controls. SAC succeeds in 262 of them. The main SAC rollout supplies the no-intervention outcome; simulator clones at common first-trigger states supply the filtered branches. Exact-state/RNG cloning is a simulator diagnostic only; it is not an onboard prediction model.

| Phase-1 measure | V4 | V7 |
| --- | ---: | ---: |
| Episodes where the filter fires | 337 | 311 |
| Policy failure among episodes with a fire | 43.6% | 48.9% |
| Fires among policy-failure episodes | 88.0% | 91.0% |
| Fires among policy-success episodes | 72.5% | 60.7% |
| Helpful first-fire branches | 68 | 79 |
| Harmful first-fire branches | 23 | 13 |
| Net goals with ordinary triggering | +45 | +66 |
| Net goals if an oracle forbids harmful first fires | +68 | +79 |

There were 819 phase-1 replay branches. The oracle uses already-known outcomes and is not deployable. It estimates the value of choosing among those particular recorded first-fire alternatives; it is not an upper bound on different trigger timings or a redesigned recovery controller.

Within TS2's 128 failures, V7 rescues 58: 34/76 deployment and 24/52 frozen-source failures. It breaks 9/151 successful controls (six target collisions; three involve reactive/non-compliant labels). Phase 2 adds the remaining 721 SAC successes, yielding the full **904/1,000** V7 result. Even a perfect first-fire chooser among those branches would give **872 + 58 = 930 goals**: preventing 26 lost successes still leaves 70 original failures unrescued. The rescue controller itself, not just the trigger, needs improvement.

`results/safety_dev/trigger_counterfactual/analysis.txt` and `gate.json` preserve phase 1. The CSVs were later expanded to 1,150 scenario rows and 1,378 branches; exclude `set == "ts2_rest"` to reconstruct phase 1. Do not mix the phase-1 percentages with the expanded denominator.

### 8.3 SAC critic gate: suggestive scores, insufficient control evidence

The hypothesis was to gate first intervention using `min(Q1(o, pi(o)), Q2(o, pi(o)))` or its change over the previous four decisions (2 s). The critic is trained for discounted return, not calibrated collision probability; the underlying SAC reference is Haarnoja et al. (R20). Treating its value as a safety gate was a project experiment.

At V7's first fires, median Q is -5.237 for policy failures and -2.910 for successes, but distributions overlap. A fixed hash split separated threshold fitting from validation, searching 50 training quantile thresholds. Chosen V7 gate `Q < -1.9982` yields **33 rescues / 6 losses** on 148 validation first-fire cases, versus **34 / 8** when always acting: only one extra net goal, with one rescue sacrificed to avoid two losses. The trend gate admits all 148 and gives no benefit. V4's critic gate gives 31/9 versus 31/13, four extra net goals in that offline comparison.

These are choices among recorded branch outcomes, **not a closed-loop validation of a live gate**: delaying one intervention changes subsequent states, observations and opportunities. No significance or generalization claim is justified, and no critic gate was integrated. Also, the diagnostic `speed` column used `env.u_body` truth; the Q/Q-trend gates did not use it. Do not describe every recorded feature as deployable simply because the fitted gate used policy-observation features.

## 9. Detailed mechanism cases to carry into the next research session

For readability, audit paths beginning `v10_iterations/` or `v18_development/` in sections 9-10 are relative to `results/safety_dev/`.

### 9.1 CH-HO-CV-073: corrected target extent produces a real rescue

Source: `results/safety_dev/v10_iterations/audits/v14_ch_ho_cv073_mechanism/audit.md`.

OFF reaches its goal in 54 decisions. V14 first overrides at decision 13 and contacts the target at decision 28. The forecast of OFF's actual subsequent commands gives target clearance -0.3608 m; using actual OFF own-position samples still gives -0.3633 m against the forecast target. Own-position error is only about 0.097 m at 4 s and 0.235 m at 8 s. Thus this example points strongly to target representation rather than own motion alone.

The base target center is biased about 0.723 m and the hull heading about 29.14 degrees. Velocity course is about 175 degrees versus true hull heading 179.14, while the fitted axis is about 150. The observed beam is 0.424 m but observed length only 0.357 m versus a known hull length near 1.73 m. Completing the visible bow away from the sensor reduces center error to about 0.0123 m and makes the recorded sequence pass at +0.06685 m. V16 qualifies the observed geometry at decisions 12/13 and the fresh episode reaches the goal with one intervention. This is both a supported mechanism and a measured closed-loop rescue.

### 9.2 DV3-BO-CV-04: another partial-hull rescue

Source: `v10_iterations/audits/bo04_v15_preservation/REPORT.md`.

V15 preserves decision 11 with a searched policy-prefix clearance +0.139726 m despite the parent proposal at -0.42126 m. At decision 12 every prefix fails, best -0.05771 m. OFF's actual successful future receives predicted target clearance -0.2870 m; actual OFF own positions against that target forecast still give -0.33746 m. The target center is biased about 0.651 m and heading about 27 degrees. True **current** target state under CV makes the same own forecast pass at +0.18867 m. True target motion is verified CV over the checked horizon **on the recorded V15 branch**; OFF future target truth was not recorded. Own eight-second error is only about 0.0754 m and 1.028 degrees.

V16 uses observed clusters 10-12 to obtain about 0.049 m center error and 1.64 degree course/heading error, making the recorded continuation pass at +0.09362 m. The fresh V16 episode reaches the goal with seven interventions. This supports the specific geometry change, not universally replacing heading by course.

### 9.3 BAS-NU-CV-070: own-vessel forecast rejects a successful maneuver

Sources: `v10_iterations/audits/{bas070_recorded_policy_future,bas070_initial_state_decomposition,bas_nu_v15_gate}/REPORT.md`.

SAC succeeds; V15/V16 eventually hit the boundary. At the common decision-10 state, rolling out SAC's **actual next 16 recorded commands** predicts static clearance -0.2081 m. Scoring the actual recorded own positions against the same static cloud gives +0.0856 m at 0.5 s samples. This noncausal comparison exposes a forecast error; it is not an online policy rollout or an inter-sample safety proof.

At 2.5 s the ordinary predicted heading is wrong by **15.02 degrees** and position by 0.1726 m; eight-second position error is approximately 0.6488 m. Raw yaw is -11.16 deg/s, true current yaw -10.223, and filtered yaw -6.85. Using true initial u/v/r still leaves **13.61 degrees** of heading error at 2.5 s; raw initialization leaves 14.06. Reducing prediction timestep from 0.125 to 0.025 s changes that error by only 0.0323 degrees. Neither initial-state correction nor finer integration solves the longer-horizon mismatch.

V14's earlier decision-10 prefix had best +0.103391 m, below 0.15 but above the parent's ordinary selection floor +0.0647659. V15's exact-floor rule preserves that action. At the new decision-11 intervention, the parent's CEM backup is +0.150102828, so the extra prefix search is skipped as adequate. However, replaying the unchanged 192-plan search at that state finds **zero hard passes**, best -0.007452623 m. Removing the gate alone therefore does not provide a passing backup. The actual successful SAC continuation at decision 11 is still predicted at -0.154723 m against static geometry.

True plant parameters, servo state and actual delay FIFO were not captured in these traces. The remaining error cannot be fully attributed to parameter randomization versus actual actuator-history mismatch. It is evidence against assuming nominal multi-step prediction is accurate, not identification of the exact physical fault.

### 9.4 BAS-HO-NC-059: policy preference, maneuvering target and stale refresh

Source: `v10_iterations/audits/bas_ho_v16_first_divergence/REPORT.md`.

OFF succeeds in 54 decisions. V16 first differs at decision 20 and contacts the target at 28 after seven interventions; V15 and V16 have identical commands throughout that filtered branch. At first divergence they have the same recorded pre-state and SAC action.

The selected full-astern command, rudder -0.544810, has a checked full-plan clearance of **+0.500991 m**. SAC followed by exactly the same tail also hard-passes at **+0.099951 m**. V7 rejects the latter under its 0.15 m requirement; V10 prefers the larger margin; V15 skips extra search because the parent is already above 0.15. An existing broad-prefix episode at the same state finds **122 hard-passing plans among 192**, best +0.137479, but accepts none at 0.15. This is a concrete unnecessary first override relative to available hard-passing SAC tails. It is not proof that retaining that one action would save the mission: the fresh any-feasible and broader-prefix episode probes still fail.

The known successful OFF future commands receive target clearance **-0.793390 m** with the onboard forecast; true current target state but still CV gives **-0.675899 m**. Replacing own forecasts with actual saved OFF own positions still gives **-0.830506 m** against the onboard target. Static and boundary margins are positive (+0.147592 and +1.866083). Better own yaw or target center alone cannot make that successful continuation pass under the assumed target motion.

At decision 20 the persistent center error is about 0.198 m, while fresh raw fitted center error is about 0.033 m; a partial-beam prior is retained. The measured headings at decisions 13-20 are 132, 128, 124, 120, 116, 112, 108 and 104 degrees, already showing a -8 deg/s turn. On the **filtered branch**, true target heading then jumps from 104.0837 to 185.9683 degrees in 0.5 s. Its position diverges from current-state CV by 0.242 m after 0.5 s and 1.835 m after 2.5 s. Persistent geometry keeps heading 104 for two decisions because refresh rejects oversized clusters; current raw axes are also wrong (124 and 96). By decision 23 the new approximately 186-degree estimate arrives too late.

The OFF trace did **not** save its future target truth. Do not substitute the filtered branch's post-intervention target trajectory as the counterfactual SAC target trajectory: reactive behavior can depend on own-ship actions. A CV-plus-turn union also cannot remove this rejection while it retains the failing CV branch. The problem is selecting/calibrating a defensible target behavior model, not simply adding more worst-case hypotheses.

### 9.5 P2-L1-CRP-FIX-05: target stops and its fitted orientation disappears

Sources: `v10_iterations/audits/fix05_v13_final_audit/REPORT.md`; `v18_development/heading_memory_saved/REPORT.md`.

The older V13 episode lasts 39 decisions with 11 interventions. At decision 29 a new base source 21 coexists with persistent source 4; their center errors are roughly 0.4485 and 0.0974 m. The true target stops at 32 while its old CV hypothesis remains. At 34 the current hull fit loses heading and falls back to noisy course **147.56 degrees**, versus true **115.78** and prior fitted **114**. The same plan appears to pass: weak-brake margin +0.1341 m, fast-brake +0.1542. Retaining the earlier measured hull axis changes those to **-0.2907/-0.2499 m**: the missing-heading fallback turns a failing check into a passing check under the same biased center. That does not establish the plan's true physical safety either way.

True full stationary target state instead gives +0.0071/+0.0871 m, so center error also matters. Heading memory is not a complete oracle fix. At decisions 35-39 no escape is found. The last passing plan was not retained by that preset; an offline check of its hypothetically shifted tail already fails at decision 35, so retention alone is not supported as a rescue. Of 33 audited plans, 28 weak-model passes contained **zero weak-pass/fast-fail** cases, so faster reverse does not explain this isolated passing check.

V18's saved replay holds heading on decisions 34-36 and expires at 37 while leaving synthetic views alone. Yet on fresh V16-derived episodes it activates only at terminal 34 and changes no action. This illustrates the difference between correcting a diagnosed forecast and preventing the earlier decision chain that caused a collision.

### 9.6 DV3-CRS-CV-04: dynamic returns also constrain the boat as static points

Sources: `planning/archive/safety/SAFETY_LAYER_V19_PLAN.md` and `results/safety_dev/v18_development/probe12_trace_audit/REPORT.md`.

At decision 13, raw source 77 is absent from the ordinary published track list but is present as a freshly admitted persistent target. Nine of 58 static snapshot points are exact current returns of that target; the worst static-contact point is about **2.645 m from every true static obstacle**, measured offline only. Removing those nine changes static clearance of the recorded successful SAC tail from **-0.430486 to +0.118128 m**, retaining the other 49 points. The target-CV constraint still fails at **-0.235371 m**.

Thus duplicate occupancy is a real mechanism but not the only failing constraint. V19 removes the qualified points, slightly changes decision-13 rudder from about -0.345242 to -0.338779 under full astern, and ends in an obstacle contact instead of V16's boundary contact. It does not restore SAC's goal. Per-point appearance evidence alone would remove none of those nine returns; the implemented transfer relies on the existing cluster-motion admission and therefore still has association uncertainty.

### 9.7 Initial recoverability and blind-zone examples

Source: `results/safety_dev/development_v6_budget150/offline/blind_zone_oracle_replay/REPORT.md`.

In **DV3-BO-CV-10**, initial inflated-hull gap is only **0.217902 m**. An oracle replay with true geometry and seed-derived plant parameters tests **10,426 constant first commands** (401 rudders x 26 propulsion values); every one contacts by 0.5 s, including immediate straight braking. The first empty actuator delay queue is filled by the initial command, so the audit already allows its immediate arrival at the servo. This is strong evidence of an extremely constrained start, **not a proof over all continuous time-varying controls**. The case is retained in evaluation denominators.

In **DV3-BO-VS-05**, gap is 0.460558 m. There are 401 sampled commands clear through 1.2 s; immediate straight braking remains clear for 3 s with minimum gap 0.135734 m. Waiting for the recorded first decision and then braking at 0.5 s contacts at 0.9 s for all three checked rudders. Initial astern is also excluded by the existing no-traffic candidate gate, and the demonstrated clearance is below the ordinary 0.15 m trigger before other gap subtraction. This shows a potentially actionable earlier-intervention/admission issue, not proof of a successful whole mission.

These examples justify studying admission and recoverability. They do not establish that the current seven V16 failures, the six broken SAC successes, or all hard cases are physically unavoidable. Relevant theory is inevitable collision states (R22), with the same distinction between sampled evidence and an actual inevitability proof.

### 9.8 Complete remaining V16 development-failure inventory

These are all failures from the disjoint 32- and 40-case V16 cohorts, read from their validated `paired.csv` files. “Goal” in OFF marks a lost SAC success. None is excluded from its denominator. Detailed mechanisms are established only for the subset discussed above; the rest remain research cases, not diagnosed inevitabilities.

| Cohort | Namespaced case | OFF outcome | V16 contact |
| --- | --- | --- | --- |
| 32 | `DV3:DV3-HO-VS-01` | Boundary | Obstacle |
| 32 | `TS2:BAS-HO-NC-059` | **Goal** | Target |
| 32 | `TS2:BAS-NU-CV-070` | **Goal** | Boundary |
| 32 | `TS2:CH-CR-CV-059` | Target | Target |
| 32 | `TS2:P2-L1-CRP-FIX-05` | Target | Target |
| 32 | `TS2:P2-L1-CRS-VAR-08` | Target | Obstacle |
| 32 | `TS2:P2-L2-HO-FIX-11` | Obstacle | Target |
| 40 | `DV3:DV3-BO-CV-10` | Obstacle | Obstacle |
| 40 | `DV3:DV3-CRS-CV-04` | **Goal** | Boundary |
| 40 | `DV3:DV3-NT-CV-15` | Obstacle | Obstacle |
| 40 | `TS2:BAS-CR-CV-027` | Target | Target |
| 40 | `TS2:BAS-CR-RE-022` | Target | Target |
| 40 | `TS2:BAS-HO-NC-002` | Target | Target |
| 40 | `TS2:BAS-NU-CV-043` | Boundary | Boundary |
| 40 | `TS2:CH-HO-RE-057` | Obstacle | Obstacle |
| 40 | `TS2:P2-L2-CRS-FIX-19` | **Goal** | Boundary |
| 40 | `TS2:P2-L2-CRS-VAR-14` | Obstacle | Obstacle |
| 40 | `TS2:P2-L2-HO-VAR-14` | Obstacle | Boundary |
| 40 | `TS2:P2-L2-HO-VAR-16` | Obstacle | Boundary |
| 40 | `TS2:P2-L3-BO-FIX-09` | **Goal** | Boundary |
| 40 | `TS2:P2-L3-CRP-FIX-03` | **Goal** | Target |
| 40 | `TS2:P2-L3-CRP-FIX-17` | Obstacle | Target |

## 10. Model audits and rejected approaches that should not be repeated blindly

### 10.1 Observation and actuator diagnostics

An audit of **50 saved controller-episode exposures / 2,837 fresh frames** compared raw and filtered body motion. Pooled frame-level RMSE changed as follows (longer episodes receive more weight):

| Component | Raw | Filtered | Interpretation |
| --- | ---: | ---: | --- |
| Surge, m/s | 0.04987 | 0.02460 | Improves in all 50 exposures. |
| Sway, m/s | 0.04965 | 0.02137 | Improves in all 50. |
| Yaw, deg/s | 1.01315 | 1.82227 | Worsens in 40/50. |

For fresh noninitial frames the observer prior can be reconstructed as `(filtered - 0.3 * raw) / 0.7`; its yaw RMSE is 2.5622 deg/s. This algebra describes the recorded prior contribution, not a fresh independent model validation. Repeated scenarios and correlated frames are not independent trials. Source: `v10_iterations/audits/ego_observer_completed/REPORT.md`. This motivated V17, whose mixed episode result is already reported.

Separate nominal actuator checks on BAS-NU's 78 issued commands found exact copied servo/FIFO bookkeeping, maximum error zero. Comparing nominal servo propagation at plant-like 0.05 s versus 0.125 s gave only about 4.44e-16 rad difference; the nominal 0.730117 s delay rounds to 0.75 s in both representations. No deterministic actuator bookkeeping bug was found in that case.

Conditional mismatches remain: bank rollouts do not represent an enabled bridge command-rate limiter (the relevant BAS trace has it disabled); stale snapshots can propagate u/v/r while holding pose (BAS has no stale frames); a hand-back-start predictor can have incomplete actuator prehistory (these cached cases do not use those starts); and endpoint-clock brake onset first applies at 0.875 s for a nominal 0.75 s threshold. BAS's specific successful future is nonbraking, so those facts do not explain its observed false rejection. Missing true plant actuator/parameter logs limit further attribution.

### 10.2 Brake-response calibration

The V6 diagnostic used **53 training pulses across 13 development cases**, giving 59 scored endpoints, and nine held pulses from two other development cases (CRP-VS-04, CRS-CV-04). Forty-eight training pulses lasted one decision, four two and one three. The fit used raw measured motion and issued commands, with truth only for scoring; some source states came from earlier oracle-controller diagnostics and were therefore selected operating states.

The effective first-decision impulse is about 0.232603 m/s, equivalent to 0.465206 m/s^2 at -24 rpm-units. Held-case true-surge MAE improves **0.208909 -> 0.024333 m/s**; yaw MAE changes **1.187948 -> 1.041396 deg/s**, with mixed trajectories. Zero versus 0.125 s effective delays have nearly equal weighted objectives (176.89 versus 178.67). Other reverse command levels were not observed, and long-duration braking remains extrapolation. The combined calibration/geometry controller scores only 1/8 goals, so calibrated braking stays disabled. Source: `results/safety_dev/brake_calibration/case_validation/`.

### 10.3 Causal scalar yaw-response gain: rejected

Method: fit only `predicted yaw acceleration = g * nominal yaw acceleration` using the first ten past measured transitions and already-issued commands, then freeze before decision 11. No hidden state, future command or outcome selects the gain. This is a diagnostic prediction-error model (R15), not an identified physical parameter. No outcome-driven clipping, added bias or new gain bound was introduced.

On V15's 32 saved exposures, two early-braking traces and one nonpositive gain are excluded, leaving 29 valid fits; the V16 probe contributes five. One-step V15 yaw RMSE improves in 20/29, mean **3.4557 -> 3.2029 deg/s**. But continuous available-horizon heading worsens in 19/28; full-eight-second heading worsens in **12/19**, mean **4.143 -> 6.943 degrees**. V16's full-eight-second heading worsens in 2/4; CH-HO-CV-073 degrades from about 1.484 to 25.070 degrees.

BAS-NU alone looks promising: fitted gain 0.33585 reduces held one-step yaw error 4.60 -> 2.59 deg/s and a local 2.5 s heading error 11.79 -> 0.55 degrees, but clearance remains slightly negative (about -0.0074 m). Combining raw initial yaw with the gain makes one local margin positive, but changes two factors and is not a rescue. The cross-case results reject production integration. Source: `v10_iterations/audits/causal_yaw_response_cohort/REPORT.md`.

### 10.4 Whole-parameter model bank: both selection objectives rejected

Thirty physical vectors were tested: identified, mean of 28 existing bootstrap vectors, and the 28 whole bootstrap vectors. Each candidate reconstructs its own servo/delay history from reset. Selection uses the first ten measured transitions, fixed inverse sensor-variance weighting (0.05 m/s u/v, 1 deg/s yaw) and freezes before decision 11. The finite bank may not contain the randomized plant, which can be blended and perturbed. This is neither an IMM filter nor a calibrated posterior/uncertainty tube.

There are 37 saved exposures (V15 32 + V16 five); excluding two early-braking V15 traces leaves 30 V15 and five V16 eligible episode exposures, each assessed with the same 30-model bank. Continuous validation ends at eight seconds, braking or trace end; denominators differ by available horizon. Future recorded commands are conditional forecast inputs only, not information available during selection. Continuous heading/position rows below use the unchanged filtered initial state.

| Metric (means of per-exposure RMSE) | N | Nominal | One-step-objective bank | Continuous-objective bank |
| --- | ---: | ---: | ---: | ---: |
| V15 held one-step yaw, deg/s | 30 | 3.4714 | 2.7939 | 2.9069 |
| V15 available continuous heading, deg | 29 | 3.2079 | 4.3914 | 3.9737 |
| V15 full-eight-second heading, deg | 20 | 4.0224 | 4.6856 | 4.8152 |
| V15 full-eight-second position, m | 20 | 0.18559 | 0.17528 | 0.21165 |
| V16 full-eight-second heading, deg | 4 | 4.2446 | 5.4462 | 2.4005 |

The first objective improves V15 one-step yaw in 23/30 but worsens full-horizon heading in **12/20** and one-step sway in **28/30**. The second changes only the training objective to a continuous five-second forecast without intermediate measurement reset; it still worsens V15 full-horizon heading in **10/20** and position in **12/20**, although the small V16 probe improves. V15 P2-L3-CRS-FIX-05 heading RMSE becomes 15.7497 degrees from 1.5812. BAS-NU selects bootstrap vector 01 and improves locally, but does not justify cross-case adoption.

The two audits use identical validation endpoints and reproduce identical nominal predictions. Four pure continuous-selector checks passed. **Neither selector is integrated; stop this particular bank/window/weight family rather than repeatedly tuning it to these outcomes.** This does not rule out a properly identified/adaptive uncertainty model; it rules out claiming these particular selectors as an improvement. Sources: `v10_iterations/audits/bootstrap_model_bank_cohort/REPORT.md` and `bootstrap_continuous_bank_cohort/REPORT.md`, with evaluated reproducer scripts and JSON inputs.

### 10.5 Other negative evidence and unintegrated work

Longer horizons, broad course recovery, indiscriminate policy priority, larger hypothesis sets, raw-yaw substitution and locally better fits have not established a general solution. The original V3 source investigation also checked path-rejoin reference, stale plans, throttle preference and room slack; some were real implementation problems, but fixing them did not remove the residual walls.

`src/safety_fast_geometry.py` contains an exact tiled point-clearance implementation with 13 tests, but no reported controller uses it. Treat it as unintegrated performance work, not a new safety result. Unit tests validate arithmetic, gates, state isolation, disabled-option parity and dispatch; they cannot establish collision avoidance by themselves.

## 11. Why it still does not work reliably

The evidence supports several coupled limitations. Calling the layer too pessimistic explains only part of the result: there are also false-safe forecasts and missing hazards.

| Limitation | Concrete evidence | What remains missing |
| --- | --- | --- |
| Own-motion forecast bias | BAS-NU rejects SAC's successful static-obstacle maneuver; true initial state and finer integration leave large multi-step heading error. | A validated multi-step model/uncertainty description, including actuator state and low-speed/reverse behavior. |
| Target extent/heading bias | BO04 and CH-HO geometry repairs yield actual rescues; FIX05 loses a hull axis; BAS-HO persistent center can be worse than fresh center. | A consistent partial-extent estimator and association uncertainty, with reliable adaptation after turns/stops. |
| Target behavior mismatch | CV rejects a successful BAS-HO continuation even with true current target state; large maneuver error follows on the filtered branch. | A defensible behavior model or bounded motion set; valid treatment of reactive counterfactuals. |
| Missing or duplicate obstacle evidence | Three historical V8/V9 never-triggered target contacts; deleted tracks; target returns remaining static in DV3-CRS-CV-04. | Reliable visibility/occupancy/identity handling with calibrated confidence; aggressive deletion can hide real static obstacles. |
| Limited search | Primitive banks fail; CEM rescues some cases but finite samples still miss possible maneuvers. | A distinction between search failure, actual infeasibility and a certified contingency. |
| Margin/preference tradeoff | BAS-HO has a hard-passing SAC tail below 0.15 m; accepting any feasible tail elsewhere loses goals. | Uncertainty-grounded margins and future recoverability, rather than a new threshold fitted to one outcome. |
| Loss of future feasibility | Old plans fail after new observations; bounded `last certificate` and `no escape` survive through the reference lineage. | A terminal safe region/backup and a mechanism preserving its reachability after each admitted action. |
| Physical/time limitations | Steering needs thrust, delay is substantial relative to 0.5 s decisions, close initial states can leave little time; stopping does not prevent another ship striking the ASV. | An explicit operating/admission envelope, latency limits, and mission-level fallback options. |
| Policy distribution shift | Recovery creates slow/off-path/wall-adjacent states, and global rejoin heads through panels. | Validated recovery-to-policy transition; any later retraining must be labeled filter-informed and evaluated separately. |

The implemented fixes have been local and conditional. V16 succeeds where fresh motion and partial-hull geometry satisfy its gates; it intentionally leaves synthetic persistent tracks and uncertain observations unchanged. V18 often acts too late. V19 removes a real static duplication but cannot fix a simultaneously wrong target-CV constraint. These are reasons for limited effect, not evidence that the methods were never executed.

The desired decision “intervene only when SAC would fail and the intervention would help” contains two future counterfactuals. The current controller sees a noisy present/past history and tests a finite modeled backup, so it cannot directly answer either. A policy trained with a roughly ten-second discounted reward horizon also does not provide a calibrated collision-risk certificate through its critic. Saved branch results show that perfect selection among current first-fire branches still leaves many failures because the recovery itself is inadequate.

There is a fundamental information tradeoff: two worlds can have the same current onboard observations but different future target maneuvers. SAC may succeed in one and collide in the other. A deterministic robust filter cannot promise both to leave every eventually successful SAC run untouched and to prevent collision under every indistinguishable admissible future, unless stronger information/behavior assumptions make those futures distinguishable or jointly avoidable. This does **not** mean all observed regressions are necessary; several were demonstrable modeling mistakes and were repaired.

## 12. What could justify a safety guarantee, and what can be claimed now

### 12.1 Current claim

The supported description is an **empirically evaluated predictive intervention layer** with measured rescues and regressions. It is not “100% safe,” not a verified barrier function, and not a guarantee that all generated missions can finish. The name “safety layer” describes its purpose; it does not establish its property.

The finite checks lack several conditions required for a formal guarantee: a state/model uncertainty set known to contain reality, a recoverable admitted initial state, an invariant terminal backup, assured feasibility under input/delay constraints, and continuous or bounded inter-sample collision checks. A higher success rate, zero errors in unit tests or even zero collisions in one finite test set supplies none of those proofs by itself.

Predictive safety filtering as in Wabersich and Zeilinger requires a terminal safe set/controller and appropriate model assumptions; the uncertain version attaches its guarantee to a specified probabilistic model bound. The local project adaptations implement neither the full terminal construction nor that uncertainty argument. [R4, especially Section 4.1, Eq. 5f and Assumption 4.2](https://arxiv.org/html/1812.05506v4#S4.SS1).

Control barrier functions express forward invariance through conditions that must be achievable by admissible inputs. Replacing this code with a QP called “CBF” would still require a valid barrier, feasible control authority and the correct augmented dynamics. [Ames et al., R21](https://authors.library.caltech.edu/records/jnhr0-1ww05). In this ASV, u/v/r, delayed commands and servo state matter; a kinematic distance barrier with instantaneous turning is not automatically valid.

### 12.2 A meaningful conditional mission-safety specification

A prospective guarantee should have the form: **for every admitted initial state/belief in a specified recoverable set, and for every disturbance/target motion within stated bounds, the executed controls keep the inflated own hull disjoint from all modeled hazards throughout the mission or a verified safe abort.** Each term requires evidence or a proof.

The work still needed to support that statement is:

1. Define the operational envelope: sensor coverage/uncertainty, map accuracy, own dynamics including reverse and delay, target acceleration/turn behavior, allowed initial gaps/speeds and computation deadline. Cases outside it require explicit admission rejection or a different system-level response, not a hidden removal from the test denominator.
2. Establish a recoverable region or backup set with a feasible controller. A low surge speed is not a safe set: sway/yaw, map proximity and moving traffic remain. A static geometric A* route at generation does not prove dynamic avoidability.
3. Before admitting SAC's first action, verify that all modeled possible successor states retain a feasible backup. Retain and execute that backup when replanning fails. Repeatedly accepting a nominal plan without maintaining recoverability can defer the escape until too late.
4. Validate inter-sample hull motion and latency. Sampled point clearances and a 0.5 s trace can miss contact between samples. An optimization timeout must have a known safe continuation, not an unchecked nominal action.
5. Separate safety from task completion. A safe wait, reroute or abort may not reach the goal within 90 s. A guarantee of both collision avoidance and goal completion additionally needs reachability/traffic assumptions. Physical stop is not universally safe against another moving ship.

An unavoidable-collision argument requires quantification over admissible controls and future hazards, not failure of 192 sampled plans. The inevitable-collision-state framework is relevant for this distinction. [Fraichard and Asama, R22](https://emotion.inrialpes.fr/fraichard/publications/journals/04-rsjar-fraichard-asama.pdf).

If uncertainty bounds are statistical, the claim must be probabilistic and specify whether risk is per decision or per mission; repeatedly applying a per-step confidence level does not automatically give the same mission confidence. Even zero collisions in 1,000 independent identically distributed trials would only yield an approximately 0.3% one-sided 95% upper failure-probability bound, not a proof of zero risk. These development-enriched, overlapping scenario sets do not meet that simple sampling interpretation.

### 12.3 Research questions for the next phase

The next session should perform an in-depth literature/design investigation against the following concrete questions, rather than continue changing a global trigger threshold:

- **Own dynamics:** what estimator or robust prediction formulation can bound multi-step yaw/position error under randomized parameters, noisy u/v/r and delayed commands? Why do one-step fitting gains fail on continuous forecasts? Evaluate on entire withheld development cases and eight-second errors, not only one-step RMSE. The rejected gain/bank results are required negative controls.
- **Target geometry and motion:** can partial-hull extent, association and motion mode be estimated jointly, with uncertainty that remains useful in confined water? Handle fresh evidence of stopping/turning without unsafe raw-fit switching or eight seconds of stale CV. Preserve the two demonstrated V16 geometry rescues.
- **Policy-aware recoverability:** what certificate can pass the current SAC action with a verified future contingency while avoiding the myopic “any feasible tail” regressions? A learned rollout/value may propose candidates, but it requires independent validity checks; simulator cloning and known future SAC actions are diagnostic oracles only.
- **Backup/terminal design:** does an invariant or finite-mission recoverable set exist for this augmented underactuated/delayed system within an explicit traffic envelope? Consider braking, powered escape, retreat/reroute or mission admission, rather than assuming a stopped boat is safe. Do not repeat the old unsupported “no invariant set exists” assertion.
- **Perception safety:** how should unseen regions, transient track deletion and static/dynamic ownership enter a certificate? V19 illustrates that overconservative double occupancy and unsafe point deletion are competing risks.
- **Evaluation:** freeze the hypothesis, source and selection before testing; preserve all six lost SAC successes as regression cases, existing rescues and no-intervention successes as controls; keep primary TS3 untouched for tuning. Report rescue/loss pairs and collision types, not only net goals or intervention counts.

Potential families to assess include robust predictive filters, backup-CBF/reachability methods and uncertainty-aware extended-object tracking (R4-R5, R14, R16, R21-R22). None is preselected as the solution by this summary. A useful proposal must show which existing failure mechanism it addresses, what assumptions it needs, whether those assumptions hold here, and how it will be falsified on the saved cases before a new campaign.

## 13. Operational constraints, evaluation tools and artifact map

### 13.1 Persistent constraints and current state

- Work in `asv-lidar/static_dynamic_obstacles/`. The separate Paper 2 project `asv-lidar/static_obstacles/` is **read-only**. The main-project helper `tools/tiers/paper2_suite.py` is a different file; its name does not make the whole main project read-only.
- **Do not edit `src/constants.py`, change the SAC checkpoint, rewrite completed ledgers, or normalize existing line endings.** Use constructor options, new source modules and archived runtime settings for experiments.
- Put controller code in `src/`, diagnostics in `tools/diagnostics/safety/`, results in `results/safety_dev/`, plans in `planning/`, and focused tests in `tests/test_safety_*.py`.
- TS2 and DV3 were explicitly authorized for development. **TS3 is now the primary evaluation set**; do not tune from its results. Other held-out sets are not development data without an explicit scope change. TS3's 755-case overlap with TS2 must still be disclosed.
- Keep at most **one or two evaluation processes**, each with one Torch/native numerical thread. Protect other training jobs, including the previously observed `pilot_v4_A` work; no training process was signaled or stopped. Check current process state rather than assume it is still running.
- Historical 100-run/150-run caps and saved-only restrictions applied to their respective stages and were superseded by a later authorization to continue testing. They are documented for accounting, not a reason to retroactively relabel completed runs. This wrap-up itself adds no evaluation and queues nothing.
- No old queue should resume automatically, especially the stopped 8,670-record campaign. Never silently retry/overwrite a failed or partial attempt. Inspect and preserve unrelated concurrent source changes.

### 13.2 Primary test-set-v3 preparation and provenance

Canonical files are `results/test_set/v3/definition.csv`, `definition.json` and `set_v3.0.pkl`. The full frozen selection is `results/safety_dev/testset_v3_main/selection.json`; inventory/verification files record all identities in definition order without regenerating scenarios.

- Definition digest: `b8212ecf2a4f2529b6dec3a17dfaf178dabe0cf6590c606dd0d3427afdabb4bb`.
- Cache SHA-256: `4f2f2bc781750e0086b2a5e250e1f6221d7b3489ae9654a17a9f17afeda2679e`.
- Historical `results/test_set/v3/sacs0_bl3/episodes.csv`: **853 goals, 71 obstacle, 66 target, 10 boundary**. It combines 755 reused TS2 outcomes and 245 new runs, and lacks a frozen source/config/checkpoint archive. Use it as context, not the fresh baseline for a current filtered comparison.
- Quick subset: `quick_v18_case_list.json` (name predates V19 selection), `quick_v18_coverage.json`. Source/encounter grouping plus fixed identity hash selected 18 cases before outcomes; second target variants are used where available. Actual evaluated controllers were **OFF/V16/V19**, despite a prospective sentence still present in the older V18 plan.
- Advancement and choice records: `results/safety_dev/v18_development/primary_advancement_rule.json` and `primary_advancement_decision.json`.

`tools/diagnostics/safety/prepare_testset_v3.py` validates the trusted cached identities without environment construction, reset or scene generation. It accepts identical repeated preparation and refuses changed overwrite. **Preparation does not run episodes.**

`tools/diagnostics/safety/safety_candidate_iteration.py` is the current generic runner. Its default selection is **all 1,000 TS3 cases**, and default OFF/V16 pairing means **2,000 new episode runs**. Do not invoke it assuming a quick smoke test. Explicit `--cases` creates a labeled subset; explicit old `--selection` is needed for TS2/DV3. TS3 outputs go under `results/safety_dev/testset_v3_main/runs/<tag>/`; historical development selections retain `v10_iterations/<tag>/`. A fresh unique tag is required. A STOP marker is checked before an episode; no implicit retries occur.

Each run freezes source/loader/model/settings/selection identities, reserves attempts, explicitly installs and verifies the requested controller class after reset, and stores result JSON and decision traces. Full snapshots are optional and much larger. Source drift is checked before each episode and at completion. The current runner supports conventional `safety_vN` modules through the evaluated versions; native environment/suite dispatch was independently tested through 19.

`report_safety_iteration.py` verifies completion and hashes before pairing, prioritizes a fresh same-run OFF reference, never borrows a TS2 record for TS3 just because the bare ID matches, and distinguishes historical references. Strict cohorts require disjoint cases and compatible controller/options/source/runtime settings. Any permitted archive-inventory difference must be recorded explicitly. Numerical thread settings matter: even deterministic policy inference near thresholds can differ under runtime changes; source/hash identity alone is not sufficient.

The older general test-set runner can reuse results by tag/ID/digest without proving the same actual seed/model hash, defaults to three workers and can rewrite definitions. Avoid it for this controlled safety comparison. Also avoid blanket `pytest`: some old safety tests construct/run episodes. Select reviewed synthetic/unit or report-integrity tests when the task authorizes no new runs.

### 13.3 Essential documents and result directories

The information needed to reason about the work is embedded above. These links provide detailed records/reproduction, not missing conceptual prerequisites:

| Evidence | Primary artifact |
| --- | --- |
| Historical chronology / original constraints | [SAFETY_LAYER_NOTES.md](archive/safety/SAFETY_LAYER_NOTES.md) |
| Early V2/V3 experiments | [V2 plan](archive/safety/SAFETY_LAYER_V2_PLAN.md), [V3 plan](archive/safety/SAFETY_LAYER_V3_PLAN.md), `results/safety_v2_dev_*.csv` |
| V4 observer/perception and oracles | [V4 plan](archive/safety/SAFETY_LAYER_V4_PLAN.md), `results/safety_dev/final_fullset_summary.csv` |
| Final stopped sweep | [Final stopped report](../results/safety_dev/v4_stopped_full_sweep_report.md) |
| V5 variants / original inventory | [V5 plan](archive/safety/SAFETY_LAYER_V5_PLAN.md), [quick report](../results/safety_dev/quick_v5_budget100/runs/policy_feedback_v1/report.md) |
| V6 149-attempt accounting | [V6 report](../results/safety_dev/development_v6_budget150/report.md), [V6 plan](archive/safety/SAFETY_LAYER_V6_PLAN.md) |
| TS2 failure analysis / initial V7 | [Saved-data report](../results/safety_dev/testset_v2_offline/report.md), [failure-pattern report](../results/safety_dev/testset_v2_offline/analysis/report.md), [V7 plan](archive/safety/SAFETY_LAYER_V7_PLAN.md) |
| Branch/critic studies | `results/safety_dev/trigger_counterfactual/`, [V8 plan](archive/safety/SAFETY_LAYER_V8_PLAN.md) |
| Full V8 reconstruction and fallback inventory | [V8 saved audit](../results/safety_dev/v8_followup_offline/report.md), `audit/all_1150.csv`, `audit/summary.json` within that directory |
| Fresh OFF/V8/V9 pilot | [Pilot report](../results/safety_dev/v9_paired_pilot/report.md), [V9 plan](archive/safety/SAFETY_LAYER_V9_PLAN.md) |
| V10-V17 campaign | [Campaign report](../results/safety_dev/v10_iterations/report.md), [V10-V15 history](archive/safety/SAFETY_LAYER_V11_PLAN.md), [V16 plan](archive/safety/SAFETY_LAYER_V16_PLAN.md), [V17 plan](archive/safety/SAFETY_LAYER_V17_PLAN.md) |
| Strict 32/40-case comparisons | [V16 32-case report](../results/safety_dev/v10_iterations/reports/motion_axis32/report.md), [40-case report](../results/safety_dev/v10_iterations/reports/v16_broader40_paired/report.md) |
| V18/V19 latest 90 runs | [Latest report](../results/safety_dev/v18_development/report.md), [V18 plan](archive/safety/SAFETY_LAYER_V18_PLAN.md), [V19 plan](archive/safety/SAFETY_LAYER_V19_PLAN.md), `final_verification.json` in latest report directory |
| Primary protocol and 18-case results | [Protocol](SAFETY_EVALUATION_PROTOCOL.md), [TS3 README](../results/safety_dev/testset_v3_main/README.md), [fresh quick report](../results/safety_dev/testset_v3_main/quick_v19_report/report.md) |

The original scratch diagnostics remain `dev_eval.py`, `trace_case.py` and `trace_dead_ahead.py` under `tools/diagnostics/safety/`. They can run episodes: do not mistake them for saved-only analysis. Saved-only tools include the appropriately named `audit_saved_policy_future.py`, `audit_saved_policy_initial_state.py`, `audit_ego_observer.py`, `audit_causal_yaw_cohort.py`, `audit_bootstrap_model_bank.py`, `audit_bootstrap_continuous_bank.py` and `audit_heading_memory_saved.py`. Consult their CLI/default inputs before using them; archived `reproducer_evaluated_bytes.py` files preserve exact evaluated diagnostics.

### 13.4 Tests, source lineage and storage maintenance

Test files are in `tests/`; important newer suites cover prefix search/requests, track persistence/fallback/consistency, motion-axis geometry, yaw observer, heading memory, source-point ownership, strict reports, selection and native dispatch. Counts quoted across plans overlap and should not be summed as unique tests. V16/core/report checks, subsequent 23 yaw checks, and latest 26 V18 + 33 V19 + 32 native-dispatch checks all passed in their recorded stages. Unit checks do not replace an episode comparison or a safety proof.

Native-dispatch integration sometimes happened after a candidate's evaluation freeze. The runner already installed the exact candidate explicitly; source/newline integration audits record the later native changes. Use each run's archived evaluated source, not just the current working-tree version number, to reproduce its behavior. The project contains other work and uncommitted changes; do not reset the tree to make it look clean.

Disk use also had to be reduced. Completed trace JSONL dominated safety-result storage. Instead of deleting evidence or old modules needed by inheritance, `compact_safety_artifacts.py` applied transparent NTFS compression to **585 completed traces across 24 runs**: the 96-run V9 pilot, 399-run campaign and 90 latest runs. It reclaimed **974,323,274 allocated bytes (0.907 GiB)**, with every before/after SHA-256 identical and all paths/source archives/caches/results retained. **Zero files were deleted.** A 28,852-byte bytecode deletion preflight stopped on OneDrive reparse attributes, leaving those files intact.

Maintenance audits are under `results/safety_dev/maintenance/ntfs_compact_20261003T111517Z/` (495 old traces; 945,003,157 bytes reclaimed) and `ntfs_compact_20261003T115342Z/` (90 new traces; 29,320,117 bytes). Hashes were also checked against prior reports. Allocation savings are not the same as net whole-drive free-space change because new files and other jobs also use disk. Fifteen synthetic maintenance guard tests passed. Do not delete old controller modules merely because a newer version exists: they remain in the inheritance/import chain.

### 13.5 Known documentation traps

- “895/1,000 SAC” is unsupported by the audited saved versions; TS2 is 872, old test-set-v1 is 857, historical TS3 is 853. These are different scenario sets.
- An old V7/V9 header saying “unmeasured” refers to its implementation stage. The later branch study/fresh pilot supersedes it.
- The final stopped-sweep report supersedes the earlier cancellation snapshot.
- V12/V13 are sibling experiments; V17/V18/V19 are separate V16 branches. Do not claim V19 includes all prior version ideas.
- V4's dual-braking model, V5's many switches, V6 calibration/history, V9 turn hypotheses and V16 any-feasible-policy option exist in code but are not all enabled in the reported reference.
- The 12-case probe is deliberately regression-heavy. Its 5/12 result is not the overall V16 success rate.
- The filename `quick_v18_case_list.json` refers to the frozen primary selection; actual tested candidate is V19. V18 has no primary-18 result.
- The 1,150-row V8 reconstruction, 1,378 counterfactual branches, 8,670 planned sweep records, 585 compressed traces and 1,000 primary cases are different quantities. There is no justified grand total of independent trials obtained by adding them.
- A diagnostic oracle/current true state, actual future recorded SAC controls and post-intervention target truth have different meanings. None can silently enter runtime action selection.
- “No invariant set exists,” “no sampled escape,” “100% on a subset,” “last certificate,” and “coupled set” must not be inflated into mathematical impossibility, guaranteed safety or physical field validation.

## 14. Method bibliography and exact scope of attribution

The R-numbers used above map to primary papers or official sources below. Attribution distinguishes a method actually adapted from literature, a retrospective architectural relationship, and an engineering choice made in this project. No citation should be read as transferring a theorem to code that does not meet its assumptions.

**R1. International Maritime Organization.** [Convention on the International Regulations for Preventing Collisions at Sea, 1972 (COLREGs)](https://www.imo.org/en/about/conventions/pages/colreg.aspx). Official general reference for the Rules 8/17 rationale recorded by V1. This summary neither interprets every legal duty nor claims compliance certification.

**R2. Fox, D.; Burgard, W.; Thrun, S. (1997).** [The Dynamic Window Approach to Collision Avoidance](https://publications.ri.cmu.edu/the-dynamic-window-approach-to-collision-avoidance). Related sampled admissibility/forward-simulation architecture for the classical comparator and early V2; the ASV recovery bank is an adaptation, not the original velocity-space algorithm.

**R3. Bastani, O. (2019; revised 2020).** [Safe Reinforcement Learning with Nonlinear Dynamics via Model Predictive Shielding](https://arxiv.org/abs/1905.10691). Learning action followed by backup motivates V2/V3's architectural relationship and V5's one-decision/policy-preserving experiments. The project does not implement the paper's complete safety argument. Bastani is the sole author of this paper; older “Bastani et al.” wording in plans is imprecise.

**R4. Wabersich, K. P.; Zeilinger, M. N. (2021 version).** [A predictive safety filter for learning-based control of constrained nonlinear dynamical systems](https://arxiv.org/abs/1812.05506v4); [Section 4.1 full text](https://arxiv.org/html/1812.05506v4#S4.SS1). First-input preservation, feasible backup and terminal-safe-set requirements inform V3, V5, V7, V9/V10 and later prefix methods. Local same-tail rules, clearance rankings and trigger gates are project choices; uncertainty tubes and robust terminal construction are absent.

**R5. Chen, Y.; Jankovic, M.; Santillo, M.; Ames, A. D. (2021).** [Backup Control Barrier Functions: Formulation and Comparative Study](https://arxiv.org/abs/2104.11332). Forward integration of a backup feedback controller motivates V3/V5 backup ideas. Re-evaluating feedback rudder during a sampled forecast does not itself implement a backup CBF.

**R6. Luenberger, D. G. (1971).** [An introduction to observers](https://doi.org/10.1109/TAC.1971.1099826); [primary paper copy](https://lab.prd.vanderbilt.edu/taha/wp-content/uploads/sites/154/2017/10/Observers_Original_Paper.pdf). Model plus measurement-error correction, Section II.B, relates retrospectively to V4's observer and motivates V17's isolated yaw-correction ablation. No observer pole-placement, covariance propagation or proved gain is claimed.

**R7. Hornung, A.; Wurm, K. M.; Bennewitz, M.; Stachniss, C.; Burgard, W. (2013).** [OctoMap: An efficient probabilistic 3D mapping framework based on octrees](https://doi.org/10.1007/s10514-012-9321-0); [author manuscript](https://www.arminhornung.de/Research/pub/hornung13auro.pdf). Retrospective relationship to V4's finite-ray free-space evidence. The project directly clears 2D remembered points; it does not use OctoMap, an octree or its probability update.

**R8. Wabersich, K. P.; Zeilinger, M. N. (2023 journal publication).** [Predictive control barrier functions: Enhanced safety mechanisms for learning-based control](https://doi.org/10.1109/TAC.2022.3175628); [preprint](https://arxiv.org/abs/2105.10241). Soft-constrained recovery motivates V5's constraint-deficit ranking. The implementation is not a predictive CBF with the paper's terminal/stability construction.

**R9. Eriksen, B.-O. H.; Breivik, M.; Wilthil, E. F.; Flaaten, A. L.; Brekke, E. F. (2019).** [The branching-course model predictive control algorithm for maritime collision avoidance](https://doi.org/10.1002/rob.21900); [preprint](https://arxiv.org/abs/1907.00039). Course/speed trajectory branching motivates V5 feedback candidates. Their number/directions in this project are engineering choices.

**R10. Fossen, T. I.; Pettersen, K. Y.; Galeazzi, R. (2015).** [Line-of-Sight Path Following for Dubins Paths with Adaptive Sideslip Compensation of Drift Forces](https://doi.org/10.1109/TCST.2014.2338354); [accepted manuscript](https://backend.orbit.dtu.dk/ws/files/129469382/FossenPettersenGaleazzi2014.pdf). V5 uses the course/heading/sideslip kinematic relationship, not the full adaptive law or its stability proof.

**R11. Zheng, L.; Yang, R.; Wu, Z.; Pan, J.; Cheng, H. (2022 version).** [Safe Learning-based Gradient-free Model Predictive Control Based on Cross-entropy Method](https://arxiv.org/abs/2102.12124v3). Sampling and elite-refitting inspire V6 trajectory search and later policy-prefix search. The project does not inherit that paper's GP, CLF/CBF or probabilistic guarantees.

**R12. Yoon, D. J.; Tang, T. Y.; Barfoot, T. D. (2018).** [Mapless Online Detection of Dynamic Objects in 3D Lidar](https://arxiv.org/abs/1809.06972). Free-space consistency as motion evidence informs V6 provisional admission and V11 persistence. The project uses existing 2D finite-ray evidence, not the complete 3D algorithm.

**R13. Zhang et al. (2017).** [Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners](https://publications.ri.cmu.edu/efficient-l-shape-fitting-for-vehicle-detection-using-laser-scanners); [DOI](https://doi.org/10.1109/IVS.2017.7995698). Existing hull-fit geometry underlies V6/V11 and V16's extent checks. Known-hull completion and motion gates are additional project rules.

**R14. Granstrom, K.; Baum, M.; Reuter, S. (2016).** [Extended Object Tracking: Introduction, Overview and Applications](https://arxiv.org/abs/1604.00970). Separating object state/extent from measured points and accounting for association motivates V6 geometry, V12-V16 representation choices, V18 heading memory and V19 ownership. These deterministic wrappers are not the paper's full extended-object estimators.

**R15. Ljung, L. (2002; linked technical report 2001).** [Prediction Error Estimation Methods](https://doi.org/10.1007/BF01211648); [technical report](https://liu.diva-portal.org/smash/get/diva2:316694/FULLTEXT01.pdf). Past measured input/output prediction-error fitting motivates V6 effective braking calibration and the rejected scalar-yaw/physical-bank diagnostics. It does not validate their identifiability, bank coverage or control safety.

**R16. Hsu, K.-C.; Hu, H.; Fisac, J. F. (2023 preprint; 2024 publication).** [The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems](https://arxiv.org/abs/2309.05837). Monitor/intervention separation informs V7's shadow monitor and the prospective architecture discussion. The one-second diagnostic horizon/persistence labels were project choices.

**R17. Li, X. R.; Jilkov, V. P. (2003).** [Survey of maneuvering target tracking. Part I: Dynamic models](https://doi.org/10.1109/TAES.2003.1261132). Constant-turn kinematics inform V9's optional target ensemble. Three hand-sampled motions are not an IMM estimator, calibrated reachable set or a guarantee.

**R18. Johansen, T. A.; Cristofaro, A.; Perez, T. (2016).** [Ship Collision Avoidance Using Scenario-Based Model Predictive Control](https://torarnj.folk.ntnu.no/colregs_cams.pdf); [DOI](https://doi.org/10.1016/j.ifacol.2016.10.315). Multiple possible obstacle-motion scenarios, particularly Sections 3.3-3.4, motivate the optional V9 target hypotheses. This option remains disabled in the evaluated reference.

**R19. Nuss et al. (2016).** [A Random Finite Set Approach for Dynamic Occupancy Grid Maps with Real-Time Application](https://arxiv.org/abs/1605.02406). Dynamic occupancy and persistence motivate V11-V14 views and V19's distinction between stationary constraints and moving-target prediction. No PHD/MIB estimator or probabilistic ownership bound was implemented.

**R20. Haarnoja, T.; Zhou, A.; Abbeel, P.; Levine, S. (2018).** [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290). Background for the nominal learner/critic. Using critic magnitude or trend as a first-fire safety gate was a separate, unvalidated project hypothesis.

**R21. Ames, A. D.; Xu, X.; Grizzle, J. W.; Tabuada, P. (2017).** [Control Barrier Function Based Quadratic Programs for Safety Critical Systems](https://authors.library.caltech.edu/records/jnhr0-1ww05); [preprint](https://arxiv.org/abs/1609.06408). Used to explain forward-invariance/input-feasibility requirements for a prospective guarantee, not to claim a QP/CBF was implemented in V1-V19.

**R22. Fraichard, T.; Asama, H. (2004).** [Inevitable Collision States: A Step Towards Safer Robots?](https://emotion.inrialpes.fr/fraichard/publications/journals/04-rsjar-fraichard-asama.pdf); DOI `10.1163/1568553042674662`. Relevant to admission/recoverability and the difference between sampled failure and a true inevitability argument. The project's blind-zone audit does not compute a proved complete inevitable-collision set.

**R23. Fossen, T. I.; Breivik, M.; Skjetne, R. (2003).** [Line-of-Sight Path Following of Underactuated Marine Craft](https://doi.org/10.1016/S1474-6670(17)37809-6); [author's publication record](https://www.fossen.biz/php/research/path_following.php). Shared LOS guidance used by classical comparators and the initial V3 path-rejoin reference. Path following alone cannot route around panels on the reference path.

## 15. Starting brief for the next phase

> Read this document as the complete safety-controller history. Research a defensible way to improve collision avoidance for the unchanged SAC baseline-3/3M ASV policy while preserving certifiably safe SAC actions. Start from V16 as the reference, not V19 as an assumed improvement. Address nominal multi-step own-dynamics error, partial/maneuvering-target estimation, and loss of backup feasibility; explain the assumptions required for any guarantee. Use the saved BAS-NU, BAS-HO, FIX05, BO04, CH-HO and DV3-CRS cases as falsifiable mechanism tests, retain all six policy-success regressions and prior rescues, and do not tune on primary TS3 outcomes. Distinguish diagnostic truth/future commands from deployable information. Propose and cite a method before implementing a new candidate, and state how it differs from the rejected approaches above. Respect the source, benchmark, training-process and disk-preservation constraints in section 13.

## 16. Continuation, 2026-10-10: saved-trace cascade audit and V20 design plan

This section is appended; sections 1-15 are unchanged. **No episode was run, no controller was added or promoted, and no test-set outcome was read.**

A read-only audit of the saved V16 traces of the 32- and 40-case development cohorts (72 cases, 3,801 decisions) labels each decision by V16's recorded branch. Two branches issue commands that V16's own checker did not pass at that decision: `last certificate` (171 decisions) and `no escape` (213). **Twenty-one of the 22 V16 contacts occur immediately after one or more such unchecked decisions** (18 `no escape`, 3 `last certificate`); the remaining contact is the DV3-BO-CV-10 blind-zone start. The unchecked run before contact lasts 1-15 decisions (median 7). Braking, coming to rest and `last certificate` phases are about as frequent in the 19 rescues as in the six lost SAC successes, and SAC with the proposal's own tail hard-passes the nominal checker at V16's first intervention in 2 of 6 losses and 8 of 19 rescues. The first intervention therefore cannot be relaxed without also removing rescue interventions. This is an association; the audit does not show that any contact was avoidable. Source: [audit report](../results/safety_dev/v20_development/phase1_saved_cascade_audit/REPORT.md), produced by `tools/diagnostics/safety/audit_v20_saved_cascades.py`.

Two model facts were also checked from the identified equations: rest with zero propulsion and zero rudder angle is an exact equilibrium for every parameter vector, so rest is an invariant set for static hazards in the simulator (which has no environmental forcing); a deflected rudder held at rest still turns the hull through the minimum inflow floor (about 51 degrees in 30 s at 35 degrees). The old assertion that no invariant set exists should be read as: no invariant set safe against arbitrary target behavior has been constructed.

The resulting design is recorded in the [V20 plan](SAFETY_V20_PLAN.md): V16 keeps every decision it can check; a committed contingency that brings the vessel to rest with the rudder centred is certified at every decision against static points, signed boundary containment and a declared constant-velocity target contract, using calibrated forecast-error allowances; certified commands replace V16's two unchecked branches only. The plan states its conditional claim, assumptions, offline gates and pre-registered probe predictions, and awaits approval before implementation.

The constants file hash quoted in section 2.3 refers to the file before a comment-only edit on 2026-10-04; the current file differs only in comment, docstring and error-message text, with an identical syntax tree once string literals are blanked.

## 17. Continuation, 2026-10-10: V20 phase 2 (gates, development episodes, decision)

This section is appended; sections 1-16 are unchanged. Full record: [V20 plan](SAFETY_V20_PLAN.md) sections 13-16 and the [V20 development results](../results/safety_dev/v20_development/README.md). **No controller was promoted; V16 remains the reference. Test set v4 was authorized but not run for V20, because no configuration met the pre-registered development condition. No test-set outcome was read.**

**Runtime.** Fresh episodes ran in a Python 3.11.17 environment (numpy 1.26.4, torch 2.2.1, gymnasium 0.29.1, stable-baselines3 2.3.2) with one thread per process. Fresh V16 reproduces the historical probe outcomes but not bit-identical commands, so all pairing uses fresh OFF and V16 runs from the same environment.

**Offline gates (saved V16 states only).**
- G1: held-out own-ship forecast coverage passes for every horizon a stop contingency uses.
- G2/G3, 3,569 decisions in 68 development episodes:
  - The default configuration, with calibrated allowances, fails: one-step survival with a target present is 0.805, and 3 of 21 contact precursors are certified at their last checked decision.
  - The nominal option, without allowances, meets the rule narrowly: survival 0.920 and 0.940, and 11 of 21 precursors.
  - Only 11 % of V16's unchecked decisions have any certified option.
  - At 9 of the 11 certified precursors the commitment fails its recheck exactly when V16 turns unchecked.
- Recheck failures split into small own-state errors against thin slack (104 of 194) and target-estimate updates (58).
- A commit margin of 0.127 m raises survival to 0.965 and 0.969.
- Constant-velocity target forecasts cover truth 68 % of the time at 0.5 s.

**Episodes (fresh, paired by test ID, seed and scenario digest).**

| Configuration | Cases | Goals (V16) | Contacts (V16) | Timeouts | Rescued / lost SAC vs OFF (V16) |
| --- | ---: | ---: | ---: | ---: | --- |
| Nominal V20 | 12 | 3 (5) | 7 (7) | 2 | 0 / 7 (1 / 6) |
| Revision 1: gatekeeper, extended tails, margin | 12 | 2 (5) | 7 (7) | 3 | 0 / 8 (1 / 6) |
| Revision 2: traffic-gated gatekeeper and stop | 12 | 6 (5) | 5 (7) | 1 | 1 / 5 (1 / 6) |
| Revision 2 | 60 | 40 (45) | 16 (15) | 4 | 16 / 3 (18 / 0) |
| Revision 3: no uncertified stops | 72 | 46 (50) | 24 (22) | 2 | 16 / 7 (19 / 6) |

Contact types for revision 3 on the 72 cases are obstacle 8, boundary 7 and target 9; V16 has 6, 7 and 9, and OFF has 16, 3 and 16 with 37 goals.

**Mechanisms.**
- Uncertified stops turn some boundary and target contacts into timeouts. They also let crossing targets strike the stopped vessel, and let a braked hull drift or spin into walls. Rest is invariant only for an exactly stationary hull, and braked turns are not.
- Without a target allowance, contacts follow certified decisions. With the allowance, certificates are rarely available.
- Even certified interventions alone (revision 3) perturb V16 and SAC into new contacts more often than they recover failures.

**Decision: redirect.** The stop-terminal backup is not pursued further. Plan section 16.3 names the directions:
- a target estimator with calibrated short-horizon forecasts;
- a moving, target-safe terminal set;
- closed-loop counterfactual screening of interventions before cohort runs.

New code is confined to `src/safety_v20*.py`, its tests and V20 tools. V16, the environment, constants, checkpoint and configuration are unchanged.
