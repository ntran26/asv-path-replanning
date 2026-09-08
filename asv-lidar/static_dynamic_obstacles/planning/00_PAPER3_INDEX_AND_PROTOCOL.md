# Paper 3 — Index and Experiment Protocol

**Revision 2** — repositioned to two-vessel encounters with static obstacles.
Supersedes the multi-vessel version of this document set.

**Scope:** Path following with static obstacles and one dynamic target vessel,
COLREGs-compliant, in a narrow waterway.
**Predecessors:** ICMCR 2026 conference paper (SAC vs PPO, static); MDPI *Drones*
journal paper (Paper 2 — LiDAR sector pooling, staged curriculum, sim-to-field).

---

## 1. How to use this set

Six documents plus a draft skeleton. This one is the anchor; the others are
self-contained handovers intended to be opened in **separate threads**.

| # | Document | Target surface | Depends on |
|---|---|---|---|
| 00 | `00_PAPER3_INDEX_AND_PROTOCOL.md` | — (anchor) | — |
| 01 | `01_PERCEPTION_AND_OBSERVATION.md` | Claude Code | 02 (classifier definition) |
| 02 | `02_REWARD_AND_COLREGS.md` | Claude chat, then Code | — |
| 03 | `03_ENVIRONMENT_AND_TARGETS.md` | Claude Code | 01 (tracker interface) |
| 04 | `04_SCENARIOS_AND_EVALUATION.md` | Claude chat, then Code | 03 (target behaviours) |
| 05 | `05_VESSEL_MODEL_AND_SIM2REAL.md` | Claude chat + field work | — (parallel track) |
| — | `PAPER3_DRAFT_SKELETON.md` | Cowork | all |

Also carry `PROJECT_BRIEF.md` into every thread — file contents do not persist across
Claude conversations.

**Work order.** `02` moved to the front in this revision: the Rule 9 precedence table is
now the deliverable that gates the encounter classifier, the reward terms, and the
width-sweep design. Then `05` in parallel (basin booking lead time), then `01 → 03 → 04`.

---

## 2. Decision log

### Scope decisions (Revision 2)

| # | Decision |
|---|---|
| S1 | Two-vessel encounters. One dynamic target plus up to 3 static obstacles. `N_max` kept as a config parameter so multi-vessel extension costs a retrain, not a redesign |
| S2 | COLREGs scope: Rules 13–16, with Rule 9 governing precedence and Rule 8 governing action quality |
| S3 | Own ship gives way in **all** crossing encounters, justified by **Rule 9(b)** — not Rule 18. The Rule 18 route fails because own ship and target are similarly sized model vessels |
| S4 | Five encounter classes: none, head-on, crossing, overtaking, being overtaken |
| S5 | Rule 17(a)(i) passive course-keeping retained for being-overtaken. Active release under 17(a)(ii) is out of scope and moved to future work |
| S6 | Observation reduced to ≈56 dimensions |

### Carried forward

| # | Decision |
|---|---|
| D1 | Targets constant-velocity in training; reactive and non-compliant in evaluation only |
| D2 | COLREGs via reward terms **plus** an explicit encounter-class feature — not shaping alone, not a separate safety layer |
| D5 | Channel boundary from the map via virtual raycast; LiDAR `c_t` reserved for obstacles |
| D6 | Raw LiDAR 360°/720 beams/0.5°; pooled `c_t` forward-biased to ±135°, ~27 non-uniform sectors |
| D7 | No warm start from Paper 2. Train from scratch; Paper 2 SAC is a frozen baseline only |
| D8 | Imazu dropped (open-water, scale-incompatible) |
| D9 | Evaluation generated in two tiers plus an external named benchmark |
| D10 | Reward and observation both fully redesigned, not patched |

### Superseded

| # | Was | Now |
|---|---|---|
| D3 | Fixed 3 slots + mask, shared encoder, DeepSets flag | One target slot + presence bit. Slot machinery kept parameterised but not exercised |
| D4 | Slots assigned by track ID, sorted by CRI | Single track; CRI sorting moot. Track-ID persistence still used for the one slot |

### Resolved open items

| # | Question | Resolution |
|---|---|---|
| O1 | External named benchmark | **Adopted** — Waltz & Okhrin "Around the Clock", 24 single-ship cases. The two-vessel scope makes it an exact fit rather than an adaptation |
| O2 | Give-way only vs reciprocity | **Give-way only**, via Rule 9(b) |
| O3 | Sim-to-real as RQ4 or separate paper | **Retained as RQ4** — domain randomisation ablation evaluated in the field |

| O4 | Corridor dimensions | **Resolved** — simulation matches basin, max width 10 m (20 B). The open-water "Around the Clock" variant supplies the unconfined reference case, so the sweep covers degrees of confinement only |
| O5 | Barrier vs software gating | **Resolved — software gating**, geometric against the pool polygon. A physical barrier would occlude the facility-wall features that are the only localisation reference. Gating is also mandatory rather than optional because operators standing on the deck sit at scan height and would otherwise be tracked as targets |
| O6 | Ground-truth instrumentation | **Reframed** — no external instrumentation. Scan-to-map registration against surveyed facility geometry, validated by static tests and closed-loop drift. A software deliverable, not a purchase; removes the longest lead time in the project |
| — | Ship domain geometry | **Resolved (provisional)** — compressed asymmetric: 2.0·Lpp ahead, 1.0·Lpp astern, 0.75·Lpp abeam. Final values derived from the identified turning circle in 05 |

| IMU | **Confirmed — will be added.** Removes the yaw-rate observability constraint and, via the accelerometer, rescues surge measurement. Specification in 05 §4.7 | 05 |
| Precedence | **Structure resolved** (02 §3.2). Rule 9 constrains the space; Rule 8(e) supplies the action when space is unavailable. Width thresholds remain an output of Study 1 | 02 |
| Boundary conflict | **Resolved** — when compliance would push the vessel into the boundary, slacken speed or stop under Rule 8(e). Boundary stays a hard constraint | 02 |
| Overtaking | **Resolved** — overtake whenever geometry allows, passing to port of the target (follows from 9(a)). Fallback: hold astern at reduced speed | 02 |
| Head-on band | **Resolved** — ±10°, widened from the source value of ±5° | 01 |
| Propulsion | **Resolved** — authority widens; Rule 8(e) requires it | 03 |

### Still open

| # | Question | Owner |
|---|---|---|
| Reflectivity | Measure black-wall LiDAR return rate from retained logs before committing to continuous-wall registration. **A measurement, not a decision** | 05 |
| Compute | Wall-clock estimate per run in the new environment, then the final comparator list. **A measurement, not a decision** | 04 |
| Width thresholds | Output of the Study 1 sweep | 04 |
| — | Reverse thrust capability of the vessel | 03 |
| — | Progress-penalty carve-out design and gating for Rule 8(e) | 02 |
| — | Head-on band width (source value ±5° is narrow) | 01 |
| — | Whether recurrence is added for occlusion | 01 |
| — | Compute estimate → final comparator list | 04 |
| — | Target vessel platform and repeatability protocol | 05 |

---

## 3. Novelty and study structure

Dropping both Rule 17 active release and multi-vessel left the paper close to
"Paper 2 plus one moving obstacle". The rebuilt position:

| | Claim |
|---|---|
| **N1** *(headline)* | COLREGs compliance driven by target state estimated entirely from onboard 2D LiDAR rather than AIS or a simulation oracle, with perception noise characterised from field logs and injected in training |
| **N2** | Rule 9 precedence over Rules 13–16 where channel width makes the open-water manoeuvre inadmissible |
| **N3** | Full sim-to-real pipeline — system identification, domain randomisation over identified uncertainty, field validation |

Three studies replace the lost depth, all achievable with one moving target:

| | Study | Owner | Cost |
|---|---|---|---|
| **Study 1** | Channel-width sweep, open-water-equivalent down to where the Rule 14 alteration no longer fits | 04 | Simulation only |
| **Study 2** | Perception degradation — pose drift, detection dropout, occlusion duration, velocity noise | 01 / 04 | Simulation only, no basin time |
| **Study 3** | Domain randomisation ablation evaluated in the field (RQ4) | 05 | Basin time |

**Two-vessel justification belongs in the problem formulation, not the limitations
section.** Rules 13–16 are formulated pairwise; confined geometry precludes simultaneous
close-quarters conflicts so encounters are sequential; and every reported behaviour
becomes physically reproducible. It is a scope decision defended by the structure of the
regulations and the geometry of the domain, not a constraint conceded after the fact.

---

## 4. Cross-cutting protocol

### 4.1 Seeds and statistics

- **Five** independent training seeds for every reported configuration (Paper 2 used
  three and was criticised for it)
- Bootstrap 95% confidence intervals on all aggregate metrics
- Report full seed spread, not only the mean
- State that five seeds is a limited estimate of training variability

### 4.2 Metric set

*Task:* success rate; collision rate separately for static obstacle / boundary / target
vessel; RMS and maximum cross-track error; path length ratio; action smoothness.

*COLREGs:* violation rate per encounter class; minimum CPA distribution as a CDF, not a
mean; ship-domain intrusion rate and depth; time to first evasive action (Rule 8(a));
magnitude of first evasive action (Rule 8(b)); course-keeping stability while being
overtaken (Rule 17(a)(i)); side-of-passing correctness.

*Perception (new in Rev 2):* track acquisition range; classification latency and
stability; velocity estimate error; occlusion duration. These carry N1 and Study 2.

**Aggregation is a design decision, not a reporting detail.** Decide the presentation
format before running the campaign; a per-axis table or Pareto view is safer than a
weighted scalar.

### 4.3 Ablation matrix

|  | Encounter feature OFF | Encounter feature ON |
|---|---|---|
| **COLREGs reward terms OFF** | Avoidance only | Told, not rewarded |
| **COLREGs reward terms ON** | Learned from kinematics | Full method |

Answers "is compliance learned or handed to the agent?" Design the campaign around it.

Supplementary leave-one-out: each COLREGs reward term; the boundary observation branch;
recurrence if added.

### 4.4 Reproducibility artefacts

Committed **before** the first training run:

- Frozen, versioned, hashed evaluation suite
- Scenario generator source and seed
- Full reward specification with coefficients and normalisation ranges
- Complete numerical vessel model and actuator parameters
- System identification dataset (see 05 — the direct answer to Paper 2 Reviewer 1.4,
  where the concession was that identification data was not archived reportably)

### 4.5 Compute budget

Roughly halved by the two-vessel cut, which is what buys room for Studies 1 and 2.

Shape: 5 seeds × (1 headline + 4 ablation cells + N learned comparators), plus the
width sweep and the perception sweep. **Estimate wall-clock per run in the new
environment before finalising the comparator list** — still the most likely cause of a
late scope cut.

---

## 5. Repository attachment sets

Repository: `github.com/ntran26/asv-path-replanning`, directory `asv-lidar/`.

**Core set — every thread:** `paper_pooling/README.md`, `paper_pooling/src/README.md`,
`paper_pooling/src/config.py`, `requirements.txt` (~26 KB).

| Doc | Add |
|---|---|
| 01 | `src/lidar.py`, `src/env.py`, `field_deployment/lidar_pooling.py`, `field_deployment/asv_lidar.py` |
| 02 | `src/env.py`, `src/config.py`, `src/metrics.py` |
| 03 | `src/env.py`, `src/obstacles.py`, `src/path.py`, `src/ship.py`, `dynamic_obstacles/rl_env_dynamic.py` |
| 04 | `src/generate_suite.py`, `src/scenarios.py`, `src/curriculum.py`, `src/eval_layouts.py`, `src/evaluate_suite.py`, `src/baselines/los_apf.py`, `OOD_PROTOCOL.md` |
| 05 | `bluefin_modelling/bluefin_model_derivation_explanation.md`, `ship_model_bluefin_4dof.py`, `ship_model_bluefin_v2.py`, `bluefin_4dof/validate_bluefin_4dof.py`, `real_vessel_performance/log_parser.py`, `test_3_metrics.json` |

**Notes.** `src/` is the authoritative code surface — do not attach the superseded
top-level `rl_env.py` (60 KB) or `rl_env_reward_v2.py` (64 KB). Field logs are 130 KB to
2 MB and are Claude Code territory only, never chat attachments. `REWARD_ANALYSIS.md`,
`REWARD_REDESIGN.md` and `PAPER3_BASELINES.md` are referenced but not committed — commit
them so the repository is self-describing.

---

## 6. Standing principles

1. **Concessions trace to experimental design, not writing.** Lock baselines, seeds,
   metrics and ablations before drafting.
2. **Reward scale mismatches are invisible until audited.** Paper 2's `r_oa` was ~49×
   weaker than `r_pf` at contact distance, masked by the λ framing.
3. **Staged repair without redesign degrades performance.** Four incremental repair
   attempts all fell below the 94% SAC baseline.
4. **The narrow waterway constraint is the differentiator.** Foreground Rule 9 and
   restricted-water scope in positioning.
5. **A better nominal model is not robustness.** Pair system ID with domain randomisation
   over identified parameters ± their confidence intervals.
6. **Turn direction from yaw rate, not rudder angle.** Dynamics delay the sign change by
   several timesteps.
