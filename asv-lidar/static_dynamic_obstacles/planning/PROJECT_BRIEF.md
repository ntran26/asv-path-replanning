# PROJECT BRIEF — Paste at thread start

> Update the two fields below each time. Everything else is stable until a decision changes.
> **Revision 2** — repositioned to two-vessel encounters. Supersedes the multi-vessel version.

**Last updated:** 2026-09-04
**This thread's task:** `______________________________________________`

---

## 1. Overview

PhD candidate, Australian Maritime College / University of Tasmania (ID 675973).
Supervisors: Dr Hung Nguyen, Dr Peter King, Dr Minh Tran. Thesis target ~April 2027.

Research: machine-learning navigation and control for Autonomous Surface Vessels —
path following combined with real-time obstacle avoidance via deep RL. Multi-paper series.

| Output | Status |
|---|---|
| Conference (ICMCR 2026) | Published. Established SAC over PPO, static obstacles |
| Paper 2 (MDPI *Drones*) | Accepted after revision. LiDAR sector pooling, staged curriculum, sim-to-field validation |
| **Paper 3** | **In design — this is the current work** |

Paper 2 is settled. Don't re-open its design; it's the baseline and the source of the
review lessons in §7.

---

## 2. Platform and stack

**Vessel (field):** model-scale Bluefin. 64.55 kg, LOA 1.73 m, LBP 1.57 m, breadth 0.50 m,
draft 0.19 m, Iz 10.45 kg·m². RPLidar C1 (360°, 720 beams, 0.5°, 10 Hz),
`rf2o_laser_odometry`, UDP offboard control at 10 Hz.

**Simulation:** custom Gymnasium env, 0.1 s control period, 700-step horizon, 3-DOF
Fossen model. Corridor width sweeps 10 m → 3.5 m (20 → 7 breadths); simulation matches the
basin so every width is physically reproducible.

**RL:** Stable-Baselines3 + `sb3-contrib`. SAC primary; PPO, RecurrentPPO, TQC as
comparators.

**Codebase:** `github.com/ntran26/asv-path-replanning`, directory `asv-lidar/`.
`paper_pooling/src/` is the authoritative code surface (modular rewrite, ~3,100 lines,
behaviourally identical to the original scripts). Do **not** work from the top-level
`rl_env.py` or `rl_env_reward_v2.py` — superseded. `dynamic_obstacles/rl_env_dynamic.py`
already contains early Paper 3 work. `bluefin_modelling/` holds the system-ID pipeline.
`field_deployment/` holds the hardware stack and trial logs (logs are Claude Code
territory only — 130 KB to 2 MB each).

**Known gaps carried forward:** field RMS cross-track error roughly 2× simulation. The LiDAR
does not register the pool edge at all — it registers the facility walls 1–2 m beyond it, which
is the architectural justification for the map-derived boundary branch. No IMU: u, v and r are
differentiated from pose, so the `ego` observation branch carries error in the field that
simulation does not model. Static obstacles are suspended panels, confirmed stable in the
water, so apparent motion of static objects comes almost entirely from ego-pose error.
One full-length facility wall is matte black — verify LiDAR return rate on that side before
committing to scan-to-map localisation.

---

## 3. Paper 3 scope

Path following with **static obstacles and one dynamic target vessel**, COLREGs-compliant,
in a **narrow / restricted waterway**.

**Repositioned from the earlier multi-vessel plan.** Two-vessel encounters are the unit of
analysis: Rules 13–16 are formulated pairwise, confined geometry precludes simultaneous
close-quarters conflicts so encounters are sequential, and every reported behaviour becomes
physically reproducible in the basin. Multi-vessel extension only if time allows — `N_max`
is a config parameter so it costs a retrain, not a redesign.

### Novelty axes

| | Claim |
|---|---|
| **N1** *(headline)* | COLREGs compliance driven by target state estimated entirely from onboard 2D LiDAR, not AIS or a simulation oracle. Perception noise characterised from field logs and injected in training |
| **N2** | Rule 9 precedence over Rules 13–16 where channel width makes the open-water manoeuvre inadmissible |
| **N3** | Full sim-to-real pipeline — system ID, domain randomisation over identified uncertainty, field validation |

### Three studies carrying the depth

- **Study 1** — channel-width sweep, from open-water-equivalent down to where the Rule 14
  starboard alteration no longer fits
- **Study 2** — perception degradation: pose drift, detection dropout, occlusion duration,
  velocity noise
- **Study 3** — domain randomisation ablation evaluated in the field (this is RQ4)

---

## 4. Locked decisions

| | Decision |
|---|---|
| S1 | Two-vessel encounters. One dynamic target + up to 3 static obstacles. `N_max` kept configurable |
| S2 | COLREGs scope: Rules 13–16, Rule 9 governs precedence, Rule 8 governs action quality |
| S3 | Own ship gives way in **all** crossing encounters, justified by **Rule 9(b)** — not Rule 18 |
| S4 | Five encounter classes: none, head-on, crossing, overtaking, being overtaken |
| S5 | Rule 17(a)(i) passive course-keeping retained for being-overtaken. Active release under 17(a)(ii) is out of scope → future work |
| S6 | Observation ≈56 dims (see below) |
| D1 | Targets constant-velocity in training; reactive and non-compliant in evaluation only |
| D2 | COLREGs via reward terms **+** explicit encounter-class feature in the observation |
| D5 | Channel boundary from the map via virtual raycast; LiDAR `c_t` reserved for obstacles |
| D6 | Raw LiDAR 360°/720 beams; pooled `c_t` forward-biased to ±135°, ~27 non-uniform sectors |
| D7 | No warm start from Paper 2. Train from scratch; Paper 2 SAC is a frozen baseline only |
| D8 | Imazu dropped (open-water, scale-incompatible) |
| D10 | Reward and observation both fully redesigned, not patched |

**Observation (≈56 dims, Dict):** `lidar` 27 (obstacles only) · `boundary` 7 (map raycast,
pose noise injected) · `ego` 3 (u, v, r) · `path` 3 (e_y, χ̃, χ̃_LA) · `target` 16.

Target features: distance to ship domain, sin/cos bearing, sin/cos heading-intersection
angle, target speed, relative speed, DCPA, TCPA, CRI, 5-way encounter one-hot, presence bit.

**Required additions beyond the literature:** a "being overtaken" class; one shared encounter
classifier feeding *both* observation and reward; a Rule 9 vs Rules 13–16 precedence table;
a non-compliant target stratum; a mandatory per-term reward scale audit.

---

## 5. Open decisions

| | Question | Resolved? |
|---|---|---|
| — | Black-wall LiDAR return rate — a *measurement*, do first | |
| — | Compute estimate per run → final comparator list — a *measurement* | |
| — | Width thresholds (output of Study 1) | |
| — | Reverse thrust capability | |
| — | Progress-penalty carve-out for Rule 8(e) | |
| — | Whether recurrence is added for occlusion | |
| — | Compute estimate → final comparator list | |
| — | Target vessel platform and repeatability protocol | |

**Resolved.** O1 — "Around the Clock" adopted as the external benchmark. O2 — give-way only,
via Rule 9(b). O3 — sim-to-real retained as RQ4. O4 — simulation matches the basin, max width
10 m (20 B); the open-water "Around the Clock" variant supplies the unconfined reference.
O5 — software gating, not a physical barrier; the facility walls are the only localisation
reference and must not be occluded. O6 — no external instrumentation; scan-to-map registration
against surveyed facility geometry, validated by static tests and closed-loop drift.
Ship domain — compressed asymmetric, provisionally 2.0/1.0/0.75 · Lpp (ahead/astern/abeam),
final values derived from the identified turning circle. **IMU confirmed** — will be added;
log raw gyro and accelerometer at 100 Hz+, time-synced to the LiDAR. **Precedence structure
resolved** — Rule 9 constrains the space, Rule 8(e) supplies the action when space is
unavailable; boundary conflict resolves to slacken speed or stop; overtaking permitted whenever
geometry allows, passing to port; head-on band ±10°; propulsion authority widens.

---

## 6. Evaluation protocol

- **5 seeds** per configuration; bootstrap 95% CIs; report full spread
- Frozen, versioned, hashed suite committed **before** the first training run
- Scenario generator released (seed + source)
- Difficulty defined **geometrically** — channel width in breadths, spawn TCPA, static
  clutter count. Never by baseline performance
- Include a stratum where the proposed method also fails
- Collisions reported separately: static obstacle / boundary / target vessel
- COLREGs metrics: violation rate per class, min-CPA CDF, ship-domain intrusion,
  time-to-first-action, first-action magnitude, course-keeping stability, passing side
- Perception metrics: track acquisition range, classification latency and stability,
  velocity error, occlusion duration
- Primary ablation is a 2 × 2: COLREGs reward terms on/off × encounter feature on/off

**Classical comparators:** LOS-PID + DWA; COLREGs-VO (Kuwata et al.); encounter-specific
VO (Thyri & Breivik, which also serves as the reactive target model).
**Learned:** Paper 2 SAC frozen zero-shot, PPO, RecurrentPPO, TQC, COLREGs-ablated.

---

## 7. Standing principles

1. Concessions in peer review trace to **experimental design, not writing**. Lock
   baselines, seeds, metrics and ablations before drafting.
2. Reward scale mismatches are invisible until audited — Paper 2's `r_oa` was ~49×
   weaker than `r_pf` at contact distance, masked by the λ framing. Audit per-term
   episode-integrated contribution, not coefficients.
3. Staged reward repair without redesign degrades performance; four attempts all fell
   below the 94% SAC baseline.
4. A better nominal vessel model reduces bias but does not create robustness — pair
   system ID with domain randomisation over identified parameters ± their CIs.
5. Turn direction is judged from **yaw rate, not rudder angle** — dynamics delay the sign
   change by several timesteps.

---

## 8. How to work with me

**Workstream separation.** Claude chat = thinking and decisions. Claude Code = anything
touching the repository. Cowork = document deliverables and formatting.

**Output style.** Structured, technically grounded drafts with explicit decision points
flagged. Not exhaustive option lists — give a recommendation and the reasoning, then the
alternatives.

**Literature.** Q1 or high-ranking journals, recent. Avoid arXiv unless strongly justified.
Separate "cite for framing" from "read intensively for implementation."

**Key sources.** Waltz & Okhrin (2023, *Neural Networks* 165:634–653) — §3.3 CPA/CRI and
§4.3 encounter table are reusable, **but constants are tuned for a 320 m KVLCC2 and must be
re-derived in ship lengths for the 1.57 m Bluefin; the 3·Lpp ship domain does not fit the
channel.** Also Waltz, Paulig & Okhrin (2025, *ESWA* 274:126933); Heiberg et al. (2022,
*Neural Networks* 152:17–33); Hansen et al. (2022, *IFAC-PapersOnLine* 55(31):222–228);
Krasowski & Althoff (2024, *IEEE T-IV* 9(12):7617–7634); Villa, Aaltonen & Koskinen
(*IEEE/ASME Trans. Mechatronics*); Han et al. (2020, *J. Field Robotics* 37(6):987–1002);
Skjetne, Smogeli & Fossen (2004, *MIC* 25(1):3–27).

---

## 9. Attach alongside this

`00_PAPER3_INDEX_AND_PROTOCOL.md` (always), plus whichever task doc applies:
`01_PERCEPTION_AND_OBSERVATION` · `02_REWARD_AND_COLREGS` ·
`03_ENVIRONMENT_AND_TARGETS` · `04_SCENARIOS_AND_EVALUATION` ·
`05_VESSEL_MODEL_AND_SIM2REAL`. Plus `PAPER3_DRAFT_SKELETON.md` when drafting.

Repository files: see `00` §5 for the per-thread attachment sets.

Work order: **02 first** (the precedence table now gates several other decisions),
05 in parallel (gates on basin booking), then 01 → 03 → 04.
