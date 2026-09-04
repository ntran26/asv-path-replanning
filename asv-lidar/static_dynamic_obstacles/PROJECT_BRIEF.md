# PROJECT BRIEF

> Update the two fields below each time. Everything else is stable until a decision changes.

**Last updated:** 2026-09-02
**This thread's task:** `______________________________________________`

---

## 1. Overview

Research: machine-learning navigation and control for Autonomous Surface Vessels —
path following combined with real-time obstacle avoidance via deep RL. Multi-paper series.

| Output | Status |
|---|---|
| Conference (ICMCR 2026) | Published. Established SAC over PPO, static obstacles |
| Paper 2 (MDPI *Drones*) | Accepted after revision. LiDAR sector pooling, staged curriculum, sim-to-field validation |
| **Paper 3** | **In design — this is the current work** |

Paper 2 is settled. Don't re-open its design; it's the baseline and the source of the
review lessons below.

---

## 2. Platform and stack

**Vessel (field):** model-scale Bluefin. 64.55 kg, LOA 1.73 m, LBP 1.57 m, breadth 0.50 m,
draft 0.19 m, Iz 10.45 kg·m². RPLidar C1 (360°, 720 beams, 0.5°, 10 Hz),
`rf2o_laser_odometry`, UDP offboard control at 10 Hz.

**Simulation:** custom Gymnasium env. 10 × 25 m workspace, 20 m reference path,
700-step horizon, 0.1 s control period. 3-DOF Fossen model.

**RL:** Stable-Baselines3 + `sb3-contrib`. SAC primary; PPO, RecurrentPPO, TQC, TD3 as
comparators.

**Codebase:** `paper_pooling/` — `rl_env.py` (~1400 lines), `ship_model.py`,
`asv_lidar.py`, `lidar_pooling.py`, plus training/eval scripts. Separate repos for
3-DOF system ID and the LOS+APF evaluation suite.

**Known gaps carried forward:** field RMS cross-track error roughly 2× simulation;
LiDAR is mounted higher than the test basin wall and cannot be repositioned, so field
scans contain beyond-wall returns.

---

## 3. Paper 3 scope

Extend Paper 2 from static-only to **static and dynamic obstacles**, with **COLREGs
compliance (Rules 9, 13–17)** in **narrow / restricted waterways**.

Positioning: most COLREGs DRL work targets open water. The restricted-waterway
constraint (Rule 9) and Rule 17 stand-on behaviour are the novelty differentiators —
Rule 17 is nearly absent from the DRL literature.

Up to 3 dynamic targets in simulation; 1 in field deployment.

---

## 4. Locked decisions

Confirmed. Treat as settled unless I say otherwise.

| | Decision |
|---|---|
| D1 | Targets: constant velocity for training; reactive for evaluation only |
| D2 | COLREGs via reward terms **+** explicit encounter-role feature in the observation. Not shaping alone, not a separate safety layer |
| D3 | Variable target count: fixed 3 slots + valid mask, shared per-slot encoder; concatenation headline, sum-pooling (DeepSets) behind a config flag |
| D4 | Slot assignment by track-ID persistence, sorted by CRI (not TCPA) |
| D5 | Channel boundary from the map via virtual raycast; LiDAR `c_t` reserved for obstacles only |
| D6 | Raw LiDAR 360°/720 beams; pooled `c_t` forward-biased to ±135°, ~27 non-uniform sectors |
| D7 | No warm-start from Paper 2. Train from scratch; Paper 2 SAC is a frozen baseline only |
| D8 | Imazu 22 problems dropped (open-water, scale-incompatible) |
| D9 | Evaluation generated in two tiers: ~40–60 deterministic named cases + ~900-case stratified holdout |
| D10 | Reward and observation both fully redesigned, not patched |

**Observation (~91 dims, Dict):** `lidar` c_t (27, obstacles only) · `boundary` virtual
raycast (7 rays, pose noise injected) · `ego` u,v,r (3) · `path` e_y,χ̃,χ̃_LA (3) ·
`targets` (3 × 16 features + 3 mask bits).

Per-slot features: distance-to-ship-domain, sin/cos bearing, sin/cos heading-intersection
angle, target speed, relative speed, DCPA, TCPA, CRI, 6-way encounter one-hot.

**Required additions beyond the literature:** a fifth "being overtaken" encounter class;
one shared encounter classifier feeding *both* observation and reward; a Rule 9 vs
Rules 13–17 precedence table; a non-compliant target stratum (without it Rule 17 release
can never be exercised); a mandatory per-term reward scale audit.

---

## 5. Open decisions

Not yet resolved. Don't assume an answer — flag if a task depends on one.

| | Question | Resolved? |
|---|---|---|
| O1 | Adopt Waltz & Okhrin "Around the Clock" (24 cases) as the external named benchmark replacing Imazu? *(recommended)* | |
| O2 | Give-way-only vs full give-way/stand-on reciprocity | |
| O3 | Sim-to-real as standalone RQ4, or split into a separate short paper? | |
| O4 | Does the 10 × 25 m workspace need to grow for meaningful encounter geometry? | |
| O5 | Physical barrier at basin edge vs software gating of beyond-wall returns | |
| O6 | External ground-truth instrumentation for system ID (total station / overhead camera / mocap) — gates basin booking | |

---

## 6. Evaluation protocol

- **5 seeds** for every reported configuration; bootstrap 95% CIs; report full spread
- Frozen, versioned, hashed evaluation suite committed **before** the first training run
- Scenario generator released (seed + source) — mandatory, since dropping Imazu leaves
  no externally-defined set
- Difficulty defined **geometrically** (channel width in breadths, spawn TCPA,
  simultaneous conflict count), never by baseline performance
- Include a stratum where the DRL agent also fails
- Collisions reported separately: obstacle / boundary / target ship
- COLREGs metrics: violation rate per encounter type, min-CPA CDF, ship-domain intrusion,
  time-to-first-action, first-action magnitude, Rule 17 hold and release timing
- Primary ablation is a 2 × 2: COLREGs reward terms on/off × role feature on/off

**Classical baselines:** LOS-PID + DWA; COLREGs-VO (Kuwata); encounter-specific VO
(Thyri & Breivik); COLREGs-NMPC (optional).
**Learned comparators:** Paper 2 SAC unmodified, retrained PPO, RecurrentPPO, TQC,
COLREGs-ablated policy, optionally TD3 and frame-stacked SAC.

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

---

## 8. How to work with me

**Workstream separation.** Claude chat = thinking and decisions (design rationale, RQ
formulation, literature positioning). Claude Code = anything touching the repositories.
Cowork = document deliverables and formatting.

**Output style.** Structured, technically grounded drafts with explicit decision points
flagged for my input. Not exhaustive option lists — give me a recommendation and the
reasoning, then the alternatives.

**Literature.** Q1 or high-ranking journals, recent. Avoid arXiv unless strongly
justified. Separate "cite for framing" from "read intensively for implementation."

**Key sources for Paper 3:** Waltz & Okhrin (2023, *Neural Networks* 165:634–653) —
§3.3 CPA/CRI and §4.3 encounter table are directly reusable, **but their constants are
tuned for a 320 m KVLCC2 and must be re-derived in ship lengths for the 1.57 m Bluefin;
their 3·Lpp ship domain does not fit a 10 m channel.** Also Waltz, Paulig & Okhrin
(2025, *ESWA*); Heiberg et al. (2022, *Neural Networks*); Fan et al. (2025, *Ocean Eng.*);
Kim et al. (2025, *JMSE*).

---

## 9. Attach alongside this

`00_PAPER3_INDEX_AND_PROTOCOL.md` (always), plus whichever task doc applies:
`01_PERCEPTION_AND_OBSERVATION` · `02_REWARD_AND_COLREGS` ·
`03_ENVIRONMENT_AND_TARGETS` · `04_SCENARIOS_AND_EVALUATION` ·
`05_VESSEL_MODEL_AND_SIM2REAL`.

Work order: 05 in parallel now (gates on basin booking), then 01 → 03 → 02 → 04.
