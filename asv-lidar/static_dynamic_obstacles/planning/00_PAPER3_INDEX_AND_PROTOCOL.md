# Paper 3 — Index and Experiment Protocol

**Project:** ML-based navigation and control for ASVs (PhD, AMC/UTas)
**Scope of Paper 3:** Path following with static *and dynamic* obstacle avoidance and
COLREGs compliance in narrow/restricted waterways.
**Predecessors:** ICMCR 2026 conference paper (SAC vs PPO, static); MDPI *Drones*
journal paper (Paper 2 — LiDAR sector pooling, staged curriculum, sim-to-field).

---

## 1. How to use this set

Six documents. This one is the anchor; the other five are self-contained handovers
intended to be opened in **separate threads**.

| # | Document | Target surface | Depends on |
|---|---|---|---|
| 00 | `00_PAPER3_INDEX_AND_PROTOCOL.md` | — (anchor) | — |
| 01 | `01_PERCEPTION_AND_OBSERVATION.md` | Claude Code | — |
| 02 | `02_REWARD_AND_COLREGS.md` | Claude chat, then Code | 01 (encounter classifier) |
| 03 | `03_ENVIRONMENT_AND_TARGETS.md` | Claude Code | 01 (LiDAR/tracker interface) |
| 04 | `04_SCENARIOS_AND_EVALUATION.md` | Claude chat, then Code | 03 (target behaviour models) |
| 05 | `05_VESSEL_MODEL_AND_SIM2REAL.md` | Claude chat + field work | — (parallel track) |

Also carry `PROJECT_BRIEF.md`, `REWARD_REDESIGN.md`, and `PAPER3_BASELINES.md` into
any thread — file contents do not persist across Claude conversations.

**Suggested order of work.** 05 starts immediately and in parallel (it gates on basin
booking, which has lead time). 01 → 03 → 02 → 04 in sequence, because 02 needs the
encounter classifier built in 01, and 04 needs the target behaviour models from 03.

---

## 2. Decision log

Decisions confirmed in the design session that produced these documents. Recorded so
they are not silently re-opened.

| # | Decision | Status |
|---|---|---|
| D1 | Target ships: constant-velocity for **training**; reactive for **evaluation only** | Confirmed |
| D2 | COLREGs enforced via **reward terms + explicit encounter-role feature** in the observation | Confirmed |
| D3 | Variable target count: **fixed 3 slots + valid mask**, shared per-slot encoder, concatenation headline, sum-pooling behind a config flag | Confirmed |
| D4 | Slot assignment by **track ID persistence**, sorted by CRI (not TCPA) | Confirmed |
| D5 | Channel boundary supplied **from the map**, not from LiDAR; `c_t` reserved for obstacles | Confirmed |
| D6 | Raw LiDAR swath **360°, 720 beams (0.5°)**; pooled `c_t` forward-biased to ±135° | Confirmed |
| D7 | **No warm-start** from the Paper 2 policy. Train from scratch; Paper 2 SAC is a frozen baseline only | Confirmed |
| D8 | Imazu 22 problems **dropped** (open-water, scale-incompatible) | Confirmed |
| D9 | Evaluation set **generated**, two tiers (deterministic named cases + stratified randomised holdout) | Confirmed |
| D10 | Reward and observation both **fully redesigned**, not patched from Paper 2 | Confirmed |

### Decisions still open

| # | Question | Owner document |
|---|---|---|
| O1 | Adopt Waltz & Okhrin "Around the Clock" (24 single-ship cases) as an external named benchmark to replace Imazu? | 04 |
| O2 | Give-way-only vs full give-way/stand-on reciprocity in the COLREGs encoding | 02 |
| O3 | Does sim-to-real transfer become a standalone RQ (RQ4)? | 05 |
| O4 | Does the 10 × 25 m workspace need to grow for meaningful encounter geometry? | 03 |
| O5 | Physical barrier at the basin edge vs software gating of beyond-wall returns | 01 / 05 |
| O6 | External ground-truth instrumentation for system identification (total station / overhead camera / mocap) | 05 |

---

## 3. Cross-cutting protocol

### 3.1 Seeds and statistics

- **Five** independent training seeds for every reported configuration (Paper 2 used
  three and was criticised for it).
- Bootstrap 95% confidence intervals on all aggregate metrics.
- Report full seed spread, not only the mean.
- Do not interpret intervals from five seeds as a definitive estimate of training
  variability; state the limitation.

### 3.2 Metric set

Task metrics (carried from Paper 2):

- Success rate
- Collision rate, **reported separately** for obstacle / boundary / target ship
- RMS and maximum cross-track error
- Path length ratio
- Action smoothness

COLREGs metrics (new — define precisely in 02 and 04):

- Violation rate **per encounter type**
- Minimum CPA distribution (report the CDF, not just the mean)
- Ship-domain intrusion rate and depth
- Time-to-first-evasive-action (Rule 8: "in ample time")
- Magnitude of first evasive action (Rule 8: "large enough to be readily apparent")
- Stand-on hold duration and release timing (Rule 17)
- Side-of-passing correctness

**Aggregation is a design decision, not a reporting detail.** A single scalar score
combining safety, COLREGs compliance and path-following will be contested. Decide the
presentation format before running the campaign; a Pareto or per-axis table is safer
than a weighted sum.

### 3.3 Ablation matrix

The D2 decision yields a clean 2 × 2 for free:

|  | Role feature OFF | Role feature ON |
|---|---|---|
| **COLREGs reward terms OFF** | Baseline (avoidance only) | Told, not rewarded |
| **COLREGs reward terms ON** | Learned from kinematics | Full method |

This directly answers "is COLREGs compliance learned or handed to the agent?", which
is the question a sceptical reviewer will ask. Design the training campaign around it
rather than retrofitting.

Additional leave-one-out ablations (Paper 2 conceded these; do not concede again):

- Each COLREGs reward term individually
- Boundary observation branch on/off
- Sum-pooling vs concatenation aggregation
- Frame stacking / recurrence for partial observability

### 3.4 Reproducibility artefacts

Committed **before** the first training run:

- Frozen, versioned, hashed evaluation suite
- Scenario generator source and seed
- Full reward specification with coefficients
- Complete numerical vessel model and actuator parameters
- System identification dataset (see 05 — this is the direct answer to Paper 2
  Reviewer 1.4, where the concession was that the sys-ID data was not archived in a
  reportable form)

### 3.5 Compute budget

Cost this before committing to the baseline list in `PAPER3_BASELINES.md`.

Rough shape: 5 seeds × (1 headline + 4 ablation cells + N learned comparators)
× a harder environment than Paper 2. With seven learned comparators this is upward of
60 training runs. **Estimate wall-clock per run in the new environment before
finalising the comparator set** — this is the most likely cause of a late scope cut.

### 3.6 Scope risk

Points 1–5 are one paper. Point 6 is arguably its own short contribution: an
identification and sim-to-real study using the existing Paper 2 field logs plus one
dedicated trial. Splitting de-risks both and produces a publishable artefact while the
COLREGs training campaign runs. Revisit this decision once 05 has a concrete
manoeuvre plan and basin booking date.

---

## 4. Standing principles

Carried from the Paper 2 review cycle and the reward-repair experience:

1. **Concessions trace to experimental design, not writing.** Every concession in
   Paper 2's review came from a decision made before drafting started. Lock baselines,
   seeds, metrics and ablations before writing begins.
2. **Reward scale mismatches are invisible until audited.** The Paper 2 `r_oa` term was
   ~49× weaker than `r_pf` at contact distance, masked by the λ weighting framing.
   Explicit per-term scale audits are mandatory at design time.
3. **Staged repair without redesign degrades performance.** Four incremental repair
   attempts all fell below the 94% SAC baseline; principled redesign outperformed them.
4. **The narrow waterway constraint is the novelty differentiator.** Most COLREGs DRL
   work targets open water. Foreground Rule 9 and restricted-water scope in positioning.
5. **A better nominal model is not robustness.** Pair system identification with domain
   randomisation over the identified parameters ± their confidence intervals.
