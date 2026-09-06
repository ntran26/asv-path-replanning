# 04 — Scenario Generation and Evaluation Suite

**Revision 2** — single target, "Around the Clock" adopted, width sweep added.
**Handover target:** Claude chat (design), then Claude Code
**Depends on:** 03 (target behaviours, corridor geometry), 02 (precedence thresholds)

---

## 1. Purpose

One scenario generator serving three consumers: the randomised training distribution, the
frozen evaluation suite, and the two parameter sweeps (Studies 1 and 2). Building them
separately invites drift.

---

## 2. Generator design

**Do not sample initial conditions and hope encounters emerge.** Random spawning frequently
produces targets that pose no threat, wasting samples.

**Parameterise by encounter class, then solve backwards:**

1. Sample the intended class — head-on, crossing, overtaking, being overtaken, none
2. Sample a heading intersection angle from that class's valid interval
3. Sample a target speed — slower for overtaking, faster for being overtaken
4. Sample a spawn TCPA
5. **Solve backwards** for the spawn position producing that geometry at that TCPA against
   the own ship's projected track
6. Sample corridor width (in ship breadths), bend geometry, and path offset
7. Add static obstacles independently

This is Waltz & Okhrin's routine (§5.1), which they contrast explicitly with random
spawning that may create no threat.

**Include the null class** — a target on a course similar to the own ship. It never arises
from a purely class-conditioned spawner but does occur in practice.

---

## 3. Training distribution

### 3.1 Sampling

- Target present in a substantial majority of episodes, but **a meaningful fraction with no
  target at all** — the static-only configuration must not be out of distribution
- Encounter class balanced across the five classes plus null. **Report the realised
  distribution** in the paper
- Target behaviour: constant velocity only (D1)
- Corridor width sampled across the full sweep range, so the policy is not specialised to
  one geometry

### 3.2 Curriculum

Warm-starting from Paper 2 is not viable (D7) — reward, observation and LiDAR semantics
have all changed. The curriculum does that work instead:

| Stage | Content |
|---|---|
| 1 | Static obstacles only, straight constant-width corridor |
| 2 | Static obstacles, variable width and bends |
| 3 | Single dynamic target, generous spawn TCPA, wide corridor |
| 4 | Single dynamic target, reduced TCPA, narrowed corridor |
| 5 | Full difficulty range with static clutter |

Axes: spawn TCPA, corridor width, static clutter density, speed ratio.

---

## 4. Evaluation suite

**Imazu is dropped (D8)** — open water and scale-incompatible (spawn radii ~6 NM, which at
LBP 1.57 m maps to a ~110 m domain).

### 4.1 External benchmark — adopted (O1 resolved)

**Waltz & Okhrin's "Around the Clock":** 24 single-ship encounters at equally spaced target
headings, φ_TS,j = (j/25)·2π for j = 1…24, own ship and target set to meet at the origin.

The Revision 2 scope makes this an **exact fit rather than an adaptation** — it is a
single-target benchmark and the paper is now a single-target paper. It sweeps every
encounter classification boundary systematically, including the astern sector, which is
directly relevant given the 360° swath and the being-overtaken class.

Run in two variants:

- **Open water** — comparability against the published literature
- **Channel-constrained** — same 24 constellations with corridor walls added. Novel in its
  own right, and where classical VO and APF baselines should begin to fail

This is the principal defence against the criticism that the benchmark was constructed by
the authors. **Releasing the generator (§4.5) remains mandatory regardless.**

### 4.2 Tier A — deterministic named cases (~30–40)

Hand-specified: one per encounter class × corridor width condition, plus static-clutter
variants. These carry the **trajectory figures** — overlays, rudder traces, CPA-vs-time
curves, per-case commentary. Small enough that every case can appear in an appendix figure.

### 4.3 Tier B — stratified randomised holdout

| Stratum | Levels |
|---|---|
| Encounter class | 5 (none, head-on, crossing, overtaking, being overtaken) |
| Target behaviour | 3 (constant velocity, compliant reactive, non-compliant) |
| Corridor width | 3 (wide, intermediate, narrow — thresholds from 02) |
| Static clutter | 0–3 obstacles |

45 cells before clutter. At ~20 episodes per cell, ~900 cases — same order as Paper 2.
Cost this against the compute estimate before fixing the episode count.

### 4.4 Difficulty definition

**Geometric only** — corridor width in ship breadths, spawn TCPA, static clutter count.
Never defined by baseline performance. Then "classical methods degrade in the narrow
stratum" is a *result* rather than a construction.

Avoid framing any stratum as "challenging for classical methods"; a reviewer reads that as
designing the benchmark to produce the conclusion.

**Include cases the proposed method also fails.** A suite the method aces everywhere reads
as constructed regardless of how it was built. The narrowest corridor crossed with a
non-compliant target is the natural candidate — report it.

### 4.5 Three disciplines

**Freeze before training.** Generate, version, hash and commit the suite **before the first
training run**.

**Release the generator.** Seed, source, and the frozen suite as a data artefact.

**Disjoint seeds** between the training distribution and the evaluation suite, and a
separate development suite for any algorithm selection, so the frozen suite is never
contaminated by selection decisions.

---

## 5. Study 1 — channel-width sweep

**Primary evidence for N2 and contribution C3.**

Sweep corridor width from open-water-equivalent down to the point at which the Rule 14
starboard alteration no longer fits within the channel. Hold everything else fixed;
use the Tier A named cases as the base constellations so the sweep is interpretable.

Report per width: success rate, compliance rate per class, minimum CPA, boundary collision
rate, and the governing rule from the precedence table. Identify the width at which each
classical comparator becomes inadmissible — that transition point is the headline figure.

Requires the enlarged workspace (O4).

---

## 6. Study 2 — perception degradation

**Primary evidence for N1 and contribution C4.**

Degrade the tracker along four axes independently, then jointly:

| Axis | Range |
|---|---|
| Pose drift magnitude | nominal → several × the rf2o characterisation from 05 |
| Detection dropout rate | 0 → the point of track loss |
| Occlusion duration | 0 → beyond the tracker's coast time |
| Velocity estimate noise | nominal → the point of encounter misclassification |

Report compliance and safety as a function of each. The key result is **where failures stop
being conservative and start being unsafe** — a policy that becomes cautious under
degradation is deployable; one that misclassifies and turns the wrong way is not.

Requires no basin time. Run on the frozen suite with degradation as an environment
parameter (hooks specified in 03 §7).

---

## 7. Metrics

Full list in `00_PAPER3_INDEX_AND_PROTOCOL.md` §4.2. Suite-specific points:

- Per-class violation rates, not a pooled number
- **Minimum CPA as a CDF**, not a mean — the tail is the safety claim
- Collisions separated into static obstacle / boundary / target vessel. In a narrow
  corridor a successful avoidance can still fail by pushing the vessel into the boundary,
  and that distinction was one of Paper 2's accepted contributions
- Perception metrics (acquisition range, classification latency and stability, velocity
  error, occlusion duration) reported for the nominal case and across the Study 2 sweep

---

## 8. Comparators

| Family | Method |
|---|---|
| Classical | LOS-PID + DWA |
| Classical | COLREGs-VO (Kuwata et al., 2014) |
| Classical | Encounter-specific VO (Thyri & Breivik, 2022) — also the reactive target model |
| Classical | *Optional* NMPC with COLREGs constraints (Gonzalez-Garcia et al., 2022) |
| Learned | Paper 2 SAC, unmodified, zero-shot (frozen) |
| Learned | PPO, RecurrentPPO, TQC retrained on the same environment |
| Learned | COLREGs-ablated policy (avoidance only) |

Classical baselines run against reactive targets as well, or the comparison is not
like-for-like. **Note the information-parity argument:** the VO comparators require tracked
target state, which is exactly what the perception pipeline provides — so they consume the
same estimates, including their errors. State this; it is what makes the comparison fair
and it strengthens N1.

`[TBC — final list after the compute estimate.]`

---

## 9. Open items

- Fix episodes per Tier B cell after the compute estimate
- Define corridor width thresholds numerically, from the 02 precedence table
- Specify the Tier A case list explicitly
- Decide the presentation format for multi-axis results — a weighted scalar will be
  contested
- Decide whether the channel-constrained "Around the Clock" variant is reported alongside
  the open-water version or in an appendix
