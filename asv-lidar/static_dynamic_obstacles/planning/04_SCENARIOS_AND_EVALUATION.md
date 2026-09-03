# 04 — Scenario Generation and Evaluation Suite

**Handover target:** Claude chat (design), then Claude Code (implementation)
**Depends on:** 03 (target behaviour models), 01 (encounter classifier)
**Merges original task points 4 and 5** — they share one generator.

---

## 1. Purpose

One scenario generator serving two consumers: the randomised training distribution and
the frozen evaluation suite. Merging them is deliberate — the same
backwards-from-encounter-class code produces both, and building it twice invites drift.

---

## 2. Generator design

**Do not sample initial conditions and hope encounters emerge.** Random spawning
frequently produces target ships that pose no threat at all, wasting training samples.

**Parameterise by encounter type, then solve backwards:**

1. Sample the intended encounter class (head-on, crossing give-way, crossing stand-on,
   overtaking, being overtaken, none)
2. Sample a heading intersection angle from that class's valid interval
3. Sample a target speed — including slower targets for the overtaking class and faster
   ones for being-overtaken
4. Sample a spawn TCPA
5. **Solve backwards** for the target spawn position that produces that geometry at that
   TCPA against the own ship's projected track

This is Waltz & Okhrin's routine (§5.1), and they contrast it explicitly with random
spawning that may create no threat. It also yields class-balanced training for free.

**Include the null class.** They found it essential to include targets with a course
similar to the own ship, because such cases never arise from a purely
COLREGs-conditioned spawner but do occur in practice.

Static obstacles are generated independently and crossed with the target configuration.

---

## 3. Training distribution

### 3.1 Sampling

- Target count: 0–3. Weight so 0- and 1-target episodes are well represented — field
  deployment has one target and two masked slots, and that configuration must not be
  out of distribution (01 §6.2).
- Encounter class: balanced across the five classes plus null. **Report the realised
  distribution** in the paper.
- Target behaviour: constant velocity only (D1).

### 3.2 Curriculum

Warm-starting from the Paper 2 policy is **no longer viable** (D7) — the reward,
observation and LiDAR semantics have all changed. The curriculum does the work instead:

| Stage | Content |
|---|---|
| 1 | Static obstacles only, straight constant-width channel |
| 2 | Static obstacles, variable width and bends |
| 3 | Single dynamic target, generous spawn TCPA |
| 4 | Single dynamic target, reduced TCPA, narrower channel |
| 5 | Multi-target (2–3), full difficulty range |

Curriculum axes: target count, spawn TCPA, channel width, static clutter density,
speed ratio.

Train Paper 3 from scratch; treat the Paper 2 SAC policy purely as a frozen baseline.
This is cleaner to describe, removes a fragile engineering dependency, and strengthens
the "unmodified Paper 2 SAC" comparator by making it genuinely independent rather than
an ancestor of the new agent.

---

## 4. Evaluation suite

**Imazu is dropped (D8)** — open water and scale-incompatible (spawn radii ~6 NM, which
at LBP 1.57 m maps to a ~110 m domain, over four times the basin's long dimension).

### 4.1 The cost of dropping it — read this

There is now **no externally-defined scenario set**. Every case a reviewer sees was
designed by the author. Paper 2 already drew fire on baselines and rigour, and *"the
authors constructed their own benchmark and then reported that classical methods do
poorly on it"* is the easiest version of that criticism to make.

Two mitigations, in order of preference.

**Mitigation A (recommended, O1): adopt "Around the Clock".** Waltz & Okhrin's 24
single-ship encounters at equally spaced target headings, φ_TS,j = (j/25)·2π for
j = 1…24, own ship and target set to meet at the origin. Single-vessel, deterministic,
published, citable, and trivially adapted to a channel because each case is one target on
one bearing. It sweeps every encounter classification boundary systematically —
including the astern sector, now directly relevant given the 360° swath. Far cheaper to
set up than Imazu and it meets the open-water objection because each case can be walled.

**Mitigation B (mandatory regardless): release the generator.** Seed, source, and the
frozen suite as a data artefact. With no external benchmark, scenario reproducibility is
the only thing standing between this work and the self-designed-benchmark criticism.

### 4.2 Tier A — deterministic named cases (~40–60)

Hand-specified constellations: one per encounter type × channel-width condition, plus a
few multi-target additions.

These are what gets **plotted**: trajectory overlays, rudder traces, CPA-vs-time curves,
per-case commentary. They replace Imazu's interpretability role. Small enough that every
case can appear in an appendix figure.

### 4.3 Tier B — stratified randomised holdout

The statistical tier, successor to Paper 2's 500-case suite.

| Stratum | Levels |
|---|---|
| Encounter type | 5 (head-on, crossing give-way, crossing stand-on, overtaking, being overtaken) |
| Target behaviour | 3 (constant velocity, compliant reactive, non-compliant) |
| Difficulty | 3 (from channel width in breadths, spawn TCPA, simultaneous conflict count) |

45 cells before static clutter. Cross with 0–2 static obstacles. At 20 episodes per cell
this gives ~900 cases — same order as Paper 2.

**Cost it before committing:** ~900 cases × 5 seeds × every comparator and ablation. This
is where the campaign budget actually goes (see 00 §3.5).

### 4.4 Three disciplines, now load-bearing

**Freeze before training.** Generate, version, hash, and commit the suite **before the
first Paper 3 training run**. Direct application of the standing principle that
concessions trace to pre-writing experimental decisions.

**Define difficulty geometrically only.** Width, TCPA, conflict count — all measurable
from initial conditions, none referencing baseline performance. Then "classical methods
degrade in the hardest stratum" is a *result* rather than a construction.

Avoid framing the suite as "challenging for classical methods." A reviewer reads that as
designing the benchmark to produce the conclusion. Define difficulty by geometry and let
the baselines fail where they fail.

**Include cases the DRL agent also fails.** A suite where the proposed method succeeds
everywhere and the baselines fail everywhere reads as constructed regardless of how it
was actually built. Include a stratum narrow enough that no method passes cleanly, and
report it. It costs nothing and buys considerable credibility.

---

## 5. Metrics

Full list in `00_PAPER3_INDEX_AND_PROTOCOL.md` §3.2. Points specific to the suite:

- Report per-encounter-type violation rates, not a pooled number
- Report the **minimum CPA distribution as a CDF**, not a mean — the tail is the safety
  claim
- Separate obstacle / boundary / target-ship collisions, as Paper 2 did. In a narrow
  channel a successful avoidance manoeuvre can still fail by pushing the vessel into the
  boundary, and that distinction was one of Paper 2's accepted contributions
- Rule 17 release timing is a metric, not just a behaviour — it only exists in the
  non-compliant stratum

---

## 6. Open items

- **O1** — adopt "Around the Clock" as the external named benchmark? Recommended.
- Fix the number of episodes per Tier B cell after the compute estimate
- Define difficulty thresholds numerically (channel width in breadths, TCPA bands,
  conflict count)
- Specify the Tier A case list explicitly
- Decide the presentation format for multi-axis results (per-axis table vs Pareto);
  a weighted scalar will be contested
