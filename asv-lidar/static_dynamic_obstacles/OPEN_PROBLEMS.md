# OPEN PROBLEMS — what is blocking, and what resolving each one takes

**Companion to `PROJECT_STATE.md`**, which records what is built. This file
records only what is *not settled*, ordered so the highest-leverage decision is
first.

**Last updated:** 2026-09-13 · **Tests:** 334 passing, 2 xfailed

Three kinds of problem, and they need different things from you:

| Part | Kind | Needs |
|---|---|---|
| **A** | Decisions | a call from you — 6 items, none needs new data |
| **B** | Measurements | basin or log time — 5 sessions, ~28 constants |
| **C** | Build work | my time — 7 items, no decision needed |

---

# Part A — decisions only you can make

## A1. The operating speed `U_nom` — 0.55 or 1.14 m/s

**This is the highest-leverage open item in the project.** It sets every time
constant: spawn TCPAs, spawn geometry, the episode horizon, classification
latency, the Rule 8(a) early-action metric, and three of the four width
thresholds.

### The disagreement

03a §1.1 decides **0.55 m/s**, describing it as "taken from the field
measurement and treated as authoritative". It is not a measurement. It is the
placeholder F20 identified: it entered the first `constants.py` under a comment
claiming a field origin, 02b C2 adopted it as ground truth, and 03a is the third
document to cite it as authoritative.

T1 mined the retained logs and found a **median of 1.14 m/s at 12 RPM across 18
runs**, spread 0.56–1.25. That is a real measurement, with one caveat worth
stating: these are short avoidance runs and the vessel never holds a steady
state, so 1.14 is a run-mean past the acceleration ramp, not a measured steady
speed.

### What each choice gives you

| | 0.55 m/s | 1.14 m/s |
|---|---|---|
| Froude | 0.14 | 0.29 |
| λ=50 full scale | 78.5 m at **7.6 kn** | 78.5 m at **15.7 kn** |
| 20 m transit | 36 s | 17.5 s |
| Head-on spawn TCPA | 12.5–17.3 s | **6.0–8.3 s** |
| **Rule 8 urgency at spawn** | **0.00–0.17** | **0.44–0.60** |
| Provenance | a chosen operating point | a measurement with a caveat |

### The argument that decided my recommendation, and it is not the Froude one

02a §11.2 warns that if spawn TCPA drops below `T_act = 15 s`, "the agent is
already late at spawn and `v_r8` is pinned at maximum for the whole episode —
teaching nothing except that the term is unavoidable."

**At 1.14 m/s, urgency at spawn is 0.44–0.60.** The agent begins every head-on
encounter already late. The Rule 8 term would carry almost no gradient, and
`v_r8` is the term that distinguishes "learned to alter" from "learned *when* to
alter" — which is the M5 ablation and a substantial part of the contribution.

At 0.55 m/s urgency starts at ~0 and rises through the encounter, which is the
shape the term was designed to have.

### Recommendation

**Adopt 0.55 m/s, but as a *chosen operating point*, not as a measurement.**

That resolves the provenance problem honestly and keeps 03a's decision intact.
The sentence for the methods is something like: *the campaign runs at a nominal
0.55 m/s, chosen to sit at Fr = 0.14 for full-scale plausibility and to give the
Rule 8 timing term usable dynamic range in a 25 m basin; the platform's measured
cruise at 12 RPM is 1.14 m/s, so the operating point is well inside its
envelope.* Nobody has to defend 0.55 as something it isn't, and the Froude table
and the λ=50 column both stand.

**What this costs:** `CRUISE_RPM` drops from 12 to about 6–7, and `THRUST_CAL`
is re-solved against the new operating point. Both are one-line changes.
Everything else is already a formula of `U_NOM`.

### To resolve

Tell me which value, and whether to reword 03a §1.1's provenance claim. I will
change one constant and re-run the suite; acceptance test `T1` flips from xfail
to pass automatically if you choose 0.55.

---

## A2. The vessel cannot take way off — Rule 8(e) may not be executable

03a §4.3 reasons that shedding 3 m of along-track position needs "of order 5 N
of net decelerating force" and concludes "coasting alone plausibly achieves it;
reverse thrust is probably not required". Acceptance test T9 sets the criterion
at head reach ≤ 1.5·Lpp = **2.36 m**.

Measured against the carried-over hull, thrust cut:

| From | To | Head reach | Time |
|---|---|---|---|
| 1.14 m/s | 0.23 m/s | **29.9 m (19.1 Lpp)** | 53 s |
| 0.51 m/s | 0.10 m/s | **13.5 m (8.6 Lpp)** | 54 s |

It fails by 6–13×, at **both** candidate speeds, so A1 does not rescue it. Fifty
three seconds is also more than half the 90 s episode.

**Why it matters beyond the test.** `R-2`, `R-5`, `v_r8`'s speed-reduction
branch, and Study 1's headline "manoeuvre-mode share" curve all assume the vessel
can slow meaningfully *within* an encounter. If it cannot, Study 1's headline
result is about a manoeuvre the platform cannot perform.

### Three possible resolutions, and they are not equally likely

1. **The identified surge drag is too low.** The Paper 2 hull was fitted to
   path-following runs, which never test deceleration, so the drag coefficients
   are unconstrained in exactly this regime. This is my guess, and 05's
   deceleration test settles it.
2. **The platform can reverse**, and `allow_reverse` should be set. 03a §4.3
   requires this to be *verified*, not taken from a datasheet.
3. **It genuinely cannot stop**, in which case 03a §4.3 already says what to do:
   state the limitation rather than claiming the manoeuvre, and Study 1's
   framing changes from "alteration vs speed reduction" to something narrower.

### To resolve

A deceleration run is the cheapest possible basin measurement: bring the vessel
to cruise, cut thrust, record the track. Ten minutes. Add it to 05's manoeuvre
set at the top. Until then `T9` stays xfail with the numbers in its reason.

---

## A3. `D_SAFE` — 02a's value breaches 02a's own invariant

The last `TODO(decision)` in the tree.

02a §2 asserts `d_safe < c_wall − B/2`, so the geometry defining a compliant
narrow-channel manoeuvre cannot itself trigger the boundary penalty. With
`c_wall = 0.65` and `B = 0.50` the ceiling is **0.40 m**, and 02a §5.2's stated
`d_safe = 0.50 m` does not clear it — in the same sentence that says 0.50 was
chosen *because of* this invariant.

Currently `D_SAFE = 0.35`, the largest 5 cm value that clears with margin. `d_safe`
is the free parameter of the pair: `c_wall` drives all four width thresholds and
is a `TODO(05)` measurement, so moving it would move published predictions.

**Recommendation:** keep 0.35, and note in the methods that the boundary penalty
begins 0.35 m from the channel limit. **To resolve:** confirm, or give a value
below 0.40.

---

## A4. The crossing width threshold — 02a says 6.80 m, 04a says 7.60 m

04a §11 lists this as an open item owned by 02, and it is 0.8 m.

| Source | Reasoning | `W_crossing` |
|---|---|---|
| 02a §2.2 | own ship has `W/2` of starboard room | 6.80 m |
| **04a §1.1** | own ship is at the starboard quarter-width under 9(a), so only `W/4` | **7.60 m** |

They also disagree on the ordering: 02a predicts crossing > head-on >
overtaking; 04a predicts crossing > overtaking > head-on, on an exposure-margin
argument that 04a itself flags for checking.

**Recommendation: adopt 04a's.** It is 02a's own N2 insight carried one step
further — a vessel already keeping starboard under Rule 9(a) has *spent* its
starboard room before the Rule 14 alteration becomes tight. That is the more
interesting claim and the more defensible one.

Both are computed and reported. The suite currently uses 04a's, which sets the
Tier B width strata (wide ≥ 7.60, intermediate 4.02–7.60, narrow < 4.02).

**To resolve:** confirm 04a's, and I will mark 02a §2.2's crossing row
superseded so the two documents stop disagreeing in print.

---

## A5. Curriculum stage 3 cannot carry a bend

F25: the corridor must fit a 10 m basin, so bend magnitude is capped by width.

| `W` (m) | 10 | 9 | 8 | 7 | 6 | 5 | 4 | 3.5 |
|---|---|---|---|---|---|---|---|---|
| max `Δψ` | 0° | 5° | 16° | 27° | 39° | 51° | 60° | 60° |

04a §3.7 sets stage 3 at **7–10 m**, where the ceiling runs from 27° down to 0.
Across that range almost no episode carries a ≥20° bend, so **stage 3 trains no
`r_path` signal at all** — and it is the stage that introduces the encounter
machinery, so the agent meets its first targets on a channel that never bends
and then meets bends and targets together in stage 4.

**Options:** widen stage 3 to 5–10 m; or accept it and note that bend exposure
begins at stage 4; or drop stage 3's lower bound only for no-target episodes.

**Recommendation:** widen stage 3 to 5–10 m. Its purpose is to introduce the
encounter machinery on geometry where the compliant response is available at
every width, and 5 m still satisfies that for head-on and null.

**To resolve:** pick one. It is a one-line change to `CURRICULUM_STAGES`.

---

## A6. Study 2's condition count, and its pose-noise nominal

Two small things on the same axis.

**Count.** 04a §7.1 specifies 4 axes × 5 levels `{0, 0.5, 1, 2, 4}` plus one
joint corner, and states 21 conditions. The 1× level is the same condition on
every axis, so deduplicating gives 4×4 + 1 + 1 = **18**. Implemented as 18.
Running the shared nominal four times would be three wasted conditions.
**To resolve:** confirm 18, or say 21 if a table depends on the number.

**Nominal.** Measured track uptime against pose σ, four seeds, head-on:

| σ (m) | 0.00 | 0.05 | 0.10 | 0.25 | 0.50 |
|---|---|---|---|---|---|
| uptime | 90% | 90% | 90% | 59% | 25% |

Flat to 0.10 m, then falling. The knee is set by `TRACK_GATE_DIST` = 0.30 m: a
displacement inside the association gate costs nothing.

**So if 05 measures pose noise at or below 0.10 m, three of the five sweep levels
land on the flat part and the axis reports robustness that is really
insensitivity.** Worth deciding now whether the sweep multipliers should be
re-centred on the knee rather than on the measured nominal. **To resolve:** flag
it for when 05 lands; no action needed today.

---

# Part B — measurements, owned by 05

About 28 constants, grouped by the session that would produce them. None needs a
decision; all need time.

## B1. Deceleration run — 10 minutes, and it is the most urgent

Settles **A2**. Cruise, cut thrust, record the track. Produces head reach and
constrains the surge drag coefficients, which are currently unconstrained in the
deceleration regime.

## B2. Straight-line runs — settles the speed caveat

Confirms `U_REF` at a held steady state, which the avoidance logs cannot give
(no run holds a ≥3 s near-straight plateau). Also produces the thrust map at
each RPM, replacing `THRUST_CAL`.

## B3. rf2o pose drift — **the single most consequential group**

`BOUNDARY_POSE_NOISE_XY`, `_HEADING_DEG`, `_WALK`, plus `RHO_POSE_DRIFT` and
`w_wall`. **All are currently 0.0**, which means the 01 §3.3 sim-to-real gap is
wide open and no headline run should start. It also gates Study 2's first axis
and the whole N1 claim.

This was T1.2 in 02b's ordering and has not been attempted. It needs pose and
scan streams cross-referenced against surveyed geometry.

## B4. Static-spin recording — 10 minutes

`LIDAR_AFT_MASK_HALF_DEG`, `LIDAR_NO_RETURN_GRAZING_DEG`, beams per revolution
(`TODO(03-6)` — the C1 at 10 Hz may deliver ≈500 rather than 720), and `D_max`
effective including the matte-black wall side (`TODO(04-2)`).

02b §3.2 decided not to block on the aft mask and to bound it by sensitivity
instead (train at 0° and at 30°). That sensitivity run has not been done either.

## B5. Turning circle — the ship domain and the yaw thresholds

`DOMAIN_FORE`, `_AFT`, `_LATERAL`, `R_REF`, `R_DEAD` (gyro noise floor),
`KAPPA_N`, actuator rate limits and lag (`TODO(03-3)`), and `TURN_RATE_DPS` for
the reactive target.

**Note on the lag:** a constant LiDAR/IMU time offset appears in the
identification fit as actuator lag, so this must come from a synchronised fit or
it encodes a clock error as a physical property.

## B6. Target platform top speed — gates a whole encounter class

The being-overtaken class needs the target at 1.5–2.2× own-ship speed. At
`U_nom = 0.55` that is 0.83–1.21 m/s. If the target platform cannot do it, 04a
§1.4 suggests running being-overtaken trials at a reduced own-ship reference of
≈0.35 m/s rather than dropping the class.

---

# Part C — build work outstanding

Mine, not yours. Listed so the picture is complete, roughly in dependency order.

## C1. The scenario generator is not wired into the environment — **I overstated this**

`scenario.py` is built and tested — five classes, backward solve, round-trip
class check, rejection ledger — but **`env.py` does not import it**. Episode
resets still call `_sample_target`, the head-on-only placeholder.

So the corridor generator *is* live in training and the scenario generator is
**not**. My previous summary said "03a and 04a implemented", which was true of
the modules and not true of the training loop. Correcting that here.

Wiring it means routing `reset()` through `ScenarioGenerator` with the curriculum
stage, which also touches `_load_scenario` and the Tier A case path. It is the
next thing I should do and it is not small.

## C2. Throughput — 74 steps/s against 04a's assumed 500

04a §8.3 says to measure before committing to the comparator list. One
policy-seed suite is **6.0 h**, not the 50 minutes assumed. A 30-run protected
core is not affordable on that basis.

Two optimisations are already in (polygon decimated to 0.5 m; the gate computes
perpendicular distances only for beams that fell outside). The measurement is
single-environment, so vectorised envs are the obvious next move and may close
most of the gap. Worth doing before `TODO(04-4)` can be answered.

## C3. Classical comparators do not exist

LOS-PID + DWA, COLREGs-VO, and the encounter-specific VO. Every comparison in
the paper needs them, and 03a §5.3 requires the encounter-specific VO and the
`T-RE` reactive target to be **one implementation** — if they drift apart the
reactive stratum and the VO baseline stop being comparable.

## C4. `T-RE` is a placeholder

A starboard alteration stands in for the velocity obstacle. Ships with C3.

## C5. Occlusion and conflict obstacle placement

04a §3.6. The scenario record carries the `conflict` and `occlusion` flags; the
placement logic is not written. Blocks Tier A's `A-CLT-*` and `A-OCC-*` cases,
acceptance test T4, and Study 2's occlusion axis.

## C6. `metrics.py` does not read the reward keys

02a §10.3 says 00 §4.2's metric set should be a *read* of the emitted keys
rather than a separate computation. The keys are emitted; the read is not
written. Paper 2's concessions came from metrics that were not designed in
before the campaign ran.

## C7. The scale audit, the claim ledger, and the empty result tables

The audit accumulators and pre-committed predictions exist; running 1,000
random-policy and 1,000 Paper 2 SAC episodes through the stage-5 distribution
has not been done — and cannot be until C1 lands, because stage 5 is a scenario
distribution. The claim ledger and empty tables are 04a §9.3 freeze items.

---

# Quick reference

| # | Problem | Kind | Blocks | Cost to resolve |
|---|---|---|---|---|
| **A1** | operating speed 0.55 vs 1.14 | decision | everything timed | one call |
| **A2** | vessel cannot stop; 8(e) unexecutable | decision + B1 | Rule 8(e), Study 1 headline | 10 min basin |
| A3 | `D_SAFE` breaches its own invariant | decision | last `TODO(decision)` | one call |
| A4 | crossing threshold 6.80 vs 7.60 | decision | Tier B strata, Study 1 | one call |
| A5 | stage 3 cannot bend | decision | `r_path` exposure in training | one line |
| A6 | Study 2 count and nominal | decision | a table; the noise axis | one call |
| **B3** | pose drift all 0.0 | measurement | **N1, Study 2, any headline run** | log session |
| B1/B2/B4/B5/B6 | drag, speed, sensor, turning circle, target speed | measurement | domain, thresholds, one class | basin time |
| **C1** | generator not wired into the env | build | **all five encounter classes in training** | mine |
| C2 | 74 steps/s | build | campaign size | mine |
| C3/C4 | comparators and `T-RE` | build | every comparison | mine |
| C5 | occlusion/conflict placement | build | 8 Tier A cases, T4, Study 2 axis | mine |
| C6 | metrics read | build | 04a §10 metric set | mine |
| C7 | audit, ledger, tables | build | freeze checklist | mine |

**If you resolve only one thing:** A1. It settles the timing structure, flips an
acceptance test, and unblocks the re-derivation of every speed-scaled constant.

**If you resolve only one thing that needs the basin:** B1, the deceleration
run. Ten minutes, and it decides whether Rule 8(e) is real.
