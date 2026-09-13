# PROJECT STATE — Paper 3 implementation

**A living document.** Updated on every change to the tree. It records what is
built, what each piece decides, and — the part that matters — **what is
currently blocking a full training run and a full evaluation.**

| | |
|---|---|
| **Revision** | 5 — tasks 03a and 04a implemented; see `OPEN_PROBLEMS.md` |
| **Last updated** | 2026-09-13 |
| **Tests** | **334 passing, 2 xfailed** (both deliberate; see §3.1, §3.2) |
| **Blocking a headline training run** | **4 items** (§2) |
| **Blocking a full evaluation** | **5 items** (§2) |
| **Open `TODO(decision)`** | 1 (`D_SAFE`) |
| **Open measurements owned by 05** | 20 |

---

## 1. What exists

### 1.1 Modules

| Module | Owns | Spec |
|---|---|---|
| `constants.py` | every constant, one place, each with a `TODO` marker | kickoff §5 |
| `ship.py` | 3-DOF Fossen hull, thrust calibration | 05 |
| `path.py` | reference path, arclength, signed curvature, `r_path` | 02b T3 |
| `corridor.py` | **channel generator** — centreline, width profile, bends, polygon | 03a §3 |
| `targets.py` | **oriented hull, five behaviour models, confinement** | 03a §5 |
| `scenario.py` | **generator, backward solve, rejection ledger** | 04a §3 |
| `suite.py` | **Tier A/B, Around the Clock, Study 2, freeze manifest** | 04a §4–§9 |
| `asv_lidar.py` | 720-beam raycast, sector pooling, dead zone, aft mask | 01 §2 |
| `boundary_raycast.py` | virtual boundary scan, beam gating, pose noise | 01 §3 |
| `tracking.py` | cluster → associate → Kalman → static/dynamic | 01 §4 |
| `cpa_cri.py` | CPA, ship domain, collision risk index | 01 §5 |
| `encounter.py` | five-class classifier + hysteresis — **one definition** | 01 §5.3 |
| `colregs/` | `EncounterContext`, engagement machine, admissibility | 02a §10.1 |
| `reward/` | eight dense terms, group clipping, audit | 02a |
| `observation.py` | five branches, 56 dims, frozen index order | 01 §6 |
| `env.py` | the Gymnasium environment | 03a §4 |
| `render.py` | field view + seven-block telemetry panel | RENDER_PANEL_SPEC |
| `play.py` | manual and random harness | — |

### 1.2 Task status

| Task | State |
|---|---|
| 01 perception and observation | **done** |
| 02 / 02a / 02b reward | **done** — T1–T4, C1–C4 |
| 03a environment and target | **done** — §1.2, §3, §4, §5, §6.3, §7, §10 |
| 04a scenario and evaluation | modules done — §3, §4, §5, §6, §7, §9 — **but see B10** |
| 05 vessel model and sim2real | **not started** — 20 measurements |
| Training campaign | **blocked** — §2 |

---

## 2. What is blocking the campaign

Ordered by what they block, not by size. **The first is one decision and it
moves almost everything else.**

### 2.1 Blocking a headline training run

| # | Blocker | Owner | Effect |
|---|---|---|---|
| **B1** | **F24 — the operating speed is contested** | you | every time constant in the paper |
| **B2** | **F28 — the vessel cannot stop; Rule 8(e) is unexecutable** | 05 / you | the whole 8(e) contribution |
| **B3** | **Throughput is 74 steps/s against 04a's assumed 500** | 04 | the comparator list length |
| **B4** | **Pose-noise constants are all 0.0** | 05 | the N1 sim-to-real claim |

### 2.2 Additionally blocking a full evaluation

| # | Blocker | Owner | Effect |
|---|---|---|---|
| **B5** | Curriculum steps per stage and total budget (`TODO(04-3)`, `TODO(04-4)`) | 04 | cannot size the campaign |
| **B6** | Claim ledger and empty result tables not written | you | 04a §9.3 freeze checklist |
| **B7** | `T-RE` reactive target is a placeholder, not the VO comparator | 03 | Tier B's `re` behaviour stratum |
| **B8** | Classical comparators (LOS-PID+DWA, COLREGs-VO, encounter VO) do not exist | 04 | every comparison in the paper |
| **B9** | `metrics.py` does not read the reward keys | 04 | 04a §10's metric set |
| **B10** | **the scenario generator is not wired into `env.py`** | me | **all five encounter classes in training** |

**B10 corrects an overstatement.** `scenario.py` is built and tested, but `env.py`
does not import it: episode resets still call `_sample_target`, the head-on-only
placeholder. The *corridor* generator is live in training; the *scenario* generator
is not. Revision 4 of this document said "03a and 04a implemented", which was
true of the modules and not of the training loop.

---

## 3. Findings

Numbered continuously. Each is a place where the specification and the
implementation disagreed, and the disagreement was resolved *or* is still open.

### 3.1 F24 — the operating speed, and it is the same placeholder for a third time

**Status: OPEN. This is B1 and it is the most consequential open item.**

03a §1.1 decides `U_nom = 0.55 m/s`, calling it "taken from the field
measurement and treated as authoritative". **It is not a measurement.** It is
the placeholder F20 identified: a number that entered the first `constants.py`
under a comment claiming a field origin, was adopted by 02b C2 as ground truth,
and is now cited as authoritative in a third document.

T1 mined the retained logs and found a **median of 1.14 m/s at 12 RPM across 18
runs**, spread 0.56–1.25.

| | `U` | Fr | 20 m transit | λ=50 full scale |
|---|---|---|---|---|
| 03a §1.1 "field-measured" | 0.55 | 0.140 | 36.4 s | 7.6 kn |
| **T1, measured** | **1.14** | **0.290** | **17.5 s** | **15.7 kn** |
| 02a §1 assumed | 0.80 | 0.204 | 25.0 s | 11.0 kn |
| simulator before T1 | 1.77 | 0.451 | 11.3 s | 24.3 kn |

**03a's Froude argument genuinely favours 0.55**, and it is a good argument: at
λ=50 it gives a 78.5 m vessel doing 7.6 kn in a 175–500 m fairway, which is a
plausible restricted-water transit, whereas 1.14 gives 15.7 kn, which is fast
for a channel. But the two claims are different in kind. The logs measure what
the vessel *did*; the Froude argument reasons about what a *scaled model* should
do. If the Bluefin is not intended as a Froude-scaled model of a full-scale
ship, the measurement is the truth and the λ=50 table is an interpretive device
applied afterwards.

**What is implemented:** `U_NOM = U_REF = 1.14` (the measurement), and every
04a quantity derived from it as a formula rather than a table — which 04a's own
status note asks for. Switching is one constant. `cfg.froude()` and
`cfg.full_scale()` produce either column.

**What moves if you choose 0.55:** every spawn TCPA, every spawn range, the
overtaking width threshold, `N_REF_PROG`, `DU_MIN`, `TARGET_SPEED_RANGE`,
`BEING_OVERTAKEN_SPEED_MARGIN`, `TRACK_GATE_DIST`, `SPEED_SCALE`, and the
thrust calibration. Nothing needs re-coding.

Acceptance test `T1` is `xfail(strict)` on Fr = 0.14, so the suite states the
disagreement rather than hiding it, and flips the day it is settled.

### 3.2 F28 — the vessel cannot take way off, and Rule 8(e) depends on it

**Status: OPEN. This is B2.**

03a §4.3 reasons that shedding 3 m of along-track position needs "of order 5 N
of net decelerating force — comparable to the hull's own quadratic drag at
0.55 m/s", concludes "coasting alone plausibly achieves it; reverse thrust is
probably not required", and sets acceptance test T9 at **head reach ≤ 1.5·Lpp =
2.36 m**.

Measured against the carried-over hull, with thrust cut:

| From | To | Head reach | Time |
|---|---|---|---|
| 1.14 m/s (cruise) | 0.23 m/s | **29.9 m (19.1 Lpp)** | 53 s |
| 0.51 m/s | 0.10 m/s | **13.5 m (8.6 Lpp)** | 54 s |

It fails by **6–13×**, and it fails at *both* candidate speeds, so it does not
turn on F24. Fifty-three seconds is also more than half the 90 s episode.

03a §4.3 states the consequence itself: `allow_reverse` must be set and the
platform's actual reverse capability verified — and if the platform cannot
reverse, "take all way off" is unavailable and the paper states the limitation
rather than claiming the manoeuvre. Both routes run through 05, because the
other possibility is that the identified surge drag is badly low.

**This is not cosmetic.** `R-2`, `R-5`, `v_r8`'s speed-reduction branch and the
whole "manoeuvre-mode share" headline curve of Study 1 assume the vessel can
slow meaningfully within an encounter. If it cannot, Study 1's headline result
is about a manoeuvre the platform cannot perform.

Acceptance test `T9` is `xfail(strict)` with the measurement in its reason.

### 3.3 F25 — bend magnitude is capped by channel width

**Status: RESOLVED, with a consequence for the curriculum.**

The corridor must stay inside a 10 m basin, so a wide channel cannot bend.
Measured ceiling, with the chord aligned to the basin's long axis:

| `W` (m) | 10 | 9 | 8 | 7 | 6 | 5 | 4 | 3.5 |
|---|---|---|---|---|---|---|---|---|
| max `Δψ` (deg) | 0 | 5 | 16 | 27 | 39 | 51 | 60 | 60 |

Aligning the **chord** rather than the entry heading is worth more than a factor
of three: a dogleg's far end swings 5.3 m off axis for a 25° turn, while the
same turn as an arc deviates by its sagitta, about 1.4 m.

The generator clamps to the ceiling and records both requested and realised
bend, so a silently straightened channel is visible. 04a's "≥ 40% of episodes
carry a ≥ 20° bend" comes out at **43%** over the stage-5 width range.

**Consequence: curriculum stage 3 (7–10 m) can carry almost no bend**, so it
trains no `r_path` signal — and it is the stage that introduces the encounter
machinery. Worth deciding whether stage 3 should widen its lower bound.

### 3.4 F26 — the null class was unconstructible

**Status: RESOLVED.**

04a §3.4's null row asks for no CPA in the horizon *and* a spawn range of
8–15 m. With no CPA there is no `T_0` to solve backwards from, so the backward
solve degenerates to `R_0 = DCPA`, which the same row caps at 4–7 m. The two
windows are disjoint and every draw capped out at 200 attempts.

Null is now **placed** rather than solved: sampled range and near-parallel
course, then verified to have no CPA inside the horizon. That is what "no CPA in
the horizon" actually specifies.

### 3.5 F27 — being-overtaken does not fit the corridor

**Status: RESOLVED, with one metre of slack.**

04a §3.4 puts the target 4–6 m astern of the own ship and inside the channel,
while §3.2 sets the corridor at 25 m and the reference path at 20 m. Starting
the own ship at the corridor mouth leaves 5 m astern, so a 6 m spawn is outside
the water. 20 + 6 = 26 > 25.

The own ship now starts `astern_room` metres along for this class and the spawn
window is capped at what the channel provides. The class 04a §1.4 already flags
as most likely to fail in the field for perception reasons would otherwise have
been lost to a metre of geometry.

### 3.6 F29 — `r_dom` was reading perception, not ground truth

**Status: RESOLVED. This was the most serious defect found this round.**

`R-1` splits the reward: physical consequence on ground truth, rule regime on
the perceived state — "the agent pays for hitting things whether or not it saw
them". `r_dom` was iterating the tracked `EncounterContext` objects, which exist
only where a *track* exists. **A target inside the ship domain but not yet
tracked cost exactly nothing.**

Found by writing acceptance test T12, where a target 1.0 m abeam sits inside the
1 m sensor dead zone, is never published by the tracker, and drew no penalty.

It inverts the Study 2 design: the whole point is that COLREGs obligations
degrade with perception while physical safety obligations do not, and this made
`r_dom` the most perception-dependent term in the reward.

`r_dom` now reads `state.dom_intrusion`, computed by the environment over its
true targets. The panel's domain margin comes from truth too, so it can display
an intrusion the tracker has not seen.

### 3.7 The pooled LiDAR branch was built from the ungated scan

**Status: RESOLVED.**

`_perceive`'s docstring has always read "raycast → gate → pool → cluster". The
code pooled before gating, which returned the same answer for as long as the
gate had nothing to remove. Adding facility walls (03a §1.2) exposed it: **624
of 720 beams reached the obstacle branch**, and the policy would have learned to
treat the basin as an obstacle field.

### 3.8 The width profile's kinks were spiking `r_path`

**Status: RESOLVED.**

04a §3.2 asks for a "piecewise-linear" width profile over 2–4 control points.
Taken literally that puts a kink at every knot, and since the reference path is
the centreline offset by a fraction of the *local* half-width, a kink in the
width becomes a corner in the path and a delta in its curvature. Measured at
**0.49 rad/s on a dead-straight 10 m channel** — a 2.3 m turn radius that does
not exist. `R-8` differences the vessel's yaw against `r_path`, so those spikes
would have been charged as evasive manoeuvres the agent never made.

A 1 m Hann window over the profile removes the corners and preserves the width
ratio (1.79 realised against a 1.8 cap).

### 3.9 The Study 2 condition count is 18, not 21

**Status: minor, flagged.**

04a §7.1 specifies 4 axes × 5 levels `{0, 0.5, 1, 2, 4}` plus one joint corner
and states 21 conditions. The 1× level is the same condition on every axis, so
deduplicating gives 4×4 + 1 + 1 = **18**. Running the shared nominal four times
is three wasted conditions. Implemented as 18; say which if the count matters
for a table.

### 3.10 The pose-noise knee sits above the Study 2 nominal

**Status: flagged, matters for choosing the nominal.**

Track uptime against pose σ, four seeds, head-on:

| σ (m) | 0.00 | 0.05 | 0.10 | 0.25 | 0.50 |
|---|---|---|---|---|---|
| uptime | 90% | 90% | 90% | 59% | 25% |

Flat to 0.10 m then falling. The knee is set by `TRACK_GATE_DIST` =
`max(2.5·U_REF·Δt, 0.30)` = 0.30 m: a displacement inside the association gate
costs nothing.

**Consequence for 04a §7.1**, which sweeps this axis at `{0, 0.5, 1, 2, 4} ×
nominal`: a nominal at or below 0.10 m puts three of the five levels on the flat
part of the curve, and the axis would report robustness that is really
insensitivity. When 05 measures the real pose noise, check it against this knee
before freezing the sweep.

### 3.11 04a §1.1 supersedes 02a §2.2 on the crossing threshold

**Status: OPEN, owned by 02.** 04a §11 lists it as an open item.

Two derivations, 0.8 m apart:

| Source | Reasoning | `W_crossing` |
|---|---|---|
| 02a §2.2 | own ship has `W/2` of starboard room | 6.80 m |
| **04a §1.1** | own ship is at the starboard quarter-width under 9(a), so `W/4` | **7.60 m** |

04a's is the better reading and it is 02a's own N2 insight carried a step
further: a vessel already keeping starboard has *spent* its starboard room
before the Rule 14 alteration becomes tight. The suite uses 04a's; both are
computed and reported. The predicted ordering also differs — 02a says crossing >
head-on > overtaking, 04a says crossing > overtaking > head-on, on the exposure
margin. 04a flags this against itself.

### 3.12 Earlier findings, still standing

| # | Finding | Status |
|---|---|---|
| F20 | 0.55 m/s was an unsourced placeholder adopted as measurement | see F24 |
| F21 | the domain floor excluded the provisional domain | resolved: `d_abeam` = 1.25 m |
| F22 | `N_ref = 250` inverts `R-9` at the measured speed | resolved: derived from `U_REF` |
| F23 | `d_safe = 0.50` breaches 02a §2's own invariant | **open** — `D_SAFE` = 0.35, the last `TODO(decision)` |
| F23b | the timeout ordering fails undiscounted at a 900-step horizon | flagged; holds at SAC's γ, marginal for PPO's |

---

## 4. Throughput — 04a §8.3's gate

04a §8.3 says to measure before committing to the comparator list. Measured:

| | Value |
|---|---|
| Single environment, stage-5 geometry, one obstacle | **74 steps/s** |
| 04a's assumption | 500 steps/s |
| One policy-seed full suite (1.6 M steps) | **6.0 h** against 04a's "50 minutes" |

The dominant costs are the boundary gate (720 beams against the channel
polygon), the channel-room raycasts in the admissibility predicate (9 per target
per step), and the hull containment test. Two optimisations are already in: the
polygon is decimated to 0.5 m spacing (a 1000-vertex channel made it 250× slower
than the rectangle it replaced), and the gate computes perpendicular distances
only for beams that fell outside.

**This is B3.** At 74 steps/s a 30-run protected core is not affordable on the
timescale 04a assumes. Options, none of which is Claude Code's to choose:
vectorised environments (the obvious first move, and the measurement above is
single-env), a coarser admissibility sample, or a shorter budget per run.

---

## 5. Acceptance tests — 03a §10

| # | Test | State |
|---|---|---|
| T1 | Froude at `U_nom` is 0.14 ± 0.01 | **xfail** — F24 |
| T2 | boundary branch differs from ground truth under pose noise | pass |
| T3 | `\|corr(e_y, b_i)\| < 0.9` for all 7 rays | pass (60 episodes; run 1000 before the freeze) |
| T4 | target invisible when fully occluded | **not written** — needs 04a §3.6's occlusion placement |
| T5 | facility walls in the raw scan, absent after the gate | pass |
| T6 | corridor boundary never in the raw scan | pass |
| T7 | crossing targets leave the corridor, others do not | pass |
| T8 | static panels classified dynamic rarely | pass |
| T9 | head reach ≤ 1.5·Lpp, or `allow_reverse` | **xfail** — F28 |
| T10 | timeout rate below 5% on stage 5 | **not written** — needs a trained policy |
| T11 | env class matches the reward gate's class every frame | pass |
| T12 | domain intrusion does not terminate | pass |

---

## 6. Generator feasibility — 04a §3.5

Rejection rate is a **result**, not a diagnostic: it says at what width each
class stops being constructible, independently of any policy. Stage 5, 200
scenarios, 190 constructed:

| Class | Rejection rate |
|---|---|
| overtaking | 0.76 |
| head-on | 0.87 |
| crossing | 0.93 |
| being overtaken | 0.96 |
| null | 0.99 |

These are high, and they are the cost of the round-trip class check — the
sampled `CT` and the realised bearing are independent, so a large fraction of
draws land in a different class than requested. That check is what keeps the
per-class partition clean. The rates also feed Study 1's feasibility envelope
directly.

---

## 7. Freeze checklist — 04a §9.3

Machine-checked in `suite.freeze_checklist()`.

| | Item | Note |
|---|---|---|
| ✅ | generator source committed | `corridor.py`, `scenario.py`, `suite.py` |
| ✅ | seed namespaces disjoint | asserted in `test_acceptance.py` |
| ✅ | suite generated and hashed | `build_tier_a()` + `manifest()` |
| ✅ | regeneration reproduces every hash | asserted |
| ✅ | throughput measured | 74 steps/s — see §4 |
| ❌ | every `TODO(04-*)` resolved or deferred | four outstanding |
| ❌ | operating speed settled | F24 |
| ❌ | claim ledger committed | including Study 1's §6 predictions |
| ❌ | empty result tables committed | matching the draft skeleton |

---

## 8. Deliberately not done

* **03a §9's reconciliation of `dynamic_obstacles/rl_env_dynamic.py`** — the
  directory does not exist. It was deleted in the first session. There is
  nothing to reconcile and the checklist has no target.
* **The scale audit (T6 in 02b's ordering)** — the accumulators and predictions
  exist; running 1,000 random-policy and 1,000 Paper 2 SAC episodes is its own
  task.
* **`T-RE`'s velocity obstacle** — a starboard alteration stands in for it. 03a
  §5.3 requires `T-RE` and the C3 comparator to be one implementation, so the
  real one arrives with the comparator.
* **Occlusion and conflict obstacle placement** (04a §3.6) — the flags are
  carried in the scenario record; the placement logic is not written.
* **Study 3's domain-randomisation ablation** — the config block exists
  (03a §8); the sweep does not.

---

## 9. Changelog

| Rev | Date | Change |
|---|---|---|
| 1 | 2026-09-06 | 01 perception and observation |
| 2 | 2026-09-08 | 02b T1–T3, constant decisions; F20, F21 |
| 3 | 2026-09-08 | 02b T4 reward, telemetry panel; F22, F23 |
| **4** | **2026-09-13** | **03a + 04a: corridor generator, targets, scenario generator, suite, acceptance tests; F24–F29** |
| 5 | 2026-09-13 | `OPEN_PROBLEMS.md` added; B10 recorded (the generator is not wired into the env) |
