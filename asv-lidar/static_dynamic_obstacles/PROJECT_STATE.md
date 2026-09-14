# PROJECT STATE — Paper 3 implementation

**A living document.** Updated on every change to the tree. It records what is
built, what each piece decides, and — the part that matters — **what is
currently blocking a full training run and a full evaluation.**

| | |
|---|---|
| **Revision** | 8 — 0.55 m/s; virtual corridors back; generator wired in; e-stop reward; noise and randomisation on; scale audit; first PPO formulation run; see `OPEN_PROBLEMS.md` |
| **Last updated** | 2026-09-15 |
| **Tests** | **406 passing, none expected to fail** (T1 passes since F24 was decided) |
| **Blocking a headline training run** | **nothing decision-side**: A15–A20 decided and built (F53, F56, F58); run 3 done (F57). Before run 4: the tiered checks, including the 5 m stop rise in F58. The full budget then waits on run 4 confirming the formulation, and on B5 (`OPEN_PROBLEMS.md`) |
| **Blocking a full evaluation** | **7 more** — B2, B5–B10 (§2) |
| **Open `TODO(decision)`** | 1 (`D_SAFE`) |
| **Open measurements** | 05 part 1 done from logs; basin sessions pending — `OPEN_PROBLEMS.md` Part B |

---

## 1. What exists

### 1.1 Modules

| Module | Owns | Spec |
|---|---|---|
| `constants.py` | every constant, one place, each with a `TODO` marker | kickoff §5 |
| `ship.py` | **the identified hull**, imported from `bluefin/`; substepping, reverse braking | 05 part 1 |
| `ship_v2.py` | the Paper 2 hull, kept for comparison only | — |
| `emergency_stop.py` | Rule 8(e) stop latch and trigger predicates — simulator-free; **the bridge imports it** | user spec |
| `path.py` | reference path, arclength, signed curvature, `r_path` | 02b T3 |
| `corridor.py` | **channel generator** — centreline, width profile, bends, polygon | 03a §3 |
| `targets.py` | **oriented hull, five behaviour models, confinement** | 03a §5 |
| `scenario.py` | **generator, backward solve, rejection ledger** | 04a §3 |
| `suite.py` | **Tier A/B, Around the Clock, Study 2, freeze manifest** | 04a §4–§9 |
| `asv_lidar.py` | 720-beam raycast, sector pooling, dead zone, aft mask | 01 §2 |
| `boundary_raycast.py` | virtual boundary scan, beam gating, pose noise | 01 §3 |
| `tracking.py` | cluster → associate → Kalman → **free-space static/dynamic test** (F37) | 01 §4, replaces 03a §6.3 |
| `cpa_cri.py` | CPA, ship domain, collision risk index | 01 §5 |
| `encounter.py` | five-class classifier + hysteresis — **one definition** | 01 §5.3 |
| `colregs/` | `EncounterContext`, engagement machine, admissibility | 02a §10.1 |
| `reward/` | eight dense terms, group clipping, audit | 02a |
| `observation.py` | five branches, 56 dims, frozen index order | 01 §6 |
| `env.py` | the Gymnasium environment — **2 Hz decisions, 0.1 s physics and collision sub-steps, optional pose staleness** | 03a §4 |
| `render.py` | field view + seven-block telemetry panel | RENDER_PANEL_SPEC |
| `play.py` | manual and random harness | — |
| `train_formulation.py` | **single-seed PPO test of the formulation** — generator curriculum, per-class evaluation | rev 8 |
| `tools/scale_audit.py` | 02a §8.2's reward scale audit over the generator | rev 8 |

### 1.2 Task status

| Task | State |
|---|---|
| 01 perception and observation | **done** |
| 02 / 02a / 02b reward | **done** — T1–T4, C1–C4 |
| 03a environment and target | **done** — §1.2, §3, §4, §5, §7, §10; **§6.3 replaced** by the free-space classifier (F37); §4.1's 0.1 s step replaced by 2 Hz (F38) |
| 04a scenario and evaluation | modules done — §3, §4, §5, §6, §7, §9 — and the generator drives training (F44) |
| 05 vessel model and sim2real | **part 1 validated and integrated**; refit v4 run, **not adopted** (F40); crash-stop block added to part 2; basin sessions pending |
| Training campaign | **blocked** — §2 |

---

## 2. What is blocking the campaign

Ordered by what they block, not by size. **The first is one decision and it
moves almost everything else.**

### 2.1 Blocking a headline training run

| # | Blocker | Owner | Effect |
|---|---|---|---|
| ~~B1~~ | ~~F24, the operating speed~~ — **decided: 0.55 m/s, `CRUISE_RPM = 6`** (F42) | — | — |
| **B2** | **F28 — Rule 8(e): resolved in simulation by the emergency stop; unverified on the water** | 05 | the field claim of 8(e) — moved to evaluation |
| **B3** | **Throughput** — 38 steps/s at 2 Hz (19 simulated s per s); one policy-seed suite ~2.3 h against 04a's 50 min (§4) | 04 | the comparator list length |
| B4 | Pose noise is **nominal**, not measured (F46) | 05 | the N1 sim-to-real claim, until S1-A |
| ~~B11~~ | ~~Phantom dynamic tracks~~ — **fixed** (F37); confirm as §6.3's replacement (`OPEN_PROBLEMS.md` A7) | — | — |
| ~~B12~~ | ~~Spawn boundary penalty~~ — **fixed** (F35) | — | — |
| ~~B13~~ | ~~fixed corridor, 2 Hz reward~~ — **corridors restored, reward confirmed** (F43) | — | — |

### 2.2 Additionally blocking a full evaluation

| # | Blocker | Owner | Effect |
|---|---|---|---|
| **B5** | Curriculum steps per stage and total budget (`TODO(04-3)`, `TODO(04-4)`) | 04 | cannot size the campaign |
| **B6** | Claim ledger and empty result tables not written | you | 04a §9.3 freeze checklist |
| **B7** | `T-RE` reactive target is a placeholder, not the VO comparator | 03 | Tier B's `re` behaviour stratum |
| **B8** | Classical comparators (LOS-PID+DWA, COLREGs-VO, encounter VO) do not exist | 04 | every comparison in the paper |
| **B9** | `metrics.py` does not read the reward keys | 04 | 04a §10's metric set |
| ~~B10~~ | ~~the scenario generator is not wired into `env.py`~~ — **wired** (F44) | — | — |

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

**Status: DECIDED (revision 8) — 0.55 m/s, `CRUISE_RPM = 6` (F42).** The history follows.

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

**Status (revision 8): at 0.55 m/s the stop meets T9 for reverse efficiency ≥ 0.05 (≥ 0.065 with a 0.73 s thrust delay).** **Status (revision 7): resolved in simulation at the vessel's 2 Hz** for reverse efficiency ≥ 0.27 (≥ 0.55 if thrust shares the rudder's 0.73 s delay); basin block S1-C2 measures both. **Status (revision 6):** the emergency stop (full astern until stopped, then zero thrust) met T9 at the 0.1 s control step for any reverse efficiency >= 0.19. Coasting on the identified hull takes 8.5 m, not the 29.9 m below -- the v2 hull under-predicted drag. On the water the stop needs reverse efficiency >= 0.26 at 2 Hz, or >= 0.56 if thrust shares the rudder's 0.73 s delay; neither is measured. The original analysis follows.

**Status (revision 5): OPEN. This is B2.**

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

**Revision 8:** the width sweep is back, so this stands again.

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

**Revision 7:** at 2 Hz the gate is 1.40 m and the knee moved out — head-on
tracked frames are flat at 7 of 11 through 0.25 m, 4 at 0.50 m, 0 at 0.75 m.

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

### 3.12 Revision 6 — 05 part 1, the emergency stop, and what they exposed

Full detail in `bluefin/REVIEW.md` (the model) and `OPEN_PROBLEMS.md` (the
decisions). Summary:

**F30 — the identified hull is in, and it settles what the vessel does.**
`src/ship.py` imports `bluefin/` rather than copying it, and forward-thrust
trajectories are asserted bit-identical to the report's `ShipModel`. `U_REF` is
now derived from the plant — `sqrt(T12/X_uu)` = **1.116 m/s** — and replaces
`THRUST_CAL`. That agrees with 02b T1's independent log median (1.14 m/s) to 2 %,
so cruise at 12 RPM is no longer in question; F24 is purely a choice of operating
point, and 0.55 m/s is exactly `CRUISE_RPM = 6`. The bridge's 50 %/s rudder
command limiter is now applied every step, and hull randomisation from the
bootstrap is available (off by default).

**F31 — phantom dynamic tracks on 28–33 % of frames, and T8 never tested it.**
Static panels are promoted to dynamic tracks as the vessel passes them: the
cluster centroid slides across the visible face. Median 0.32 m from a panel,
apparent speed 0.17–0.24 m/s (p90 0.5–0.6), with pose noise off. It happens on
the v2 hull too (27 %); T8 passed only because v2 covered 2.4 m in the test
window and never reached the panels. No threshold or hysteresis setting reaches
03a's 1 in 10⁴ (best: 6.5 %). It also **fired the emergency stop falsely** — 5
stops in 18 target-free episodes — so the stop's trigger defaults to manual.
T8 is rewritten to run 300 steps and is a strict xfail.
**Correction:** §1.2 previously listed 03a §6.3 as done. It was not
implemented; the tracker still uses 01's 0.15 m/s threshold and a 5-step hold.

**F32 — pose staleness is 40.8 %, not every frame.** 443 of 1,085 July frames
used the previous frame's pose, because the pose line races the LiDAR line at
the bridge. The identification report models it as every frame. The
`DeploymentTiming` wrapper uses the measured rate.

**F33 — the report's holdout claims are narrower than stated.** Naive predictors
also beat v3 at 2 s and in free run; the V6 drift regression flipped sign rather
than going away; and three training segments contain several seconds of motion
against the heading (most likely un-trimmed retrievals). Nine corrections in
`bluefin/REVIEW.md` §5.

**F34 — throughput was overstated about 4×.** Revision 5's 74 steps/s came from
a target-free narrow corridor. The default environment measured 13–14 steps/s;
fixing my confinement check (polygon rebuilt per call, hull vertices tested one
at a time — 40 % of every step) brought it to 19–21. The hull swap itself costs
about 8 %.

**F35 — every episode starts with a boundary penalty.** The path is inset from
the corridor's start edge by exactly the inflated hull half-length (1.0125 m),
so the stern spawns on that edge, inside `d_safe`. `r_bnd` is negative for the
first 16–19 steps: **−22.9 per episode** on average, against 02a §8.1's
predicted zero. Corrupts the scale audit. Recorded, not yet fixed
(`OPEN_PROBLEMS.md` C10).

**F36 — `corridor_width` ignores a supplied channel.** An explicit 8 m channel
reports 10 m (20 B). `W_local` and the reward are unaffected; the labels on
named cases and Study 1 levels are wrong (`OPEN_PROBLEMS.md` C11).

**The emergency stop** (user specification): full astern `S2 = −100` until speed
≤ 0.05 m/s, then zero thrust, then back to the policy. Reverse thrust is modelled
by operator splitting because the identified model has none; a release condition
and a braking time-out were added because the specification had no exit.
`field_deployment/udp_live_rl.py` is **not** edited — it drives the real vessel —
and needs three changes before the stop exists on the water
(`OPEN_PROBLEMS.md` C8).

### 3.14 Revision 7 — 2 Hz, the fixed corridor, the tracker, the refit, the bridge

**F37 — phantom tracks fixed by a free-space consistency test.** A static solid
cannot occupy space a ray has seen through, nor leave space a ray now sees
through, from any viewpoint; the cluster centroid's slide is invisible to that
test. Over a 2 s window a track shows motion when its returns *appear* where an
earlier ray returned from beyond, or *vacate* space a ray now returns from beyond.
Two defects were found on the way: returns were lifted from the vessel origin
instead of the sensor 0.86 m ahead (static objects moved when the vessel turned),
and the first version treated an empty beam as free space — the C1's 1 m dead
zone makes that false, and T8 caught it at 1.1 %. The second version checked the
tolerance only along the ray; T8 with 3 cm of pose noise caught that at 0.17 %,
on panel faces seen edge-on, and every beam within the tolerance's lateral width
must now clear the point too.

| Classifier | Phantom frames, target-free, three panels | With 3 cm / 0.2° pose noise |
|---|---|---|
| 01's centroid speed, at 2 Hz | **23.9 %** (959 of 4,015) | — |
| free space v1 — an empty beam counted as free | 1.1 % | 1.5 % |
| free space v2 — returned beams only, tolerance along the ray | 0 of 12,012 | 0.17 % |
| **free space v3 — tolerance laterally too (current)** | **0 of 12,012** | **0 of 12,012** |

Zero in 24,024 frames bounds the rate at about 1.25 in 10⁴ at 95 %.

Detection is essentially unchanged. Head-on, crossing, slow overtaken and fast overtaking targets placed around the own ship at cruise, 25 seeds each:

| | head-on | crossing | overtaking | being overtaken |
|---|---|---|---|---|
| promoted, median (p90) delay — speed classifier | 2.0 (4.0) s | 2.0 (4.0) s | 2.0 (3.3) s | 3.0 (4.5) s |
| promoted, median (p90) delay — **free space** | 2.0 (4.0) s | 2.5 (4.5) s | 2.5 (4.0) s | 3.0 (5.8) s |
| never promoted among 3 panels, of 25 — speed / **free space** | 6 / 7 | 0 / 0 | 0 / 3 | 15 / 15 |

No target is missed without panels. The being-overtaken misses are the scene itself -- occluded or colliding early -- under both classifiers. Slow overtaking targets among panels are the one place the free-space test costs recall.

**F38 — everything runs at 2 Hz** (your decision). `UPDATE_RATE = 0.5`; every
step count is a duration through `steps_for`; every dense reward weight is × 5 so
per-second rates and 02a §8.1's integrals are unchanged; the training discounts
are converted (0.99 → 0.951). Physics and collision are sub-stepped at 0.1 s,
because a closing target moves 1.3 m per decision. Measured pose staleness is
native: a stale frame repeats every pose-derived observation and skips the
tracker, and the `DeploymentTiming` wrapper is gone. Default 0.0 since the
bridge waits for each pose line (F41). The emergency stop's
break-even reverse efficiency at 2 Hz is **0.27** (0.55 with a 0.73 s thrust
delay), and the astern speed the latch would impart after the vessel stops is
reported per step because the hull cannot simulate it.

**F39 — the corridor was fixed at 10 m** (your decision; **reversed in revision 8**, F43). Every configured width
is the basin. Consequences in `OPEN_PROBLEMS.md` A11: Study 1's sweep and the
width-driven half of N2 are gone (all four predicted transitions lie below
10 m); no bend fits, so `r_path` is always zero; Tier B realises 13 of 39 cells;
28 of 34 Tier A cases no longer differ in geometry; and a crossing target had no
water outside the corridor to start from, so that containment rule is relaxed
at a basin-wide corridor. Acceptance T3 still passes, decorrelated by the path offset alone.

**F40 — the refit, and the retrievals confirmed.** The July videos are the
bridge's own display; its telemetry at t = 21 s of `trial_2` reads surge −0.60 m/s
under S2 = 62.5 forward thrust, so the three windows are hauled retrievals.
Refit v4, pre-registered (trim, `N_uv` widened, same pipeline): better than v3 at
every windowed horizon from 2 s and in position, but **A5 passed 11 of 12 draws,
so v4 is not adopted** and the simulator keeps v3. Six parameters sit at bounds
and the yaw and actuator parameters moved by more than their bootstrap
intervals. `fit_final.py` documents `N_rr` as fixed at zero and bounds it
[0, 60]; post-hoc v4b fixes it: best heading of any fit, the first to beat the freeze baseline in free run (21.1° against 22.9°), but path-length error 37 % worse and A5 11 of 12, so **also not adopted**. Recommendation: keep v3, and carry v4b's structure (`N_rr` fixed) into the basin session-1 fit, where coast-downs and turning circles pin what these logs cannot.

**F41 — the bridge latch.** `udp_live_rl.py` now imports `emergency_stop.py`:
E latches full astern (S2 = −100) until **signed** surge reads stopped, then 0;
R releases after the minimum hold. The policy's range is still 0–100, and the
latch is the only path below zero. **Then, on your go-ahead, the bridge's two
parity defects were fixed:** the rudder limiter is a true 50 %/s rate limit when enabled (it was overwritten to off, and moved half the error per frame for small commands), and is **off by default in bridge and simulator alike**, since the identified model predicts the July runs that sent raw commands at least as well as limited ones, and `PoseSync` makes each frame wait for its own pose line — replayed through every July log, **0 of 1,085 frames stale** against 443 without it, at 6 ms worst added latency. `--rudder-limit` and
`--no-pose-sync` switch the other behaviour back for comparison runs.
The simulator's `POSE_STALE_PROB` is now 0.0; the measured 0.408 is kept as
`MEASURED_POSE_STALE_PROB`.

**F35 and F36 are fixed.** The own ship spawns 1.41 m in (half inflated hull +
`d_safe` + 0.05), and the scenario generator honours the same inset; a supplied
channel reports its own width.

**The crash-stop block** is in `PART2_BASIN_PLAN.md`: S1-C2 (RPM 9/12/15 × 3,
bridge latch), a P-8 bench check that the ESC reverses at all, thresholds
T-9–T-11, three holdout stops in S2-B, and O-6/O-7 on reversal safety. S1-E gave
up ten minutes for it.

### 3.15 Revision 8 — your decisions, the generator, the audit, the formulation run

**F42 — the operating speed is 0.55 m/s.** `CRUISE_RPM = 6`, so `U_REF` =
`steady_speed(6)` = **0.558 m/s** (Fr 0.142; acceptance T1 now passes). The
propulsion stages keep their authority relative to cruise (stage 4: 0–12
rpm-units, stop to 1.116 m/s). Everything derived follows: `N_REF_PROG` 71.7,
`TRACK_GATE_DIST` 0.70 m, the spawn TCPAs return to 04a's table values. Two
consequences worth stating in the paper:

* the emergency stop meets T9 for reverse efficiency ≥ 0.05, or ≥ 0.065 with a
  0.73 s thrust delay, against 0.27 and 0.55 at 1.1 m/s — the unmeasured
  reverse thrust no longer decides whether 8(e) is possible;
* detection is **faster**, not slower: every class is promoted within a median
  1.0–2.0 s with no panels, tracked on 86–92 % of visible frames.

**F43 — the narrower virtual corridors are back**: the 10–3.5 m sweep, width
variation and bends (revision 7's fixed 10 m reversed). The crossing-containment
relaxation added then stays, and only applies to basin-wide draws.

**F44 — C1: the scenario generator drives the environment.**
`ASVLidarEnv(scenario_stage=s)` draws each episode from `ScenarioGenerator` in
the episode's seed namespace. The own ship starts **at `U_nom` on the scenario
heading** — 04a's backward solve assumes it, and starting from rest moved every
CPA — the target gets its class, behaviour and confinement, and generated
obstacles are dropped if they sit within ±0.4·T₀ of own-ship travel around the
CPA or within 2 m of the target spawn. `set_scenario_stage` switches the
curriculum at the next reset; `reset(options={"generated": scenario})` replays a
given scenario. The placeholder `_sample_target` remains only for the older tests.

**F45 — A8: the emergency stop in the reward.** While the latch holds, `r_pf`'s
speed gate is suspended (the vessel was stopped, it did not choose to stop), and
each stop costs `R_ESTOP = −20` once, logged as `reward/intervention`. A
validator keeps it in `(R_COLLISION, 0]`.

**F46 — nominal noise and hull randomisation for training.** Pose 3 cm / 0.2°
per frame, no walk; ego 0.05 m/s and 1 °/s — all `TODO(05)` until S1-A. Hull
randomisation at 1.0: A1–A3 pass on 12/12 draws at 1.0, 10/12 at 1.5 (2.0 passed
12/12 by chance of the draws). The free-space tracker's tolerance stays at 0.25 m.

**F47 — the scale audit** (`tools/scale_audit.py`, 240 stage-5 episodes per
policy, nominal hull):

| episode type | mean return |
|---|---|
| nominal success, no target (follower) | **+172** |
| success with a COLREGs penalty | +101 |
| any collision | **−337** |

02a §8.1's orderings hold. At the design point progress (+101 vs +107 predicted),
existence (−17 vs −15) and obstacle proximity (−12 vs −26) are within the factor
of 3; `bnd`, `dom` and `col` are exactly zero, confirming F35's fix. `pf` and
`smooth` come in near zero only because a scripted follower tracks a straight
path perfectly. One finding needs a decision (`OPEN_PROBLEMS.md` A15):
**being-overtaken episodes collide a stand-on vessel** in about a third of draws,
because the overtaker is constant-velocity and its DCPA starts at 0.

**F48 — the single-seed PPO formulation run** (`src/train_formulation.py`):

PPO, seed 0, 10 workers, curriculum 1→5, propulsion stage 4 (0–12 rpm-units).
It stopped at **1.91 M of 2 M steps** after 5.6 h on a `PermissionError` in
`monitor.csv` (OneDrive's sync client held the file); the checkpoints to
1.75 M and all nine evaluations survived. The evaluation is 36 fixed
development scenarios, 6 per class, so each class rate moves in steps of 17 pp.

| | 0.2 M | 0.6 M | 1.0 M | 1.6 M | 1.8 M |
|---|---|---|---|---|---|
| goal | 0.72 | 0.83 | 0.81 | 0.86 | 0.81 |
| collision | 0.28 | 0.17 | 0.19 | 0.14 | 0.19 |
| crossing collision | 0.50 | 0.67 | 0.67 | 0.50 | 0.50 |
| COLREGs integral | −15.5 | −13.6 | −18.3 | −15.1 | −12.0 |

Training episodes over the last 0.9 M steps (stage 5) tell a worse story, and
it is the truer one:

| class | goal | collision |
|---|---|---|
| crossing | 0.47 | 0.53 |
| head-on | 0.63 | 0.37 |
| overtaking | 0.75 | 0.25 |
| no target | 0.82 | 0.18 |
| being overtaken | 0.74 | 0.58, of which 0.34 is F49 |

Training return peaked in stage 4 and declined through stage 5 while
`approx_kl` climbed from 0.01 to 0.25, clip fraction from 0.06 to 0.53 and
action std from 0.98 to 0.27. Replaying the 1.75 M checkpoint on the
development set shows why the numbers cannot yet be read as a verdict on the
formulation: **the policy holds full throttle in every class**, at 0.9–1.1 m/s,
1.9 × cruise. Timeouts: none. Emergency stops: 0.08–0.17 per episode.

What it does and does not say:

* **Not yet interpretable: crossing.** Every crossing collision is with the
  target, 6–10 s in, at about 1 m/s. 04a's generator solves the encounter for
  the own ship at `U_NOM`; at 1.9 × that, the CPA it built is not the CPA that
  happens. Re-read after F50.
* **Being overtaken:** most of the "collisions" were F49. The remainder
  (~0.24) is A15's DCPA question, and it is also confounded by speed. At
  1.05 m/s the own ship is faster than slower overtakers, so they cannot
  overtake it.
* **What worked:** no-target and null classes reach 0.83–1.0 goal on the
  development set; head-on 0.83–1.0; overtaking 0.83–1.0 from 1.0 M steps on;
  the curriculum transitions did not collapse learning.

**F49 — the being-overtaken goal sat on the corridor's end edge.**
`own_start_s` gave this class all of `length − 20` m as astern room, so the
20 m path ended, and the goal sat, exactly on the end edge. The goal test
runs at the decision instant and the collision test on every sub-step. So a
fast ship's bow crossed the edge inside the step that reached the goal, and the
reward, which ranks collision first, paid −300 for arriving. It was 34 % of all
being-overtaken training episodes. The development evaluation hid it by
labelling goal before collision. **Fixed:** the path end keeps
`GOAL_END_INSET_M` = 0.37 m (derived: a full step at the propulsion ceiling
after the last failed goal test). Using the full `SPAWN_INSET_M` instead cut
the class's generator yield from 200 to 129 in 200; 0.37 m keeps 198. Test:
`test_the_being_overtaken_goal_is_clear_of_the_corridor_end`.

**F50 — the speed gate is two-sided (provisional, A16).** With a one-sided
gate, speeding cost nothing in `r_pf` or `r_prog`. Meanwhile the existence
cost, the 10 s discount horizon and an earlier +100 goal all paid for it, so
full throttle was the optimum. `r_pf`'s gate is now multiplied by
`overspeed_gate(u, U_REF)`: 1 up to 1.2 · `U_REF`, falling linearly to 0 at
1.7 · `U_REF`. It is measured against the nominal `U_REF`, not `U_ref_eff`,
so R-2 and R-5 still make slowing free without making cruising costly. Tests:
`test_r_pf_charges_speeding_above_the_tolerance`,
`test_the_overspeed_gate_ignores_the_lowered_reference`.

**F51 — the training harness.**

* The evaluation now labels collision before goal, as the reward does.
* `mean_speed` is the episode mean. It was the final step's speed.
* `RetryingVecMonitor` retries a locked CSV write.
* `--runs-dir` puts long runs elsewhere, and `--tag` names them. Runs 2 and 3
  were written to `C:\Users\hntran\asv_runs` and moved back into `runs/` on
  2026-09-15; outputs now stay inside the project.
* PPO `target_kl` = 0.03 stops a KL runaway like run 1's.

Also: piping the script's stdout through `grep | tail` from a backgrounded
shell deadlocked a smoke run on its first log table. Always redirect to a
file.

Tests: **400 passed** (397 + 3).

**F52 — formulation run 2** (F49–F51 in; `runs/ppo_formulation_seed0_v2`).
The full 2 M steps completed in 6.2 h. The evaluation is 20 development
scenarios per class, 120 per evaluation. Geometry joins are in
`results/formulation_run2/join_geom.txt`.

*The fixes held.*

* Evaluation mean speed was 0.52–0.66 m/s (run 1: 0.9–1.0).
* Goal and collision on the same step fell from 0.34 to 0.01–0.02 of
  being-overtaken episodes.
* `approx_kl` stayed at 0.03–0.047 (run 1: 0.25) and action std fell to 0.50.
* fps still fell from 201 in stage 1 to 90 in stage 5 with the run outside
  OneDrive, so the fall is the environment's cost, not sync.

*Outcomes.* Evaluation at 2.0 M: goal **0.77**, collision 0.23, no timeouts.
It was still rising: 0.57 at 0.2 M and about 0.66 at 1.0–1.8 M.

| class (eval at 2.0 M) | goal | collision |
|---|---|---|
| head-on | 0.85 | 0.15 |
| crossing | 0.55 | 0.45 |
| overtaking | 0.90 | 0.10 |
| being overtaken | 0.60 | 0.40 |
| null | 0.80 | 0.20 |
| no target | 0.90 | 0.10 |

The run-1 and run-2 evaluation numbers are not comparable. Run 1 used 6 per
class, labelled goal first, and ran at 1.9 × cruise. The training episodes over
the last 0.5 M steps agree with the evaluation's ordering: crossing 0.60
collision, being overtaken 0.55, head-on 0.40, overtaking 0.28, no target
0.20.

*Where the collisions are* (evaluations at 1.6, 1.8 and 2.0 M, 60 per class,
joined to the drawn geometry):

| case | DCPA ≤ 0.7 m | 0.7–1.4 m | > 1.4 m |
|---|---|---|---|
| crossing, target to starboard (give-way) | 0.88 (n 8) | 0.83 (12) | 0.29 (24) |
| crossing, target to port | 0.17 (12) | 0.50 (12) | **0.75** (12) |
| being overtaken | 0.69 (32) | 0.39 (28) | 0.50 (20) |
| head-on, width ≤ 7 m | 1.00 (4) | 0.50 (8) | 0.88 (8) |
| head-on, width > 7 m | 0.05 (20) | 0.06 (32) | 0.00 (8) |
| overtaking | 0.19 | 0.18 | 0.25 |

Replaying the final model with the nominal hull on the development set
explains three of them:

* **A port-side crossing is scored as give-way with a starboard turn sense.**
  S3 / modification 1 collapses both sides into one crossing class, and
  `_TURN_SENSE[CROSSING] = +1`. So `v_port` charges the port turn that passing
  astern of a target from port needs, and a starboard turn carries the own ship
  along the target's track. The policy does exactly what it is paid for: +27°
  to +41° to starboard before CPA in 8 of 9 port crossings. Collisions *rise*
  with DCPA, the signature of an alteration creating a collision out of a clear
  pass. This is a locked decision, so it is **A17**, not a fix.
* **Being overtaken: the policy runs.** Throttle mean up to 0.9, speed about
  1.0 m/s, swerves of 20–50° before CPA, and 8.8 `v_hold` frames per episode.
  In 10 m corridors that ends in the wall: 10 of 20 collisions there are
  boundary. It is A15 made concrete. With a constant-velocity overtaker and
  DCPA under about 0.65 m, standing on is fatal, the observation cannot tell
  the agent which draw it is in, and −300 outweighs `v_hold` and the F50
  over-speed gate together.
* **Give-way crossing at DCPA < 1.4 m** fails 83–88 %. In 3 of the 5 collisions
  replayed the policy turned *to port* (−13° to −34°). That is the wrong way,
  and `v_port` does charge it. It looks like a learning problem, not a
  formulation defect, but it shares the class one-hot with A17's port case. So
  until A17 is resolved the agent is taught opposite geometry under one label,
  which may be why it cannot learn either.
* **Narrow head-on (≤ 7 m) collides 75 %**, against 5 % above 7 m, although
  the head-on width threshold is 3.8 m. Not yet diagnosed (C14).

**F53 — your decisions on A15, A16 and A17: option 1 for all three.**

* **A17, a crossing's turn sense is side-dependent.**
  `compliant_turn_sense(cls, crossing_side)` returns −1 for a crossing from
  port and +1 from starboard. The engagement latch and the idle path both pass
  `ctx.crossing_side`, so the side is fixed at engagement. `v_port`, `v_r8`'s
  alteration credit, `turn_admissible` (now `A_port` for a port crossing) and
  R-2's slowdown all follow the sense, with no other change. S3 still makes
  the own ship give way either way, and the observation still has one crossing
  class. Test: `test_5b_a_crossing_from_port_requires_a_port_turn`.
* **A15, the being-overtaken DCPA is floored.** 80 % of draws are uniform on
  [`BEING_OVERTAKEN_DCPA_FLOOR` = 1.0 m, 2.0 m]. The other
  `BEING_OVERTAKEN_BELOW_FLOOR_FRAC` = 20 % are uniform on [0, 1.0 m) and
  labelled `scenario.dcpa_below_floor` as the Rule 17(b) case. The label is
  an attribute, not a record field, so scenario hashes are unchanged. Only
  being-overtaken draws consume the extra random number. Test:
  `test_the_being_overtaken_dcpa_is_floored_with_a_labelled_fraction_below`.
* **A16, the two-sided speed gate stays** (F50); the constants lose their
  `TODO(decision)`.
* The formulation evaluation now records `dcpa_m`, `ct_deg` and
  `dcpa_below_floor` per episode.

Tests: **402 passed** (400 + 2). Run 3 (`runs/ppo_formulation_seed0_v3`) trains with all three.

**F54 — C14 diagnosed: narrow head-on collisions are the emergency-stop
supervisor, through domain-based admissibility.** Scripts and data are in
`results/c14_narrow_head_on/` (scripts in `tools/diagnostics/`). There were 100 head-on scenarios (20 each at 5, 6, 7, 8 and
10 m), with obstacles off and the nominal hull.

* *Run 2's 0.75 was 5 scenarios*, all on bends. At ≤ 7.5 m every sampled
  head-on corridor bends (the F25 ceiling allows it and the bend debt forces
  it), and at ≥ 7.7 m none do. So width and bend were confounded.
* *The target clamp is not it.* `clamp_to_corridor` teleports a
  wall-touching target to the nearest centreline station and snaps its
  heading. That departs from 03a §5.2's Rule 9(a) channel-keeping. But with it
  disabled, target collisions moved within noise for both a path follower and
  the run 2 model, and median first snap is at 26 s, after CPA. It should
  still be replaced (C14).
* *Room is not it.* Starboard room from the Rule 9(a) station to the wall is
  1.8 m at 5 m against a median 1.0 m shift for hull clearance: feasible in
  90–100 % of draws.
* *Bends are not it.* Holding width, bent and straight corridors do not
  differ consistently.
* **The supervisor is.** `emergency_stop.stop_required` fires on an engaged,
  give-way, in-extremis encounter whose compliant turn is inadmissible.
  `A_stbd` compares starboard room with `Dy_req = d_req − DCPA`, a domain
  separation (`d_req` = 2.5 m). At 5 m that is "inadmissible" for DCPA below
  about 1.3 m, and the starboard turn was inadmissible in 66 % of pre-CPA
  frames at 5 m, 59 % at 6 m, 27 % at 7 m and 0 at 8–10 m. The ship then
  stops ahead of a constant-velocity reciprocal target, which a stop cannot
  clear. 74 % of target collisions had stopped before CPA (10 % of
  non-collisions); run 2's narrow head-on collisions averaged 1.08 stops, the
  rest 0.

| run 2 model, target collision | 5 m | 6 m | 7 m | 8 m | 10 m |
|---|---|---|---|---|---|
| supervisor on | 0.43 | 0.25 | 0.24 | 0.16 | 0.05 |
| supervisor off | 0.24 | 0.15 | 0.14 | 0.11 | 0.05 |

Switching the supervisor off saved 10 collisions and added 1. Decision:
`OPEN_PROBLEMS.md` A18.

**F55 — regression: A17 flips head-on encounters to a port turn sense.** Class
and `crossing_side` are computed from the own ship's instantaneous heading. A
starboard alteration for a head-on moves the intersection angle out of the
head-on band. The encounter re-classifies as a crossing with the target on the
port bow, and since A17 `compliant_turn_sense` latches −1 on that class
switch. On the same 100 head-on scenarios with the current code:

* the run 2 model latched a port sense at some point in **67 %** of episodes,
  with **43 %** of engaged pre-CPA frames carrying it;
* a non-avoiding path follower did so in 20 % of episodes (6 % of frames).

F53's unit test checks the function and the latch, not this interaction.
Run 3 trains with it. Decision: `OPEN_PROBLEMS.md` A19.

**F56 — A18 and A19 built (option 1 for both); A18 works, A19 halves the
regression but leaves a residual.**

*Built.*

* `EncounterContext.dcpa_if_stopped` is the perceived DCPA with the own ship
  stationary, `rng·|sin(α − ct)|` while the target approaches.
  `stop_clears` compares it with `ESTOP_CLEAR_DCPA_M` = 0.5·LOA + 0.5·B +
  2·0.15 + `D_SAFE` = 1.76 m.
* `emergency_stop.stop_required` requires `stop_clears`. So does R-2's
  slowdown carve-out in `effective_speed_reference`, and `r8_parts` credits
  the alteration when a slowdown cannot clear.
* `ContextManager.update` classifies and takes `crossing_side` (and the true
  class) against the path tangent at the own ship, via `_path_heading`. The
  CPA products and the Rule 8 accumulator keep the instantaneous heading, and
  open water falls back to it.
* Tests: `test_the_supervisor_stops_only_when_stopping_clears`,
  `test_5c_the_class_is_judged_against_the_path_not_the_momentary_heading`,
  `test_5d_a_slowdown_that_cannot_clear_is_not_the_paid_8e_answer`.
  **405 passed.**

*Verified* on the C14 head-on set (`tools/diagnostics/a18_a19_verify.py`,
`results/c14_narrow_head_on/a18_a19_verify.*`). This is the run 2 model,
trained under the old rules, so it measures the mechanism, not a retrained
policy:

| width | 5 m | 6 m | 7 m | 8 m | 10 m |
|---|---|---|---|---|---|
| target collision, before → after | 0.43 → **0.33** | 0.25 → **0.10** | 0.24 → **0.19** | 0.16 → 0.11 | 0.05 → 0.05 |
| stops per episode, after | 0.05 | 0 | 0 | 0 | 0 |
| median minimum speed, after | 0.52 | 0.53 | 0.52 | 0.54 | 0.55 m/s |

*The A19 residual.* A port sense still latches in 30 % of the model's head-on
episodes (was 67 %), and in **24 % of a non-avoiding follower's** (was 20 %).
So it is not the own ship's manoeuvre (`tools/diagnostics/a19_residual.py`,
`results/c14_narrow_head_on/a19_residual_*`):

* on the follower's port-sense frames the target is at a median **3.6 m**, at a
  path-relative bearing of **18°**, outside the ±10° head-on bearing band. A
  reciprocal target 1–2 m to one side leaves that band as range closes;
* the **perceived** heading-intersection deviation is a median 29° against 8°
  true, with 62 % of frames more than 5° out. The tracker's course estimate
  degrades at close range, where the cluster centroid moves along the hull as
  the aspect changes;
* 62 % of those frames are truly head-on (true deviation ≤ 10°). Straight
  corridors are affected more than bent ones (6 m: 0.62 straight, 0.33 bent),
  so it is not the bend;
* the mechanism is the engagement latch. `_advance_state` re-engages, and
  re-latches the sense, whenever a different class persists for
  `N_SWITCH_STEPS` while engaged. So close-range bearing drift and tracker
  course noise re-decide a situation that COLREGs fixes when risk first
  develops.

Decision: `OPEN_PROBLEMS.md` A20. The tracker's close-range course estimate is
C15.

**F57 — formulation run 3 (A15–A17 in; before A18–A20).** Same seed, budget and
evaluation as run 2. `runs/ppo_formulation_seed0_v3`, compared by
`tools/diagnostics/run3_vs_run2.py` → `results/formulation_run2/run3_vs_run2.txt`.
2 M steps; PPO diagnostics unchanged (last-10 `approx_kl` 0.036, clip
fraction 0.22, 87 fps).

| | run 2 | run 3 |
|---|---|---|
| evaluation goal at 2.0 M (final) | 0.77 (0.70) | **0.79** (0.74) |

Late evaluations (1.6 M – final, 80 episodes per class) and training episodes
after 1.5 M:

| class | eval collision, run 2 → 3 | training collision, run 2 → 3 |
|---|---|---|
| being overtaken | 0.54 → **0.35** | 0.55 → **0.23** |
| crossing | 0.51 → **0.34** | 0.60 → 0.55 |
| null | 0.25 → 0.16 | — |
| no target | 0.10 → 0.08 | 0.20 → 0.17 |
| overtaking | 0.20 → 0.24 | 0.28 → 0.23 |
| **head-on** | 0.22 → **0.32** | 0.40 → 0.35 |

*A17 worked for crossings from starboard, not yet for far crossings from port.*

| crossing collision, late evals | DCPA ≤ 0.7 m | 0.7–1.4 m | > 1.4 m |
|---|---|---|---|
| from starboard, run 2 → 3 | 0.88 → **0.25** | 0.83 → **0.50** | 0.29 → **0.12** |
| from port, run 2 → 3 | 0.17 → 0.17 | 0.50 → 0.33 | 0.75 → **0.83** |

The port cells are 3 distinct scenarios each, evaluated four times, so the
0.83 is weak evidence either way. It is also the geometry where F56's
close-range re-classification acts, and run 3 trained without A19 or A20.

*A15 worked.* Being-overtaken collision in run 3 by drawn DCPA: 0.55 below
the floor (n 20, the labelled Rule 17(b) share), **0.19** at 1.0–1.4 m and
**0.32** above 1.4 m. Run 2 (F52) was 0.69 / 0.39 / 0.50. Boundary collisions
in this class fell from 12 to 1. But the policy still accelerates: mean
speed 0.86–0.89 m/s against run 2's 0.77, with about 7 `v_hold` frames per
episode, and obstacle collisions rose from 9 to 16. It avoids by outrunning
rather than by standing on.

*Head-on got worse, as F55 predicted.* 0.22 → 0.32: run 3 trained with A17's
port sense re-latching on head-on encounters.

*What it says for run 4:* A18–A20 are the head-on fixes, and run 3 could test
none of them. Being overtaken is now a speed problem rather than a geometry
one — `v_hold` at 0.45 of the COLREGs group does not outweigh outrunning.
That is worth reading again in run 4 before touching the weight.

*Housekeeping.* Runs 2 and 3, the smoke run, their TensorBoard logs, the C14
diagnostics and the test logs were moved from `C:\Users\hntran\asv_runs` into
`runs/`, `results/` and `tools/diagnostics/` (76 files, 71.8 MB, sizes
verified), and the home-folder copy removed. `planning/READING_LIST.md`
lists the literature behind every method in the tree.

**F58 — A20 built (option 1): an engaged encounter keeps its class, side and
turn sense; the port-sense regression is gone.**

*Built.* `ContextManager._advance_state` no longer re-engages on a class switch.
`_engage` latches the class, `crossing_side` and turn sense, and while an
encounter is engaged or clearing `ctx.cls` and `ctx.crossing_side` are the
latched values. The observation one-hot and every reward gate therefore still
read one field. `N_SWITCH_STEPS` is retired, its value kept. Test:
`test_5e_an_engaged_encounter_keeps_its_class_and_sense` — a head-on whose
geometry drifts into a crossing from port keeps `head_on` and a starboard
sense, while the classifier's own held class has moved. **406 passed.**

*Verified* on the C14 head-on set (`tools/diagnostics/a18_a19_verify.py`;
the pre-A20 output is kept as `a18_a19_verify_pre_a20.*`):

| head-on episodes latching a port sense | original | A19 | **A19 + A20** |
|---|---|---|---|
| run 2 model | 0.67 | 0.30 | **0.05** |
| path follower | 0.20 | 0.24 | **0.06** |

Target collisions for the run 2 model are unchanged from F56 (0.33, 0.10,
0.19, 0.11, 0.05 at 5, 6, 7, 8, 10 m). A20 corrects what the reward pays; it
cannot change what a policy trained under the old rules does. The remaining
~5 % engage *as* a crossing, so no latch can correct them; that is C15's
tracker course error at engagement range. One thing to look at: stops at 5 m
rose from 0.05 to 0.38 per episode. The likeliest reading is encounters now
held in a class whose stop clears, which a Tier-1 replay should confirm
before run 4.

*Throughput benchmark* (`results/throughput/`). Three 25.6 k-step stage-5
training runs, back to back on an otherwise idle machine:

| configuration | fps (last three logs) | wall time |
|---|---|---|
| 10 workers, default PyTorch threads (as runs 1–3) | **120–126** | 239 s |
| 10 workers, learner 4 threads, workers 1 | 106–111 | 278 s |
| 8 workers, learner 4 threads, workers 1 | 78–82 | 401 s |

The oversubscription hypothesis is **refuted**. Although policy inference alone
is 1.9× faster on one thread (3,079 against 1,645 calls/s), capping threads
slowed training, and fewer workers slowed it further. So throughput is set by
the environment step. One stage-5 environment runs 43 steps/s against 98 in
stage 1, and the 10-worker rate is close to linear in workers. The flags
(`--torch-threads`, `--fixed-stage`) stay, defaulting to the old behaviour.
Next: profile one stage-5 step (C2).

### 3.13 Earlier findings, still standing

| # | Finding | Status |
|---|---|---|
| F20 | 0.55 m/s was an unsourced placeholder adopted as measurement | see F24 |
| F21 | the domain floor excluded the provisional domain | resolved: `d_abeam` = 1.25 m |
| F22 | `N_ref = 250` inverts `R-9` at the measured speed | resolved: derived from `U_REF` |
| F23 | `d_safe = 0.50` breaches 02a §2's own invariant | **open** — `D_SAFE` = 0.35, the last `TODO(decision)` |
| F23b | the timeout ordering fails undiscounted at a 900-step horizon | flagged; holds at SAC's γ, marginal for PPO's |

---

## 4. Throughput — 04a §8.3's gate

> **Revision 7.** **38 steps/s at 2 Hz = 19 simulated seconds per wall second**, against about 2 at 10 Hz (20 steps/s) -- 9.5x more simulated time, most of it from casting the LiDAR and updating the tracker once per 0.5 s instead of per 0.1 s. The free-space classifier costs nothing measurable (38.1 against 38.3 steps/s). The 1.6 M-step suite sized at 10 Hz is 320 k steps at 2 Hz: **about 2.3 h per policy seed**, against 04a §8.3's 50 minutes. Vectorised environments are still the next move.

> **Corrected in revision 6.** The 74 steps/s below was measured on a narrow target-free corridor and overstated the default environment about 4×. Measured idle on the default environment with a target and random actions: **13–14 steps/s**, then **19–21** after fixing a hotspot in the confinement check. One policy-seed suite at 20 steps/s is ~22 h. The original text follows.

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

**This is B3.** At ~20 steps/s (74 as first measured) a 30-run protected core is not affordable on the
timescale 04a assumes. Options, none of which is Claude Code's to choose:
vectorised environments (the obvious first move, and the measurement above is
single-env), a coarser admissibility sample, or a shorter budget per run.

---

## 5. Acceptance tests — 03a §10

| # | Test | State |
|---|---|---|
| T1 | Froude at `U_nom` is 0.14 ± 0.01 | **pass** — Fr 0.142 (F42) |
| T2 | boundary branch differs from ground truth under pose noise | pass |
| T3 | `\|corr(e_y, b_i)\| < 0.9` for all 7 rays | pass at 10 m, on the path offset alone (60 episodes; run 1000 before the freeze) |
| T4 | target invisible when fully occluded | **not written** — needs 04a §3.6's occlusion placement |
| T5 | facility walls in the raw scan, absent after the gate | pass |
| T6 | corridor boundary never in the raw scan | pass (at 10 m) |
| T7 | crossing targets leave the corridor, others do not | pass — confinement flags; containment relaxed at a basin-wide corridor (F39) |
| T8 | static panels classified dynamic rarely | **pass** (F37) — 40 target-free episodes with zero phantoms; measured 0 in 24,024 frames with and without pose noise |
| T9 | head reach ≤ 1.5·Lpp, or `allow_reverse` | **pass** via the emergency stop at 2 Hz, η 0.3–1.0 (break-even 0.27); coasting pinned as failing |
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
| ✅ | throughput measured | 38 steps/s at 2 Hz, 19 simulated s per s — see §4 |
| ❌ | every `TODO(04-*)` resolved or deferred | four outstanding |
| ✅ | operating speed settled | 0.55 m/s (F42) |
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
| **6** | **2026-09-13** | **05 part 1 validated (`bluefin/REVIEW.md`) and integrated; emergency stop; deployment timing wrapper; F30–F36; §6.3 and throughput overstatements corrected** |
| **7** | **2026-09-14** | **2 Hz everywhere; corridor fixed at 10 m; free-space static/dynamic classifier and sensor-origin fix; bridge e-stop latch; crash-stop block in the basin plan; refit v4 (not adopted) and v4b; F35, F36 fixed; F37–F41; then the pose race fixed (0 of 1,085 stale on replay) and the rudder limiter made a true rate limit, off by default in bridge and simulator** |
| **8** | **2026-09-14** | **F24 decided (0.55 m/s); virtual corridors restored; scenario generator wired into the environment; e-stop reward (A8); nominal pose/ego noise; hull randomisation 1.0; reward scale audit; single-seed PPO formulation run; F42–F48** |
