# PROJECT STATE — Paper 3 implementation

**A living document.** Updated on every change to the tree. It records what is
built, what each piece decides, and — the part that matters — **what is
currently blocking a full training run and a full evaluation.**

| | |
|---|---|
| **Revision** | 12 — **TD3 out of the baseline; four learners x 3 seeds = 12 runs; final replay buffer kept for continuation** (F98); suite **3.1** (F97): Tier B floored at 7.5 m and the default frozen suite, Tier A extended; comparators tuned and COLREGs-VO built; **baseline-v2** (F96): Rule 17(b) draws out of training and the suite; field work delayed not deferred; formulation + five-learner framing; campaign to restart on `configs/baseline_v2.json`; method in `planning/METHODS_BRIEF.md` |
| **Last updated** | 2026-09-24 |
| **Tests** | **515 passing, none expected to fail** (T1 passes since F24 was decided) |
| **Blocking a headline training run** | **none -- the baseline-v1 campaign runs on this machine (F95)** until a cluster is set up. Known limits of v1, where a fix would mean baseline-v2 and retraining: crossing side bias under PPO (A32, a learner property per F94), narrow head-ons without starboard room (A33), being overtaken by a non-yielding vessel (A30, on hold) |
| **Blocking a full evaluation** | the Tier B runner and the COLREGs-VO comparator (B8, C3–C6); your sign-off of the suite freeze, claim ledger and result tables (B6), after the framing decision in `METHODS_BRIEF.md` §9; then B2, B7, B9 (§2) |
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
| `corridor.py` | **channel and basin geometry** — channel centreline, width profile, polygon; `Basin` (slanted legs, side-specific half-widths, head-on band) | 03a §3, 06 (F74) |
| `feasibility.py` | A* route check and obstacle thinning — every layout admits a route | 06 (F74) |
| `targets.py` | **oriented hull, five behaviour models, confinement** | 03a §5 |
| `scenario.py` | **generator, backward solve, rejection ledger** | 04a §3 |
| `suite.py` | **Tier A/B, Around the Clock, Study 2, freeze manifest** | 04a §4–§9 |
| `asv_lidar.py` | 720-beam raycast, sector pooling, dead zone, aft mask | 01 §2 |
| `boundary_raycast.py` | virtual boundary scan, beam gating, pose noise | 01 §3 |
| `tracking.py` | cluster → associate → Kalman → **free-space static/dynamic test** (F37) | 01 §4, replaces 03a §6.3 |
| `cpa_cri.py` | CPA, ship domain, collision risk index | 01 §5 |
| `encounter.py` | five-class classifier + hysteresis — **one definition** | 01 §5.3 |
| `colregs/` | `EncounterContext`, engagement latch (A20), admissibility, peak risk (F88) | 02a §10.1 |
| `stopping.py` | braking-path stop test ("would stopping clear?", A23) | F67 |
| `reward/` | eight dense terms, group clipping, audit | 02a |
| `observation.py` | **six branches, 70 dims** (`a25-v3-context`), frozen index order | `OBSERVATION_SPEC.md`, F72 |
| `features_extractor.py` | scene MLP + presence-gated shared slot encoder, every learner | `OBSERVATION_SPEC.md` §7 |
| `env.py` | the Gymnasium environment — **2 Hz decisions, 0.1 s physics and collision sub-steps, optional pose staleness** | 03a §4 |
| `render.py` | field view + seven-block telemetry panel | RENDER_PANEL_SPEC |
| `play.py` | manual and random harness | — |
| `classical/` | **LOS-PID + DWA, encounter-specific VO and COLREGs-VO (Kuwata)** (B8), perception-only; shared LOS, PID, model rollout, clearances; parameters in `constant_temp.py`, tuned and pinned in `configs/comparators_v1.json` | 04 §5, F83, F97 |
| `reference_controller.py` | CODEX classical reference controller (supplementary comparator) | F71 |
| `train_formulation.py` | **the shared trainer for every learner** (baseline: PPO, RecurrentPPO, SAC, TQC; TD3 still runnable) — curriculum, dev-set evaluation, checkpoints, `--resume`, `--config` baseline gate, replay buffers outside the repository | F76–F77, F86, F93, F95 |
| `baseline_config.py` | freeze, check and verify runs against `configs/baseline_v2.json` | F93, F96 |
| `constant_temp.py` | **the classical comparators' parameters**, staged out of `constants.py` so tuning one cannot change the formulation digest; merged in after the campaign | your call, 2026-09-23 |
| `tools/scale_audit.py` | 02a §8.2's reward scale audit over the generator | rev 8 |
| `results/baseline_campaign.sh`, `tools/campaign_status.py` | the resumable five-learner campaign and its status view (`TRAINING_GUIDE.md`) | F95 |

### 1.2 Task status

| Task | State |
|---|---|
| 01 perception and observation | **done** |
| 02 / 02a / 02b reward | **done** — T1–T4, C1–C4 |
| 03a environment and target | **done** — §1.2, §3, §4, §5, §7, §10; **§6.3 replaced** by the free-space classifier (F37); §4.1's 0.1 s step replaced by 2 Hz (F38) |
| 04a scenario and evaluation | modules done — §3, §4, §5, §6, §7, §9 — and the generator drives training (F44) |
| 05 vessel model and sim2real | **part 1 validated and integrated**; refit v4 run, **not adopted** (F40); crash-stop block added to part 2; basin sessions pending |
| Training campaign | **running** — baseline-v2, four learners × 3 seeds = 12 runs (F95, F98); formulation frozen (F93, F96) |

---

## 2. What is blocking the campaign

Ordered by what they block, not by size. **The first is one decision and it
moves almost everything else.**

### 2.1 Blocking a headline training run

| # | Blocker | Owner | Effect |
|---|---|---|---|
| ~~B1~~ | ~~F24, the operating speed~~ — **decided: 0.55 m/s, `CRUISE_RPM = 6`** (F42) | — | — |
| **B2** | **F28 — Rule 8(e): resolved in simulation by the emergency stop; unverified on the water** | 05 | the field claim of 8(e) — moved to evaluation |
| ~~B3~~ | ~~Throughput~~ — **measured for every learner** (F80) and budgeted (A26, F95): PPO ~108 steps/s, RecurrentPPO ~61, TD3/SAC/TQC 12-18 at 1.0 gradient step per transition | — | — |
| B4 | Pose noise is **nominal**, not measured (F46) | 05 | the N1 sim-to-real claim, until S1-A |
| ~~B11~~ | ~~Phantom dynamic tracks~~ — **fixed** (F37); confirm as §6.3's replacement (`OPEN_PROBLEMS.md` A7) | — | — |
| ~~B12~~ | ~~Spawn boundary penalty~~ — **fixed** (F35) | — | — |
| ~~B13~~ | ~~fixed corridor, 2 Hz reward~~ — **corridors restored, reward confirmed** (F43) | — | — |

### 2.2 Additionally blocking a full evaluation

| # | Blocker | Owner | Effect |
|---|---|---|---|
| ~~B5~~ | ~~Curriculum steps per stage and total budget~~ — **decided**: stage fractions (F75), 2 M steps × 5 seeds, ratio 1.0 (A26, F95) | — | — |
| **B6** | Claim ledger and result tables **drafted, unsigned** (`planning/`); they still name a single proposed SAC method — framing decision first (`METHODS_BRIEF.md` §9) | you | 04a §9.3 freeze checklist |
| **B7** | `T-RE` reactive target is a placeholder, not the VO comparator | 03 | Tier B's `re` behaviour stratum |
| **B8** | Classical comparators: **LOS-PID+DWA and encounter-specific VO built and on Tier 1 (F83)**; COLREGs-VO (Kuwata) not built; `T-RE` not yet swapped onto the VO | 04 | COLREGs-VO row of R1; B7 |
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

**F59 — straight paths only (your call, 2026-09-16).** To simplify the RL
environment:

* **No bends.** `CORRIDOR_BENDS = False`: bends are off in every curriculum
  stage, `corridor.sample` ignores `allow_bend`, and the generator's bend quota
  (`CORRIDOR_BEND_FRACTION`) is 0. `corridor.build` keeps the bend machinery
  for unit tests and for re-enabling.
* **A straight reference path in a varying-width corridor.**
  `STRAIGHT_REFERENCE_PATH = True`: the Rule 9(a) offset is
  `offset_frac · ½ · min(W)`, constant along the path. Before, it was a
  fraction of the *local* half-width, which bent the path at every taper even
  without a bend. The walls still vary, so the boundary branch keeps its
  information; T3's decorrelation test still passes.
* **What goes with it.** `r_path` (R-8) is identically zero in training.
  04a's "≥ 40 % of episodes carry a ≥ 20° bend" is withdrawn. Tier A loses
  `A-BND-HO-I` and `A-BND-CRS-I` (34 → 32 cases). The F54 width/bend
  confound disappears.
* **Tests.** `test_every_generated_reference_path_is_straight` replaces the
  two tests that required some episode to bend the path. The unit test that a
  compliant port bend costs nothing in `v_port` stays. **405 passed.**
* **Generator yield is unchanged.** 60/60 per class except null: 40/60,
  against 43/60 with bends, so it is not caused by this.

**F60 — the tiered formulation tests: built, Tiers 0 and 1 run; Tier 0 found A21.**

*The protocol* (`tools/tiers/`). Every tier replays fixed development-namespace
scenarios, so a comparison changes the code and nothing else.

| tier | what | cost | command |
|---|---|---|---|
| 0 | scripted policies: does the reward point the right way? | ~80 s | `tier0_scripted.py` |
| 1 | replay a saved model under today's code | ~2 min | `tier1_replay.py --model …` |
| 2 | short fine-tune from a saved model, stage 5 | ~1 h | `train_formulation.py --init-model … --init-vecnormalize … --fixed-stage 5 --timesteps 300000` |
| 3 | full run from scratch | ~6 h | `train_formulation.py --timesteps 2000000` |

`train_formulation.py` gains `--init-model` / `--init-vecnormalize`: today's
hyperparameters replace the saved ones, so a fine-tune differs from a fresh
run only in its starting weights. Outputs go to `results/tiers/<tier>_<tag>/`.

*Tier 0* (`results/tiers/tier0_straight_a20/`), 12 scenarios per class, obstacles
off, three scripted policies (path follower; one committed 30° alteration in the
compliant sense; the same in the wrong sense):

* **T0.1 passes for every give-way class.** COLREGs penalty, compliant against
  wrong way: head-on −15.3 vs −55.3, crossing −11.9 vs −55.5, overtaking
  −11.7 vs −32.7. The reward points the right way, including A17's port
  crossings.
* **T0.2 passes.** Being overtaken, `v_hold` integral 2.21 holding against 3.41
  for leaving the role. A frame count was the wrong metric: speed-estimate noise
  makes `v_hold` slightly positive on most frames of a held course.
* **T0.4 passes.** No head-on episode latches a port sense (A19–A20).
* **T0.5 passes.** `r_path` is zero throughout (F59).
* **T0.6 passes.** Supervisor stops: 2, none followed by a collision.
* **T0.3 fails narrowly.** A path follower collects −5.6 of COLREGs penalty on
  null encounters, against a −5 threshold. Explained below.
* **The scripted policies collide a lot, and one case is not the policy's
  fault.** A held 30° alteration runs into the walls in narrow corridors
  (head-on 7/12, overtaking 10/12 boundary collisions). That limits the
  scripted policy, not the reward. But **holding course against an overtaker
  collided 11/12**, including draws above A15's 1.0 m floor.

*Why holding course against an overtaker collides* (traces;
`tools/diagnostics/clamp_frequency.py`, `results/clamp_frequency/`):

1. **The corridor clamp teleports confined targets.** Being-overtaken crossing
   angles are drawn from ±67.5°, and the generator checks containment only at
   the spawn *point*. So the hull breaches the channel on step one, and
   `targets.clamp_to_corridor` moves the target to the nearest centreline
   station and turns it parallel. In one trace, a 1.30 m DCPA overtaker was put
   0.19 m off the own ship's track and hit its stern at 6 s. Path follower, 20
   per class: the clamp fires before CPA in **75 %** of being-overtaken, **90 %**
   of null and 10 % of overtaking episodes (median jumps 2.4, 3.5 and 3.2 m),
   and never in head-on. This also explains T0.3: most "null" targets are
   teleported onto the own track. **Runs 2 and 3 trained on this.**
2. **A15's floor is centre-to-centre.** The contact-free centre DCPA for an
   overtaker at speed ratio 1.5–2.2 is 0.66 m at 0°, 1.2–1.4 m at 15° and
   1.5–1.7 m from 30° to 67.5°. So a 1.0 m floor is collision-free only for
   near-parallel overtakers. Unclamped being-overtaken episodes still collided
   at 0.80.

Decision: `OPEN_PROBLEMS.md` **A21** (recommended: confined targets keep the
channel, and the floor is on hull clearance). It absorbs C14's clamp replacement.

*Tier 1* (`results/tiers/tier1_run3_straight_a20/`): the run 3 final model,
trained with bends and before A18–A20, replayed on straight paths with
A18–A20. That is 20 development scenarios per class plus the 100-scenario
head-on width set, 220 episodes in 114 s.

| class | collision, replay | collision, run 3's own final eval |
|---|---|---|
| overtaking | **0.00** | 0.30 |
| head-on | 0.30 | 0.35 |
| null | 0.15 | 0.20 |
| no target | 0.10 | 0.10 |
| being overtaken | 0.35 | 0.35 |
| crossing | 0.35 | 0.25 |

* **Narrow head-on no longer stands out.** On the width set, target collision
  is 0.14 / 0.15 / 0.10 / 0.21 / 0.21 at 5 / 6 / 7 / 8 / 10 m, against run 2's
  0.43 → 0.05 gradient (F54, F56). With straight paths and A18, width is no
  longer the axis the failures lie on.
* **F58's stop check.** 25 supervisor stops in 220 episodes, 24 of them head-on,
  about 0.2 per episode at every width. **6 of the 22 stopped episodes then hit
  the target (0.27).** A18 admits a stop only when the *perceived* DCPA with the
  own ship stationary reaches 1.76 m, and near CPA that estimate carries C15's
  tracker course error. So A18's residual is C15, not the rule. It is also a
  policy trained under the old supervisor, so this bounds the mechanism rather
  than measuring a retrained agent.
* **Being overtaken:** 0.43 collision at or above the floor, against 0.17
  below. It is inverted because of A21: the floor is not a clearance, and
  clamped targets ignore it.
* The crossing class's 0.55 "port sense" is correct: those are crossings from
  port (A17).

*Tier 2 and run 4 wait for A21.* Being overtaken and null are a third of the
training distribution, and their geometry is currently decided by the clamp.

**F61 — A21 built (option 1): confined targets keep the channel; Tiers 0 and 1
re-run clean.**

*Built.*

* `scenario._sample_ct`: overtaking, being-overtaken and null crossing
  angles are uniform on ±`CONFINED_CT_HALF_DEG` = 10°. The classifier's
  bands are unchanged.
* `ScenarioGenerator._containment_ok` → `_track_inside`: a confined target's
  hull must stay inside the corridor every 0.5 s from spawn to CPA (null:
  15 s).
* `targets.clamp_to_corridor` nudges instead of teleporting: heading to the
  local tangent, position inward by the breach + 0.1 m. This absorbs C14.
* `scenario.contact_free_dcpa(ct, k)`: the smallest centre DCPA at which the
  collision hulls never touch, found by bisection over the relative track and
  cached per 0.5° and 0.05 of speed ratio. It is 0.65 m parallel, 0.82–0.96 m
  at 5° and 0.99–1.23 m at 10°. The being-overtaken floor is that value plus
  `BEING_OVERTAKEN_FLOOR_MARGIN` = `D_SAFE`, per draw. Above-floor draws are
  uniform on [floor, max(2.0, floor + 0.5)]; 20 % are drawn below and
  labelled. `scenario.dcpa_floor_m` records it.
* Tests: `test_contact_free_dcpa_matches_the_hull_geometry`,
  `test_confined_targets_keep_the_channel_until_cpa`,
  `test_the_clamp_nudges_rather_than_teleports`, and the A15 test now against
  the per-draw floor. **408 passed.**
* Generator yield, stage 5, 60 draws per class: head-on, overtaking and no
  target 60; crossing 59; being overtaken **54**; null **32** (40 before).

*Tier 0* (`results/tiers/tier0_a21/`): **all 10 checks pass.**

| | before A21 | after |
|---|---|---|
| being overtaken, path follower holding course: collision | 11/12 | **1/12** |
| being overtaken: `v_hold` integral, holding vs leaving the role | 2.21 vs 3.41 | 2.44 vs 3.50 |
| null: follower COLREGs penalty (T0.3) | −5.6 (fail) | **0.0** |
| overtaking: COLREGs penalty, compliant vs wrong way | −11.7 vs −32.7 | −3.2 vs −43.6 |

Head-on and crossing are unchanged, as they should be: A21 does not touch them.

*Clamp* (`results/clamp_frequency/`, path follower): no clamp before CPA for
being overtaken (was 75 %) or overtaking (was 10 %). Null targets are still
nudged, but first at a median 24.8 s, after the 15 s check window, by a
median 0.13 m, so the encounter is not re-drawn. Being-overtaken target
collision for the follower is **0.05** (was 0.80).

*Tier 1* (`results/tiers/tier1_run3_a21/`), run 3's final model replayed. Only
the being-overtaken and null draws changed:

| class | collision, F60 replay | after A21 |
|---|---|---|
| being overtaken | 0.35 | **0.20** |
| null | 0.15 | 0.25 (target collisions 0; boundary and obstacle) |
| head-on, crossing, overtaking, no target | 0.30, 0.35, 0.00, 0.10 | same |

The null rise is a policy trained on teleported null targets meeting real
ones, with no target contact; Tier 2 is the test of it. Supervisor stops are
unchanged: 24, with 6 followed by a target collision (F60, C15).

*Tier 2 launched:* fine-tune of run 3's final model, stage 5 only, 300 k steps,
evaluation every 100 k at 20 per class
(`runs/ppo_formulation_seed0_tier2_a21/`).

**F62 — Tier 2 (fine-tune on A18–A21 and straight paths), crossing
feasibility, and a class-share bug.**

*Tier 2* (`runs/ppo_formulation_seed0_tier2_a21/`). Run 3's final weights and
reward statistics, 302 k stage-5 steps, 1.25 h. PPO stayed stable: `approx_kl`
0.025 → 0.036, clip fraction 0.17 → 0.22, action std 0.43. Throughput averaged
71 fps; each 120-episode evaluation cost 140–300 s.

| evaluation (20 per class) | 100 k | 200 k | 300 k | final |
|---|---|---|---|---|
| goal | 0.83 | 0.81 | 0.80 | **0.82** |
| collision | 0.17 | 0.19 | 0.20 | **0.18** |
| head-on collision | 0.25 | 0.25 | 0.20 | 0.15 |
| crossing | 0.35 | 0.40 | 0.55 | 0.45 |
| overtaking | **0.00** | 0.00 | 0.00 | 0.00 |
| being overtaken | 0.15 | 0.25 | 0.30 | 0.35 |
| null | 0.20 | 0.15 | 0.10 | 0.05 |
| no target | 0.05 | 0.10 | 0.05 | 0.05 |

Training-episode collision by 100 k block: crossing 0.60 / 0.58 / 0.58,
head-on 0.39 / 0.39 / 0.37, being overtaken 0.27 / 0.23 / 0.21, overtaking
0.12 / 0.11 / 0.13, no target 0.20 / 0.16 / 0.11. The evaluation's rise in
being overtaken (0.15 → 0.35) runs against the training trend (0.27 → 0.21).
With 20 scenarios, one scenario is 5 pp, so it reads as noise until a larger
set says otherwise.

What it says:

* **Overtaking is solved** under the current formulation: 0 collisions at
  every evaluation, 0.11–0.13 in training.
* **Null improved as the Tier 1 replay predicted** once the policy saw real
  channel-keeping null targets: 0.25 → 0.05.
* **Head-on is middling**, 0.15–0.25; supervisor stops fell to 0.02 per
  episode.
* **Crossing is now the dominant failure**, and flat in training at ~0.58.
  Both sides fail alike (port 0.48, starboard 0.44 over the last two
  evaluations), so it is no longer A17's side problem.

*Crossing feasibility* (`tools/diagnostics/crossing_feasibility.py`,
`results/crossing_feasibility/`). The 20 development crossings, obstacles
and supervisor off, each replayed under scripted responses taken **from
t = 0**, so late engagement cannot be the excuse:

| response | target hit, from port | from starboard |
|---|---|---|
| hold path and speed | 0.82 | 0.56 |
| half speed | 0.73 | 0.67 |
| full astern (coasts: no reverse at stage 4) | 0.45 | 0.67 |
| 30° compliant alteration | 0.45 | 0.89 |
| 60° compliant alteration | 0.27 | 0.67 |

**Even the best of these for each scenario still hits the target in 5 of 20
(0.25).** Unavoidable draws have a median TCPA of 8.8 s and spawn range 6.7 m,
against 9.9 s and 8.9 m for avoidable ones. So about a quarter of generated
crossings cannot be escaped by the own ship, which is A15's problem in another
class. Decision: `OPEN_PROBLEMS.md` **A22**. Tier 2's ~0.45 crossing collision
sits above that 0.25 floor, so the class is partly geometry and partly still
learning.

*C16 — the class share is not the configured one (fixed).* When a draw capped
out, `env._load_generated` retried with a fresh seed, which redrew the *class*
too. Classes that cap out often lost share to the rest: null caps on about
half its draws and trained at **4.3 %** against an intended 11 %. The class is
now drawn once from `CLASS_SAMPLE_WEIGHTS` and only the seed is retried. 400
stage-5 resets now give null 0.12 (0.11), crossing 0.22 (0.22), overtaking
0.16 (0.16), being overtaken 0.16 (0.14), no target 0.20 (0.17), head-on 0.15
(0.20). (A reading slip on the way: pandas parses the class label "null" as a
missing value, which briefly made null look absent from training. Read
`monitor.csv` with `keep_default_na=False`.)

*Where the tiers leave run 4.* Tiers 0 and 1 are clean on A18–A21. Tier 2 shows
the formulation trains stably and improves overtaking, null and head-on. The
one open formulation issue is A22. A 6 h run 4 before A22 would spend a
quarter of its crossing episodes on draws no policy can win.

**F63 — A22 built (option 1): crossings must be escapable, 20 % labelled; Tiers
0 and 1 re-run.**

*Built.* `scenario.crossing_escape_feasible(own, own_heading, solved, channel)`
rolls the nominal hull out from the environment's start state (on the path, at
`U_NOM`). It holds course for `CROSSING_ESCAPE_DELAY_S` = 1.5 s, then either
coasts (`CROSSING_ESCAPE_STOP_RPM` = 0, the policy's floor at propulsion stage
4) or steers a 60° alteration in the A17 compliant sense, with commands at the
2 Hz decision rate. The target is constant-velocity and unconfined. An escape
counts if the hulls never overlap and the own hull stays inside the corridor
until `CROSSING_ESCAPE_TAIL_S` = 6 s past TCPA; the rollout stops early once the
target has passed and is opening beyond 3 m.

`ScenarioGenerator.sample` draws `want_unescapable` (p = 0.20) **once per
sample**. Per attempt, rejection would cut the realised share to about 8 %.
`_attempt` rejects crossing draws whose feasibility disagrees with the label
(`escape_label_mismatch`) and records `scenario.crossing_escapable`.

Cost after optimisation (a decimated wall outline, 2 Hz steering, early exit):
0.21 s per crossing sample, down from 0.6 s. Across all 80 draws, 78 built;
wanted unescapable 16 %, built unescapable 17 % (13 of 78), and cap-outs are
flat across width strata. Test:
`test_crossings_are_escapable_except_a_labelled_fraction`. **409 passed.**

*Crossing feasibility re-run* (`results/crossing_feasibility/`; the pre-A22 file
is kept as `summary_pre_a22.txt`): **every one of the 20 development crossings
is now escapable by some scripted response** (was 15 of 20). Hit rates: coast
0.1 / 0.2, 60° turn 0.0 / 0.2, hold 0.8 / 0.5 (port / starboard). The
generator's labels agree with the replay in all 20.

*Caution on comparisons.* The label draw consumes one random number before the
corridor is sampled, so development crossing seeds now produce different
corridors. The 20-crossing development set happens to carry no unescapable
draw (p ≈ 0.03) and is narrower: median width 5.6 m, against 7.7 m before. So
crossing results before and after A22 are **not like-for-like**. Other classes
are unchanged.

*Tier 0* (`results/tiers/tier0_a22/`): **all 10 checks pass.** On crossings,
scripted compliant collision 0.58 (was 0.83); COLREGs penalty compliant −7.2
against wrong way −35.3. Other classes are identical to F61.

*Tier 1* (`results/tiers/tier1_tier2model_a22/`), replaying the Tier 2 (A21)
final model:

* Crossing collision **0.80** on the new, narrower, all-escapable set (0.45 on
  its old set). The policy never saw this distribution, so this is the gap
  Tier 2 has to close, not a regression in the code.
* Head-on 0.15, null 0.05, no target 0.05, overtaking 0.00, being overtaken
  0.35: as in its own evaluation.
* Head-on width set: 0.41 at 5 m (0.16–0.20 elsewhere), with **5 of 6** stopped
  episodes then hitting the target at 5 m. In total, 12 stops, 6 followed by a
  target collision. This model was fine-tuned under A18; the stop-then-hit pattern
  at narrow widths is the perceived-DCPA error of C15, and is now worth building.

*Tier 2 launched* on A22: fine-tune from the A21 Tier 2 final model, stage 5,
300 k steps (`runs/ppo_formulation_seed0_tier2_a22/`).

**F64 — C15, a hull-fitted tracker measurement: built behind a switch, not
adopted.** `TRACK_MEASUREMENT` stays `"centroid"`.

*Why C15 exists.* The Kalman filter was fed the centroid of a cluster's
returns. Those lie on the faces nearest the sensor, so the centroid sits up to
half a hull length toward the own ship and slides as the aspect changes or as
the 1 m dead zone clips the near end. Close in, that slide reads as velocity
(F56, F60).

*Built* (`src/tracking.py`):

* `hull_fit_centre(points, origin, prior_axis_deg=…)`: an L-shape fit
  (Zhang et al., 2017) over orientation, then each axis completed to the known
  `LOA` × `BREADTH` away from the sensor. An observed end on the dead-zone
  circle is treated as clipped, and the far end is anchored instead. It
  returns `None` when the length axis is unknowable (a short cluster with no
  course prior).
* `Cluster.origin` records the sensor position.
* `Tracker(measurement="hull_fit")`: each track learns the offset from its
  centroid to the fitted centre (blend 0.5, fits need ≥ 8 returns) and is
  always measured as centroid + offset; association uses the same.
* Tests: synthetic returns from bow-on, quarter and beam-on views recover the
  centre within 5 cm, and within 15 cm with the near end clipped by the dead
  zone (the first version failed both: a tie completed the hull toward the
  sensor, and it anchored on the dead-zone cut). **50 tracking tests pass.**

*Measured* (`tools/diagnostics/c15_track_error.py`,
`results/c15_track_error/`): path follower, obstacles and supervisor off, 40
head-on width-set and 24 development scenarios, each step's nearest dynamic
track paired with the truth. Three variants:

| true range | stop-test disagreement with truth: centroid | v1 fit fed directly | **v2 learned offset** | v3 range-dependent gain |
|---|---|---|---|---|
| < 2 m | 0.34 | 0.06 | **0.06** | 0.06 |
| 2–3 m | 0.20 | 0.04 | **0.03** | 0.03 |
| 3–4 m | 0.06 | 0.05 | **0.03** | 0.03 |
| 4–6 m | 0.01 | 0.16 | 0.05 | 0.07 |
| 6–9 m | 0.01 | 0.32 | **0.23** | 0.24 |
| 9–16 m | 0.06 | 0.34 | 0.09 | 0.09 |
| overall | 0.090 | 0.137 | 0.083 | 0.087 |

| v2 against centroid | < 2 m | 2–4 m | 6–9 m |
|---|---|---|---|
| position error, median | 0.33 → 0.18 m | 0.37–0.55 → 0.05–0.07 m | 0.73 → 0.08 m |
| course error, p90 | **162° → 22°** | 7–11° → 10–12° | **7° → 33°** |
| frames tracked | 304 → 328 | similar | 447 → 447 |

v1 also lost up to 80 % of long-range tracked frames: sparse far clusters fell
back to the centroid, and the ~0.7 m jump between the two measurements read as
velocity and broke associations. v2's learned offset fixed that. v3 learned
slowly beyond 4 m and did no better.

*Why not adopted.* The hull fit sharply improves exactly what C15 was opened
for: close-range position, and the A18 stop test inside 4 m. But every variant
worsens course at 6–9 m, which is where encounters engage and where A20 now
freezes the class. The mechanism is structural. A track arrives from range
with little learned offset and acquires ~0.6 m of it while closing, and any
time-varying correction to a measurement reads as velocity. Trading
engagement-range course for close-range position is the wrong trade while the
class is latched at engagement.

*What would make it adoptable* (not done): estimate the centre offset as
filter state rather than correcting the measurement, e.g. a
constant-velocity-plus-extent model whose offset has its own slow process noise,
so it is not differentiated into velocity. Alternatively, use the fitted centre
only in the stop test's DCPA, not in the track state. Both are real work. The
second is the smaller, and targets the one consumer C15 was measured against.
Recorded as C15's next step.

**F65 — Tier 2 on A22.** A fine-tune of the A21 Tier 2 final model, 302 k stage-5
steps, 1.50 h. That is slower than F62's 1.25 h because the C15 diagnostics
shared the CPU (48–58 fps). PPO stayed stable: `approx_kl` 0.044 → 0.037, clip
fraction 0.23 → 0.18, std 0.43. Class shares in training now match the
configured weights (C16): crossing 0.22, head-on 0.20, no target 0.17,
overtaking 0.16, being overtaken 0.14, null 0.11.

| evaluation, 20 per class | 100 k | 200 k | 300 k | final | A21 Tier 2 final (F62) |
|---|---|---|---|---|---|
| goal | 0.79 | 0.82 | 0.81 | **0.81** | 0.82 |
| collision | 0.21 | 0.18 | 0.19 | **0.19** | 0.18 |
| head-on | 0.15 | 0.15 | 0.10 | 0.15 | 0.15 |
| crossing | 0.75 | 0.70 | 0.60 | **0.60** | 0.45 (different set) |
| overtaking | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 |
| being overtaken | 0.15 | 0.10 | 0.15 | 0.20 | 0.35 |
| null | 0.15 | 0.10 | 0.25 | 0.15 | 0.05 |
| no target | 0.00 | 0.00 | 0.05 | 0.05 | 0.05 |

Training-episode collision by 100 k block: crossing 0.58 / 0.61 / 0.53,
head-on 0.40 / 0.33 / 0.36, being overtaken 0.23 / 0.20 / **0.18**,
overtaking 0.08 / 0.12 / 0.14, null 0.12 / 0.16 / 0.18, no target
0.14 / 0.12 / 0.15.

What it says:

* **Crossing is learning but far from learned.** On A22's escapable set the
  policy started at 0.80 (Tier 1, F63) and reached 0.60, with
  target collisions 0.50 from port and 0.37 from starboard over the last
  two evaluations. Training includes the labelled 20 % unescapable, so its
  0.53 means about 0.41 of *escapable* crossings still collide. A scripted
  60° turn or a coast escapes every one of them by construction, so this is
  learning, not geometry.
* **Being overtaken improved further** (evaluation 0.35 → 0.20 against F62;
  training 0.23 → 0.18), consistent with A21.
* **Head-on, overtaking and no target held.** Null wandered (0.05–0.25,
  obstacles not targets) at 20 scenarios.
* **The limit of Tier 2 is showing.** 300 k fine-tuning steps from a policy
  trained on bent corridors and earlier generator geometry cannot say whether
  crossing converges. That is a from-scratch question: run 4.

*Stop-test note (C15).* Supervisor stops rose to 0.06–0.08 per episode (F62:
0.02). The close-range perceived-DCPA error F64 measured is the likely reader;
the stop-test-only use of the fitted centre is C15's next step.

**F66 — run 4 launched; C15's stop-test view built, measured more accurate, and
switched off, because it exposes A18's premise.**

*Run 4* (`runs/ppo_formulation_seed0_v4/`): from scratch, 2 M steps, curriculum
1 → 5, 20 evaluation scenarios per class, with A15–A22 and C16 in and
`TRACK_MEASUREMENT = "centroid"`. Its processes loaded the code before the stop
view existed, so it trains without it.

*Built.*

* `Track.last_fit_centre` / `last_fit_heading_deg`: each matched update's hull
  fit, in either measurement mode. `hull_fit_centre(..., return_heading=True)`
  gives the length axis pointed along the track's course.
* `ContextManager._attach_stop_view` sets `ctx.stop_rng`, `stop_alpha` and
  `stop_ct` from the fit when it lies within `STOP_TEST_FIT_RANGE_M` = 4 m.
  `EncounterContext.dcpa_if_stopped` reads them when set. Nothing else does, so
  the track state and every other consumer are untouched.
* Switch: `STOP_TEST_USES_HULL_FIT`. Tests:
  `test_the_stop_test_reads_the_close_range_hull_fit_view` and
  `test_tracks_carry_their_latest_hull_fit`.
* Tier 1 gained `--processes` and `--stop-test-fit on|off`, which apply
  `constants` overrides in every worker.

*Against ground truth* (`results/c15_track_error/summary_f66_stop_view.txt`):
the stop test's disagreement with the truth about `stop_clears` fell from
0.090 to **0.046**. By range: < 2 m 0.34 → 0.12, 2–3 m 0.20 → 0.03, 3–4 m
0.06 → 0.08, unchanged beyond 4 m. Position and course are identical.

*Against behaviour* (Tier 1 A/B on the Tier 2 A22 final model;
`results/tiers/tier1_f66_stopfit_off/`, `…_on/`):

| | stop view off | **on** |
|---|---|---|
| supervisor stops (220 episodes) | 13 | **29** |
| … followed by a target collision | 6 | **15** |
| head-on target collision at 5 / 6 / 7 / 8 / 10 m | 0.32 / 0.25 / 0.25 / 0.16 / 0.05 | 0.32 / **0.35 / 0.30** / 0.16 / 0.05 |
| development set, per class | — | identical |

**A more accurate stop test made outcomes worse.** A18 asks whether the target
would clear *an own ship stationary where it is now* (`dcpa_if_stopped` ≥
`ESTOP_CLEAR_DCPA_M` = 1.76 m). But a stopping vessel does not stop where it
is. At stage 4 it cannot reverse, so it coasts. Even the supervisor's astern
brake takes 1.1–3.7 s (F28), during which it keeps closing on the target's
track. The centroid's bias toward the own ship had been understating that
DCPA and so suppressing stops, which masked the flaw. With accurate geometry
the supervisor fires in draws whose stationary DCPA is 1.76–2.5 m, the vessel
slides forward while stopping, and the hulls meet.

**`STOP_TEST_USES_HULL_FIT` is `False`**, which matches run 4. The view stays
built for A23, the decision on how "stopping clears" should be evaluated. **418 tests pass.**

**F67 — A23 built (option 1): the stop test follows the braking path. Stops
that end in a collision fall to zero; C15's view stays off.**

*Built.*

* `src/stopping.py`, kept out of `emergency_stop.py`, which the bridge
  imports simulator-free. `braking_profile(u0)` is the supervisor latch's full
  astern (`S2 = −100`) on the nominal identified hull, from surge `u0` to
  `ESTOP_STOP_SPEED` (cap `ESTOP_MAX_BRAKE_S`), sampled at `STOP_TEST_DT_S` =
  0.05 s and cached per cm/s. `dcpa_over_stop(rng, alpha, ct, speed_ts,
  u_own)` is the minimum centre distance between the target's
  constant-velocity track and the own ship along that path, then stopped.
* `EncounterContext.u_own` holds the own surge when the context was built;
  `dcpa_if_stopped` now calls `dcpa_over_stop`, still against
  `ESTOP_CLEAR_DCPA_M` = 1.76 m. At `u_own` below the stop speed it equals
  A18's stationary DCPA exactly, so every A18 test passes unchanged.
* The supervisor's trigger, R-2's slowdown carve-out and `r8_parts`'
  alteration credit all read `stop_clears`, so all three now account for
  braking.
* Braking from cruise (0.56 m/s) takes 1.05 s over 0.32 m; from 1.05 m/s,
  1.9 s over 0.99 m. A crossing target clearing a stationary ship by 1.9 m
  passes within 1.58 m of a braking one. 0.06 ms per evaluation.
* Tests: `test_the_braking_profile_stops_and_grows_with_speed`,
  `test_the_stop_test_follows_the_braking_path`.

*Tier 1 A/B* (the Tier 2 A22 final model; `results/tiers/tier1_a23_stopfit_off/`,
`…_on/`; F66's rows for comparison):

| stop test | C15 view | stops | then a target collision | head-on target collision 5 / 6 / 7 / 8 / 10 m |
|---|---|---|---|---|
| A18, stationary | off | 13 | 6 | 0.32 / 0.25 / 0.25 / 0.16 / 0.05 |
| A18, stationary | on | 29 | 15 | 0.32 / 0.35 / 0.30 / 0.16 / 0.05 |
| **A23, braking path** | **off** | **7** | **0** | 0.32 / 0.25 / 0.25 / 0.16 / 0.05 |
| A23, braking path | on | 23 | 8 | 0.32 / 0.35 / 0.30 / 0.16 / 0.05 |

Development-set outcomes per class are unchanged in all four.

*Reading it.* The braking path removes every stop that ended in a collision,
and halves the stops. The hull-fitted view still doubles stops and still ends
8 of them in collisions, so geometry accuracy was not the whole story. The
likeliest remainder is the fitted *heading*: `stop_ct` comes from the fitted
length axis, whose orientation at < 2 m is set by one or two faces. Two
further candidates are the nominal hull under-predicting stopping distance
(reverse efficiency 0.5 is unmeasured, B1) and the stop test's constant-velocity
target. **`STOP_TEST_USES_HULL_FIT` stays `False`**; A23 is on by default.

Run 4 trains without A23 (its processes started first). **420 tests pass.**

**F68 — your five suggestions: supervisor as a runtime layer, not a training
signal. Built behind switches, and one confound found and fixed on the way.**

1. **Train with the supervisor off.** `train_formulation.py --train-supervisor
   off` builds the training environments with `emergency_stop=False`. `R_ESTOP`
   and the latch-held speed-gate suspension only act while the latch holds, so
   both drop out of the training reward with nothing else changed. Actions stay
   0–100. The environment default is still on, so run 4 and every earlier
   result are reproducible.
   **R-2 keeps a slowdown test, now about the agent's own slowdown.** A23's
   `stop_clears` models the latch's full astern. The policy cannot reverse at
   propulsion stage 4 (RPM floor 0), so its slowdown is a coast.
   `stopping.braking_profile(u0, "coast")` (RPM `POLICY_SLOWDOWN_RPM` = 0, cap
   `SLOWDOWN_TEST_MAX_S` = 20 s) backs `EncounterContext.dcpa_if_slowed` /
   `slowdown_clears`. R-2's carve-out and the Rule 8 alteration credit read that;
   the supervisor still reads `stop_clears`. From cruise the latch stops in
   1.05 s over 0.32 m; a coast is still moving after 20 s, 6.0 m on.
2. **The stop stays as a runtime layer.** The environment, bridge and latch are
   unchanged; only where it is applied moves.
3. **Evaluate with it off and on; report interventions.** `--eval-supervisor
   off|on|both`. With `both`, every evaluation runs the development set twice,
   each row tagged `supervisor`, and the summaries carry `intervention_rate` (share
   of episodes with at least one stop) overall and per class. The best model is
   chosen on the policy's own score, supervisor off. Tier 1 gained `--supervisor
   on|off` and an intervention column. Tier B has no runner yet (C3–C6); the
   metric is ready for it.
4. **8(e) as two layers.** A writing item, recorded here: the policy slackens
   speed (learned, reported with the supervisor off); the safety layer takes all
   way off (engineered, reported as the intervention rate). Only the first is a
   learned-compliance claim.
5. **Low-speed starts.** `ASVLidarEnv(low_speed_start_frac=…)`, default 0.
   `--low-speed-start-frac` sets it for training. The chosen share of generated
   episodes starts from rest (half, `LOW_SPEED_START_ZERO_SHARE`) or uniform on
   (0, 0.5 `U_NOM`], on its own seeded stream, so the scenario draw is
   unchanged. `info["start_speed"]` records it.

**Confound found: the supervisor perturbed the noise stream.** The latch read
the own speed by drawing a fresh noisy estimate from the environment's shared
random stream. Switching the supervisor on therefore shifted every later noise
draw (pose, ego, tracker dropout). A replay with the supervisor on differed from
one with it off *even when no stop fired*, so an on/off comparison would have
attributed noise to the stop. The latch now reads the surge the controller
perceived in its last observation (`_observed_surge`), which is also the more
faithful field model. Verified: with no stop, all six classes replay
identically with the supervisor off and on; replays were already
deterministic, and a reused environment matches a fresh one.

Tests: `test_the_policy_slowdown_is_a_coast_not_the_latch`,
`test_a_fraction_of_episodes_can_start_slow`. A smoke run
(`runs/ppo_formulation_seed0_f68_smoke/`, 4 k steps, supervisor off in training,
20 % low-speed starts, evaluation both ways) exercised every flag. **422 tests pass.**

*Experiment, next:* (a) when run 4 finishes, Tier 1 of its final policy with the
supervisor off and on — how much it leans on the stop it trained with, and its
intervention rate; (b) run 5 from scratch with `--train-supervisor off
--low-speed-start-frac 0.15 --eval-supervisor both`, compared with run 4 on the
same development set in both modes. A Tier 2 fine-tune cannot answer this
question: weights trained with the supervisor have already learned around it.
Run 5 also carries A23, the coast test and the noise fix, which run 4 lacks, so
the comparison is of the formulation as a whole.

**F69 — run 4 finished; its policy barely leans on the stop it trained with.
The runtime layer intervened in 5 % of development episodes and rescued 3
crossings.**

*Run 4* (`runs/ppo_formulation_seed0_v4/`, 2 M steps, 7.6 h, supervisor on in
training, pre-A23 stop test). Final in-training evaluation: **goal 0.87,
collision 0.13** (boundary 0.03, obstacle 0.03, target 0.07), the best
formulation run so far. The curve rose to 0.85–0.86 from 1.4 M and held.

| class | run 3 | Tier 2 A22 | **run 4** |
|---|---|---|---|
| head-on | 0.65 | 0.85 | **0.95** |
| crossing | 0.75 | 0.40 | **0.60** |
| overtaking | 0.70 | 1.00 | **0.95** |
| being overtaken | 0.65 | 0.80 | **0.85** |
| null | 0.80 | 0.85 | **0.95** |
| no target | 0.90 | 0.95 | **0.90** |

Goal rate per class. Run 3's development set predates A21/A22, so its column is
indicative only. Crossings remain the weak class: all 8 in-training crossing
failures were target collisions with no stop, at mean speed 0.39–0.71 m/s — the
policy steers but does not slacken. Compliance integrals worsened for head-on
(−25.8) and overtaking (−35.9) against the A22 fine-tune (−17.2, −25.9).

*Tier 1, current code, supervisor off vs on* (`results/tiers/tier1_run4_supervisor_off/`,
`…_on/`; paired episodes, identical noise since F68):

| | supervisor off | supervisor on |
|---|---|---|
| crossing goal / collision / target | 0.45 / 0.55 / 0.50 | **0.60 / 0.40 / 0.35** |
| head-on goal | 1.00 | 1.00 |
| overtaking, being overtaken, null, no target | 1.00, 0.85, 0.85, 0.95 | identical |
| stops (development set + head-on width set) | 0 | 8 in 8 episodes, **0 then hit** |
| intervention rate, development set | — | **0.05** (crossing 0.20, head-on 0.10) |
| intervention rate, head-on width set | — | 0.02 |

*Reading it.* Paired episode by episode, the stop changed exactly three outcomes,
all crossings from target collision to goal (fired at TCPA 4.4–4.9 s, predicted
DCPA 0.19–0.88 m). It harmed none. The four head-on stops fired at TCPA
0.1–1.4 s with the target already at ~2 m and a predicted DCPA of 1.8–2.0 m;
they changed neither outcome nor closest range, so they are late, harmless
triggers near the 1.76 m clear threshold. Seven crossing target collisions got
no stop: the in-extremis condition did not hold in time, not a stop that failed.

So run 4's policy, trained with the latch, performs the same without it in
every class but crossings, where the layer adds 0.15 goal rate. That is the
runtime-assurance picture F68 asked for, measured on a policy that was not
trained for it. Run 5 (`--train-supervisor off --low-speed-start-frac 0.15
--eval-supervisor both`, same seed and budget, all of A23 and F68) is training;
adopt it if its supervisor-off outcomes match or beat run 4's 0.87 / crossing
0.45 and its intervention rate stays near 5 %.

**F70 — run 5 (the F68 formulation) did not improve on run 4: goal 0.76
against 0.87. Low-speed starts are ruled out. The likeliest cause is the R-2
coasting test, which all but switched off the 8(e) slowdown in crossings. Run 6
isolates it.**

*Run 5* (`runs/ppo_formulation_seed0_v5/`, 2 M steps, 11.1 h — evaluation runs
twice; supervisor off in training, 15 % low-speed starts, R-2 on the coast
test, A23 and the noise fix). Final: **goal 0.76, collision 0.24** with the
supervisor off *and* on; intervention rate 0.03. The curve sat at 0.64–0.73 from
0.2 M to 1.8 M, then 0.78 at 2.0 M.

| class (goal) | run 4, Tier 1, supervisor off | run 5, supervisor off | run 5, supervisor on |
|---|---|---|---|
| head-on | 1.00 | 0.90 | 0.90 |
| crossing | 0.45 | 0.40 | 0.40 (target 0.50 → 0.40) |
| overtaking | 1.00 | 0.95 | 0.95 |
| being overtaken | 0.85 | 0.80 | 0.80 |
| null | 0.85 | **0.65** | 0.65 |
| no target | 0.95 | 0.85 | 0.85 |
| intervention rate | — | — | 0.03 (crossing 0.15); 3 stops, 1 then hit |

Run 5 is faster everywhere (crossing mean speed 0.79 m/s against run 4's 0.55;
no target 0.70 against 0.64), and hits static obstacles more (development
obstacle collisions 0.13 against 0.04, Tier 1). It is better on the head-on width set
(target collisions 0.00–0.10 against 0.10–0.20). In training, its last quarter
had shorter episodes (45 against 56 steps) and lower being-overtaken success
(0.52 against 0.80, stochastic actions, 15 % slow starts included).

*Low-speed starts ruled out* (`tools/diagnostics/f70_low_speed_start.py`,
`results/f70_low_speed_start/`). The development set, supervisor off, every
episode at cruise, then every episode from rest:

| policy | goal, cruise | goal, rest | target collision, cruise | rest |
|---|---|---|---|---|
| path follower | 0.38 | 0.45 | 0.26 | 0.16 |
| run 4 | 0.85 | 0.84 | 0.08 | 0.08 |
| run 5 | 0.76 | 0.75 | 0.11 | 0.14 |

Starting from rest does not break the solved encounters: the follower never
collides with an overtaker from rest, and crossings become *easier* (target
collision 0.65 → 0.10) because the late own ship misses the CPA. Run 4, which
never trained from rest, loses 0.01 overall from rest, though its head-on goal
rate falls from 1.00 to 0.80 (3 target collisions). Suggestion 5's
out-of-distribution concern shows only in head-ons, and run 5, trained on slow
starts, scores the same from rest as from cruise; the start share is not what
cost run 5.

*The R-2 slowing test* (`tools/diagnostics/f70_r2_activation.py`). Run 5's
final policy on development crossings and head-ons; every frame with a
give-way context engaged and its compliant alteration inadmissible (the R-2
candidates):

| class | frames (episodes) | carve-out admitted, coast test (F68) | stop test (A23) |
|---|---|---|---|
| crossing | 16 (4) | **0.06** | **0.38** |
| head-on | 69 (7) | 0.00 | 0.01 |

Mean predicted DCPA in those crossing frames: coasting 0.66 m, stopping 1.44 m.
From cruise, a coast is still at most of its speed when the target passes, so it
barely differs from holding course. Under F68 the policy is almost never paid to
slacken in a crossing, which fits the faster, worse crossings. The sample is
small (4 episodes) and is run 5's own trajectory, so this is a lead, not proof.

*The physics behind it.* At propulsion stage 4 the policy has no reverse; its
only slowdown is a coast, which takes way off too slowly to matter within a
crossing's TCPA. The coast test is therefore the honest model of "the agent's
own slowdown" — and it says that slowdown rarely avoids anything. Reading your
suggestion 1 literally, R-2 kept the pre-F68 "does slowing clear" test
(`stop_clears`); I had substituted the coast.

*Run 6, training* (`runs/ppo_formulation_seed0_v6/`): run 5's command plus
`--r2-slowdown-test stop`. One change from run 5, so run 5 against run 6 is the
R-2 test alone; run 6 against run 4 is supervisor-off training plus low-speed
starts plus A23 and the noise fix. The switch is `R2_SLOWDOWN_TEST` ("coast"
default, F68 as built), read by R-2 and the Rule 8 alteration credit. Test
`test_r2_reads_the_configured_slowing_test`. **423 tests pass.**

*Caveat.* Every run is seed 0 and no two runs share a formulation, so run-to-run
spread at a fixed formulation is unmeasured. A 0.11 gap on 120 episodes is about
two binomial standard errors, before seed variance.

**F71 — review of `CODEX/` (a separate Codex working copy, copied from this tree
after F68). Nothing in it trained a policy; one real observation defect, a
usable comparator, and several formulation changes that need your call. Run 6
was stopped (by you) before its first evaluation.**

*What CODEX is.* A full copy of `src/`, `tests/` and the trainer with: a
synchronised perception state, a sixth observation branch (70 values), a
clearing-latch change, a stricter goal, scene strata with mastery-gated
curriculum, a rewritten trainer, a predictive LOS reference controller, and
behaviour contracts. Its verification is 460 tests plus 20/20 reference-
controller cases on ten hand-built scenes in a 10 m channel, and two 1 k-step
PPO smokes. No learned-policy result exists.

*Checked against this tree.*

| CODEX change | Verified here | Worth |
|---|---|---|
| Path branch cross-track error divided by the local half-width | **real defect**: `observation.py` divides by `max(MAP_WIDTH, MAP_HEIGHT)` = 25 m and observations are not normalised, so a 2.5 m offset in a 5 m corridor reads 0.1 | **port**; retrain needed (same shape, new meaning) |
| Context branch: latch state, turn sense, heading/speed change since engagement, admissibility, slowdown-clears, gates, age; previous executed action | the reward reads `psi_engage`, `u_engage`, latched sense and clearing state the policy cannot observe, so R-8/R-hold are non-Markov in the observation | **strong candidate** (A25); 56 → 70, retrain |
| Clearing latch: risk returning restores the obligation; release only after opening outside the required separation | current code releases after `n_clear` steps of CLEARING whatever happens in them | candidate (A25) |
| One pose and ego draw per decision; boundary scan, tracker, path errors and CPA all from that estimate; stale frames hold the last received pose | real, but small at nominal settings (pose 3 cm / 0.2 deg jitter, `WALK` 0, `POSE_STALE_PROB` 0): the boundary scan draws a second jitter, CPA uses true own pose against tracks built from the estimate | port for fidelity; negligible performance effect expected |
| Goal overshoot guard; `metrics.EpisodeRecorder` collision kind from the physics sub-step | edge-case bug fixes | port |
| `GOAL_CTE_RADIUS` 1.60 → 0.60 m | a stricter task; goal rates stop being comparable with runs 1–5 | decision (A25) |
| Training excludes unescapable crossings and below-floor being-overtaken draws | reverses the labelled 20 % of A15/A22 in *training* (the development set already has no unescapable crossings) | decision (A25) |
| Scene strata 20 % empty / 25 % static / 40 % dynamic / 15 % combined, 35 % recovery starts in empty/static, curriculum advancing only on ≥ 0.9 goal and zero collisions per stratum and class, no step override | run 5's obstacle collisions argue for more static practice, but with crossings at 0.45–0.70 for every policy measured, the gate would never pass level 4 | strata and recovery starts: candidate; the gate: **do not adopt** |
| R-8 slowdown credit only when the coast clears | the opposite direction to F70's evidence | do not adopt (A24 open) |
| Rewritten trainer (`ent_coef` 0.01, own development cases, lexicographic checkpoint ranking) | drops the development set, Tier 1 comparability, low-speed starts and `RetryingVecMonitor` | do not replace; cherry-pick |

*The reference controller on the development set* (`tools/diagnostics/codex_reference_devset.py`,
`results/codex_reference_devset/`; CODEX's own environment, goal tolerance
restored to 1.60 m, supervisor off, nominal noise, seeds as Tier 1):

| class | reference goal | collisions (target / obstacle / boundary) | run 4 Tier 1 goal | run 5 |
|---|---|---|---|---|
| head-on | 0.95 | 0.05 / 0 / 0 | 1.00 | 0.90 |
| crossing | **0.70** | 0.15 / 0.15 / 0 | 0.45 | 0.40 |
| overtaking | 0.90 | 0 / 0.05 / 0.05 | 1.00 | 0.95 |
| being overtaken | 0.75 | 0 / 0.20 / 0.05 | 0.85 | 0.80 |
| null | 0.90 | 0 / 0.10 / 0 | 0.85 | 0.65 |
| no target | 1.00 | 0 | 0.95 | 0.85 |
| **overall** | **0.87** | 0.13 | 0.85 | 0.76 |

It matches run 4 overall and beats every PPO run on crossings. Paired by
scenario, 7 crossings are reference-only goals, 2 run-4-only, 4 failed by both:
at least 16 of 20 development crossings are solvable from perception, so the
PPO crossing weakness is not mainly infeasible draws. 95 % of development
scenarios are solved by at least one of the two. It costs ~1.3 s of CPU per
decision (budget 0.5 s) and fell back on 39 % of crossing steps, so it is a
comparator (C3–C6), not a deployable controller. It gives way to both crossing
sides under the project's narrow-channel convention; CODEX's README notes this
is not the open-water Rule 15 role table, which the paper should state.

**F72 — A25 option 1 built: the CODEX fixes and the observable encounter latch
are in, the rest is not. Observation 56 -> 70. Run 6 trains on it.**

*Ported from `CODEX/` (F71), by copying its `env.py`, `observation.py`,
`features_extractor.py`, `metrics.py` and `colregs/context.py`, then re-applying
this tree's F70 work on top:*

1. **Cross-track error is scaled by the local channel half-width**, not by
   `max(MAP_WIDTH, MAP_HEIGHT)` = 25 m. In a 5 m corridor a 2.5 m offset read
   0.10 and now reads 1.0. Observations are not normalised (`norm_obs=False`),
   so this was the policy's whole view of where it sat in the channel.
2. **A sixth observation branch, `context`, 14 values**: per slot, ENGAGED and
   CLEARING indicators, the latched compliant turn sense, heading and surge
   change since engagement, `turn_admissible`, `slowdown_clears`,
   `admissibility_known`, `a_req`, `rho`, `in_extremis` and engagement age; then
   the previous executed rudder and throttle. `r_pf`'s gate, `v_hold` and `v_r8`
   are all defined against state latched at engagement, which the policy could
   not see: the reward was non-Markov in the observation. The slot encoder now
   takes `TARGET_FEATURES + CONTEXT_FEATURES` and shares weights across slots as
   before; the previous action joins the scene branch. Truth fields never enter
   it (`test_context_inputs_do_not_depend_on_target_truth`).
3. **Clearing is a confirmation window, not a grace period.** CLEARING used to
   release the latch after `N_CLEAR_STEPS` whatever happened in them, so a
   give-way vessel could turn back into the target and lose the obligation.
   Renewed risk now restores ENGAGED, and release requires the range to be
   opening *and* outside the required separation.
4. **One perceived state per decision.** `estimated_pose()` returns the last
   received estimate instead of drawing fresh noise on every call; the boundary
   scan, the tracker, the path errors and the encounter CPA all use it, with the
   ego estimate drawn once per frame; consecutive stale frames hold the last
   estimate actually received. Truth-side geometry moved to
   `ContextManager.attach_truth`, for diagnostics only.
5. **Two bug fixes.** The goal is no longer reached by overshooting the last
   path point along its extension (signed cross-track error is zero there), and
   `metrics.EpisodeRecorder` takes the collision kind from the physics sub-step,
   so ship contacts stop being recorded as obstacle contacts.
6. **A24 decided with A25 (option 1): `R2_SLOWDOWN_TEST = "stop"`.** R-2 and the
   Rule 8 credit read the braking-path test again, as before F68.

*Deliberately not taken* (F71 has the table): CODEX's Rule 8 credit gated on its
coast test (A24 the other way), `GOAL_CTE_RADIUS` 0.60 m, excluding the A15/A22
labelled hard cases from training, the mastery-gated curriculum, and its
rewritten trainer. Its scene strata and recovery starts are not in run 6 (A25
option 2 keeps them available; `env.reset` accepts the recovery options).

*Cost.* 26-27 steps/s per process against 30 before (~12 %), from the extra
path projection and context assembly; run 6 should take 9-11 h.

*Compatibility.* `OBSERVATION_SCHEMA_VERSION = "a25-v3-context"`, recorded in
each run's `config.json` with the dimension. **Runs 1-5's checkpoints cannot be
loaded any more**, so Tier 1 and Tier 2 on them are frozen at the numbers in
F69-F71; their results stand as recorded.

*Tests.* CODEX's `test_perception_consistency.py` and `test_context_regression.py`
were added (minus three tests belonging to changes not adopted, each with the
reason in the file), and its adapted `test_observation.py`, `test_env.py`,
`test_path_geometry.py`, plus its stale-frame and cleared-encounter tests.
**445 pass.** A 5 k-step smoke ran with the schema stamped.

*Run 6* (`runs/ppo_formulation_seed0_v6/`): 2 M steps, seed 0, 10 workers,
supervisor off in training, 15 % low-speed starts, evaluation both ways,
`R2_SLOWDOWN_TEST = "stop"`. Compare with run 4 (0.87) and run 5 (0.76) on the
development set, and with the reference controller's 0.87 / crossing 0.70.
The stopped R-2-only run is kept as `ppo_formulation_seed0_v6_stopped_r2stop/`.

**F73 — run 6 (A25) finished: goal level with run 4, the best COLREGs compliance
of any run, and crossings still unsolved. The supervisor changes nothing.**

*Run 6* (`runs/ppo_formulation_seed0_v6/`, 8.95 h). Final in-training
evaluation: goal 0.82 / collision 0.17 with the supervisor off and on.
Tier 1 under current code (`results/tiers/tier1_run6_supervisor_off/`, `…_on/`):

| development set | run 4 | run 5 | **run 6** | reference controller |
|---|---|---|---|---|
| goal | 0.85 | 0.76 | **0.83** | 0.87 |
| target / obstacle / boundary collision | 0.08 / 0.04 / 0.03 | 0.11 / 0.13 / 0 | 0.12 / 0.06 / 0 | 0.03 / 0.08 / 0.02 |
| mean COLREGs integral | −17.0 | −16.9 | **−12.2** | — |
| crossing goal | 0.45 | 0.40 | 0.40 | **0.70** |
| head-on goal | 1.00 | 0.90 | 0.85 | 0.95 |
| being overtaken / overtaking / null / no target | 0.85 / 1.00 / 0.85 / 0.95 | 0.80 / 0.95 / 0.65 / 0.85 | 0.90 / 1.00 / 0.85 / 0.95 | 0.75 / 0.90 / 0.90 / 1.00 |

Run 4 and run 5 rows are their F69/F70 replays (pre-A25 code); runs 1–5 can no
longer be replayed (F72). The goal gap to run 4 (0.83 against 0.85) is under one
binomial standard error. Compliance improved in every target class (head-on
−15.9 against run 4's −25.8 in training, overtaking −19.9 against −35.9), and
run 5's obstacle collisions are gone. Crossings did not move: 11 of 12 failures
are target collisions. Head-on width set: target collisions 0.09 / 0.15 / 0.20 /
0.21 / 0.21 at 5 / 6 / 7 / 8 / 10 m.

*Supervisor.* 8 stops in 6 of 220 episodes (7 head-on, 1 crossing), none then
hit, no outcome changed; development intervention rate 0.02. With supervisor-off
training the policy stands alone, which is the F68 framing.

**F74 — basin mode (06) built, as amended by your calls: basin is the default
geometry, Paper 2's layout, head-on-only banding, and every layout feasible.**

*Your calls.* (1) Basin mode follows Paper 2: start y 2.0 m, goal y 22.0 m, both
x drawn independently in [2.5, 7.5] m, so slant follows from the draw (up to
14.0 deg) instead of 06 §3.2's sampled angle and midpoint. (2) The 10 m basin is
already narrow water: confined traffic keeps to the whole navigable polygon,
and only **head-on** traffic keeps the path band (Rule 9(a) with Rule 14).
(3) **Basin is the default**: `DEFAULT_GEOMETRY_MODE = "basin"` for the
environment and the generator; channel mode carries only the classes whose rule
the width decides -- head-on, crossing, overtaking -- at 15 % (stage 3) or 25 %
(stages 4-5) of their draws; null, being-overtaken and no-target are always
basin. (4) Feasible but challenging (below).

*Built.*
* `corridor.Basin` (a `Corridor` with the same interface): `P_nav` is the basin
  inset by `d_safe + 0.05` = 0.40 m; `h_+(s)`, `h_-(s)` by ray to `P_nav`;
  `width = W_eff`; `band()` (clipped strip, head-on only); vertices at Paper
  2's 0.2 m so a slanted float32 leg stays under `CURVATURE_EPS` (7.6e-5 worst
  over 300 legs; 0.05 m stations gave 3.4e-4 and a phantom `r_path`).
  `sample_basin`, `build_basin`, stage slant cap with clamp-and-record.
* Generator: mode per draw (`p_basin`), rejection ledger keyed "basin", 06 §6
  record fields (`geometry_mode`, slants, midpoint, clearance profile,
  `w_eff_at_cpa`, `field_replicable`); being-overtaken starts 6.86 m along the
  leg for water astern; basin crossings keep their hull in the water to CPA
  (T15 caught them leaving through a wall); `suite_version` 3.0.
* 06 M-5 side-specific normalisation: `r_pf` and the observation's cross-track
  scale use the clearance on the deviation's side, clipped [0.60, 5.00] m; in a
  channel this is exactly `W/2` (T14).
* **Static feasibility** (`src/feasibility.py`): Paper 2's A* on a 0.25 m grid,
  walls inflated 0.40 m, panels 0.45 m, route ≤ 2.25 × the leg. Every layout is
  checked at reset, redrawn up to 20 times, then thinned nearest-the-path first.
  The Paper 3 environment had no such check before.
* `CHANNEL_MIN_WIDTH_BY_CLASS`: null and being-overtaken channels start at the
  4.26 m narrow edge (06 M-6).

*Deviations from 06, each measured.* Null traffic in basin mode keeps to `P_nav`
not the band (37 of 40 draws capped out on the band); slant cap 14.0 deg not
18.1 (Paper 2's endpoint box); Tier A basin leg 14.0 deg not 15.

*Tests.* `test_basin_mode.py` (19): T13 legs and slant clamp, affine clearances,
T14 channel reduction, side-specific normalisation, T15 (to CPA, and side walls
through an episode), T3 **per mode passes**, T7 banding rule, default geometry,
class mix, every class drawable, record fields, gate width, thinning, every
generated layout routable.

*Development set, basin default* (`tools/diagnostics/basin_devset_baselines.py`):
104 basin + 16 channel scenarios, **all A*-feasible, none needed a redraw**.

| class (mode) | path follower | reference controller |
|---|---|---|
| overall | 0.48 | **0.89** |
| crossing (basin, 18) | 0.11 | 0.61 |
| head-on (basin 13 / channel 7) | 0.54 / 0.71 | 0.92 / 1.00 |
| overtaking (basin 13 / channel 7) | 0.69 / 0.29 | 0.92 / 0.86 |
| being overtaken / null / no target | 0.80 / 0.45 / 0.35 | 0.95 / 1.00 / 0.95 |

Feasible but not easy: a follower that avoids nothing reaches the goal in half,
and the classical planner fails 11 % of feasible cases -- mostly crossings.

**F75 — crossings diagnosed (port crossings turn the wrong way), SAC and
SAC-IQN built and throughput-gated, and the suite rebuilt to 3.0 with its named
cases realised. The freeze now waits only on you.**

*1. Crossing diagnosis* (`tools/diagnostics/crossing_diagnosis.py`, run 6 vs
the reference controller, the 20 new development crossings, supervisor off):

| | reference | run 6 |
|---|---|---|
| goal | 0.60 | 0.45 |
| TCPA at engagement (median) | 7.0 s | 7.1 s |
| TCPA at first alteration > 10 deg | 3.1 s | 5.9 s |
| first alteration in the compliant sense | **0.79** | **0.25** |
| peak compliant alteration | 30 deg | 17 deg |
| speed at closest approach | 0.40 m/s | 0.52 m/s |

Paired: both 8, reference only 4, run 6 only 1, neither 7. Engagement timing is
identical, so detection and the latch are not the problem. **Direction is.** In
port crossings the compliant sense is a port turn (A17: pass astern); run 6
turned starboard first in 8 of 12 and reached the goal in 4 (reference 6 of 12);
in starboard crossings it reached 5 of 8 (reference 6). In its failures its peak
compliant alteration is 2 deg against 20 deg the wrong way. `v_port` is
symmetric in the sense (s_c = -1 penalises a starboard turn), so this is not a
reward bug: the policy generalised "give way = starboard" from head-ons and
starboard crossings, which are most give-way cases. It also slows less. A27.

*2. SAC and SAC-IQN* (`src/sac_iqn.py`; `train_formulation.py --algo
ppo|sac|sac_iqn`): the same environment, curriculum, development set,
two-mode evaluation and checkpointing for all three. SAC-IQN keeps SAC's actor
and entropy tuning with implicit-quantile critics (cosine embedding of tau,
quantile Huber loss, 32 x 32 quantiles, clipped double-Q per sample) -- the
DSAC construction, since IQN itself is discrete-action. The quantile loss
recovers a median exactly; save, load and predict work; tier tools load any
learner from its `config.json`. Smoke runs complete training and evaluation.

*Throughput gate (04a §8.3)*, 10 workers, 12 cores, learner 4 threads:

| learner | gradient steps per transition | steps/s | 2 M steps |
|---|---|---|---|
| PPO | -- | ~108 | ~6 h (9 h with two-mode evaluation) |
| SAC | 1.0 | 12 | ~46 h |
| SAC | 0.2 | 38 | ~15 h |
| SAC-IQN | 1.0 | 3 | ~185 h |
| SAC-IQN | 0.2 | 10 | ~55 h |

The protected core is 30 runs. At SAC's 15 h that is ~19 days of this machine;
with SAC-IQN in the tail, far more. TODO(04-4) is a budget and hardware call:
A26.

*3. Suite 3.0* (06 §5). Tier B restructured to 48 cells x 20 = 960 (basin 13,
channel wide 13, intermediate 13, narrow 9 without null and being-overtaken):
**960 of 960 realised**. Tier A: 38 named cases (the 32, plus six `A-BSN-*` on a
fixed 14.0-deg leg). **Before this, Tier A's named features were recorded but
never realised** -- crossing side, speed ratio, path offset, conflict and
occlusion panels were all ignored, so `A-CRP-N` was any crossing at 4 m. Now:
side and speed ratio are forced in the backward solve, the offset is set, basin
cases run their leg, and the environment places the conflict panel (beside the
path before CPA on the compliant side) and the occlusion panel (on the initial
line of sight), each kept only if the layout stays A*-feasible. **35 of 38
realised**; A-BO-N, A-NU-I and A-NU-N cannot be placed (the geometries 06 M-6
calls infeasible) and are reported by `tier_a_shortfall()`. Also fixed: the
frozen namespace holds 10,000 seeds, so `index + 10,000 x retry` retried the
same seed and Tier B cells 0 and 10 shared seeds; Tier B now takes 200 seeds per
cell and Tier A a disjoint block above. Draft manifest:
`results/suite_v3/SUITE_MANIFEST_draft.json`, 995 cases, digest `2c713ca7...`.
`test_suite_v3.py` (10).

*4. Freeze checklist* (`suite.freeze_checklist()`, now checked against git
rather than asserted): TODO(04-3) **resolved** (stage fractions of the budget,
`CURRICULUM_STAGE_FRACTIONS`); TODO(04-1) and (04-2) **explicitly deferred** to
basin session 1 at their nominal values; claim ledger and result tables
**drafted** (`planning/CLAIM_LEDGER.md` with the 04a §6 predictions,
`planning/RESULT_TABLES.md`). Open, all yours: TODO(04-4) budget (A26),
sign-off of the ledger and tables, and a commit so the generator has a SHA.

*Run 7* (`runs/ppo_formulation_seed0_v7/`): run 6's setup on the basin-default
geometry, training. The 06 §7 edits to planning documents 01, 02a, 03a and 04a
are left for Claude chat, which owns them.

**F76 — the five baseline learners share one trainer. PPO tests the reward
now; after the freeze PPO, RecurrentPPO, TD3, SAC and SAC-IQN all run over
multiple seeds (your call).**

`train_formulation.py --algo ppo|recurrent_ppo|td3|sac|sac_iqn`: one
environment, curriculum, development set, two-mode evaluation, checkpointing
and `config.json`. `sb3-contrib` 2.3.0 installed (matches SB3 2.3.2; nothing
upgraded), recorded in the new `requirements.txt`.

* **TD3**: the SAC replay settings, policy delay 2, target smoothing 0.2 clipped
  at 0.5, Gaussian exploration 0.1 on the normalised action.
* **RecurrentPPO**: PPO's settings with a 256-unit LSTM after the features
  extractor, separate for actor and critic (`MultiInputLstmPolicy`).
* **`EpisodeActor`**: every evaluation (the in-run callback, Tier 1, the
  crossing diagnosis) now acts through one object that carries RecurrentPPO's
  LSTM state through the episode; calling `predict` statelessly would have
  reset its memory every step and scored a different policy from the one trained.

Tests: `test_learners.py` (7) -- all five build and act on the Dict
observation, and the recurrent state is carried and reset. Smoke runs of TD3 and
RecurrentPPO trained, evaluated both ways and saved. Throughput for both is
measured once run 7 frees the CPU, as SAC's was (F75).

**F77 — TQC, not SAC-IQN (your correction).** The distributional arm is TQC
(Kuznetsov et al., 2020; 04a §8.2's original rank-1 entry), from `sb3-contrib`
2.3.0: SAC's settings, 2 critics x 25 quantiles, the top 2 per critic dropped.
`--algo tqc`; `src/sac_iqn.py` removed. The baselines after the freeze: PPO,
RecurrentPPO, TD3, SAC, TQC, multiple seeds. SAC-IQN's throughput rows in F75
no longer apply; TQC's is measured with TD3's and RecurrentPPO's once run 7
finishes. `test_learners.py` passes with TQC.

**F78 — port crossings are solvable, and A17's port turn is the right answer
in this simulator. Run 6's failure is learned, and one reward gap invites it.**

*Scripted responses* (`tools/diagnostics/port_crossing_responses.py`,
`results/port_crossing_responses/`): 140 crossings (20 development + 60 per
side, basin default, obstacles off, supervisor off). Each response starts when
the encounter engages, holds until the target is past and opening, then resumes
the path at cruise. Goal rate over the 109 crossings that engaged (port 57,
starboard 52):

| response | port | starboard |
|---|---|---|
| hold course and speed (Rule 17 stand-on) | 0.32 | 0.40 |
| slow (coast to the RPM floor) | 0.40 | 0.54 |
| A17 sense, 30 deg | 0.51 | 0.65 |
| A17 sense, 60 deg | 0.37 | 0.44 |
| **A17 sense, 60 deg + slow** | **0.65** | **0.67** |
| other way, 30 deg | 0.33 | 0.27 |
| other way, 60 deg | 0.12 | 0.19 |
| other way, 60 deg + slow | 0.42 | 0.48 |
| **some response solves it** | **0.81** | **0.81** |

A17's sense beats the other way on both sides: the A17 family solves 0.67 of
port crossings against 0.56, with 14 crossings only A17 solves and 8 only the
other way solves. Holding course solves 0.32, because a constant-velocity
target never gives way. Port crossings are **not** harder than starboard
crossings once the turn is combined with slowing (0.65 against 0.67). By drawn
DCPA (port): below 0.7 m only an A17 turn works (0.61 with slowing, other way
0.09); above 1.4 m holding course is enough (0.78).

(A first version held each turn to the end of the episode. 40-50 % of turned
episodes then ended on a wall before the encounter could be judged; it is kept
as `results/port_crossing_responses_v1_heldturn/` and not used.)

*The reward gap.* A held wrong-way heading costs exactly what no action costs.
Port crossing, engaged, TCPA 4 s: `v_port` 0.000 and `v_r8` 0.733 both for no
action and for 30 deg to starboard held; a compliant 30 deg port turn scores
`v_r8` 0.000. `v_port` measures the wrong-way **yaw rate**, which is zero once
the heading is held, and `v_r8` credits compliant displacement but charges none
for wrong-way displacement. So a swerve to starboard costs only the seconds of
turning, and in port crossings that swerve is what run 6 learned (8 of 12 first
turns, F75).

*Conclusion.* The geometry is solvable (0.81 by the best scripted response),
A17 is the better rule for these targets, and the policy's failure is learned.
A27 has the options.

**F79 — run 7 (basin default) matches run 6 on goals but not on compliance,
and confirms the port-crossing failure on the basin geometry.**

*Run 7* (`runs/ppo_formulation_seed0_v7/`, 6.0 h, run 6's setup on the
basin-default distribution): in-run 0.84 goal / 0.16 collision with the
supervisor off and on. Tier 1 on the basin-default development set, with run 6
replayed on the same set for comparison:

| development set (basin default) | run 6 (trained on channels) | run 7 (trained basin default) |
|---|---|---|
| goal | **0.875** | 0.842 |
| target / boundary / obstacle collision | 0.12 / 0 / 0.01 | 0.13 / 0.03 / 0 |
| mean COLREGs integral | **−11.1** | −19.8 |
| mean / max speed (m/s) | 0.61 / 0.81 | 0.66 / 0.84 |
| crossing goal (port / starboard) | 0.45 (0.33 / 0.62) | 0.30 (0.17 / 0.50) |
| head-on / overtaking / being overtaken | 1.00 / 1.00 / 0.85 | 0.85 / 1.00 / 0.90 |
| null / no target | 1.00 / 0.95 | 1.00 / 1.00 |

Reference controller on the same port / starboard crossings: 0.50 / 0.75.

*Reading it.*
* **Run 6 transfers to the basin without training on it** (0.875, and its best
  compliance), which is C-8's claim; run 7 gains nothing from the basin
  distribution on goals and loses on compliance. One seed each.
* **Port crossings, confirmed and worse:** 2 of 12 (reference 6). The first
  alteration is compliant in 0.22 of port crossings, and speed at closest
  approach is 0.72 m/s (reference 0.40), so the policy neither turns the A17
  way nor slows. A27's options 1 and 2 target exactly this.
* **Run 7 outruns overtakers:** being-overtaken max speed 0.94 m/s against run
  6's 0.68, and the stand-on `v_hold` integral doubles (8.7 against 4.3). Goals
  hold (0.90), but it is the stand-on violation `v_hold` exists to price;
  watch it in run 8.
* Supervisor: 10 stops in 8 of 220 episodes (6 head-on, 4 overtaking), none
  then hit; outcomes identical off and on. Head-on width set: target
  collisions 0.09 / 0.15 / 0.15 / 0.00 / 0.00 at 5 / 6 / 7 / 8 / 10 m.

**F80 — throughput for all five baselines** (10 workers, 12 cores; 1.0 / 0.2
gradient steps per transition for the off-policy learners): PPO ~108 steps/s,
RecurrentPPO 61, TD3 18 / 53, SAC 12 / 38, TQC 13 / 32. Per 2 M-step seed:
~6, ~9, ~31 / ~10.5, ~46 / ~15, ~43 / ~17 h. One seed of all five is ~135 h at
1.0 and ~58 h at 0.2; five seeds one at a time are ~28 or ~12 days, before
ablations. A26 carries the decision.

**F81 — A27 options 1 and 2 built (your call); run 8 trains on them.**

1. **`v_port` charges a held wrong-way heading.** Besides the wrong-way yaw
   rate, it now reads the heading displaced against the compliant sense since
   engagement, `(held - 5 deg) / DPSI_MIN`, clipped to [0, 1]; the larger form is
   charged, ENGAGED only, on the perceived heading the latch recorded. Port
   crossing, TCPA 4 s: no action `v_port` 0, compliant 30 deg port 0, 4 deg
   starboard 0 (deadband), 10 deg starboard held 0.25, 30 deg held 1.00. Held
   wrong-way now costs more than doing nothing, which costs more than complying.
   `V_PORT_HEADING_DEAD_DEG` = 5.0 (TODO(05), heading noise).
2. **Port crossings earlier and more.** Stage 3 now draws crossings from both
   sides alongside head-on and null (stage-3 crossing yield 59 of 60).
   Training crossings come from port at `CROSSING_PORT_SHARE_TRAINING` = 0.60
   (measured 0.58 over 297 draws). The development and frozen namespaces keep
   the even integer draw: the development set is identical to run 7's (classes
   and crossing angles checked), so runs 7 and 8 compare on the same scenarios.

*Reward-scale audit* (`results/scale_audit_a27.json`, 240 episodes per
policy): the orderings hold -- clean success 169, success with COLREGs
penalties 97, any collision −323 (before: 172 / 101 / −337). The COLREGs term's
episode integral rises by about 5 (random policy −18.7 → −23.8, follower
−16.4 → −21.1). The earlier audit predates basin mode, so other terms moved
with the geometry too.

Tests: `test_a27_a_held_wrong_way_heading_costs_more_than_doing_nothing`,
`test_a27_returning_after_the_encounter_is_not_charged`,
`test_a27_training_crossings_favour_port_and_other_namespaces_do_not`,
`test_a27_stage_3_teaches_crossings`.

*Run 8* (`runs/ppo_formulation_seed0_v8/`): run 7's setup plus both changes.
Compare with run 7 on the same development set; the port-crossing diagnosis
and the being-overtaken speed (F79) are the things to read.

**F83 — B8: LOS-PID + DWA and encounter-specific VO built, on Tier 1.**

`src/classical/` (`common.py`, `los_dwa.py`, `encounter_vo.py`). Both read only
what the policies and the reference controller read -- held pose and ego, the
gated scan, tracks and their contexts, path and map -- and a test runs them
against an environment with the truth attributes removed
(`tests/test_classical.py`, 12 tests; suite 497 passing). Both plan on the
nominal identified hull; the episode's hull is randomised, as for the policies.

* **LOS-PID + DWA.** LOS (2.5 m lookahead) and a heading PID follow the path;
  DWA searches yaw-rate x speed set-points plus the LOS-PID command, each
  predicted closed-loop on the identified hull (the dynamic window, since a
  one-step `(u, r)` window admits arcs this hull cannot fly), held 3 or 6 s then
  handed back to LOS-PID. Fox's heading/dist/velocity objective, plus path
  deviation and smoothness. No COLREGs.
* **Encounter-specific VO** (our reading of Thyri & Breivik 2022, not a port):
  hard hull VO; soft own-ship domain scaled per class; passing-side constraint
  for give-way classes (head-on port-to-port, crossing astern from either side
  per A17, overtaking on `compliant_turn_sense`), released in extremis; Rule 17
  stand-on for being overtaken, released 12 s before a hard violation. Each
  candidate velocity is swept along a turn-lag / turn-rate / surge-lag model and
  flown by a cascaded course autopilot at that rate. `select_velocity` is
  environment-free so `T-RE` can call it (03a §5.3) -- **not yet wired**.

*What the hull forced* (each found on a tuning set disjoint from Tier 1 --
development namespace, indices 600,000+, 8 per class): a textbook VO's
instantaneous velocity change procrastinates into obstacles on a hull that
needs ~5 s to build its yaw rate; a fixed time horizon rewards slowing (DWA now
judges static obstacles over equal travelled distance); "slowest" fallbacks
are wrong on a hull that coasts ~60 s; the reference controller's 10 s scan
memory smears an unconfirmed target into a phantom wall (2 s here); the VO
needed a soft static margin (no-target goal 0.40 -> 0.95 on Tier 1; the
hard-margin run is kept as `tier1_encounter_vo_supervisor_off_hardstatic`).
Tuning set, final: DWA 0.92, VO 0.83.

*Tier 1* (`results/classical_comparison/`, `tools/tiers/compare_tier1.py`;
paired exact McNemar against run 7):

| development set (120), supervisor off | goal | target coll. | crossing goal (port / stbd) | McNemar p vs run 7 |
|---|---|---|---|---|
| PPO run 7 | 0.842 | 0.133 | 0.17 / 0.50 | — |
| LOS-PID + DWA | 0.883 | 0.100 | 0.42 / 0.62 | 0.30 |
| encounter-specific VO | 0.842 | 0.108 | 0.25 / 0.50 | 1.00 |
| reference (CODEX, F74 replay) | 0.892 | 0.075 | — | 0.24 |

Head-on width set (100, obstacles off): run 7 0.85, DWA 0.96 (p = 0.013), VO
0.99 (p = 0.0001). Supervisor on changes one VO episode and nothing else.
**Crossing is the hard class for every method**, classical included; neither
comparator beats run 7 significantly on the development set. The classical
methods succeed as often but pay more in the COLREGs term in head-on (DWA −35,
VO −33 vs run 7 −18) and overtaking (VO −54); DWA overtakes slowly (0.35 m/s).

**F82 — run 8 (A27 options 1 + 2) fixes port crossings; starboard crossings
now start with a port swerve, and the stand-on speeding persists.**

*Run 8* (`runs/ppo_formulation_seed0_v8/`, 6.3 h): in-run 0.85 final, peak 0.91
at 1.8 M (`best_model`). Tier 1 and the crossing diagnosis on the same
development set as run 7 (F79), supervisor off:

| | run 7 | run 8 final | run 8 best (1.8 M) | reference |
|---|---|---|---|---|
| goal | 0.842 | 0.850 | 0.908 | -- |
| crossing goal | 0.30 | 0.55 | **0.75** | 0.60 |
| port crossings | 2 / 12 | **6 / 12** | **8 / 12** | 6 / 12 |
| starboard crossings | 4 / 8 | 5 / 8 | 7 / 8 | 6 / 8 |
| first alteration compliant, port | 0.22 | **0.70** | **0.90** | -- |
| first alteration compliant, starboard | 0.14 | 0.14 | 0.00 | -- |
| speed at closest approach, port (m/s) | 0.72 | 0.45 | 0.44 | 0.40 (all) |
| mean COLREGs integral | −19.8 | −18.2 | −22.3 | -- |
| being overtaken: max speed / `v_hold` integral | 0.94 / 8.7 | 0.80 / 10.3 | -- / -- | -- |

Head-on width set, run 8 final: target collisions 0.05 / 0 / 0 / 0 / 0.05 at
5 / 6 / 7 / 8 / 10 m (run 7: up to 0.15). Supervisor: 2 stops in 220 episodes,
none then hit.

*Reading it.*
* **Port crossings fixed.** 6 of 12 (final) and 8 of 12 (best) against run 7's
  2; the first alteration is now the A17 port turn in 0.70-0.90 of them, and
  the ship slows (0.45 m/s at closest approach against 0.72). The best
  checkpoint's crossings (0.75) beat the reference controller's (0.60).
* **New: starboard crossings start with a port swerve.** The first >10 deg
  alteration is compliant in 0.14 (final) and 0.00 (best) of starboard
  crossings, with a 30-50 deg wrong-way peak; the policy recovers and reaches
  the goal in 5-7 of 8, but the swerve is a Rule 8 violation in the log. In
  several of them it starts before the encounter engages, where the new
  heading form does not charge. The 60/40 port share is the likely cause:
  crossings now mostly want a port turn, and the policy learned "crossing ->
  port" before it learned the side.
* **Stand-on speeding persists** (being-overtaken max 0.80 m/s, `v_hold`
  integral 10.3 against run 7's 8.7). Not addressed by A27.
* The best checkpoint was selected on this development set, so its 0.908 is
  optimistic; the final model's 0.850 is the fair number. One seed.

**F83 — A28 (your call: options 1 and 3). Run 9 trains with the port share
back at 0.5; the stand-on speeding is diagnosed; the development set has a
gallery.**

*Option 1 -> run 9* (`runs/ppo_formulation_seed0_v9/`): run 8's setup with
`CROSSING_PORT_SHARE_TRAINING` back to 0.50; the wrong-way heading term and
stage-3 crossings stay. It also attributes A27: if port crossings hold, the
heading term did the work.

*Option 3 -- why the policy outruns an overtaker*
(`tools/diagnostics/standon_speed.py`, the 20 development being-overtaken
episodes, supervisor off; per-step means while the being-overtaken context is
ENGAGED):

| | run 8 | run 6 |
|---|---|---|
| engaged steps per episode | 26.6 | 16.9 |
| speed (m/s) / throttle | 0.69 / +0.31 | 0.48 / −0.28 |
| steps above the overspeed tolerance | 0.54 | 0.00 |
| steps `v_hold` fires | 0.65 | 0.41 |
| COLREGs term / path term / progress term | −1.43 / −0.58 / +1.48 | −0.93 / −0.46 / +1.49 |
| target collisions (of 20) / median closest range | 1 / 2.30 m | 3 / 1.72 m |

Speeding does not pay in progress (the progress term is the same per step) and
it lengthens the encounter (the overtaker takes longer to pass). What it buys
is distance from a target that never gives way: the training overtaker is
constant-velocity (D1), so holding course is only as safe as the A15 floor
makes it, and 20 % of draws are labelled below it. And it is almost free:
**`v_hold` saturates** -- its surge part reaches full severity at
`DU_HOLD` = 0.10 m/s off the engagement speed, so once the policy deviates at
all, deviating further costs nothing more. Run 6 slowed instead, which is also a
`v_hold` violation, and was rear-ended 3 times. Options: A29.

*Gallery* (`tools/diagnostics/devset_gallery.py` -> `results/devset_gallery/`):
one figure per development scenario (geometry, realised panels, the leg, the
target's track to CPA, outcomes of run 7, run 8 final and best, and the
reference controller), a contact sheet per class, `index.csv` and a README.
The set is **104 basin + 16 channel**; basin legs have mean |slant| 5.2 deg,
max 12.6 deg (43 % above 5 deg, 13 % above 10 deg), because two independent x
draws in [2.5, 7.5] m over a 20 m leg rarely differ by much.

**F85 — A29 (your call): `v_hold`'s speed part keeps rising past 0.10 m/s.
Built behind `V_HOLD_GROWS`; run 10 switches it on after run 9 is scored.**

The option as first written (a linear ramp to full severity at 0.30 m/s) would
have *cut* the penalty at the speeds run 8 flees at: 0.13 m/s over its
engagement speed costs 1.00 today and 0.43 on that ramp. Built instead:
unchanged up to `DU_HOLD` = 0.10 m/s (full severity there, as before), then +1
per further 0.10 m/s, capped at 3x (`V_HOLD_EXCESS_SPAN`, `V_HOLD_CAP`):

| speed change (m/s) | 0.05 | 0.10 | 0.13 | 0.25 | 0.30+ |
|---|---|---|---|---|---|
| before | 0.25 | 1.00 | 1.00 | 1.00 | 1.00 |
| now | 0.25 | 1.00 | 1.30 | 2.50 | 3.00 |

The COLREGs group still clips at 1, so being overtaken can cost up to ~2.2x what
it did (0.45 x 3 = 1.35, clipped). Test
`test_a29_fleeing_an_overtaker_keeps_costing_more`; the reward suite passes
(59). `V_HOLD_GROWS` stays False until run 9's analysis has run, so run 9 is
scored with the reward it trained on; `results/after_run9_run10.sh` then
switches it on, re-runs the reward tests and the scale audit
(`results/scale_audit_a29.json`), launches run 10 (run 9's setup plus A29) and
analyses it, including `standon_speed.py`.

**F84 — run 9 (port share 0.50, A28 option 1) loses the port-crossing fix and
keeps the starboard swerve: the share drove the fix, and did not cause the
swerve. Run 10 goes back to 0.60 and adds A29 (your call).**

*Run 9* (`runs/ppo_formulation_seed0_v9/`, 5.0 h): in-run 0.82 final. Tier 1
and the diagnoses on the development set, supervisor off, against run 8:

| | run 8 (0.60) | run 9 (0.50) |
|---|---|---|
| goal | 0.850 | 0.817 |
| crossing / head-on / overtaking | 0.55 / 0.85 / 0.95 | 0.40 / 0.90 / **0.80** |
| being overtaken / null / no target | 0.90 / 0.90 / 0.95 | 0.85 / 0.95 / 1.00 |
| port crossings | **6 / 12** | **2 / 12** |
| first alteration compliant, port | 0.70 | 0.56 |
| speed at closest approach, port (m/s) | 0.45 | 0.74 |
| starboard crossings / first alteration compliant | 5 / 8, 0.14 | 6 / 8, **0.12** |
| mean COLREGs integral | −18.2 | −19.0 |
| stand-on (engaged): speed, above tolerance, `v_hold` fires | 0.69, 0.54, 0.65 | 0.65, 0.38, 0.60 |
| head-on width set, target collision at 5 m | 0.05 | 0.18 |

*Reading it.* A27's attribution comes out the other way from the working
guess: the heading term alone (run 9) moves the first alteration toward
compliance (0.22 in run 7, 0.56 here) but does not fix the outcomes -- the
policy neither completes the port turn nor slows (0.74 m/s at closest
approach). The 60/40 share is what made port crossings succeed. The port swerve
at the start of starboard crossings survives the 50/50 share (0.12 compliant),
so the share did not cause it; A28 option 2 (charge the wrong-way heading from
detection, not engagement) remains the candidate. Stand-on speed is unchanged,
as expected before A29. One seed each: run-to-run spread is unmeasured, and
4 of 12 port crossings is within what a seed could move.

*Run 10* (`runs/ppo_formulation_seed0_v10/`): `CROSSING_PORT_SHARE_TRAINING`
back to 0.60 plus A29 (`V_HOLD_GROWS` on). A29's audit
(`results/scale_audit_a29.json`): orderings unchanged (169 / 95 / −323); the
random policy's COLREGs integral −23.8 → −24.7. The range test now allows
`v_hold` its 3x cap. **Process note:** a queue copy believed stopped was still
running, flipped the switch early and started a second audit; it was killed,
and a first run-10 launch on the 0.50 share was aborted within minutes
(`runs/ppo_formulation_seed0_v10_aborted_share050/`, not used). Outcomes
unaffected; run 9 was scored with the reward it trained on.

**F86 — run 10 paused at 1.40 M of 2 M steps (your call, 2026-09-19 20:08),
with two ways to continue.**

* **In place (no loss):** its learner and 10 workers are suspended
  (`tools/pause_run.ps1 -Tag v10 -Action pause`). Continue with
  `powershell -ExecutionPolicy Bypass -File tools\pause_run.ps1 -Tag v10 -Action resume`;
  `results/run10.sh` is still waiting on it and runs the analysis at the end.
  Suspension does not survive a reboot or sign-out.
* **After a reboot:** `bash results/run10_resume_after_reboot.sh` continues from
  the latest checkpoint (1.25 M steps; 0.15 M re-trained) and then analyses.
  Use one route, never both.

`train_formulation.py --resume RUN_DIR` (new): reads the learner, budget and
flags from the run's `config.json`, loads the latest checkpoint and its reward
normaliser, continues the step count (curriculum and checkpoints follow it),
trains only the remaining budget, keeps the evaluation history and best score up
to the checkpoint (later rows are dropped and redone), writes a separate
`resume_<steps>.monitor.csv`, and records the resume in `config.json`. It
**refuses to resume** if the formulation switches (`R2_SLOWDOWN_TEST`, the port
share, `V_HOLD_GROWS`, the heading deadband, the geometry default, the
observation schema) differ from the run's; runs now record them, and run 10's
config was stamped with the values its launch script asserted. Off-policy
checkpoints now also save the replay buffer, so a resumed TD3, SAC or TQC run
keeps its data. `--checkpoint-every` sets the interval. Tested end to end on a
smoke run (resumed at 4,096 of 8,192 steps; evaluation and curriculum
continued). The environment episode stream after a resume is not the one an
uninterrupted run would have seen.

**F87 — run 10 (60/40 + A29) is the best final policy so far; A29 stops the
stand-on speeding; the starboard-crossing swerve remains.**

*Run 10* (`runs/ppo_formulation_seed0_v10/`, 5.3 h of training, paused once at
1.40 M and resumed in place, F86): in-run 0.88 final. Tier 1 and the diagnoses
on the development set, supervisor off:

| | run 8 | run 9 | **run 10** |
|---|---|---|---|
| goal | 0.850 | 0.817 | **0.875** |
| mean COLREGs integral | −18.2 | −19.0 | **−16.5** |
| crossing / head-on / overtaking | 0.55 / 0.85 / 0.95 | 0.40 / 0.90 / 0.80 | **0.65** / 0.90 / **1.00** |
| being overtaken / null / no target | **0.90** / 0.90 / 0.95 | 0.85 / 0.95 / 1.00 | 0.75 / 0.95 / 1.00 |
| port crossings (first alteration compliant) | 6 / 12 (0.70) | 2 / 12 (0.56) | **8 / 12** (0.67) |
| starboard crossings (first alteration compliant) | 5 / 8 (0.14) | 6 / 8 (0.12) | 5 / 8 (**0.00**) |
| stand-on, engaged: speed / above tolerance / `v_hold` fires | 0.69 / 0.54 / 0.65 | 0.65 / 0.38 / 0.60 | **0.55 / 0.05 / 0.46** |
| stand-on, engaged: throttle | +0.31 | +0.29 | −0.07 |

Head-on width set: target collisions 0.14 / 0.10 / 0 / 0.05 / 0.05 at
5 / 6 / 7 / 8 / 10 m. Supervisor: 8 stops in 6 of 220 episodes, none then hit.

*Reading it.*
* **A29 works.** Engaged and being overtaken, the ship now holds cruise
  (0.55 m/s against 0.56 nominal; above the overspeed tolerance in 5 % of steps
  against 54 %), and `v_hold` fires on fewer steps.
* **Holding course costs being-overtaken goals** (0.75 against 0.90): 2 of the 5
  failures are the development set's below-floor draws (A15's labelled 20 %,
  where the overtaker's track reaches the hull and holding cannot be safe), 1 an
  above-floor target collision after slowing, and 2 boundary collisions. That is
  the price of a non-yielding overtaker (A29 option 2 would have addressed it
  by making some overtakers keep clear).
* **Port crossings best yet** (8 of 12) with the 60/40 share restored;
  crossings overall 0.65, above the reference controller's 0.60.
* **The starboard-crossing port swerve persists** (first alteration compliant
  0.00): A28 option 2 is still open.
* One seed per configuration: run-to-run spread is unmeasured.

**F88 — the starboard-crossing swerve discounts its own penalty; run 11 weighs
`v_port` by the peak risk since engagement. Being overtaken, laid out (A30).**

*When the swerve starts* (`tools/diagnostics/swerve_timing.py`, run 10, the 8
development starboard crossings): every first alteration is to port, 2.0-4.5 s
into the episode. In 3 the target is not yet tracked (tracked at 2.5-5.5 s); in
5 the encounter is already ENGAGED (engaged at 1.5-2.0 s). The "detected but not
engaged" window that A28 option 2 would have charged never occurs at the turn,
so option 2 would change nothing and was not built.

*Why the engaged swerves are cheap.* The COLREGs group is not saturated while
swerving (pre-clip 0.30 on average, never above 1). `v_port` averages 0.35 there
although the heading is held 25-50 deg the wrong way, because every COLREGs
severity is multiplied by `rho = clip(1 - DCPA / (kappa * d_req))`, and a swerve
in **either** direction opens DCPA: median `rho` 0.52 while swerving. The
wrong-way manoeuvre lowers the weight of its own violation.

*Fix (run 11).* `v_port` is weighted by the largest `rho` since engagement
(`EncounterContext.rho_latched`, kept in the latch and reset with it;
`V_PORT_LATCHED_RHO`). Every other term keeps the current `rho`. Head-on and
overtaking wrong-way turns are weighted the same way. The 3 swerves that begin
before tracking cannot be charged by a COLREGs term -- no vessel is perceived --
but once the encounter engages, holding the swerve now costs at the encounter's
peak risk. Tests: `test_f88_a_swerve_cannot_discount_its_own_penalty`,
`test_f88_the_latched_risk_is_the_peak_since_engagement`; the reward suite
passes (61). Run 11 = run 10's setup + this; `results/run11.sh` audits the
reward first (`results/scale_audit_f88.json`) and analyses after, including
the swerve timing.

*Being overtaken, by stratum* (development set: 17 draws at or above A15's
floor, 3 below it):

| run | above floor | below floor | max speed (m/s) | `v_hold` integral | behaviour |
|---|---|---|---|---|---|
| 6 | **17 / 17** | 0 / 3 | 0.68 | 4.3 | slows |
| 7 | 15 / 17 | **3 / 3** | 0.94 | 8.7 | flees |
| 8 | 16 / 17 | 2 / 3 | 0.80 | 10.3 | flees |
| 9 | 16 / 17 | 1 / 3 | 0.82 | 10.1 | flees |
| 10 (A29) | 14 / 17 | 1 / 3 | 0.79 | 6.3 | holds cruise |

Fleeing is what rescues the below-floor draws, where the overtaker's track
reaches the hull and holding course cannot be safe; A29 now charges it, so run
10 holds and loses them. Above the floor, holding should always work, yet run 10
loses 3 (1 target collision after slowing, 2 boundary collisions after late
evasions). A30 has the options.

**F89 — run 11 (latched `v_port` weight) is the best policy yet and fixes the
starboard-crossing swerve -- but port crossings now start with a starboard turn.**

*Run 11* (`runs/ppo_formulation_seed0_v11/`, 5.0 h): in-run 0.93 final. Reward
audit (`results/scale_audit_f88.json`): orderings unchanged (169 / 94 / −324);
the random policy's COLREGs integral −24.7 -> −26.9. Tier 1 and the diagnoses,
development set, supervisor off:

| | run 8 | run 10 | **run 11** |
|---|---|---|---|
| goal | 0.850 | 0.875 | **0.925** |
| head-on / crossing / overtaking | 0.85 / 0.55 / 0.95 | 0.90 / 0.65 / 1.00 | **1.00 / 0.75** / 0.95 |
| being overtaken (above / below floor) | 16 / 17, 2 / 3 | 14 / 17, 1 / 3 | **17 / 17, 2 / 3** |
| null / no target | 0.90 / 0.95 | 0.95 / 1.00 | 0.95 / 0.95 |
| port crossings, goal (first alteration compliant) | 6 / 12 (0.70) | 8 / 12 (0.67) | 8 / 12 (**0.10**) |
| starboard crossings, goal (first alteration compliant) | 5 / 8 (0.14) | 5 / 8 (0.00) | **7 / 8 (1.00)** |
| speed at closest approach, port / starboard (m/s) | 0.45 / 0.33 | 0.47 / 0.41 | 0.68 / 0.64 |
| stand-on engaged speed / above tolerance | 0.69 / 0.54 | 0.55 / 0.05 | 0.57 / 0.02 |
| mean COLREGs integral (crossing) | −18.2 | −16.5 (−21.6) | −17.4 (−28.6) |
| head-on width set, target collision 5 / 6 / 7 / 8 / 10 m | 0.05 / 0 / 0 / 0 / 0.05 | 0.14 / 0.10 / 0 / 0.05 / 0.05 | 0 / 0.25 / 0.20 / 0.16 / 0.05 |

Supervisor: 7 stops in 6 of 220 episodes, none then hit.

*Reading it.*
* **The starboard swerve is gone:** every starboard crossing now starts with the
  compliant starboard turn (1.00, against 0.00 in run 10), and 7 of 8 reach the
  goal. Swerve timing: the first alteration now comes 2.5-8.0 s in, mostly
  after engagement, and is starboard in 6 of 8.
* **Port crossings mirror it:** the first alteration is compliant in only 0.10
  of them -- the policy now opens *every* crossing with a starboard turn. Goals
  hold (8 of 12), and the crossing COLREGs integral worsens (−28.6).
* Across runs 8-11 the policy has picked one opening direction for all crossings
  (port in runs 8-10, starboard in run 11) instead of conditioning it on the
  side the target comes from. Whether that is the formulation or the seed is
  unknown with one seed per configuration.
* Being overtaken recovers to 0.95 while holding cruise (A29 kept): all 17
  above-floor draws, and 2 of the 3 below-floor ones. A30 is held.
* The head-on width set regressed at 6-8 m (0.16-0.25 target collisions).

**F90 — the seed replicate: run 11 scores 0.925 and 0.875 on the same
formulation, and the crossing opening direction is a seed accident, not a bias.
Single-seed differences below ~0.05 mean nothing.**

*Run 11, seed 1* (`runs/ppo_formulation_seed1_v11/`, same settings as seed 0;
paused overnight, the suspended processes did not survive it, so it finished
from its 1.75 M checkpoint, F86):

| development set, supervisor off | run 11 seed 0 | run 11 seed 1 | run 10 (seed 0) |
|---|---|---|---|
| goal | 0.925 | 0.875 | 0.875 |
| head-on | 1.00 | 1.00 | 0.90 |
| crossing | 0.75 | 0.55 | 0.65 |
| overtaking / null / no target | 0.95 / 0.95 / 0.95 | 0.90 / 0.95 / 0.95 | 1.00 / 0.95 / 1.00 |
| being overtaken (above / below floor) | 0.95 (17/17, 2/3) | 0.90 (16/17, 2/3) | 0.75 (14/17, 1/3) |
| mean COLREGs integral | −17.4 | −19.7 | −16.5 |
| port crossings: goal, first alteration compliant | 8/12, 0.10 | 6/12, 0.30 | 8/12, 0.67 |
| starboard crossings: goal, first alteration compliant | 7/8, **1.00** | 5/8, **0.43** | 5/8, 0.00 |

*Reading it.*
* **Seed spread is about 0.05 on the headline** (0.925 against 0.875) and **0.20
  in the crossing class** (0.75 against 0.55). Every formulation comparison so
  far has been one seed against one seed: run 8 -> run 10 (+0.025) and run 10 ->
  run 11 (+0.05) are inside that spread, so only the large, mechanism-backed
  movements are safe to read -- port crossings under A27 (2/12 -> 8/12), the
  stand-on speed under A29 (0.69 -> 0.55 m/s), the starboard first alteration
  under F88 (0.00 -> 1.00 at seed 0).
* **The opening direction is not systematic.** Seed 0 opens every crossing to
  starboard (port compliant 0.10, starboard 1.00); seed 1 is mixed (0.30 /
  0.43). Neither reads the side reliably, and the bias differs by seed, so this
  is a formulation gap rather than the latched weight pushing one way: A31.
* **What holds across both seeds:** head-on 1.00, being overtaken 0.90-0.95 with
  all-but-one above-floor draws and 2 of 3 below-floor ones, the supervisor
  intervening in 1-2 % of episodes, and no stop followed by a collision.

**F91 — the head-on regression: run 11 alters to starboard even where there is
no starboard room. Fixed for run 12, with A31's stage-3 crossing share.**

*Diagnosis* (`tools/diagnostics/channel_headon.py`, the head-on width set,
obstacles off). Narrow channels (5-6 m), split by whether the compliant
starboard alteration was admissible when the encounter engaged:

| narrow head-ons | run 8 | run 11 seed 0 | run 11 seed 1 |
|---|---|---|---|
| starboard **inadmissible**: altered to starboard first | 0.27 | **1.00** | **1.00** |
| starboard inadmissible: target collision | 0.05 | 0.16 | 0.33 |
| starboard admissible: target collision | 0.00 | 0.06 | 0.05 |
| first alteration to starboard, all widths | 0.21-0.32 | 0.95-1.00 | 1.00 |

Run 11 is *more* compliant in the head-on sense and *worse* in a channel: A27's
held-heading charge, weighted at peak risk by F88, prices any port heading, so
where the starboard turn does not fit the policy turns into the squeeze instead
of slackening speed. 02 §4.4 says the answer there is 8(e), and R-2 already pays
for it.

*Fix (F91).* The held-heading form of `v_port` applies only where the compliant
alteration is admissible (`V_PORT_HEADING_NEEDS_ADMISSIBLE`); the yaw-rate form
still charges a wrong-way *turn* everywhere. Test
`test_f91_no_held_heading_charge_where_the_compliant_turn_has_no_room`.

*A31, revised by measurement.* The planned fix -- exposing the compliant sense
before the latch -- was falsified: the sense is already in the observation
pre-latch, and the policy **does** read it. Probing the two run 11 seeds at
engagement, flipping that one input moves the rudder by 0.49-1.03 (seed 0) and
0.15-0.20 (seed 1). Seed 0 commands +0.79 rudder (starboard) for a target from
starboard and +0.21 for one from port: it responds to the input, but on top of a
standing starboard bias the response cannot overturn. So the fix is training
exposure, not observability: **stage 3 now draws crossings as often as head-ons**
(`weights` per stage; stage 3 is 0.35 head-on / 0.35 crossing / 0.15 null /
0.15 no-target), so the side distinction is learned where head-on's starboard
answer is learned. Test
`test_a31_stage_3_draws_crossings_as_often_as_head_ons`.

*Run 12* (`results/run12.sh`): both changes, **two seeds**, each with Tier 1,
the crossing diagnosis, the stand-on and swerve checks, then a channel head-on
comparison against runs 8 and 11. **502 tests pass.**

**F92 -- run 12 scored: F91 halves the narrow-channel collisions but the two
seeds solve it by opposite means, and A31 does not remove the crossing side
bias. The formulation is not ready to freeze.**

*Headline* (development set, supervisor off, final checkpoint). Run 12 seed 0
0.83, seed 1 0.88, against run 11's 0.92 / 0.88 and run 8's 0.85 -- inside the
0.05 seed spread of F90, and neither change was aimed at it.

| class | run 8 s0 | run 11 s0 | run 11 s1 | run 12 s0 | run 12 s1 |
|---|---|---|---|---|---|
| all | 0.85 | 0.92 | 0.88 | 0.83 | 0.88 |
| head-on | 0.85 | 1.00 | 1.00 | 0.90 | 0.95 |
| crossing | 0.55 | 0.75 | 0.55 | **0.40** | **0.60** |
| being overtaken | 0.90 | 0.95 | 0.90 | 0.90 | **0.80** |

*F91 worked where it was aimed, and over-shot* (`results/channel_headon/`, the
head-on width set, grouped by whether the starboard alteration was admissible at
engagement):

| group | | run 8 | run 11 s0 | run 12 s0 | run 12 s1 |
|---|---|---|---|---|---|
| narrow (5-6 m), starboard **inadmissible** | altered starboard first | 0.27 | **1.00** | **0.00** | 0.64 |
| | any collision | 0.05 | **0.36** | 0.15 | 0.18 |
| wide (8-10 m), starboard **admissible** | altered starboard first | 0.22 | 1.00 | **0.06** | 0.68 |
| | any collision | 0.03 | 0.11 | 0.00 | 0.05 |
| | speed at closest point (m/s) | 0.84 | 0.81 | 0.91 | **0.56** |

Removing the held-heading charge where the compliant turn has no room did stop
the squeeze: run 11's 0.36 collision rate in inadmissible narrow head-ons falls
to 0.15-0.18. But the charge was the only thing pricing a held course there, and
the two seeds replaced it differently. **Seed 0 abandoned the starboard
alteration altogether** -- 0.06 starboard-first in wide channels where starboard
is admissible in 100 % of draws, turning port at t = 2.0 s at 0.91 m/s. That is
a Rule 14 failure bought at the price of a Rule 8(e) one. **Seed 1 found the
intended answer** -- 0.68 starboard-first in wide channels, altering late
(t = 8.5 s) at 0.56 m/s, i.e. slackening speed and still turning the right way.
F91 is therefore necessary but under-determined: it removes a wrong pressure
without supplying the right one, and which behaviour fills the gap is left to
the seed. Run 8 remains the best on collisions (0.05 narrow, 0.03 wide) while
altering starboard in only 0.22-0.27 of head-ons -- it avoids collisions by not
committing to a sense at all.

*A31 failed: exposure changes which way the bias points, not that there is one.*
First alteration compliant with 02 §4.3's turn sense, by the side the target
came from (development crossings, `results/crossing_diagnosis/`):

| run | target from port | target from starboard | crossing goal |
|---|---|---|---|
| run 8 s0 | 0.58 | 0.12 | 0.55 |
| run 10 | 0.50 | 0.00 | 0.65 |
| run 11 s0 | 0.08 | 0.75 | 0.75 |
| run 11 s1 | 0.25 | 0.38 | 0.55 |
| run 12 s0 | **0.42** | **0.12** | 0.40 |
| run 12 s1 | **0.08** | **0.38** | 0.60 |

Drawing crossings as often as head-ons in stage 3 did not teach the side. Run 12
seed 0 came out port-biased (42 deg the compliant way for a port-side target,
7.8 deg for a starboard-side one while swinging 40 deg the wrong way); seed 1
came out starboard-biased. Six policies across four runs, each picking one
direction and applying it to both sides. With the run 11 probe already showing
that the policy **reads** the compliant sense, neither observability (A31 option
1, falsified in F91) nor exposure (option 2, falsified here) is the cause. The
remaining candidate is that the reward does not pay enough for the distinction
to survive the variance of the progress and path-following terms: **A32**.

*The seed-0 speed jump was seed, not F91.* Seed 0's crossing speed at CPA was
0.81 m/s (0.93 against starboard-side targets); seed 1's is 0.63, inside the
run 8-11 range of 0.38-0.66. The run-12 change is not what raised it.

*Stand-on regressed on seed 1* -- 16 of 20 goals with 3 target collisions
against run 11's 19/1 and 18/1, at a lower maximum speed (0.60 vs 0.70-0.72).
A29's growing `v_hold` is holding the speed down, but holding course into an
overtaker that does not give way is now costing contacts: folded into A30.

*Verdict.* Do not freeze. F91 stays (it removes a real wrong pressure and halves
the narrow-channel collisions) but needs its complement, and the crossing side
bias is now the blocking formulation gap. Both are A32.

**F93 -- baseline-v1: the run 11 formulation, saved for the multi-learner
campaign (your call to prepare it, 2026-09-22).**

*Why run 11.* Development set, supervisor off, final checkpoint:

| run | all | head-on | crossing | being overtaken | overtaking | no target |
|---|---|---|---|---|---|---|
| run 8 | 0.85 | 0.85 | 0.55 | 0.90 | 0.95 | 0.95 |
| run 10 | 0.88 | 0.90 | 0.65 | 0.75 | 1.00 | 1.00 |
| **run 11 s0** | **0.92** | **1.00** | **0.75** | **0.95** | 0.95 | 0.95 |
| **run 11 s1** | **0.88** | **1.00** | 0.55 | 0.90 | 0.90 | 0.95 |
| run 12 s0 | 0.83 | 0.90 | 0.40 | 0.90 | 0.90 | 0.95 |
| run 12 s1 | 0.88 | 0.95 | 0.60 | 0.80 | 0.95 | 1.00 |

Run 11 has the best headline (0.90 over two seeds against run 12's 0.855), the
only head-on 1.00 on both seeds, the best being-overtaken rate, and -- what
matters most for a campaign in which five learners each draw their own seeds --
the **same qualitative behaviour on both seeds** (Rule 14 starboard alteration in
open water on 1.00 of head-ons). Run 12's narrow-channel gain is real but its
seed 0 abandoned the starboard alteration in open water (F92), so a learner's
Rule 14 result would depend on its seed. Run 8 is single-seed, alters to
starboard in only 0.22-0.27 of head-ons, and predates A29 and F88. Run 10 lost
the stand-on (0.75).

*What was done.*
1. **F91 and A31 switched off**, their code kept: `V_PORT_HEADING_NEEDS_ADMISSIBLE
   = False` (A33 builds on it) and `A31_STAGE3_WEIGHTS = False` (stage 3's
   `weights` is `None`, so the global class shares apply, as in run 11). No other
   formulation file changed after run 11 launched.
2. **`configs/baseline_v1.json`** records the formulation (every upper-case
   constant, the formulation switches, observation schema, curriculum), run 11's
   run arguments (2 M steps, 10 workers, supervisor off in training and both in
   evaluation, low-speed starts 0.15, 20 per class every 200 k) and each
   learner's hyperparameters and network. Formulation digest `02e853268828a32a`.
3. **`src/baseline_config.py`** -- `--check` (code against the file),
   `--verify-run` (a finished run against it), `--write` (a new version).
   Verified: **run 11 seeds 0 and 1 match baseline-v1**; run 12 differs by
   exactly the two switches; run 8 predates recorded switches.
4. **`train_formulation.py --config`** applies the file's run arguments (only
   `--algo`, `--seed` and `--tag` come from the CLI) and refuses to start if the
   code no longer matches. The CLI defaults differ from run 11's arguments
   (supervisor on in training, 6 per class), so a campaign launched on defaults
   would have silently trained a different formulation; with `--config` it cannot.
   `config.json` records the baseline id and digest.
5. **Fixed:** `config.json` recorded `"algorithm": "PPO"` for every learner; it
   now records the learner's class.
6. **`results/baseline_campaign.sh`** (not launched): check, then each learner x
   seed with `--config`, then Tier 1 off/on. PPO seeds 0 and 1 are run 11 and
   are skipped unless `RERUN_PPO_01=1` (run 11 seed 1 was resumed from 1.75 M
   after a pause, F86 -- identical formulation, a fresh rollout at the resume).
7. Tests `tests/test_baseline_config.py`: code matches the file, run 12's
   switches off, a changed constant is caught, every learner has settings,
   run 11 verifies. **508 tests pass.**
8. **Launch check** (4,096 steps each, `--smoke` under `--config`): PPO,
   RecurrentPPO, TD3, SAC and TQC all start, train, evaluate and save under
   baseline-v1, each recording its own learner class, the baseline id and
   digest, and run 11's arguments; the Tier 1 loader opens all four new model
   types. The smoke runs were deleted.

*Carried into the paper as known limits of baseline-v1:* the crossing opening
direction is not side-conditioned (A32); narrow head-ons where starboard has no
room collide in 0.36 (F91, A33); being overtaken by a vessel that never gives
way (A30). Fixing any of them changes the formulation, which means baseline-v2
and rerunning every learner. Still A26's: the seed count and budget -- the file
records 2 M steps and 1.0 gradient steps per transition, ~135 h for one seed of
all five (F80).

**F94 -- A32 measured: the reward already pays for the compliant crossing turn,
strongly and symmetrically. The side bias is not a reward problem.**

`tools/diagnostics/a32_reward_gap.py` (`results/a32_reward_gap/`): 102 crossings
(20 development + training-namespace draws with the side forced; 57 from port,
45 from starboard; basin, obstacles off, supervisor off) branched at the step
the encounter engages into scripted responses -- a 30 or 60 deg alteration in
the latched compliant sense, the same alteration the other way, or stand-on --
held until the latch clears, then the path resumed. Same seed and follower up to
the branch. The latched sense agreed with A17's side rule in all 102.

| compliant minus wrong, 30 deg, discounted at PPO's gamma from the branch | from port | from starboard |
|---|---|---|
| COLREGs term | +34.2 | +31.9 |
| terminal (goal / collision) | +25.7 | +27.8 |
| path following + progress + boundary | +2.2 | +2.0 |
| **total** | **+61.3** | **+61.6** |
| pairs where both reach the goal: total (COLREGs alone) | +25.3 (+16.6) | +18.3 (+16.6) |
| share of those pairs where the reward prefers compliance | **1.00** (n = 6) | **1.00** (n = 8) |
| gap / across-crossing std of the discounted return | 0.64 | 0.68 |

The 60 deg alteration gives the same picture (+57.9 / +67.4). Complying is also
safer: target collisions 0.46 against 0.70 (from port) and 0.36 against 0.60
(from starboard) at 30 deg.

*What this settles.* The reward prefers the compliant opening in every clean
pair, by the same margin from either side, and the compliant turn costs nothing
on the path terms (it is slightly *cheaper*). So the gap is neither small nor
side-dependent: **A32 option 2 (raise the wrong-sense price) is ruled out** --
it would scale a signal that is already large and symmetric. With observability
(F91) and exposure (F92) also ruled out, what remains is how PPO turns a clear
signal into a side-conditioned action: six policies read the side input (F91's
probe) yet settle on one direction.

*Checked and set aside:* the opening turn made **before** the encounter engages,
which `v_port` does not judge. It happens in only 1-3 of 13-17 turned crossings
per policy (all six runs), always the wrong way, but after-engagement first
turns are still only 0.27-0.67 compliant, so it is a minor contributor.

*Caveats.* Scripted fixed-heading alterations, not the policy's continuous
actions; obstacles off; only 14 (30 deg) and 6 (60 deg) pairs reach the goal
both ways, so the clean-pair margin rests on few pairs, though the sign agrees in
all of them. Magnitudes are before `VecNormalize`; the ratios are unaffected.

*Consequence for baseline-v1.* The formulation prices crossing compliance
correctly, so whether a learner opens the right way is a property of the
**learner**, which is exactly what the five-learner comparison measures. The side
bias is a PPO result to report and compare, not a defect of baseline-v1 that
forces a v2.

**F95 -- A26 decided; the baseline campaign runs on this machine until a cluster
is set up (your calls, 2026-09-22).**

*Decided:* off-policy **1.0** gradient step per transition; **5 seeds** per
learner; each seed represented by its **best-on-dev checkpoint** (`best_model.zip`);
2 M steps; **this machine** for now (~28 days back to back), moving to a cluster
once access is granted. Recorded in `configs/baseline_v1.json`'s campaign block
(formulation digest unchanged, `02e853268828a32a`).

*Built for a month-long unattended run:*
1. **`results/baseline_campaign.sh`** -- 23 jobs (PPO seeds 0-1 are run 11) in
   order (your call): **seed 0 of RecurrentPPO, TD3, SAC and TQC first** (a first
   look at every learner, ~5.5 days), then PPO seeds 2-4 and RecurrentPPO 1-4,
   then the off-policy seeds 1-4 seed by seed, so the campaign can move to a
   cluster at any point with every learner partly done. Re-runnable: finished runs are skipped, a run with a checkpoint
   `--resume`s, a run that died before its first checkpoint is set aside and
   restarted; `runs/CAMPAIGN_STOP` stops it cleanly between runs. Tier 1 (off/on)
   on the best-on-dev model after each run, and on run 11's.
2. **Replay buffers outside the repository (your calls):** ~0.58 GB each at 1 M
   transitions -- past GitHub's 100 MB limit, and C: has 17.5 GB free. They were
   saved at every checkpoint inside the run folder (~8 per run, ~4.6 GB); now
   `ReplayBufferCheckpoint` writes only the latest to
   **`PhD/asv_replay_buffers/<run>/`** in OneDrive, beside the repository (on a
   cluster clone, beside the clone; `ASV_REPLAY_BUFFER_DIR` overrides), skips the
   save below 3 GB free, retries while OneDrive holds a file, and empties the
   folder when the run finishes. (Never under AppData: the Microsoft Store Python
   silently redirects writes there into its package folder -- found in testing.)
   A resume without its buffer refills `learning_starts` steps before updating
   instead of training on an empty buffer. `.gitignore` blocks
   `*_replay_buffer_*.pkl`. **Verified** in the OneDrive folder by killing a SAC
   run after its second checkpoint (the first buffer had been replaced by the
   second) and resuming: restored from `PhD/asv_replay_buffers`, finished, buffer
   deleted, none in the run directory. OneDrive held the emptied folder for a
   moment, so removing it is best-effort.
   *Cost to know:* OneDrive uploads each new buffer -- ~0.58 GB about every 6 h
   during SAC -- and syncs its deletion.
3. Each run's `config.json` records its **platform** (Python, torch, SB3,
   sb3-contrib, numpy, OS, machine) and **git commit**, so runs split between this
   machine and a cluster can be shown to share software.
4. **`tools/campaign_status.py`** -- one line per learner x seed. **510 tests pass.**
5. **`TRAINING_GUIDE.md`** -- how to start, check, hold, pause, resume and move
   the campaign, with troubleshooting.
6. **Fixed:** the baseline check read constants from the live module, so a check
   made after `curriculum.apply_stage` (which rewrites the RPM settings in memory)
   reported a drift the source did not have. It now reads `constants.py` as
   written; test `test_runtime_staging_is_not_a_drift`.

**F96 -- baseline-v2: the Rule 17(b) below-floor draws leave training and the
suite; field work delayed, not deferred; the paper is a formulation plus a
five-learner comparison (your three calls, 2026-09-23, after the Rev 2 review).**

*What changed in the formulation.* `BEING_OVERTAKEN_BELOW_FLOOR_FRAC` 0.20 ->
**0.0**. Every being-overtaken draw now passes at or above the contact-free floor
(A15, A21), in training and in the frozen suite alike, so Rule 17(a)(i) holding
course is always the lawful answer and the policy is never asked to choose
between 17(a)(i) and 17(b). S5 put active release out of scope in Rev 2; A15
later put a labelled 20 % below the floor into the distribution, which is the
drift the review caught. `dcpa_below_floor` stays in the record (always False)
so earlier runs remain readable. **This is the only difference from
baseline-v1** (`baseline_config.diff` reports exactly that one constant).

| | baseline-v1 | **baseline-v2** |
|---|---|---|
| formulation digest | `02e853268828a32a` | **`3d697858e95e5adf`** |
| below-floor being-overtaken draws | 0.20, labelled | **0.0** |
| development set | 120, 20 per class | unchanged, 0 below floor |
| Tier A | 38 named, 35 realised (A-BO-N, A-NU-I, A-NU-N infeasible) | unchanged |
| runs | run 11 (PPO seeds 0-1), RecurrentPPO seed 0 | none yet |

*Cost.* The three finished runs were trained on baseline-v1 and **do not carry
over**: run 11's two PPO seeds and RecurrentPPO seed 0 (~17 h). The campaign is
25 runs again, tagged `bl2`, ordered seed 0 of all five learners first (~6 days),
then seeds 1-4. Taken now rather than after the off-policy runs, the change
costs hours instead of weeks.

*The other two calls.* **Field work is delayed within this paper**, so N3, RQ4
and the "physically reproducible" clause that justified the two-vessel scope all
stand, and the basin sessions (`PART2_BASIN_PLAN.md`) remain on the critical
path. **The paper is a formulation plus a five-learner comparison**, so
`RESULT_TABLES.md`'s "Proposed (SAC, full)" row and the ablation rows need
restating before sign-off, and the classical comparators (C3-C6) become
essential rather than optional -- they carry N2.

*Also from the review, fixed in `planning/METHODS_BRIEF.md`:* the crossing
convention now leads with **Rule 9(b)** and reports the 0.32 stand-on figure as
corroboration rather than as the basis (the earlier ordering let the target
model decide the rule interpretation); **Around the Clock** is named in the
evaluation section (it was always built -- `suite.around_the_clock`, 24 open
water + 24 channel x 10 seeds, table R8 -- and only missing from the brief); the
runtime safety layer is recorded as a **deliberate change to D2**; C-4's 2 x 2
ablation is flagged as needing a **third rung** (no feature / class one-hot /
full context branch) since the observation gained the context branch; and the
0.55 m/s speed caveat moved into the model section, where the sim-to-real claim
lives.

**F97 -- suite 3.1: Tier B channels floored at 7.5 m, Tier B is the default
frozen suite and Tier A the extended set (your calls, 2026-09-23). Comparators
tuned (C3a) and COLREGs-VO built.**

*The suite change.* Tier B sampled channels down to 3.5 m. PPO seed 0's first
frozen-suite run measured what that costs: the 3.5-4.26 m stratum succeeded in
0.43 of episodes and **a third of those failures were wall contacts**, which
reads the geometry rather than the policy. The floor is now 7.5 m -- the 10 m
basin is already narrow water for a 1.7 m vessel -- and the strata become basin,
channel-wide 8.75-10 m and channel-intermediate 7.5-8.75 m: **39 cells x 20 =
780 episodes**. The floor lives in `suite.py` (`TIER_B_MIN_WIDTH_M`), not in
`constants.py`, because the formulation digest covers every constant there and a
campaign was running.

*What it costs, and where the claim moves.* The derived head-on (3.80 m) and
overtaking (4.26 m) thresholds now lie outside the sampled range, and the
crossing threshold (7.60 m) sits just inside the lower stratum, so **Tier B no
longer partitions widths by governing rule**. Claims C-2 and C-3 rest on the
Study 1 width sweep (R4), which keeps widths to 3.5 m, and on Tier A's `A-*-N`
and `A-FAIL-*` cases. The claim ledger's evidence column needs that edit (B6).

*Default and extended.* Every frozen-suite evaluation now runs **Tier B only**;
Tier A runs when asked (`--tiers a,b`). PPO seed 0's suite-3.0 result (Tier B
0.727) is superseded and set aside as `ppos0_bl2_suite30_superseded`; it will be
rerun on 3.1 after the campaign.

*Two defects found and fixed in the runner.* It wrote to `results/tiers/frozen_suite/`
while the campaign's "already evaluated" guard looked at `results/frozen_suite/`,
so the 2,000-episode suite would have rerun after every learner; and the
`stratum` column was blank on every row, because the stratum belongs to the cell
rather than to the built scenario -- the headline could not be split by geometry
at all, which is the one axis C-2 needs. Both corrected; the summary now also
splits by target behaviour (R3).

*Comparators (C3a).* `src/classical/colregs_vo.py` implements **COLREGs-VO
(Kuwata et al., 2014)** -- candidate velocities, velocity-obstacle rejection, and
the give-way constraint that the relative velocity stays to starboard of the
bearing line -- classified by the **open-water** roles, so a crossing target to
port makes the own ship stand on. `encounter_vo` is kept: it follows this
paper's A17 channel convention, and the difference between them is what the
Rule 9 precedence claim measures. A defect found in testing: with only hard
static rejection it grazed obstacles and scored 0.30-0.50 on episodes with **no
encounter at all**; with the soft clearance margin the other comparators use,
0.63 -> 0.86 overall.

*Tuning, and a false positive caught.* Both comparators were fitted on the
development set by coordinate descent over a declared grid, scored by the rule
RL checkpoints are selected by (goal - 2 x collision), then the winners were
re-scored on all 120 development episodes:

| | 60-episode search | full 120 | kept |
|---|---|---|---|
| COLREGs-VO | 0.500 -> 0.550 | 0.525 -> **0.575** | tuned (`W_CHANGE` 0.3 -> 0.8) |
| LOS-PID + DWA | 0.700 -> **0.850** | 0.650 -> 0.650 | **defaults** -- the gain was noise |

Pinned in `configs/comparators_v1.json` (digest `0dd8eab26bbf954f`, 67 values)
with the method, the score rule and both validation results. The DWA episode is
worth a sentence in the paper: it is why a search is validated rather than
trusted.

**F98 -- TD3 dropped from the baseline; the final replay buffer is kept so a run
can be continued (your calls, 2026-09-24).**

*The baseline set is now four learners:* PPO, RecurrentPPO, SAC, TQC, at **3
seeds each** (your call, 2026-09-24; was 5) -- **12 runs**, about 10 days at
current rates. TD3 stays
implemented in `train_formulation.py` and can be run by hand; it is out of the
campaign, `configs/baseline_v2.json`'s campaign block and the paper's
comparison. The campaign is **20 runs**, not 25. The formulation digest is
unchanged (`3d697858e95e5adf`) -- the learner set lives in the campaign block,
not the formulation -- so the config was re-frozen and the running campaign will
still start TQC without a mismatch.

*Replay buffers.* Off-policy runs now keep two files at most while training --
the latest checkpoint's buffer, and (optionally) the best model's -- and
`KEEP_FINAL_BUFFER` now leaves the **final** buffer in place when a run ends, so
training can be continued past the budget. That matters because 2 M may be
short: across six on-policy runs the best checkpoint fell at **2.0 M three
times and 1.8 M twice**, so the budget, not the learner, may be the binding
constraint (A26). SAC seed 0 predates the switch, so a watcher copies its
1.75 M and 2.0 M buffers aside before its own cleanup deletes them.

*Learning curves, for A26.* PPO rises steeply to ~1.2 M (0.53 -> 0.83) then
drifts on a noisy plateau, +-0.04 between evaluations; RecurrentPPO starts high
(0.80 at 200 k), dips, and recovers to 0.86 by 1.6 M. Reporting the maximum over
a noisy plateau slightly favours whichever learner spikes late, so either the
budget rises to 3 M (~5 extra days over the campaign) or the tables report the
mean of the last three evaluations instead of the best.

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
| **10** | **2026-09-23** | **baseline-v2 (F96): below-floor being-overtaken draws removed from training and the suite (S5); Rev 2 review answered — Rule 9(b) leads the crossing convention, Around the Clock named, D2 change recorded, C-4 needs a third rung; field work delayed within this paper; formulation + five-learner framing** |
| **9** | **2026-09-22** | **Formulation work F49–F95 (runs 1–12, basin mode, five learners) recorded in §3; baseline-v1 frozen (F93); A26 decided and the campaign launched (F95); `OBSERVATION_SPEC.md`, `CONSTANTS_AND_SCALES.md` and this header brought up to date; stale planning specs carry dated status notes; `planning/METHODS_BRIEF.md` added** |
