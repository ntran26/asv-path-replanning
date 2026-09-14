# OPEN PROBLEMS — what is blocking, and what resolving each one takes

**Companion to `PROJECT_STATE.md`**, which records what is built and the test
count. This file records only what is *not settled*, ordered so the
highest-leverage item is first.

**Last updated:** 2026-09-14, revision 6 — your option-1 calls on A15, A16 and
A17 (`PROJECT_STATE.md` F53), after formulation runs 1 and 2 (F48–F52).
Revision 4 followed your answers on speed,
corridor widths, the 2 Hz reward, the emergency-stop reward, the tracker, the
generator, the scale audit, noise and randomisation.

| Part | Kind | Needs |
|---|---|---|
| **A** | Decisions | a call from you — 3 items (A10, A3/A4/A6) |
| **B** | Measurements | basin time — `PART2_BASIN_PLAN.md`, with one addition proposed |
| **C** | Build work | my time — 7 items |

### Decided in revision 6 — option 1 for all three (`PROJECT_STATE.md` F53)

| Item | Resolution |
|---|---|
| A15 being-overtaken DCPA | **floored at 1.0 m** for 80 % of draws; 20 % below, labelled `dcpa_below_floor` (Rule 17(b)) |
| A16 speed | **two-sided speed gate kept** — free to 1.2 × `U_REF`, full path penalty from 1.7 × |
| A17 crossing turn sense | **side-dependent** — starboard turn for a crossing from starboard, port turn from port; S3 and the one crossing class unchanged |

Run 3 trains with all three; the A15–A17 sections below are kept as the record
of why.

### Closed since revision 3

| Item | Resolution |
|---|---|
| A1 operating speed | **0.55 m/s**, `CRUISE_RPM = 6` (Fr 0.142); propulsion stages rescaled around it |
| A11 fixed 10 m corridor | **reversed** — the 10–3.5 m virtual corridors, width variation and bends are back |
| A12 reward at 2 Hz | **confirmed** — weights ×5, step counts as durations, discounts converted |
| A8 e-stop reward | **built** — speed gate suspended while the latch holds; one-off `R_ESTOP = −20` |
| A7 free-space tracker | **confirmed** as 03a §6.3's replacement |
| C1 generator in the environment | **built** — `ASVLidarEnv(scenario_stage=…)` |
| C7 scale audit | **run** — `tools/scale_audit.py`, `results/scale_audit.json` |
| B4 pose noise at zero | **nominal noise on** — 3 cm / 0.2° pose, 0.05 m/s / 1 °/s ego |
| hull randomisation | **on at 1.0** |

---

# Part A — decisions only you can make

## A15. Being-overtaken scenarios make standing on a collision

**New, from the scale audit.** The generator draws the overtaker's DCPA
uniformly over 0–2 m (04a §3.4), and in training the overtaker is
constant-velocity (D1): it never gives way. At a DCPA under ~0.65 m the hulls
touch. So in about a third of being-overtaken episodes, holding course and speed
— the lawful Rule 17(a) action — ends in a collision unless the own ship moves
inside the last few seconds.

| policy (scale audit, stage 5) | being-overtaken episodes | collided |
|---|---|---|
| path follower, ignores targets | 27 | **100 %** |
| follower that slows for engaged targets | 33 | 85 % |

It is learnable in principle: `v_hold` is released inside `T_EXTREMIS` = 5 s,
and a 0.65 m sidestep fits that window at 0.55 m/s. But the policy is also
charged −300 for the cases it cannot save, and the cheapest way to avoid that is
to leave the stand-on role early, which `v_hold` exists to discourage.

**Options:**

1. floor the being-overtaken DCPA at a collision-free separation (e.g. ≥ 1.0 m)
   for most draws, and keep a labelled fraction below it as the Rule 17(b) case;
2. keep 04a's distribution and treat early evasion as the expected cost of a
   non-compliant overtaker;
3. make the overtaker keep clear in training — the one class where D1's
   constant velocity is itself non-compliant.

**Run 2 confirms it (F52).** Being-overtaken collision is 0.69 at DCPA ≤ 0.7 m,
0.39 at 0.7–1.4 m and 0.50 above. The policy does not stand on in *any* draw.
It opens the throttle to about 1.0 m/s and swerves 20–50° before CPA, taking
8.8 `v_hold` frames and the F50 over-speed penalty. In 10 m corridors that
ends in the wall, 10 of 20 collisions there. It cannot see which draw is the
fatal one, so it treats every overtaker as one. Evaluation goal rate: 0.60.

**Recommendation:** (1), with about 20 % of draws below the floor. **To resolve:**
say which.

## A16. The formulation run — what to change before the full budget

Run 1 (1.91 M steps; full account in `PROJECT_STATE.md` F48) found three
defects I have fixed, F49–F51, and one that is a formulation call:

**The policy drives at 1.9 × cruise.** It holds full throttle in every class,
because speeding was free in the path and progress terms. At the same time,
the existence cost, the 10 s discount horizon and an earlier goal reward all
paid for it. That breaks more than Rule 6: the generator solves every
encounter for the own ship at 0.56 m/s, so at 1.05 m/s the crossing and
being-overtaken geometry is not the one that was drawn, and those two classes'
collision rates (0.53 and 0.58) cannot be read.

**Options:**

1. a two-sided speed gate in `r_pf`: free to 1.2 × `U_REF`, full path
   penalty from 1.7 × (**F50, built, on by default**);
2. cap the propulsion ceiling at about 1.3 × cruise (stage 4's 12 rpm-units
   becomes about 8). That is simple and hard, but it takes the speed-up out of
   the action space altogether, including in a crossing where Rule 17(b)
   might want it;
3. leave speed free and generate encounters at the realised speed. That
   makes the generator policy-dependent, which 04a exists to avoid.

**Recommendation:** (1), with the 0.2 / 0.5 constants as `TODO(decision)`.
Run 2 (below) uses it. **To resolve:** confirm (1), or pick another.

**Also changed for run 2, my call rather than yours:** 20 development
scenarios per class instead of 6, because 6 moves a class rate in 17-pp steps.
Runs are also written outside OneDrive.

**Run 2 (F52): the gate did its job.** Evaluation mean speed was 0.52–0.66 m/s
against run 1's 0.9–1.0; goal 0.77 at 2.0 M and still rising. The one class
where the agent still runs is being overtaken, and that is A15, not the gate.
**Recommendation unchanged:** (1).

## A17. A crossing from port is scored with a starboard turn sense

**New, from run 2 — the largest formulation defect found so far.** S3
(modification 1) collapses port and starboard crossings into one give-way
class, and the reward gives every crossing the same compliant turn sense,
`+1`, starboard. For a target crossing **from starboard** that is right: turn
to starboard and pass astern. For a target crossing **from port** it is
backwards. Passing astern of it needs a *port* turn or a slowdown, while a
starboard turn carries the own ship along the target's track. `v_port`
charges the port turn, `v_bow` (correctly side-independent) charges crossing
ahead, and nothing charges turning into the target's path.

The policy does what it is paid for:

| crossing, target from port (late evals, n 36) | DCPA ≤ 0.7 m | 0.7–1.4 m | > 1.4 m |
|---|---|---|---|
| collision rate | 0.17 | 0.50 | **0.75** |

Collisions rise with DCPA, which is the signature of an alteration turning a
clear pass into a collision. In replay it turned +27° to +41° to starboard
before CPA in 8 of 9. The starboard-side give-way case also fails at small
DCPA (0.83–0.88) with port turns in 3 of 5 replays. With one class label
covering opposite geometry, the agent may be unable to learn either.

**Options:**

1. keep S3's "own ship gives way either way", but make the turn sense
   side-dependent: `crossing_side == port` → `s_c = −1` (pass astern by
   turning to port or slackening), starboard → `+1`. Nothing else in S3 moves;
   `crossing_side` is already computed for exactly this (`encounter.py`). The
   observation keeps one crossing class; the bearing in the slot features
   already separates the sides;
2. as (1) with `s_c = 0` for the port case: no turn sense, only `v_bow` and
   `v_r8`, leaving how to pass astern to the agent;
3. revert the port case to stand-on (Rules 15/17), with `v_hold`. That
   inherits A15's problem with a constant-velocity target that never gives way;
4. keep as is.

**Recommendation:** (1). It is the smallest change consistent with S3, it gives
each side one geometrically correct answer, and `v_r8` and R-2 already cover
the slowdown. **To resolve:** say which. Run 3 should wait for this and A15;
together they account for most of the collisions left.

## A10. The vessel model — keep v3

Unchanged from revision 3. Both refits failed the pre-registered rule, so the
simulator keeps v3; v4b's structure (`N_rr` fixed) should go into the basin
session-1 fit. **To resolve:** confirm.

## A3, A4, A6. Carried over

| # | Item | Recommendation |
|---|---|---|
| A3 | `D_SAFE` breaches 02a §2's invariant; set to 0.35 | keep 0.35 |
| A4 | crossing width threshold, 02a 6.80 m vs 04a 7.60 m | adopt 04a's |
| A6 | Study 2 has 18 conditions, not 21 | confirm 18. The nominal pose noise (3 cm) sits below the tracking knee (flat to 0.25 m at 2 Hz), so a {0, 0.5, 1, 2, 4}× sweep of it is insensitive — sweep around a larger nominal |

---

# Part B — measurements

## B5. The operating point is half the identification speed — proposed addition

**New.** The hull was identified at 12 rpm-units (1.1 m/s); the campaign now runs
at 6 (0.55 m/s). Rudder force scales with inflow speed squared, so manoeuvring at
0.55 m/s is the model **extrapolated**: the July logs contain no running at that
speed. `PART2_BASIN_PLAN.md` S1-D runs turning circles at RPM 12 and 18 and S1-E
zig-zags at RPM 12. **Proposed:** move S1-D's secondary circles from RPM 18 to
RPM 6, and run one zig-zag pair at RPM 6. Hull randomisation (1.0) covers some of
this uncertainty; it cannot cover a systematic low-speed bias.

## Carried

| # | Measurement | Where |
|---|---|---|
| B1 | reverse efficiency, thrust delay, astern overshoot — the stop now needs only ≥ 0.05 efficiency at 0.55 m/s | S1-C2, P-8 |
| B2 | speed-estimate noise floor for `ESTOP_STOP_SPEED` | S1-A |
| B4 | real pose and ego noise, replacing the nominal values | S1-A |
| — | servo rate (the limiter is off), thrust curve, turning circles | S1-B, S1-C, S1-D |

---

# Part C — build work

## C2. Throughput

Run 1, 10 `SubprocVecEnv` workers: **160 steps/s in stage 1, falling to 96 in
stage 5** — 95 on average, **5.6 h for 2 M steps**. The fall tracks the
curriculum (a target, the tracker and obstacles all arrive by stage 5), but the
run also saved 250 k-step checkpoints into a OneDrive folder being synced,
so how much of it is the environment is not separated. A development
evaluation at 6 per class costs about 30 s; at 20 per class, about 100 s.

**Next:** run 2 is written outside OneDrive, which separates the two; then a
per-stage single-environment timing. At 5.6 h per 2 M-step seed, a 5-seed
full budget is a day of wall clock per configuration.

**Run 2 separated the two:** outside OneDrive the rate still fell from 201 to
90 steps/s, so the fall is stage 5's environment cost (target, tracker,
obstacles). The run took 6.2 h, of which about 24 min was ten 120-episode
evaluations. Next is profiling one stage-5 step.

## C3–C6. Unchanged

Classical comparators; `T-RE`'s velocity obstacle; occlusion and conflict
obstacle placement (04a §3.6 — the generator's obstacles are currently only kept
clear of the CPA); `metrics.py` reading the reward keys.

## C14. Narrow head-on collides 75 %

At width ≤ 7 m head-on collides 0.75 (n 20) against 0.05 (n 60) above, although
the head-on width threshold is 3.8 m. Not yet diagnosed: replay the narrow
cases, and check whether the confined target's Rule 9(a) station leaves the
passing gap 02a §2 assumes. Mine; it may become a decision.

## C12. T8 at its own criterion; T3, T4, T10

T8's 30,000-frame run and T3's 1,000 episodes are pre-freeze scripts. T4
(occlusion) waits on C5; T10 (timeout rate on stage 5) needs a trained policy —
the formulation run is the first candidate.

## C13. Deployment — deferred

Per your call: the bridge's Paper 3 observation adapter (tracker, boundary scan,
encounter contexts), the shadow run and P-8 wait until the simulation is
settled.

---

# Quick reference

| # | Problem | Kind | Blocks | Cost |
|---|---|---|---|---|
| ~~A15~~ | being-overtaken DCPA floor | **decided** (option 1) | — | built, F53 |
| ~~A16~~ | two-sided speed gate | **decided** (option 1) | — | built, F50 |
| ~~A17~~ | side-dependent crossing turn sense | **decided** (option 1) | — | built, F53 |
| A10 | keep v3 | decision | model provenance | confirm |
| A3, A4, A6 | carried | decision | various | one call each |
| **B5** | manoeuvring at 0.55 m/s is extrapolated | measurement | sim-to-real at the operating point | plan edit |
| C2 | throughput | build | campaign size | mine |
| C3–C6 | comparators, occlusion placement, metrics | build | evaluation | mine |
| C14 | narrow head-on collisions | build | head-on in Study 1's narrow widths | mine |
