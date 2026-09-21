# OPEN PROBLEMS — what is blocking, and what resolving each one takes

**Companion to `PROJECT_STATE.md`**, which records what is built and the test
count. This file records only what is *not settled*, ordered so the
highest-leverage item is first.

**Last updated:** 2026-09-21, revision 30 — seed replicate: spread 0.05 headline, 0.20 crossing; opening direction is seed-dependent (F90); A31 opened.
Revision 29: 2026-09-20 — run 11 analysed: best yet, crossing opening direction not side-conditioned (F89); A30 on hold.
Revision 28: 2026-09-19 — swerve root cause, run 11 training (F88); A30 opened.
Revision 27: 2026-09-19 — run 10 analysed: best so far, A29 holds stand-on speed, starboard swerve open (F87).
Revision 26: 2026-09-19 — run 9 analysed against run 8 (F84); run 10 = 60/40 + A29, training.
Revision 25: 2026-09-19 — A29 decided and built behind a switch; run 10 queued after run 9 (F85).
Revision 24: 2026-09-19 — A28 options 1 + 3 (run 9 training; stand-on diagnosed, F83); A29 opened.
Revision 23: run 8 analysed (F82); A28 opened.
Revision 22: 2026-09-19 — A27 options 1 + 2 built, run 8 training (F81); run 7 analysed (F79); all five learners timed (F80).
Revision 21: 2026-09-18 — basin mode as default (F74); crossings diagnosed, SAC built, suite 3.0 (F75); TQC replaces SAC-IQN (F77); A26, A27 opened.
Revision 20: run 6 finished: goal 0.83, best compliance, crossings 0.40 (F73).
Revision 19: your option-1 call on A25 (and A24 with it) built (F72).
Revision 18: CODEX reviewed; A25 opened (F71).
Revision 17: run 5 did not improve on run 4; low-speed starts ruled out; A24 opened (F70).
Revision 16: run 4 finished and replayed with the supervisor off and on (F69).
Revision 15: your supervisor-as-runtime-layer suggestions, built (F68).
Revision 14: your option-1 call on A23 (F67).
Revision 13: run 4 launched, C15's stop-test view (F66), A23.
Revision 12: your option-1 call on A22 (F63).
Revision 11: Tier 2 and A22 (F62).
Revision 10: your option-1 call on A21 (F61).
Revision 9: straight paths (F59), the tiered tests (F60), a recommended
resolution for every open item, and A21 from Tier 0.
Revision 8 recorded A20 (F58), revision 7 A18 and A19 (F56), revision 6
A15–A17 (F53), after formulation runs 1–3 (F48–F57).
Revision 4 followed your answers on speed,
corridor widths, the 2 Hz reward, the emergency-stop reward, the tracker, the
generator, the scale audit, noise and randomisation.

| Part | Kind | Needs |
|---|---|---|
| **A** | Decisions | a call from you — 6 items (A31, A30, A26, A10, A3/A4/A6), plus the freeze sign-off |
| **B** | Measurements | basin time — `PART2_BASIN_PLAN.md`, with one addition proposed |
| **C** | Build work | my time — 8 items |

### Open in revision 21 (`PROJECT_STATE.md` F74, F75)

**A26 — training budget and the off-policy update ratio (TODO(04-4)).** PPO is the development vehicle for testing and adjusting the reward now (your call, 2026-09-18). **Baseline learner set: PPO, RecurrentPPO, TD3, SAC and TQC (your correction: TQC, not SAC-IQN), multiple seeds, trained once everything is decided and the suite is frozen.** All five are built in the shared trainer (F76, F77); `sb3-contrib` 2.3.0 supplies RecurrentPPO and TQC. All five go through the A26 budget. Measured on this machine (10 workers, 12 cores; steps/s, and hours per 2 M-step seed), at 1.0 / 0.2 gradient steps per transition for the off-policy learners:

| Learner | steps/s | h per seed |
|---|---|---|
| PPO | ~108 | ~6 |
| RecurrentPPO | 61 | ~9 |
| TD3 | 18 / 53 | ~31 / ~10.5 |
| SAC | 12 / 38 | ~46 / ~15 |
| TQC | 13 / 32 | ~43 / ~17 |

**One seed of all five: ~135 h at 1.0, ~58 h at 0.2. Five seeds, run one at a time: ~28 days at 1.0, ~12 days at 0.2** -- before the ablations (2 x 2, leave-one-out) and the P-clean and randomisation-off contrasts.

| Option | What it means | Cost |
|---|---|---|
| **1. Move the headline runs to GPU/HPC (recommended)** | SAC at 1.0 gradient step per transition, 2 M steps, 5 seeds; the machine here keeps doing development runs | needs access; the code is device-agnostic apart from `device="cpu"` |
| 2. This machine, off-policy at 0.2, 2 M steps | ~12 days for five learners × five seeds, ablations extra | a lower update ratio than SAC is usually run at; state it |

**A27 — port crossings (F75, F78) — decided 2026-09-19: options 1 and 2, built (F81); run 8 trains on them.** Port crossings are solvable (the best scripted response reaches the goal in 0.81 of engaged crossings, on both sides), and A17's port turn beats the starboard turn (0.67 against 0.56). Run 6 turns starboard first in 8 of 12 port crossings anyway. The reward charges a wrong-way *turn* but not a wrong-way *heading*: holding 30 deg to starboard scores exactly like doing nothing.

| Option | Change | Why | Cost |
|---|---|---|---|
| **1. Charge the wrong-way heading (recommended)** | `v_port` also reads the heading displaced the wrong way since engagement, `ρ · clip(max(0, −s_c · Δψ) / Δψ_min, 0, 1)` taken as the max with the yaw-rate form, so a held wrong-way heading keeps costing while the encounter is engaged | closes the measured gap; it is the mirror of the displacement credit `v_r8` already gives | a reward change: reward-scale audit, `test_reward` additions, one PPO run |
| **2. Train port crossings earlier and more (recommended, with 1)** | stage 3 gains crossings from both sides; stages 4-5 weight crossings 60/40 port/starboard | stage 3 teaches only head-on (starboard), so "give way = starboard" is learned first and never unlearned | no formulation change; the development set is unchanged |
| 3. Pay turn plus slowdown | R-2 / `v_r8` credit the combined response more than either alone | the scripted optimum is the A17 turn *with* slowing (0.65-0.67 against 0.37-0.44 turning alone) | touches A24's settled slowing test; hold unless 1 + 2 leave crossings slow |
| 4. Revisit A17 (Rule 15/17 stand-on for a target from port) | hold course and speed, act only by starboard turn or slowing | textbook open-water roles | **not recommended**: against a constant-velocity target that never gives way, holding course solves 0.32 and the Rule 17(c) direction does worse than A17's; the paper should state the narrow-channel convention instead (CODEX flagged the same) |

Recommendation: 1 and 2 together in run 8. **Run 7 confirmed the pattern on the basin geometry (F79): 2 of 12 port crossings, first alteration compliant in 0.22, 0.72 m/s at closest approach.** Both aim at the same learned failure from two sides; this is PPO's debugging phase, so fewer runs outweighs clean attribution. If run 8 fixes it, a one-off run with 2 alone attributes it.

**Freeze sign-off.** `planning/CLAIM_LEDGER.md` and `planning/RESULT_TABLES.md` are drafts for you to sign off; then a commit, so the generator has a SHA and the manifest can be finalised.

**06 deviations to confirm:** basin null traffic keeps `P_nav` not the band; slant cap 14.0 deg (Paper 2's endpoint box) not 18.1; Tier A basin leg 14.0 deg not 15.

**A28 — after run 8 (F82) — decided 2026-09-19: options 1 and 3 (F83). Run 9 tested option 1 and it failed (F84): port crossings fell to 2 of 12 and the starboard swerve stayed, so run 10 returns to 0.60 (your call). Option 2 is the open candidate for the swerve.** Port crossings fixed (6-8 of 12, first alteration compliant 0.70-0.90). Two things left:

| Option | Change | Targets |
|---|---|---|
| **1. Side share back to 0.5, keep the heading term and stage-3 crossings (recommended)** | `CROSSING_PORT_SHARE_TRAINING` 0.60 → 0.50 | the port swerve in starboard crossings (first alteration compliant 0.00-0.14), likely from the 60/40 share; also attributes the fix -- if port crossings hold, option 1 of A27 did the work |
| 2. Charge wrong-way heading from detection, not engagement | `v_port`'s heading form while the target is tracked and classified as give-way, not only while ENGAGED | the swerve starting before engagement |
| **3. Stand-on speed (recommended, with 1)** | investigate why the policy outruns overtakers (being-overtaken max speed 0.80-0.94 m/s, `v_hold` integral 8.7-10.3): the overspeed gate, `v_hold`'s weight 0.45, or progress reward outpacing it -- diagnose first, then change one | the Rule 17(a)(i) stand-on violation |

Recommendation: 1 now as run 9 (one change, and it attributes A27), with a Tier 1-cheap diagnosis of 3 alongside; 2 only if the swerve survives 1.

**A29 — stand-on speeding (F83) — decided 2026-09-19: option 1 as amended, the speed part keeps rising past 0.10 m/s to 3x at 0.30 (F85); run 10, after run 9.** Engaged and being overtaken, run 8 averages 0.69 m/s (above the overspeed tolerance in 54 % of steps) against run 6's 0.48; `v_hold` fires on 65 % of those steps. Speeding keeps the non-yielding overtaker further off (closest range 2.30 m against 1.72 m), and it costs almost nothing extra because `v_hold` saturates at 0.10 m/s of speed change.

| Option | Change | Effect | Cost |
|---|---|---|---|
| **1. Grade `v_hold`'s speed part (recommended)** | the surge deviation rises linearly to full severity at a larger span (e.g. 0.30 m/s) instead of squaring to saturation at 0.10, so fleeing faster costs more than a small correction | removes the "violate once, then free" gap, the same shape as A27's | a reward change: audit and tests; one PPO run |
| 2. Some reactive overtakers in training | a share of being-overtaken targets keep clear under Rule 13 (the Tier B `re` behaviour), so holding course is safe in those and the policy can learn stand-on | trains the behaviour the rule assumes | relaxes D1 (constant-velocity training targets); a formulation change |
| 3. Leave it | fleeing is a Rule 17(b) response when the give-way vessel does not act, and goals hold at 0.90 | none | the stand-on compliance claim weakens; `v_hold` stays high |

Recommendation: 1 after run 9 reports, as run 10, so run 9 stays a single-change run.

**A31 (new) — crossings do not read which side the target comes from (F89, F90).** Run 11's two seeds open crossings differently (seed 0 starboard in every crossing; seed 1 mixed), and neither aligns the first alteration with the side. The compliant sense is in the observation only once the encounter is **latched**; before that the context branch carries 0, and the first alteration is usually made 2-6 s in, often at or just after engagement.

| Option | Change | Why it should help | Cost |
|---|---|---|---|
| **1. Show the sense as soon as the class is known (recommended)** | the context branch carries `compliant_turn_sense(cls, crossing_side)` whenever the classifier has a class, not only while latched; the latch still fixes it at engagement (A20) | the policy can align its opening alteration with the side before the latch, which is when it commits | observation meaning changes (no new values); retrain; the pre-latch sense can flip with classifier noise, which is what A20 exists to stop -- the latched value must still govern once engaged |
| 2. A crossing-only curriculum stage | a stage of crossings alternating sides before the full mixture | forces the distinction early, as stage 3 did for head-on | another stage; longer curriculum |
| 3. Leave it, report by side | none | -- | the paper reports crossing compliance by side, with the opening direction wrong in one of them |

Recommendation: 1, then a two-seed check, since one seed cannot resolve a 0.05 difference (F90).

**A30 — being overtaken by a vessel that never gives way (F87, F88) — on hold (your call, 2026-09-20).**

*The situation.* Under Rule 13 the overtaking vessel keeps clear; under Rule 17(a)(i) the own ship, as stand-on, keeps course and speed. Training overtakers are constant-velocity (D1) at 1.5-2.2x the own ship's speed, so they never keep clear. A15 places 80 % of draws at or above a floor where holding course is safe (hull clearance + D_SAFE), and 20 % below it, labelled, where the overtaker's track reaches the hull and only the own ship's own action (Rule 17(a)(ii)/(b)) avoids contact. The reward's only release is `in_extremis`: DCPA < d_req **and** TCPA < 5 s, which is too late to evade a faster vessel from astern.

*What the runs did* (above floor / below floor, of 17 / 3): run 6 slowed, 17 / 0; runs 7-9 fled, 15-16 / 1-3; run 10, with A29 charging speed change, held cruise, 14 / 1. Fleeing buys the below-floor draws and costs the stand-on rule on every draw; holding keeps the rule and loses the below-floor draws. Run 10 also lost 3 above-floor draws it should win by holding.

| Option | Change | What it teaches | Cost / risk |
|---|---|---|---|
| **1. An earlier Rule 17(a)(ii) release (recommended)** | release `v_hold` (not the other terms) when the perceived overtaker is on a contact course -- its predicted DCPA below A15's contact-free floor -- and TCPA is within the time a stand-on evasion needs (derived from the turning and speed-change response, ~10 s rather than 5); keep charging otherwise | hold when holding is safe; act, without penalty, when the give-way vessel is visibly not keeping clear -- what 17(a)(ii) permits and 17(b) requires | a reward change; the release threshold rests on the perceived DCPA, so tracker error matters (C15); keeps D1 |
| 2. Some overtakers keep clear in training | a share (e.g. 30 %) of being-overtaken targets use `T-RE`, the compliant reactive model, and pass wide | stand-on is learnable where the other vessel co-operates | relaxes **D1**, whose rationale is exactly that a policy trained on co-operating targets learns to rely on them; `T-RE` is a simple starboard-turn placeholder, not a Rule 13 model |
| 3. Train without the below-floor draws | keep them labelled for evaluation (a stress stratum), drop them from training | removes the one case where the rule and survival conflict | the policy never practises 17(b); the evaluation stratum becomes out of distribution |
| 4. Accept and report by stratum | none | -- | the headline being-overtaken rate stays ~0.75-0.85; report above / below floor separately |

Recommendation: 1, with 4's stratum split reported whatever is chosen. Option 1 targets the conflict directly and keeps D1; 2 changes what the policy may assume about other vessels, which the paper would have to defend.

### Under test in revision 15 — the supervisor as a runtime layer (`PROJECT_STATE.md` F68)

| Item | State |
|---|---|
| Train with the supervisor off; R-2 tests the agent's own (coasting) slowdown | **built** (`--train-supervisor off`, `slowdown_clears`) |
| Stop kept as a runtime safety layer | unchanged in the environment and bridge |
| Evaluate off and on; intervention rate as a metric | **built** (`--eval-supervisor both`, Tier 1 `--supervisor`) |
| Paper: 8(e) as two layers — policy slackens (learned), safety layer stops (engineered) | recorded; a writing item |
| 10–20 % of training episodes start slow or at rest | **built** (`--low-speed-start-frac`) |
| Fixed on the way: the supervisor's speed read drew from the shared noise stream | **fixed** — on/off replays are identical when no stop fires |

**Test:** Tier 1 of run 4 off/on, then run 5 from scratch with all of it, against run 4. Adopt if run 5's policy-only (supervisor off) outcomes match or beat run 4's and its intervention rate is low.

**Run 4 baseline (F69):** goal 0.87 / collision 0.13 in training. Tier 1 under current code — supervisor off: crossing goal 0.45, all other classes 0.85–1.00; supervisor on: crossing 0.60, others identical. Intervention rate 0.05 on the development set (crossing 0.20, head-on 0.10), 0.02 on the head-on width set; 8 stops, 0 then hit; 3 outcomes changed, all crossing collisions turned to goals.

**Run 5 (F70):** goal 0.76 / collision 0.24, the same with the supervisor off and on; intervention rate 0.03. Worse than run 4 in every class, most in null (0.65 against 0.85) and crossing (0.40); faster everywhere. **Not adopted as built.**
Low-speed starts are ruled out: run 4 scores 0.85 from cruise and 0.84 from rest (head-on 1.00 → 0.80), and starting from rest makes solved encounters easier, not infeasible.
Lead: R-2's coast test admitted the 8(e) carve-out on 6 % of candidate crossing frames against 38 % for the stop test. See A24.

### Decided in revision 19 — option 1 for A25, and A24 with it (`PROJECT_STATE.md` F72)

| Item | Resolution |
|---|---|
| A25 what to take from CODEX | **port the fixes and the observable latch**: cross-track error scaled by the local half-width; a 14-value `context` branch (latch state, turn sense, change since engagement, admissibility, gates, age) plus the previous action, 56 -> 70; clearing restores the obligation when risk returns; one perceived state per decision; goal-overshoot and collision-kind fixes. Our trainer, development set, curriculum, 1.60 m goal and the A15/A22 labelled cases stay. CODEX's coast-gated Rule 8 credit, 0.60 m goal, excluded hard cases, mastery gate and trainer are not taken. Run 6 trains on it; the CODEX reference controller is the comparator |
| A24 R-2's slowing test | **the braking-path stop** (`R2_SLOWDOWN_TEST = "stop"`), as before F68 |

### A24 (decided in revision 19 — option 1) — which slowing test R-2 reads

| Option | What it means | Evidence |
|---|---|---|
| **1. The braking-path stop (`stop_clears`, A23), as before F68 (recommended if run 6 ≥ run 5)** | R-2 pays a slowdown where stopping would clear. The policy cannot itself stop (no reverse), so the reward credits an intent the safety layer completes; the paper's two-layer framing covers that | run 4 used the stationary form of this test (0.87); run 6 tests it under the F68 formulation |
| 2. The policy's own coast (`slowdown_clears`, F68) | R-2 pays only a slowdown this hull can deliver unaided. Honest, but a coast from cruise rarely clears within a crossing's TCPA, so 8(e) slackening is almost never paid | run 5: 0.76, crossings 0.40, faster everywhere |
| 3. Unlock reverse for the policy (propulsion stage 5) | makes the agent's slowdown a real brake, so option 2 becomes meaningful | untested; changes the action space and Part B's reverse measurements (B1) |

**Test:** run 6 = run 5 + `--r2-slowdown-test stop` — stopped before its first evaluation; folded into A25's next run.

### A25 (new, open) — what to take from CODEX (`PROJECT_STATE.md` F71)

| Option | Contents | Cost |
|---|---|---|
| **1. Port the fixes and the observable latch, one run (recommended)** | cross-track scaled by local half-width; context branch + previous action (56 → 70); clearing-latch fix; synchronised perception; goal-overshoot and metrics fixes. Keep our trainer, development set, curriculum, goal tolerance 1.60 m, A15/A22 labelled cases, and R-2 on the stop test (A24 option 1). Run 7 from scratch; the CODEX reference controller becomes the comparator | ~1 day build and tests, one 8–11 h run; old checkpoints retire |
| 2. Option 1 plus CODEX's scene strata and recovery starts (not its mastery gate) | more static-only and recovery practice, aimed at run 5's obstacle collisions | two changes in one run; harder to attribute |
| 3. Adopt CODEX wholesale | its trainer, gate, 0.60 m goal, excluded hard cases | loses comparability with runs 1–5; the gate would stall at level 4 given every measured crossing rate |
| 4. Fix only the cross-track scaling | smallest change, same observation shape | leaves the reward non-Markov in the observation |

### Decided in revision 14 — option 1 for A23 (`PROJECT_STATE.md` F67)

| Item | Resolution |
|---|---|
| A23 stop test | **along the braking path**: the target's closest approach to the own ship as it brakes under the latch's full astern, then stopped (`stopping.dcpa_over_stop`), against 1.76 m. Tier 1: stops 13 → 7, stop-then-hit 6 → 0. C15's hull-fitted view stays off (with it: 23 stops, 8 then hit) |

### Decided in revision 12 — option 1 for A22 (`PROJECT_STATE.md` F63)

| Item | Resolution |
|---|---|
| A22 crossing feasibility | **a crossing is accepted only if a lawful escape clears it** — coasting to a stop, or a 60° alteration in the A17 compliant sense, after 1.5 s at cruise, in the own ship's physics, hulls apart and own hull inside the corridor; 20 % drawn unescapable and labelled `crossing_escapable = False` |

### Decided in revision 10 — option 1 for A21 (`PROJECT_STATE.md` F61)

| Item | Resolution |
|---|---|
| A21 confined targets | **keep the channel**: overtaking, being-overtaken and null crossing angles drawn within ±10°; the target hull's track must stay inside the corridor to CPA; the clamp nudges instead of teleporting (absorbs C14); being-overtaken floor = hull clearance for the draw + `D_SAFE`, 20 % below, labelled |

### Revision 9 — straight paths, and my recommended resolution for every open item

**Done (your call, `PROJECT_STATE.md` F59):** straight paths only. No bends in
any stage, and a constant Rule 9(a) path offset, so the path stays straight in
varying-width corridors. Tier A drops its two bend cases (34 → 32).

| Item | Recommendation | Why, briefly |
|---|---|---|
| **A22** (new) | **option 1** — feasibility-floor crossings, 20 % labelled below | a quarter of crossings are unavoidable by any lawful response; see A22 |
| A10 vessel model | **keep v3** in simulation; fit v4b's structure (`N_rr` fixed) in basin session 1 | both refits failed the pre-registered rule; changing the plant mid-formulation would confound every run comparison |
| A3 `D_SAFE` | **keep 0.35 m** | satisfies 02a §2's invariant; A18's `ESTOP_CLEAR_DCPA_M` is built on it |
| A4 crossing width threshold | **adopt 04a's 7.60 m** | the generator's width strata already use it, and bends no longer confound width (F59) |
| A6 Study 2 | **confirm 18 conditions**; sweep pose noise around a measured nominal, not 3 cm | 3 cm sits below the tracker's knee, so {0–4}× of it is flat; wait for B4 |
| B5 low-speed manoeuvring | **make the plan edit**: S1-D secondary circles at RPM 6, one S1-E zig-zag pair at RPM 6 | the operating point is outside the identification data |
| B1, B2, B4 | measure in basin session 1 as planned | nothing to decide |
| C2 throughput | profile one stage-5 step next (mine) | threads ruled out (F58); the tiers cover debugging meanwhile |
| C14 clamp teleport | **fold into A21** | it is now load-bearing, not tidy-up |
| C15 tracker course | after A21 (mine) | ~5 % of head-ons still engage as crossings |
| C3–C6, C12 | after the formulation is frozen (mine) | evaluation machinery, not formulation |

### Decided in revision 8 — option 1 for A20 (`PROJECT_STATE.md` F58)

| Item | Resolution |
|---|---|
| A20 engaged encounters | **class, crossing side and turn sense frozen for the life of an engagement** — latched at engagement, reported as `ctx.cls` while engaged or clearing, released only when the encounter clears; `N_SWITCH_STEPS` retired |

Run 4 trains with A18–A20.

### Decided in revision 7 — option 1 for A18 and A19 (`PROJECT_STATE.md` F56)

| Item | Resolution |
|---|---|
| A18 supervisor stop | **only when stopping clears** — the target's DCPA with the own ship stationary must reach `ESTOP_CLEAR_DCPA_M` = 1.76 m; the same test gates R-2's slowdown, and where a slowdown cannot clear the Rule 8 term credits the alteration |
| A19 classification | **against the path tangent** — the own ship's alteration can no longer re-label a head-on as a crossing from port |

Run 3 trained before both; its head-on results are not a test of them.

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
Runs 2 and 3 were also written outside OneDrive; both were moved back into
`runs/` on 2026-09-15.

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

## A20. An engaged encounter is re-classified at close range

**New, from verifying A19 (`PROJECT_STATE.md` F56).** A19 halved the port-sense
regression for the run 2 model (67 % of head-on episodes → 30 %), but a
**non-avoiding path follower** still latches a port sense in 24 %. So the cause
is not the own ship's manoeuvre. On those frames:

| | median |
|---|---|
| range to the target | **3.6 m** |
| path-relative bearing (head-on band ±10°) | **18°** |
| perceived heading-intersection deviation from reciprocal | **29°** |
| true deviation | 8° |

Two things push a truly head-on encounter (62 % of those frames) out of its
class as the vessels close:

* **bearing.** A reciprocal target 1–2 m to one side passes outside the ±10°
  head-on bearing band inside about 8 m;
* **tracker course.** The course estimate degrades at close range — the
  cluster centroid slides along the hull as the aspect changes — and the
  perceived intersection angle is 21° worse than the truth.

Whenever a different class persists for `N_SWITCH_STEPS` (2) while engaged,
the engagement latch re-engages and re-latches the turn sense. COLREGs decides
the situation when risk of collision first develops, not at 3.6 m. Since A17
the re-decision can flip the sense to port.

**Options:**

1. **freeze class and turn sense for the life of an engagement.** The latched
   class becomes `ctx.cls` until the encounter clears, so the observation and
   the reward still read one field (01 §5.3). A new encounter starts only
   after CLEARING → IDLE;
2. freeze only close in: allow class switches while TCPA > `T_EXTREMIS` or
   range > `kappa_rel · d_req`, and freeze inside that;
3. keep the switch rule and fix the inputs instead: widen the head-on bearing
   band with range, and repair the tracker's close-range course (C15). This
   treats the symptoms, and leaves a latch that can still be re-decided by
   any future perception error;
4. keep as is.

**Recommendation:** (1). It is the rule itself — the situation is decided
once, when the obligation arises. It also removes a whole class of
perception-driven flips rather than the two found so far. Its cost is that a
target which *genuinely* changes encounter geometry mid-encounter keeps its
first label. A `T-NC` target in evaluation could do that; D1 training targets
cannot. C15 is worth doing either way. **To resolve:** say which.

**Run 3 (F57), which trained with A15–A17 but before A18–A20:** evaluation
goal 0.79 (run 2: 0.77). Being-overtaken collision fell 0.54 → 0.35 and
starboard-side crossings improved sharply (0.88 → 0.25 at DCPA ≤ 0.7 m). But
head-on rose 0.22 → 0.32, as F55 predicted, and far port-side crossings stay
at 0.83. Both are the geometry this item and A19 act on. Run 4 waits for this
call.

## A19. A17 re-labels head-on encounters as port crossings — run 3 is affected

**New, and caused by my implementation of A17 (`PROJECT_STATE.md` F55).** The
encounter class and the crossing side are computed from the own ship's
*instantaneous* heading. When it turns to starboard for a head-on, the heading
intersection angle leaves the head-on band. So the same encounter is
re-classified as a crossing with the target on the port bow, and A17 then
latches a **port** turn sense. The reward pays the agent to turn back toward
the target. Before A17 this was harmless, because crossing and head-on shared
the starboard sense.

| head-on scenarios, current code (100) | path follower, no avoidance | run 2 model |
|---|---|---|
| episodes that ever latch a port sense | 0.20 | **0.67** |
| engaged pre-CPA frames with port sense | 0.06 | **0.43** |
| first engagement classified head-on | 0.84 | 0.45 |

Run 3 is training with this. Its crossing and being-overtaken splits are
still worth reading; its head-on learning is not.

**Options:**

1. classify the encounter, and take `crossing_side`, against the **path
   tangent** (the course the own ship is keeping) instead of its instantaneous
   heading. The agent's own manoeuvre can then neither change the class nor
   flip the sense. The observation shares the classifier (01 §5.3), so the
   one-hot moves consistently with it;
2. keep the turn sense latched at first engagement across class switches.
   It is simpler, but first engagement is already "crossing" in 52 % of the
   model's head-ons, because it alters before engaging;
3. revert A17.

**Recommendation:** (1), then stop run 3 and restart it with A18 and A19 in.
**To resolve:** say which, and whether to stop run 3.

## A18. The emergency-stop supervisor stops in front of head-on targets

**New, from the C14 diagnosis (`PROJECT_STATE.md` F54).** The supervisor fires
when an engaged give-way encounter is in extremis (DCPA < `d_req` = 2.5 m,
TCPA < 5 s) and the compliant alteration is inadmissible. Admissibility asks
whether the starboard room to the wall, less B/2 and `c_wall`, covers
`Dy_req = d_req − DCPA`. That is a *ship-domain* separation, not hull
clearance. At a 5 m width, with the usual Rule 9(a) station, a starboard turn
is "inadmissible" for any DCPA under about 1.3 m (0.4 m at 7 m, never at
10 m), although about 1.8 m of physical starboard room exists. The own ship
then stops dead ahead of a constant-velocity head-on target. A stop cannot
change a reciprocal target's DCPA, so the target hits it.

| head-on, run 2 model, obstacles off (20 per width) | 5 m | 6 m | 7 m | 8 m | 10 m |
|---|---|---|---|---|---|
| target collision, supervisor on | 0.43 | 0.25 | 0.24 | 0.16 | 0.05 |
| target collision, **supervisor off** | **0.24** | **0.15** | **0.14** | 0.11 | 0.05 |
| starboard alteration inadmissible (pre-CPA frames) | 0.66 | 0.59 | 0.27 | 0 | 0 |

74 % of target collisions had stopped before CPA, against 10 % of
non-collisions. Switching the supervisor off saved 10 collisions and caused 1.
The remaining narrow excess (0.24 against 0.05) is a policy trained *with* the
supervisor, and R-2 still pays the same futile slowdown.

**Options:**

1. the supervisor stops only when stopping clears: the DCPA recomputed with
   the own ship stationary must reach hull clearance plus `D_SAFE`. That is
   true for a crossing target that will pass ahead, and false for a reciprocal
   head-on. Apply the same test to R-2, so a slowdown that cannot clear is not
   the paid 8(e) answer; the agent must take the lateral room it has;
2. measure admissibility against hull clearance instead of `d_req`, for the
   supervisor and R-2. The domain-based `v_*` severities stay as they are;
3. both;
4. keep as is: narrow head-on stays a stop-and-be-hit case against D1
   targets.

**Recommendation:** (1). It asks the physically right question ("does
stopping help?") rather than re-tuning a width, and it leaves the domain
geometry the COLREGs terms are built on untouched. **To resolve:** say which.

## A22. A quarter of generated crossings cannot be avoided

**New, from Tier 2 and a feasibility replay (`PROJECT_STATE.md` F62).** Crossing
is now the dominant failure: evaluation 0.35–0.55 in Tier 2, and flat at
~0.58 in training, on both sides alike. So I replayed the 20 development
crossings under scripted responses taken from t = 0 (hold, half speed, full
astern, a 30° or 60° compliant alteration), with obstacles and the supervisor
off. **The best response per scenario still hits the target in 5 of 20.**

| drawn geometry, medians | TCPA | spawn range | DCPA | speed ratio |
|---|---|---|---|---|
| avoidable | 9.9 s | 8.9 m | 1.26 m | 0.90 |
| unavoidable | **8.8 s** | **6.7 m** | 1.00 m | 0.87 |

The generator draws crossing TCPA and range from 04a's windows, scaled to
0.55 m/s. At the short end, a target on a crossing course reaches the own
ship's track before the own ship can leave it or stop short of it. No reverse
is available at propulsion stage 4 (`RPM_FLOOR` = 0), so "full astern" coasts.

**Options:**

1. **feasibility-floor crossings, as A15 did for being overtaken.** Accept a
   crossing draw only if a scripted escape (the best of stop, and a 60°
   compliant alteration, from engagement) avoids the target hull. Keep a
   labelled 20 % below that floor as the last-moment case;
2. raise the crossing windows instead (e.g. TCPA ≥ 10 s and spawn range
   ≥ 8 m) and leave the rest as drawn. Simpler, but the threshold is a proxy
   and some unavoidable draws would remain;
3. keep as is and report crossing against the feasible subset only. Training
   still pays −300 for draws no policy can win, which is the pressure that made
   the being-overtaken policy abandon its role in run 2.

**Recommendation:** (1). It is the A15/A21 principle applied consistently —
the drawn encounter must be winnable by a lawful response, except in a
labelled fraction — and the check reuses the environment's own physics, so it
stays true when the vessel model changes. Cost: an escape rollout per accepted
draw at generation time, about 10–20 simulated steps × 2 responses. **To
resolve:** say which. Run 4 should wait for it.

## A23. "Stopping clears" assumes the own ship stops where it is

**New, from C15's stop-test view (`PROJECT_STATE.md` F66).** A18 lets the
supervisor stop only when the target's DCPA *with the own ship stationary at
its current position* reaches `ESTOP_CLEAR_DCPA_M` = 1.76 m. With the
hull-fitted geometry that test agrees with the truth twice as often (0.090 →
0.046 disagreement), yet in a Tier 1 replay it **more than doubled stops (13 →
29) and stop-then-hit episodes (6 → 15)**, and raised head-on collision at 6
and 7 m. A stopping vessel keeps moving toward the target's track: it coasts at
propulsion stage 4, and even the astern latch takes 1.1–3.7 s. The centroid's
bias had understated DCPA-if-stopped and masked this.

**Options:**

1. **evaluate the stop where the vessel will actually stop.** Project the own
   ship along its heading by the stopping distance of the latch the supervisor
   would use (the identified hull's braking, as in F28), and require the target
   hull to clear that swept stretch by hull clearance + `D_SAFE`. Then re-enable
   the hull-fitted view (C15), since the test would at last ask the right
   question with the right geometry;
2. keep the current test and raise `ESTOP_CLEAR_DCPA_M` by an empirical margin
   (e.g. to ~2.5 m). Simple, but a proxy for stopping distance that is wrong at
   other speeds;
3. keep as is, with the view off (today). The centroid's bias acts as an
   accidental margin, which is fragile: any perception improvement brings the
   failure back.

**Recommendation:** (1). It keeps A18's principle, "stop only when stopping
helps", and makes it physically true. It also lets the better perception help
instead of hurt. **To resolve:** say which. Run 4 is unaffected either way.

## A21. Confined targets leave their corridor, and A15's floor is centre-to-centre

**New, from Tier 0 (`PROJECT_STATE.md` F60).** With obstacles off, a path
follower that simply holds course — the lawful Rule 17(a) action — collides
with the overtaker in **11 of 12** being-overtaken episodes. That includes
passes drawn at 1.0–1.9 m, above A15's floor. Two causes, both measured:

1. **The generator draws confined targets that leave the corridor, and the
   clamp teleports them.** Being-overtaken crossing angles are drawn from
   ±67.5° (the classifier's band), and containment checks only the spawn
   *point*. The hull breaches the channel on the first step.
   `clamp_to_corridor` then moves the target to the nearest *centreline*
   station and turns it parallel: a median 2.4 m jump, onto the own ship's
   track. Over the development set (path follower, 20 per class), the clamp
   fires **before CPA** in:

   | class | clamped before CPA | median jump |
   |---|---|---|
   | being overtaken | **0.75** | 2.4 m |
   | null | **0.90** | 3.5 m |
   | overtaking | 0.10 | 3.2 m |
   | head-on | 0 | — |

   So most null episodes are not null, and most being-overtaken episodes are
   not the geometry drawn. Runs 2 and 3 trained on this.
2. **The floor ignores hull extent.** A15 floors the *centre* DCPA at 1.0 m.
   Two hulls pass without contact at 0.66 m only when parallel. The
   contact-free centre DCPA (overtaker speed ratio 1.5–2.2) is:

   | crossing angle | 0° | 15° | 30° | 45–67.5° |
   |---|---|---|---|---|
   | contact-free centre DCPA | 0.66 m | 1.2–1.4 m | 1.5–1.7 m | 1.5–1.7 m |

**Options:**

1. **confined targets keep the channel, and the floor is on hull clearance.**
   (a) Draw confined-class crossing angles from a channel-keeping band
   (|CT| ≤ 10° for overtaking and being overtaken; head-on is already ±10°;
   null ±10°), and require the target hull's whole pre-CPA track to stay
   inside the corridor. (b) Replace the teleport with a heading nudge that
   keeps the target's lateral position (C14). (c) Express A15's floor as hull
   clearance — contact-free DCPA at the drawn angle plus `D_SAFE` — keeping
   the labelled 20 % below it;
2. keep the ±67.5° draws and fix only the clamp (b) and the floor (c). Oblique
   overtakers still leave narrow corridors within seconds, so much of the
   class keeps being re-drawn by the clamp;
3. make confined overtaking targets unconfined (no clamp). The geometry drawn
   is then the geometry that happens, but a target leaving through the channel
   wall is not a channel user, which undercuts the Rule 9 framing.

**Recommendation:** (1). Classification bands stay as they are — this narrows
what the *generator* draws for channel users, not what the classifier accepts.
In a straight corridor a vessel keeping the fairway runs near-parallel, so this
is also what F59's simplification implies. **To resolve:** say which. Tier 2
and run 4 should wait for it.

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

**Thread oversubscription ruled out (F58).** 10 workers plus a 10-thread
learner on 10 physical cores looked like contention, but capping the learner
at 4 threads with 1 per worker cut stage-5 training from 120–126 to 106–111
fps, and 8 workers gave 78–82. The rate scales with workers, so the cost is
the environment step itself: 43 steps/s per stage-5 environment against 98 in
stage 1. A 2 M-step curriculum run therefore stays near 6 h until one
stage-5 step is profiled and cut. Until then the cheaper route to reliable
debugging is the tiered protocol, not the thread count.

## C3–C6. Unchanged

Classical comparators; `T-RE`'s velocity obstacle; occlusion and conflict
obstacle placement (04a §3.6 — the generator's obstacles are currently only kept
clear of the CPA); `metrics.py` reading the reward keys.

## C15. The tracker's course estimate at close range

F56: at a median 3.6 m the perceived heading-intersection angle is 21° worse
than the truth, and 62 % of close frames are more than 5° out. A centroid-based
constant-velocity filter sees the cluster centre move along the hull as the
aspect changes. Candidates: a hull-fitted (oriented box) centre and heading,
or inflating the course variance with angular extent. Mine; no decision
needed. It matters for A20's options 2 and 3, and for every class boundary
near CPA.

## C14. Narrow head-on collides — diagnosed, now A18

Run 2's 0.75 rested on only 5 development scenarios, all on bends. A 100-scenario
diagnosis (`PROJECT_STATE.md` F54) found:

* **not the target clamp.** `clamp_to_corridor` does teleport a wall-touching
  target to the centreline. That is not 03a's 9(a) station-keeping either, and
  is worth replacing with a tangent nudge that keeps the lateral offset. But
  disabling it changed collisions by noise, and most snaps come after CPA;
* **not room.** A contact-free pass fits in 90–100 % of draws even at 5 m;
* **not bends.** There is no consistent bend effect once width is held;
* **the supervisor stop**, via domain-based admissibility (A18).

Left for me: the clamp replacement (small, no decision needed).

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
| ~~A22~~ | crossings must be escapable, 20 % labelled | **decided** (option 1) | — | built, F63 |
| ~~C16~~ | class share lost on generator cap-outs | **fixed** (mine) | — | F62 |
| ~~A21~~ | confined targets keep the channel; hull-clearance floor | **decided** (option 1) | — | built, F61 |
| A10 | keep v3 | decision | model provenance | confirm (recommended) |
| A3, A4, A6 | carried | decision | various | one call each |
| **B5** | manoeuvring at 0.55 m/s is extrapolated | measurement | sim-to-real at the operating point | plan edit |
| C2 | throughput | build | campaign size | mine |
| C3–C6 | comparators, occlusion placement, metrics | build | evaluation | mine |
| ~~A20~~ | freeze class and sense per engagement | **decided** (option 1) | — | built, F58 |
| ~~A19~~ | classify against the path tangent | **decided** (option 1) | — | built, F56 (residual → A20) |
| ~~A18~~ | stop only when stopping clears | **decided** (option 1) | — | built, F56 |
| ~~A23~~ | stop test along the braking path | **decided** (option 1) | — | built, F67 |
| C15 | tracker close-range bias — hull fit and stop-test view built, both **off** (F64, F66, F67); the view still over-stops with A23 — next suspect is the fitted heading | build | stop test | mine |
| C14 | narrow head-on — diagnosed (A18); clamp teleport to replace | build | tidy | mine |
