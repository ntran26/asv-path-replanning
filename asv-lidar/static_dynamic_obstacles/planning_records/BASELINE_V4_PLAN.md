# Baseline-v4 plan: above 90 % on the test set (2026-10-02)

**Status:** plan with validation gates. **Nothing trains until gates G1–G4 pass.**

Goal (decision, 2026-10-02): a policy above **0.90 success on test set v2**. Test set v2 is the
frozen suite and the Paper 2 set merged and trimmed to 1,000 episodes, with the 15 impossible L1
being-overtaken starts replaced (`tools/tiers/test_set.py`).

The plan has two parts: option 2, the static-obstacle conflicts, and option 3, crossings.

## 1. Where the kept SAC 3 M policy stands

On test set v1, SAC scores **0.857**: 143 failures. Of these, 15 were the impossible L1-BO
episodes, now replaced in v2. On the 985 valid episodes the score is about 0.870. Reaching 0.90
on v2 needs roughly **+30 successes**.

| Failure group | Failures (v1) | Mechanism |
|---|---|---|
| Crossings, frozen (basin + channel) | 28 (21 target) | Target collisions early, about 9 s in: a late or small manoeuvre, less Rule 8 activity, slower |
| Crossings, field (L1–L3) | 48 (25 target, 21 obstacle, 2 boundary) | The same, plus obstacles boxing in the avoidance room |
| **L2 head-on** | **15 / 15** | The starboard turn is blocked by the gate panel. The policy swings to port and hits a panel within about 23 steps. The right move is to slow down and pass through the gate |
| L1 overtaking | 7 (6 obstacle) | Overtaking past the on-path (5, 8) panel |
| Other frozen (head-on, BO, null, …) | about 30 | Scattered |

**Why the policy has not learned the conflicts:**
- **Too little exposure.** A near-L2 head-on is about 0.9 % of stage-7 episodes: 45 % coupled ×
  20 % HO × 30 % near-deployment × about 1/3 L2. Coupled layouts only start at 1.5 M.
- **The reward favours compliance-shaped motion in a conflict.** COLREGs penalties arrive every
  step, up to about 5 per step, while the collision comes later and is discounted (about −180).
  Without fix 1, nothing tells the policy the compliant turn is blocked, so its encounter context
  keeps asking for it.
- **It rarely slows down.** Cruise is about 0.70 m/s, against 0.52 for COLREGs-VO.

**What fix 1 showed.** As a 0.5 M fine-tune, fix 1 (blocked turns flagged, so R-2 and the Rule 8
credit pay for slowing) did not lift L2 HO, and it cost 5.7 points on the frozen suite. A
plausible reason is that it changed the meaning of the encounter context half-way through
training. That is the reason to test it **from the start of the static-obstacle stages**, not as a
late fine-tune.

## 2. What baseline-v4 changes

baseline-v4 is a new overlay, `src/formulation_v4.py`, built like v3's overlay with its own digest
and `configs/baseline_v4.json`. `constants.py` is untouched, and v2 and v3 stay reproducible.

| | Change | Targets | Why |
|---|---|---|---|
| A | **Fix 1 from step 0** (`ADMISSIBILITY_STATIC` on throughout) | L2 HO, coupled crossings | Blocked turns are flagged consistently, and slowing is paid when the turn is blocked |
| B | **Conflict curriculum earlier and denser** | L2 HO, L1 OT, coupled crossings | Stage 5 (1.0 M): 20 % coupled (CPA guard 0.2). Stage 6 (1.5 M): 40 % (guard 0). Stage 7 (2.0 M): 55 %. Coupled-layout weights HO .25, CRS .20, CRP .20, OT .10, BO .10, NT .15. Near-deployment share 0.5 |
| C | **Crossing emphasis** in the generator stages | Frozen crossings | Crossing weight raised in stages 5–7 (as A31 does for stage 3) |
| D | **Start-clear rule for basin BO** | Data validity | Training draws whose own hull starts within 0.3 m of a panel are rejected. Near-L1 BO gets a start past the panel, as in test set v2 |
| E | *(only if G1 fails)* Reward: no wrong-way penalty while the compliant turn is flagged blocked | L2 HO | Only if the scripted-behaviour check shows the reward still ranks "turn into the panel" above "slow and pass" |

Unchanged: the reward (unless E is triggered), the observation layout, the learners, the 3 M budget
and the validation set.

## 3. Validation gates (cheap) before any long run

All gates use the **development side** only: the validation sets and near-deployment layouts from
training seeds. Test set v2, the frozen suite and the Paper 2 set are never used.

| Gate | What | Cost | Pass if |
|---|---|---|---|
| **G0** Headroom | Test set v2 for SAC 3 M (running), LOS-DWA and COLREGs-VO (running). Space-time solvability of every failing test-set cell. | Running; no extra cost | Some controller, or the space-time check, solves L2 HO and the coupled crossings. If nothing does, they are near-impossible: drop them from the 90 % arithmetic and report them separately |
| **G1** Reward ordering | On about 40 conflict scenarios (near-L2 HO, near-L1 OT, coupled crossings), roll out scripted behaviours: (a) compliant turn at cruise; (b) port turn; (c) slow down / hold, then pass (Rule 8 / 2(b)); (d) COLREGs-VO. Compute episode return under v3, and under v4 (fix 1 on). | About 1 h CPU | Under v4, the safe behaviour (c or d) has the **highest** return in at least 90 % of scenarios. Under v3 it does not (confirms the diagnosis). Otherwise add E and recheck |
| **G2** Observation | With fix 1: (i) share of conflict states flagged blocked on near-deployment dev scenarios (target over 80 %); (ii) false flags on decoupled dev scenarios the policy now passes. | About 30 min | (i) at least 80 %; (ii) at most 10 %. High false flags explain the fine-tune's frozen loss and need a tighter room test first |
| **G3** Curriculum | Sample 2,000 episodes per stage from the v4 generator: conflict share, crossing share, start-clear rejects, space-time redraw rate, reset time. | About 1 h | The shares match the design, there are no own-start contacts, and resets stay under 1 s (PPO collection is environment-bound) |
| **G4** Pilot | **PPO from its own 1.5 M checkpoint** (start of stage 6, where obstacles first meet the encounter, so fix 1 changes little before it) with v4 stages 6–7 for 0.5 M. The **control** is the running PPO v3 run's own 1.5 → 2.0 M segment: same seed and start, already paid for. Compare both at 2.0 M on the dev conflict subset and the decoupled dev subset. | About 3–5 h, after PPO v3 ends | Conflict subset (near-L2 HO, field HO and crossings) **≥ +15 points** over the control; decoupled dev **≥ −2 points** |

**Crossing diagnosis (part of G1):** trace 20 failing frozen crossings to see when the first
avoidance action comes against TCPA, by side. If the give-way turn starts too late, add a
look-ahead feature for TCPA or an earlier Rule 16 credit (its own small gate), not just more
weighting.

## 4. After the gates: the full run

- **Learner (decided 2026-10-02): develop and test with PPO**, because it is quicker.
  - The G4 pilot and the full v4 confirmation run are both PPO. The comparison is against PPO v3
    (running now), on test set v2.
  - **If PPO v4 solves the conflict cases** (L2 head-on, coupled crossings) with the fix, train **SAC
    v4**, the stronger learner on v3 (frozen 0.935; a 3 M run is about 35 h).
- **Budget:** 3 M steps, from scratch, seed 0.
- **Evaluation:** test set v2, safety off (and the frozen suite and Paper 2 set for continuity).
  Report test-set v2 success against 0.90 with a 95 % interval; at n = 1,000 the interval is
  about ±1.9 points.

## 5. Risks

- **G0 may show L2 HO unsolvable** under the field rules. The 90 % target then has to come from
  crossings, about +20 from 76 failures, which is more than options 2 and 3 alone are likely to
  deliver.
- **Fix 1's frozen-suite loss** may be intrinsic: false "blocked" flags making the policy slow
  down where it should turn. G2 measures this before any run.
- **Denser coupled curriculum** may cost frozen-suite performance (overfitting to coupled
  layouts). The G4 decoupled gate guards against it, and near-deployment layouts never equal the
  tested ones.
- **One seed.** A single 3 M run cannot show seed robustness; three seeds are needed for the paper
  claim.

## 5a. Build state (2026-10-02, 23:40)

- `src/formulation_v4.py`: the draft overlay (A, B, C), not frozen.
- `tools/diagnostics/v4_gates/gates.py`: gates G1–G3, running detached (G2, then G1, then G3).
  Results go to `results/v4_gates/`.
- **Discount, noted for G1:** both learners use γ = 0.951, an effective horizon of about 20 steps
  (10 s). A panel collision 20 steps ahead weighs about 0.37 × −300; 40 steps ahead, about 0.13 ×.
  If G1 fails, the horizon is a candidate alongside reward change E: a longer γ, or an earlier
  collision-risk term.
- **Smoke test on one near-L2 head-on.** The compliant turn hit the panel, and slowing and
  COLREGs-VO hit the target. Fix 1 flagged the blocked turn on 11 of 25 engaged steps (0 without
  it). For these scripted behaviours the returns were identical with fix 1 off and on. So in this
  case fix 1 changes the observation, not the reward ranking. G1 checks this across all scenarios.

## 5b. Gate results (2026-10-03, 01:45)

Results are in `results/v4_gates/`.

| Gate | Result | Reading |
|---|---|---|
| **G2** fix-1 flags | **PASS.** Recall 86 % on blocked conflict episodes (18 % without fix 1). Added flags on validation episodes the compliant run passes: 10 %, on n = 10, so a weak estimate | Fix 1 marks blocked COLREGs turns. The false-flag estimate needs more data before a final run |
| **G1** reward ordering | **FAIL at γ 0.951**: best-return behaviour safe in 88 % of 26 script-solvable conflicts with fix 1, 92 % without. **96 % at γ 0.97, 100 % at 0.98 and 0.99** | The reward already prefers obstacle avoidance over the compliant turn; the compliant turn never reaches the goal in head-on or overtaking conflicts. Fix 1 does not change the returns. The failures are late collisions, 61–91 steps after engagement, which γ 0.951 discounts to almost nothing. **The discount, not the reward terms, is the problem.** 14 of 40 conflicts are solved by no scripted or classical behaviour |
| **G3** curriculum | **CHECK.** Coupled-layout shares 0.18 / 0.38 / 0.58 (design 0.20 / 0.40 / 0.55); crossing share up (0.33–0.44 against 0.22–0.26); 1 own-start contact in 200 stage-7 draws (v3 has 1 too); mean reset 2.6–4.5 s back-to-back (v3 stages 6–7: 1.5–2.3 s) | Shares are as designed. Start contacts are rare but real, so **item D (start-clear redraw) is added**: `env._start_is_clear`, stage key `start_clear`, v4 stages only. Back-to-back resets overstate training cost, because the field prefetch thread refills between episodes, but v4 generates more coupled layouts, so PPO collection may be slower than v3's |

**G4 (queued, `results/v4_pilot_run.sh`).** The run starts after PPO v3 finishes training and its
evaluations. Arm **A** is v4 at γ 0.951; arm **B** is v4 at γ 0.98. Each runs 0.5 M steps from PPO
v3's 1.5 M checkpoint. Both are evaluated against that run's 2.0 M checkpoint on the conflict set
(coupled dev HO/CRP/CRS plus the G1 conflicts), the coupled validation set, and the decoupled
validation set (`tools/diagnostics/v4_gates/pilot.py`). Arm B's value function was fitted for
γ 0.951, so it starts at a disadvantage.

## 5c. Queued for after G4: draft 4.1 and the validation-set extension (2026-10-03)

The trigger is test set v3 (`tools/tiers/test_set.py --version 3`). It has more straight survey
lanes and new dense arrangements. SAC 3 M scores 0.853 overall, and is weakest on:

| Group | SAC 3 M |
|---|---|
| Varying-speed targets | 0.61 (constant speed: 0.78) |
| Dense crossing from port (FS-CRP) | 0.45 |
| Dense being overtaken (FS-BO) | 0.64 |

The changes are queued by `results/v4_after_g4.sh`. After `== G4 done` it runs
`tools/diagnostics/v4_gates/revise_v4_after_g4.py`, then gate G3. The G4 pilots read draft 4.0,
so nothing changes before they finish.

| Change | Draft 4.0 | Draft 4.1 |
|---|---|---|
| Straight legs: dense generator draws (≥ 3 panels) | not set | 70 % (`dense_straight_share`) |
| Straight legs: coupled layouts | not set | 70 % (`field_straight_share`) |
| Varying-speed coupled-layout targets, stages 5 / 6 / 7 | 0 / 15 / 30 % | 20 / 50 / 50 % |
| Coupled-layout weights NT / HO / CRP / CRS / OT / BO | .15 / .25 / .20 / .20 / .10 / .10 | .15 / .20 / .175 / .175 / .15 / .15 |
| Validation set | DV3 (150) + decoupled (120) | the same, **plus** `dev_set_v4.extension()` (60) |

- The validation-set extension (`src/dev_set_v4.py`) has 10 episodes each of NT, HO, CRP, CRS, OT
  and BO, with 70 % straight legs and half the targets at varying speed. It uses generator seeds
  520,000+, which no other set uses.
- Code support is in place: opt-in stage keys in `scenario.py`, `field_training.py` and `env.py`,
  and tests in `tests/test_dense_straight.py`. Baseline v2 and v3 are unchanged (`baseline_config`
  check passes).
- **Still to do before a full v4 run:** wire `train_formulation.py` to a `configs/baseline_v4.json`
  that installs `formulation_v4`. Its validation evaluation currently hard-codes
  `formulation_v3.field_development_set`.

**Applied (2026-10-03, 21:33).** `formulation_v4.py` is now `4.1-draft`. G3 on it passed its
checks: no own-start contacts; coupled-layout shares 0.18 / 0.38 / 0.57; straight legs in 55–76 % of
3-panel draws; varying-speed targets in 29–51 % of coupled-layout targets; mean reset 1.6 s.

## 5d. G4 result: both arms fail (2026-10-03, 21:29)

Results are in `results/v4_gates/g4_summary.txt`. Pairs are compared on the same episodes and
seeds.

| | Conflict set (118) | Coupled dev (150) | G1 conflicts (40) | Decoupled dev (120) |
|---|---|---|---|---|
| Control (PPO v3, 2.0 M) | 0.356 | 0.473 | 0.300 | 0.883 |
| A: draft 4.0, γ 0.951 | 0.356 (**+0.0**; 12 gained, 12 lost) | 0.473 | 0.275 | 0.842 (**−4.2**) |
| B: draft 4.0, γ 0.98 | 0.364 (**+0.8**; 13 gained, 12 lost) | 0.507 | 0.350 | 0.842 (**−4.2**) |

How to read it:
- Neither the overlay (fix 1, a denser coupled curriculum, more crossings) nor the longer horizon
  moved the conflict set within 0.5 M steps.
- Both arms lose the same 4.2 points on the decoupled set. That loss belongs to the overlay
  rather than the discount; the fix-1 fine-tune lost 5.7 points the same way.
- G1 found that 14 of its 40 conflicts are solved by no scripted or classical behaviour. If
  conflicts like these are a large share of the conflict set, the +15-point gate cannot be met
  by any learner. The oracle below measures that.

## 5e. Near-impossible episodes and the crossing trace (2026-10-03)

**Decision (2026-10-03):** near-impossible episodes should be left out of the test set and the
validation set, or kept to a very small share. That way they neither confuse the policy nor
mislead readers of the statistics.

**Oracle feasibility (`src/oracle_feasibility.py`).** `feasibility_st` treats the own ship as a
point that can move in any direction at once. It therefore passes encounters that a vessel with
a turning circle can no longer escape.

The oracle works differently. From the episode's true initial state (same seed), it rolls out a
library of manoeuvres in a copy of the environment:

```
follower at cruise until t0  ->  hold heading + dpsi at throttle tau for D s  ->  follower
```

The library covers t0 at 8 values from 0 to 18 s, dpsi at 7 values from −70° to +70°, tau at 4
values from −1 to +1, and D of 4, 8 and 14 s: up to 574 manoeuvres. The oracle has three
properties:
- **Perfect foresight:** the target moves exactly as it will in the episode.
- **The true vessel model,** checked against `env.step` to about 1e-7 m.
- **The policy's own action space,** with no astern.

There is no COLREGs condition, so it asks only whether the episode can be survived at all. It is
a lower bound on solvability, and its outputs are graded rather than a single verdict:
- the number of solving manoeuvres;
- the best minimum clearance;
- `latest_start_s`, the latest start time from which a manoeuvre still works.

**The rule is set on the development side before it is applied to the test set:**
- Run the oracle on the validation sets (`tools/diagnostics/feasibility/oracle_sets.py dev`).
- Check it against outcomes already recorded: any episode the oracle calls unsolvable but a
  controller solved is a miss.
- Fix the threshold, for example "no manoeuvre keeps 0.2 m clearance".
- Rebuild test set v4 and the validation set without these episodes, or with a capped share
  that is reported separately.

**Crossing trace (`tools/diagnostics/crossing/crossing_trace.py`).** This traces SAC 3 M step by
step on the 108 development crossings and joins the oracle's `latest_start_s`. Each failure goes
into one bin: infeasible, engaged too late, wrong way, no action, late, or acted in time but not
enough.

Both run detached via `results/feasibility_run.sh`; results go to `results/feasibility/` and
`results/crossing_trace/`.

**Oracle v1 was too weak, so it was replaced (2026-10-03, 23:45).** v1 returned to the reference
path on a timer and ignored the panels while following it. On the development side, SAC solved
9 of the 12 episodes v1 called unsolvable. Its rows are kept in `oracle_dev_v1_weak.csv`.

v2 changes three things:
- It follows an A* route around the panels.
- It holds each alteration until the target has passed and is clear.
- Its speed-only manoeuvres stay on that route.

Validation on the 34 episodes v1 found hard, plus 16 others: v2 solves all 50. No controller
disproves any of its verdicts. Only 4 episodes lack a 0.2 m margin, and SAC solved all 4.

**What this means.** With perfect foresight, nearly every development episode is physically
solvable. The near-impossible class is therefore not "physically impossible". It is "decided
before the onboard perception can see the target".

v2 also records, for each episode, when the tracker first tracks the target while the own ship
follows the route at cruise. This uses the full `env.step` run, so occlusion by panels counts. The
tracker usually confirms a target about 2 s into the episode. From this, v2 reports:
- `solved_after_track`: is there any solution that starts at or after that time?
- `decision_window_s`: the latest feasible start minus the tracking time.

Example: `DV3-CRS-CV-09` is solvable only by manoeuvres starting at t = 0. Nothing that starts
after tracking at 2 s works. SAC still solved it, but by what it does before it sees anything,
not by reacting.

**Candidate rule (to fix on the development side):** an episode is near-impossible if no
manoeuvre that starts at or after the first track avoids collision with 0.2 m clearance, even
with perfect foresight. Such episodes would be excluded, or capped and reported separately.
Controller outcomes on the same 370 validation episodes (SAC, LOS-DWA and COLREGs-VO, in
`results/feasibility/dev_outcomes.csv`) show how often a controller "solves" these episodes by
its behaviour before detection.

**Development-side results (2026-10-04, 02:10; `results/feasibility/oracle_summary.txt`).** Each
of the 370 episodes falls into one tier:

| Tier | Episodes | Share | SAC 3 M | LOS-DWA | COLREGs-VO |
|---|---|---|---|---|---|
| Solvable after the first track, with 0.2 m margin | 324 | 87.6 % | 0.80 | 0.72 | 0.62 |
| Solvable after the first track, but only under 0.2 m margin | 26 | 7.0 % | 0.73 | 0.23 | 0.15 |
| Unsolvable once tracked: decided before the target can be seen | 13 | 3.5 % | 0.54 | 0.23 | 0.00 |
| Unsolvable with perfect foresight (oracle lower bound) | 7 | 1.9 % | 0.14 | 0.14 | 0.00 |

The hard tiers concentrate in the dense coupled sets, mostly in the crossings. Among the
coupled dev, dv4x and G1 crossings, 20–40 % are tight or decided before tracking. The decoupled
dev_v2 set has almost none.

SAC "solves" 54 % of the episodes decided before tracking, against at most 23 % for the classical
methods. It does so through its behaviour before it sees the target, not through any reaction. Excluding these
episodes therefore removes cases SAC tends to win: it does not favour the policy, and it makes
the success rate measure reaction rather than luck. The oracle misses at least one solution: SAC
solved one of the 7 "unsolvable" episodes, and LOS-DWA another.

**Crossing trace results (2026-10-04; `results/crossing_trace/summary.txt`).** SAC 3 M solves
70 of the 108 development crossings (0.65; port 0.64, starboard 0.65). Each failure is binned
on its response after the target is first tracked. Turns are measured from the heading held at
that moment, because every episode opens with a heading transient of about 10° while the own
ship settles on the path, in successes and failures alike.

| Failure bin | Episodes |
|---|---|
| Wrong way: the first 10° turn after tracking is against the compliant side | 17 |
| Insufficient: acted in time and on the compliant side, but too little | 10 |
| Decided before tracking | 6 |
| Late: first action after the oracle's latest feasible start | 3 |
| Detected late, or no action | 1 each |

What the timing and response data show:
- **The window is short but usually open.** The target is tracked at a median of about 2 s. The
  latest feasible start is a median of 4–6 s, which leaves a 2–4 s decision window. SAC's first
  action comes about 2 s before that limit.
- **The failures are failures of choice and magnitude, not of timing.** In 27 of 38, SAC turns
  the wrong way first or turns too little.
- **Starboard-crossing failures turn little.** They make a median of 6° of compliant alteration
  in the 4 s after tracking, against 17° in successes.
- **Failures slow down more.** 11 of 38 drop below 0.4 m/s within 4 s of tracking, against 6 of
  70 successes. Slowing during a crossing keeps the own ship on the target's track.
- **Speed changes rarely explain these failures.** Only 1 of the 6 varying-speed failures saw
  the speed change before SAC acted.
- **Fix 1 had no part in these results.** The blocked-turn flag is off in the baseline-v3 policy,
  so its share is 0 throughout.

**Test set v3 tiers (2026-10-04, 04:32; `results/feasibility/oracle_test.csv`).** These counts are
policy-independent. Five rows lost on the first pass, through appends on OneDrive, were rerun; the
runner now checks for completeness itself.

| Tier | Episodes | Where |
|---|---|---|
| A: solvable after the first track, with 0.2 m margin | 933 | everywhere |
| B: tight, solvable after the first track but only under 0.2 m margin | 35 | 15 Paper 2 crossings, 7 Paper 2 overtaking, 5 frozen crossings, 8 other |
| C: decided before tracking | 25 | 12 Paper 2 crossings (FS 6, L2 3, L3 2, L1 1), 6 frozen crossings, 4 Paper 2 head-on, 3 other |
| D: unsolvable with perfect foresight | 7 | 3 Paper 2 being overtaken (FS), 4 scattered |

- **The hard tiers sit in the Paper 2 crossings.** Of those 101 episodes, 13 % are in C or D and a
  further 15 % in B.
- **L2 head-on is not one of them.** All 5 of its episodes are in tier A, so SAC's failures there
  are policy failures, not infeasibility.

**Decision (2026-10-04): tiers B, C and D leave the test set and the validation set, replaced
so the totals and the cell balance stay.**

- **Test set v4** (`tools/tiers/test_set.py --version 4`). It is v3 with the 67 failing episodes
  replaced. Each replacement is a fresh draw of the same cell, target behaviour or speed profile,
  and leg type, from new seed blocks (`V4_*`). A draw is accepted only if it passes the same
  oracle test (`results/test_set/v4/replacement_candidates.csv`). The default `--version` stays 3
  until v4 is adopted.
- **Validation set v4.2** (`formulation_v4.development_set`, `field_development_set`). The
  decoupled 120, coupled validation set v3 (150) and the 4.1 extension (60) are graded with the
  training evaluation's own seeds (`oracle_sets.py deveval`), because the tracking time depends on
  the seed. Each failing episode is replaced in place, so it keeps its position and seed, by
  the next draw of its own generator that passes. The accepted recipes are in
  `configs/dev_set_v4_replacements.json`, written by
  `tools/diagnostics/feasibility/dev_replacements.py`.
- **Not filtered:** the G1 conflict set stays as recorded gate evidence.
- **Training draws are not filtered either.** The oracle costs 10–270 s an episode, against a mean
  reset of 1.6 s. A validated layout pool built offline is the option if training needs the same
  rule.
- **Queue (detached):** the v4 build, then SAC 3 M on v4 (`results/test_v4_sac.sh`; the 933 shared
  episodes reuse the v3 rows), the validation-set oracle (`results/feasibility_deveval.sh`) and
  the replacements (`results/dev_v4_replace.sh`).
- **Still to do before a v4 training run:** `train_formulation.py` must install `formulation_v4`
  and evaluate on `formulation_v4.development_set` and `field_development_set`.

**Built and evaluated (2026-10-04).**
- **Test set v4.** Digest `a8c40beb234ea7f9`, built in 28 min. 146 candidates were graded over
  2 rounds and 109 passed.
- **Validation set.** Graded with the evaluation seeds, 37 of the 330 episodes fail. All 37 are
  replaced (`configs/dev_set_v4_replacements.json`).

SAC 3 M on test set v4, safety off (`results/test_set/v4/sacs0_bl3/`). The 933 shared episodes
reuse the v3 rows (same id, digest and seed).

| | Test set v3 | Test set v4 |
|---|---|---|
| Overall | 0.853 (95 % CI 0.831–0.875) | **0.865** (0.844–0.886) |
| Frozen suite | 0.922 | 0.931 |
| Coupled (L1–L3 and FS cells) | 0.717 | 0.735 |
| Coupled crossing from port (FS-CRP) | 0.448 | 0.345 |
| L1-CRP / L2-HO / L2-CRS | 0.27 / 0.00 / 0.33 | 0.18 / 0.00 / 0.33 |
| Varying-speed Paper 2 targets | 0.613 | 0.620 |

- **The 67 swapped episodes.** SAC solved 35 of the 67 removed episodes (52 %), including 12 of
  the 25 decided before tracking. It solves 47 of the 67 replacements (70 %).
- **Every remaining failure is a policy failure.** Each v4 episode has a solution that starts
  after the first track and keeps 0.2 m. The 135 failures split into 70 obstacle, 53 target and
  12 boundary contacts.
- **The FS-CRP drop is real.** SAC had solved 7 of the 11 removed FS-CRP episodes, mostly
  through its behaviour before detection, but solves 4 of the 11 fair replacements. So the
  weakness in crossings from port belongs to the policy, not to the test set.
- **0.90 needs about 35 more successes.**

**Reward check (2026-10-04; `tools/diagnostics/crossing/reward_check.py`, `results/crossing_trace/reward_check_summary.txt`).**
The question was whether the reward prefers SAC's failing crossing response. Method:
- In each of the 105 development crossings where the target is tracked, SAC is replayed to the
  first track and the environment is copied there.
- From that state, SAC's own continuation is compared with up to 16 oracle manoeuvres that
  succeed from the same state (starting 0–3 s later). Every run goes through the full
  `env.step`, so it collects exactly the reward the policy would.

Results:

| | SAC failures (28 with a successful alternative) | SAC successes (65), control |
|---|---|---|
| Reward prefers SAC's own continuation, γ 0.951 / 0.98 / 1 | **0 / 0 / 0** | 14 / 26 / 21 |
| Median return from the first track (γ 0.951): SAC vs best alternative | −161 vs −28 | −23 vs −19 |

Per-term difference for the failures, SAC minus the best alternative at γ 0.951 (means):

| Term | Difference |
|---|---|
| Terminal (collision) | −124 |
| Progress | −8 |
| Smoothness | −1 |
| Path following | +5 |
| Obstacle proximity | +7 |
| COLREGs | +6 |
| Existence | +2 |

What the check shows:
- **The ordering is right.** In every failure, the reward ranks a successful manoeuvre far above
  what SAC did. The collision term decides it even at the 10 s horizon.
- **The dense shaping leans the other way, but only slightly.** Path following, obstacle
  proximity and COLREGs pay SAC's "stay near the path, keep off the panels, small turn" response
  about 10–20 more over the window, because the successful manoeuvres leave the path and pass
  closer to panels. That can slow learning; it never flips the preference.
- **7 of the 35 failures are already lost at the first track.** From SAC's own state at that
  moment, no manoeuvre succeeds: its behaviour before detection decided them.

**Conclusion.** The crossing failures are a learning problem, not a reward-ranking problem. The
policy has not found the action the reward already prefers, in rare states with a 2–4 s window.

Candidate levers, in order of expected effect:
1. **More exposure** to dense crossings (drafts 4.1 and 4.2).
2. **Oracle demonstrations** in the off-policy replay buffer, with their real rewards (learning
   from demonstrations, e.g. DDPGfD). The demonstrations must come from training-namespace
   layouts only.
3. **Optionally, softer path and obstacle shaping while an encounter is engaged.**

**Decisions (2026-10-04).**
- **The collision penalty stays at −300.** Under the reward normalisation, −300 already
  normalises to −9.1, just inside the clip at ±10; −1000 would be clipped to −10. The larger value
  would also inflate the return spread, so the other terms would shrink. In the reward check the
  ordering is already decided by the collision term.
- **No demonstrations in the learner comparison.** Replay-buffer demonstrations reach only the
  off-policy learners. The on-policy routes, a behaviour-cloning warm start or DAPG, are different
  mechanisms, so the four-learner comparison would confound learner and demonstration method.
- **Item G (draft 4.3): hard-state starts, the same for every learner.**
  - **The share.** 15 % of episodes from stage 6 start at the first track of a dense
    training-namespace encounter. In that state, standing on fails and a manoeuvre started 0–2 s
    later still succeeds with 0.2 m margin (`tools/diagnostics/feasibility/harvest_hard_starts.py`,
    `results/hard_starts/pool_v4.pkl`).
  - **The candidates.** 1,200 layouts: crossings 70 %, head-on 15 %, overtaking 8 %, being
    overtaken 7 %. Straight legs 70 %, varying speed 50 %, near-deployment 50 %.
  - **The approach.** The oracle's route follower at cruise, so no learner chooses the states.
  - **The replay.** The hand-back mechanism replays each record, with ±10° and ±20 % jitter on the
    last state. The pool gives practice in these states and never supplies an action.
- **The recommended baseline is v4.3:** v4.1 (exposure, E) plus 4.2 (feasible validation set,
  F) plus 4.3 (hard starts, G). None of 4.1–4.3 has trained yet; G4 tested 4.0 only.

**The hard-state pool (2026-10-04, 13:35).** 505 of 1,200 candidates were kept:

| Encounter | Kept |
|---|---|
| Crossing from port | 148 |
| Crossing from starboard | 160 |
| Head-on | 112 |
| Overtaking | 58 |
| Being overtaken | 27 |

Of the rest, 332 needed no action (standing on succeeds), 342 had no solution with margin from
the start state, and 21 ended before reaching it.

**The v4.3 SAC pair (decision, 2026-10-04: a short SAC test before a full run).**
- **Why SAC, and why short.** PPO is a weak proxy for these cases: on the coupled validation set
  it scores 0.47 at 2 M, against SAC's 0.71. The 2.5 M SAC checkpoint no longer exists, so both
  arms continue SAC baseline-v3 seed 0 from its final 3.0 M model and replay buffer, by +0.5 M.
- **What the two arms share.** Reset seeds (offset 77777), safety off, and selection on
  formulation v4.2's validation sets (`dev_set: "v4"`).
- **The v4.3 arm** (`configs/finetune_v4_pair_v43.json`). v4's stage 7, built as the overlay
  builds it, plus 15 % hard-state starts. Fix 1 stays off.
- **The v3 arm** (`configs/finetune_v4_pair_v3.json`). v3's stage 7: the control.
- **The runner.** `src/finetune_field.py` gained the opt-in spec keys `overlay_stage`,
  `start_pool`, `dev_set` and `eval_supervisor`. Both arms passed a 4,096-step smoke test, and the
  resolved stage 7 matches v4.1.
- **The gate, fixed before the runs** (`tools/diagnostics/v4_gates/pair_compare.py`). At the
  selected checkpoints, v4.3 minus v3 must reach coupled set +5 points or more, with the
  decoupled set no worse than −2 points. Test set v4 is reported for both, but not used for the
  decision.
- **Requirement for SAC v4 (decision, 2026-10-04, revised the same day): no regression on
  no-target episodes against the v3 SAC 3 M baseline**, on the same episodes and seeds. An earlier
  "pass every no-target episode" was judged too harsh.

  | No-target set | v3 SAC 3 M, to match or beat | When checked |
  |---|---|---|
  | Validation set 4.2, evaluation seeds (50) | **44 of 50**: decoupled 20/20, coupled validation set v3 14/20, 4.1 extension 10/10 | during development |
  | Test set v4 (30) | **28 of 30**: coupled 25/27, L1–L3 3/3 | once, at the end |
  | Paper 2 set, safety off (L1–L3 × 10 noise seeds) | **30 of 30** | once, at the end |

  - **Null episodes** (a target present but never in conflict) are tracked alongside, against v3's
    71 of 75 on test set v4.
  - **The misses v3 already has.** On the test set they are FS-NT-014 and FS-NT-023, both obstacle
    contacts. On the validation set they are 6 coupled validation set v3 layouts with gates, on-path panels
    or slaloms, mostly on slanted legs. The route follower clears 8 of the 9 failing development
    layouts with 0.34–1.2 m margin, and LOS-DWA and COLREGs-VO each solve 8.
  - **The pair is not gated on this.** `pair_compare.py` reports the no-target counts for both arms
    and the start model. The pair's gate stays as fixed before the runs.
  - **Two baselines.** The pair's start model is the final 3.0 M model, which passes 43 of 50 on the
    validation set. The reference above is the kept best 3 M model behind the v3 results: 44 of
    50. The two differ by one decoupled no-target episode.
  - Diagnosis: `tools/diagnostics/static/nt_trace.py` (`results/static_trace/summary.txt`). SAC 3 M
    was traced on all 80 no-target episodes (validation and test set v4) and the 75 null
    episodes of test set v4, and each run was compared with the oracle's A* route.

**The v4.3 SAC pair: result (2026-10-05, 10:11; `results/v4_gates/pair_summary.txt`).** Both arms
are scored on validation set 4.2 with the same seeds. The 3.0 M start evaluations are identical,
which confirms the pairing.

| | v4.3 arm | v3 arm (control) |
|---|---|---|
| Selected checkpoint | 3.4 M | 3.0 M (the start; never improved on) |
| Coupled set (210) | 0.776 | 0.738 |
| Decoupled set (120) | 0.908 | 0.892 |
| Coupled crossings | 0.653 | 0.611 |
| No-target (50) | 45 | 43 |

- **Gate: FAIL.** The coupled set gains +3.8 points against the +5 required. The decoupled set
  gains +1.7, which passes its −2 limit. Over the 330 episodes, 35 were gained and 25 lost.
- **The final checkpoints are level.** At 3.5 M against 3.5 M, the coupled set is −0.2 points and
  decoupled +6.2. The v4.3 advantage depends on its peak at 3.4 M.
- **Test set v4, reported only and not used for the decision:**

  | Selected models | Overall | Paper 2 part | FS-CRP | Varying speed | No-target |
  |---|---|---|---|---|---|
  | v4.3 arm | **0.886** (±0.020) | 0.798 | 0.62 | 0.713 | 30/30 |
  | v3 arm | 0.859 | 0.720 | 0.31 | 0.633 | 26/30 |

  Paired, the v4.3 arm gains 71 episodes and loses 44. The frozen part is unchanged (0.931 against
  0.929). The kept SAC 3 M scores 0.865.
- **The tension.** By the rule fixed before the runs, v4.3 is not adopted. Adopting it because of
  the test-set numbers would make the test set a selection instrument. Any revision of the
  decision has to rest on development evidence only, for example the same checkpoints scored on
  more development seeds.
- **A counting bug.** The final 3.5 M checkpoint was evaluated twice, so the "final" rows count
  each episode twice (no-target n = 100). Their rates are unaffected.

**Added analysis: re-scoring on new seeds (decision, 2026-10-05; declared before it runs).**
- **Why.** The gate failed narrowly, and development evidence only may revise the decision.
- **Checkpoints.** The same ones the gate compared: the v4.3 arm's selected 3.4 M model and the
  v3 arm's selected model, which is its 3.0 M start. Both arms' final 3.5 M models are scored as
  secondary.
- **Episodes.** The same 330 validation episodes (formulation v4.2) under three new episode-seed
  sets (1,100,000 / 1,200,000 / 1,300,000 + position), 990 episodes per checkpoint, safety off
  (`tools/diagnostics/v4_gates/pair_reseed.py`).
- **The decision uses the new seeds only.** The 3.4 M checkpoint was chosen as the best of six
  evaluations on the original seeds, so those seeds flatter it.
- **The rule is unchanged.** v4.3 minus v3 at the selected checkpoints: coupled set +5 points or
  more and decoupled set −2 or better, reported with a 95 % interval from a bootstrap over
  development scenarios.
- **Outcomes.** If it passes, the full SAC v4.3 3 M run is the next step. If not, v3 stays.

**Result (2026-10-05, 12:15; `results/v4_gates/pair_reseed_summary.txt`). Gate PASS, narrowly.**

| New seeds, 990 episodes per checkpoint | v4.3 selected (3.4 M) | v3 selected (3.0 M) | Difference (95 % CI) |
|---|---|---|---|
| Coupled set | 0.759 | 0.708 | **+5.1** (+0.2 to +10.5) |
| Decoupled set | 0.931 | 0.936 | −0.6 (−3.3 to +2.2) |
| Coupled crossings | 0.648 | 0.606 | +4.2 |
| No-target | 137 of 150 | 129 of 150 | +8 |

- **The margin sits at the threshold.** The coupled-set gain is +5.1 against the +5 required, and its
  interval reaches down to +0.2. v4.3 is better by this protocol, but the size of the gain is
  uncertain. Over the 990 episodes, 111 were gained and 81 lost.
- **The final 3.5 M checkpoints are level.** The coupled-set difference is +0.0 (−4.3 to +4.3). The
  advantage depends on the selected checkpoint, chosen by the rule that applies to both arms.
- **No regression on no-target.**
- **Consequence.** v4.3 is adopted as the formulation for the next full run. A SAC v4.3 run of
  3 M steps from scratch confirms it, after `train_formulation.py` is wired to `formulation_v4`.

**Full run launched (2026-10-05, 13:02).** `configs/baseline_v4.json` (digest 851e5baf06bc088a;
fix 1 off; hard-state starts at 15 % from stage 6; stage switches at the kept SAC v3 absolute
steps) through `results/v4_full_run.sh`: training, then Tier 1, the frozen suite and the Paper 2
set on the best checkpoint, then test set v4 (report only). Run directory
`runs/sac_formulation_seed0_bl4`.
- **Wiring checks.** A 4,096-step smoke run passed all 7 stages with the pool on and finished its
  final evaluation. The trainer's validation set (120 decoupled + 210 coupled) matches the pair's
  re-scoring set at all 330 positions, with the 37 replacements in place.
- **Fix in `dev_set_v4.development_sets`.** Replacement positions are recorded on the
  20-per-class set. A smaller decoupled set (the smoke run's 1 per class) now keeps the coupled
  replacements and skips the decoupled ones, instead of indexing past the end.
- **Acceptance after the run.** No-target non-regression against the kept SAC v3 3 M: development
  at least 44 of 50, test set v4 at least 28 of 30, Paper 2 set 30 of 30. The explainer-artifact
  briefs switch to v4.3 only if it becomes the best formulation.

**Full-run result (2026-10-07, 16:21). v4.3 wins on the validation set and loses on the held-out
sets; it does not replace v3.** Training took 50.1 h. The selected checkpoint is the final 3.0 M
evaluation (combined 0.82, score +0.46).

| SAC seed 0, safety off | v4.3 (3.0 M selected) | v3 (kept 3 M) | Difference |
|---|---|---|---|
| Validation set, 330 (selection) | 0.82 | 0.79 | +3 |
| Test set v4, 1,000 (report only) | 0.833 | 0.865 | **−3.2 (−5.7 to −0.7)** |
| of which decoupled, 664 | 0.864 | 0.931 | **−6.6 (−9.3 to −3.9)** |
| of which coupled, 336 | 0.771 | 0.735 | +3.6 (−1.5 to +8.6) |
| Frozen suite headline, 800 | 0.890 | 0.935 | −4.5 |
| Paper 2 deployment-layout set, 630 | 0.649 | 0.643 | +0.6 |

Test set differences are paired over the same episodes (bootstrap 95 % intervals): 66 episodes
gained, 98 lost.

- **Where it loses.** Decoupled being overtaken (test basin cell 0.79 against 0.97; frozen suite
  0.80 against 0.94) and the null target that needs no action (0.84 against 0.95; suite 0.87
  against 0.97), with more obstacle contacts (suite 5.8 % against 2.5 %). This matches the earlier
  null-failure pattern: answering a target that needs no answer and running out of room.
- **Where it gains.** Coupled crossings from starboard (FS-CRS 0.85 against 0.74) and the
  coupled set overall; on the Paper 2 layout set it is level.
- **No-target acceptance: fails by one episode.** Development 48 of 50 (pass), test set v4 27 of 30
  (needs 28; v3 had 28), Paper 2 set 30 of 30 (pass).
- **Reading.** The development gain did not transfer. Selection weights the coupled set 210 of 330,
  so it favours coupled-set gains, and the decoupled part of the validation set (0.89 for both) did not
  reveal the being-overtaken and null regression the larger held-out sets show.
- **Consequence.** v3 stays the best formulation; the explainer-artifact briefs stay on v3. v4.3 is
  a documented negative result for the paper's development record.
- **Next (decision, 2026-10-07).** The baseline set continues on v3: PPO seed 0 is done, and
  RecurrentPPO seed 0 started 16:59 (`results/rppo_bl3_run.sh`: 2.5 M, extended to 3.0 M, then
  Tier 1, frozen suite, Paper 2 set and test set v4).

**Why the no-target failures happen (2026-10-04).**

Where they fail. The failures are dense coupled layouts, mostly on slanted legs:

| No-target set | Success |
|---|---|
| Development: coupled validation set v3 (20; 16 slanted) | 14 of 20 |
| Test set v4: coupled FS-NT (27; mostly straight) | 25 of 27 |
| Development: decoupled (20) | all |
| Development: 4.1 extension (10; mostly straight) | all |
| Test set v4: Paper 2 layouts L1–L3 (3) | all |

By motif, gate plus on-path panel scores 0.69 on slanted legs and 0.93 on straight ones.

What the failures are not:
- **Not infeasible.** The route follower clears these layouts, and LOS-DWA and COLREGs-VO
  solve 8 of the 9.
- **Not the wrong side.** SAC passes every panel it reaches on the side the A* route takes.
- **Not unseen.** The panel that is hit shows on the LiDAR 2–4 m ahead, 3–6 s before the contact.

What they are: late, wavering and slowed avoidance close to the panels. The no-target failures
against the successes:

| | Failures | Successes |
|---|---|---|
| Rudder sign reversals per step within 1.5 m of a panel | 0.38 | 0.13 |
| Rudder hard over (|command| > 0.95) within 1 m of a panel | 55 % of steps | 10 % |
| Speed within 1 m of a panel | 0.59 m/s | 0.92 m/s |
| Median peak cross-track error | 2.9 m (up to 4 m) | — |

Typical cases:
- **FS-NT-023** swings the rudder from one side to the other in front of an on-path panel and
  hits it.
- **FS-NT-014** holds a grazing course with the rudder near zero until one step before the
  contact.
- **DV3-NT-CV-12** squeezes between a panel and the wall on the side the route avoids.

Slowing near a panel costs the turn rate the manoeuvre then needs. Cutting back toward the path
is not more common in failures (0.25 against 0.42).

The null failures on test set v4 have a different pattern. Three of the four are wall contacts
about 30 s in. Each follows a heading excursion of 20–48° off the path, with full astern throttle
and a reversing rudder, while the non-conflicting target is 2–5 m away: the policy answers a
target that needs no answer and runs out of room at the wall. The fourth cuts back into a panel
on a straight three-panel basin leg.

Likely causes, in order of evidence:
1. **Exposure.** In stages 6–7, slanted dense no-target layouts are about 4–9 % of episodes. v4.1
   lowers the no-target coupled-layout weight (0.20 → 0.15) and straightens 70 % of legs, so it moves away
   from this failure.
2. **Reactive control without a route, which is an interpretation.** The policy steers from
   pooled LiDAR and path errors. In tight motifs, the path term pulling back and the obstacle term
   pushing away would explain the wavering.

Remedies after the pair, cheapest and fairest first:
- **Static hard-state starts.** Pool G gains no-target and static states just before an on-path
  panel or gate, oracle-checked, mostly on slanted legs.
- **More slanted dense no-target layouts in stages 6–7.** The no-target weight goes back up, and
  no-target layouts are exempt from the 70 % straight share.
- **Only if those fall short:** a stronger rudder-reversal (smoothness) cost near panels, or a
  free-side cue in the observation. Both are formulation changes.
- **Launch.** Started 13:47 (`results/v4_pair_run.sh`), arms one at a time, each about 11–13 h.
  0.5 M steps at roughly 12–15 steps/s plus validation evaluations is slower than the earlier
  9 h estimate.

## 6. Order

1. Test set v2 and its SAC and classical results (running).
2. G1, G2 and G3 (CPU-light; can run while PPO v3 finishes).
3. Build `formulation_v4.py`, `configs/baseline_v4.json` and tests.
4. G4 pilot once PPO v3 ends.
5. Decide the learner, then run the full v4.
