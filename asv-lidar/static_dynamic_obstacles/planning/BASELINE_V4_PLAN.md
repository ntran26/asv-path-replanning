# Baseline-v4 plan: above 90 % on the test set (2026-10-02)

**Status:** plan with validation gates. **Nothing trains until gates G1–G4 pass.**

Goal (your call, 2026-10-02): a policy above **0.90 success on test set v2**. Test set v2 is the
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
- **Too little exposure.** A near-L2 head-on is about 0.9 % of stage-7 episodes: 45 % field ×
  20 % HO × 30 % near-deployment × about 1/3 L2. Field layouts only start at 1.5 M.
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
| A | **Fix 1 from step 0** (`ADMISSIBILITY_STATIC` on throughout) | L2 HO, field crossings | Blocked turns are flagged consistently, and slowing is paid when the turn is blocked |
| B | **Conflict curriculum earlier and denser** | L2 HO, L1 OT, field crossings | Stage 5 (1.0 M): 20 % field (CPA guard 0.2). Stage 6 (1.5 M): 40 % (guard 0). Stage 7 (2.0 M): 55 %. Field weights HO .25, CRS .20, CRP .20, OT .10, BO .10, NT .15. Near-deployment share 0.5 |
| C | **Crossing emphasis** in the generator stages | Frozen crossings | Crossing weight raised in stages 5–7 (as A31 does for stage 3) |
| D | **Start-clear rule for basin BO** | Data validity | Training draws whose own hull starts within 0.3 m of a panel are rejected. Near-L1 BO gets a start past the panel, as in test set v2 |
| E | *(only if G1 fails)* Reward: no wrong-way penalty while the compliant turn is flagged blocked | L2 HO | Only if the scripted-behaviour check shows the reward still ranks "turn into the panel" above "slow and pass" |

Unchanged: the reward (unless E is triggered), the observation layout, the learners, the 3 M budget
and the development set.

## 3. Validation gates (cheap) before any long run

All gates use the **development side** only: the dev sets and near-deployment layouts from
training seeds. Test set v2, the frozen suite and the Paper 2 set are never used.

| Gate | What | Cost | Pass if |
|---|---|---|---|
| **G0** Headroom | Test set v2 for SAC 3 M (running), LOS-DWA and COLREGs-VO (running). Space-time solvability of every failing test-set cell. | Running; no extra cost | Some controller, or the space-time check, solves L2 HO and the field crossings. If nothing does, they are near-impossible: drop them from the 90 % arithmetic and report them separately |
| **G1** Reward ordering | On about 40 conflict scenarios (near-L2 HO, near-L1 OT, field crossings), roll out scripted behaviours: (a) compliant turn at cruise; (b) port turn; (c) slow down / hold, then pass (Rule 8 / 2(b)); (d) COLREGs-VO. Compute episode return under v3, and under v4 (fix 1 on). | About 1 h CPU | Under v4, the safe behaviour (c or d) has the **highest** return in at least 90 % of scenarios. Under v3 it does not (confirms the diagnosis). Otherwise add E and recheck |
| **G2** Observation | With fix 1: (i) share of conflict states flagged blocked on near-deployment dev scenarios (target over 80 %); (ii) false flags on frozen-like dev scenarios the policy now passes. | About 30 min | (i) at least 80 %; (ii) at most 10 %. High false flags explain the fine-tune's frozen loss and need a tighter room test first |
| **G3** Curriculum | Sample 2,000 episodes per stage from the v4 generator: conflict share, crossing share, start-clear rejects, space-time redraw rate, reset time. | About 1 h | The shares match the design, there are no own-start contacts, and resets stay under 1 s (PPO collection is environment-bound) |
| **G4** Pilot | **PPO from its own 1.5 M checkpoint** (start of stage 6, where obstacles first meet the encounter, so fix 1 changes little before it) with v4 stages 6–7 for 0.5 M. The **control** is the running PPO v3 run's own 1.5 → 2.0 M segment: same seed and start, already paid for. Compare both at 2.0 M on the dev conflict subset and the frozen-like dev subset. | About 3–5 h, after PPO v3 ends | Conflict subset (near-L2 HO, field HO and crossings) **≥ +15 points** over the control; frozen-like dev **≥ −2 points** |

**Crossing diagnosis (part of G1):** trace 20 failing frozen crossings to see when the first
avoidance action comes against TCPA, by side. If the give-way turn starts too late, add a
look-ahead feature for TCPA or an earlier Rule 16 credit (its own small gate), not just more
weighting.

## 4. After the gates: the full run

- **Learner (decided 2026-10-02, your call): develop and test with PPO**, because it is quicker.
  - The G4 pilot and the full v4 confirmation run are both PPO. The comparison is against PPO v3
    (running now), on test set v2.
  - **If PPO v4 solves the conflict cases** (L2 head-on, field crossings) with the fix, train **SAC
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
- **Denser field curriculum** may cost frozen-suite performance (overfitting to Paper 2-style
  layouts). The G4 frozen-like gate guards against it, and near-deployment layouts never equal the
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
| **G2** fix-1 flags | **PASS.** Recall 86 % on blocked conflict episodes (18 % without fix 1). Added flags on development episodes the compliant run passes: 10 %, on n = 10, so a weak estimate | Fix 1 marks blocked COLREGs turns. The false-flag estimate needs more data before a final run |
| **G1** reward ordering | **FAIL at γ 0.951**: best-return behaviour safe in 88 % of 26 script-solvable conflicts with fix 1, 92 % without. **96 % at γ 0.97, 100 % at 0.98 and 0.99** | The reward already prefers obstacle avoidance over the compliant turn; the compliant turn never reaches the goal in head-on or overtaking conflicts. Fix 1 does not change the returns. The failures are late collisions, 61–91 steps after engagement, which γ 0.951 discounts to almost nothing. **The discount, not the reward terms, is the problem.** 14 of 40 conflicts are solved by no scripted or classical behaviour |
| **G3** curriculum | **CHECK.** Field shares 0.18 / 0.38 / 0.58 (design 0.20 / 0.40 / 0.55); crossing share up (0.33–0.44 against 0.22–0.26); 1 own-start contact in 200 stage-7 draws (v3 has 1 too); mean reset 2.6–4.5 s back-to-back (v3 stages 6–7: 1.5–2.3 s) | Shares are as designed. Start contacts are rare but real, so **item D (start-clear redraw) is added**: `env._start_is_clear`, stage key `start_clear`, v4 stages only. Back-to-back resets overstate training cost, because the field prefetch thread refills between episodes, but v4 generates more field layouts, so PPO collection may be slower than v3's |

**G4 (queued, `results/v4_pilot_run.sh`).** The run starts after PPO v3 finishes training and its
evaluations. Arm **A** is v4 at γ 0.951; arm **B** is v4 at γ 0.98. Each runs 0.5 M steps from PPO
v3's 1.5 M checkpoint. Both are evaluated against that run's 2.0 M checkpoint on the conflict set
(field-dev HO/CRP/CRS plus the G1 conflicts), the field development set, and the frozen-like
development set (`tools/diagnostics/v4_gates/pilot.py`). Arm B's value function was fitted for
γ 0.951, so it starts at a disadvantage.

## 5c. Queued for after G4: draft 4.1 and the development-set extension (2026-10-03)

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
| Straight legs: field layouts | not set | 70 % (`field_straight_share`) |
| Varying-speed field targets, stages 5 / 6 / 7 | 0 / 15 / 30 % | 20 / 50 / 50 % |
| Field weights NT / HO / CRP / CRS / OT / BO | .15 / .25 / .20 / .20 / .10 / .10 | .15 / .20 / .175 / .175 / .15 / .15 |
| Development set | DV3 (150) + frozen-like (120) | the same, **plus** `dev_set_v4.extension()` (60) |

- The development-set extension (`src/dev_set_v4.py`) has 10 episodes each of NT, HO, CRP, CRS, OT
  and BO, with 70 % straight legs and half the targets at varying speed. It uses generator seeds
  520,000+, which no other set uses.
- Code support is in place: opt-in stage keys in `scenario.py`, `field_training.py` and `env.py`,
  and tests in `tests/test_dense_straight.py`. Baseline v2 and v3 are unchanged (`baseline_config`
  check passes).
- **Still to do before a full v4 run:** wire `train_formulation.py` to a `configs/baseline_v4.json`
  that installs `formulation_v4`. Its development evaluation currently hard-codes
  `formulation_v3.field_development_set`.

**Applied (2026-10-03, 21:33).** `formulation_v4.py` is now `4.1-draft`. G3 on it passed its
checks: no own-start contacts; field shares 0.18 / 0.38 / 0.57; straight legs in 55–76 % of
3-panel draws; varying-speed targets in 29–51 % of field targets; mean reset 1.6 s.

## 5d. G4 result: both arms fail (2026-10-03, 21:29)

Results are in `results/v4_gates/g4_summary.txt`. Pairs are compared on the same episodes and
seeds.

| | Conflict set (118) | Field dev (150) | G1 conflicts (40) | Frozen-like dev (120) |
|---|---|---|---|---|
| Control (PPO v3, 2.0 M) | 0.356 | 0.473 | 0.300 | 0.883 |
| A: draft 4.0, γ 0.951 | 0.356 (**+0.0**; 12 gained, 12 lost) | 0.473 | 0.275 | 0.842 (**−4.2**) |
| B: draft 4.0, γ 0.98 | 0.364 (**+0.8**; 13 gained, 12 lost) | 0.507 | 0.350 | 0.842 (**−4.2**) |

How to read it:
- Neither the overlay (fix 1, a denser field curriculum, more crossings) nor the longer horizon
  moved the conflict set within 0.5 M steps.
- Both arms lose the same 4.2 points on the frozen-like set. That loss belongs to the overlay
  rather than the discount; the fix-1 fine-tune lost 5.7 points the same way.
- G1 found that 14 of its 40 conflicts are solved by no scripted or classical behaviour. If
  conflicts like these are a large share of the conflict set, the +15-point gate cannot be met
  by any learner. The oracle below measures that.

## 5e. Near-impossible episodes and the crossing trace (2026-10-03)

**Your call (2026-10-03):** near-impossible episodes should be left out of the test set and the
development set, or kept to a very small share. That way they neither confuse the policy nor
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
- Run the oracle on the development sets (`tools/diagnostics/feasibility/oracle_sets.py dev`).
- Check it against outcomes we already have: any episode the oracle calls unsolvable but a
  controller solved is a miss.
- Fix the threshold, for example "no manoeuvre keeps 0.2 m clearance".
- Rebuild test set v4 and the development set without these episodes, or with a capped share
  that is reported separately.

**Crossing trace (`tools/diagnostics/crossing/crossing_trace.py`).** This traces SAC 3 M step by
step on the 108 development crossings and joins the oracle's `latest_start_s`. Each failure goes
into one bin: infeasible, engaged too late, wrong way, no action, late, or acted in time but not
enough.

Both run detached via `results/feasibility_run.sh`; results go to `results/feasibility/` and
`results/crossing_trace/`.

## 6. Order

1. Test set v2 and its SAC and classical results (running).
2. G1, G2 and G3 (CPU-light; can run while PPO v3 finishes).
3. Build `formulation_v4.py`, `configs/baseline_v4.json` and tests.
4. G4 pilot once PPO v3 ends.
5. Decide the learner, then run the full v4.
