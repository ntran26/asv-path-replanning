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

## 6. Order

1. Test set v2 and its SAC and classical results (running).
2. G1, G2 and G3 (CPU-light; can run while PPO v3 finishes).
3. Build `formulation_v4.py`, `configs/baseline_v4.json` and tests.
4. G4 pilot once PPO v3 ends.
5. Decide the learner, then run the full v4.
