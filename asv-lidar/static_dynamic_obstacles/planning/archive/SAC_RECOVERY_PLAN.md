# SAC seed 0 (baseline-v3): diagnosis and recovery plan within 3 M steps

**Written 2026-09-30** while the run continues 2.5 M → 3.0 M. Constraint
(decision): no restart from scratch; any improvement starts from an existing
checkpoint and the total stays at 3 M steps.

## 0. Agreed back-up plan (decision, 2026-09-30)

> **Outcome and status (2026-09-30 21:45).** The 3 M policy passed the section 3 bar
> (frozen headline 0.935 vs 0.892; Paper 2 set 0.64 vs 0.45, no-target 30/30,
> fixed-speed target cases 0.63) and is **kept as the best policy**:
> `runs/sac_formulation_seed0_bl3/kept_best_3M/` (best_model.zip,
> best_vecnormalize.pkl, eval_summary.json, config.json). Stage 8 was run anyway
> to see whether it improves further: `configs/finetune_v3_stage8.json`,
> `results/stage8_run.sh`, run folder `runs/sac_formulation_seed0_bl3_stage8`
> (launched 21:42; ~9 h training, then the frozen suite and Paper 2 set). It
> replaces the 3 M policy only if it beats it.

**Trigger:** the 3 M evaluation (Tier 1, frozen suite, Paper 2 set) comes out
"bad" by the criteria in section 3.

**Then:**

1. **Option 0 first** — score the 2.0–3.0 M checkpoints on the three-panel
   development set; if one meets section 3, select it and stop.
2. **Otherwise, a focused stage 8 from the 2.5 M checkpoint to 3.0 M**
   (0.5 M steps, about 9 h training plus 3–4 h evaluation):
   - three-panel layouts in **70 %** of episodes (stage 7: 45 %), weighted
     toward no-target spread layouts and the weakest encounters (crossing from
     starboard, head-on, crossing from port); other settings as stage 7
     (no CPA guard, 30 % varying-speed targets, space-time solvability);
   - **higher exploration** for the first 100 k steps: SAC target entropy −2 →
     −1 (the setting that actually controls exploration);
   - **failure replay:** each worker re-issues a scenario it collided in with
     probability 0.3;
   - reward and observation unchanged.
   Start files: `runs/sac_formulation_seed0_bl3/sac_2500000_steps.zip`,
   `sac_vecnormalize_2500000_steps.pkl`, and the replay buffer kept at
   `PhD/asv_replay_buffers/kept_checkpoints/sac_formulation_seed0_bl3_sac_replay_buffer_2500000_steps.pkl`.
   To build (about half a day incl. tests): a stage-8 overlay in
   `formulation_v3.py`, the env failure-replay pool, and a start-from-2.5 M
   launcher that restores this buffer and applies the entropy setting.
3. Keep Option 3 (earlier obstacle-clearance reward) in reserve; build the
   runtime safety filter (Option 4) for the field regardless.

**Considered and set aside — restarting from an earlier checkpoint (1.0–1.5 M)
with the three-panel stages pulled earlier at 70 %.** The aim (more three-panel
practice) is right and is kept above, but going back in time does not help:

- **Exploration was not higher earlier.** SAC holds its policy entropy at a
  fixed target by auto-tuning; the logged coefficient reached about 0.003 by
  100 k steps and stayed within 0.0023–0.0036 up to 2.6 M.
- **Only the 2.5 M replay buffer survives**; earlier checkpoints would restart
  with an empty buffer.
- **Time:** 2 M steps from 1.0 M ≈ 36 h + evaluation (about two days), against
  about 9 h from 2.5 M.
- **Risk to general skills:** the development score was 0.75 at 1.0 M and
  reached 0.83–0.88 only during 1.2–1.6 M; crossings (0.65) and being
  overtaken (0.70) are still the weakest development classes.
- **The share is not the bottleneck:** the three-panel score rose fastest in
  stage 6 at a 25 % share (0.39 → 0.53 in 300 k steps) and slowed at 45 %.
- An earlier three-panel schedule remains an option for the **other** learners
  and seeds if v3 is adopted, at the cost of SAC seed 0 following a different
  schedule.

> **Stage 8 result (2026-10-01 08:33): no improvement; the 3 M policy stays best.**
> Best stage-8 checkpoint 2.9 M (dev 0.89, three-panel 0.73 — level with 3 M).
> Frozen suite headline 0.905 vs **0.935** (being overtaken 0.87 vs 0.94,
> crossings 0.79 vs 0.82–0.83). Paper 2 set overall 0.66 vs 0.64 (noise), but
> **no-target 25/30 vs 30/30** (L1 5/10 — fails the hard requirement);
> fixed-speed targets 0.64 vs 0.63 (head-on and overtaking up, crossings and
> being overtaken down). The heavier three-panel share traded general skill for
> no net field gain. Kept policy: `runs/sac_formulation_seed0_bl3/kept_best_3M/`.

> **Fix 1 (2026-10-01): static obstacles in the admissibility test.** Diagnosis
> (180 field crossing/head-on replays of the 3 M policy): the compliant-turn
> admissibility test (`colregs/geometry.py: channel_room`) ray-casts only the map
> boundary, so it flagged a blocked turn in 9 of 180 episodes although a static
> obstacle sat on the compliant side ahead in about half; the context therefore
> said "turn", R-2 never credited slowing, and the policy cruised (0.70 m/s vs
> 0.52 for the rule-following VO), slowing in 0-21 % of engaged crossings. L2
> head-on 0/20: a 35° starboard turn into the gate obstacle, never slowing.
> Fix: `geometry.obstacle_room` (perceived static LiDAR returns, the target's own
> removed at 1.8 m, along the passage to the projected CPA) bounds the room;
> switch `ADMISSIBILITY_STATIC` (run-time, off by default, recorded in the run's
> config and applied by the evaluation tools). With it on, blocked turns are
> flagged in 100 % of L2 head-on and 80-90 % of L1 episodes (0-5 % before).
> Fine-tune `configs/finetune_v3_fix1.json` (`results/fix1_run.sh`), launched
> 2026-10-01 14:18: the identical 2.5 → 3.0 M stage-7 continuation with only the
> fix on. Fix 2 (permit the other side when blocked and slowing cannot clear) is
> held in reserve. Test: `tests/test_admissibility_static.py`.

## 1. Where the policy stands (2.5 M, safety layer off)

| Set | Goal | Notes |
|---|---|---|
| Development (120, general) | 0.88 | crossing 0.65, being overtaken 0.70, head-on 0.90, overtaking 1.00, no target 1.00 |
| Three-panel development (150) | 0.68 | no target 0.75, head-on 0.62, crossing port 0.62, starboard 0.58, overtaking 0.81, being overtaken 0.73 |

Trend of the three-panel score: 0.39 (1.4 M) → 0.43 → 0.53 → 0.57 → 0.60 →
0.62 → 0.68 (2.5 M) — still rising, gains shrinking per 200 k steps.

## 2. Diagnosis — is it the formulation?

**How the failures happen (2.4 M and 2.5 M evaluations).** On the three-panel
set, 30 of 48 failures at 2.5 M are **static-obstacle** collisions, 9 boundary
and only 9 target collisions. The same holds for every encounter type
(crossing from starboard: 7 panel vs 4 target). Varying-speed targets are no
harder than constant-speed ones (0.67 vs 0.68). So the weakness is threading
three-panel layouts, with or without traffic, not the rule behavior.

**Test on the 20 no-target three-panel layouts** (2.5 M policy, same seeds as
the evaluation; `scratchpad/nt_diag.py`):

| Candidate cause | Test | Result | Verdict |
|---|---|---|---|
| Perception (a panel tracked as a moving vessel, so it drops out of the LiDAR branch) | frames with a published target in scenes with no target | **0** in all 20 episodes | not the cause |
| Scenario feasibility for this hull (A* and the space-time check ignore turning dynamics) | run LOS-PID + DWA (same perception, same vessel model) on the same layouts | DWA reaches the goal in **18/20**, including **4 of SAC's 5 failures** | layouts are solvable; not the cause (one layout, CV-04, defeats both — hard but not impossible to keep) |
| Time limit | outcomes | no timeouts | not the cause |
| Policy skill in tight static geometry | where the hull touches | 4 of 5 contacts: panel **1.3–1.5 m ahead, 0.4–0.9 m to the side** — the bow or forward shoulder, 11–20 s in (the first panel group); 1 contact at the stern quarter | the cause: late or too-shallow avoidance of a panel close to the path |

**Conclusion.** The observation, reward and dynamics are capable: a classical
controller with the same perception and hull solves 90 % of these layouts, the
policy solves 75 % and is still improving, and perception makes no errors here.
The deficit is **learning**: too little exposure and exploration for tight
three-panel geometry. In the paper's terms the lever is the curriculum and
training distribution (part of the formulation), not the reward or
observation design. Two secondary contributors worth a check: SAC's entropy
coefficient has decayed to about 0.003 (little exploration late in training),
and the obstacle term gives no signal beyond 2.0 m, so it does not reward
opening clearance early.

## 3. What counts as "bad" at 3 M

Judge on the full evaluation that runs automatically after 3 M (Tier 1, frozen
suite, Paper 2 set), against baseline-v2 SAC:

- **Good enough:** frozen-suite headline within noise of baseline-v2 SAC
  (0.89), and the Paper 2 set no-target 30/30 with fixed-speed target cases
  clearly above baseline (0.45 overall; head-on 0.20, starboard crossing 0.27).
- **Bad:** any no-target failure on the Paper 2 set, fixed-speed target
  success not clearly above baseline, or a frozen-suite drop of more than
  about 0.05.

Also check whether an earlier checkpoint does better on the three-panel
development set than the one selected by the combined score; if so, the
selection rule, not training, is the cheap fix (Option 0).

## 4. Recovery options if 3 M is bad (from the 2.5 M checkpoint, 0.5 M steps)

**Start point: the 2.5 M checkpoint.** Its model (`sac_2500000_steps.zip`,
`sac_vecnormalize_2500000_steps.pkl`) is in the run folder and its replay
buffer is preserved in
`PhD/asv_replay_buffers/kept_checkpoints/sac_formulation_seed0_bl3_sac_replay_buffer_2500000_steps.pkl`
(the run's own copy is rotated away at 2.75 M). The 2.0 M checkpoint has no
replay buffer left, so starting there would mean an empty buffer and more
steps for less; 2.5 M is the right start. Budget: 0.5 M steps ≈ 9 h, plus
3–4 h of evaluation.

| # | Option | What changes | Cost | Expected effect |
|---|---|---|---|---|
| 0 | **Re-select the checkpoint** | pick the checkpoint by the three-panel development score (or the Paper 2-style no-target score) instead of the combined score | evaluation only (≈ 1 h per checkpoint) | helps if a checkpoint between 2.0 and 3.0 M is better on three-panel layouts |
| 1 | **Focused stage 8** (recommended) | continue stage 7 with three-panel layouts at 0.70 of episodes (was 0.45); weight toward no-target spread layouts and the weakest encounters (crossing from starboard, head-on, crossing from port); **failure replay** — each worker keeps the scenarios it collided in and re-issues one with probability 0.3 (prioritized level replay, Jiang et al., 2021 [VERIFY]) | code: a stage-8 overlay + a small env replay pool (~½ day incl. tests) | more practice exactly where it fails; no change to reward or observation |
| 2 | **Exploration reset** | for the first 100 k steps raise SAC's target entropy (−2 → −1) or floor the entropy coefficient at 0.01 | config only | re-opens exploration so new avoidance lines can be found; combine with 1 |
| 3 | **Earlier clearance signal** | reward-only change: obstacle term reaches 2.5 m with decay 0.8 m (was 2.0 / 0.6) | constant in the overlay; formulation change to report | rewards starting the avoidance earlier; changes the formulation, so only if 1 + 2 are not enough |
| 4 | **Runtime safety filter** (`SAFETY_LAYER_V2_PLAN.md`) | DWA-style shield around the policy at deployment | ~2 days, no training | prevents collisions in the field; reported as a system result, not learned behavior |

**Do not** change the observation (the network input would no longer match the
checkpoint), and do not re-shape the COLREGs terms (not implicated).

**Recommended sequence if 3 M is bad:** Option 0 first (cheap). If no checkpoint
is good enough, run Options 1 + 2 from 2.5 M to 3.0 M; keep Option 3 in
reserve; build Option 4 for the field regardless.

## 5. For the paper

- Record the extension (2.5 → 3.0 M) and any stage 8 as deviations from the
  frozen baseline-v3 budget, and apply the same budget and stages to the other
  learners and seeds if v3 becomes the paper's formulation.
- The diagnosis itself is reportable: failures in three-panel layouts are
  policy skill, not perception or infeasibility (DWA 18/20 with the same
  perception and hull).
