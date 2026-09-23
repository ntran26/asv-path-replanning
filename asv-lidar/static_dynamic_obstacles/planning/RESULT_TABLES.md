# Pre-committed result tables — suite 3.0 (draft for sign-off)

**Status:** DRAFT, empty by design (04a §9.3). Committed with the frozen suite before the
first headline evaluation. Rows and columns are fixed now; values are filled only from
frozen-namespace runs. Every cell reports the mean over 5 seeds with a 95 % CI.

> **Status note (2026-09-23).** Three decisions are now made and this draft needs restating before sign-off (B6). **Framing:** the paper is a **formulation plus a five-learner comparison**, so there is no "Proposed (SAC, full)" method — the five learners are the comparison, and the classical comparators carry N2. **Formulation:** baseline-v2 (F96) — the Rule 17(b) below-floor being-overtaken draws are out of training and the suite, so any row or claim resting on them goes. **Field work** is delayed within this paper, so N3, RQ4 and C-6 stand. Still to settle: which learner carries the ablations, and C-4's comparison, which needs a **third rung** (no encounter feature / class one-hot / full context branch) now that the observation has a context branch.

Learners (your call, 2026-09-18): PPO, RecurrentPPO, TD3, SAC and TQC, each over multiple seeds. TQC is the distributional arm (04a §8.2). PPO is also the development vehicle for the reward; only its frozen-suite runs are reported.
The CODEX reference controller is a supplementary comparator, not one of the pre-registered three.

## R1 — Tier B holdout (48 cells × 20 = 960 episodes per seed)

| Method | Success | Static coll. | Boundary coll. | Target coll. | RMS CTE (m) | Path ratio | Intervention rate (supervisor on) |
|---|---|---|---|---|---|---|---|
| Proposed (SAC, full) | | | | | | | |
| SAC, no COLREGs terms | | | | | | | |
| TQC (distributional) | | | | | | | |
| TD3 | | | | | | | |
| RecurrentPPO | | | | | | | |
| PPO | | | | | | | |
| Prior SAC (Paper 2, frozen) | | | | | | | |
| Encounter-specific VO | | | | | | | |
| COLREGs-VO | | | | | | | |
| LOS-PID + DWA | | | | | | | |
| Reference controller (supplementary) | | | | | | | |

**R1b — success by stratum.** Same rows. Columns: basin, channel wide [7.60, 10.00], channel intermediate [4.26, 7.60], channel narrow [3.50, 4.26].

## R2 — violation rate per class (supervisor off)

| Method | Head-on | Crossing (stbd) | Crossing (port) | Overtaking | Being overtaken |
|---|---|---|---|---|---|

## R3 — by target behaviour

| Method | Constant velocity | Compliant reactive | Non-compliant |
|---|---|---|---|

## R4 — channel-width sweep (Study 1, channel mode only)

| Width (m) | B | Success | Compliant success | Min CPA (m, median) | Mode share, crossing (alter / 8(e)) | Mode share, overtaking | Governing rule (predicted) | Rejection rate |
|---|---|---|---|---|---|---|---|---|
| 10.0 | 20 | | | | | | | |
| 8.0 | 16 | | | | | | | |
| 7.0 | 14 | | | | | | | |
| 6.0 | 12 | | | | | | | |
| 5.0 | 10 | | | | | | | |
| 4.5 | 9 | | | | | | | |
| 4.0 | 8 | | | | | | | |
| 3.5 | 7 | | | | | | | |

Plus: width at which each classical comparator becomes inadmissible.

## R5 — perception degradation (Study 2, basin primary + one channel-narrow level)

| Axis | 0× | 0.5× | 1× | 2× | 4× | FMR at 2× |
|---|---|---|---|---|---|---|
| Pose drift | | | | | | |
| Detection dropout | | | | | | |
| Occlusion duration | | | | | | |
| Velocity-estimate noise | | | | | | |
| Joint corner (2×) | | | | | | |

## R6 — ablation (2 × 2 plus leave-one-out)

| COLREGs reward terms | Encounter feature | Success | Violation rate | Target coll. |
|---|---|---|---|---|
| on | on | | | |
| on | off | | | |
| off | on | | | |
| off | off | | | |

Leave-one-out rows (3 seeds, stated): −v_port, −v_bow, −v_side, −v_hold, −v_r8.

## R7 — reward scale audit

Per-term episode-integrated contribution, random policy and trained policy, both geometry modes, with the 02a §8.1 ordering checked.

## R8 — Tier A (38 named; 35 realised, 3 reported infeasible) and Around the Clock (24 + 24 × 3 widths)

One row per case: success over 10 rollouts, min CPA, compliant, first-alteration sense.

## R9 — domain randomisation in the field (RQ4)

| Policy | Field success | Field min CPA | Sim success | Gap |
|---|---|---|---|---|
| Nominal model only | | | | |
| Randomised over identified uncertainty | | | | |
