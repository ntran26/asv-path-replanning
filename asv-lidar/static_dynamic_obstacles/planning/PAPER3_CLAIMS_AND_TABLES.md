# Paper 3 claims and result tables (drafts for sign-off)

Merged on 2026-10-04 from `CLAIM_LEDGER.md` (part 1) and `RESULT_TABLES.md` (part 2): the claims
and the tables that decide them, in one file.

> **Status note (2026-10-04).** Both parts are the pre-registered drafts of 18-28 Sep and still
> need restating before sign-off. Since then:
>
> - **The headline evaluation is the 1,000-episode test set**, not the frozen suite on its own
>   (`tools/tiers/test_set.py`). The test set merges the frozen suite and the Paper 2
>   deployment-layout set, trimmed of near-duplicates. Version 4, the current one, replaces the
>   episodes no controller can solve after the target is first tracked
>   (`planning/BASELINE_V4_PLAN.md`, section 5e).
> - **Results so far, all safety off, none of them run under the pre-registered protocol of
>   multiple seeds:**
>
>   | Policy | Test set | Success |
>   |---|---|---|
>   | SAC 3 M, baseline-v3 seed 0 | v4 | 0.865 |
>   | SAC 3 M, baseline-v3 seed 0 | v1 | 0.857 |
>   | LOS-DWA | v1 | 0.804 |
>   | COLREGs-VO | v1 | 0.703 |
>
> - **Runs.** Of the four-learner, three-seed campaign, only baseline-v3 SAC and PPO seed 0
>   exist. The baseline-v1 and baseline-v2 runs were removed on 2026-10-04, but their result
>   summaries are kept. The formulation for the campaign, v3 or v4, is decided by the v4.3 SAC
>   pair.
> - **The tables below still describe suite 3.0.** R1, the Tier B holdout, becomes the test-set
>   table, and cells are filled only once the final formulation is fixed.

## Part 1. Claim ledger (draft for sign-off)

**Status:** DRAFT, written 2026-09-18 before any headline evaluation run (04a section 9.3).
It becomes binding once signed off and committed with the frozen suite.
After that, a claim may be weakened or withdrawn, but not added or moved to fit a result.

> **Status note (2026-09-28).** The Introduction is now draft 4 (`planning/Paper3_Introduction_draft4.docx`), with the literature review moved into section 1.2 and the paper renumbered to six sections (`PAPER3_DRAFT_SKELETON.md` revision 4). Contributions are now **C1-C4** (C1 formulation; C2 geometric framework for constrained encounter responses, absorbing the old width-sweep C4; C3 four-learner comparison with classical references; C4 evaluation design, absorbing the old perception C5 and protocol C6); the old field C7 and **RQ4 / C-6 are open (S13)** because draft 4 calls physical transfer "planned". Draft 4 adds **RQ6** (how the response changes as maneuvering room decreases), which C-2 and C-3 now serve. Paper sections cited below use the revision-4 numbers.
>
> **Status note (2026-09-23).** Three decisions are now made and this draft needs restating before sign-off (B6). **Framing:** the paper is a **formulation plus a five-learner comparison**, so there is no "Proposed (SAC, full)" method — the five learners are the comparison, and the classical comparators carry N2. **Formulation:** baseline-v2 (F96) — the Rule 17(b) below-floor being-overtaken draws are out of training and the suite, so any row or claim resting on them goes. **Field work** is delayed within this paper, so N3, RQ4 and C-6 stand. Still to settle: which learner carries the ablations, and C-4's comparison, which needs a **third rung** (no encounter feature / class one-hot / full context branch) now that the observation has a context branch.

Every claim in the abstract and conclusion must map to a row here. **Evidence** names
the table or figure that decides it. **Prediction** is written before the run that
tests it. **Status** is `pending` until that evidence exists.

### 1. Claims

| # | Claim | RQ / contribution | Evidence | Prediction (pre-registered) | Status |
|---|---|---|---|---|---|
| C-1 | One end-to-end policy does path following, static avoidance and COLREGs-compliant manoeuvring in a laterally bounded basin and channel from onboard range sensing alone | RQ1 | R1 (Tier B, suite 3.4: 800 constant-velocity episodes × 3 seeds). **R8 dropped**: Tier A is out of the paper and Around the Clock is not built (2026-09-24) | Proposed method's success rate beats every classical comparator on Tier B overall; per-cell intervals reported | pending |
| C-2 | Rule 9 governs Rules 13–15 below a width threshold, and the policy's manoeuvre mode switches from alteration to 8(e) speed reduction there | RQ6, C2, Study 1 | **R4 + manoeuvre-mode curve only** — suite 3.4 draws Tier B channels at 7.5–10 m, above the head-on (3.80 m) and overtaking (4.26 m) thresholds, so Tier B cannot show the switch (2026-09-24, 2026-09-27) | **Crossing** mode share crosses 50 % between **8 m and 7 m**; **overtaking** between **4.5 m and 4.0 m**; **head-on** is resolved by channel-keeping at every width and never crosses (04a section 6) | pending |
| C-3 | The width at which each classical comparator becomes inadmissible is predicted by the precedence-table thresholds (head-on, overtaking 4.26 m, crossing 7.60 m) | RQ6, C2, Study 1 | R4 (the sweep keeps widths to 3.5 m; comparators are COLREGs-VO (Kuwata) and the encounter-specific VO, tuned on the development set and pinned in `configs/comparators_v1.json`) | Each comparator's admissibility loss falls within ±0.5 m of its predicted threshold | pending |
| C-4 | Supplying the encounter class as an observation feature improves compliance over reward shaping alone | RQ2 | R6 (2 × 2 ablation) | Violation rate lower with the feature in both reward conditions; effect size reported with seed CIs | pending |
| C-5 | Under degraded perception the policy fails conservatively before it fails unsafely | RQ3, C1/C4, Study 2 | R5, failure-mode ratio FMR | FMR ≥ 1 at every level up to 2× nominal on each axis; the first axis to drive FMR < 1 is occlusion duration | pending |
| C-6 | Domain randomisation over identified uncertainty closes more of the sim-to-field gap than nominal identification alone | RQ4 [open, S13] | R9 (field) | Randomised policy's field success exceeds the nominal policy's; if not, the gap is attributed to localisation or disturbance, which is also reportable | pending |
| C-7 | **8(e) in two layers**: the learned policy slackens speed; an engineered runtime safety layer takes all way off. Only the first is a learned-compliance claim | F68 | R1/R2 with safety layer off; intervention rate as its own column | Development-set intervention rate ≤ 5 % for the proposed method; compliance metrics reported with the safety layer **off** | pending |
| C-8 | The policy transfers between basin geometry (slanted legs, oblique walls, Paper 2's layout) and parallel-walled channels without a separate model | 06, F74 | R1 by stratum (basin / channel), over the classes both strata hold (head-on, crossing, overtaking) | Basin-stratum success within 0.10 of channel success | pending |
| C-9 | Every evaluation layout admits a route; classical failures on the suite are not infeasibility artefacts | F74 | A* feasibility filter, rejection ledger, reference controller. **Tier A is out of the paper (2026-09-24)**, so the named infeasible cases are not reported and this claim rests on the A* filter and the rejection ledger alone — Tier B cells that come up short are reported by the generator | 100 % of frozen cases pass the feasibility filter; the named infeasible cases (A-BO-N, A-NU-I, A-NU-N; Tier B narrow × null / being-overtaken) are reported, not sampled | pending |

### 2. What the paper will not claim

- Open-water Rule 15/17 role tables. The own ship gives way to a crossing target from either side (narrow-channel convention, A17). The paper states this.
- Rule 3(g) status for the own ship, Rule 18 responsibilities, Rule 19 restricted visibility, sound signals, or multi-target encounters.
- Legal compliance or certification. Violation fractions and trajectory contracts are declared behavioural proxies.
- That operating bounds establish the legal applicability of Rule 9, or that a channel width is a universal legal criterion for a narrow channel: width is a geometric experimental variable (Introduction draft 4, section 1.2.3).
- That every dynamic encounter is avoidable: the A* route filter is a static check (F74).

### 3. Framing rules

- Strata are described by geometry only, never as "challenging for classical methods" (04a section 4.5).
- Pre-committed expected failures (`A-FAIL-*`) are reported at whatever level they come out.
- Every result reports seed spread. Selection used the development namespace only; the frozen namespace is touched once per policy.

### Sign-off

| | |
|---|---|
| Signed off by | |
| Date | |
| Suite manifest digest | |
| Generator git SHA | |

## Part 2. Pre-committed result tables, suite 3.0 (draft for sign-off)

**Status:** DRAFT, empty by design (04a section 9.3). Committed with the frozen suite before the
first headline evaluation. Rows and columns are fixed now; values are filled only from
frozen-namespace runs. Every cell reports the mean over **3 seeds** with a 95 % CI
(registered as 5 on 2026-09-18; **changed to 3 on 2026-09-24 by decision**, with the
campaign at 4 learners x 3 seeds = 12 runs). At 3 seeds a headline difference of
about 0.05 is at the edge of separability and the measured 0.20 crossing spread is
not separable, so learner comparisons are made on the headline and crossing
behaviour is reported descriptively, per seed.

> **Status note (2026-09-28).** The Introduction is now draft 4 (`planning/Paper3_Introduction_draft4.docx`), with the literature review moved into section 1.2 and the paper renumbered to six sections (`PAPER3_DRAFT_SKELETON.md` revision 4). Contributions are now **C1-C4** (C1 formulation; C2 geometric framework for constrained encounter responses, absorbing the old width-sweep C4; C3 four-learner comparison with classical references; C4 evaluation design, absorbing the old perception C5 and protocol C6); the old field C7 and **RQ4 / C-6 are open (S13)** because draft 4 calls physical transfer "planned". Draft 4 adds **RQ6** (how the response changes as maneuvering room decreases), which C-2 and C-3 now serve. Paper sections cited below use the revision-4 numbers.
>
> **Status note (2026-09-23).** Three decisions are now made and this draft needs restating before sign-off (B6). **Framing:** the paper is a **formulation plus a five-learner comparison**, so there is no "Proposed (SAC, full)" method — the five learners are the comparison, and the classical comparators carry N2. **Formulation:** baseline-v2 (F96) — the Rule 17(b) below-floor being-overtaken draws are out of training and the suite, so any row or claim resting on them goes. **Field work** is delayed within this paper, so N3, RQ4 and C-6 stand. Still to settle: which learner carries the ablations, and C-4's comparison, which needs a **third rung** (no encounter feature / class one-hot / full context branch) now that the observation has a context branch.

Learners: PPO, RecurrentPPO, SAC and TQC, 3 seeds each (**TD3 dropped 2026-09-24**). TQC is the distributional arm (04a section 8.2). PPO is also the development vehicle for the reward; only its frozen-suite runs are reported.
The prototype's reference controller is a supplementary comparator, not one of the pre-registered three.

### R1 — Tier B holdout (**800 constant-velocity episodes per seed**, suite 3.4: 8 cells × 100, drawn like the development set)

| Method | Success | Static coll. | Boundary coll. | Target coll. | RMS CTE (m) | Path ratio | Intervention rate (safety layer on) |
|---|---|---|---|---|---|---|---|
| SAC | | | | | | | |
| TQC (distributional) | | | | | | | |
| RecurrentPPO | | | | | | | |
| PPO | | | | | | | |
| SAC, no COLREGs terms (ablation) | | | | | | | |
| Prior SAC (Paper 2, frozen) | | | | | | | |
| Encounter-specific VO | | | | | | | |
| COLREGs-VO | | | | | | | |
| LOS-PID + DWA | | | | | | | |
| Reference controller (supplementary) | | | | | | | |

**R1b — success by stratum.** Same rows. Columns: basin, channel wide [7.60, 10.00], channel intermediate [4.26, 7.60], channel narrow [3.50, 4.26].

### R2 — violation rate per class (safety layer off)

| Method | Head-on | Crossing (stbd) | Crossing (port) | Overtaking | Being overtaken |
|---|---|---|---|---|---|

### R3 — by target behaviour (the robustness set, suite 3.4)

The headline's scenarios and seeds again with a reactive target (every encounter class, 700) and a non-compliant one (head-on, 200), each paired with its constant-velocity twin.

Behaviours as realised since suite 3.2 (A34): `cv` constant velocity (`T-CV`), `re` compliant reactive (`T-RE`, the COLREGs-VO rule from the target's side), `nc` non-compliant (`T-NC2`, alters to port, in head-on; `T-NC1`, stands on when give-way, elsewhere). `T-NC1` moves like `T-CV`; outside head-on the `nc` rows measure the case where the target should have given way and did not.

| Method | Constant velocity | Compliant reactive | Non-compliant |
|---|---|---|---|

### R4 — channel-width sweep (Study 1, channel mode only)

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

### R5 — perception degradation (Study 2, basin primary + one channel-narrow level)

| Axis | 0× | 0.5× | 1× | 2× | 4× | FMR at 2× |
|---|---|---|---|---|---|---|
| Pose drift | | | | | | |
| Detection dropout | | | | | | |
| Occlusion duration | | | | | | |
| Velocity-estimate noise | | | | | | |
| Joint corner (2×) | | | | | | |

### R6 — ablation (2 × 2 plus leave-one-out)

| COLREGs reward terms | Encounter feature | Success | Violation rate | Target coll. |
|---|---|---|---|---|
| on | on | | | |
| on | off | | | |
| off | on | | | |
| off | off | | | |

Leave-one-out rows (3 seeds, stated): −v_port, −v_bow, −v_side, −v_hold, −v_r8.

### R7 — reward scale audit

Per-term episode-integrated contribution, random policy and trained policy, both geometry modes, with the 02a section 8.1 ordering checked.

### R8 — Around the Clock (24 + 24 × 3 widths)

**Tier A is out of this paper (2026-09-24, decision).** The 38 named cases stay in the
suite and can be reported later; nothing in the paper depends on them. Around the Clock
has no scenario builder yet (C3–C6), so this table currently has no content.

One row per case: success over 10 rollouts, min CPA, compliant, first-alteration sense.

### R9 — domain randomisation in the field (RQ4; open, S13)

| Policy | Field success | Field min CPA | Sim success | Gap |
|---|---|---|---|---|
| Nominal model only | | | | |
| Randomised over identified uncertainty | | | | |
