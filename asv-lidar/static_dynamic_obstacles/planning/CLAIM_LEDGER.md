# Claim ledger — Paper 3 (draft for sign-off)

**Status:** DRAFT, written 2026-09-18 before any headline evaluation run (04a §9.3).
It becomes binding when you sign it off and it is committed with the frozen suite.
After that, a claim may be weakened or withdrawn, but not added or moved to fit a result.

> **Status note (2026-09-23).** Three decisions are now made and this draft needs restating before sign-off (B6). **Framing:** the paper is a **formulation plus a five-learner comparison**, so there is no "Proposed (SAC, full)" method — the five learners are the comparison, and the classical comparators carry N2. **Formulation:** baseline-v2 (F96) — the Rule 17(b) below-floor being-overtaken draws are out of training and the suite, so any row or claim resting on them goes. **Field work** is delayed within this paper, so N3, RQ4 and C-6 stand. Still to settle: which learner carries the ablations, and C-4's comparison, which needs a **third rung** (no encounter feature / class one-hot / full context branch) now that the observation has a context branch.

Every claim in the abstract and conclusion must map to a row here. **Evidence** names
the table or figure that decides it. **Prediction** is written before the run that
tests it. **Status** is `pending` until that evidence exists.

## 1. Claims

| # | Claim | RQ / contribution | Evidence | Prediction (pre-registered) | Status |
|---|---|---|---|---|---|
| C-1 | One end-to-end policy does path following, static avoidance and COLREGs-compliant manoeuvring in a laterally bounded basin and channel from onboard range sensing alone | RQ1 | R1 (Tier B, 780 episodes × 3 seeds, suite 3.1). **R8 dropped**: Tier A is out of the paper and Around the Clock is not built (2026-09-24) | Proposed method's success rate beats every classical comparator on Tier B overall; per-cell intervals reported | pending |
| C-2 | Rule 9 governs Rules 13–15 below a width threshold, and the policy's manoeuvre mode switches from alteration to 8(e) speed reduction there | C2, Study 1 | **R4 + manoeuvre-mode curve only** — suite 3.1 floors Tier B channels at 7.5 m, above the head-on (3.80 m) and overtaking (4.26 m) thresholds, so Tier B can no longer show the switch (2026-09-24) | **Crossing** mode share crosses 50 % between **8 m and 7 m**; **overtaking** between **4.5 m and 4.0 m**; **head-on** is resolved by channel-keeping at every width and never crosses (04a §6) | pending |
| C-3 | The width at which each classical comparator becomes inadmissible is predicted by the precedence-table thresholds (head-on, overtaking 4.26 m, crossing 7.60 m) | C2/C3, Study 1 | R4 (the sweep keeps widths to 3.5 m; comparators are COLREGs-VO (Kuwata) and the encounter-specific VO, tuned on the development set and pinned in `configs/comparators_v1.json`) | Each comparator's admissibility loss falls within ±0.5 m of its predicted threshold | pending |
| C-4 | Supplying the encounter class as an observation feature improves compliance over reward shaping alone | RQ2 | R6 (2 × 2 ablation) | Violation rate lower with the feature in both reward conditions; effect size reported with seed CIs | pending |
| C-5 | Under degraded perception the policy fails conservatively before it fails unsafely | RQ3, C1/C4, Study 2 | R5, failure-mode ratio FMR | FMR ≥ 1 at every level up to 2× nominal on each axis; the first axis to drive FMR < 1 is occlusion duration | pending |
| C-6 | Domain randomisation over identified uncertainty closes more of the sim-to-field gap than nominal identification alone | RQ4 | R9 (field) | Randomised policy's field success exceeds the nominal policy's; if not, the gap is attributed to localisation or disturbance, which is also reportable | pending |
| C-7 | **8(e) in two layers**: the learned policy slackens speed; an engineered runtime safety layer takes all way off. Only the first is a learned-compliance claim | F68 | R1/R2 with supervisor off; intervention rate as its own column | Development-set intervention rate ≤ 5 % for the proposed method; compliance metrics reported with the supervisor **off** | pending |
| C-8 | The policy transfers between basin geometry (slanted legs, oblique walls, Paper 2's layout) and parallel-walled channels without a separate model | 06, F74 | R1 by stratum (basin / channel wide / intermediate / narrow) | Basin-stratum success within 0.10 of channel-wide success | pending |
| C-9 | Every evaluation layout admits a route; classical failures on the suite are not infeasibility artefacts | F74 | A* feasibility filter, rejection ledger, reference controller. **Tier A is out of the paper (2026-09-24)**, so the named infeasible cases are not reported and this claim rests on the A* filter and the rejection ledger alone — Tier B cells that come up short are reported by the generator | 100 % of frozen cases pass the feasibility filter; the named infeasible cases (A-BO-N, A-NU-I, A-NU-N; Tier B narrow × null / being-overtaken) are reported, not sampled | pending |

## 2. What the paper will not claim

- Open-water Rule 15/17 role tables. The own ship gives way to a crossing target from either side (narrow-channel convention, A17). The paper states this.
- Rule 3(g) status for the own ship, Rule 18 responsibilities, Rule 19 restricted visibility, sound signals, or multi-target encounters.
- Legal compliance. Violation fractions and trajectory contracts are proxies.

## 3. Framing rules

- Strata are described by geometry only, never as "challenging for classical methods" (04a §4.5).
- Pre-committed expected failures (`A-FAIL-*`) are reported at whatever level they come out.
- Every result reports seed spread. Selection used the development namespace only; the frozen namespace is touched once per policy.

## Sign-off

| | |
|---|---|
| Signed off by | |
| Date | |
| Suite manifest digest | |
| Generator git SHA | |
