# V8 saved counterfactual follow-up audit

All results below are reconstructed from existing saved runs. No simulator reset, policy inference, new episode or new rollout was performed. The nohold run used V7 with hold-back gain set to infinity; it is the evaluated behavior subsequently named V8.

## Reconstruction

The base contains 1,150 unique episodes (1,000 test-set v2 and 150 DV3). The nohold subset contains exactly the 549 cases where base V7 fired. Of these, 526 have a first-fire branch and 23 never fire under nohold. The other 601 cases never fired in base V7 and retain their recorded policy outcome. Missing branches are never treated as successful: a branch is required for every nonzero-fire episode. Contiguous step counts, fire counts, first-fire indices, unique identities, seeds and policy outcomes are checked against the CSVs. Reconstruction inherits the runner's first-fire shadow/replay equivalence claim; CSV integrity checks do not independently prove simulator-clone equivalence.

| Set | N | Policy goals | V4 goals | V7 goals | V8 goals | V8 rescued | V8 broken |
|---|---:|---:|---:|---:|---:|---:|---:|
| ts2 | 1000 | 872 | 880 | 904 | 912 | 57 | 17 |
| dv3 | 150 | 111 | 123 | 128 | 127 | 20 | 4 |
| all | 1150 | 983 | 1003 | 1032 | 1039 | 77 | 21 |

V8 test-set outcomes: 912 goals, 31 target, 27 obstacle and 30 boundary contacts. Its 88 failures comprise 71 unrescued policy failures and 17 broken policy successes. The 71 have first reasons: no fire 17, turn 32, brake 9, searched escape 5, last certificate 2, continue 6. These are first interventions, not terminal failure causes.

## Current-certification gap

V8 disables hold-back but retains the inherited `last certificate` fallback. This follows a stored continuation despite failure of its current check. Therefore disabling hold-back alone does not establish that every intervention has a currently passing escape plan.

| First-fire last certificate | Cases | Rescued | Broken | Both fail | Both goal |
|---|---:|---:|---:|---:|---:|
| ts2 | 27 | 0 | 3 | 2 | 22 |
| dv3 | 9 | 4 | 1 | 2 | 2 |

All 27 test-set first-fire last-certificate clearances are negative. The three broken successes are CH-CR-RE-038 (step 16, selected −1.2419 m, repaired policy −1.0783 m), BAS-CR-RE-072 (step 6, −0.2721 m versus +0.0430 m), and BAS-HO-NC-059 (step 20, −0.6706 m versus −0.9155 m). The latter has a worse repaired-policy clearance, so a same-tail clearance comparison would not suppress all three.

The counterexamples matter: DV3-HO-CV-03, DV3-HO-VS-01, DV3-CRP-CV-04 and DV3-CRP-CV-17 were rescued after first firing through last certificate. DV3-BO-CV-04 was broken. Two of those four rescues also have repaired-policy clearance greater than the selected continuation's clearance. A clearance-only dominance rule or blanket expiry removal therefore has no established net benefit.

## A concrete experimental guard

Recheck the actual selected backup and the same backup with only its first action replaced by the policy command, using identical snapshot, actuator history, horizon and constraints. A policy-preserving decision should depend on that actual comparison, including hard feasibility and first-violation time, rather than a rounded best-of-bank policy margin or a uniform new clearance threshold. A V9 guard built around this comparison remains experimental; these records cannot establish its closed-loop result, and it must not be promoted as improving 912/1,000.

Any oracle total derived here is restricted to selecting between the already observed first-fire branch and the recorded SAC-only episode. It does not bound different trigger times, different future interventions or new rescue methods.

Raising all intervention margins to 0.15 m is not supported here: 22 of 57 rescues and 9 of 17 breaks began below 0.15 m (the nine include three negative last certificates). All first-fire TS2 rows have risk-monitor urgent=false, including every rescue; urgency-only gating cannot be justified by a first-fire oracle calculation. This does not predict a later-triggered controller's outcome.

## What can and cannot be checked offline

`steps.csv` records pre-decision summaries along the policy-alone shadow trajectory. `branches.csv` records final branch outcomes and intervention counts, not branch decision traces. Only the first-fire branch corresponds to the complete filtered episode; later phase-1 branches are conditioned on a different policy-only history. We never pool them as independent closed-loop episodes.

`checked_clearance` and `v7_repair_clearance` compare selected versus repaired same-tail trajectories, but the CSV omits repair first-violation time, candidate first-violation time, full controls, actuator delay buffer, state snapshot, static points and target tracks. The original Python environment/filter copies were local runtime objects; they were not serialized to these outputs. Thus exact hard checks cannot be rerun from these CSVs without new state acquisition. `margin_opportunities.csv` only flags numerical clearance opportunities, not certified dominance or expected rescued episodes.

`policy_margin` is rounded and takes the best passing primitive recovery, with −inf denoting no passing primitive; it is not the same trajectory as the prefix repair. `best_margin` excludes brake candidates. Neither should be directly substituted for the selected/repaired plan comparison. The recorded `speed` comes from `env.u_body`, so it is simulator state rather than a verified noisy onboard measurement. The saved gate fit uses critic Q and its trend, not speed; this logging caveat does not by itself invalidate those critic-gate results.

## Sources and provenance

The matching/data sources are [the V8 plan](<../../../../planning/SAFETY_LAYER_V8_PLAN.md>), [base episodes](../../trigger_counterfactual/episodes.csv), [base steps](../../trigger_counterfactual/steps.csv), [base branches](../../trigger_counterfactual/branches.csv), and the corresponding [nohold episodes](../../trigger_counterfactual/v7_nohold/episodes.csv), [steps](../../trigger_counterfactual/v7_nohold/steps.csv) and [branches](../../trigger_counterfactual/v7_nohold/branches.csv). The implementation inspected is [the shadow runner](../../../../tools/diagnostics/safety/trigger_counterfactual.py), [V6 selection](../../../../src/safety_v6.py), [V7 prefix repair](../../../../src/safety_v7.py), and [V8](../../../../src/safety_v8.py). Input bytes and current review-source SHA256 values are in provenance.json. Current source hashes are not historical run snapshots; the old configs record model path/settings but do not pin source/model/runtime hashes. This report makes no new method-performance or formal safety claim.
