# Safety-layer development: V10 through V17

V16 improves the completed 32-case development comparison to **25 goals**, up from **16 with SAC alone or V9**. It rescues 11 SAC failures and preserves 14 of SAC's 16 successes. **Seven contacts remain, including two cases SAC solves. The requested 100% safety and no-regression target is not met.** On the separate, previously selected 40-case paired challenge, V16 reaches 25 goals versus fresh SAC 21, with eight rescues and four broken SAC successes.

The fixed policy is SAC baseline 3, kept best at 3M timesteps. The user explicitly designated test-set-v2 and DV3 for development. No retraining, constants changes or tuning on other held-out sets is involved. These cohorts deliberately emphasize failures and broken successes; their percentages are not full-suite success rates.

## Matched 32-case comparison

| Controller | Goals / 32 | Rescued SAC failures / 16 | Preserved SAC successes / 16 | Broken SAC successes |
|---|---:|---:|---:|---:|
| SAC alone | 16 | 0 | 16 | 0 |
| V8 | 16 | 10 | 6 | 10 |
| V9 | 16 | 5 | 11 | 5 |
| V10 | 18 | 11 | 7 | 9 |
| V11 | 18 | 12 | 6 | 10 |
| V14 | 22 | 11 | 11 | 5 |
| V15 | 23 | 11 | 12 | 4 |
| V16 | **25** | **11** | **14** | **2** |

V16 has four target contacts, two obstacle contacts and one boundary contact, with no timeout. All six successful controls remain goals. It gains DV3-BO-CV-04 and CH-HO-CV-073 over V15 with no lost V15 goal. The two remaining broken SAC successes are BAS-HO-NC-059 (target) and BAS-NU-CV-070 (boundary).

[Verified V16 cohort report](reports/motion_axis32/report.md), [preservation by selection stratum](reports/motion_axis32/stratum_preservation.md), and [per-case paired results](reports/motion_axis32/paired.csv). SAC/V8/V9 references come from the separately completed, fresh matched pilot; they are not additional runs in this campaign. The cohort report preserves source differences and the integration audits rather than treating all controller versions as identical code.

## Separate 40-case paired challenge

| Controller | Goals / 40 | Target contacts | Obstacle contacts | Boundary contacts | Timeouts |
|---|---:|---:|---:|---:|---:|
| Fresh SAC alone | 21 | 8 | 10 | 1 | 0 |
| Default V16 | 25 | 5 | 4 | 6 | 0 |

V16 rescues 8/19 SAC failures and preserves 17/21 SAC successes. All 10 successful controls remain goals. It preserves 7/11 of the remaining known V8-broken SAC successes, retains 8/11 rescue controls and rescues 0/8 cases that SAC and V8 both failed. Fresh SAC reproduces all 40 historical outcomes. Against the explicitly historical V8 reference, V16 gains seven goals and loses three; no new V8 run is implied.

The four broken SAC successes are DV3-CRS-CV-04, P2-L2-CRS-FIX-19 and P2-L3-BO-FIX-09 (boundary), and P2-L3-CRP-FIX-03 (target). Boundary contacts increase from one to six, so the net goal gain must not be described as eliminating every type of collision.

This selection was frozen before expanded-candidate results and is disjoint from the first 32, but deliberately enriched and authorized for development. Keep its denominator separate. [Verified paired report](reports/v16_broader40_paired/report.md), [all 80 records paired by case](reports/v16_broader40_paired/paired.csv).

## Changes supported by the evidence

**Perception, then selective policy preservation.** V10 compared the proposed override with the current SAC command followed by the same checked backup. V11 added independently observed motion admission and bounded track persistence. V13/V14 corrected partial-hull centre estimates and selected a single persistent estimate for an exact source. V15 restricted additional policy-prefix search to failing proposals or ordinary proposals below the existing margin, using the actual selection floor. V16 corrects eligible moving partial-hull base views using measured course and known hull dimensions.

The strongest V16 evidence is a matched mechanism and outcome: inaccurate target centroids/axes rejected SAC's actual successful continuation in BO04 and CH-HO-CV-073; an onboard-only geometry correction improved the estimate, and fresh closed-loop runs then reached the goal. It requires fresh motion and extent evidence, preserves all static points and collision checks, and falls back to the original estimate when that evidence is absent.

Method inspirations and limitations are cited in every new implementation and in [the V16 plan](../../../planning/SAFETY_LAYER_V16_PLAN.md) and [the V10–V15 development history](../../../planning/SAFETY_LAYER_V11_PLAN.md): [Wabersich and Zeilinger](https://arxiv.org/html/1812.05506v4#S4.SS1) for feasible first-input filtering, [Zheng et al.](https://arxiv.org/abs/2102.12124v3) for cross-entropy trajectory search, [Nuss et al.](https://arxiv.org/abs/1605.02406) and [Yoon et al.](https://arxiv.org/abs/1809.06972) for motion persistence/free-space evidence, and [Granstrom et al.](https://arxiv.org/abs/1604.00970) plus [Zhang et al.](https://doi.org/10.1109/IVS.2017.7995698) for extended-object geometry. These are engineering adaptations, without those papers' formal safety guarantees.

## Rejected alternatives

- Disabling V9's blanket current-plan veto alone reproduces V8's 16/32 outcomes.
- Preferring any currently feasible SAC backup under V10 reaches only 15/32, with two gains and five losses versus default V10.
- The isolated shorter policy-prefix option reaches 17/32: two gains but three lost V10 rescues. Its two slices have an explicitly audited archive-inventory difference; they are not silently treated as a strict source-identical cohort.
- Broad prefix search on V14 reaches 21/32, below default V14's 22/32.
- Retesting the existing any-feasible-policy option with V16 on nine targeted cases reaches **3/9**, versus default V16's **6/9**: zero gains and three losses. The lost goals are DV3-HO-CV-03, P2-L1-CRP-VAR-12 and P2-L2-HO-FIX-19. The option remains disabled. [Verified probe](reports/v16_feasible_probe9/report.md).
- V17 uses measured yaw directly on fresh frames and the inherited model prior on stale frames. It reaches **6/9**, equal to V16: it rescues DV3-HO-VS-01 but loses P2-L2-HO-FIX-19. The same two SAC successes remain broken. Do not promote it. [Verified probe](reports/v17_fresh_yaw_probe9/report.md), [method and Luenberger citation](../../../planning/SAFETY_LAYER_V17_PLAN.md).

These tests explain why merely intervening less does not ensure better episode outcomes. A currently passing backup can be replaced later, or cease to pass when the state/target estimate changes.

## What still blocks the requested result

**Own-vessel prediction error.** BAS-NU-CV-070 is a false rejection of a successful SAC continuation against static geometry. At V15/V16's first changed action, an unchanged 192-plan search has zero passing policy-prefix backups. Removing its gate alone therefore does not solve the case. Correct initial body state and finer integration do not remove the longer-horizon yaw error. The plant is randomized; the forecaster uses nominal dynamics. [Gate audit](audits/bas_nu_v15_gate/REPORT.md), [initial-state decomposition](audits/bas070_initial_state_decomposition/REPORT.md).

**Changing target behaviour.** BAS-HO-NC-059 has a passing same-tail SAC backup at +0.100 m, rejected for its lower margin. However, both fresh broader-prefix and any-feasible trials still fail the episode. Even perfect current target geometry with constant-velocity prediction rejects SAC's successful future trajectory. On the recorded V16 branch the target changes direction sharply, and persistent tracking retains the old heading for two decisions. Future target states on the successful SAC-only branch were not recorded; that turn is not a known SAC counterfactual. [First-divergence audit](audits/bas_ho_v16_first_divergence/REPORT.md).

**Recoverability of the starting state.** Earlier diagnostics found DV3-BO-CV-10, outside the current 32-case cohort, where all 10,426 sampled constant first commands contact within the first decision interval. This is not proof over the continuous action space, but it prevents assuming every generated episode is recoverable by a better trigger. This does not establish that any of the current seven failures is unavoidable. No case or contact is removed from its relevant denominator. [Existing plant feasibility audit](../development_v6_budget150/offline/blind_zone_oracle_replay/REPORT.md).

## Evaluation accounting

Every tag preserves its selected scenes, seeds, source archive, effective settings, checkpoint hash, reserved attempts, individual result JSON and per-decision traces. Reports validate those files before computing paired metrics. No automatic retries or old queues are used. At most two evaluation processes run, with one Torch/native thread each. Training and Paper 2 are untouched.

The live [campaign ledger](campaign_status.json) distinguishes completed records from planned work. Repeated cases across versions are separate experiments, not extra scenario coverage. The earlier 96-run V9 pilot and all earlier ledgers remain separate and immutable.

All **399 new runs are complete**, across 19 experiment tags and 72 distinct scenarios, with no retries, unfinished attempts or queued episodes. Constants and checkpoint hashes are unchanged. Native versions through 17 are selectable through the environment, suite CLIs and counterfactual dispatch; default selection is unchanged. The V16 and V17 native-integration audits preserve exact changes and newline conventions. V16 remains the stronger measured candidate; V17 is an ablation with no net gain.

Validation passed: 184 focused core/dispatch/report tests, 23 new yaw-observer tests, 28 final native-dispatch checks and 76 final reporting checks. These suites overlap and must not be summed as unique tests. The only test warning was a third-party NumPy deprecation. [Final verification](final_verification.json).


A separate causal yaw-response gain fit is **rejected for production**. It uses only the first 10 past measured transitions, but improved one-step errors do not transfer to continuous prediction: at the full 8-second horizon it worsens heading forecasts in 12/19 valid V15 cases and 2/4 V16 cases. See [the cross-case audit](audits/causal_yaw_response_cohort/REPORT.md). No fitted gain enters V16 or the fresh-yaw ablation.
