# Safety layer V10-V16 development

Status: experimental. Completed V16 matched comparison:25/32 goals; separate40-case paired challenge complete:25 goals versus freshSAC21, eight rescues and four broken SAC successes. Current details: `planning/SAFETY_LAYER_V16_PLAN.md` and `results/safety_dev/v10_iterations/report.md`. No claim of 100% safety or preservation of every SAC success. The checkpoint is SAC baseline 3, kept best at 3M timesteps. No policy retraining or constants edits.

## Objective and acceptance

Count a rescue only when a baseline failure becomes a goal. Count every baseline goal becoming contact/timeout as a regression. Report target, obstacle and boundary contacts separately; stopping indefinitely is not success. Compare identical cached geometry and episode seeds. Selection is intentionally enriched for regressions, rescues and missing-target failures and is not a population success-rate estimate.

The prior 32-case V9 pilot reached 16 goals for each of SAC, V8 and V9. V9 rescued five SAC failures but lost five SAC successes. Its current-plan veto also discarded six V8 rescues. A rejected recovery trajectory does not establish that SAC is safe.

## V10: preserve a checked policy alternative

`src/safety_v10.py` retains the V8 recovery when neither alternative passes. It preserves SAC if the current SAC action followed by the same complete backup tail passes and the proposed recovery does not, or if both pass and SAC has at least the same clearance. An independently selectable any-feasible-policy preference is disabled by default. Evaluations use the same pre-command actuator history and complete horizon; signed braking and the active target checks remain intact.

Inspiration: Wabersich and Zeilinger (2021), *A predictive safety filter for learning-based control of constrained nonlinear dynamical systems*, Section 4.1, first-action deviation minimization subject to backup feasibility. [Paper](https://arxiv.org/html/1812.05506v4#S4.SS1). The paired-tail rule is an engineering adaptation; it does not implement their terminal invariant set or recursive feasibility proof.

A separate `v9_margin` ablation disables V9's blanket current-plan veto while retaining its margin preference. Both candidates are frozen for the same 32 development cases.

## V11: observed motion admission and track persistence

Three fresh V8 diagnostic replays reproduced target contacts with zero interventions. Full raw traces show a target present in LiDAR but absent from the safety snapshot. Motion-evidence admission fails against open water, partial occlusion biases the cluster-centroid velocity toward zero, and deleting a raw track ID deletes its provisional hypothesis.

`src/safety_track_persistence.py` requires coherent motion of both longitudinal endpoints across at least three full-length observed hull clusters. It anchors measured position and velocity from those observations and coasts independently of raw track deletion. It retains all static returns, expires hypotheses after an explicit 8-second experimental maximum, and clears them only when finite rays contradict their hull. Missing, stale or occluded measurements are not proof of free space. Admission and persistence are independently selectable. No scene ID, target truth, future policy observation or outcome enters the runtime rule.

Saved-snapshot replay represents all three targets continuously from decision 3 through contact, with no panel admissions in those three traces. This is representation evidence, not a collision-avoidance result or calibrated error bound. Older coasted anchors can be inaccurate and conservative; episode regressions must be measured.

Method inspiration: dynamic-state prediction/update in Nuss et al. (2016), [A Random Finite Set Approach for Dynamic Occupancy Grid Maps](https://arxiv.org/abs/1605.02406), and finite free-space consistency in Yoon et al. (2018), [Mapless Online Detection of Dynamic Objects in 3D Lidar](https://arxiv.org/abs/1809.06972). The deterministic extent/translation gates are project-specific engineering hypotheses, not implementations of the papers' full estimators.

## Optional policy-prefix search

The inherited search fixes SAC for COMMIT_S=1 second although the environment requests a new decision every 0.5 seconds. `src/safety_policy_prefix.py` fixes exactly the current decision, then searches a complete explicit backup. It checks at most 192 plans including primitive/warm seeds, followed by three cross-entropy rounds. Default acceptance retains the existing 0.15 m trigger margin and all hard checks. Failure does not authorize a pass or discard the parent's fallback. A passing plan is retained exactly and advanced normally next decision.

Inspiration: the first-input objective in Wabersich and Zeilinger above; sampled trajectory optimization in Zheng et al. (2022), [Safe Learning-based Gradient-free Model Predictive Control Based on Cross-entropy Method](https://arxiv.org/abs/2102.12124v3). The finite search does not inherit the paper's uncertainty, CLF/CBF or safety guarantees. This option is separately ablated and initially disabled.

## Computation and evidence

`src/safety_fast_geometry.py` tiles the existing point-clearance arithmetic to bound temporary array size. It changes neither geometry nor margins; exact-output parity and timing are checked before optional integration.

Use `tools/diagnostics/safety/safety_iteration.py` for new uniquely named runs, at most two evaluation processes. Each run reserves attempts, freezes code/settings/selection/checkpoint hashes, saves per-decision traces, and refuses automatic retries or changed source. Full perception snapshots are enabled for diagnostic cases only. Saved-data report generation does not call the simulator.

Current artifacts are under `results/safety_dev/v10_iterations/`. `baseline_source_audit.json` records the unrelated opt-in field-generator change since the prior pilot and exact reproduction checks. Cached scenes bypass that generator. The 40-case follow-up selection is disjoint from the 32 pilot cases and covers every remaining known V8 regression plus matched rescues, active/passive successful controls and failures. It is development data, not a newly claimed held-out benchmark.

Paper 2's project is read-only; `constants.py`, checkpoints, other training processes and previous evaluation ledgers are preserved. New evaluations are authorized to keep improving and testing. Results and completed-run counts will be appended after verification.

## First completed episode results

The frozen 32-case comparison is complete (64 new candidate runs). Margin-only V9: 16 goals, unchanged outcomes relative to V8. Conditional V10: 18 goals, 9 target contacts, 2 obstacle contacts and 3 boundary contacts. V10 gains BAS-CR-RE-072 and P2-L2-HO-FIX-19 without losing any V8 goal, but still breaks nine SAC successes while rescuing eleven SAC failures. This misses the requirement. [Verified report](../results/safety_dev/v10_iterations/reports/v10_comparison/report.md).

The three perception-diagnostic V11 runs are complete: CRS-VAR-08 and CRP-VAR-14 reach goals; CRP-FIX-05 still contacts the target after eight interventions. Those three were all target contacts without any interventions in V8/V10. Do not infer success preservation from this failure-only sample. The other 29 cases have completed with the identical V11 constructor. Together they yield 18/32 goals: two gains and two losses versus V10. The three diagnostic cases were not rerun for that comparison. See reports/persistence32 in the iteration folder.

A saved V10 audit finds 21 paired checks across five broken-SAC cases where SAC's tail passes but V10 retains the higher-clearance proposal. Four occur at the first changed action. The separate `prefer_any_feasible_policy=True` 32-case ablation completed at 15 goals: two gains and five losses versus V10. Reject that option; local feasibility preference did not improve episode-level preservation.

## Follow-ups: V12 source priority and V13 consistent estimates

The first V11 regression against V10, BAS-CR-RE-072, contains both base source 4 and its synthetic alternative on decisions 4-41. The stored backup first differs at decision 4, and the issued rudder first differs at decision 8 despite identical pre-state and SAC action. V12 is an isolated exact-ID ablation: an existing base view takes priority, while the independent anchor remains cached for a later loss. Existing does not imply fresh or accurate. In the remaining FIX05 failure a base view has three misses and approximately 1.257 m position error versus 0.336 m for its persistent alternative. Cross-ID association is not changed. See `v11_duplicate_source_audit.json` and `v12_existing_track_priority_caveat.json` in the iteration folder.

The FIX05 prediction audit isolates a second issue. Its target moves at constant velocity. At decision 18 the persistent centre is within approximately 0.020 m of truth. At decision 19 a near-dead-zone return flips the fitted lateral hull completion and refreshes that centre with approximately 0.254 m error. The stored backup scores -0.125 m under onboard target hypotheses, but +0.207 m with the recorded true CV target and the SAME predicted own trajectory. Truth is used only to score this diagnostic; it is never a controller input. Static and boundary checks remain more than 1 m clear in that comparison.

V13 implements two explicit, separately selectable engineering changes. A prior centre projected into the centre interval consistent with a partially observed hull prevents an ambiguous visible-face completion from jumping across the hull. For an observed axis interval [lo, hi] and known size d, the consistent centre interval is [hi-d/2, lo+d/2]; complete observed axes retain their measured midpoint. Original motion admission and existing extent tolerances remain in force. Secondly, each exact source retains its initial estimator: an already represented base source retains priority; an independently admitted safety source retains its own estimate if a base view later appears, instead of publishing inconsistent duplicate hulls. Anchors still expire, all static points remain, and different IDs remain independent. This is continuity management, not proof that the chosen estimate is correct.

Method inspiration: Granstrom, Baum and Reuter, *Extended Object Tracking: Introduction, Overview and Applications*, [survey and measurement modelling](https://arxiv.org/abs/1604.00970), together with the Nuss prediction/update reference above. The deterministic interval projection and source continuity rule are engineering derivations made here; they do not implement the survey's random-matrix or Bayesian multi-object estimators. An eight-run diagnostic comparison has completed: V12 reaches 3/4 goals, V13 2/4. Both preserve BAS-CR-RE-072 and rescue CRP-VAR-14; V13 loses CRS-VAR-08 to an obstacle, and both still contact the target in CRP-FIX-05. These four deliberately selected cases do not establish a best full-cohort candidate.

## Known limits of the 100% objective

Earlier development already includes an initially unobservable near-panel start, DV3-BO-CV-10, where all 10,426 sampled first commands contact within 0.5 s. This is not a continuous-control impossibility proof, but it is evidence that controller tuning alone may not rescue every generated start. Another case permits an immediate brake but fails after delaying one decision. These original seeds and outcomes are retained; no case is removed from a denominator to inflate safety. [Existing offline plant feasibility audit](../results/safety_dev/development_v6_budget150/offline/blind_zone_oracle_replay/REPORT.md).


## Completed one-decision prefix subset and V14 estimate selection

The isolated `v11_prefix` option (V10 control, both persistence switches off) retains 9/16 SAC-success cases, versus V10's 7/16. BAS-BO-CV-063 and CH-CR-CV-007 are the two additional preserved goals. All six successful controls remain goals. The complementary sixteen SAC-failure cases must be measured before claiming that the option retains earlier rescues.

A saved-state audit at BAS-CR-RE-072 decision 8 uses the actual next sixteen commands from the successful SAC-only trace. With the existing base target estimate the SAME command sequence fails at a predicted margin of -0.226 m. Replacing that exact source by its corrected persistent estimate gives +0.215 m. Current target position error decreases from 0.448 to 0.077 m and velocity error from 0.156 to 0.075 m/s; heading error increases, so the replacement is not uniformly more accurate. The anchor is 0.5 s old. Truth is used only for error scoring, never the runtime selection. Future reactive target truth is not available on this SAC branch. See `bas_re_known_sac_continuation_audit.json` and `bas_re_persistent_replacement_audit.json`.

V14 tests corrected persistent-estimate priority independently of whether the source was first admitted by the base tracker. It replaces only a matching raw source ID while an admitted anchor remains valid. No cross-ID merging, threshold relaxation, future policy information, target truth, or changes to policy observations are used. The disabled option reproduces V13. Admission, finite-ray contradiction, static points and age expiry remain intact. The method is an engineering predict/update and estimate-selection heuristic inspired by the Nuss and Granstrom references above; it is not a calibrated confidence ranking. An old CV anchor can be less accurate than a newly measured changing target. Closed-loop tests must therefore include both earlier rescues and SAC successes.

The new V13 FIX05 trace lasts long enough for its target genuinely to stop. A weak-braking versus nominal-fast-braking audit finds no weak-pass/fast-fail checked plans, so braking pessimism is not established as the cause. A fresh raw track ID later appears, and at decision 34 its fitted heading is missing; the inherited fallback substitutes noisy velocity direction, producing a roughly 32-degree orientation jump and one isolated passing plan. Keeping its last fitted heading instead rejects that same plan. Centre error remains material, and heading-only tightening has no demonstrated rescue. This is a remaining model/association issue, not evidence that the target continues at constant velocity forever.


## Full V14 and prefix comparisons (237 cumulative new runs complete)

The isolated shorter-prefix option is not a net improvement: its two disjoint16-case slices yield17/32 goals, two preserved SAC successes gained but three earlier V10 rescues lost. Reports/prefix32_pieces validates both pieces and explicitly documents their unequal archived inventories (an unrelated, unused development generator was added between slices); all shared sources and canonical scene identities match.

V14 with default prefix search disabled reaches22/32 goals:11 rescued SAC failures and five broken SAC successes. Its four diagnostic cases plus28 remaining cases are disjoint. V14 with broad prefix search enabled reaches21/32 goals. All six successful controls are retained. Neither configuration satisfies success preservation or100percent safety. These are enriched development comparisons, not full-test-set success rates. The evaluated source archives remain authoritative.

The strongest local success-preservation evidence is BAS-CR-RE-072: V13 makes28 interventions, default V14 makes one, and V14 with prefix search makes zero. The last configuration exactly reproduces all49 recorded SAC-only commands, policy actions and pre/post [x,y,heading,surge] states. This parity statement does not cover unrecorded sensor observations or internal plant state. See audits/v14_bas_mechanism.

V15 is the conditional-prefix experiment. Extra search is considered only when the current issued override explicitly fails V10's paired full-plan check, or when a passing override has less than the existing trigger margin and belongs to the current ordinary candidate pool. In the first case require hard feasibility with nonnegative clearance; in the second use the parent's exact existing candidate floor, clamped at zero. Adequate currently passing overrides, unmodelled commands, missing evidence and separate search/brake branches keep the parent result. No new numerical margin is fitted to case outcomes. The hook's default retains V11/V14's original0.15m search; exact candidate-floor diagnostics are added without changing V6 selection. V15 subsequently completed23/32 goals:11 rescues and four broken SAC successes. V16 subsequently completed25/32:11 rescues and two broken SAC successes; its measured motion-axis geometry recovered BO04 and CH-HO-CV-073 with no V15 goal lost. See the V16 plan and verified cohort report.

Inspiration remains first-action feasibility filtering in Wabersich and Zeilinger, linked above. Conditional search and floor consistency are engineering choices, not that paper's uncertainty bound, terminal invariant set or proof. A currently passing plan can still become unsafe because predictions are wrong, target motion changes or a later controller decision replaces it.

Saved prediction audits further separate the remaining limits. BAS-NU-CV-070's first override follows a predicted static-point collision on the actual successful SAC command continuation; boundary clearance is positive there. Correcting the initial body state barely changes its2.5s yaw error, and finer integration does not resolve it. The physical plant uses randomized parameters while the predictor uses the nominal identified hull; captured data do not include true actuator state, so actuator-vs-dynamics causes cannot yet be isolated. Separately, CH-HO-CV-073's target centroid/axis errors reject even the actual successful SAC continuation; a motion-axis known-shape completion greatly reduces current geometric bias in saved frames. This motivates a geometry experiment, not a claim of closed-loop rescue.
