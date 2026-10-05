# V16: measured target geometry and policy-success preservation

Primary benchmark update: test set v3 is now the main evaluation set. See [the current protocol](../../SAFETY_EVALUATION_PROTOCOL.md). Results below remain their original v2/DV3 development comparisons.

Status: experimental. The 32-case development comparison is complete; the separate40-case paired challenge is complete (V16 25 goals versus freshSAC21, eight rescues and four broken SAC successes). Neither collision avoidance nor preservation of every SAC success has been achieved. Use SAC baseline 3's kept best 3M checkpoint throughout.

## Measured result

On the same enriched 32 cases, SAC alone reaches 16 goals, V9 16, V10 18, V11 18, V14 22, V15 23 and default V16 25. V16 rescues 11 SAC failures and breaks two SAC successes: BAS-HO-NC-059 and BAS-NU-CV-070. It preserves all six successful controls. There are four target contacts, two obstacle contacts and one boundary contact; no timeout. These deliberately difficult cases are not a population success-rate estimate.

V16 gains DV3-BO-CV-04 and CH-HO-CV-073 over V15 with no lost V15 goal. The exact evaluated sources, checkpoint, scene identities and per-decision records are retained in `results/safety_dev/v10_iterations/motion_axis_probe5` and `motion_axis_remaining27`. Their verified disjoint cohort report is `reports/motion_axis32` in that folder.

## What changed

V15 admits an extra current-policy-prefix search when the current override fails its complete paired check, or when an ordinary candidate passes below the existing trigger margin. It uses the parent's exact selection floor, clamped at zero, for the latter branch. Missing evidence and already adequate overrides retain the parent decision. The common V11 hook preserves its old default behavior; V6 additions only expose the existing selection branch, floor and best margin. Audits verify the V6 behavior AST is unchanged apart from these diagnostics.

This builds on the first-input objective in Wabersich and Zeilinger, [A predictive safety filter for learning-based control of constrained nonlinear dynamical systems](https://arxiv.org/html/1812.05506v4#S4.SS1), and the finite cross-entropy search inspired by Zheng et al., [Safe Learning-based Gradient-free Model Predictive Control Based on Cross-entropy Method](https://arxiv.org/abs/2102.12124v3). The conditional rule is an engineering adaptation, without their terminal safe set, calibrated uncertainty model or safety proof.

V16 adds `src/safety_motion_axis.py`, wrapped by `src/safety_v16.py`. A partially observed moving hull can have a badly biased cluster centroid and a poorly fitted axis. For an existing, freshly observed, confirmed raw track, the wrapper uses measured travel direction to complete the known hull length away from the sensor-facing observed end. It requires coherent endpoint translation in at least three actual observations, recorded observation times and sensor origins, adequate motion evidence, a plausible observed beam and partial length, and consistency between fitted and measured motion. It never infers elapsed time from track-ID or scan-serial gaps.

The correction changes only the geometry of the matching existing base view. It preserves IDs, ordering, measured velocity, static points, persistent hypotheses and all inherited collision checks. It does not coast its correction when the evidence disappears. Disabling `motion_axis_geometry` reproduces V15. No scenario ID, saved outcome, future policy action or hidden target state is available to this rule.

Inspiration: Granstrom, Baum and Reuter, [Extended Object Tracking: Introduction, Overview and Applications](https://arxiv.org/abs/1604.00970), for spatial measurement and extent modelling; Zhang et al., [Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners](https://publications.ri.cmu.edu/efficient-l-shape-fitting-for-vehicle-detection-using-laser-scanners), for the project's underlying hull-fit representation. The visible-end completion and evidence gates are an engineering construction made here, not implementations of the survey's Bayesian estimators. Travel direction can differ from hull heading; partial occlusion can conceal the actual end. The gates do not establish an error bound.

## Evidence behind the change

At the first intervention in CH-HO-CV-073, the old target view rejected even the recorded successful SAC continuation. Replacing its centroid/axis by the observed motion-axis completion reduced current centre error from about 0.72 m to 0.01 m in the saved diagnostic. The new closed-loop V16 run then reached the goal. In DV3-BO-CV-04, the analogous centre error decreased from about 0.65 m to 0.05 m; V16 also reached the goal. Truth was used to score these errors, not to construct runtime estimates.

These mechanisms are documented under `audits/v14_ch_ho_cv073_mechanism` and `audits/bo04_v15_preservation`. The actual fresh episode results, rather than a successful offline counterfactual, determine the reported rescues.

## Remaining failure mechanisms

- BAS-NU-CV-070: V15/V16 preserve SAC at decision10, then intervene at decision11. The unchanged prefix search has no passing backup there, and the actual successful SAC continuation is falsely predicted to hit static geometry. Correcting the initial measured body state and reducing integration step size did not remove the future yaw/position mismatch. The simulator randomizes the plant while the forecaster uses nominal parameters; true actuator state was not recorded. See `audits/bas_nu_v15_gate` and `audits/bas070_initial_state_decomposition`.
- BAS-HO-NC-059: a hard-passing SAC backup is rejected for its smaller margin, while target behaviour changes sharply after the decision. Even a perfect current-state constant-velocity target model rejects the successful future SAC trajectory. The abrupt turn was observed on the V16 branch; future target states on the SAC-only branch were not recorded. Geometry-only correction is insufficient evidence of a fix. See the completed `audits/bas_ho_v16_first_divergence/REPORT.md`.
- The earlier DV3-BO-CV-10 start has contact within the first decision interval under all10,426 sampled constant first commands. This is evidence of limited recoverability, not a continuous-action impossibility proof. It is outside the current32-case cohort and does not establish that any remaining failure here is unavoidable. Original cases remain in their relevant denominators.

## Next checks and reproducibility

The separate `selection_broader40.json` was frozen before expanded-candidate outcomes and is disjoint from the pilot. Its 40 cases include every remaining known V8-broken SAC success, matched rescues, successful active/passive controls and failures. `v16_broader40_paired` completed fresh SAC and default V16 on all40 cached scenes. V16 preserves17/21 SAC successes; all10 successful controls remain goals, but boundary contacts increase1 to6. Report its strata and paired gains/losses separately from the 32-case development cohort.

The additional `v16_feasible_probe9` is an explicit existing-option ablation, `prefer_any_feasible_policy=True`. It checks both remaining broken SAC successes, both geometry rescues and the five cases previously lost by this option under V10. That earlier full ablation reached only 15/32 and was rejected. The probe completed3/9 goals versus defaultV16 6/9, with zero gains and three losses. Reject the option. Do not make it the default or combine its records with defaultV16.

All runs reserve attempts, archive source/settings/checkpoint/scene hashes, write individual results and decision traces, and refuse automatic retries or changed archived sources. Keep at most two evaluation processes and do not disturb training. Constants, checkpoints, Paper 2 and older result ledgers are unchanged. New evaluations are authorized; earlier saved-only restrictions and numerical run caps belong to completed phases.

NativeV16 dispatch was added after all390 then-planned runs completed. Environment, suite CLIs and counterfactual dispatch accept16; defaults are unchanged. See `v16_native_dispatch_source_audit.json`. The diagnostic runner explicitly installed and verified the exactV16 class throughout the earlier runs.
