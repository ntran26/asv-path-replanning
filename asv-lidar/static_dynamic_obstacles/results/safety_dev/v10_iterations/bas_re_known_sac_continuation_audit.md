# Saved successful-SAC continuation check: BAS-CR-RE-072

At decision8, V12/V13 first issue a different command from the saved successful SAC episode. All earlier issued commands, the recorded four-component pre-state, and the policy action match exactly; scene, seed, checkpoint and configuration identities match. The old SAC trace does not include sway, yaw rate or full plant actuator state.

The actual next16 SAC commands (decisions8?23) were passed through the existing predictor from the saved V13 onboard snapshot and pre-command actuator history. This is offline, noncausal diagnostic input, unavailable to an online controller. The sequence fails the current target check at3.375s, with minimum target margin?0.22557m at4s. Static and boundary margins remain positive (+0.65723m and+1.41032m). The rejection occurs within the prediction horizon, not only in its terminal extension.

Using SAC's recorded actual own positions at decision endpoints against the same onboard constant-velocity target still produces a negative target margin (?0.14977m). Own-model position error reaches0.717m at8s. Therefore, a larger search bank cannot certify this exact known-successful sequence under these unchanged checks. The evidence does not isolate target-centre bias from target-motion forecasting: this target is reactive, and its future true positions were not recorded in the old SAC trace. The filtered episode's later true target trajectory cannot substitute for that missing counterfactual.

See [the JSON audit](bas_re_known_sac_continuation_audit.json) for the actual command sequence, component margins, initial target measurement error, endpoint model errors and source/trace hashes. No environment was constructed, reset or stepped.
