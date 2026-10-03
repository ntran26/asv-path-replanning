# Frozen physical-model bank: saved development audit

**This model-selection rule is not supported for controller integration.** It improves BAS-NU and many measured one-step predictions, but worsens continuous heading forecasts across cases. No controller, margin, sensor weighting, selection window or episode was changed after these results.

The audit uses 37 completed saved episode versions: V15's 32-case cohort and V16's five-case perception probe. Repeated scenarios across versions are separate exposures, not independent scenario coverage. Exactly two V15 traces are excluded because their first ten transitions include braking: P2-L2-CRP-VAR-16 and P2-L2-CRP-VAR-02. The other 30 V15 and five V16 traces support model selection. No simulator construction, reset, step, policy evaluation or new episode occurs.

## Fixed method

The 30 candidates are the existing identified vector, the arithmetic mean of the 28 existing whole bootstrap vectors, and those 28 whole vectors. Whole-vector selection preserves each candidate's parameter relationships; it does not fit independent coefficients. This finite bank need not contain the randomized plant, which may be blended and perturbed.

For each candidate, reconstruct its own estimated servo and transport-delay queue from the empty-queue/zero-servo episode reset and the actual issued commands. Keep the existing predictor timestep of 0.125 seconds, RK4 dynamics, state clipping and rounded-delay convention. No actual plant servo, hidden parameters or true state is read.

Select once, before decision 11, by minimizing the sum of squared prediction errors on measured transitions 1–10. Each transition begins from its raw measured u/v/r; its endpoint is the next raw measurement. Errors are divided by the recorded sensor scales: 0.05 m/s for u and v, and one degree/s for yaw. These scales are fixed, not fitted. Ties retain the earliest candidate, with identified first. No target, future observation, future command, scenario feature or outcome enters selection.

The objective is an engineering use of prediction-error estimation, related to [Ljung (2002), Section 1](https://doi.org/10.1007/BF01211648). It is not an interacting multiple-model filter, calibrated posterior, safety envelope or physical-parameter identification guarantee. Both measured endpoints are noisy and consecutive residuals are correlated.

The selected vector stays frozen. Held one-step validation uses fresh nonbraking transitions starting at decisions 11–26, with each candidate's causally advanced actuator history. Continuous validation starts once at decision 11 and carries state and actuator history without measurement resets. It ends at eight seconds, the first braking command, or the last available pre-decision endpoint. Future commands are conditional validation inputs, not information available during selection. Only fresh measured endpoints are scored. Raw and unchanged filtered initializations are reported separately.

The identified candidate's reconstructed servo/FIFO matches the captured original predictor history, and its rollout matches the original V3 rollout to 1e-13. All required model-source hashes match the evaluated manifests. These checks isolate parameter selection from a changed integrator or command-history implementation.

## Results

Numbers below are means of per-episode RMSE, with each row using its own explicit denominator. Heading rows use the existing filtered initial state.

| Cohort / metric | N | Identified | Selected bank | Improved / worse |
| --- | ---: | ---: | ---: | ---: |
| V15 held one-step yaw, deg/s | 30 | 3.4714 | 2.7939 | 23 / 7 |
| V15 continuous heading, available horizon, deg | 29 | 3.2079 | 4.3914 | 11 / 18 |
| V15 continuous heading, full eight seconds, deg | 20 | 4.0224 | 4.6856 | 8 / 12 |
| V16 held one-step yaw, deg/s | 5 | 4.4938 | 2.9294 | 5 / 0 |
| V16 continuous heading, available horizon, deg | 5 | 3.7037 | 5.4052 | 2 / 3 |
| V16 continuous heading, full eight seconds, deg | 4 | 4.2446 | 5.4462 | 2 / 2 |

V15 CH-CR-CV-073 starts braking at decision 11, so it has one-step observations later but zero continuous nonbraking horizon. This explains 30 selected versus 29 continuously scored traces. The earlier yaw-gain audit also excluded one nonpositive fitted gain; this bank has no fitted gain and therefore has a different eligible count. Do not compare those aggregate denominators as identical samples.

For surge and sway, selected-bank V15 held one-step mean RMSE is 0.06763 and 0.06853 m/s, versus identified 0.06754 and 0.06493 m/s. Sway worsens in 28/30 cases. V16 sway also worsens in 4/5. Thus the yaw one-step improvement does not describe all estimated body components.

Full-eight-second filtered position RMSE improves in the mean (V15 0.18559 to 0.17528 m; V16 0.23845 to 0.18678 m), but V15 position prediction worsens in 13/20 cases. Examples of substantial heading degradation are V15 P2-L1-CRP-VAR-12 (1.0883 to 9.9036 degrees), V15 BAS-HO-NC-071 (1.3616 to 9.5142), and V16 CH-HO-CV-073 (1.4841 to 12.1237).

The unselected fixed bootstrap mean is included as a control, not chosen from outcomes. Full-eight-second filtered heading means are 3.7655 degrees for V15 and 4.0126 for V16, versus identified 4.0224 and 4.2446. It still worsens 9/20 V15 and 1/4 V16 cases. These mixed sensitivities do not establish a safe replacement.

## BAS-NU local improvement and limits

Using only its first ten transitions, BAS-NU selects bootstrap vector 01. Held one-step yaw RMSE improves from 4.5988 to 2.5317 degrees/s. Continuous filtered heading RMSE improves from 8.3146 to 4.7103 degrees and position RMSE from 0.39138 to 0.16209 m. The selected vector also improves surge and sway on this local case. The wider results show why that example cannot justify integration.

The scoring deliberately does not ask which model makes the successful SAC continuation pass, and does not select a model by collision outcome. The bank has not been tested as a controller and has no episode-success evidence. Limited early excitation, errors in measured predictor inputs, objective/horizon mismatch and imperfect bank coverage remain plausible explanations; this audit does not isolate their respective causes. No revised weights, longer window or threshold are selected from these results.

## Reproduction and provenance

From the project directory, with a new output tag:

```powershell
python -B tools/diagnostics/safety/audit_bootstrap_model_bank.py --tag NEW_TAG
```

The optional `--case TS2:BAS-NU-CV-070` limits the diagnostic to that existing saved case. [audit.json](audit.json) preserves all candidate vectors, training scores, selected identities, per-endpoint validation errors, exclusions, source/input hashes and baseline parity checks. [reproducer_evaluated_bytes.py](reproducer_evaluated_bytes.py) preserves the executed script. Its stripped measurement/JSON helpers come from `tools/diagnostics/safety/audit_causal_yaw_response.py`, whose hash is also recorded. No test-set-v3 data is read.
