# Continuous-objective finite model bank: saved development audit

**Do not integrate this model-selection rule. Stop this bank family without further window or weight tuning.** A continuous training objective helps the small V16 probe but still degrades the larger V15 cohort, including substantial individual failures. This does not change V18/V19 or provide new episode evidence.

## Fixed comparison

This is one prespecified objective ablation of the [prior one-step bank audit](../bootstrap_model_bank_cohort/REPORT.md). The 30 complete parameter vectors, raw sensor-noise weights, first ten transitions, candidate-specific actuator history, exclusions and held validation are identical. Select once before decision 11. The only predictor change during training is to initialize once from decision 1's raw measured body state, carry dynamics continuously through commands 1?10, and score the ten measured endpoints 2?11. Intermediate measurements never reset the training state. Decision 11's command and later data do not enter selection.

Selection is an engineering prediction-error calculation, related to [Ljung (2002), Section 1](https://doi.org/10.1007/BF01211648). It is not a posterior, IMM filter, robust envelope or physical-parameter identification guarantee. Initial measurement error can propagate throughout this continuous training objective. A five-second past training segment also differs from an eight-second future use; that limitation is reported, not addressed by outcome-guided window changes.

The saved cohort contains 37 episode versions: V15's 32 cases and V16's five-case probe. Two V15 cases have early braking and remain excluded, leaving 30 and five selected models. Repeated scenarios across versions are not independent coverage. Validation starts afresh at decision 11 with its recorded raw or unchanged filtered body state and the candidate-specific command history. The selected physical vector stays fixed. Future issued commands condition these offline forecasts; they are not known online during selection. Continuous validation stops at eight seconds, the first braking command or the end of the available trace. Only fresh observed endpoints are scored.

## Results

Values are means of per-episode RMSE. Heading and position rows below use the existing filtered initial state. The prior-objective and new-objective columns use exactly the same cases and endpoints.

| Cohort / metric | N | Identified | Prior one-step objective | Continuous objective | New improved / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| V15 held one-step yaw, deg/s | 30 | 3.47137 | 2.79389 | 2.90685 | 22 / 8 |
| V15 available continuous heading, deg | 29 | 3.20788 | 4.39141 | 3.97369 | 13 / 16 |
| V15 full eight-second heading, deg | 20 | 4.02237 | 4.68558 | 4.81525 | 10 / 10 |
| V15 full eight-second position, m | 20 | 0.18559 | 0.17528 | 0.21165 | 8 / 12 |
| V16 held one-step yaw, deg/s | 5 | 4.49384 | 2.92938 | 3.29292 | 4 / 1 |
| V16 available continuous heading, deg | 5 | 3.70366 | 5.40524 | 2.19402 | 4 / 1 |
| V16 full eight-second heading, deg | 4 | 4.24464 | 5.44625 | 2.40052 | 3 / 1 |
| V16 full eight-second position, m | 4 | 0.23845 | 0.18678 | 0.18460 | 3 / 1 |

V15's full-eight-second filtered heading worsens in 10/20 cases and position in 12/20. Across its available continuous horizons, heading worsens in 16/29 and position in 16/29. V15 CH-CR-CV-073 starts braking at decision 11, so it has no continuous nonbraking forecast; this accounts for 30 selected versus 29 available continuous results.

V16's full-eight-second heading and position improve in 3/4 cases. With a raw initial state, all four V16 full-eight-second heading forecasts improve; V15 raw heading still worsens in 12/20. Consequently the mixed conclusion is not solely a choice of filtered initialization.

A severe V15 degradation is P2-L3-CRS-FIX-05: filtered eight-second heading RMSE rises from 1.5812 to 15.7497 degrees. CH-HO-CV-043 rises from 0.8459 to 8.6907, and CH-CR-CV-015 from 2.4181 to 9.0522. BAS-NU again selects bootstrap vector 01, so its local improvement is unchanged from the prior audit and is insufficient evidence for integration.

The new objective changes selected vectors in 20/30 V15 and 5/5 V16 eligible exposures. This demonstrates sensitivity to the objective. It does **not** establish objective mismatch as the main cause of failed forecasts: changing that objective alone does not produce reliable cross-case predictions. Limited excitation, noisy initial conditions, finite-bank coverage and uncertain plant/actuator parameters remain unresolved explanations. No revised window, noise weight, bank subset or exception is chosen from these results.

## Verification and reproduction

Four pure selector tests passed: a continuous forecast is initialized once, endpoints align correctly, future command 11 is unused, intermediate measured states only affect residuals, inputs remain unchanged and the ten-transition window is enforced. A second agent independently reviewed timing and data separation without finding a blocker. Runtime assertions also retain the prior identified-model history/rollout parity checks.

[comparison.json](comparison.json) verifies that both audits use identical cases, exclusions, seeds, scene digests, training steps and held validation endpoints. Identified and unselected bootstrap-mean validation values are exactly equal across the two audits. [audit.json](audit.json) records every candidate score, chosen vector, forecast error, input hash and numerical thread cap. [reproducer_evaluated_bytes.py](reproducer_evaluated_bytes.py) preserves the executed script.

```powershell
python -B tools/diagnostics/safety/audit_bootstrap_continuous_bank.py --tag NEW_TAG
```

The diagnostic used one numerical CPU thread, no simulator construction/reset/step, no policy evaluation and no new episode. No test-set-v3 data was read. The source-only candidates and their active evaluations were untouched.
