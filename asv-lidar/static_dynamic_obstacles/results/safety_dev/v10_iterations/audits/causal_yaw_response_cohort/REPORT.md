# Cross-case causal yaw gain: do not integrate into the safety predictor

The BAS-only improvement does not generalize to continuous forecasts. The same fixed causal fitting procedure often improves the next measured yaw rate, but worsens the heading trajectory used for collision checks. These saved-data results do not justify production integration or a safety-margin relaxation. No new episodes, environment calls or policy calls were made.

The method was fixed before this extension: fit only `dr = g * dr_identified`, with no additive bias, gain bounds, clipping, alternative sample counts or outcome-dependent rules. Each fit uses exactly the first ten transitions, measured states at decisions 1–11, commands already issued at decisions 1–10, and copied estimated actuator history. The fit freezes at decision 11. All fitting and cross-case validation use onboard measurements; no simulator truth or future observations enter fitting. The method follows the prediction-error setup in [Ljung, Section 1, Eqs. 1–2](https://liu.diva-portal.org/smash/get/diva2:316694/FULLTEXT01.pdf), with this effective gain as the diagnostic model choice here.

| Saved run | Records | Positive valid fits | Exclusions |
| --- | ---: | ---: | --- |
| V15 `conditional_prefix32` | 32 | 29 | Two contain braking in the first ten transitions; one produces a negative gain |
| V16 `motion_axis_probe5` | 5 | 5 | None |

The excluded V15 braking cases are `P2-L2-CRP-VAR-16` and `P2-L2-CRP-VAR-02`, each with two braking commands in the training window. `CH-CR-CV-031` converges to gain -0.042324; it is reported as nonpositive and is not clipped or used for validation. There are no short-trace, stale-training-endpoint, solver-failure, nonfinite-fit or numerically zero-excitation exclusions in these records. V15 positive gains range from 0.13027 to 0.99513, median 0.40458; V16 positive gains range from 0.16193 to 0.99513, median 0.38062.

One-parameter Jacobian condition numbers are trivially one when nonzero and do not demonstrate useful identifiability. The JSON therefore also records absolute sensitivity and the implied scale from the existing one-degree/second gyro noise. The latter spans 0.0366–0.2257 gain units and ignores noisy u/v, correlated endpoint noise and unknown actuator error; it is not a confidence interval. For `BAS-HO-NC-071`, the fitted gain 0.1303 is smaller than that simple noise scale 0.1467. No new threshold is derived from these numbers.

**One-step validation:** freeze the past fit, then predict measured yaw at the next decision using that decision's measured initial body state and estimated actuator history. Score valid nonbraking, fresh-endpoint transitions in decisions 11–26. This provides 422 V15 and 76 V16 measured transitions. Means below average per-case RMSE, giving cases equal weight rather than treating correlated transitions as independent trials.

| Run | Cases improving / degrading | Mean original RMSE deg/s | Mean fitted RMSE deg/s |
| --- | ---: | ---: | ---: |
| V15 | 20 / 9 | 3.4557 | 3.2029 |
| V16 probe | 4 / 1 | 4.4938 | 3.2686 |

**Continuous validation:** initialize once at decision 11, carry predicted body state and estimated actuator history, and condition on the recorded future commands without later measurement corrections. Stop before the first braking command or at the available trace endpoint, up to eight seconds. No braking-response model is introduced. Each original/fitted pair uses exactly the same command sequence, time horizon and measurement endpoints. These forecasts score the recorded controller trajectory; they do not re-run control or establish that any changed action would succeed.

| Run / initialization | Cases improving / degrading in heading RMSE | Mean original heading RMSE deg | Mean fitted heading RMSE deg |
| --- | ---: | ---: | ---: |
| V15 / raw measured body state | 8 / 20 | 4.8362 | 6.9497 |
| V15 / unchanged filtered body state | 9 / 19 | 3.2609 | 5.9637 |
| V16 / raw measured body state | 2 / 3 | 6.1149 | 9.3389 |
| V16 / unchanged filtered body state | 2 / 3 | 3.7037 | 8.8996 |

V15 has 28 continuous forecasts: `CH-CR-CV-073` starts braking immediately, so its continuous horizon is zero and it is omitted from this table. Nine further V15 forecasts end early at braking or trace end. The deterioration also holds on the common **full eight-second subset**: V15 filtered-state heading RMSE increases from 4.1433 to 6.9426 degrees, with 12 of 19 cases worsening; V16 increases from 4.2446 to 9.3553 degrees, with two of four worsening. Thus the conclusion is not explained by mixing short and long horizons.

The V16 `CH-HO-CV-073` trace is a concrete counterexample: the fitted gain 0.16193 gives filtered-state heading RMSE 25.0696 degrees, versus 1.4841 degrees for the identified model. V15 `BAS-HO-NC-071` similarly worsens from 1.3616 to 19.5172 degrees. These failures coexist with the useful BAS-NU local improvement. A low one-step residual and a finite, positive fit are insufficient gates for changing the multi-step safety predictor.

The actual plant servo, delay buffer and sampled dynamics are unavailable. This procedure scales total yaw acceleration, including damping, drift and rudder effects; it cannot assign the mismatch to one physical parameter. No observer/predictor code is changed. Any later model-identification method would need validation of continuous multi-step trajectories, not just next-rate error, and explicit handling of latent actuator state and noisy measured initial states. This audit rejects this particular fixed-window gain method for release; it does not demonstrate that all causal identification methods fail.

The 37 episode versions represent the same selected development pool; the five V16 cases overlap V15 and are not five independent new scenarios. No population safety rate or closed-loop success change is estimated.

Reproduce with a new tag:

```powershell
python -B tools/diagnostics/safety/audit_causal_yaw_cohort.py --tag NEW_TAG
```

`audit.json` stores all per-case fits, exclusions, paired errors, forecast lengths, current script/helper hashes and every input hash. `reproducer_evaluated_bytes.py` and `helper_evaluated_bytes.py` preserve the evaluated diagnostic code. Original production sources and evaluated campaign artifacts are unchanged.
