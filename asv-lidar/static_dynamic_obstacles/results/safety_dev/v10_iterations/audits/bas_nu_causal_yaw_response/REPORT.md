# Causal yaw-response audit at BAS-NU decision 11

Ten measured transitions available at decision 11 support an effective yaw-response correction on this case. They do not identify the physical source of the mismatch or establish a safe controller change. No environment, policy or new episode was run.

The fitted model changes only total yaw acceleration, `dr_corrected = g * dr_identified + b`, retaining the identified body dynamics, numerical scheme and copied predictor rudder/delay history. With `g=1,b=0`, each integration step is asserted identical to the original model to 1e-13. Fitting uses fresh measured u/v/r at decisions 1–11 and issued commands 1–10. Future observations, target geometry, outcomes and simulator truth do not enter the objective. The method is a small prediction-error parameterization motivated by [Ljung, Section 1, Eqs. 1–2](https://liu.diva-portal.org/smash/get/diva2:316694/FULLTEXT01.pdf), published in 2002 as [Prediction Error Estimation Methods](https://doi.org/10.1007/BF01211648).

| Model | Fitted gain | Bias deg/s² | Past next-yaw RMSE deg/s | Frozen forward next-yaw RMSE deg/s |
| --- | ---: | ---: | ---: | ---: |
| Identified | 1 | 0 | 4.2014 | 4.5988 |
| Gain only | 0.33585 | 0 | 2.0542 | 2.5896 |
| Bias only | 1 | -2.2159 | 4.1136 | 4.6723 |
| Gain and bias | 0.34010 | -1.2443 | 1.9735 | 2.7406 |

Forward next-yaw predictions cover issued commands 11–26, resetting each forecast from that decision's raw measured state and captured estimated actuators. All fit parameters remain frozen. The gain-only leave-one-transition range is 0.27264–0.38910; these overlapping-data ranges measure sensitivity, not a confidence interval. The gain-and-bias Jacobian has two nonzero singular values, but rank does not resolve unobserved actuator state or errors in measured u/v/r.

Continuous eight-second forecasts on the recorded V15 commands also improve overall heading error: raw-state heading RMSE falls from 7.8383 to 2.2755 degrees; unchanged filtered-state initialization falls from 8.3146 to 2.9196 degrees. This is separate from one-step validation: the continuous forecast carries its model state and actuator history without measurement resets. It remains imperfect: gain-only filtered-state heading error at eight seconds is -6.90 degrees, compared with -0.86 degrees for the original model at that endpoint. Improving the average does not improve every future time.

The successful recorded SAC continuation is an additional conditional forecast, not an online-known command sequence. With the original filtered initial state, gain-only reduces its 2.5-second heading error from 11.7889 to 0.5514 degrees and improves full predicted clearance from -0.154723 to -0.007420 m, which still fails. Raw-state initialization plus gain gives +0.021839 m, combining two changes. Gain-and-bias gives +0.013360 m with filtered initialization but is worse on held-forward one-step error than gain-only. No method is selected because its clearance becomes positive, and no closed-loop rescue is demonstrated.

The first six controls are almost full negative rudder; later past controls include a reversal. The gain is therefore supported by limited local excitation. It scales total yaw acceleration, including damping, drift and rudder response; it is not a recovered rudder gain. The actual plant servo and delay buffer are unavailable. Raw yaw noise occurs in both transition endpoints, other measured states are noisy, and consecutive residuals are correlated. Whole-case saved-data validation is required before production integration. No threshold change is supported by this audit.

Reproduce from the project directory using a new output tag:

```powershell
python -B tools/diagnostics/safety/audit_causal_yaw_response.py --tag NEW_TAG
```

Omit `--tag` for read-only output. `audit.json` contains every fit, measured forward residual, conditional SAC-future score and input/source hash. `reproducer_evaluated_bytes.py` preserves the exact evaluated script (SHA-256 `73db590aacadb0d3c22aea731ef59e41aa2d03e17f997290432821deac1dd378`).
