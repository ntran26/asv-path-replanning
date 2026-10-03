# V17: fresh measured yaw ablation

Primary benchmark update: the user now designates test set v3 as the main evaluation set. See [the current protocol](SAFETY_EVALUATION_PROTOCOL.md). Results below remain their original v2/DV3 development comparisons.

Status: evaluated on nine targeted development cases; **no net improvement over V16**. Keep V16 as the stronger measured candidate. Neither version reaches the requested collision-free, success-preserving objective.

V17 reaches 6/9 goals, the same count as default V16. It rescues DV3-HO-VS-01 but loses P2-L2-HO-FIX-19 to a boundary contact. BAS-HO-NC-059 and BAS-NU-CV-070 remain broken SAC successes. The probe has one target contact and two boundary contacts. There is no evidence to expand or promote this option based on these results.

See `results/safety_dev/v10_iterations/reports/v17_fresh_yaw_probe9/report.md` for validated identities, exact sources, paired outcomes and per-decision traces. No case is removed and no retry is hidden.

## Method and citation

`src/safety_yaw_observer.py` tests a measurement-dominant yaw update: fresh frames use the supplied measured yaw directly, while surge/sway retain their existing correction and stale frames retain model prediction. It adds no sensor call or hidden-state access and changes no rollout dynamics or safety thresholds. Changed yaw can affect later surge/sway priors through the coupled model.

Related prior work is Luenberger (1971), *An introduction to observers*, Section II.B, Eq. (2.6), which combines model dynamics with measurement-error correction and discusses the response-speed/noise tradeoff. [DOI](https://doi.org/10.1109/TAC.1971.1099826), [original paper, p.597](https://lab.prd.vanderbilt.edu/taha/wp-content/uploads/sites/154/2017/10/Observers_Original_Paper.pdf). The fresh-frame gain-one choice here is an engineering ablation, not that paper's observer design, a Kalman filter, or a convergence/safety guarantee.

`SafetyFilterV17(fresh_yaw_measurement=False)` retains V16's exact observer instance. Enabled construction fails clearly if the inherited model observer is disabled, rather than silently changing its surge/sway algorithm. Initialization, validation, reset, stale updates and issued-command prediction retain their existing semantics.

The motivating saved audit found raw yaw-rate RMSE about1.01 degrees/s versus filtered1.82 across50 recorded episode versions, while filtering improved surge/sway. This component-level accuracy result did not establish better closed-loop control, as the nine-case result demonstrates.

## Rejected dynamics fit

A separate causal gain-only yaw-response fit used only the first10 past measured transitions and froze its parameters before future validation. It improved one-step errors on many cases but worsened continuous heading forecasts:12/19 full8-second V15 forecasts and2/4 V16 forecasts deteriorated. No fitted gain is used in V17. [Cross-case audit](../results/safety_dev/v10_iterations/audits/causal_yaw_response_cohort/REPORT.md).

## Validation and integration

The new observer and wrapper have23 passing tests, including fresh/stale/reset behaviour, disabled-option identity, coupled issued-command integration and no extra sensor/truth access. Independent review found no implementation blocker. The evaluated runner explicitly installed and verified V17; native environment, suite CLI and counterfactual selection were added only after the probe completed. That later integration is separately archived in `v17_native_dispatch_source_audit.json` and is not retroactively attributed to the evaluated source archive.

Existing defaults, constants, the SAC checkpoint, Paper 2 and other jobs remain unchanged. All399 new runs in this continuation are complete; there are no queued episode evaluations.
