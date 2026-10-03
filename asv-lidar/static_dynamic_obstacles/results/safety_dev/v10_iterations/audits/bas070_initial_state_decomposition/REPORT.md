# BAS070 prediction-error decomposition

Saved-only sensitivity of the successful SAC command continuation at the common decision-10 state. No environment or new episode. The default forecast remains entirely onboard. True initial body-state substitutions below are explicitly scoring-only; they never select a controller command.

| Initial quantity | Filtered onboard | Raw measured | True, for scoring |
| --- | ---: | ---: | ---: |
| Surge u | 0.580205 m/s | 0.532538 m/s | 0.553576 m/s |
| Sway v | 0.145538 m/s | 0.155699 m/s | 0.144126 m/s |
| Yaw r | -6.850752 deg/s | -11.160680 deg/s | -10.223267 deg/s |
| Heading | 303.928289 deg | ? | 304.190488 deg |

| Forecast initialization / integration | Heading error at 0.5 s | Heading error at 2.5 s | Position error at 2.5 s | Full clearance |
| --- | ---: | ---: | ---: | ---: |
| Original onboard, dt .125 s | 2.4398 deg | **15.0211 deg** | 0.17265 m | -0.20811 m |
| Raw measured u/v/r | 0.8127 deg | 14.0635 deg | 0.15477 m | -0.24580 m |
| True initial yaw only | 1.1535 deg | 14.0925 deg | 0.14754 m | -0.17908 m |
| True initial u/v/r | 1.0825 deg | **13.6140 deg** | 0.13114 m | -0.19511 m |
| True initial pose and u/v/r | 1.3447 deg | 13.8762 deg | 0.14631 m | -0.21706 m |
| Onboard, dt .025 s | 2.4357 deg | **15.0534 deg** | 0.17277 m | -0.20874 m |

Observer yaw lag contributes to the first half-second error but accounts for only about 1.41 degrees of the 15.02-degree error at 2.5 seconds. Correcting all initial ego quantities does not remove the mismatch or restore predicted feasibility. Finer numerical integration changes that error by only 0.0323 degrees and does not explain it either.

The finer-integration branch retains the same identified dynamics, copied servo, piecewise captured command history, delay duration, and recorded future commands. At dt .125 it reproduces the original rollout positions/headings to 1e-12. Thus this is a controlled numerical sensitivity, not a replacement plant simulation.

Captured predictor actuator state is servo 0.384002 rad and six pending delay samples `[-.07324825,-.07324825,-.00652974,-.00652974,-.00652974,-.00652974]`. **True plant servo, actual delay buffer and sampled model parameters are absent from the trace.** The residual cannot be separated into actual actuator-history error versus identified body/actuator dynamics mismatch from these records. Do not label it an integration bug or infer hidden plant parameters.

Next model work should validate multi-step yaw response against fresh measured onboard yaw and issued commands, with actuator history carried continuously and whole-case holdouts. Raw measurement substitution alone is not supported as the fix here. Neither better prediction on this sequence nor a lower acceptance buffer demonstrates a closed-loop rescue.

Reproduce:

```powershell
python -B tools/diagnostics/safety/audit_saved_policy_initial_state.py --run-dir results/safety_dev/v10_iterations/persistent_prefix32 --trace 014_v14.jsonl --tag NEW_TAG
```

Omit `--tag` for read-only output. `audit.json` contains all ten sensitivities, errors every half-second, exact input/model hashes and the unavailable-actuator limitation. `reproducer_evaluated_bytes.py` preserves the evaluated script bytes.
