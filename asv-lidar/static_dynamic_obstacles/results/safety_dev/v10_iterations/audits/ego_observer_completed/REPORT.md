# Saved own-state observer accuracy

The model observer improves surge and sway, but worsens yaw-rate estimation in most completed episode versions. This supports a **yaw-specific** measurement/prior investigation; replacing the complete ego estimate with raw measurements would discard measured benefits in both speed components.

The audit uses 50 completed episode versions across 32 distinct development cases: 2,837 pre-decision frames, all fresh. A committed episode result and exact matching trace length are required. Unfinished traces are excluded without reading. Every raw reading matches the captured held sensor input, and every truth value matches the recorded pre-decision state and timestamp. Truth is used only for offline scoring. No simulator, controller, or policy is invoked.

| Component | Raw MAE | Filtered MAE | Raw RMSE | Filtered RMSE | Versions where filtering improves RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Surge u, m/s | 0.03945 | 0.01888 | 0.04987 | 0.02460 | 50/50 |
| Sway v, m/s | 0.03973 | 0.01664 | 0.04965 | 0.02137 | 50/50 |
| Yaw rate, deg/s | 0.80909 | 1.23261 | 1.01315 | 1.82227 | 10/50 |

These pooled values weight longer episodes more heavily. Equal-episode mean MAEs agree in direction: surge 0.03969 to 0.01889 m/s, sway 0.03990 to 0.01665 m/s, and yaw 0.80287 to 1.20177 deg/s. The maximum observed yaw error rises from 3.6842 deg/s raw to 10.0496 deg/s filtered. Overall signed yaw bias is near zero because signed errors cancel; that does not make large state-dependent errors negligible.

| Completed cohort | Frames | Raw yaw RMSE | Filtered yaw RMSE |
| --- | ---: | ---: | ---: |
| missing_target_diagnostics, V8 | 56 | 1.0442 | 1.6807 |
| persistence_three, V11 | 195 | 0.9875 | 2.1735 |
| consistent_track_probe8, V12/V13 | 618 | 1.0249 | 1.9390 |
| persistent_preference_probe4, V14 | 252 | 1.0319 | 2.3113 |
| persistent_prefix32, V14 plus prefix search | 1,716 | 1.0079 | 1.6500 |

The cases repeat across controller variants and their trajectories differ. These rows are descriptive cohorts, not independent trials or a causal comparison of controller quality. `episodes.json` identifies each run/attempt/mode/case/seed, constructor options, source versions, outcome, trace hash, and all per-component MAE/RMSE/bias. `frames.csv` permits case-level inspection without rereading the large sensor traces.

## Prior contribution and the BAS-NU example

All archived observers share source SHA256 `9d5feb13e0b50982c76bb3b0c4af5afea7cef5fa0a2e35e41a23cbced141cb1f` and measurement weight 0.3. The prior is **not directly logged**. For fresh frames after initialization its value can be algebraically reconstructed from the implemented correction:

`prior = (filtered - 0.3 * raw) / 0.7`.

Across the 2,787 frames with a preceding prior, reconstructed yaw-prior MAE is 1.7063 deg/s and RMSE is 2.5622 deg/s. This is not independent validation of a newly simulated model prediction; it describes the prior contribution already present in the recorded filtered state.

At BAS-NU-CV-070 decision 10 in `persistent_prefix32`:

| Yaw quantity | deg/s | Error relative to current truth |
| --- | ---: | ---: |
| Recorded truth, scoring only | -10.22327 | 0 |
| Raw sensor | -11.16068 | -0.93741 |
| Filtered estimate | -6.85075 | +3.37251 |
| Reconstructed prior | -5.00364 | +5.21963 |

Here the existing 70% prior contribution pulls an approximately correct fresh yaw measurement toward a substantially incorrect prediction. This instance is consistent with the aggregate yaw result.

All included manifests record vessel-randomization scale 1.0, speed noise 0.05 m/s, yaw noise 1 deg/s, and pose-stale probability zero. The observer predicts with nominal identified parameters. The audit does **not** identify each randomized vessel, prove that randomization alone causes the error, or calibrate any parameter from these outcomes. Actuator-history approximation, unmodelled response, and model discretization can also contribute.

The evidence justifies a separately tested yaw-only fresh-measurement or innovation-weight hypothesis while retaining the present u/v estimator and stale-frame handling. It establishes neither a new gain nor a collision/goal improvement. Eight-second trajectory errors and the consequences of noisier control decisions still require separate validation. No adaptive controller was developed here.

Reproduce with a fresh tag:

```powershell
python -B tools/diagnostics/safety/audit_ego_observer.py --tag NEW_EGO_AUDIT
```

The command reads completed saved traces only and refuses to overwrite an existing output tag.
