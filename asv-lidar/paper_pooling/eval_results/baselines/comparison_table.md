# Comparison on the frozen 500-episode evaluation set

Point estimate with a stratified bootstrap 95 % CI (stratified by obstacle count, pooled over runs, 5000 resamples).

**What each interval covers is not the same, and the rows are not interchangeable.** `SAC (published)` is a single checkpoint, so its interval reflects episode/layout variance only. The `SAC (retrained)` and `PPO` rows pool three from-scratch training seeds, so theirs reflect seed *and* episode variance. `LOS+APF` pools three independent 250-configuration random searches under different search seeds: the controller itself is deterministic -- re-running it reproduces its CSV byte for byte -- so its interval reflects tuning-procedure variance, which is the analogue of a training seed for a non-learned method.

`SAC (published)` and `SAC (retrained)` are different objects and are deliberately not pooled: the published checkpoint carries a long history of resumed, hand-staged training that the 1M-step retrains do not.

Rate metrics (marked *) are per-episode 0/1 outcomes and are reported as **means**, which is what a success or collision rate is. The remaining, continuous metrics are reported as **IQM** (interquartile mean). IQM is not used for the rates because it is degenerate on a binary variable -- the middle 50 % of a mostly-successful set is all ones, so every method would read exactly 1.000.

| Metric | SAC (published) | SAC (retrained, final) | SAC (retrained, best) | PPO (final @1M) | PPO (best ckpt) | LOS+APF (tuned) |
|---|---|---|---|---|---|---|
| Success rate * | 0.950 [0.930, 0.968] | 0.816 [0.797, 0.835] | 0.876 [0.859, 0.892] | 0.272 [0.249, 0.294] | 0.910 [0.895, 0.924] | 0.960 [0.950, 0.969] |
| Obstacle collision rate * | 0.038 [0.022, 0.056] | 0.158 [0.140, 0.176] | 0.079 [0.065, 0.092] | 0.199 [0.179, 0.219] | 0.078 [0.065, 0.091] | 0.033 [0.024, 0.042] |
| Border collision rate * | 0.012 [0.004, 0.022] | 0.026 [0.018, 0.034] | 0.045 [0.035, 0.056] | 0.511 [0.487, 0.537] | 0.012 [0.007, 0.017] | 0.007 [0.003, 0.012] |
| Timeout rate * | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.018 [0.011, 0.025] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Rudder saturation fraction * | 0.098 [0.090, 0.107] | 0.042 [0.039, 0.044] | 0.009 [0.008, 0.010] | 0.527 [0.509, 0.544] | 0.115 [0.108, 0.121] | 0.098 [0.091, 0.105] |
| RMS cross-track error (m) | 0.908 [0.879, 0.936] | 0.874 [0.848, 0.900] | 0.955 [0.933, 0.977] | 1.572 [1.508, 1.638] | 1.264 [1.240, 1.288] | 1.328 [1.312, 1.344] |
| Min obstacle clearance (m) | 0.233 [0.217, 0.249] | 0.231 [0.213, 0.248] | 0.299 [0.283, 0.315] | 0.518 [0.302, 0.592] | 0.505 [0.478, 0.534] | 0.768 [0.749, 0.788] |
| Min border clearance, all walls (m) | 0.989 [0.989, 0.989] | 0.989 [0.989, 0.989] | 0.989 [0.989, 0.989] | 0.425 [0.376, 0.473] | 0.989 [0.989, 0.989] | 0.989 [0.989, 0.989] |
| Min border clearance, lateral (m) | 2.514 [2.433, 2.595] | 2.477 [2.420, 2.530] | 2.299 [2.243, 2.352] | 0.722 [0.617, 0.832] | 2.447 [2.406, 2.490] | 2.524 [2.493, 2.556] |
| Control effort (int. sq. rudder cmd) | 9.846 [9.580, 10.122] | 7.869 [7.727, 8.005] | 7.266 [7.153, 7.377] | 12.459 [11.845, 13.082] | 5.219 [4.950, 5.501] | 5.845 [5.632, 6.062] |
| Mean abs. rudder rate (deg/s) | 74.91 [73.40, 76.29] | 69.20 [68.39, 70.01] | 51.84 [50.78, 52.86] | 6.68 [6.12, 7.28] | 11.05 [10.88, 11.22] | 8.96 [8.83, 9.10] |
| Completion time (s) | 20.9 [20.7, 21.0] | 19.6 [19.5, 19.8] | 20.6 [20.4, 20.7] | 19.5 [19.2, 19.8] | 21.6 [21.3, 22.0] | 27.2 [27.0, 27.4] |

| | SAC (published) | SAC (retrained, final) | SAC (retrained, best) | PPO (final @1M) | PPO (best ckpt) | LOS+APF (tuned) |
|---|---|---|---|---|---|---|
| Runs | sac_1M | sac_seed0_final, sac_seed1_final, sac_seed2_final | sac_seed0_best, sac_seed1_best, sac_seed2_best | ppo_seed0_final, ppo_seed1_final, ppo_seed2_final | ppo_seed0_best, ppo_seed1_best, ppo_seed2_best | los_apf_s1, los_apf_s2, los_apf_s3 |
| Episodes | 500 | 1500 | 1500 | 1500 | 1500 | 1500 |
