# Comparison on the frozen 500-episode evaluation set

Point estimate with a stratified bootstrap 95 % CI (stratified by obstacle count, pooled over runs, 5000 resamples).

**What each interval covers is not the same, and the rows are not interchangeable.** `SAC (published)` is a single checkpoint, so its interval reflects episode/layout variance only. The `SAC (retrained)` and `PPO` rows pool three from-scratch training seeds, so theirs reflect seed *and* episode variance. `LOS+APF` pools three independent 250-configuration random searches under different search seeds: the controller itself is deterministic -- re-running it reproduces its CSV byte for byte -- so its interval reflects tuning-procedure variance, which is the analogue of a training seed for a non-learned method.

`SAC (published)` and `SAC (retrained)` are different objects and are deliberately not pooled: the published checkpoint carries a long history of resumed, hand-staged training that the 1M-step retrains do not.

Rate metrics (marked *) are per-episode 0/1 outcomes and are reported as **means**, which is what a success or collision rate is. The remaining, continuous metrics are reported as **IQM** (interquartile mean). IQM is not used for the rates because it is degenerate on a binary variable -- the middle 50 % of a mostly-successful set is all ones, so every method would read exactly 1.000.

| Metric | SAC (deployed) | SAC (retrained, 3 seeds) | PPO (3 seeds) | LOS+APF (3 searches) |
|---|---|---|---|---|
| Success rate * | 0.950 [0.930, 0.968] | 0.924 [0.911, 0.937] | 0.905 [0.890, 0.919] | 0.960 [0.950, 0.969] |
| Obstacle collision rate * | 0.038 [0.022, 0.056] | 0.047 [0.037, 0.058] | 0.079 [0.066, 0.093] | 0.033 [0.024, 0.042] |
| Border collision rate * | 0.012 [0.004, 0.022] | 0.029 [0.021, 0.037] | 0.017 [0.011, 0.023] | 0.007 [0.003, 0.012] |
| Timeout rate * | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Rudder saturation fraction * | 0.098 [0.090, 0.107] | 0.032 [0.030, 0.035] | 0.130 [0.123, 0.138] | 0.098 [0.091, 0.105] |
| RMS cross-track error (m) | 0.908 [0.879, 0.936] | 1.101 [1.083, 1.119] | 1.286 [1.258, 1.316] | 1.328 [1.312, 1.344] |
| Min obstacle clearance (m) | 0.233 [0.217, 0.249] | 0.440 [0.422, 0.459] | 0.528 [0.497, 0.562] | 0.768 [0.749, 0.788] |
| Min border clearance, all walls (m) | 0.989 [0.989, 0.989] | 0.989 [0.989, 0.989] | 0.989 [0.989, 0.989] | 0.989 [0.989, 0.989] |
| Min border clearance, lateral (m) | 2.514 [2.433, 2.595] | 2.451 [2.398, 2.500] | 2.484 [2.441, 2.528] | 2.524 [2.493, 2.556] |
| Control effort (int. sq. rudder cmd) | 9.846 [9.580, 10.122] | 8.792 [8.664, 8.920] | 5.560 [5.255, 5.893] | 5.845 [5.632, 6.062] |
| Mean abs. rudder rate (deg/s) | 74.91 [73.40, 76.29] | 77.30 [76.50, 78.05] | 11.33 [11.18, 11.47] | 8.96 [8.83, 9.10] |
| Completion time (s) | 20.9 [20.7, 21.0] | 20.3 [20.2, 20.4] | 22.0 [21.7, 22.4] | 27.2 [27.0, 27.4] |

| | SAC (deployed) | SAC (retrained, 3 seeds) | PPO (3 seeds) | LOS+APF (3 searches) |
|---|---|---|---|---|
| Runs | sac_1M | sac_gs4_seed0_best, sac_gs4_seed1_best, sac_gs4_seed2_best | ppo_fx_seed0_best, ppo_fx_seed1_best, ppo_fx_seed2_best | los_apf_s1, los_apf_s2, los_apf_s3 |
| Episodes | 500 | 1500 | 1500 | 1500 |
