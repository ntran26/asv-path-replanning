# Paired statistics on the frozen 500-episode evaluation set

McNemar (exact) on success; Wilcoxon signed-rank on RMS cross-track error and on per-episode minimum obstacle clearance. `both_succeeded` restricts to episodes where both methods reached the goal -- a collision truncates the trajectory and flatters its RMS CTE, so the all-episode figure mixes tracking quality with failure timing.

| Comparison | Scope | n | Succ A | Succ B | McNemar p | RMS CTE A | RMS CTE B | HL diff | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| SAC deployed vs SAC retrained (seed0) | all_paired | 500 | 0.950 | 0.940 | 0.511 | 0.929 | 1.260 | -0.309 | 9.44e-71 |
| SAC deployed vs SAC retrained (seed0) | both_succeeded | 454 | 0.950 | 0.940 | 0.511 | 0.931 | 1.282 | -0.318 | 9.17e-75 |
| SAC deployed vs SAC retrained (seed1) | all_paired | 500 | 0.950 | 0.900 | 0.00126 | 0.929 | 1.100 | -0.167 | 2.63e-28 |
| SAC deployed vs SAC retrained (seed1) | both_succeeded | 434 | 0.950 | 0.900 | 0.00126 | 0.906 | 1.147 | -0.185 | 3.88e-53 |
| SAC deployed vs SAC retrained (seed2) | all_paired | 500 | 0.950 | 0.932 | 0.289 | 0.929 | 1.046 | -0.180 | 6.97e-19 |
| SAC deployed vs SAC retrained (seed2) | both_succeeded | 442 | 0.950 | 0.932 | 0.289 | 0.945 | 1.061 | -0.184 | 7.98e-23 |
| SAC deployed vs PPO (seed0) | all_paired | 500 | 0.950 | 0.894 | 0.00152 | 0.929 | 1.210 | -0.396 | 9.08e-33 |
| SAC deployed vs PPO (seed0) | both_succeeded | 424 | 0.950 | 0.894 | 0.00152 | 0.937 | 1.259 | -0.455 | 2e-51 |
| SAC deployed vs PPO (seed1) | all_paired | 500 | 0.950 | 0.898 | 0.00294 | 0.929 | 1.875 | -1.036 | 9.27e-72 |
| SAC deployed vs PPO (seed1) | both_succeeded | 426 | 0.950 | 0.898 | 0.00294 | 0.972 | 1.985 | -1.094 | 1.96e-70 |
| SAC deployed vs PPO (seed2) | all_paired | 500 | 0.950 | 0.922 | 0.0925 | 0.929 | 1.101 | -0.291 | 7.08e-31 |
| SAC deployed vs PPO (seed2) | both_succeeded | 438 | 0.950 | 0.922 | 0.0925 | 0.931 | 1.108 | -0.332 | 5.8e-38 |
| SAC deployed vs LOS+APF (los_apf_s1) | all_paired | 500 | 0.950 | 0.968 | 0.211 | 0.929 | 1.217 | -0.189 | 1.22e-35 |
| SAC deployed vs LOS+APF (los_apf_s1) | both_succeeded | 459 | 0.950 | 0.968 | 0.211 | 0.929 | 1.174 | -0.176 | 2.52e-28 |
| SAC deployed vs LOS+APF (los_apf_s2) | all_paired | 500 | 0.950 | 0.972 | 0.108 | 0.929 | 1.378 | -0.333 | 1.4e-55 |
| SAC deployed vs LOS+APF (los_apf_s2) | both_succeeded | 461 | 0.950 | 0.972 | 0.108 | 0.949 | 1.366 | -0.307 | 9.88e-53 |
| SAC deployed vs LOS+APF (los_apf_s3) | all_paired | 500 | 0.950 | 0.940 | 0.576 | 0.929 | 1.532 | -0.496 | 1.56e-65 |
| SAC deployed vs LOS+APF (los_apf_s3) | both_succeeded | 447 | 0.950 | 0.940 | 0.576 | 0.928 | 1.513 | -0.453 | 6.11e-56 |
| SAC retrained seed0 vs PPO seed0 | all_paired | 500 | 0.940 | 0.894 | 0.0152 | 1.260 | 1.210 | -0.081 | 0.00487 |
| SAC retrained seed0 vs PPO seed0 | both_succeeded | 417 | 0.940 | 0.894 | 0.0152 | 1.276 | 1.253 | -0.117 | 6.46e-08 |
| SAC retrained seed1 vs PPO seed1 | all_paired | 500 | 0.900 | 0.898 | 1 | 1.100 | 1.875 | -0.887 | 6.51e-64 |
| SAC retrained seed1 vs PPO seed1 | both_succeeded | 403 | 0.900 | 0.898 | 1 | 1.163 | 1.963 | -0.897 | 5.59e-66 |
| SAC retrained seed2 vs PPO seed2 | all_paired | 500 | 0.932 | 0.922 | 0.59 | 1.046 | 1.101 | -0.089 | 4.73e-13 |
| SAC retrained seed2 vs PPO seed2 | both_succeeded | 436 | 0.932 | 0.922 | 0.59 | 1.038 | 1.118 | -0.121 | 7.04e-18 |
