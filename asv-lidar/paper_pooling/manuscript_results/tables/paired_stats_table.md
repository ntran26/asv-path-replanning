# Paired statistics on the frozen 500-episode evaluation set

McNemar (exact) on success; Wilcoxon signed-rank on RMS cross-track error and on per-episode minimum obstacle clearance. `both_succeeded` restricts to episodes where both methods reached the goal -- a collision truncates the trajectory and flatters its RMS CTE, so the all-episode figure mixes tracking quality with failure timing.

| Comparison | Scope | n | Succ A | Succ B | McNemar p | RMS CTE A | RMS CTE B | HL diff | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| SAC deployed vs SAC retrained (seed0) | all_paired | 500 | 0.950 | 0.814 | 9.29e-11 | 0.929 | 1.200 | -0.264 | 7.33e-27 |
| SAC deployed vs SAC retrained (seed0) | both_succeeded | 384 | 0.950 | 0.814 | 9.29e-11 | 0.959 | 1.337 | -0.347 | 1.44e-44 |
| SAC deployed vs SAC retrained (seed1) | all_paired | 500 | 0.950 | 0.780 | 1.35e-16 | 0.929 | 0.767 | -0.039 | 0.0122 |
| SAC deployed vs SAC retrained (seed1) | both_succeeded | 375 | 0.950 | 0.780 | 1.35e-16 | 0.889 | 0.911 | -0.121 | 1.32e-28 |
| SAC deployed vs SAC retrained (seed2) | all_paired | 500 | 0.950 | 0.842 | 7.68e-08 | 0.929 | 0.873 | -0.049 | 0.032 |
| SAC deployed vs SAC retrained (seed2) | both_succeeded | 397 | 0.950 | 0.842 | 7.68e-08 | 0.959 | 0.950 | -0.096 | 7.87e-09 |
| SAC deployed vs PPO (seed0) | all_paired | 500 | 0.950 | 0.870 | 8.58e-06 | 0.929 | 1.193 | -0.363 | 7.02e-35 |
| SAC deployed vs PPO (seed0) | both_succeeded | 415 | 0.950 | 0.870 | 8.58e-06 | 0.930 | 1.204 | -0.373 | 2.9e-44 |
| SAC deployed vs PPO (seed1) | all_paired | 500 | 0.950 | 0.912 | 0.0295 | 0.929 | 1.826 | -0.991 | 2.06e-74 |
| SAC deployed vs PPO (seed1) | both_succeeded | 431 | 0.950 | 0.912 | 0.0295 | 0.958 | 1.890 | -1.020 | 5.06e-72 |
| SAC deployed vs PPO (seed2) | all_paired | 500 | 0.950 | 0.832 | 1.79e-09 | 0.929 | 0.841 | -0.056 | 0.00989 |
| SAC deployed vs PPO (seed2) | both_succeeded | 396 | 0.950 | 0.832 | 1.79e-09 | 0.937 | 0.891 | -0.148 | 3.07e-10 |
| SAC deployed vs LOS+APF (los_apf_s1) | all_paired | 500 | 0.950 | 0.968 | 0.211 | 0.929 | 1.217 | -0.189 | 1.22e-35 |
| SAC deployed vs LOS+APF (los_apf_s1) | both_succeeded | 459 | 0.950 | 0.968 | 0.211 | 0.929 | 1.174 | -0.176 | 2.52e-28 |
| SAC deployed vs LOS+APF (los_apf_s2) | all_paired | 500 | 0.950 | 0.972 | 0.108 | 0.929 | 1.378 | -0.333 | 1.4e-55 |
| SAC deployed vs LOS+APF (los_apf_s2) | both_succeeded | 461 | 0.950 | 0.972 | 0.108 | 0.949 | 1.366 | -0.307 | 9.88e-53 |
| SAC deployed vs LOS+APF (los_apf_s3) | all_paired | 500 | 0.950 | 0.940 | 0.576 | 0.929 | 1.532 | -0.496 | 1.56e-65 |
| SAC deployed vs LOS+APF (los_apf_s3) | both_succeeded | 447 | 0.950 | 0.940 | 0.576 | 0.928 | 1.513 | -0.453 | 6.11e-56 |
| SAC retrained seed0 vs PPO seed0 | all_paired | 500 | 0.814 | 0.870 | 0.0158 | 1.200 | 1.193 | -0.078 | 0.0144 |
| SAC retrained seed0 vs PPO seed0 | both_succeeded | 358 | 0.814 | 0.870 | 0.0158 | 1.291 | 1.149 | +0.059 | 0.00832 |
| SAC retrained seed1 vs PPO seed1 | all_paired | 500 | 0.780 | 0.912 | 1.69e-08 | 0.767 | 1.826 | -0.995 | 1.16e-75 |
| SAC retrained seed1 vs PPO seed1 | both_succeeded | 354 | 0.780 | 0.912 | 1.69e-08 | 0.923 | 1.925 | -0.947 | 1.79e-59 |
| SAC retrained seed2 vs PPO seed2 | all_paired | 500 | 0.842 | 0.832 | 0.668 | 0.873 | 0.841 | -0.026 | 0.0498 |
| SAC retrained seed2 vs PPO seed2 | both_succeeded | 375 | 0.842 | 0.832 | 0.668 | 0.927 | 0.882 | -0.057 | 0.00379 |
