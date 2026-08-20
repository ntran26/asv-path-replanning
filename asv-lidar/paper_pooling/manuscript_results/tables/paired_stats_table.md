# Paired statistics on the frozen 500-episode evaluation set

McNemar (exact) on success; Wilcoxon signed-rank on RMS cross-track error and on per-episode minimum obstacle clearance. `both_succeeded` restricts to episodes where both methods reached the goal -- a collision truncates the trajectory and flatters its RMS CTE, so the all-episode figure mixes tracking quality with failure timing.

| Comparison | Scope | n | Succ A | Succ B | McNemar p | RMS CTE A | RMS CTE B | HL diff | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| SAC published vs PPO final (seed0) | all_paired | 500 | 0.950 | 0.870 | 8.58e-06 | 0.929 | 1.193 | -0.363 | 7.02e-35 |
| SAC published vs PPO final (seed0) | both_succeeded | 415 | 0.950 | 0.870 | 8.58e-06 | 0.930 | 1.204 | -0.373 | 2.9e-44 |
| SAC published vs PPO final (seed1) | all_paired | 500 | 0.950 | 0.912 | 0.0295 | 0.929 | 1.826 | -0.991 | 2.06e-74 |
| SAC published vs PPO final (seed1) | both_succeeded | 431 | 0.950 | 0.912 | 0.0295 | 0.958 | 1.890 | -1.020 | 5.06e-72 |
| SAC published vs PPO final (seed2) | all_paired | 500 | 0.950 | 0.832 | 1.79e-09 | 0.929 | 0.841 | -0.056 | 0.00989 |
| SAC published vs PPO final (seed2) | both_succeeded | 396 | 0.950 | 0.832 | 1.79e-09 | 0.937 | 0.891 | -0.148 | 3.07e-10 |
| SAC published vs LOS+APF (los_apf_s1) | all_paired | 500 | 0.950 | 0.968 | 0.211 | 0.929 | 1.217 | -0.189 | 1.22e-35 |
| SAC published vs LOS+APF (los_apf_s1) | both_succeeded | 459 | 0.950 | 0.968 | 0.211 | 0.929 | 1.174 | -0.176 | 2.52e-28 |
| SAC published vs LOS+APF (los_apf_s2) | all_paired | 500 | 0.950 | 0.972 | 0.108 | 0.929 | 1.378 | -0.333 | 1.4e-55 |
| SAC published vs LOS+APF (los_apf_s2) | both_succeeded | 461 | 0.950 | 0.972 | 0.108 | 0.949 | 1.366 | -0.307 | 9.88e-53 |
| SAC published vs LOS+APF (los_apf_s3) | all_paired | 500 | 0.950 | 0.940 | 0.576 | 0.929 | 1.532 | -0.496 | 1.56e-65 |
| SAC published vs LOS+APF (los_apf_s3) | both_succeeded | 447 | 0.950 | 0.940 | 0.576 | 0.928 | 1.513 | -0.453 | 6.11e-56 |
