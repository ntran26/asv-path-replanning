# Paired statistics on the frozen 500-episode evaluation set

McNemar (exact) on success; Wilcoxon signed-rank on RMS cross-track error and on per-episode minimum obstacle clearance. `both_succeeded` restricts to episodes where both methods reached the goal -- a collision truncates the trajectory and flatters its RMS CTE, so the all-episode figure mixes tracking quality with failure timing.

| Comparison | Scope | n | Succ A | Succ B | McNemar p | RMS CTE A | RMS CTE B | HL diff | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| SAC published vs SAC retrained best (sac_seed0) | all_paired | 500 | 0.950 | 0.888 | 0.000752 | 0.929 | 1.249 | -0.321 | 4.87e-32 |
| SAC published vs SAC retrained best (sac_seed0) | both_succeeded | 419 | 0.950 | 0.888 | 0.000752 | 0.907 | 1.299 | -0.318 | 2.85e-36 |
| SAC published vs SAC retrained best (sac_seed1) | all_paired | 500 | 0.950 | 0.818 | 2.12e-12 | 0.929 | 0.889 | -0.001 | 0.633 |
| SAC published vs SAC retrained best (sac_seed1) | both_succeeded | 395 | 0.950 | 0.818 | 2.12e-12 | 0.859 | 0.862 | -0.014 | 0.00811 |
| SAC published vs SAC retrained best (sac_seed2) | all_paired | 500 | 0.950 | 0.922 | 0.098 | 0.929 | 0.938 | -0.042 | 0.0121 |
| SAC published vs SAC retrained best (sac_seed2) | both_succeeded | 437 | 0.950 | 0.922 | 0.098 | 0.930 | 0.960 | -0.045 | 0.000215 |
| SAC published vs SAC retrained final (sac_seed0) | all_paired | 500 | 0.950 | 0.816 | 1.49e-10 | 0.929 | 1.080 | -0.162 | 1.41e-13 |
| SAC published vs SAC retrained final (sac_seed0) | both_succeeded | 385 | 0.950 | 0.816 | 1.49e-10 | 0.975 | 1.204 | -0.230 | 5.11e-30 |
| SAC published vs SAC retrained final (sac_seed1) | all_paired | 500 | 0.950 | 0.794 | 6.4e-15 | 0.929 | 0.741 | -0.055 | 0.000484 |
| SAC published vs SAC retrained final (sac_seed1) | both_succeeded | 382 | 0.950 | 0.794 | 6.4e-15 | 0.870 | 0.938 | -0.120 | 4.83e-36 |
| SAC published vs SAC retrained final (sac_seed2) | all_paired | 500 | 0.950 | 0.838 | 3.22e-08 | 0.929 | 0.893 | -0.052 | 0.019 |
| SAC published vs SAC retrained final (sac_seed2) | both_succeeded | 395 | 0.950 | 0.838 | 3.22e-08 | 0.969 | 0.944 | -0.102 | 1.36e-09 |
| SAC published vs PPO retrained best (ppo_seed0) | all_paired | 500 | 0.950 | 0.902 | 0.00558 | 0.929 | 1.174 | -0.315 | 2.32e-34 |
| SAC published vs PPO retrained best (ppo_seed0) | both_succeeded | 428 | 0.950 | 0.902 | 0.00558 | 0.915 | 1.199 | -0.355 | 1.99e-53 |
| SAC published vs PPO retrained best (ppo_seed1) | all_paired | 500 | 0.950 | 0.906 | 0.0103 | 0.929 | 1.856 | -1.023 | 1.61e-71 |
| SAC published vs PPO retrained best (ppo_seed1) | both_succeeded | 430 | 0.950 | 0.906 | 0.0103 | 0.964 | 1.940 | -1.075 | 2.11e-71 |
| SAC published vs PPO retrained best (ppo_seed2) | all_paired | 500 | 0.950 | 0.922 | 0.0925 | 0.929 | 1.101 | -0.291 | 7.08e-31 |
| SAC published vs PPO retrained best (ppo_seed2) | both_succeeded | 438 | 0.950 | 0.922 | 0.0925 | 0.931 | 1.108 | -0.332 | 5.8e-38 |
| SAC published vs PPO retrained final (ppo_seed0) | all_paired | 500 | 0.950 | 0.080 | 2.25e-131 | 0.929 | 1.710 | -1.020 | 1.19e-62 |
| SAC published vs PPO retrained final (ppo_seed0) | both_succeeded | 40 | 0.950 | 0.080 | 2.25e-131 | 0.193 | 1.097 | -0.524 | 1.82e-12 |
| SAC published vs PPO retrained final (ppo_seed1) | all_paired | 500 | 0.950 | 0.000 | 2.05e-143 | 0.929 | 2.309 | -1.481 | 8.12e-81 |
| SAC published vs PPO retrained final (ppo_seed1) | both_succeeded | 0 | 0.950 | 0.000 | 2.05e-143 | nan | nan | +nan | nan |
| SAC published vs PPO retrained final (ppo_seed2) | all_paired | 500 | 0.950 | 0.736 | 6.68e-21 | 0.929 | 0.739 | +0.039 | 0.0751 |
| SAC published vs PPO retrained final (ppo_seed2) | both_succeeded | 350 | 0.950 | 0.736 | 6.68e-21 | 0.906 | 0.800 | -0.121 | 3.58e-06 |
| SAC published vs LOS+APF (los_apf_s1) | all_paired | 500 | 0.950 | 0.968 | 0.211 | 0.929 | 1.217 | -0.189 | 1.22e-35 |
| SAC published vs LOS+APF (los_apf_s1) | both_succeeded | 459 | 0.950 | 0.968 | 0.211 | 0.929 | 1.174 | -0.176 | 2.52e-28 |
| SAC published vs LOS+APF (los_apf_s2) | all_paired | 500 | 0.950 | 0.972 | 0.108 | 0.929 | 1.378 | -0.333 | 1.4e-55 |
| SAC published vs LOS+APF (los_apf_s2) | both_succeeded | 461 | 0.950 | 0.972 | 0.108 | 0.949 | 1.366 | -0.307 | 9.88e-53 |
| SAC published vs LOS+APF (los_apf_s3) | all_paired | 500 | 0.950 | 0.940 | 0.576 | 0.929 | 1.532 | -0.496 | 1.56e-65 |
| SAC published vs LOS+APF (los_apf_s3) | both_succeeded | 447 | 0.950 | 0.940 | 0.576 | 0.928 | 1.513 | -0.453 | 6.11e-56 |
