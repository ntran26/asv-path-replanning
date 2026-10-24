# Results on test set v4 (safety layer off)

Point estimate with a stratified bootstrap 95 % interval (stratified by test-set cell, pooled over runs, 5,000 resamples). Rate metrics (marked *) are means of per-episode 0/1 outcomes; continuous metrics are interquartile means (IQM). Each run is represented by its best validation-set checkpoint. Definitions follow Paper 2 (`src/metrics.py`).

| Method | Runs | Episodes per scope (all / decoupled / coupled) |
|---|---|---|
| SAC | sacs0_bl3, sacs1_bl3, sacs2_bl3 | 3000 / 1992 / 1008 |
| TQC | tqcs0_bl3, tqcs1_bl3, tqcs2_bl3 | 3000 / 1992 / 1008 |
| PPO | ppos0_bl3, ppos1_bl3, ppos2_bl3 | 3000 / 1992 / 1008 |
| RecurrentPPO | recurrent_ppos0_bl3, recurrent_ppos1_bl3, recurrent_ppos2_bl3 | 3000 / 1992 / 1008 |
| LOS-DWA | los_dwa | 1000 / 664 / 336 |

## All episodes

| Metric | SAC | TQC | PPO | RecurrentPPO | LOS-DWA |
|---|---|---|---|---|---|
| **Success rate** * | 0.846 [0.834, 0.857] | 0.830 [0.817, 0.842] | 0.750 [0.736, 0.764] | 0.717 [0.702, 0.731] | 0.770 [0.747, 0.793] |
| Target collision rate * | 0.063 [0.055, 0.072] | 0.067 [0.058, 0.076] | 0.154 [0.141, 0.166] | 0.184 [0.172, 0.196] | 0.080 [0.065, 0.095] |
| Obstacle collision rate * | 0.074 [0.065, 0.083] | 0.082 [0.073, 0.091] | 0.066 [0.058, 0.075] | 0.065 [0.056, 0.073] | 0.109 [0.093, 0.126] |
| Boundary collision rate * | 0.017 [0.013, 0.022] | 0.021 [0.016, 0.026] | 0.030 [0.024, 0.036] | 0.035 [0.029, 0.041] | 0.038 [0.027, 0.050] |
| Timeout rate * | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.000, 0.007] |
| RMS cross-track error (m) | 1.370 [1.342, 1.398] | 1.269 [1.241, 1.298] | 0.994 [0.968, 1.022] | 0.905 [0.877, 0.932] | 0.515 [0.482, 0.550] |
| Completion time, successful episodes (s) | 25.8 [25.8, 26.0] | 27.3 [27.1, 27.5] | 30.1 [29.6, 30.2] | 31.0 [30.8, 31.2] | 36.5 [36.2, 36.9] |
| Path length / reference, successful episodes | 0.990 [0.988, 0.991] | 0.986 [0.985, 0.987] | 0.970 [0.969, 0.971] | 0.965 [0.964, 0.966] | 0.953 [0.952, 0.954] |
| Mean speed (m/s) | 0.749 [0.746, 0.752] | 0.713 [0.711, 0.716] | 0.608 [0.604, 0.612] | 0.602 [0.599, 0.605] | 0.495 [0.491, 0.499] |
| Minimum speed (m/s) | 0.504 [0.500, 0.508] | 0.503 [0.499, 0.506] | 0.389 [0.383, 0.395] | 0.429 [0.423, 0.434] | 0.322 [0.312, 0.333] |
| Min obstacle clearance (m) | 0.664 [0.641, 0.687] | 0.599 [0.578, 0.620] | 0.746 [0.718, 0.776] | 0.838 [0.808, 0.868] | 0.634 [0.591, 0.678] |
| Min boundary clearance (m) | 0.813 [0.810, 0.816] | 0.818 [0.816, 0.821] | 0.818 [0.815, 0.821] | 0.816 [0.814, 0.819] | 0.824 [0.820, 0.828] |
| *Target ship* | | | | | |
| Ship-domain intrusion rate * | 0.112 [0.101, 0.123] | 0.125 [0.114, 0.137] | 0.239 [0.225, 0.253] | 0.251 [0.236, 0.265] | 0.142 [0.123, 0.162] |
| Deepest intrusion, episodes with one (fraction of radius) | 0.208 [0.188, 0.228] | 0.192 [0.175, 0.209] | 0.232 [0.218, 0.247] | 0.271 [0.257, 0.285] | 0.215 [0.182, 0.251] |
| Time inside the domain, episodes with an intrusion (s) | 1.567 [1.420, 1.636] | 1.622 [1.430, 1.684] | 1.755 [1.619, 1.813] | 2.296 [1.811, 2.353] | 2.117 [1.900, 2.318] |
| Closest approach, centre to centre (m) | 2.828 [2.783, 2.872] | 2.705 [2.659, 2.748] | 2.462 [2.418, 2.505] | 2.351 [2.311, 2.393] | 2.473 [2.405, 2.540] |
| Min target clearance, hull to hull (m) | 1.619 [1.580, 1.657] | 1.519 [1.481, 1.558] | 1.299 [1.257, 1.340] | 1.205 [1.162, 1.247] | 1.330 [1.279, 1.384] |
| *Actuation* | | | | | |
| Rudder saturation fraction * | 0.050 [0.048, 0.053] | 0.110 [0.106, 0.113] | 0.303 [0.298, 0.308] | 0.342 [0.335, 0.349] | 0.279 [0.271, 0.288] |
| Mean abs. rudder rate (deg/s) | 23.7 [23.5, 23.9] | 26.8 [26.6, 27.0] | 31.5 [31.0, 32.0] | 38.6 [38.2, 39.1] | 26.2 [25.7, 26.6] |
| Control effort (int. sq. rudder cmd) | 10.4 [10.3, 10.5] | 12.9 [12.8, 13.1] | 12.4 [12.2, 12.6] | 11.7 [11.5, 12.0] | 14.1 [13.6, 14.5] |

## Decoupled

| Metric | SAC | TQC | PPO | RecurrentPPO | LOS-DWA |
|---|---|---|---|---|---|
| **Success rate** * | 0.901 [0.888, 0.914] | 0.891 [0.878, 0.905] | 0.822 [0.807, 0.840] | 0.799 [0.783, 0.815] | 0.870 [0.846, 0.895] |
| Target collision rate * | 0.060 [0.049, 0.070] | 0.056 [0.046, 0.066] | 0.132 [0.118, 0.147] | 0.161 [0.146, 0.177] | 0.060 [0.042, 0.078] |
| Obstacle collision rate * | 0.025 [0.019, 0.032] | 0.033 [0.025, 0.040] | 0.021 [0.015, 0.028] | 0.022 [0.016, 0.029] | 0.038 [0.024, 0.053] |
| Boundary collision rate * | 0.015 [0.010, 0.020] | 0.020 [0.014, 0.027] | 0.025 [0.018, 0.032] | 0.018 [0.012, 0.024] | 0.032 [0.020, 0.045] |
| Timeout rate * | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| RMS cross-track error (m) | 1.245 [1.215, 1.274] | 1.165 [1.134, 1.197] | 0.960 [0.929, 0.990] | 0.806 [0.775, 0.838] | 0.436 [0.400, 0.477] |
| Completion time, successful episodes (s) | 25.8 [25.6, 26.1] | 27.5 [27.2, 27.6] | 29.6 [29.5, 30.0] | 30.8 [30.6, 31.0] | 35.7 [35.5, 36.0] |
| Path length / reference, successful episodes | 0.983 [0.982, 0.984] | 0.980 [0.979, 0.981] | 0.965 [0.964, 0.966] | 0.960 [0.959, 0.961] | 0.951 [0.950, 0.952] |
| Mean speed (m/s) | 0.745 [0.741, 0.748] | 0.708 [0.705, 0.711] | 0.617 [0.612, 0.622] | 0.607 [0.604, 0.610] | 0.516 [0.512, 0.519] |
| Minimum speed (m/s) | 0.513 [0.508, 0.517] | 0.509 [0.505, 0.513] | 0.417 [0.409, 0.425] | 0.467 [0.461, 0.472] | 0.377 [0.364, 0.389] |
| Min obstacle clearance (m) | 0.975 [0.931, 1.021] | 0.784 [0.748, 0.821] | 1.142 [1.088, 1.202] | 1.192 [1.134, 1.251] | 0.874 [0.805, 0.947] |
| Min boundary clearance (m) | 0.798 [0.795, 0.801] | 0.799 [0.796, 0.802] | 0.802 [0.800, 0.805] | 0.802 [0.799, 0.804] | 0.800 [0.796, 0.803] |
| *Target ship* | | | | | |
| Ship-domain intrusion rate * | 0.104 [0.091, 0.117] | 0.108 [0.094, 0.122] | 0.200 [0.184, 0.217] | 0.221 [0.205, 0.238] | 0.101 [0.080, 0.122] |
| Deepest intrusion, episodes with one (fraction of radius) | 0.205 [0.180, 0.231] | 0.181 [0.158, 0.208] | 0.233 [0.213, 0.253] | 0.266 [0.245, 0.287] | 0.205 [0.153, 0.268] |
| Time inside the domain, episodes with an intrusion (s) | 1.536 [1.398, 1.780] | 1.465 [1.405, 1.675] | 1.728 [1.549, 1.802] | 2.108 [1.777, 2.322] | 2.081 [1.671, 2.425] |
| Closest approach, centre to centre (m) | 2.755 [2.707, 2.806] | 2.657 [2.607, 2.707] | 2.446 [2.400, 2.496] | 2.361 [2.314, 2.412] | 2.546 [2.461, 2.637] |
| Min target clearance, hull to hull (m) | 1.589 [1.547, 1.631] | 1.502 [1.460, 1.547] | 1.341 [1.294, 1.385] | 1.284 [1.239, 1.333] | 1.466 [1.406, 1.531] |
| *Actuation* | | | | | |
| Rudder saturation fraction * | 0.040 [0.037, 0.043] | 0.095 [0.090, 0.099] | 0.285 [0.279, 0.291] | 0.267 [0.259, 0.276] | 0.229 [0.219, 0.239] |
| Mean abs. rudder rate (deg/s) | 23.2 [23.0, 23.4] | 26.5 [26.2, 26.7] | 30.8 [30.2, 31.3] | 36.0 [35.4, 36.6] | 24.5 [24.0, 25.0] |
| Control effort (int. sq. rudder cmd) | 10.1 [9.9, 10.2] | 12.7 [12.5, 12.9] | 12.1 [11.9, 12.3] | 10.4 [10.1, 10.6] | 12.7 [12.3, 13.2] |

## Coupled

| Metric | SAC | TQC | PPO | RecurrentPPO | LOS-DWA |
|---|---|---|---|---|---|
| **Success rate** * | 0.737 [0.711, 0.762] | 0.709 [0.683, 0.735] | 0.608 [0.579, 0.635] | 0.555 [0.526, 0.583] | 0.571 [0.524, 0.622] |
| Target collision rate * | 0.070 [0.057, 0.085] | 0.088 [0.072, 0.105] | 0.196 [0.174, 0.219] | 0.228 [0.205, 0.252] | 0.119 [0.089, 0.149] |
| Obstacle collision rate * | 0.171 [0.149, 0.193] | 0.180 [0.156, 0.202] | 0.156 [0.135, 0.179] | 0.149 [0.127, 0.171] | 0.250 [0.208, 0.295] |
| Boundary collision rate * | 0.022 [0.014, 0.031] | 0.023 [0.015, 0.032] | 0.040 [0.029, 0.052] | 0.068 [0.054, 0.083] | 0.051 [0.030, 0.074] |
| Timeout rate * | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.009 [0.000, 0.021] |
| RMS cross-track error (m) | 1.759 [1.697, 1.818] | 1.588 [1.528, 1.648] | 1.104 [1.047, 1.162] | 1.137 [1.087, 1.190] | 0.671 [0.618, 0.727] |
| Completion time, successful episodes (s) | 26.0 [25.8, 26.1] | 27.2 [26.9, 27.4] | 30.4 [30.1, 31.0] | 31.9 [31.5, 32.4] | 39.9 [39.2, 41.1] |
| Path length / reference, successful episodes | 1.011 [1.008, 1.013] | 1.007 [1.004, 1.011] | 0.988 [0.985, 0.992] | 0.982 [0.980, 0.985] | 0.961 [0.959, 0.963] |
| Mean speed (m/s) | 0.759 [0.753, 0.765] | 0.726 [0.720, 0.731] | 0.582 [0.572, 0.591] | 0.577 [0.566, 0.587] | 0.452 [0.444, 0.459] |
| Minimum speed (m/s) | 0.486 [0.478, 0.493] | 0.489 [0.482, 0.496] | 0.337 [0.328, 0.347] | 0.342 [0.330, 0.353] | 0.243 [0.230, 0.257] |
| Min obstacle clearance (m) | 0.440 [0.412, 0.467] | 0.449 [0.420, 0.477] | 0.474 [0.445, 0.503] | 0.590 [0.557, 0.622] | 0.422 [0.246, 0.478] |
| Min boundary clearance (m) | 0.841 [0.836, 0.845] | 0.849 [0.845, 0.852] | 0.842 [0.839, 0.845] | 0.840 [0.836, 0.843] | 0.856 [0.851, 0.860] |
| *Target ship* | | | | | |
| Ship-domain intrusion rate * | 0.129 [0.109, 0.148] | 0.163 [0.142, 0.185] | 0.324 [0.296, 0.350] | 0.314 [0.288, 0.340] | 0.232 [0.193, 0.275] |
| Deepest intrusion, episodes with one (fraction of radius) | 0.214 [0.181, 0.248] | 0.205 [0.184, 0.227] | 0.232 [0.209, 0.255] | 0.278 [0.259, 0.296] | 0.224 [0.186, 0.267] |
| Time inside the domain, episodes with an intrusion (s) | 1.493 [1.424, 1.756] | 1.653 [1.427, 1.909] | 1.791 [1.659, 2.202] | 2.320 [1.921, 2.559] | 2.148 [1.912, 2.362] |
| Closest approach, centre to centre (m) | 3.009 [2.917, 3.096] | 2.815 [2.730, 2.902] | 2.492 [2.406, 2.583] | 2.319 [2.241, 2.399] | 2.372 [2.267, 2.485] |
| Min target clearance, hull to hull (m) | 1.692 [1.611, 1.771] | 1.547 [1.466, 1.629] | 1.173 [1.081, 1.268] | 0.658 [0.600, 1.077] | 1.060 [0.976, 1.146] |
| *Actuation* | | | | | |
| Rudder saturation fraction * | 0.071 [0.067, 0.076] | 0.139 [0.132, 0.146] | 0.339 [0.331, 0.348] | 0.488 [0.476, 0.500] | 0.380 [0.366, 0.394] |
| Mean abs. rudder rate (deg/s) | 24.9 [24.5, 25.3] | 27.6 [27.1, 28.1] | 33.2 [32.3, 34.2] | 44.0 [43.1, 44.9] | 30.3 [29.3, 31.4] |
| Control effort (int. sq. rudder cmd) | 11.1 [10.9, 11.4] | 13.5 [13.2, 13.7] | 12.9 [12.5, 13.3] | 15.2 [14.7, 15.7] | 17.5 [16.5, 18.6] |

Notes: completion time and path length are over successful episodes only, because a collision truncates an episode; RMS cross-track error is over all episodes, as in Paper 2, so a collision can flatter it. Target-ship rows cover the episodes with a target. The ship domain is own ship's (3.14 m ahead, 1.57 m astern, 1.25 m abeam), entered by the target's centre; intrusion depth and time are over the episodes with an intrusion, since most have none and an IQM over all would read zero. COLREGs behaviour is shown by the snapshot figure, not by per-rule rates. The classical controllers are deterministic, so their intervals reflect episode variance only. Clearances are from the hull polygon with its 0.15 m margin. Path length below 1 reflects the goal test (within 1.25 m of the path end) and cut corners.

## Paired statistics

Against SAC, same episodes (first run of each method). McNemar (exact) on success; Wilcoxon signed-rank on RMS cross-track error and on the minimum hull-to-hull clearance to the target. `both succeeded` restricts to episodes both methods completed.

| Comparison | Episodes | Scope | n | Success A | Success B | McNemar only A / only B | McNemar p | RMS CTE median A / B | RMS CTE median diff | RMS CTE Wilcoxon p | Target clearance median A / B | Target clearance median diff | Target clearance Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SAC vs TQC | All episodes | all paired | 1000 | 0.865 | 0.837 | 91 / 63 | 0.0292 | 1.414 / 1.161 | +0.080 | 2.86e-13 | 1.670 / 1.497 | +0.145 | 2.17e-11 |
| SAC vs TQC | All episodes | both succeeded | 774 | 0.865 | 0.837 | 91 / 63 | 0.0292 | 1.468 / 1.206 | +0.066 | 1.3e-10 | 1.777 / 1.606 | +0.170 | 9.07e-12 |
| SAC vs TQC | Decoupled | all paired | 664 | 0.931 | 0.895 | 51 / 27 | 0.00877 | 1.307 / 1.076 | +0.130 | 2.54e-17 | 1.650 / 1.487 | +0.180 | 1.13e-08 |
| SAC vs TQC | Decoupled | both succeeded | 567 | 0.931 | 0.895 | 51 / 27 | 0.00877 | 1.310 / 1.111 | +0.111 | 2.23e-13 | 1.738 / 1.567 | +0.196 | 1.64e-09 |
| SAC vs TQC | Coupled | all paired | 336 | 0.735 | 0.723 | 40 / 36 | 0.731 | 1.862 / 1.653 | -0.006 | 0.456 | 1.716 / 1.525 | +0.100 | 0.000598 |
| SAC vs TQC | Coupled | both succeeded | 207 | 0.735 | 0.723 | 40 / 36 | 0.731 | 2.152 / 2.053 | -0.012 | 0.868 | 1.963 / 1.783 | +0.095 | 0.00226 |
| SAC vs PPO | All episodes | all paired | 1000 | 0.865 | 0.751 | 195 / 81 | 5.17e-12 | 1.414 / 0.859 | +0.383 | 5.52e-77 | 1.670 / 1.236 | +0.328 | 1.45e-16 |
| SAC vs PPO | All episodes | both succeeded | 670 | 0.865 | 0.751 | 195 / 81 | 5.17e-12 | 1.418 / 0.942 | +0.302 | 3.7e-53 | 1.766 / 1.413 | +0.259 | 3.48e-10 |
| SAC vs PPO | Decoupled | all paired | 664 | 0.931 | 0.827 | 103 / 34 | 2.9e-09 | 1.307 / 0.818 | +0.399 | 1.21e-55 | 1.650 / 1.255 | +0.305 | 9.42e-13 |
| SAC vs PPO | Decoupled | both succeeded | 515 | 0.931 | 0.827 | 103 / 34 | 2.9e-09 | 1.308 / 0.916 | +0.323 | 4.91e-42 | 1.761 / 1.407 | +0.240 | 9.05e-08 |
| SAC vs PPO | Coupled | all paired | 336 | 0.735 | 0.601 | 92 / 47 | 0.000169 | 1.862 / 0.922 | +0.359 | 5.6e-24 | 1.716 / 1.151 | +0.511 | 3.88e-06 |
| SAC vs PPO | Coupled | both succeeded | 155 | 0.735 | 0.601 | 92 / 47 | 0.000169 | 2.133 / 1.115 | +0.258 | 9.42e-13 | 1.895 / 1.485 | +0.487 | 0.000412 |
| SAC vs RecurrentPPO | All episodes | all paired | 1000 | 0.865 | 0.757 | 174 / 66 | 2.1e-12 | 1.414 / 0.911 | +0.291 | 2.08e-60 | 1.670 / 1.211 | +0.351 | 9.81e-20 |
| SAC vs RecurrentPPO | All episodes | both succeeded | 691 | 0.865 | 0.757 | 174 / 66 | 2.1e-12 | 1.419 / 1.019 | +0.222 | 1.48e-42 | 1.822 / 1.384 | +0.301 | 3.89e-14 |
| SAC vs RecurrentPPO | Decoupled | all paired | 664 | 0.931 | 0.834 | 92 / 28 | 3.77e-09 | 1.307 / 0.811 | +0.302 | 1.85e-48 | 1.650 / 1.281 | +0.303 | 5.77e-13 |
| SAC vs RecurrentPPO | Decoupled | both succeeded | 526 | 0.931 | 0.834 | 92 / 28 | 3.77e-09 | 1.307 / 0.898 | +0.254 | 8e-35 | 1.781 / 1.395 | +0.225 | 1.22e-08 |
| SAC vs RecurrentPPO | Coupled | all paired | 336 | 0.735 | 0.604 | 82 / 38 | 7.29e-05 | 1.862 / 1.145 | +0.221 | 3.12e-15 | 1.716 / 1.033 | +0.582 | 9.36e-09 |
| SAC vs RecurrentPPO | Coupled | both succeeded | 165 | 0.735 | 0.604 | 82 / 38 | 7.29e-05 | 2.079 / 1.372 | +0.142 | 7.61e-09 | 2.053 / 1.317 | +0.576 | 2.48e-08 |
| SAC vs LOS-DWA | All episodes | all paired | 1000 | 0.865 | 0.770 | 176 / 81 | 3.02e-09 | 1.414 / 0.484 | +0.733 | 1.21e-111 | 1.670 / 1.250 | +0.253 | 1.05e-05 |
| SAC vs LOS-DWA | All episodes | both succeeded | 689 | 0.865 | 0.770 | 176 / 81 | 3.02e-09 | 1.363 / 0.487 | +0.714 | 3.9e-88 | 1.761 / 1.352 | +0.210 | 0.00383 |
| SAC vs LOS-DWA | Decoupled | all paired | 664 | 0.931 | 0.870 | 75 / 35 | 0.000172 | 1.307 / 0.391 | +0.687 | 2.18e-80 | 1.650 / 1.342 | +0.180 | 0.392 |
| SAC vs LOS-DWA | Decoupled | both succeeded | 543 | 0.931 | 0.870 | 75 / 35 | 0.000172 | 1.288 / 0.400 | +0.695 | 5.01e-72 | 1.742 / 1.464 | +0.152 | 0.824 |
| SAC vs LOS-DWA | Coupled | all paired | 336 | 0.735 | 0.571 | 101 / 46 | 6.67e-06 | 1.862 / 0.645 | +0.917 | 5.83e-34 | 1.716 / 0.992 | +0.668 | 7.32e-11 |
| SAC vs LOS-DWA | Coupled | both succeeded | 146 | 0.735 | 0.571 | 101 / 46 | 6.67e-06 | 2.046 / 0.730 | +0.954 | 6.03e-18 | 1.947 / 0.987 | +0.877 | 1.13e-09 |

