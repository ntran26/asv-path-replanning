# Results on test set v4 (safety layer off)

Point estimate with a stratified bootstrap 95 % interval (stratified by test-set cell, pooled over runs, 5,000 resamples). Rate metrics (marked *) are means of per-episode 0/1 outcomes; continuous metrics are interquartile means (IQM). Each run is represented by its best development-set checkpoint. Definitions follow Paper 2 (`src/metrics.py`).

| Method | Runs | Episodes per scope (all / decoupled / coupled) |
|---|---|---|
| SAC | sacs0_bl3 | 1000 / 664 / 336 |
| PPO | ppos0_bl3 | 1000 / 664 / 336 |
| RecurrentPPO | recurrent_ppos0_bl3 | 1000 / 664 / 336 |
| LOS-DWA | los_dwa | 1000 / 664 / 336 |
| COLREGs-VO | colregs_vo | 1000 / 664 / 336 |

## All episodes

| Metric | SAC | PPO | RecurrentPPO | LOS-DWA | COLREGs-VO |
|---|---|---|---|---|---|
| **Success rate** * | 0.865 [0.846, 0.883] | 0.751 [0.727, 0.776] | 0.757 [0.732, 0.781] | 0.770 [0.748, 0.793] | 0.687 [0.661, 0.711] |
| Target collision rate * | 0.053 [0.041, 0.066] | 0.142 [0.122, 0.163] | 0.155 [0.135, 0.175] | 0.080 [0.065, 0.096] | 0.135 [0.115, 0.154] |
| Obstacle collision rate * | 0.070 [0.056, 0.085] | 0.091 [0.075, 0.107] | 0.078 [0.063, 0.094] | 0.109 [0.092, 0.126] | 0.133 [0.114, 0.152] |
| Boundary collision rate * | 0.012 [0.006, 0.019] | 0.016 [0.009, 0.024] | 0.010 [0.004, 0.016] | 0.038 [0.027, 0.050] | 0.045 [0.033, 0.058] |
| Timeout rate * | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.000, 0.007] | 0.000 [0.000, 0.000] |
| RMS cross-track error (m) | 1.422 [1.375, 1.469] | 0.853 [0.813, 0.895] | 0.915 [0.869, 0.961] | 0.515 [0.483, 0.547] | 0.413 [0.383, 0.445] |
| Completion time, successful episodes (s) | 26.2 [26.0, 26.5] | 33.2 [33.1, 33.6] | 31.2 [31.0, 31.6] | 36.5 [36.2, 36.9] | 35.3 [35.0, 35.3] |
| Path length / reference, successful episodes | 0.991 [0.989, 0.993] | 0.966 [0.965, 0.968] | 0.965 [0.964, 0.967] | 0.953 [0.952, 0.954] | 0.949 [0.948, 0.950] |
| Mean speed (m/s) | 0.743 [0.737, 0.748] | 0.545 [0.538, 0.550] | 0.575 [0.569, 0.581] | 0.495 [0.491, 0.499] | 0.532 [0.531, 0.534] |
| Minimum speed (m/s) | 0.513 [0.507, 0.518] | 0.306 [0.299, 0.313] | 0.385 [0.376, 0.394] | 0.322 [0.312, 0.333] | 0.396 [0.390, 0.401] |
| Min obstacle clearance (m) | 0.655 [0.619, 0.692] | 0.617 [0.574, 0.662] | 0.748 [0.700, 0.798] | 0.634 [0.592, 0.677] | 0.334 [0.308, 0.363] |
| Min boundary clearance (m) | 0.819 [0.814, 0.823] | 0.823 [0.819, 0.827] | 0.819 [0.814, 0.823] | 0.824 [0.820, 0.828] | 0.827 [0.823, 0.831] |
| *Target ship* | | | | | |
| Ship-domain intrusion rate * | 0.087 [0.070, 0.104] | 0.254 [0.228, 0.279] | 0.248 [0.226, 0.273] | 0.142 [0.123, 0.163] | 0.203 [0.181, 0.225] |
| Deepest intrusion, episodes with one (fraction of radius) | 0.207 [0.172, 0.244] | 0.215 [0.196, 0.236] | 0.265 [0.239, 0.288] | 0.215 [0.184, 0.251] | 0.257 [0.224, 0.288] |
| Time inside the domain, episodes with an intrusion (s) | 1.408 [1.311, 1.658] | 1.858 [1.664, 2.219] | 2.330 [1.965, 2.536] | 2.117 [1.902, 2.324] | 1.928 [1.819, 2.267] |
| Closest approach, centre to centre (m) | 2.898 [2.826, 2.971] | 2.468 [2.398, 2.540] | 2.385 [2.318, 2.458] | 2.473 [2.408, 2.542] | 2.149 [2.094, 2.203] |
| Min target clearance, hull to hull (m) | 1.687 [1.624, 1.750] | 1.276 [1.201, 1.353] | 1.265 [1.194, 1.337] | 1.330 [1.281, 1.382] | 0.992 [0.950, 1.036] |
| *Actuation* | | | | | |
| Rudder saturation fraction * | 0.053 [0.050, 0.058] | 0.234 [0.227, 0.241] | 0.370 [0.362, 0.379] | 0.279 [0.271, 0.287] | 0.156 [0.151, 0.162] |
| Mean abs. rudder rate (deg/s) | 22.3 [21.9, 22.6] | 22.7 [22.4, 23.0] | 41.5 [40.6, 42.3] | 26.2 [25.7, 26.6] | 29.9 [29.5, 30.3] |
| Control effort (int. sq. rudder cmd) | 9.750 [9.545, 9.959] | 12.5 [12.1, 12.8] | 14.3 [13.8, 14.8] | 14.1 [13.6, 14.5] | 11.0 [10.6, 11.3] |

## Decoupled

| Metric | SAC | PPO | RecurrentPPO | LOS-DWA | COLREGs-VO |
|---|---|---|---|---|---|
| **Success rate** * | 0.931 [0.911, 0.949] | 0.827 [0.798, 0.854] | 0.834 [0.807, 0.860] | 0.870 [0.846, 0.893] | 0.783 [0.755, 0.812] |
| Target collision rate * | 0.041 [0.027, 0.056] | 0.134 [0.110, 0.160] | 0.130 [0.107, 0.154] | 0.060 [0.044, 0.078] | 0.111 [0.089, 0.134] |
| Obstacle collision rate * | 0.021 [0.011, 0.033] | 0.027 [0.015, 0.039] | 0.032 [0.020, 0.045] | 0.038 [0.024, 0.053] | 0.071 [0.053, 0.090] |
| Boundary collision rate * | 0.008 [0.002, 0.015] | 0.012 [0.005, 0.021] | 0.005 [0.000, 0.011] | 0.032 [0.018, 0.045] | 0.035 [0.021, 0.048] |
| Timeout rate * | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| RMS cross-track error (m) | 1.313 [1.266, 1.359] | 0.814 [0.763, 0.863] | 0.824 [0.765, 0.880] | 0.436 [0.398, 0.477] | 0.320 [0.287, 0.357] |
| Completion time, successful episodes (s) | 26.4 [26.1, 26.7] | 33.2 [32.8, 33.5] | 30.7 [30.5, 30.9] | 35.7 [35.5, 36.0] | 35.0 [34.9, 35.0] |
| Path length / reference, successful episodes | 0.987 [0.985, 0.989] | 0.963 [0.962, 0.965] | 0.961 [0.959, 0.962] | 0.951 [0.950, 0.952] | 0.948 [0.947, 0.949] |
| Mean speed (m/s) | 0.736 [0.730, 0.742] | 0.556 [0.550, 0.562] | 0.605 [0.600, 0.610] | 0.516 [0.512, 0.520] | 0.537 [0.536, 0.538] |
| Minimum speed (m/s) | 0.516 [0.509, 0.522] | 0.328 [0.320, 0.337] | 0.451 [0.441, 0.461] | 0.377 [0.364, 0.389] | 0.414 [0.408, 0.419] |
| Min obstacle clearance (m) | 0.932 [0.863, 1.006] | 0.971 [0.888, 1.061] | 1.033 [0.940, 1.130] | 0.874 [0.806, 0.951] | 0.414 [0.364, 0.477] |
| Min boundary clearance (m) | 0.801 [0.797, 0.805] | 0.804 [0.799, 0.808] | 0.804 [0.800, 0.807] | 0.800 [0.796, 0.803] | 0.800 [0.796, 0.803] |
| *Target ship* | | | | | |
| Ship-domain intrusion rate * | 0.071 [0.053, 0.090] | 0.217 [0.187, 0.248] | 0.212 [0.185, 0.241] | 0.101 [0.080, 0.123] | 0.151 [0.125, 0.176] |
| Deepest intrusion, episodes with one (fraction of radius) | 0.200 [0.158, 0.251] | 0.234 [0.207, 0.262] | 0.251 [0.212, 0.292] | 0.205 [0.152, 0.269] | 0.248 [0.197, 0.300] |
| Time inside the domain, episodes with an intrusion (s) | 1.371 [1.217, 1.652] | 1.816 [1.688, 2.377] | 2.345 [1.736, 2.483] | 2.081 [1.679, 2.420] | 1.906 [1.661, 2.236] |
| Closest approach, centre to centre (m) | 2.842 [2.763, 2.927] | 2.429 [2.347, 2.511] | 2.396 [2.315, 2.484] | 2.546 [2.461, 2.635] | 2.150 [2.085, 2.217] |
| Min target clearance, hull to hull (m) | 1.671 [1.601, 1.746] | 1.293 [1.208, 1.377] | 1.336 [1.258, 1.415] | 1.466 [1.404, 1.532] | 1.056 [1.009, 1.107] |
| *Actuation* | | | | | |
| Rudder saturation fraction * | 0.043 [0.039, 0.048] | 0.225 [0.216, 0.234] | 0.297 [0.285, 0.309] | 0.229 [0.219, 0.239] | 0.147 [0.141, 0.153] |
| Mean abs. rudder rate (deg/s) | 21.6 [21.2, 21.9] | 22.5 [22.2, 22.8] | 37.5 [36.4, 38.5] | 24.5 [24.0, 25.0] | 29.9 [29.5, 30.3] |
| Control effort (int. sq. rudder cmd) | 9.560 [9.310, 9.820] | 12.2 [11.8, 12.7] | 12.0 [11.4, 12.6] | 12.7 [12.2, 13.2] | 11.1 [10.7, 11.5] |

## Coupled

| Metric | SAC | PPO | RecurrentPPO | LOS-DWA | COLREGs-VO |
|---|---|---|---|---|---|
| **Success rate** * | 0.735 [0.696, 0.774] | 0.601 [0.554, 0.649] | 0.604 [0.557, 0.652] | 0.571 [0.524, 0.619] | 0.497 [0.449, 0.545] |
| Target collision rate * | 0.077 [0.054, 0.104] | 0.158 [0.122, 0.196] | 0.205 [0.167, 0.244] | 0.119 [0.089, 0.149] | 0.182 [0.146, 0.217] |
| Obstacle collision rate * | 0.167 [0.128, 0.205] | 0.217 [0.179, 0.259] | 0.170 [0.134, 0.208] | 0.250 [0.208, 0.295] | 0.256 [0.211, 0.298] |
| Boundary collision rate * | 0.021 [0.009, 0.036] | 0.024 [0.009, 0.042] | 0.021 [0.006, 0.036] | 0.051 [0.030, 0.074] | 0.065 [0.042, 0.092] |
| Timeout rate * | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.009 [0.000, 0.021] | 0.000 [0.000, 0.000] |
| RMS cross-track error (m) | 1.776 [1.673, 1.874] | 0.963 [0.880, 1.052] | 1.124 [1.038, 1.214] | 0.671 [0.616, 0.728] | 0.613 [0.562, 0.667] |
| Completion time, successful episodes (s) | 26.0 [25.7, 26.2] | 34.1 [33.4, 34.7] | 34.1 [33.6, 35.1] | 39.9 [39.2, 41.1] | 36.1 [35.9, 36.4] |
| Path length / reference, successful episodes | 1.006 [1.002, 1.010] | 0.979 [0.974, 0.985] | 0.981 [0.977, 0.985] | 0.961 [0.959, 0.963] | 0.954 [0.952, 0.956] |
| Mean speed (m/s) | 0.759 [0.749, 0.768] | 0.506 [0.489, 0.522] | 0.486 [0.472, 0.499] | 0.452 [0.443, 0.459] | 0.521 [0.517, 0.523] |
| Minimum speed (m/s) | 0.506 [0.495, 0.515] | 0.260 [0.249, 0.270] | 0.276 [0.265, 0.287] | 0.243 [0.230, 0.257] | 0.351 [0.335, 0.364] |
| Min obstacle clearance (m) | 0.439 [0.395, 0.483] | 0.361 [0.215, 0.407] | 0.532 [0.473, 0.589] | 0.422 [0.245, 0.477] | 0.180 [0.157, 0.306] |
| Min boundary clearance (m) | 0.846 [0.840, 0.852] | 0.848 [0.842, 0.852] | 0.840 [0.835, 0.845] | 0.856 [0.851, 0.860] | 0.859 [0.855, 0.862] |
| *Target ship* | | | | | |
| Ship-domain intrusion rate * | 0.121 [0.091, 0.154] | 0.333 [0.288, 0.382] | 0.327 [0.284, 0.373] | 0.232 [0.193, 0.275] | 0.317 [0.275, 0.363] |
| Deepest intrusion, episodes with one (fraction of radius) | 0.217 [0.164, 0.270] | 0.189 [0.164, 0.220] | 0.276 [0.249, 0.303] | 0.224 [0.186, 0.267] | 0.264 [0.225, 0.301] |
| Time inside the domain, episodes with an intrusion (s) | 1.619 [1.000, 2.020] | 1.893 [1.520, 2.371] | 2.482 [2.021, 2.900] | 2.148 [1.915, 2.347] | 2.217 [1.809, 2.449] |
| Closest approach, centre to centre (m) | 3.017 [2.869, 3.158] | 2.555 [2.409, 2.702] | 2.354 [2.230, 2.485] | 2.372 [2.263, 2.489] | 2.147 [2.062, 2.240] |
| Min target clearance, hull to hull (m) | 1.713 [1.576, 1.844] | 1.226 [1.064, 1.384] | 1.064 [0.628, 1.209] | 1.060 [0.978, 1.146] | 0.819 [0.736, 0.903] |
| *Actuation* | | | | | |
| Rudder saturation fraction * | 0.073 [0.066, 0.082] | 0.251 [0.238, 0.263] | 0.516 [0.504, 0.528] | 0.380 [0.365, 0.393] | 0.175 [0.165, 0.185] |
| Mean abs. rudder rate (deg/s) | 24.0 [23.3, 24.7] | 23.0 [22.5, 23.6] | 48.8 [47.5, 50.0] | 30.3 [29.3, 31.4] | 29.9 [29.3, 30.6] |
| Control effort (int. sq. rudder cmd) | 10.1 [9.8, 10.5] | 12.9 [12.3, 13.5] | 19.6 [18.7, 20.5] | 17.5 [16.5, 18.5] | 10.7 [10.1, 11.4] |

Notes: completion time and path length are over successful episodes only, because a collision truncates an episode; RMS cross-track error is over all episodes, as in Paper 2, so a collision can flatter it. Target-ship rows cover the episodes with a target. The ship domain is own ship's (3.14 m ahead, 1.57 m astern, 1.25 m abeam), entered by the target's centre; intrusion depth and time are over the episodes with an intrusion, since most have none and an IQM over all would read zero. COLREGs behaviour is shown by the snapshot figure, not by per-rule rates. The classical controllers are deterministic, so their intervals reflect episode variance only. Clearances are from the hull polygon with its 0.15 m margin. Path length below 1 reflects the goal test (within 1.25 m of the path end) and cut corners.

## Paired statistics

Against SAC, same episodes (first run of each method). McNemar (exact) on success; Wilcoxon signed-rank on RMS cross-track error and on the minimum hull-to-hull clearance to the target. `both succeeded` restricts to episodes both methods completed.

| Comparison | Episodes | Scope | n | Success A | Success B | McNemar only A / only B | McNemar p | RMS CTE median A / B | RMS CTE median diff | RMS CTE Wilcoxon p | Target clearance median A / B | Target clearance median diff | Target clearance Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
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
| SAC vs COLREGs-VO | All episodes | all paired | 1000 | 0.865 | 0.687 | 240 / 62 | 7.17e-26 | 1.414 / 0.375 | +0.815 | 7.14e-132 | 1.670 / 0.963 | +0.630 | 9.78e-38 |
| SAC vs COLREGs-VO | All episodes | both succeeded | 625 | 0.865 | 0.687 | 240 / 62 | 7.17e-26 | 1.335 / 0.372 | +0.775 | 2.14e-93 | 1.822 / 1.069 | +0.583 | 9.2e-21 |
| SAC vs COLREGs-VO | Decoupled | all paired | 664 | 0.931 | 0.783 | 128 / 30 | 1.29e-15 | 1.307 / 0.273 | +0.809 | 4.21e-94 | 1.650 / 1.020 | +0.554 | 4.22e-19 |
| SAC vs COLREGs-VO | Decoupled | both succeeded | 490 | 0.931 | 0.783 | 128 / 30 | 1.29e-15 | 1.266 / 0.269 | +0.762 | 8.92e-77 | 1.792 / 1.086 | +0.522 | 2.12e-13 |
| SAC vs COLREGs-VO | Coupled | all paired | 336 | 0.735 | 0.497 | 112 / 32 | 1.33e-11 | 1.862 / 0.579 | +0.858 | 1.42e-39 | 1.716 / 0.781 | +0.830 | 8.18e-23 |
| SAC vs COLREGs-VO | Coupled | both succeeded | 135 | 0.735 | 0.497 | 112 / 32 | 1.33e-11 | 2.047 / 0.671 | +0.828 | 4.21e-18 | 2.005 / 0.949 | +1.020 | 9.61e-11 |

