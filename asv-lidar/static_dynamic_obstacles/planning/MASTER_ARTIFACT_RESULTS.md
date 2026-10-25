# Paper 3 results for the master artifact (generated)

Generated 2026-10-25 07:54 by `tools/diagnostics/artifact_results.py` from the recorded result files; chart data in `master_artifact_results.json`. Safety layer off throughout. Test set v4: 1,000 episodes (664 decoupled, 336 coupled). Each run is represented by its best validation-set checkpoint; the final 3.0 M model is the selection-sensitivity check. Rates are fractions of episodes; the bracket after a run's success is its Wilson 95 % interval.

## 1. Status

- SAC: seeds done 0, 1, 2
- TQC: seeds done 0, 1, 2
- PPO: seeds done 0, 1, 2
- RecurrentPPO: seeds done 0, 1, 2
- LOS-PID + DWA: comparators v1 (tuned on the 120 decoupled validation episodes; DWA kept its defaults); re-run under comparators v2
- Encounter-specific VO: pending: comparators v2 (decision S18, A35)
- Deployed policy: the better of SAC and TQC by three-seed mean success on test set v4; within it, the seed with the best validation-set score (decision 2026-10-10). Status: decided by the rule. Deployed: SAC seed 0 (validation score +0.411).

## 2. Test set v4 by run (success; collisions by type)

| Run | Best step | Validation score | Success (Wilson 95 %) | Decoupled | Coupled | Target coll. | Obstacle coll. | Boundary coll. | Timeout | Final 3.0 M model: success (dec. / coup.) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| SAC seed 0 | 3.0 M | +0.411 | 0.865 (0.842–0.885) | 0.931 | 0.735 | 0.053 | 0.070 | 0.012 | 0.000 | 0.859 (0.929 / 0.720) |
| SAC seed 1 | 3.0 M | +0.300 | 0.827 (0.802–0.849) | 0.884 | 0.714 | 0.074 | 0.076 | 0.023 | 0.000 | 0.819 (0.875 / 0.708) |
| SAC seed 2 | final (end check) | +0.367 | 0.845 (0.821–0.866) | 0.887 | 0.762 | 0.063 | 0.076 | 0.016 | 0.000 | same model (the best checkpoint is the final model) |
| TQC seed 0 | final (end check) | +0.267 | 0.837 (0.813–0.859) | 0.895 | 0.723 | 0.069 | 0.075 | 0.019 | 0.000 | same model (the best checkpoint is the final model) |
| TQC seed 1 | 2.8 M | +0.311 | 0.815 (0.790–0.838) | 0.883 | 0.682 | 0.069 | 0.092 | 0.024 | 0.000 | 0.814 (0.881 / 0.682) |
| TQC seed 2 | 3.0 M | +0.378 | 0.838 (0.814–0.860) | 0.896 | 0.723 | 0.063 | 0.079 | 0.020 | 0.000 | 0.839 (0.904 / 0.711) |
| PPO seed 0 | 2.8 M | -0.033 | 0.751 (0.723–0.777) | 0.827 | 0.601 | 0.142 | 0.091 | 0.016 | 0.000 | 0.712 (0.780 / 0.577) |
| PPO seed 1 | 2.4 M | +0.089 | 0.747 (0.719–0.773) | 0.830 | 0.583 | 0.189 | 0.044 | 0.020 | 0.000 | 0.777 (0.849 / 0.634) |
| PPO seed 2 | 3.0 M | +0.100 | 0.753 (0.725–0.779) | 0.810 | 0.640 | 0.130 | 0.064 | 0.053 | 0.000 | 0.756 (0.818 / 0.634) |
| RecurrentPPO seed 0 | 2.4 M | +0.011 | 0.757 (0.729–0.783) | 0.834 | 0.604 | 0.155 | 0.078 | 0.010 | 0.000 | 0.774 (0.858 / 0.607) |
| RecurrentPPO seed 1 | 1.6 M | -0.289 | 0.638 (0.608–0.667) | 0.730 | 0.455 | 0.237 | 0.066 | 0.059 | 0.000 | 0.669 (0.789 / 0.432) |
| RecurrentPPO seed 2 | final (end check) | +0.033 | 0.756 (0.728–0.782) | 0.833 | 0.604 | 0.159 | 0.050 | 0.035 | 0.000 | same model (the best checkpoint is the final model) |
| LOS-PID + DWA (v1) | — | — | 0.770 (0.743–0.795) | 0.870 | 0.571 | 0.080 | 0.109 | 0.038 | 0.003 | deterministic |
| Encounter-specific VO | — | — | pending | | | | | | | |

## 3. Learner means over the finished seeds (best checkpoints; range in brackets)

| Learner | Seeds | Success | Decoupled | Coupled |
|---|---|---|---|---|
| SAC | 0, 1, 2 | 0.846 (0.827–0.865) | 0.901 (0.884–0.931) | 0.737 (0.714–0.762) |
| TQC | 0, 1, 2 | 0.830 (0.815–0.838) | 0.891 (0.883–0.896) | 0.709 (0.682–0.723) |
| PPO | 0, 1, 2 | 0.750 (0.747–0.753) | 0.822 (0.810–0.830) | 0.608 (0.583–0.640) |
| RecurrentPPO | 0, 1, 2 | 0.717 (0.638–0.757) | 0.799 (0.730–0.834) | 0.555 (0.455–0.604) |

## 4. Success by encounter type, whole test set

Success: seed mean (seed range), n = episodes per run.

| Method | Head-on | Crossing from port | Crossing from starboard | Overtaking | Being overtaken | Null (target, no risk) | No target |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAC | 0.901 (0.89–0.91) n=243 | 0.741 (0.70–0.78) n=166 | 0.743 (0.73–0.78) n=135 | 0.859 (0.83–0.88) n=211 | 0.886 (0.85–0.94) n=140 | 0.942 (0.91–0.97) n=75 | 0.911 (0.87–0.93) n=30 |
| TQC | 0.889 (0.87–0.91) n=243 | 0.769 (0.75–0.78) n=166 | 0.714 (0.70–0.73) n=135 | 0.831 (0.80–0.85) n=211 | 0.840 (0.81–0.88) n=140 | 0.929 (0.89–0.95) n=75 | 0.911 (0.90–0.93) n=30 |
| PPO | 0.793 (0.73–0.84) n=243 | 0.641 (0.61–0.69) n=166 | 0.602 (0.49–0.69) n=135 | 0.812 (0.78–0.88) n=211 | 0.719 (0.57–0.84) n=140 | 0.969 (0.96–0.97) n=75 | 0.844 (0.77–0.93) n=30 |
| RecurrentPPO | 0.801 (0.77–0.83) n=243 | 0.554 (0.34–0.69) n=166 | 0.489 (0.35–0.56) n=135 | 0.735 (0.66–0.81) n=211 | 0.812 (0.76–0.87) n=140 | 0.938 (0.89–0.99) n=75 | 0.844 (0.70–0.93) n=30 |
| LOS-PID + DWA (v1) | 0.840 | 0.699 | 0.630 | 0.735 | 0.829 | 0.973 | 0.700 |

## 5. Success by encounter type, decoupled family

Success: seed mean (seed range), n = episodes per run.

| Method | Head-on | Crossing from port | Crossing from starboard | Overtaking | Being overtaken | Null (target, no risk) |
|---|---:|---:|---:|---:|---:|---:|
| SAC | 0.928 (0.91–0.94) n=198 | 0.852 (0.83–0.90) n=115 | 0.800 (0.76–0.85) n=85 | 0.965 (0.94–0.98) n=96 | 0.895 (0.84–0.97) n=95 | 0.942 (0.91–0.97) n=75 |
| TQC | 0.918 (0.91–0.92) n=198 | 0.855 (0.83–0.88) n=115 | 0.812 (0.79–0.85) n=85 | 0.951 (0.94–0.97) n=96 | 0.860 (0.81–0.94) n=95 | 0.929 (0.89–0.95) n=75 |
| PPO | 0.854 (0.80–0.89) n=198 | 0.719 (0.70–0.75) n=115 | 0.671 (0.52–0.78) n=85 | 0.941 (0.92–0.98) n=96 | 0.782 (0.57–0.96) n=95 | 0.969 (0.96–0.97) n=75 |
| RecurrentPPO | 0.879 (0.84–0.92) n=198 | 0.597 (0.42–0.71) n=115 | 0.537 (0.44–0.61) n=85 | 0.917 (0.82–0.98) n=96 | 0.884 (0.84–0.96) n=95 | 0.938 (0.89–0.99) n=75 |
| LOS-PID + DWA (v1) | 0.914 | 0.765 | 0.647 | 0.979 | 0.916 | 0.973 |

## 6. Success by encounter type, coupled family

Success: seed mean (seed range), n = episodes per run.

| Method | Head-on | Crossing from port | Crossing from starboard | Overtaking | Being overtaken | No target |
|---|---:|---:|---:|---:|---:|---:|
| SAC | 0.785 (0.78–0.80) n=45 | 0.490 (0.37–0.67) n=51 | 0.647 (0.62–0.66) n=50 | 0.771 (0.75–0.79) n=115 | 0.867 (0.80–0.91) n=45 | 0.911 (0.87–0.93) n=30 |
| TQC | 0.763 (0.67–0.84) n=45 | 0.575 (0.55–0.61) n=51 | 0.547 (0.52–0.58) n=50 | 0.730 (0.66–0.77) n=115 | 0.800 (0.76–0.84) n=45 | 0.911 (0.90–0.93) n=30 |
| PPO | 0.526 (0.42–0.64) n=45 | 0.464 (0.39–0.57) n=51 | 0.487 (0.44–0.54) n=50 | 0.704 (0.65–0.79) n=115 | 0.585 (0.58–0.60) n=45 | 0.844 (0.77–0.93) n=30 |
| RecurrentPPO | 0.459 (0.44–0.49) n=45 | 0.458 (0.16–0.63) n=51 | 0.407 (0.20–0.56) n=50 | 0.583 (0.53–0.66) n=115 | 0.659 (0.60–0.69) n=45 | 0.844 (0.70–0.93) n=30 |
| LOS-PID + DWA (v1) | 0.511 | 0.549 | 0.600 | 0.530 | 0.644 | 0.700 |

## 7. Success by decoupled cell

Success: seed mean (seed range), n = episodes per run.

| Method | basin-head_on | basin-crossing | basin-overtaking | basin-being_overtaken | basin-null | channel-head_on | channel-crossing | channel-overtaking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAC | 0.940 (0.92–0.98) n=100 | 0.820 (0.79–0.86) n=100 | 0.963 (0.96–0.96) n=54 | 0.895 (0.84–0.97) n=95 | 0.942 (0.91–0.97) n=75 | 0.915 (0.88–0.96) n=98 | 0.840 (0.81–0.89) n=100 | 0.968 (0.90–1.00) n=42 |
| TQC | 0.927 (0.91–0.95) n=100 | 0.847 (0.82–0.88) n=100 | 0.938 (0.89–0.96) n=54 | 0.860 (0.81–0.94) n=95 | 0.929 (0.89–0.95) n=75 | 0.908 (0.90–0.92) n=98 | 0.827 (0.78–0.89) n=100 | 0.968 (0.93–1.00) n=42 |
| PPO | 0.810 (0.74–0.87) n=100 | 0.700 (0.68–0.72) n=100 | 0.926 (0.89–0.98) n=54 | 0.782 (0.57–0.96) n=95 | 0.969 (0.96–0.97) n=75 | 0.898 (0.86–0.93) n=98 | 0.697 (0.57–0.80) n=100 | 0.960 (0.95–0.98) n=42 |
| RecurrentPPO | 0.823 (0.79–0.88) n=100 | 0.577 (0.38–0.69) n=100 | 0.901 (0.78–0.96) n=54 | 0.884 (0.84–0.96) n=95 | 0.938 (0.89–0.99) n=75 | 0.935 (0.90–0.96) n=98 | 0.567 (0.47–0.62) n=100 | 0.937 (0.88–1.00) n=42 |
| LOS-PID + DWA (v1) | 0.890 | 0.720 | 0.963 | 0.916 | 0.973 | 0.939 | 0.710 | 1.000 |

## 8. Success by stratum, decoupled (all classes)

Success: seed mean (seed range), n = episodes per run.

| Method | basin | channel |
|---|---:|---:|
| SAC | 0.905 (0.89–0.93) n=424 | 0.893 (0.87–0.94) n=240 |
| TQC | 0.895 (0.89–0.90) n=424 | 0.885 (0.87–0.90) n=240 |
| PPO | 0.821 (0.79–0.86) n=424 | 0.825 (0.78–0.85) n=240 |
| RecurrentPPO | 0.809 (0.73–0.86) n=424 | 0.782 (0.74–0.82) n=240 |
| LOS-PID + DWA (v1) | 0.880 | 0.854 |

## 9. Success by stratum, classes both strata hold (head-on, crossing, overtaking)

Success: seed mean (seed range), n = episodes per run.

| Method | basin | channel |
|---|---:|---:|
| SAC | 0.898 (0.88–0.91) n=254 | 0.893 (0.87–0.94) n=240 |
| TQC | 0.898 (0.87–0.91) n=254 | 0.885 (0.87–0.90) n=240 |
| PPO | 0.791 (0.76–0.83) n=254 | 0.825 (0.78–0.85) n=240 |
| RecurrentPPO | 0.743 (0.63–0.81) n=254 | 0.782 (0.74–0.82) n=240 |
| LOS-PID + DWA (v1) | 0.839 | 0.854 |

## 10. Success by target behaviour, decoupled family

Success: seed mean (seed range), n = episodes per run.

| Method | CV: constant velocity | RE: reactive (gives way by the COLREGs-VO rule from its own side) | NC: non-compliant (head-on: turns to port) |
|---|---:|---:|---:|
| SAC | 0.903 (0.87–0.94) n=340 | 0.901 (0.89–0.92) n=258 | 0.889 (0.79–0.97) n=66 |
| TQC | 0.885 (0.87–0.90) n=340 | 0.895 (0.88–0.92) n=258 | 0.904 (0.86–0.92) n=66 |
| PPO | 0.775 (0.76–0.78) n=340 | 0.877 (0.84–0.91) n=258 | 0.854 (0.83–0.88) n=66 |
| RecurrentPPO | 0.755 (0.72–0.77) n=340 | 0.823 (0.71–0.88) n=258 | 0.934 (0.88–0.97) n=66 |
| LOS-PID + DWA (v1) | 0.835 | 0.888 | 0.985 |

## 11. Success by target behaviour, decoupled head-on only

Success: seed mean (seed range), n = episodes per run.

| Method | CV | RE | NC |
|---|---:|---:|---:|
| SAC | 0.975 (0.97–0.98) n=66 | 0.919 (0.85–0.98) n=66 | 0.889 (0.79–0.97) n=66 |
| TQC | 0.934 (0.91–0.95) n=66 | 0.914 (0.88–0.94) n=66 | 0.904 (0.86–0.92) n=66 |
| PPO | 0.783 (0.65–0.88) n=66 | 0.924 (0.91–0.94) n=66 | 0.854 (0.83–0.88) n=66 |
| RecurrentPPO | 0.818 (0.73–0.89) n=66 | 0.884 (0.85–0.95) n=66 | 0.934 (0.88–0.97) n=66 |
| LOS-PID + DWA (v1) | 0.818 | 0.939 | 0.985 |

## 12. Success by target speed, coupled family

Success: seed mean (seed range), n = episodes per run.

| Method | constant speed | one speed change on the approach | no target |
|---|---:|---:|---:|
| SAC | 0.776 (0.73–0.81) n=156 | 0.662 (0.62–0.71) n=150 | 0.911 (0.87–0.93) n=30 |
| TQC | 0.724 (0.72–0.73) n=156 | 0.653 (0.59–0.69) n=150 | 0.911 (0.90–0.93) n=30 |
| PPO | 0.613 (0.60–0.63) n=156 | 0.556 (0.50–0.61) n=150 | 0.844 (0.77–0.93) n=30 |
| RecurrentPPO | 0.528 (0.41–0.59) n=156 | 0.524 (0.45–0.57) n=150 | 0.844 (0.70–0.93) n=30 |
| LOS-PID + DWA (v1) | 0.564 | 0.553 | 0.700 |

## 13. Pooled table (seeds pooled; stratified bootstrap 95 % intervals)

From `results/baseline_v3/paper_table.csv`. Runs pooled per method: SAC: sacs0_bl3, sacs1_bl3, sacs2_bl3; TQC: tqcs0_bl3, tqcs1_bl3, tqcs2_bl3; PPO: ppos0_bl3, ppos1_bl3, ppos2_bl3; RecurrentPPO: recurrent_ppos0_bl3, recurrent_ppos1_bl3, recurrent_ppos2_bl3; LOS-DWA: los_dwa.

### All episodes

| Metric | SAC | TQC | PPO | RecurrentPPO | LOS-DWA |
|---|---|---|---|---|---|
| Success rate | 0.846 [0.834, 0.857] | 0.830 [0.817, 0.842] | 0.750 [0.736, 0.764] | 0.717 [0.702, 0.731] | 0.770 [0.747, 0.793] |
| Target collision rate | 0.063 [0.055, 0.072] | 0.067 [0.058, 0.076] | 0.154 [0.141, 0.166] | 0.184 [0.172, 0.196] | 0.080 [0.065, 0.095] |
| Obstacle collision rate | 0.074 [0.065, 0.083] | 0.082 [0.073, 0.091] | 0.066 [0.058, 0.075] | 0.065 [0.056, 0.073] | 0.109 [0.093, 0.126] |
| Boundary collision rate | 0.017 [0.013, 0.022] | 0.021 [0.016, 0.026] | 0.030 [0.024, 0.036] | 0.035 [0.029, 0.041] | 0.038 [0.027, 0.050] |
| Timeout rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.000, 0.007] |
| RMS cross-track error (m) | 1.370 [1.342, 1.398] | 1.269 [1.241, 1.298] | 0.994 [0.968, 1.022] | 0.905 [0.877, 0.932] | 0.515 [0.482, 0.550] |
| Completion time, successful episodes (s) | 25.8 [25.8, 26.0] | 27.3 [27.1, 27.5] | 30.1 [29.6, 30.2] | 31.0 [30.8, 31.2] | 36.5 [36.2, 36.9] |
| Path length / reference, successful episodes | 0.990 [0.988, 0.991] | 0.986 [0.985, 0.987] | 0.970 [0.969, 0.971] | 0.965 [0.964, 0.966] | 0.953 [0.952, 0.954] |
| Mean speed (m/s) | 0.749 [0.746, 0.752] | 0.713 [0.711, 0.716] | 0.608 [0.604, 0.612] | 0.602 [0.599, 0.605] | 0.495 [0.491, 0.499] |
| Minimum speed (m/s) | 0.504 [0.500, 0.508] | 0.503 [0.499, 0.506] | 0.389 [0.383, 0.395] | 0.429 [0.423, 0.434] | 0.322 [0.312, 0.333] |
| Min obstacle clearance (m) | 0.664 [0.641, 0.687] | 0.599 [0.578, 0.620] | 0.746 [0.718, 0.776] | 0.838 [0.808, 0.868] | 0.634 [0.591, 0.678] |
| Min boundary clearance (m) | 0.813 [0.810, 0.816] | 0.818 [0.816, 0.821] | 0.818 [0.815, 0.821] | 0.816 [0.814, 0.819] | 0.824 [0.820, 0.828] |
| Ship-domain intrusion rate | 0.112 [0.101, 0.123] | 0.125 [0.114, 0.137] | 0.239 [0.225, 0.253] | 0.251 [0.236, 0.265] | 0.142 [0.123, 0.162] |
| Deepest intrusion, episodes with one (fraction of radius) | 0.208 [0.188, 0.228] | 0.192 [0.175, 0.209] | 0.232 [0.218, 0.247] | 0.271 [0.257, 0.285] | 0.215 [0.182, 0.251] |
| Time inside the domain, episodes with an intrusion (s) | 1.567 [1.420, 1.636] | 1.622 [1.430, 1.684] | 1.755 [1.619, 1.813] | 2.296 [1.811, 2.353] | 2.117 [1.900, 2.318] |
| Closest approach, centre to centre (m) | 2.828 [2.783, 2.872] | 2.705 [2.659, 2.748] | 2.462 [2.418, 2.505] | 2.351 [2.311, 2.393] | 2.473 [2.405, 2.540] |
| Min target clearance, hull to hull (m) | 1.619 [1.580, 1.657] | 1.519 [1.481, 1.558] | 1.299 [1.257, 1.340] | 1.205 [1.162, 1.247] | 1.330 [1.279, 1.384] |
| Rudder saturation fraction | 0.050 [0.048, 0.053] | 0.110 [0.106, 0.113] | 0.303 [0.298, 0.308] | 0.342 [0.335, 0.349] | 0.279 [0.271, 0.288] |
| Mean abs. rudder rate (deg/s) | 23.7 [23.5, 23.9] | 26.8 [26.6, 27.0] | 31.5 [31.0, 32.0] | 38.6 [38.2, 39.1] | 26.2 [25.7, 26.6] |
| Control effort (int. sq. rudder cmd) | 10.4 [10.3, 10.5] | 12.9 [12.8, 13.1] | 12.4 [12.2, 12.6] | 11.7 [11.5, 12.0] | 14.1 [13.6, 14.5] |

### Decoupled

| Metric | SAC | TQC | PPO | RecurrentPPO | LOS-DWA |
|---|---|---|---|---|---|
| Success rate | 0.901 [0.888, 0.914] | 0.891 [0.878, 0.905] | 0.822 [0.807, 0.840] | 0.799 [0.783, 0.815] | 0.870 [0.846, 0.895] |
| Target collision rate | 0.060 [0.049, 0.070] | 0.056 [0.046, 0.066] | 0.132 [0.118, 0.147] | 0.161 [0.146, 0.177] | 0.060 [0.042, 0.078] |
| Obstacle collision rate | 0.025 [0.019, 0.032] | 0.033 [0.025, 0.040] | 0.021 [0.015, 0.028] | 0.022 [0.016, 0.029] | 0.038 [0.024, 0.053] |
| Boundary collision rate | 0.015 [0.010, 0.020] | 0.020 [0.014, 0.027] | 0.025 [0.018, 0.032] | 0.018 [0.012, 0.024] | 0.032 [0.020, 0.045] |
| Timeout rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| RMS cross-track error (m) | 1.245 [1.215, 1.274] | 1.165 [1.134, 1.197] | 0.960 [0.929, 0.990] | 0.806 [0.775, 0.838] | 0.436 [0.400, 0.477] |
| Completion time, successful episodes (s) | 25.8 [25.6, 26.1] | 27.5 [27.2, 27.6] | 29.6 [29.5, 30.0] | 30.8 [30.6, 31.0] | 35.7 [35.5, 36.0] |
| Path length / reference, successful episodes | 0.983 [0.982, 0.984] | 0.980 [0.979, 0.981] | 0.965 [0.964, 0.966] | 0.960 [0.959, 0.961] | 0.951 [0.950, 0.952] |
| Mean speed (m/s) | 0.745 [0.741, 0.748] | 0.708 [0.705, 0.711] | 0.617 [0.612, 0.622] | 0.607 [0.604, 0.610] | 0.516 [0.512, 0.519] |
| Minimum speed (m/s) | 0.513 [0.508, 0.517] | 0.509 [0.505, 0.513] | 0.417 [0.409, 0.425] | 0.467 [0.461, 0.472] | 0.377 [0.364, 0.389] |
| Min obstacle clearance (m) | 0.975 [0.931, 1.021] | 0.784 [0.748, 0.821] | 1.142 [1.088, 1.202] | 1.192 [1.134, 1.251] | 0.874 [0.805, 0.947] |
| Min boundary clearance (m) | 0.798 [0.795, 0.801] | 0.799 [0.796, 0.802] | 0.802 [0.800, 0.805] | 0.802 [0.799, 0.804] | 0.800 [0.796, 0.803] |
| Ship-domain intrusion rate | 0.104 [0.091, 0.117] | 0.108 [0.094, 0.122] | 0.200 [0.184, 0.217] | 0.221 [0.205, 0.238] | 0.101 [0.080, 0.122] |
| Deepest intrusion, episodes with one (fraction of radius) | 0.205 [0.180, 0.231] | 0.181 [0.158, 0.208] | 0.233 [0.213, 0.253] | 0.266 [0.245, 0.287] | 0.205 [0.153, 0.268] |
| Time inside the domain, episodes with an intrusion (s) | 1.536 [1.398, 1.780] | 1.465 [1.405, 1.675] | 1.728 [1.549, 1.802] | 2.108 [1.777, 2.322] | 2.081 [1.671, 2.425] |
| Closest approach, centre to centre (m) | 2.755 [2.707, 2.806] | 2.657 [2.607, 2.707] | 2.446 [2.400, 2.496] | 2.361 [2.314, 2.412] | 2.546 [2.461, 2.637] |
| Min target clearance, hull to hull (m) | 1.589 [1.547, 1.631] | 1.502 [1.460, 1.547] | 1.341 [1.294, 1.385] | 1.284 [1.239, 1.333] | 1.466 [1.406, 1.531] |
| Rudder saturation fraction | 0.040 [0.037, 0.043] | 0.095 [0.090, 0.099] | 0.285 [0.279, 0.291] | 0.267 [0.259, 0.276] | 0.229 [0.219, 0.239] |
| Mean abs. rudder rate (deg/s) | 23.2 [23.0, 23.4] | 26.5 [26.2, 26.7] | 30.8 [30.2, 31.3] | 36.0 [35.4, 36.6] | 24.5 [24.0, 25.0] |
| Control effort (int. sq. rudder cmd) | 10.1 [9.9, 10.2] | 12.7 [12.5, 12.9] | 12.1 [11.9, 12.3] | 10.4 [10.1, 10.6] | 12.7 [12.3, 13.2] |

### Coupled

| Metric | SAC | TQC | PPO | RecurrentPPO | LOS-DWA |
|---|---|---|---|---|---|
| Success rate | 0.737 [0.711, 0.762] | 0.709 [0.683, 0.735] | 0.608 [0.579, 0.635] | 0.555 [0.526, 0.583] | 0.571 [0.524, 0.622] |
| Target collision rate | 0.070 [0.057, 0.085] | 0.088 [0.072, 0.105] | 0.196 [0.174, 0.219] | 0.228 [0.205, 0.252] | 0.119 [0.089, 0.149] |
| Obstacle collision rate | 0.171 [0.149, 0.193] | 0.180 [0.156, 0.202] | 0.156 [0.135, 0.179] | 0.149 [0.127, 0.171] | 0.250 [0.208, 0.295] |
| Boundary collision rate | 0.022 [0.014, 0.031] | 0.023 [0.015, 0.032] | 0.040 [0.029, 0.052] | 0.068 [0.054, 0.083] | 0.051 [0.030, 0.074] |
| Timeout rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.009 [0.000, 0.021] |
| RMS cross-track error (m) | 1.759 [1.697, 1.818] | 1.588 [1.528, 1.648] | 1.104 [1.047, 1.162] | 1.137 [1.087, 1.190] | 0.671 [0.618, 0.727] |
| Completion time, successful episodes (s) | 26.0 [25.8, 26.1] | 27.2 [26.9, 27.4] | 30.4 [30.1, 31.0] | 31.9 [31.5, 32.4] | 39.9 [39.2, 41.1] |
| Path length / reference, successful episodes | 1.011 [1.008, 1.013] | 1.007 [1.004, 1.011] | 0.988 [0.985, 0.992] | 0.982 [0.980, 0.985] | 0.961 [0.959, 0.963] |
| Mean speed (m/s) | 0.759 [0.753, 0.765] | 0.726 [0.720, 0.731] | 0.582 [0.572, 0.591] | 0.577 [0.566, 0.587] | 0.452 [0.444, 0.459] |
| Minimum speed (m/s) | 0.486 [0.478, 0.493] | 0.489 [0.482, 0.496] | 0.337 [0.328, 0.347] | 0.342 [0.330, 0.353] | 0.243 [0.230, 0.257] |
| Min obstacle clearance (m) | 0.440 [0.412, 0.467] | 0.449 [0.420, 0.477] | 0.474 [0.445, 0.503] | 0.590 [0.557, 0.622] | 0.422 [0.246, 0.478] |
| Min boundary clearance (m) | 0.841 [0.836, 0.845] | 0.849 [0.845, 0.852] | 0.842 [0.839, 0.845] | 0.840 [0.836, 0.843] | 0.856 [0.851, 0.860] |
| Ship-domain intrusion rate | 0.129 [0.109, 0.148] | 0.163 [0.142, 0.185] | 0.324 [0.296, 0.350] | 0.314 [0.288, 0.340] | 0.232 [0.193, 0.275] |
| Deepest intrusion, episodes with one (fraction of radius) | 0.214 [0.181, 0.248] | 0.205 [0.184, 0.227] | 0.232 [0.209, 0.255] | 0.278 [0.259, 0.296] | 0.224 [0.186, 0.267] |
| Time inside the domain, episodes with an intrusion (s) | 1.493 [1.424, 1.756] | 1.653 [1.427, 1.909] | 1.791 [1.659, 2.202] | 2.320 [1.921, 2.559] | 2.148 [1.912, 2.362] |
| Closest approach, centre to centre (m) | 3.009 [2.917, 3.096] | 2.815 [2.730, 2.902] | 2.492 [2.406, 2.583] | 2.319 [2.241, 2.399] | 2.372 [2.267, 2.485] |
| Min target clearance, hull to hull (m) | 1.692 [1.611, 1.771] | 1.547 [1.466, 1.629] | 1.173 [1.081, 1.268] | 0.658 [0.600, 1.077] | 1.060 [0.976, 1.146] |
| Rudder saturation fraction | 0.071 [0.067, 0.076] | 0.139 [0.132, 0.146] | 0.339 [0.331, 0.348] | 0.488 [0.476, 0.500] | 0.380 [0.366, 0.394] |
| Mean abs. rudder rate (deg/s) | 24.9 [24.5, 25.3] | 27.6 [27.1, 28.1] | 33.2 [32.3, 34.2] | 44.0 [43.1, 44.9] | 30.3 [29.3, 31.4] |
| Control effort (int. sq. rudder cmd) | 11.1 [10.9, 11.4] | 13.5 [13.2, 13.7] | 12.9 [12.5, 13.3] | 15.2 [14.7, 15.7] | 17.5 [16.5, 18.6] |

## 14. Validation-set learning curves (score = goal − 2 × collision; safety off)

| Run | Best step | Best score | Score at 1.0 / 2.0 / 3.0 M |
|---|---:|---:|---|
| SAC seed 0 | 3.0 M | +0.411 | -0.522 / +0.122 / +0.411 |
| SAC seed 1 | 3.0 M | +0.300 | -0.467 / -0.011 / +0.300 |
| SAC seed 2 | final (end check) | +0.367 | -0.367 / -0.044 / +0.322 |
| TQC seed 0 | final (end check) | +0.267 | -0.400 / -0.067 / +0.222 |
| TQC seed 1 | 2.8 M | +0.311 | -0.578 / -0.011 / +0.200 |
| TQC seed 2 | 3.0 M | +0.378 | -0.511 / +0.144 / +0.378 |
| PPO seed 0 | 2.8 M | -0.033 | -0.489 / -0.167 / -0.100 |
| PPO seed 1 | 2.4 M | +0.089 | -0.367 / -0.267 / -0.044 |
| PPO seed 2 | 3.0 M | +0.100 | -0.467 / -0.056 / +0.100 |
| RecurrentPPO seed 0 | 2.4 M | +0.011 | -0.333 / -0.100 / -0.189 |
| RecurrentPPO seed 1 | 1.6 M | -0.289 | -0.389 / -0.911 / -0.422 |
| RecurrentPPO seed 2 | final (end check) | +0.033 | -0.344 / -0.089 / -0.033 |

## 15. Compute (ideal conditions, per run)

| Learner | Training (h) | Evaluations (h) | Steps/s, stages 1–7 |
|---|---:|---:|---|
| PPO | 11.2 | 0.9 | 223, 188, 131, 124, 118, 66, 55 |
| RecurrentPPO | 12.5 | 0.9 | 89, 96, 99, 113, 109, 62, 52 |
| SAC | 45.3 | 0.9 | 23, 21, 20, 20, 20, 18, 17 |
| TQC | 55.7 | 0.9 | 17, 16, 16, 16, 16, 15, 14 |

## 16. Notes

- RecurrentPPO seed 1 collapsed mid-training: between 1.0 and 2.4 M steps, 258 of 273 updates stopped at the first epoch on the KL limit, the validation-set score fell to -0.91 at 2.0 M, and it recovered after 2.4 M. No defect was found, so the seed is kept and reported beside seeds 0 and 2, which show none of it (seed retention, skeleton S16; planning/PAPER3_CLAIMS_AND_TABLES.md).
- Decoupled channel cells: recorded nominal widths 6.9-10.6 m (median 8.7 m); 16 of 240 below 7.60 m.

## 17. Reward scale audit of the deployed policy (Table R7)

Run `sacs0_bl3_audit`: SAC seed 0 (the deployed policy), best validation-set checkpoint, on test set v4 (1000 episodes), safety layer off. From `results/baseline_v3/reward_scale_audit.csv (tools/diagnostics/reward_scale_audit.py)`. Weight = the term's largest per-step contribution; integral = mean weighted episode integral; share = mean share of the episode's absolute dense reward; active = share of episodes in which the term was ever non-zero; per step = mean magnitude per step when active.

- Terms add up to the return: largest per-episode gap 1.28e-13 over 1000 episodes.
- Same episodes as the Table R1 run `sacs0_bl3_metrics`: outcome identical in 1000 of 1000, steps identical in 1000, largest return difference 0.00e+00.

### All episodes

| Term | Weight | Mean episode integral | Share of dense abs. | Episodes active | Per step when active |
|---|---:|---:|---:|---:|---:|
| Boundary r_bnd | 15.00 | -1.25 | 0.004 | 0.034 | 5.358 |
| Ship domain r_dom | 12.50 | -0.17 | 0.002 | 0.084 | 0.470 |
| Static obstacle r_obs | 11.00 | -20.60 | 0.094 | 0.616 | 2.059 |
| COLREGs group r_col | 9.00 | -26.86 | 0.126 | 0.840 | 2.337 |
| Path following r_pf | 3.00 | -66.39 | 0.306 | 1.000 | 1.373 |
| Progress r_prog | 1.50 | +69.71 | 0.338 | 1.000 | 1.452 |
| Smoothness r_smooth | 0.50 | -14.62 | 0.071 | 1.000 | 0.304 |
| Existence r_exist | 0.25 | -11.94 | 0.058 | 1.000 | 0.250 |
| Terminal (+100 goal, -300 collision) | — | +46.00 | — | 1.000 | — |
| Episode return | — | -26.12 | — | — | — |

Order by per-step magnitude when active: bnd > col > obs > prog > pf > dom > smooth > exist (designed, by weight: bnd > dom > obs > col > pf > prog > smooth > exist).

### Per step when active, by scope

| Term | All episodes (n = 1000) | Decoupled (n = 664) | Coupled (n = 336) | Goal episodes (n = 865) | Collision episodes (n = 135) |
|---|---:|---:|---:|---:|---:|
| Boundary r_bnd | 5.358 | 5.393 | 5.313 | 2.699 | 9.156 |
| Ship domain r_dom | 0.470 | 0.437 | 0.512 | 0.203 | 0.627 |
| Static obstacle r_obs | 2.059 | 1.623 | 2.444 | 1.818 | 3.318 |
| COLREGs group r_col | 2.337 | 2.332 | 2.346 | 2.274 | 2.710 |
| Path following r_pf | 1.373 | 1.305 | 1.508 | 1.445 | 0.915 |
| Progress r_prog | 1.452 | 1.460 | 1.437 | 1.464 | 1.381 |
| Smoothness r_smooth | 0.304 | 0.303 | 0.307 | 0.303 | 0.308 |
| Existence r_exist | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 |

