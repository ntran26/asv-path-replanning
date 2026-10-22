# Checkpoints removed from the working tree (2026-10-22)

Removed to free disk space after the disk filled during TQC seed 2 (2026-10-22 22:47). Every
file is kept in the repository history: the last commit holding them is `2ba3dbdf`, so one is
restored with

    git checkout 2ba3dbdf -- runs/<run>/<file>

and the full list is `removed_checkpoints_2026-10-22.txt` beside this note.

What was removed:

- the intermediate checkpoints (`<algo>_<N>_steps.zip` and `<algo>_vecnormalize_<N>_steps.pkl`)
  of every finished run. They serve only to resume a run, and no finished run is resumed (seed
  retention, skeleton S16);
- the model files (`.zip`, `.pkl`) of runs not adopted or not completed: the SAC baseline-v4
  run and the v3/v4.3 pair runs (`BASELINE_V4_PLAN.md`, section 5e), and the set-aside
  RecurrentPPO seed 0 attempts.

What stays: each finished run's `best_model.zip`, `final_model.zip`, `final_model_2500000.zip`
and their VecNormalize statistics, its `config.json`, `curriculum.json`, `eval_summary.json`,
`eval_episodes.csv` and `monitor.csv`, the training logs, and every file of the run in progress
(TQC seed 2, which needs its checkpoints to resume).

| Run directory | Files removed |
|---|---:|
| `pilot_v4_A` | 4 |
| `pilot_v4_B` | 4 |
| `ppo_formulation_seed0` | 14 |
| `ppo_formulation_seed0_bl3` | 24 |
| `ppo_formulation_seed0_tier2_a21` | 2 |
| `ppo_formulation_seed0_tier2_a22` | 2 |
| `ppo_formulation_seed0_v10` | 16 |
| `ppo_formulation_seed0_v11` | 16 |
| `ppo_formulation_seed0_v12` | 16 |
| `ppo_formulation_seed0_v2` | 16 |
| `ppo_formulation_seed0_v3` | 16 |
| `ppo_formulation_seed0_v4` | 16 |
| `ppo_formulation_seed0_v5` | 16 |
| `ppo_formulation_seed0_v6` | 16 |
| `ppo_formulation_seed0_v7` | 16 |
| `ppo_formulation_seed0_v8` | 16 |
| `ppo_formulation_seed0_v9` | 16 |
| `ppo_formulation_seed1_bl3` | 24 |
| `ppo_formulation_seed1_v11` | 16 |
| `ppo_formulation_seed1_v12` | 16 |
| `ppo_formulation_seed2_bl3` | 24 |
| `recurrent_ppo_formulation_seed0_bl3` | 24 |
| `recurrent_ppo_formulation_seed1_bl3` | 24 |
| `recurrent_ppo_formulation_seed2_bl3` | 24 |
| `sac_formulation_seed0_bl3_v3pair` | 8 |
| `sac_formulation_seed0_bl3_v43pair` | 8 |
| `sac_formulation_seed0_bl4` | 28 |
| `sac_formulation_seed1_bl3` | 24 |
| `sac_formulation_seed2_bl3` | 24 |
| `tqc_formulation_seed0_bl3` | 24 |
| `tqc_formulation_seed1_bl3` | 24 |
