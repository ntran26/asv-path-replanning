# `manuscript_results/` — files backing the revision

Self-contained copy of everything the revised manuscript's baseline comparison
draws on, gathered so each number in the paper can be traced to a file.

**Scope.** The frozen 500-episode evaluation set: 0–4 obstacles, straight
reference paths. Every method measured by the same harness, on the same layouts,
with the same observation, action space, reward, termination rule and episode cap.

**Rebuilt 2026-08-19** after a curriculum error was found and corrected. All PPO
results here are from runs under the *correct* fixed-RPM setup. See
`BASELINES_RESULTS.md` §4 for the retraction and the evidence.

---

## The three reported methods

| Table row | Built from | Runs |
|---|---|---|
| SAC (proposed) | `per_episode/sac_1M/` | 1 deployed checkpoint |
| PPO (final @1M) | `per_episode/ppo_fx_seed{0,1,2}_final/` | 3 training seeds |
| LOS+APF (tuned) | `per_episode/los_apf_s{1,2,3}/` | 3 tuning searches |

`ppo_fx_` = fixed-RPM, the corrected setup. PPO is reported from its **final
1M-step checkpoint** — the same "model at 1M steps" SAC is reported from, with no
best-checkpoint selection.

---

## Where each manuscript number comes from

### Tables

`tables/comparison_table.md` — formatted, with stratified bootstrap 95 % CIs.
`tables/comparison_table.csv` — machine-readable, `__mean` / `__iqm` /
`__ci_lo` / `__ci_hi` / `__per_run` per metric.

Rate metrics (success, collisions, timeout, rudder saturation) are **means** —
that is what a rate is. Continuous metrics are **IQM**. IQM is not used for rates
because it is degenerate on a binary variable.

`tables/paired_stats_table.md` / `.csv`, detail in `tables/paired_stats.json`.
Everything paired on `episode_id` against `per_episode/sac_1M/`; McNemar exact,
Wilcoxon signed-rank.

### Figures

| Figure | File | Data behind it |
|---|---|---|
| Learning curves | `figures/learning_curves.{png,svg}` | `figures/learning_curves.csv`. SAC curve recovered from `sac_log/asv_sac_2/events.out.tfevents.1781347673.*`; PPO curves from `run_config/ppo_seed*/eval_summary.json` |
| Qualitative trajectories | `figures/trajectories.{png,svg}` | Re-run live from `layouts/eval_layouts_v1.json`, cases 220 / 320 / 420 |

Both use distinct line styles **and** markers rather than colour alone,
addressing the existing reviewer comment.

`figures/learning_curves.csv` matters: the 0–1M SAC evaluation series exists
nowhere else in extractable form — the root `eval_summary.json` was overwritten
by a later fine-tuning run. Restyle the figure from this CSV, not from
TensorBoard.

---

## Supporting records a reviewer may ask for

| Question | File |
|---|---|
| PPO hyperparameters? | `run_config/ppo_seed{0,1,2}/hyperparameters.json` |
| Was a curriculum active? | `run_config/*/curriculum.json` — one entry, stage 0, fixed RPM |
| Classical baseline tuning budget? | `tuning/apf_tuning_results{,_s2,_s3}.csv` — 3 × 250 configurations, all scores |
| Which LOS+APF parameters? | `tuning/los_apf_best{,_s2,_s3}.json` |
| What was evaluated on? | `layouts/eval_layouts_v1.json` — the 500 frozen layouts |
| Was tuning done on the evaluation set? | No — `layouts/tune_layouts_v1.json`, disjoint seeds and case ids |
| Which SAC checkpoint is canonical? | `supporting/sac_1M_alt/` — the rejected alternative |

`tuning/` holds **750 configurations** (3 independent searches of 250, 25 000
episodes each). This is the record if the classical baseline is challenged as
under-tuned.

---

## Per-episode schema

Each `per_episode/*/episodes.csv` has one row per evaluation episode, joined
across methods on `episode_id`. Two definitions worth stating in the paper
because they are not the obvious ones:

* **`min_obstacle_clearance`** — exact polygon-to-polygon distance from the
  inflated hull footprint to the nearest obstacle surface, zero on contact. Not a
  LiDAR beam range: the sensor sits 0.8625 m forward of the vessel origin, so beam
  range overstates true clearance (0.523 m against a true 0.091 m on a spot check).
* **`min_border_clearance`** — minimises over all four walls and is therefore
  floored by the start pose at 0.9875 m in every episode, for every method. Use
  **`min_lateral_border_clearance`** for any corridor-keeping claim. This is the
  Table 4 correction.

---

## Deliberately not included

* **Staged-curriculum runs.** Withdrawn — they used a propulsion curriculum the
  published SAC run never had. Archived at `models/_archive_staged_curriculum/`
  and `eval_results/_archive_staged_curriculum/`.
* **The three SAC retrains.** Same defect; withdrawn with the above. If a
  multi-seed SAC result is wanted it needs a fresh retrain under the corrected
  setup.
* **The out-of-distribution study.** Exploratory, not reported in the manuscript.
  Protocol and data at `OOD_PROTOCOL.md` and `eval_results/baselines/ood_*/`.

---

## Provenance

`BASELINES_RESULTS.md` — §0 is the manuscript-ready summary: tables, figures, and
an explicit list of claims the data supports versus claims it does not. §4 is the
retraction of the earlier PPO-collapse finding.

`BASELINES_NOTES.md` — what the code actually does, and where the task
assumptions conflicted with it.

Regenerate from the working tree:

```bash
python src/make_outputs.py --all
```
