# `manuscript_results/` — files backing the revision

Self-contained copy of everything the revised manuscript's baseline comparison
draws on, gathered so each number in the paper can be traced to a file.

**Scope.** The frozen 500-episode evaluation set: 0–4 obstacles, straight
reference paths. Every method measured by the same harness, on the same layouts,
with the same observation, action space, reward, termination rule and episode
cap. The reward function and `env.py` were never modified.

**Rebuilt 2026-08-21.** All learned-method results are from runs under the
**corrected fixed-RPM setup** — the configuration the deployed SAC policy was
actually trained under. See `BASELINES_RESULTS.md` §4 for the retraction of an
earlier curriculum error, and §4a for the SAC retrain result.

---

## The four reported rows

| Table row | Built from | Runs |
|---|---|---|
| SAC (deployed policy) | `per_episode/sac_1M/` | 1 deployed checkpoint |
| SAC (retrained, 3 seeds) | `per_episode/sac_fx_seed{0,1,2}_final/` | 3 training seeds |
| PPO (3 seeds) | `per_episode/ppo_fx_seed{0,1,2}_final/` | 3 training seeds |
| LOS+APF (3 searches) | `per_episode/los_apf_s{1,2,3}/` | 3 tuning searches |

`_fx_` = fixed-RPM, the corrected setup. Learned methods are reported from their
**final 1M-step checkpoints**, with no best-checkpoint selection.

**The deployed and retrained SAC rows are different objects and must not be
pooled.** The deployed checkpoint is the controller the field trials used; the
retrains are independent from-scratch runs at the same budget. They differ by
~14 points of success — that gap is a finding, documented in §4a, not noise to
average over.

---

## Headline numbers

| Method | Success | Obst. coll. | RMS CTE (m) | Min clear. (m) | Rudder rate |
|---|---|---|---|---|---|
| SAC (deployed) | 0.950 | 0.038 | 0.908 | 0.233 | 74.9 |
| SAC (retrained) | 0.812 | 0.157 | 0.906 | 0.230 | 67.0 |
| PPO | 0.871 | 0.100 | 1.212 | 0.503 | 12.8 |
| LOS+APF | 0.960 | 0.033 | 1.328 | 0.768 | 9.0 |

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
Two families of comparison: everything against the deployed SAC checkpoint, and
retrained SAC against PPO **seed for seed** — the matched from-scratch
comparison that isolates the algorithm. McNemar exact, Wilcoxon signed-rank, all
paired on `episode_id`.

### Figures

| Figure | File | Data behind it |
|---|---|---|
| Learning curves | `figures/learning_curves.{png,svg}` | `figures/learning_curves.csv`. Deployed-SAC curve recovered from `sac_log/asv_sac_2/events.out.tfevents.1781347673.*`; retrain curves from `run_config/*/eval_summary.json` |
| Qualitative trajectories | `figures/trajectories.{png,svg}` | Re-run live from `layouts/eval_layouts_v1.json`, cases 220 / 320 / 420 |

Both use distinct line styles **and** markers rather than colour alone,
addressing the existing reviewer comment. No curriculum stage boundaries are
drawn — none of the reported runs used a curriculum.

`figures/learning_curves.csv` matters: the 0–1M deployed-SAC evaluation series
exists nowhere else in extractable form, because the root `eval_summary.json`
was overwritten by a later fine-tuning run. Restyle from this CSV, not from
TensorBoard.

---

## Supporting records a reviewer may ask for

| Question | File |
|---|---|
| Training hyperparameters? | `run_config/{sac,ppo}_seed{0,1,2}/hyperparameters.json` |
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
  deployed SAC run never had. Archived at `models/_archive_staged_curriculum/`
  and `eval_results/_archive_staged_curriculum/`.
* **The out-of-distribution study.** Exploratory, not reported in the manuscript.
  Protocol and data at `OOD_PROTOCOL.md` and `eval_results/baselines/ood_*/`.

---

## Provenance

`BASELINES_RESULTS.md` — §0 is the manuscript-ready summary: tables, figures, and
an explicit list of claims the data supports versus claims it does not. §4
retracts an earlier PPO-collapse finding. §4a covers the SAC reproducibility
result.

`BASELINES_NOTES.md` — what the code actually does, and where the task
assumptions conflicted with it.

Regenerate from the working tree:

```bash
python src/make_outputs.py --all
```
