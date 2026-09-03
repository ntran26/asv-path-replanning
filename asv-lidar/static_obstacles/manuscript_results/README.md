# `manuscript_results/` — files backing the revision

Self-contained copy of everything the revised manuscript's baseline comparison
draws on, gathered so each number in the paper can be traced to a file.

**Scope.** The frozen 500-episode evaluation set: 0–4 obstacles, straight
reference paths. Every method measured by the same harness, on the same layouts,
with the same observation, action space, reward, termination rule and episode
cap. **The reward function and `env.py` were never modified.**

**Rebuilt 2026-08-23**, after SAC was retrained with a corrected update ratio
(`gradient_steps=4`). See `BASELINES_RESULTS.md` §4a.

---

## Start here

**`MANUSCRIPT_BRIEF.md`** is the compilation brief: every claim with its
supporting number, the file it comes from, and the argument for why it holds —
plus the corrections the revision must carry, the methods disclosures required,
the limitations to state, which reviewer comment each item answers, and an
explicit list of claims that must not be made. Read that first.

---

## The four reported rows

| Table row | Built from | Runs |
|---|---|---|
| SAC (deployed policy) | `per_episode/sac_1M/` | 1 deployed checkpoint |
| SAC (retrained, 3 seeds) | `per_episode/sac_gs4_seed{0,1,2}_best/` | 3 training seeds |
| PPO (3 seeds) | `per_episode/ppo_fx_seed{0,1,2}_best/` | 3 training seeds |
| LOS+APF (3 searches) | `per_episode/los_apf_s{1,2,3}/` | 3 tuning searches |

| Method | Success | Obst. coll. | Border coll. | RMS CTE (m) | Min clear. (m) | Rudder rate |
|---|---|---|---|---|---|---|
| SAC (deployed) | 0.950 | 0.038 | 0.012 | 0.908 | 0.233 | 74.9 |
| SAC (retrained) | 0.924 | 0.047 | 0.029 | 1.101 | 0.440 | 77.3 |
| PPO | 0.905 | 0.079 | 0.017 | 1.286 | 0.528 | 11.3 |
| LOS+APF | 0.960 | 0.033 | 0.007 | 1.328 | 0.768 | 9.0 |

Per-seed success — SAC: 0.940 / 0.900 / 0.932. PPO: 0.894 / 0.898 / 0.922.
LOS+APF: 0.968 / 0.972 / 0.940.

---

## Two things the paper must disclose

**1. Checkpoint selection.** Both learned methods are reported from the
checkpoint chosen by the training callback's selection score on its **validation
grid** (60 episodes), evaluated on the **disjoint frozen 500**. Validation for
selection, test for reporting — the same rule for every method, no leakage. This
is *not* "the model after 1M steps": the SAC runs peak at 250k / 400k / 750k and
degrade by 1M.

**2. SAC's update ratio.** `gradient_steps=4` across 8 workers — one gradient
update per two environment transitions, rather than the one-per-eight of the
original configuration. Standard SAC practice is one per transition. The
environment, reward and observation are unchanged, so the field trials remain
valid. Before/after evidence at `supporting/sac_gradsteps1_seed*/`
(0.780 / 0.814 / 0.842 at the original ratio).

---

## Where each manuscript number comes from

### Tables

`tables/comparison_table.md` — formatted, with stratified bootstrap 95 % CIs.
`tables/comparison_table.csv` — machine-readable, `__mean` / `__iqm` /
`__ci_lo` / `__ci_hi` / `__per_run` per metric.

Rate metrics (success, collisions, timeout, rudder saturation) are **means** —
that is what a rate is. Continuous metrics are **IQM**, which is degenerate on a
binary variable and so is not used for rates.

`tables/paired_stats_table.md` / `.csv`, detail in `tables/paired_stats.json`.
Two families: everything against the deployed SAC checkpoint, and retrained SAC
against PPO **seed for seed** — the matched comparison that isolates the
algorithm. McNemar exact, Wilcoxon signed-rank, all paired on `episode_id`.

### Figures

| Figure | File | Data |
|---|---|---|
| **Success by obstacle count** (primary) | `figures/success_by_obstacles.{png,svg}` | `figures/success_by_obstacles.csv`. Rebuild with `python plot_success_by_obstacles.py` (copy included here). Caption must note the y-axis is zoomed to 0.70-1.01 and the bands are min-max across runs, not CIs |
| Learning curves | `figures/learning_curves.{png,svg}` | `figures/learning_curves.csv`. Deployed-SAC curve recovered from `sac_log/asv_sac_2/events.out.tfevents.1781347673.*`; retrain curves from `run_config/*/eval_summary.json` |
| Qualitative trajectories | `figures/trajectories.{png,svg}` | Re-run live from `layouts/eval_layouts_v1.json`, cases 220 / 320 / 420 |

Both use distinct line styles **and** markers rather than colour alone,
addressing the existing reviewer comment. No curriculum boundaries are drawn —
none of the reported runs used one.

`figures/learning_curves.csv` matters: the 0–1M deployed-SAC series exists
nowhere else in extractable form, since the root `eval_summary.json` was
overwritten by a later run. Restyle from this CSV, not TensorBoard.

---

## Supporting records a reviewer may ask for

| Question | File |
|---|---|
| Training hyperparameters? | `run_config/{sac,ppo}_seed{0,1,2}/hyperparameters.json` |
| What changed from the published SAC config? | same files, `deviation_from_published_config` |
| Was a curriculum active? | `run_config/*/curriculum.json` — one entry, stage 0, fixed RPM |
| Evidence for the under-training diagnosis? | `supporting/sac_gradsteps1_seed*/` |
| Classical baseline tuning budget? | `tuning/apf_tuning_results{,_s2,_s3}.csv` — 750 configurations |
| Which LOS+APF parameters? | `tuning/los_apf_best{,_s2,_s3}.json` |
| What was evaluated on? | `layouts/eval_layouts_v1.json` |
| Was tuning done on the evaluation set? | No — `layouts/tune_layouts_v1.json`, disjoint |
| Which SAC checkpoint is canonical? | `supporting/sac_1M_alt/` — the rejected alternative |

---

## Per-episode schema

Each `per_episode/*/episodes.csv` has one row per evaluation episode, joined
across methods on `episode_id`. Two definitions worth stating in the paper:

* **`min_obstacle_clearance`** — exact polygon-to-polygon distance from the
  inflated hull footprint to the nearest obstacle surface, zero on contact. Not a
  LiDAR beam range: the sensor sits 0.8625 m forward of the vessel origin, so beam
  range overstates clearance (0.523 m against a true 0.091 m on a spot check).
* **`min_border_clearance`** — minimises over all four walls and is floored by
  the start pose at 0.9875 m in every episode, for every method. Use
  **`min_lateral_border_clearance`** for corridor-keeping claims. This is the
  Table 4 correction.

---

## Deliberately not included

* **Staged-curriculum runs** — withdrawn; they used a curriculum the deployed SAC
  run never had. Archived at `models/_archive_staged_curriculum/` and
  `eval_results/_archive_staged_curriculum/`.
* **The out-of-distribution study** — exploratory, not reported. Protocol and
  data at `OOD_PROTOCOL.md` and `eval_results/baselines/ood_*/`.

---

## Provenance

`BASELINES_RESULTS.md` — §0 is the manuscript-ready summary (tables, figures, and
an explicit list of claims the data supports versus claims it does not); §4
retracts an earlier PPO-collapse finding; §4a documents the SAC reproducibility
result and the disclosure wording.

`BASELINES_NOTES.md` — what the code actually does, and where the task
assumptions conflicted with it.

```bash
python src/make_outputs.py --all
```
