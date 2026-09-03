# Manuscript compilation brief

Everything needed to write the revision's baseline-comparison content: each claim,
the number that supports it, the file the number lives in, and the argument for
why it holds. Written to be self-contained — an agent working from this folder
should not need the original working tree.

**Paper:** MDPI *Drones*, drones-4493946, Major Revision.
**Task:** Reviewers 1 and 3 require baseline comparisons against the existing SAC
policy under identical conditions.

---

## 1. What was done, in one paragraph

Three controllers were evaluated on one frozen set of 500 randomised episodes
(0–4 obstacles, straight reference paths) through a single harness, with the same
observation, action space, reward, termination rule and episode cap: the deployed
SAC policy; PPO trained from scratch on three seeds under matched conditions; and
a LOS + PID + APF classical stack whose 20 parameters were selected by three
independent 250-configuration random searches on a **disjoint** 100-episode
tuning set. SAC was additionally retrained on three seeds to give it the same
multi-seed treatment as PPO. All statistical comparisons are paired on episode
identity, using exact McNemar for success and Wilcoxon signed-rank for continuous
metrics.

**The environment, reward function and observation space were never modified.**

---

## 2. Headline results

| Method | Success | Obstacle coll. | Border coll. | RMS CTE (m) | Min obst. clear. (m) | Rudder rate (deg/s) | Time (s) |
|---|---|---|---|---|---|---|---|
| SAC (deployed policy) | **0.950** | 0.038 | 0.012 | **0.908** | 0.233 | 74.9 | 20.9 |
| SAC (retrained, 3 seeds) | 0.924 | 0.047 | 0.029 | 1.101 | 0.440 | 77.3 | **20.3** |
| PPO (3 seeds) | 0.905 | 0.079 | 0.017 | 1.286 | 0.528 | 11.3 | 22.0 |
| LOS+APF (3 searches) | 0.960 | 0.033 | **0.007** | 1.328 | **0.768** | 9.0 | 27.2 |

Per-seed success — SAC: 0.940 / 0.900 / 0.932. PPO: 0.894 / 0.898 / 0.922.
LOS+APF: 0.968 / 0.972 / 0.940.

**Source:** `tables/comparison_table.md` (with 95 % CIs), `tables/comparison_table.csv`.

### By obstacle count

| Obstacles | SAC deployed | SAC seeds | PPO | LOS+APF |
|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 0.973 | 1.000 |
| 1 | 0.960 | 0.933 | 0.887 | 1.000 |
| 2 | 0.950 | 0.923 | 0.913 | 0.953 |
| 3 | 0.950 | 0.903 | 0.887 | 0.913 |
| 4 | 0.890 | 0.860 | 0.863 | 0.933 |

**Source:** `figures/success_by_obstacles.csv`, plotted in
`figures/success_by_obstacles.png`.

---

## 3. Claims, evidence, and the argument for each

### Claim 1 — SAC reaches 95.0 % over 500 randomised episodes

**Evidence:** `per_episode/sac_1M/summary.json` → success 0.950, obstacle
collision 0.038, border collision 0.012, timeout 0.000.

**Argument:** This is a re-run of the deployed checkpoint through the new harness
on the frozen 500, not a figure quoted from an older file. It reproduces the
manuscript's reported 94 % / 3 % / 3 % closely.

⚠ **The reported value should be updated from 0.940 to 0.950.** Mean episode
length also differs (203.9 against the 214.3 previously recorded, −4.9 %). This
drift predates the present work — `src/README.md` in the code tree records that
the stored results no longer reproduce from the original `rl_env.py` either. The
re-run is the only SAC figure produced under the same conditions as the
baselines, so it is the one that belongs in a comparison table. State the delta
rather than absorbing it silently.

### Claim 2 — SAC significantly outperforms PPO under identical conditions

**Evidence:** `tables/paired_stats_table.md`, exact McNemar against the deployed
policy: p = 0.00152 (seed 0), 0.00294 (seed 1), 0.0925 (seed 2). Success 0.950
against 0.905 pooled.

**Argument:** This is the DRL baseline Reviewer 1 requested. PPO received:
identical environment, observation, action space, reward, termination rule and
episode cap; the same 1M total environment interactions; the same 8 parallel
workers; a network matched to SAC's (`[256, 256]`, ReLU) and matching gamma
(0.99); and SB3's standard defaults for every PPO-specific hyperparameter. Every
setting is recorded in `run_config/ppo_seed*/hyperparameters.json`.

⚠ **Do not write "decisively".** The matched seed-for-seed comparison — retrained
SAC against PPO on the same seed — is significant in only 1 of 3
(p = 0.0152, then 1.0 and 0.590). The defensible claim is that the *deployed
policy* significantly outperforms PPO; the two algorithms trained from scratch
are closer than that.

### Claim 3 — SAC achieves parity with a strongly-tuned classical baseline while tracking the path significantly better

**Evidence:** McNemar on success: p = 0.211, 0.108, 0.576 across the three
independently tuned LOS+APF controllers — **not significant in any**. Wilcoxon on
RMS cross-track error: p = 1.2e−35, 1.4e−55, 1.6e−65, all favouring SAC
(0.908 m against 1.328 m). Completion time 20.9 s against 27.2 s.

**Argument:** This is the conventional method Reviewer 1 asked for (LOS-PID) and
the one Reviewer 3 named (PID for path following, APF for avoidance). For a
survey ASV, remaining on the survey line is the mission objective, so a 0.42 m
tracking improvement at equal success is the substantive difference between the
two controllers, alongside a 23 % shorter completion time.

⚠ **Do not claim SAC beats the classical baseline on success.** It does not, in
any of the three comparisons. The first reviewer to run a paired test will find
p = 0.211. Frame it as parity with a decisive tracking advantage.

### Claim 4 — The classical baseline required substantial tuning to reach that parity

**Evidence:** `tuning/apf_tuning_results{,_s2,_s3}.csv` — three independent random
searches, 250 configurations each, 100 tuning layouts each, **750 configurations
and 75 000 episodes total**. Selected configurations in
`tuning/los_apf_best{,_s2,_s3}.json`. The hand-chosen default achieves 0.750 on
the tuning set; the three searches reached 0.940 / 0.980 / 0.930, an improvement
of 18–23 points.

**Argument:** Under-tuning is the standard objection to a classical baseline in a
DRL paper, and this pre-empts it with a documented record rather than an
assertion. Three independent searches also show the classical controller's
performance depends on the tuning seed (0.940–0.972 on the evaluation set), where
the learned policy needs no hand-designed gains and is the controller actually
deployed on hardware.

### Claim 5 — SAC's actuator demand exceeds the servo limit, and this reproduces across runs

**Evidence:** mean commanded rudder rate against a 20 deg/s servo limit
(`MAX_RUD_RATE_DPS`): SAC deployed 74.9, SAC alternate checkpoint 66.9, SAC
retrains 62.9–80.3 across six independent from-scratch runs; PPO 10.2–16.0;
LOS+APF 7.4–9.2. Wilcoxon against LOS+APF p ≈ 1e−83.

**Argument:** SAC demands roughly **3.5× more rudder rate than the actuator can
deliver**, while both baselines sit comfortably inside the limit. The two effort
measures diverge informatively: SAC's control effort (integral of squared rudder
command) is only ~1.7× LOS+APF's, but its rudder rate is ~8× higher — so SAC is
not holding larger rudder angles, it is chattering between them. This is a
concrete, measured mechanism for the larger oscillations already reported in the
field trials, and it reproduces across independent training runs rather than
being a quirk of one checkpoint.

**Best presented as a limitation with a physical explanation** — it strengthens
the sim-to-field discussion rather than weakening it.

### Claim 6 — The methods occupy different points on one safety/accuracy trade-off

**Evidence:** SAC tracks 0.42 m tighter than LOS+APF but passes obstacles at
0.233 m where LOS+APF keeps 0.768 m (a factor of 3.3) and PPO 0.528 m. LOS+APF
pays for its clearance in time (27.2 s against 20.9 s).

**Argument:** Neither method dominates. Against the classical baseline SAC buys
tracking accuracy and speed at the cost of clearance; the classical controller
makes the opposite trade. This is a more honest and more interesting framing than
a winner-takes-all comparison, and it is visible in the qualitative figure — SAC
bypasses to port across all three plotted layouts while PPO and LOS+APF bypass to
starboard, the direction the environment's own bypass cue suggests.

---

## 4. Two corrections the revision must carry

### 4.1 Table 4's minimum border clearance is wrong

**Current text** reports exactly **2.00 m** for all three simulation scenarios
(and 2.00 / 1.00 / 2.00 for the field trials).

**The problem:** that value corresponds to no computed quantity. The environment
does compute a genuine per-step geometric minimum (`info["true_border_clearance"]`,
used by the `r_border` reward term), and over the frozen 500 it averages
**0.914 m** and reaches **−0.010 m** at the low end (hull outside the basin).

**Worse, the metric is uninformative as defined.** It minimises over all four
walls, including the two the vessel drives between, so it is floored by the start
pose: `START_Y` = 2.0 minus an inflated hull half-length of 1.0125 gives 0.9875 m
in *every* episode before the controller acts. Three completely different
controller families all report **0.989 m**, and SAC vs LOS+APF tests at
**p = 0.936**. That is what a constant looks like, not a measurement.

**The fix:** report `min_lateral_border_clearance` — side walls only — which does
discriminate: 2.514 (SAC deployed), 2.451 (SAC seeds), 2.484 (PPO), 2.524
(LOS+APF), against 0.722 for a degraded policy. Both columns are in every
`per_episode/*/episodes.csv`.

### 4.2 Scope the generalisation sentence

**Current text:** the policy *"can generalize across different obstacle
densities"*.

**The problem:** supported for the densities evaluated (0–4 obstacles, straight
paths), and not beyond them. An exploratory out-of-distribution study — not
included in the reported results — found performance falls off outside that
range.

**The fix:** scope the claim explicitly to 0–4 obstacles and straight reference
paths. This costs a few words and removes the one sentence a reviewer could
falsify in minutes with the released code.

---

## 5. Methods-section disclosures required

Three items must appear in the methods, or the results are not reproducible from
the description:

**5.1 Checkpoint selection.** Both learned methods are reported from the
checkpoint chosen by the training callback's selection score on its **validation
grid** (60 episodes), then evaluated on the **disjoint frozen 500**. Validation
for selection, test for reporting — the same rule for every method. This is *not*
"the model after 1M steps": the SAC runs peak at 250k / 400k / 750k steps and
degrade thereafter.

**5.2 SAC's update ratio.** Suggested wording:

> SAC is trained with four gradient updates per rollout collection across eight
> parallel environments (one update per two environment transitions), rather than
> the one-per-eight ratio of the original configuration. The environment, reward
> function and observation space are unchanged.

Rationale: the original setting performed ~125 000 gradient updates per 1M
environment steps, where standard SAC practice is one update per transition. At
the original ratio, from-scratch retrains reached only 0.780 / 0.814 / 0.842
(`supporting/sac_gradsteps1_seed*/`); at the corrected ratio they reach
0.940 / 0.900 / 0.932.

**5.3 Evaluation protocol.** 500 frozen episodes, 100 at each obstacle count 0–4,
every layout pre-validated for reachability by an inflated-grid A* filter. All
methods replayed from identical serialised layouts. Determinism verified: the
same policy run twice over 10 layouts produces bit-identical per-step
cross-track-error and rudder traces to 1e−12.

---

## 6. Limitations to state

**6.1 Reproducibility of the deployed policy.** Independent retraining at the
corrected update ratio gives 0.900–0.940, with two of three seeds statistically
indistinguishable from the deployed policy's 0.950 (p = 0.511, 0.289) and one
significantly below it (p = 0.00126). The result is reproducible; the exact value
carries seed variance of roughly ±0.02.

**6.2 Late-training degradation.** Every run at the corrected update ratio
degrades before 1M steps — validation success falls to 0.617–0.767 — and the peak
occurs at an unpredictable point (250k, 400k, 750k across three seeds). A fixed
training budget would not reliably land on the peak. This is the known
instability of SAC at a high update-to-data ratio.

**6.3 Asymmetric variance in the reported intervals.** The deployed SAC row is a
single checkpoint, so its interval reflects episode variance only. The SAC and
PPO rows pool three training seeds (seed *and* episode variance). The LOS+APF row
pools three independent tuning searches — the controller itself is deterministic,
so its interval reflects tuning-procedure variance, which is the analogue of a
training seed for a non-learned method. These are not the same quantity and the
caption should say so.

**6.4 The deployed policy's training stability is unexplained.** Its validation
curve rose to 0.967 and held through 1M steps; none of six retrains reproduce
that stability. Candidate causes are the documented environment drift and
run-to-run variance.

---

## 7. Which reviewer comment each item addresses

| Reviewer point | Addressed by |
|---|---|
| R1: one DRL baseline under identical conditions | PPO, 3 seeds, §3 Claim 2 — matched env/observation/reward/budget/network |
| R1: a conventional method (LOS-PID or NMPC) | LOS+APF, 3 tuned configurations, §3 Claims 3–4 |
| R3: PID/MPC for path following, APF/dynamic window for avoidance | The LOS + PID + APF stack satisfies both |
| R3: PPO is more stable than SAC on sparse-reward long-horizon tasks | Not supported here — SAC outperforms PPO (§3 Claim 2). State it as a measured result, but note the matched from-scratch comparison is closer |
| R: actuator behaviour | §3 Claim 5 — SAC demands 3.5× the servo rate limit, reproduced across six runs |
| R: figure distinguishable by more than colour | All figures use distinct line styles **and** markers |

---

## 8. Claims that must NOT be made

* ✗ "SAC outperforms the classical baseline." Not significant in any of the three
  comparisons (p = 0.211, 0.108, 0.576).
* ✗ "SAC maintains larger clearances." LOS+APF keeps 0.768 m and PPO 0.528 m
  against SAC's 0.233 m.
* ✗ "SAC decisively beats PPO." Pooled it is ahead, but the matched seed-for-seed
  test is significant in only 1 of 3 seeds.
* ✗ "SAC trained for 1M steps achieves 0.92." The reported policies are
  mid-training checkpoints; the 1M endpoints are substantially worse.
* ✗ Any unqualified generalisation claim beyond 0–4 obstacles and straight paths.
* ✗ Presenting the deployed policy as one of the training seeds. It is a separate
  object and is plotted as a distinct reference line for that reason.

---

## 9. Compute and figures

| Item | Cost |
|---|---|
| PPO, 3 seeds × 1M steps | 3.77 h |
| SAC, 3 seeds × 1M steps (corrected update ratio) | 21.53 h |
| LOS+APF tuning, 3 × 250 configurations | ~7 h |
| All evaluations | ~1.5 h |

**Figures**, all in `figures/`, all with distinct line styles and markers:

* `success_by_obstacles.{png,svg}` — success rate by obstacle count, all methods,
  per-run markers, min–max bands, deployed policy as a separate reference line.
  **Recommended as the primary comparison figure.** Note in the caption that the
  y-axis is zoomed to 0.70–1.01 and that the bands are min–max across runs, not
  confidence intervals. Underlying numbers in `success_by_obstacles.csv`.
* `learning_curves.{png,svg}` — evaluation return vs timesteps for SAC and PPO,
  seed bands. Training-dynamics figure only; the classical baseline has no
  training curve. Underlying numbers in `learning_curves.csv`, which is the only
  surviving extractable copy of the 0–1M deployed-SAC series.
* `trajectories.{png,svg}` — cases 220 / 320 / 420 under all three methods.

---

## 10. Data layout

```
tables/        comparison_table.{csv,md}, paired_stats_table.{csv,md}, paired_stats.json
figures/       success_by_obstacles, learning_curves, trajectories (+ source CSVs)
per_episode/   one directory per reported run: episodes.csv + summary.json
tuning/        750 configurations and their scores; selected parameter sets
layouts/       eval_layouts_v1.json (500 eval), tune_layouts_v1.json (100 tuning, disjoint)
run_config/    hyperparameters.json, curriculum.json, eval_summary.json per run
supporting/    sac_gradsteps1_seed*/ (before/after evidence), sac_1M_alt/ (rejected checkpoint)
```

Per-episode CSVs all join on `episode_id`, so any additional paired test can be
run directly from this folder.

**Two metric definitions to state in the paper**, because they are not the
obvious ones:

* `min_obstacle_clearance` is exact polygon-to-polygon distance from the inflated
  hull footprint to the nearest obstacle surface, zero on contact — **not** a
  LiDAR beam range. The sensor sits 0.8625 m forward of the vessel origin, so beam
  range overstates clearance (0.523 m against a true 0.091 m on a spot check).
* `min_border_clearance` minimises over all four walls; use
  `min_lateral_border_clearance` for corridor-keeping claims (see §4.1).

Full detail, including two retracted analyses and the reasoning behind every
methodological choice, is in `BASELINES_RESULTS.md` and `BASELINES_NOTES.md`
alongside this file.
