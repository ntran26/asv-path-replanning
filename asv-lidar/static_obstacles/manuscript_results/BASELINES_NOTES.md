# BASELINES_NOTES.md — what the code actually does

Written before implementing anything, per §0 of the task brief. Everything below
was read out of `src/` (the refactored tree) and verified by running the code.
Where the brief's assumptions conflict with the code, **the code wins** and the
conflict is flagged with ⚠.

Reference: `src/README.md` documents the refactor and asserts bit-identical
behaviour against the original `rl_env.py`. I did not re-verify that claim in
full; I verified the parts this task depends on (determinism, replay, metrics).

---

## 1. Observation

⚠ **The observation is a `gymnasium.spaces.Dict`, not a flat vector.** The brief
says "observation vector" throughout. Any controller has to accept a dict of
named `np.float32` arrays. Total dimensionality 34.

Built in `env.py:_get_obs` (`src/env.py:283`). Space declared at `src/env.py:96`.

| Key | Shape | Box bounds | Meaning |
|---|---|---|---|
| `lidar` | (25,) | [0, 1] | pooled sector **closeness**, `1 - range/16`; 1 = touching |
| `u` | (1,) | [0, 5] | surge velocity, m/s |
| `v` | (1,) | [-3, 3] | sway velocity, m/s |
| `yaw_rate` | (1,) | [-180, 180] | deg/s |
| `cross_track_error` | (1,) | [-25, 25] | signed metres, **positive = left of path** |
| `course_error` | (1,) | [-180, 180] | deg, path tangent minus course |
| `lookahead_course_error` | (1,) | [-180, 180] | deg, bearing to lookahead point minus course |
| `front_clearance` | (1,) | [0, 16] | m |
| `side_clearance_diff` | (1,) | [-16, 16] | m, **right minus left** |
| `local_target_cte` | (1,) | [-25, 25] | m, lateral bypass cue |

⚠ **There is no normalisation.** Every component is in raw physical units. The
`Box` bounds above are declarations only — nothing clips or rescales the values
into them. SB3's `MultiInputPolicy` concatenates the dict and feeds it to the
MLP directly. (`MAX_EPISODE_STEPS`-length episodes routinely carry
`yaw_rate` ≈ 30 and `front_clearance` = 16, so the scales across components
differ by ~2 orders of magnitude. This is a property of the published setup,
not something to change.)

### 1.1 Which LiDAR feeds which field — this matters for the APF baseline

Three separate LiDARs are simulated every step (`env.py:_scan_lidars`,
`src/env.py:271`). All are 225 raw beams over a 270° swath, 16 m range, pooled
to 25 sectors by Meyer-style feasibility pooling (`LIDAR_POOLING_MODE="paper"`).

| LiDAR | Sees | Consumed by |
|---|---|---|
| `lidar_obs` | obstacles + walls per `OBS_BORDER_MODE` | the `lidar` observation field |
| `lidar_reward` | **obstacles only** | `r_oa` reward, `front_clearance`, `info` diagnostics |
| `lidar_border_guard` | **walls only** | `side_clearance_diff`, `local_target_cte` |

`OBS_BORDER_MODE = "none"` in the published config (`src/config.py:96`), so the
`lidar` closeness vector the policy sees contains **obstacles only** — no wall
returns.

⚠ **Correction to brief §3.2 "Boundary handling".** The brief states that both
methods face identical "boundary-blindness" because the environment excludes
boundary returns from the LiDAR observation. That is true of the `lidar` vector
but **false of the observation as a whole**. `side_clearance_diff` is computed
as `right_clearance - left_clearance` where each side clearance is
`min(obstacle_range, border_guard_range)` (`src/env.py:333`) — it carries wall
proximity. `local_target_cte` inherits the same information through its sign
(`src/env.py:355`). So the SAC policy **does** receive partial boundary
information, through two scalars rather than through the LiDAR.

Consequence for fairness: the LOS+APF baseline is entitled to use
`front_clearance`, `side_clearance_diff` and `local_target_cte`, because they
are part of the observation the SAC policy receives. Denying them to the
classical controller would handicap it, not make the comparison fair. The
constraint that *does* bind — and which I will hold to — is **no access to
`env.obstacles`, `env.map_border`, or any raw geometry**. The controller sees
the 34-dim observation and nothing else.

### 1.2 Sector bearings are not where they are labelled

`sector_angle_grid()` returns `linspace(-135, +135, 25)`, i.e. nominal centres
11.25° apart. But pooling uses `np.array_split(raw_ranges, 25)` over 225 beams
spaced 270/224 = 1.2054° apart, so sector *i* actually spans beams `9i…9i+8`,
whose angular centre is `-135 + 1.2054·(9i+4)`.

Measured discrepancy: **up to 4.82°**, largest at the swath edges, zero at
centre. The environment itself uses the *nominal* grid for its own masking in
`_update_local_planner_features`, so the mismatch is already baked into the
published features.

For the APF I will use the nominal grid by default (consistent with what the
environment does), and expose the true chunk-centre grid as a tunable
alternative in the random search, so the tuner can pick whichever works and the
choice is on the record rather than assumed.

---

## 2. Action space

`Box(-1, 1, shape=(2,), float32)` (`src/env.py:112`). Applied at
`src/env.py:419`.

- **`action[0]` — rudder.** `env.rudder = a0 · 100` (a percentage). The hull
  servo then commands `delta_cmd = -a0 · 40°` and tracks it **rate-limited at
  20 °/s** (`src/ship.py:_derivatives`). The sign inversion plus the force
  convention nets out to: **positive `a0` turns to starboard** (heading
  increases). Verified by inspection of `n_rud`.
- **`action[1]` — throttle.** `rpm = clip(12.0 + RPM_DELTA · a1, RPM_FLOOR, RPM_CEIL)`.

So saturation and rate limiting are **not** enforced on the command — they are
enforced inside the hull model. Both methods get them for free. A baseline that
emits `a0 = ±1` every step is not "violating" a limit; it is saturating the
servo, which is exactly what `rudder_saturation_fraction` should measure.

`info["rudder_deg"] = a0 · 40` — a restatement of the command, not the achieved
rudder angle. The achieved angle lives in `ShipModel.rudder_deg` and is not
exported to `info`.

---

## 3. Reward

`env._reward` (`src/env.py:466`). Ten dense terms plus terminal events. Term
names as they appear in `info`:

`r_pf`, `r_heading`, `r_oa`, `r_border`, `r_exist`, `r_progress`, `r_slow`,
`r_thrust`, plus `gamma_e_eff`. Two further terms — **`r_cte_recovery` and
`r_wrong_side` — are computed and applied to the reward but deliberately
filtered out of `info`** (`src/env.py:607`), matching the original log schema.
`info["r_local"]` and `info["r_center"]` are retired terms emitted as constant
`0.0` purely to keep CSV headers stable.

Composition:

```
dense = λ·(u_gate·r_pf) + (1-λ)·r_oa + 0.35·u_gate·r_heading
      + r_exist + r_border + r_progress + r_slow + r_thrust
      + r_cte_recovery + r_wrong_side
```

with `λ = DEFAULT_EVAL_LAMBDA = 0.5`, fixed. Terminal: collision **replaces**
the dense reward with `-1000`; goal **adds** `+50`; timeout **adds** `-1000`.

`gamma_e` is blended between 0.20 (clear) and 0.05 (blocked) by `block_alpha`,
so path-tracking strictness relaxes near obstacles.

---

## 4. Termination and success

`src/env.py:452`.

- `terminated = collided or reached_goal`
- `truncated  = step_count >= 700 and not terminated` (70 s at 10 Hz)
- `collided` is **true for border contact as well as obstacle contact**
  (`_collided` tests `_hits_border` first). To separate the two, call
  `env.hit_border()` at termination — this is what `rollout.termination_reason`
  does, and it is the existing definition I will reuse.
- **Success = `reached_goal`**: `distance_to_goal <= 0.5` **or**
  (remaining arc length `<= 1.25` **and** `|cte| <= 1.60`).

Note the episode cap interacts with the evaluation scripts: `MAX_EPISODE_STEPS`
is 700, but `evaluate_suite.py` passes `max_steps=2000`. The env truncates at
700 first, so 700 is the binding limit. `path_completion_time` will be reported
in **both steps and seconds** (`steps × 0.1 s`), stated explicitly per the
brief's request.

---

## 5. Randomisation and RNG

⚠ **The environment draws from the global `np.random` stream, not
`self.np_random`.** `reset(seed=s)` calls `super().reset(seed=s)` *and*
`np.random.seed(s)` (`src/env.py:164`). Every layout generator in
`obstacles.py` and `_random_start_goal` uses bare `np.random.*`.

Practical consequences:
- Seeding is process-global. Under `SubprocVecEnv` each worker is a separate
  process, so this is safe for training, but two envs in one process share a
  stream.
- Replay of a saved layout is nonetheless exact, because
  `reset(options={"scenario": ...})` **bypasses all sampling** — it reads
  `start`, `goal`, `obstacles` and `path` straight from the record
  (`_load_scenario`, `src/env.py:207`). The only RNG call on that path is
  `_sample_obs_border_mode`, which returns immediately without drawing when
  `OBS_BORDER_MODE != "mixed"` — and it is `"none"`. So the replay path consumes
  **zero** random numbers.

Layout families (`obstacles.py`): `normal` 0.40, `target_side` 0.35,
`field_repair` 0.15, `gate` 0.05, `offpath` 0.05; obstacle counts 0–4 with
probabilities 0.15/0.15/0.45/0.15/0.10. Start `y = 2.0`, goal `y = 22.0`, 70 %
of paths vertical, remainder slanted. Obstacles are 1.0 m axis-aligned squares.

---

## 6. The staged propulsion curriculum — the biggest conflict

⚠⚠ **There is no curriculum scheduler in the code. None. It does not exist.**

The brief (§2) requires that "the staged propulsion curriculum must be active
and scheduled against the same counter" and warns that a silent mismatch would
invalidate the comparison. The reality:

```python
# src/config.py:41
RPM_STAGE = 1
RPM_STAGES = {1: (3.0, 9.0, 15.0), 2: (4.0, 8.0, 16.0),
              3: (6.0, 6.0, 18.0), 4: (12.0, 0.0, 24.0)}
RPM_DELTA, RPM_FLOOR, RPM_CEIL = RPM_STAGES[RPM_STAGE]
```

This is evaluated **once at import time**. `grep` across the whole repository
finds no assignment to `RPM_STAGE`, `RPM_DELTA`, `RPM_FLOOR` or `RPM_CEIL`
outside these constant blocks, and no callback that touches them. There is no
counter, no schedule, no `set_stage`.

**How the curriculum was actually run:** by hand-editing `config.py` (or
`rl_env.py`) and restarting training with `--resume`. The evidence:

1. `plotting/plot_training_curves.py:96` documents the phases —
   cruise (fixed speed) 0–700k, stage 1 700–800k, stage 2 800–900k,
   stage 3 900k–1.0M.
2. `sac_log/asv_sac_2/` holds **six** separate tfevents files, i.e. six resumed
   training invocations.
3. `train.py` has `--resume` and `--replay-buffer-path` flags built for exactly
   this workflow.

**What this means for PPO.** "Identical conditions" cannot be achieved by
turning on a scheduler that was never there. I will replicate the documented
schedule explicitly, on a **total-environment-step** counter, using a callback
that mutates the `config` module globals inside each `SubprocVecEnv` worker via
`env_method`. Because `env.step` reads `cfg.RPM_DELTA` as a module attribute at
call time, mutating the module in the worker process takes effect immediately.
This requires **no change to `env.py`** — the curriculum env is a subclass in a
new file, used only by PPO training, and `train.py`'s SAC path never
constructs it.

The phase boundaries (700k/800k/900k) will be scheduled against
`model.num_timesteps`, which SB3 increments by `n_envs` per rollout step — i.e.
total environment interactions, not per-env steps. This is the counter the
brief asks for.

### 6.1 Which stage produced the manuscript numbers

Resolved, and it is not obvious from the file names:

| File | success | obstacle | border | timeout | mean rpm |
|---|---|---|---|---|---|
| `eval_suite_summary_1M.json` | **0.940** | **0.030** | **0.030** | 0.000 | 11.65 |
| `eval_suite_summary_stage_1.json` | 0.940 | 0.030 | 0.030 | 0.000 | 11.65 |
| `eval_suite_summary_stage_2.json` | 0.924 | 0.054 | 0.022 | 0.000 | 11.42 |
| `eval_suite_summary_stage_3.json` | 0.894 | 0.082 | 0.024 | 0.000 | 10.79 |
| `eval_suite_summary_stage_4.json` | 0.924 | 0.010 | 0.066 | 0.000 | 12.00 |
| `eval_suite_summary.json` ← **default output path** | 0.676 | 0.132 | 0.024 | 0.168 | 7.09 |

The manuscript's headline — 94 % success, 3 % obstacle, 3 % border over 500
episodes — is `eval_suite_summary_1M.json` / `stage_1.json`. The `stage_N` files
are the **same 1M checkpoint evaluated under each of the four RPM stage
settings**, not checkpoints from different training phases.

⚠ **`eval_suite_summary.json` and `eval_suite_details.csv` — the files
`evaluate_suite.py` writes by default, and the ones a reader would naturally
pick up — are stale and do not correspond to the manuscript.** They record 67.6 %
success with `mean_rpm` 7.09, `min_rpm` 0.0 and `max_rpm` 24.0. Those RPM values
are only reachable under `RPM_STAGE = 4` (floor 0, ceil 24); under the current
stage 1 the range is [9, 15]. So that file is a stage-4 run of some later model.
`src/README.md` already flags that these checked-in numbers no longer reproduce.
**Anything quoting `eval_suite_details.csv` as "the SAC baseline" is quoting the
wrong run.** I will regenerate SAC on the frozen set with the new harness and
check it lands at 94 %.

Current `config.py` is at `RPM_STAGE = 1`, which is the manuscript setting.
Evaluation of all three methods will be done at stage 1.

⚠ There are also **two different `sac_model_1M.zip` files**: `./sac_model_1M.zip`
(3,390,336 B) and `./models/sac_model_1M.zip` (3,390,304 B). `evaluate_suite.py`
defaults to the `models/` one.

**Resolved.** They are not the same policy repackaged — comparing
`policy.state_dict()`, **all 32 tensors differ**, with a maximum absolute weight
difference of 0.127. They are different training snapshots. Evaluating both over
the frozen 500 settles which one the manuscript used:

| | Manuscript | `models/sac_model_1M.zip` | `./sac_model_1M.zip` |
|---|---|---|---|
| Success | 0.940 | **0.950** | 0.894 |
| Obstacle collision | 0.030 | **0.038** | 0.102 |
| Border collision | 0.030 | **0.012** | 0.004 |
| Timeout | 0.000 | 0.000 | 0.000 |

`models/sac_model_1M.zip` is the canonical checkpoint. The root-level file has
2.7x the obstacle collision rate and is not a candidate. **All SAC results in
this study use `models/sac_model_1M.zip`.** The root-level duplicate should be
renamed or removed before release so nobody picks it up by accident; the
alternative run is kept at `eval_results/baselines/sac_1M_alt/` as the evidence.

### 6.2 SAC reproduces at 95.0 %, not exactly 94.0 %

| | Manuscript | Reproduced | Delta |
|---|---|---|---|
| Success | 0.940 | 0.950 | +0.010 |
| Obstacle collision | 0.030 | 0.038 | +0.008 |
| Border collision | 0.030 | 0.012 | -0.018 |
| Timeout | 0.000 | 0.000 | 0 |
| Mean episode length | 214.3 | 203.9 | **-10.4 steps (-4.9 %)** |

Per obstacle-count group: obs_0 identical at 1.000; obs_1 +0.040; obs_2 +0.030;
obs_3 identical success but the failure mode flips completely (5 border
collisions become 5 obstacle collisions); obs_4 -0.020.

The headline claim survives — SAC is ~94-95 % over the 500 layouts. But the
**~5 % systematic shortening of episodes** is a real drift, and it matches the
note already in `src/README.md`: case 0 runs 226 steps today against the 240
recorded in the stored results. `src/README.md` states that the gap reproduces
from the original `rl_env.py` as well, so it predates the refactor and is **not
introduced by this work** — something in the environment or the checkpoint
changed after the stored results were generated, and there is no surviving
per-episode file for the 94 % run to diff against.

Consequence for the revision: the SAC row of any comparison table should be the
**re-run** figure, not the stored one, because that is the only number produced
under the same conditions as PPO and LOS+APF. The delta from the published table
is small and should be stated rather than quietly absorbed.

---

## 7. Existing SAC training and evaluation setup

`src/train.py`.

**SAC hyperparameters** (`build_model`, `src/train.py:334`):
`MultiInputPolicy`, `learning_rate=5e-5`, `batch_size=512`, `gamma=0.99`,
`buffer_size=1_000_000`, `train_freq=1`, `gradient_steps=1`, `ent_coef="auto"`.
`policy_kwargs` is **not set**, so SB3's SAC default `net_arch=[256, 256]` with
ReLU applies.

**Vectorised setup:** `VecMonitor(SubprocVecEnv([...]))`, default
`--num-envs 8`, `--timesteps 1000000`, `--seed 675973` per the README command.
Checkpoints every `save_freq // num_envs` calls; eval callback every 50k steps.

**A PPO config already exists** in the same function: `learning_rate=1e-4`,
`n_steps=1024`, `batch_size=256`, `n_epochs=10`, `gamma=0.999`,
`gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.03`, `vf_coef=0.5`,
`net_arch=dict(pi=[64,64], vf=[64,64])`, `activation_fn=Tanh`.

⚠ That PPO config is **architecturally mismatched to SAC**: [64, 64] Tanh vs
SAC's [256, 256] ReLU, and `gamma` 0.999 vs 0.99. Per brief §2 ("architecture
matched to the SAC policy's hidden layer sizes where the algorithms permit") the
new PPO baseline will use **[256, 256] ReLU and gamma 0.99**, and the difference
from the pre-existing config will be documented. `models/ppo_model_1000000_steps.zip`
is 201 kB against SAC's 3.4 MB, consistent with the small [64,64] network — that
old checkpoint is not a matched baseline and will not be reused.

**The 500-episode evaluation** is driven by `src/evaluate_suite.py`, which loads
`eval_suite/asv_eval_suite.json` and calls
`run_episode(..., reset_kwargs={"seed": scenario["seed"],
"options": {"scenario": scenario}})` per case. It is configured by editing a
`USER SETTINGS` block, not argparse.

⚠ **`train.py` applies a `side_path_guard` action filter in `--mode test` only**
(`src/train.py:296`). It is a hand-written override that forces corrective
rudder. It is **not** applied in `--mode eval` or in `evaluate_suite.py`, so the
94 % figure is the raw policy. The new harness will not apply it either. Worth
knowing it exists: the rendered demo videos are not the same controller as the
evaluated one.

---

## 8. The frozen evaluation set already exists

Brief §1.1 asks me to build a 500-episode frozen set. **It is already built and
shipped**, and rebuilding it would make every existing result incomparable.

`eval_suite/asv_eval_suite.json` — 500 scenarios, 100 each for obstacle counts
0/1/2/3/4, unique `case_id` 0–499, unique seeds. Each record carries
`start`, `goal`, `obstacles` (full polygon vertices), `path` (100 points),
`seed`, `map_width`, `map_height`, `path_mode`, and `route_ratio_astar`. Every
layout passed an inflated-grid A* reachability filter with route ratio in
[1.00, 2.25], so there are no impossible cases.

That fully determines an episode, and `reset(seed=..., options={"scenario":...})`
is the deterministic constructor the brief asks for. **I will use it as-is** as
the frozen 500 set rather than generating a new one.

⚠ The suite metadata string says "600-case … 0..5"; the actual content is 500
cases over 0..4. `src/README.md` records that the wrong string is kept
deliberately so a regenerated file matches the shipped one byte-for-byte.

⚠ `src/generate_suite.py` writes to `data/env_setup/eval_suite/`, **not** the
`eval_suite/` that the evaluator reads. That is intentional guarding against
accidental regeneration. The 100-episode tuning set will be generated with the
same generator and disjoint seeds, written to a new path, and will never touch
`eval_suite/`.

### 8.1 Determinism check — PASSED

Ran `models/sac_model_1M.zip` twice over 10 frozen layouts spanning all five
obstacle-count groups, comparing not just summary metrics but the **full
per-step cross-track-error and rudder traces** rounded to 1e-12.

```
case   0 obs=0 steps=226 reason=goal     R=  100.75 identical=True
case  50 obs=0 steps=186 reason=goal     R=  103.01 identical=True
case 100 obs=1 steps=144 reason=obstacle R= -970.48 identical=True
case 150 obs=1 steps=194 reason=goal     R=   98.57 identical=True
case 200 obs=2 steps=203 reason=goal     R=   97.07 identical=True
case 250 obs=2 steps=203 reason=goal     R=   97.23 identical=True
case 300 obs=3 steps=204 reason=goal     R=   95.11 identical=True
case 350 obs=3 steps=215 reason=goal     R=   90.16 identical=True
case 400 obs=4 steps=236 reason=goal     R=   75.42 identical=True
case 450 obs=4 steps=211 reason=goal     R=   85.39 identical=True

IDENTICAL: 10/10
```

No nondeterminism to fix. `deterministic=True` on SB3 `predict`, a replay path
that draws no random numbers, and a fixed-step RK4 integrator with no stochastic
component together make the rollout exactly reproducible. Torch seeding is not
required because no sampling occurs.

---

## 9. `min_border_clearance` — the manuscript's 2.00 m is not a measured minimum

Brief §1.2 flags this as a live correction. Confirmed, and here is what is
actually going on.

**What the manuscript says.** Extracted from `manuscript.docx`, Table 4:

| Scenario | Platform | Min. obstacle clearance | Min. border clearance |
|---|---|---|---|
| 1 | Simulation | 0.68 m | **2.00 m** |
| 1 | Field | 0.54 m | **2.00 m** |
| 2 | Simulation | 0.52 m | **2.00 m** |
| 2 | Field | 1.17 m | **1.00 m** |
| 3 | Simulation | 1.09 m | **2.00 m** |
| 3 | Field | 0.50 m | **2.00 m** |
| Overall | Simulation | 0.76 m | **2.00 m** |

Values quantised to exactly 1.00 and 2.00 across six independent trajectories.
No measured per-step minimum behaves like that.

**What the environment computes.** `env._border_clearance` (`src/env.py:378`)
is a genuine per-step geometric minimum over the inflated hull polygon against
all four walls, exported as `info["true_border_clearance"]`, and it is what the
`r_border` reward term uses. `evaluate_suite.py` already records
`min_border_clearance` from it. Measured over the stored 500-episode SAC run:

```
min_border_clearance:  min = -0.010   mean = 0.914   max = 0.993
```

Not 2.00 m, and **negative at the low end** (hull outside the basin — a border
collision).

**Why it clusters at ~0.99 m, and why that is also misleading.** The minimum is
taken over all four walls including the two the vessel drives *between*. Start
is at `y = 2.0` and the inflated hull half-length is
`(1.725 + 2·0.15)/2 = 1.0125`, so the longitudinal clearance at spawn is
`2.0 - 1.0125 = 0.9875` in **every episode**, before the policy does anything.
The metric is therefore floored by the start pose and is nearly blind to
steering quality.

Splitting the two directions on five SAC episodes:

| case | n obs | min obstacle clr | min border clr (all walls) | min border clr (**lateral only**) |
|---|---|---|---|---|
| 0 | 0 | — | 0.988 | 2.356 |
| 120 | 1 | 0.365 | 0.989 | 2.259 |
| 220 | 2 | 0.091 | 0.989 | 2.125 |
| 320 | 3 | 0.152 | 0.681 | 0.681 |
| 420 | 4 | 0.206 | 0.989 | 1.922 |

The **lateral** minimum sits around 1.9–2.4 m and does respond to the policy
(case 320, which swung wide, drops to 0.68 m). That is almost certainly the
quantity Table 4 meant to report, rounded — but rounded so coarsely that it lost
all of its discriminating power and became a constant.

**What I will do.** The harness records both, under distinct names:
- `min_border_clearance` — all four walls, matching `info["true_border_clearance"]`
  and the `r_border` reward term. Comparable to the existing code.
- `min_lateral_border_clearance` — side walls only. The quantity that actually
  measures corridor-keeping, and the one I recommend for the revised Table 4.

Both reported, with this explanation, so the correction is defensible.

**`min_obstacle_clearance`** is not computed anywhere in the existing code.
`info["min_lidar_reward"]` is the nearest *beam range from the bow-mounted
sensor* (`LIDAR_OFFSET_M = 0.8625` forward of the vessel origin), which is
neither footprint-based nor a true surface distance — for case 220 it reads
0.523 where the true footprint-to-surface distance is 0.091. The harness
computes the real thing: exact polygon-to-polygon distance from the inflated
hull to each obstacle, zero on intersection.

Crucially, all of this is computed **in the harness** from `env.hull_polygon()`
and `env.obstacles`, both already public. **`env.py` needs no modification of
any kind** — the hard constraint in brief §0 is satisfied without even needing
an `info` addition.

---

## 10. Prior baseline artifacts in the tree — do not reuse

Both baselines appear to have been attempted before. The result files survive;
the source does not (`git ls-files` shows no `los_apf`/baseline source, and no
`baselines/` directory exists).

`eval_results/los_apf_baseline/` — 500 episodes, 49.4 % success, with a
`los_apf_config.json`. `eval_results/los_apf_tune_reward/` — the tuning record.
Reading those two files shows the prior attempt has exactly the weaknesses the
brief is written to pre-empt:

1. ⚠ **It was tuned on the evaluation set.** `run_args.json` shows
   `"suite_json": "eval_suite/asv_eval_suite.json"` with `"per_group_limit": 20`
   — i.e. a 100-episode subset **drawn from the frozen 500**. Tuning and
   reporting on the same layouts.
2. ⚠ **The search was 23 named presets**, not a random search.
   `tuning_summary.csv` has 24 lines. The brief asks for ≥200 configurations.
3. ⚠ **It never commanded speed.** `"throttle_action": 0.0` is fixed in both the
   tuning and the final config, so the controller ran at a constant 12 RPM while
   SAC modulated throttle. Brief §3.2 is explicit that this makes the comparison
   unfair.
4. ⚠ **Tuning and evaluation used different sensors.** `lidar_source` is
   `"reward"` (obstacle-only) during tuning but `"obs"` in the final config.

`eval_results/eval_suite_ppo/` — 500 episodes, 31.6 % success, from the
201 kB [64,64] checkpoint. Not architecture-matched to SAC, and it is precisely
the "arbitrarily crippled PPO" that brief §2's fairness note warns will read as
a straw man.

None of these will be reused. All three methods get fresh runs through the new
harness.

---

## 10.1 The SAC learning-curve data was overwritten; recovered from tfevents

⚠ Found while building the learning-curve figure. `plotting/plot_training_curves.py`
documents `eval_summary.json` as its "recommended main input", but that file
**now contains only 6 rows spanning 1,025,000–1,150,000 steps** — a later
fine-tuning run appended to and replaced the 0–1M data the published figure was
made from. The same applies to `eval_metrics.*` and `train_monitor.csv`: the
callback writes to fixed root-level paths with no run scoping, so every
subsequent run overwrites or appends to the previous one's record.

The 0–1M curve survives in TensorBoard events. Enumerating them:

| Event file | `eval/mean_ep_reward` | steps | final value | final success |
|---|---|---|---|---|
| `sac_log/asv_sac_1/...13004.0` | 19 pts | 50,688–950,272 | −213.9 | 0.70 |
| **`sac_log/asv_sac_2/...29056.0`** | **19 pts** | **50,104–950,152** | **+58.2** | **1.00** |
| `asv_sac_2/...27400.0` | — | 951k–1,449k | — | resumed run |
| `asv_sac_2/...26456.0` | — | 1,001k–1,499k | — | resumed run |
| three more in `asv_sac_2/` | — | 1,001k–1,199k | — | later experiments |

`asv_sac_2/...29056.0` is the published run — it is the only 0–1M log that ends
near the manuscript's performance. `asv_sac_1` is an earlier, worse attempt.
The curve is read straight from that file by `make_outputs.read_sac_curve`,
and the extracted numbers are written to `figures/learning_curves.csv` so the
figure never has to be regenerated from the event file again.

This is also why every PPO artifact in this task is written under
`models/ppo_seed{N}/` rather than to the repository root — see the
protected-artifact note in `src/train_ppo_baseline.py`.

---

## 11. Cost model (measured, for planning)

- Environment alone: **208 steps/s** single process, 12 cores available, **CPU-only torch**.
- SAC rollout incl. `predict`: **~3.7 s/episode** → ~31 min for 500 episodes.
  SB3 `predict` on a Dict observation dominates; the env is ~1 s of that.
- LOS+APF has no network, so ~1.1 s/episode is expected.
- Tuning: 200 configs × 100 episodes ≈ 20 000 episodes ≈ 6 h single-threaded.
  **The random search must be parallelised across the 12 cores** (~35 min).
- PPO 1M steps with 8 `SubprocVecEnv` workers plus a 50k-step eval callback:
  roughly 1.5–2 h per seed, so 2 seeds run overnight comfortably.

---

## 12. Summary of conflicts with the brief

| # | Brief assumes | Reality | Response |
|---|---|---|---|
| 1 | flat observation vector | `Dict` space, 34 dims, unnormalised | controllers accept a dict |
| 2 | both methods equally boundary-blind | `side_clearance_diff` / `local_target_cte` carry wall info | APF may use them; no raw geometry |
| 3 | curriculum active and scheduled | **no scheduler exists**; stages were hand-edited across resumed runs | replicate schedule on total env steps, in a subclass, for PPO only |
| 4 | build a frozen 500 set | already exists and is shipped | reuse `eval_suite/`; build only the tuning set |
| 5 | `min_border_clearance` may be half-width | measured value exists (mean 0.914 m) but is floored by the start pose; 2.00 m matches nothing computed | report all-wall **and** lateral-only |
| 6 | SAC baseline = `eval_suite_details.csv` | that file is a stale stage-4 run at 67.6 %; manuscript is `summary_1M.json` at 94 % | regenerate SAC on the frozen set |
| 7 | expose extra info via `info` dict | `hull_polygon()` and `obstacles` are already public | `env.py` untouched entirely |
| 8 | start from SB3 PPO defaults | a PPO config already exists but is [64,64]/γ=0.999 | match SAC: [256,256] ReLU, γ=0.99 |

Nothing here blocks the plan. Items 3 and 6 change what "identical conditions"
means concretely, and item 2 changes what the LOS+APF controller is allowed to
read — all three are settled above.
