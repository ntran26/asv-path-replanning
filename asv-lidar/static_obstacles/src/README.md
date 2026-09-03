# `src/` — cleaned-up main scripts

A rewrite of the ten main scripts in `paper_pooling/` with the same behaviour and
about a quarter fewer lines (4201 → 3096), split along the seams the original
code already had. Nothing outside this folder was modified.

Run everything **from the `paper_pooling/` directory**, so that relative paths
(`models/`, `eval_suite/`, `sac_log/`) resolve the same way they always did:

```bash
python src/train.py --mode test --model-path models/sac_model_1M.zip --test-case 2
```

## Layout

| File | Replaces | What it holds |
|---|---|---|
| `config.py` | constants at the top of `rl_env.py` | basin, reward, curriculum |
| `ship.py` | `ship_model.py` | 3-DOF hull, RK4 integrator, vessel geometry |
| `lidar.py` | `asv_lidar.py` + `lidar_pooling.py` | ray casting and the three sector-pooling modes |
| `path.py` | path helpers in `rl_env.py` | `ReferencePath`: arc length, tangents, tracking errors |
| `obstacles.py` | obstacle generators in `rl_env.py` | `ObstacleSampler`: the five layout families |
| `scenarios.py` | `test_run.py` | hand-authored test cases, as lookup tables |
| `render.py` | `rl_env.render()` + `images.py` | pygame view and MP4 capture |
| `env.py` | `rl_env.py` | the Gymnasium env: reset, step, reward, `info` |
| `rollout.py` | the episode loops in three scripts | one `run_episode`, shared by all of them |
| `train.py` | `train_test_asv.py` | `--mode train\|test\|eval`, eval callback |
| `evaluate_suite.py` | `evaluate_sac_suite.py` + `evaluate_agent_suite.py` | holdout-suite evaluation |
| `generate_suite.py` | `generate_eval_suite.py` | builds the fixed holdout |

## Commands

Training, unchanged apart from the dropped `--*-lambda` flags:

```bash
python src/train.py --mode train --algo sac --timesteps 1000000 --num-envs 8 --seed 675973 --eval-freq 50000 --save-freq 100000 --model-path sac_paper_pooling.zip
```

Visual rollout, writing `asv_data.json` and `asv_lidar.mp4`:

```bash
python src/train.py --mode test --model-path models/sac_model_1M.zip --test-case 3
```

Holdout suite. `evaluate_suite.py` still uses a USER SETTINGS block rather than
argparse; the defaults reproduce the SAC baseline files in
`eval_results/eval_suite/`. For PPO, set `ALGO="ppo"`,
`OUT_DIR="eval_results/eval_suite_ppo"` and `FILE_PREFIX="ppo_eval_suite"`
together — that reproduces what `evaluate_agent_suite.py` used to write.

```bash
python src/evaluate_suite.py
```

## What is guaranteed identical

Everything that affects a result: dynamics, LiDAR ranges and pooling, the
observation, all ten reward terms, termination, the random layout draws for a
given seed, and every key in `info`. Checked by direct comparison against the
original modules:

* `ShipModel`, 2000 steps of swept RPM and rudder — max difference `0.0`.
* `Lidar.scan` and all three pooling modes over 200 random scenes and 300
  random range vectors — max difference `0.0`.
* Full env rollouts against `rl_env.ASVLidarEnv` — 58 cases covering ten random
  seeds, forced obstacle counts 0–5, test cases 0–20 and a 12 × 30 m basin,
  comparing observations, reward, termination flags and every `info` value at
  every step — all bit-identical.
* The SAC 1M checkpoint on holdout scenarios and on the training-time eval grid
  — every metric in the details and summary rows identical.

The vectorised LiDAR is also about 7× faster than the per-beam loop, which is
the bulk of environment step time.

## Deliberate differences

Nothing here changes a number, but they are changes:

* `--train-lambda`, `--eval-lambda`, `--test-lambda` and the env's
  `lambda_override` are gone. λ has been fixed at `DEFAULT_EVAL_LAMBDA = 0.5`
  internally for a while and the argument was explicitly ignored.
* Fixed layouts are replayed with `env.reset(options={"scenario": ...})` instead
  of the evaluation script reaching into ten private env attributes.
* Test-case ids 1000–1099 are gone. They read `suite["cases"]` from
  `data/env_setup/eval_suite/asv_eval_suite_100_harder.json`, which does not
  exist in this tree and has a different schema from the suite we generate. Use
  `evaluate_suite.py` for suite evaluation. Case 99 is kept.
* `TestCase.path_waypoints()` is gone; the curves it returned are kept as data
  in `scenarios.CURVED_WAYPOINTS`. Nothing ever consumed them, so cases 8, 9 and
  13 run with straight reference paths — as they always have.
* `--mode test` no longer prints the action and clearances on every step.
* Dead constants removed: `LAMBDA_MIN/MAX`, `GAMMA_E`, `GAMMA_THETA`, `RPM_MIN`,
  `RPM_MAX`, `U_MAX`, `K_LOCAL_TARGET`, `K_CENTER_BLOCK`, `K_BORDER`,
  `OBSTACLE_MODE`, `HULL_FORWARD_SHIFT`, `SURGE_INERTIA_SCALE`,
  `YAW_INERTIA_SCALE`, `BOW_THRUSTER_YAW_GAIN`, `TR`. All were unused or fixed
  at a no-op value; `ShipModel.update()` lost its unused `thruster_rpm`
  argument with the last of those.
* `info["r_local"]` and `info["r_center"]` are still emitted as `0.0`. The terms
  are retired, but they are columns in `eval_metrics.csv`, so removing them
  would break appends to the existing file.

`info` deliberately still omits `r_cte_recovery` and `r_wrong_side`, matching
the original schema. `env._reward()` returns both by name — drop the filter at
the end of `_build_info` to start logging them, which the reward analysis asks
for before running ablations.

## Notes carried over, not fixed

* `generate_suite.py` writes to `data/env_setup/eval_suite/`, not the
  `eval_suite/` the evaluator reads. That is the original behaviour and it is
  the safer one: the holdout is fixed, and regenerating on top of it would make
  past results incomparable. Copy across deliberately.
* The suite metadata string still says "600-case … 0..5" for a 500-case 0..4
  suite, kept verbatim so a regenerated file matches the shipped one.
* The checked-in `eval_results/eval_suite/` numbers no longer reproduce from
  `rl_env.py` + `models/sac_model_1M.zip` — case 0 runs 226 steps today against
  the 240 recorded there. That gap is in the original code too (checked by
  running `evaluate_sac_suite.py` itself), so the stored results predate some
  change to the env or the checkpoint. Worth re-running before quoting them.
