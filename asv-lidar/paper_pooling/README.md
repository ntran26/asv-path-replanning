# Paper pooling branch

Pooling mode: `paper`

Observation keys:

```text
lidar, u, v, yaw_rate, cross_track_error, course_error,
lookahead_course_error, front_clearance, side_clearance_diff,
local_target_cte
```

`log10_lambda` has been removed. The reward balance is fixed internally at
`DEFAULT_EVAL_LAMBDA = 0.5`.

`front_clearance`, `side_clearance_diff`, and `local_target_cte` are kept because
they are still used by the local-planner observation and by the debug/eval
metrics. They are derived from the LiDAR sectors, not from an extra sensor.

## Train

```bash
python train_test_asv.py   --mode train   --algo sac   --timesteps 1000000   --num-envs 8   --seed 675973   --eval-freq 50000   --save-freq 100000   --model-path sac_paper_pooling.zip
```

## Generate a fixed evaluation suite

```bash
python generate_eval_suite.py
```

## Evaluate a trained model on the fixed suite

```bash
python evaluate_sac_suite.py   --model-path sac_paper_pooling.zip   --suite-json data/env_setup/eval_suite/asv_eval_suite.json   --out-dir eval_results/paper_pooling
```

## Live/shadow deployment adapter

Use this folder's matching `udp_live_rl.py` for this trained model. The branch
hardcodes the same pooling mode as simulation.
