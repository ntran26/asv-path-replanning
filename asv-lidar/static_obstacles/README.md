# ASV Path Following & Static Obstacle Avoidance with Deep RL

Simulation, training, and evaluation code for an **Autonomous Surface Vessel (ASV)** that follows a
reference path while avoiding static obstacles, trained end-to-end with **SAC** (Stable-Baselines3).
This is the `paper_pooling` branch — LiDAR sector pooling is hardcoded to `"paper"` mode so multiple
branches can be trained simultaneously without environment-variable collisions.

Supports the journal manuscript *"…Autonomous Surface Vessels: Path Following and Static Obstacle
Collision Avoidance using Deep Reinforcement Learning"*.

---

## Overview

| | |
|---|---|
| **Task** | 3-DOF ASV navigates a 10 × 25 m basin from start to goal, following a straight reference path, avoiding 0–4 static obstacles |
| **Algorithm** | SAC, `MultiInputPolicy`, 1M steps, 8 parallel envs |
| **Observation** | `Dict` — 25 pooled LiDAR sectors + 9 scalars (34 dims total) |
| **Action** | `Box(2)` in `[-1, 1]` — `[rudder, throttle]` |
| **Control rate** | 10 Hz (`UPDATE_RATE = 0.1 s`), episode cap 700 steps (70 s) |
| **Headline result** | **94% success** over the 500-case holdout suite |
| **Active env file** | `rl_env.py` (1399 lines) — the reward function lives in `step()`, lines 1161–1245 |
| **Protected checkpoint** | `models/sac_model_1M.zip` — **do not overwrite**, this is the 94% baseline |
| **Known problem** | The reward function is over-complicated and the obstacle term is ~50× too weak. See `REWARD_ANALYSIS.md` |

---

## Results

Evaluated on the fixed 500-case holdout suite (`eval_suite/`, 5 groups × 100 cases, 0–4 obstacles).

| Method | Overall | obs 0 | obs 1 | obs 2 | obs 3 | obs 4 |
|---|---|---|---|---|---|---|
| **SAC (1M, baseline)** | **0.940** | 1.00 | 0.92 | 0.92 | 0.95 | 0.91 |
| PPO (1M) | 0.316 | 0.98 | 0.37 | 0.11 | 0.10 | 0.02 |
| LOS + APF (classical) | 0.494 | 1.00 | 0.36 | 0.28 | 0.41 | 0.42 |

SAC baseline failure split: 3% obstacle collision, 3% border collision, 0% timeout.
Note border collisions *rise* with obstacle count (0% → 8% at 4 obstacles) — avoidance maneuvers push
the vessel toward the basin walls. Track both rates separately when changing the reward.

### Reward-repair experiment history (all four staged runs)

| Stage | Suite file | Overall | Verdict |
|---|---|---|---|
| 1 (= 1M baseline) | `eval_suite_summary_stage_1.json` | **0.940** | best |
| 2 | `eval_suite_summary_stage_2.json` | 0.924 | worse |
| 3 | `eval_suite_summary_stage_3.json` | 0.894 | worse |
| 4 | `eval_suite_summary_stage_4.json` | 0.924 | worse |

**Every targeted reward repair so far has degraded performance relative to the 1M baseline.** This is
the central open problem in this repo — see "Known issues" below.
(`eval_suite_summary.json` at 0.676 is a stale/earlier run, not a stage result.)

---

## Quick start

```bash
pip install -r requirements.txt
# numpy, pygame, gymnasium, stable-baselines3[extra], torch, opencv-python, rich, tqdm
```

### Train

```bash
python train_test_asv.py \
  --mode train --algo sac \
  --timesteps 1000000 --num-envs 8 --seed 675973 \
  --eval-freq 50000 --save-freq 100000 \
  --model-path sac_paper_pooling.zip
```

Checkpoints → `models/`, TensorBoard → `sac_log/`, episode monitor → `train_monitor.csv`.
Resume with `--resume --model-path <ckpt> --replay-buffer-path <buf>`.

### Evaluate on the fixed suite

`evaluate_sac_suite.py` uses a **USER SETTINGS block at the top of the file, not argparse**. Edit:

```python
MODEL_PATH = "models/sac_model_1M.zip"
SUITE_JSON = "eval_suite/asv_eval_suite.json"
OUT_DIR    = "eval_results/eval_suite"
```

then `python evaluate_sac_suite.py`. Same pattern for `evaluate_agent_suite.py` (set `ALGO` to
`sac`/`ppo`/`td3`/`ddpg`; note it currently has **two** `MODEL_PATH` assignments — the second wins).

Classical baselines *do* use argparse:

```bash
python evaluate_los_apf_suite.py  --method los_apf --out-dir eval_results/los_apf_baseline
python evaluate_los_apf_tuning.py --preset conservative_side --grid-search
```

### Regenerate the holdout suite

```bash
python generate_eval_suite.py     # writes eval_suite/asv_eval_suite.json + cases/
```

### Visual rollout / debugging

```bash
python train_test_asv.py --mode test --model-path models/sac_model_1M.zip --test-case 3
```

---

## Repository layout

```
paper_pooling/
├── rl_env.py                      # ACTIVE Gymnasium env — dynamics glue, obs, REWARD, scenarios
├── ship_model.py                  # 3-DOF MMG-style hull model, RK4 integrator
├── asv_lidar.py                   # Raw LiDAR simulation (225 beams, 270°, 16 m)
├── lidar_pooling.py               # Beam→sector pooling: "min" | "paper" | "corridor"
├── test_run.py                    # Deterministic hand-authored test cases (0–7 + extensions)
├── images.py                      # Boat sprite for render_mode='human'
│
├── train_test_asv.py              # Entry point: --mode train|test|eval, --algo sac|ppo
├── generate_eval_suite.py         # Builds the fixed 500-case holdout
├── evaluate_sac_suite.py          # SAC eval (USER SETTINGS block)
├── evaluate_agent_suite.py        # Generic SB3 eval (USER SETTINGS block)

│
├── models/                        # sac_model_1M.zip (BASELINE), 1M5, 1.1M, 1.2M, ppo_1M
├── eval_suite/                    # asv_eval_suite.json + cases/ (500 JSON scenarios)
├── eval_results/                  # Suite outputs: eval_suite/, eval_suite_ppo/, los_apf_*/
├── plotting/                      # Paper figures: training curves, sim-vs-field trajectories
├── sac_log/                       # TensorBoard events
├── rl_log_viewer.py               # Offline replay of real Bluefin field logs
├── fake_vessel_replay.py          # UDP replay harness for deployment testing
```

---

## Environment specification

### Observation — `Dict`, 34 dims

| Key | Shape | Range | Notes |
|---|---|---|---|
| `lidar` | (25,) | [0, 1] | Pooled sector *closeness* (1 = touching, 0 = clear) |
| `u` | (1,) | [0, 5] | Surge velocity, m/s |
| `v` | (1,) | [-3, 3] | Sway velocity, m/s |
| `yaw_rate` | (1,) | [-180, 180] | deg/s |
| `cross_track_error` | (1,) | ±map | **Signed** CTE, m |
| `course_error` | (1,) | [-180, 180] | deg |
| `lookahead_course_error` | (1,) | [-180, 180] | deg, lookahead = 0.25 × path length |
| `front_clearance` | (1,) | [0, 16] | m, derived from sectors |
| `side_clearance_diff` | (1,) | [-16, 16] | right − left clearance, m |
| `local_target_cte` | (1,) | ±map | m |

`log10_lambda` was **removed** from the observation; λ is fixed internally at `DEFAULT_EVAL_LAMBDA = 0.5`.
The last three keys are derived from LiDAR sectors, not an extra sensor.

### Action — `Box(2)`, `[-1, 1]`

- `action[0]` → rudder. `rudder = action[0] * 100` (a **percent** command), mapped internally to
  ±`MAX_RUD_ANGLE = 40°`, rate-limited to 20 °/s.
  **Sign convention: `delta_cmd = -clip(rud)/100 * max_rudder_rad` — positive action gives a NEGATIVE
  rudder angle.** This inversion is a recurring source of bugs; verify empirically before trusting it.
- `action[1]` → throttle. `rpm = clip(CRUISE_RPM + RPM_DELTA * action[1], RPM_FLOOR, RPM_CEIL)`,
  `CRUISE_RPM = 12.0`.

### Vessel & sensor

`VESSEL_LENGTH = 1.725 m`, `VESSEL_WIDTH = 0.50 m`, `HULL_MARGIN = 0.15 m` (inflated collision hull),
`MASS = 64.55 kg`, `DRAFT = 0.193 m`. LiDAR mounted at `VESSEL_LENGTH/2` forward.
LiDAR: **225 raw beams**, 270° swath, 16 m range → pooled to **25 sectors** in `"paper"` mode
(feasibility-based, safe width = `VESSEL_WIDTH + 2*HULL_MARGIN`).

### Termination

| Condition | Signal | Reward |
|---|---|---|
| Collision (obstacle **or** basin wall, SAT polygon test on inflated hull) | `terminated` | `-1000` (replaces all dense terms) |
| Goal reached (`GOAL_RADIUS 0.5`, `GOAL_ALONG_DIST 1.25`, `GOAL_CTE_RADIUS 1.60`) | `terminated` | `+50` |
| 700 steps elapsed | `truncated` | `-1000` (**added**, not replacing) |

### Training scenario mix (`TRAIN_SCENARIO_PROBS`)

`normal` 0.40 · `target_side` 0.35 · `field_repair` 0.15 · `gate` 0.05 · `offpath` 0.05

Start/goal are vertical with probability 0.7. Map defaults 10 × 25 m, `path-mode straight`.

### Hyperparameters

**SAC** (used for the baseline): `lr 5e-5`, `batch 512`, `gamma 0.99`, `buffer 1e6`, `train_freq 1`,
`gradient_steps 1`, `ent_coef "auto"`, `MultiInputPolicy`.
**PPO**: `lr 1e-4`, `batch 256`, `gamma 0.999`, Tanh, `net_arch pi=[64,64] vf=[64,64]`.

---

## Reward function

Ten terms. Full breakdown, measured magnitudes, and diagnosis in **`REWARD_ANALYSIS.md`** — read that
before touching the reward.

```python
reward = (
      lambda_ * (u_gate * r_pf)        # exp(-gamma_e * |CTE|), speed-gated
    + (1 - lambda_) * r_oa             # -mean(w_i / max(d_i,1)) over 225 beams
    + 0.35 * u_gate * r_heading        # cos(0.7*lookahead_err + 0.3*course_err)
    + r_exist                          # -0.5 living cost
    + r_border                         # -0.40 * max(0, 1 - clr/1.0)^2
    + r_progress                       # 0.6 * (d_prev - d_now)
    + r_slow                           # -0.10 * max(0, 0.30 - u)
    + r_thrust                         # -0.025 * |rpm - CRUISE| / RPM_DELTA
    + r_cte_recovery                   # 0.35 * (|CTE|_prev - |CTE|_now)   [not in paper]
    + r_wrong_side                     # -0.12 * |rudder| under 4-way AND  [not in paper]
)
```

`gamma_e` is **adaptive**: `block_alpha = clip((4.5 - front_clearance)/(4.5 - 2.0), 0, 1)` blends
`GAMMA_E_CLEAR 0.20` → `GAMMA_E_BLOCKED 0.05`, relaxing the path penalty near obstacles.

### Measured per-step magnitudes

| Term | Magnitude |
|---|---|
| `λ·g_u·r_pf` (on path) | 0.500 |
| `r_exist` | 0.500 |
| `r_border` (at wall) | 0.400 |
| `w_χ·g_u·r_χ` (aligned) | 0.350 |
| `r_cte_recovery` (0.3 m/step) | 0.105 |
| **`(1-λ)·r_oa` at 1 m contact** | **0.0103** |
| `(1-λ)·r_oa` open water | 0.0012 |

---

## Known issues (read before modifying)

1. **`r_oa` is ~49× weaker than `r_pf` even at contact.** λ = 0.5 is nominally a 50/50 blend but is
   effectively ~98/2. Avoidance is being learned almost entirely from the sparse `-1000` terminal, not
   from the dense signal.
2. **`r_oa` carries no directional information.** Its angular weighting `w_i = 1/(1+|θ_i|)` is symmetric,
   so it cannot express "go right rather than left." Side-choice has no dense gradient — the root cause
   of the wrong-side failure mode that `r_wrong_side` was added to patch.
3. **`r_cte_recovery` contradicts the adaptive `gamma_e`.** `block_alpha` relaxes the path penalty to
   permit deviation; `r_cte_recovery` is not gated by it and punishes that same deviation at full strength.
4. **`R_TIMEOUT == R_COLLISION == -1000`**, so a doomed episode has no incentive to avoid a late crash.
   Applying a terminal penalty on time-limit truncation is also formally incorrect (should bootstrap).
5. **Four redundant anti-stall devices** (`r_progress`, `r_slow`, `u_gate`, `r_exist`); `r_slow` maxes at
   -0.03 and `r_thrust` at ±0.025, both negligible against `r_exist = -0.5`.
6. **`r_cte_recovery` and `r_wrong_side` are not logged in `info`** — every other dense term is. Add them
   before running ablations, or the two most suspect terms are invisible.
7. **Dead code:** `r_local`, `r_center`, `K_CENTER_BLOCK = 0`, `K_BORDER = 0`.

### Documentation drift

| Item | Claim | Reality |
|---|---|---|
| `pooling_manifest.json` | `lidar_beams: 90` | `asv_lidar.py` → `LIDAR_BEAMS = 225` |
| Docstrings in `generate_eval_suite.py`, `evaluate_sac_suite.py` | "600-episode suite" | 500 cases |
| Old README eval command | `data/env_setup/eval_suite/…` | actual path `eval_suite/asv_eval_suite.json` |
| Old README | references `udp_live_rl.py` | not present in this tree |
| Manuscript Eq. 19 | 8 terms, flat `gamma_e = 0.05`, `K_border 0.25`, `d_safe 0.7`, `K_progress 0.7` | code has 10 terms, adaptive `gamma_e`, `0.40`, `1.0`, `0.6` |

The manuscript's Table 1 constants **do not currently match the trained code.** Reconcile before submission.

### Verified as *not* problems

- `r_oa` uses obstacle-only LiDAR (`map_border=None`) while `r_border` uses true border clearance →
  no double-counting.
- `_border_clearance_true()` and `_check_collision_geom()` share the same boundary definition →
  `r_border` shapes toward the real collision surface.
- `r_progress` telescopes to `K·(d_start − d_end)` → effectively potential-based, does not distort the
  optimal policy. Keep it.

---

## Env variants

All four share the same dynamics, observation space, and `GAMMA_E`/`BLOCK_D_*` constants. They differ in
the training scenario mix and which repair terms are active.

| File | Scenario modes | Probs | Repair terms |
|---|---|---|---|
| `rl_env.py` **(active)** | normal, target_side, field_repair, gate, offpath | .40/.35/.15/.05/.05 | `r_cte_recovery`, `r_wrong_side` |
| `rl_env_targeted_side_choice_repair_stage1.py` | same | same | same |
| `rl_env_updated_gate_repair.py` | normal, gate, field_repair, offpath | .30/.40/.20/.10 | none |
| `rl_env_conservative_gate_repair_stage1.py` | normal, gate, field_repair, offpath | .65/.20/.10/.05 | none |

`rl_env.py` differs from `rl_env_targeted_side_choice_repair_stage1.py` by **exactly one line**:
`OBS_BORDER_MODE = "none"` vs `"both"` (whether basin walls appear in the observation LiDAR).

---

## Working conventions

- **Never overwrite `models/sac_model_1M.zip`.** It is the 94% baseline and the reference for every ablation.
- Fine-tune **short increments** from that checkpoint rather than retraining from scratch; then re-evaluate
  on the full 500-case suite before drawing conclusions.
- Report **obstacle collision rate and border collision rate separately.** A flat headline success rate can
  hide the failure mode migrating from obstacles to walls.
- The eval suite is a **fixed holdout** — do not regenerate it mid-study or results become incomparable.
- Evaluation is deterministic (`DETERMINISTIC = True`, `MAX_STEPS = 2000` — note this is larger than the
  training cap of 700).
- Training seed of record: `675973`.
