# PORTING MANIFEST — Paper 2 → Paper 3

**Produced at:** commit `986245c73d62cde2cb1dedf16166783ecdf233c8` (branch `master`, tree clean).
**Reference read:** `static_obstacles/src/` (read-only). Nothing under `static_obstacles/` was written.
**Authoritative spec:** `planning/01_PERCEPTION_AND_OBSERVATION.md`.

Step 0 deliverable per `KICKOFF_01_PERCEPTION.md` §2, **updated after the build and
again for planning Revision 2**.
Sections 0-3 are the pre-build inventory, left as written so the findings can be read
against what was known at the time. Section 4 records what the build changed.
**Section 5 records the two-vessel repositioning** and is the current state.

---

## 0. Which tree is the baseline

`static_obstacles/` contains two parallel copies of Paper 2:

| | Files | Status |
|---|---|---|
| Root level | `rl_env.py` (1399), `ship_model.py`, `asv_lidar.py`, `lidar_pooling.py`, `test_run.py`, `images.py`, `train_test_asv.py`, `evaluate_*.py`, `generate_eval_suite.py` | Original. Kept for provenance. |
| `src/` | 25 modules, 3096 lines | Cleaned rewrite, **verified bit-identical** (`src/README.md`: ShipModel over 2000 swept steps max diff 0.0; `Lidar.scan` + all 3 pooling modes over 200 scenes / 300 range vectors max diff 0.0; 58 full-env rollouts identical on observation, reward, termination and every `info` key). |

The kickoff names `src/` and it is the better-factored source. **All porting below is from `src/`.**
Root-level files are not ported at all — porting them too would duplicate every module.

> `PROJECT_BRIEF.md` §2 calls the codebase `paper_pooling/`. No such directory exists;
> `static_obstacles/` is it, and `src/README.md` still refers to the old name internally.
> Naming only, no action needed.

---

## 1. Target layout  *(resolved: `src/constants.py`)*

Kickoff §4 lists new modules by bare filename; §5 puts `constants.py` at
`static_dynamic_obstacles/constants.py`. Paper 2 uses a flat `src/` package with flat
imports (`import config as cfg`), run from the parent directory. The two trees are kept
symmetric:

```
static_dynamic_obstacles/
    CONSTANTS_AND_SCALES.md
    OBSERVATION_SPEC.md
    PORTING_MANIFEST.md
    pytest.ini
    src/          constants.py + 19 modules
    tests/        7 files, 188 tests
```

`constants.py` sits in `src/` rather than at the package root, so the single-source-of-truth
property §5 asks for holds without `src/` modules importing from their parent. Confirmed
at review.

---

## 2. File-by-file assignment

Buckets are the kickoff §3 definitions. **D** is added for files that are out of scope for
task 01 and belong to a later task — the kickoff defines only A/B/C, but a third of the
tree is evaluation and baseline code that 04 owns.

### Bucket A — copy verbatim

| File | What it does | Note |
|---|---|---|
| `ship.py` | 3-DOF Fossen hull, RK4, vessel geometry, rudder servo. `VESSEL_LENGTH` 1.725, `VESSEL_WIDTH` 0.50, `HULL_MARGIN` 0.15 | Recalibration is 05's job, not this session's |
| `path.py` | `ReferencePath`: arc length, tangent, left-normal, `project()` → `e_y`, `χ̃`, `χ̃_LA` | Supplies the whole `path` branch unchanged |
| `rollout.py` | `run_episode`, the single act/step loop | |
| `curriculum.py` | `CurriculumASVLidarEnv` + SB3 callback for the staged RPM schedule | Header documents that the published 1M SAC run used **fixed RPM, no curriculum** — read before reusing |
| `train.py` | `--mode train\|test\|eval`, eval callback, checkpointing | Policy kwargs will need the custom extractor from build-order step 10 |
| `train_sac_baseline.py`, `train_ppo_baseline.py` | Multi-seed baseline training under a fixed protocol | |
| `evaluate.py` | Controller-agnostic eval harness, one CSV schema | |
| `evaluate_suite.py` | Holdout-suite evaluation | |
| `compare.py` | Paired statistics between two methods on the same layouts | |
| `metrics.py` | `EpisodeRecorder`, clearance geometry, IQM, stratified bootstrap CI | Copy now; 04 adds the COLREGs metrics |
| `render.py` | Pygame view + MP4 capture | Copy now; 03 adds target-ship and boundary rendering |
| `obstacles.py` | `ObstacleSampler`, five static layout families | Copy now so the ported env steps end-to-end; 03/04 own the channel generator |

### Bucket B — copy and modify, one stated change each

| File | Change |
|---|---|
| `lidar.py` | **Two changes, not one.** (i) Output becomes obstacles-only: the `map_border` argument and every border-visibility path are deleted (01 §3). (ii) Pooling accepts a **per-sector angular span Φ** and bins beams **by bearing** instead of `np.array_split` over a uniform index grid. See F5 — the second change is larger than the kickoff implies. |
| `env.py` | Split across buckets; see §2.1. The Bucket-B half is the step loop, action mapping, collision/termination geometry and `info` assembly. |

> **Note.** The kickoff lists `asv_lidar.py` and `lidar_pooling.py` as two separate Bucket B
> items. In `src/` they are merged into the single `lidar.py`. The kickoff's build-order
> item 4 (`lidar_pooling.py` with per-sector Φ) therefore maps onto splitting `lidar.py`
> back into a raycast module and a pooling module, which I intend to do — pooling needs its
> own unit tests and the field pipeline imports it independently.

### Bucket C — rebuild from specification (Paper 2 equivalents not opened)

Per D10, these are built from 01 alone. New modules, no Paper 2 ancestor:

| File | Source |
|---|---|
| `constants.py` + `CONSTANTS_AND_SCALES.md` | Kickoff §5 |
| `boundary_raycast.py` | 01 §3.2–3.4 — 7 rays, pose-noise hook |
| `tracking.py` | 01 §4 — gate → cluster → ego-motion compensate → associate → Kalman → static/dynamic split |
| `cpa_cri.py` | 01 §5.1–5.2 — CPA/DCPA/TCPA, CRI, ship domain |
| `encounter.py` | 01 §5.3 — one pure classifier + one hysteresis wrapper, single definition |
| `observation.py` | 01 §6 — 5-branch Dict, slot management, valid mask |
| `features_extractor.py` | 01 §6.3 — shared per-slot encoder φ, `aggregate` flag |

### Bucket D — out of scope for task 01

| File | Owner |
|---|---|
| `scenarios.py` | 04. Paper 2's 21 hand-authored cases are static-obstacle layouts; the named-case set is being replaced (D8 drops Imazu; O1 pending on Waltz & Okhrin "Around the Clock"). |
| `eval_layouts.py`, `generate_suite.py`, `build_hard_layouts.py` | 04. Frozen suite must be regenerated for the new scenario space. |
| `make_outputs.py`, `plot_success_by_obstacles.py` | 04. |
| `baselines/los_apf.py`, `tune_los_apf.py`, `verify_los_apf.py` | 04. Paper 3's classical baselines are LOS-PID+DWA, COLREGs-VO (Kuwata), Thyri & Breivik. LOS+APF is not on that list. Its `predict()` reads the Paper 2 observation keys and will not survive the Dict change regardless. |
| `baselines/__init__.py` | trivial |
| `config.py` | Superseded by `constants.py`; see §2.2 for the disposition of each block. |

### 2.1 `env.py` breakdown (636 lines, split three ways)

| Region | Bucket |
|---|---|
| `hull_polygon`, `_border_clearance`, `_hits_border`, `_collided`, `_polygons_intersect`, `_reached_goal`, `hit_border` | **A** — geometric collision and termination, unchanged (01 §3.4 "unchanged" clause) |
| `step()` action mapping (rudder %, RPM trim around cruise), episode bookkeeping | **A** |
| `reset()`, `_sample_layout`, `_load_scenario`, `_build_path`, `_scale_position` | **A**, extended by 03 for target spawning |
| `_scan_lidars`, `_obs_lidar_border`, `_sample_obs_border_mode` | **deleted** — the three-LiDAR scheme and the border-visibility mode both go (01 §3, kickoff §3) |
| `_get_obs`, `_update_local_planner_features` | **C** — rebuilt as `observation.py` |
| `_reward`, `_wrong_side_penalty`, reward terms in `_build_info` | **02** — not ported in this session |

### 2.2 `config.py` disposition

| Block | Disposition |
|---|---|
| Simulation (`UPDATE_RATE`, `MAX_EPISODE_STEPS`, render) | → `constants.py` verbatim |
| Basin + path (`MAP_WIDTH/HEIGHT`, `PATH_MODE`, `LOOKAHEAD_FRACTION`, start/goal) | → `constants.py`, but `MAP_WIDTH`/`MAP_HEIGHT` are **`TODO(O4)`** |
| Goal acceptance (`GOAL_RADIUS`, `GOAL_ALONG_DIST`, `GOAL_CTE_RADIUS`) | → `constants.py` verbatim |
| Actuation (`CRUISE_RPM`, `RPM_STAGES`) | → `constants.py` verbatim |
| **All reward constants** (`R_*`, `K_*`, `GAMMA_E_*`, `W_HEADING`, `U_REWARD_REF`, `DEFAULT_EVAL_LAMBDA`) | **not ported.** 02 owns these. Porting them now would violate the kickoff §8 check on Paper 2 reward names surviving into new code. |
| `OBS_BORDER_MODE`, `OBS_BORDER_P_*`, `RIGHT_WALL_OFFSET` | **deleted** — concept removed (kickoff §3) |
| `BLOCK_D_SAFE/CRIT`, `BLOCK_FRONT_DEG`, `SIDE_ARC_*`, `SIDE_CLEAR_TIE`, `BYPASS_CTE` | **deleted** — these parameterise `front_clearance` / `side_clearance_diff` / `local_target_cte`, three observation fields 01 §6 does not carry. See F3. |
| Obstacle curriculum (`TRAIN_OBS_*`, `TRAIN_SCENARIO_*`, gate/target_side/field_repair params) | → copied with `obstacles.py`; 03/04 replace |

---

## 3. Findings — things 01 or the kickoff state that the code or the data does not support

Flagged, not resolved, per kickoff §9.

### F1 — The Paper 2 observation is already a `Dict`, and it has 34 dims, not 31

Kickoff §3 Bucket C: *"Observation assembly — `Box` becomes a 5-branch `Dict`."*
There is no `Box` observation. `src/env.py:92` declares a `gymnasium.spaces.Dict` with
**ten** keys, and the root `rl_env.py:328` is identical. No Box→Dict migration exists;
the change is Dict(10 flat keys) → Dict(5 branches).

01 §1 gives the superseded observation as
`o_t = [c_t (M=25), u, v, r, e_y, χ̃, χ̃_LA]` — 31 dims. The shipped observation also
carries `front_clearance`, `side_clearance_diff` and `local_target_cte`, for 34.
No action on the code; **01 §1 needs a correction before it is quoted in the methods
section**, since it currently mis-describes the published baseline.

### F2 — The third LiDAR is not mentioned in 01

01 §3.1 credits Paper 2 with two views (obstacle-only for reward, border-configurable for
observation). There are **three**: `lidar_border_guard` scans walls only and supplies
`left_clearance`/`right_clearance`, which drive the side-choice bypass cue. Under D5 the
boundary branch subsumes it. Stating that explicitly is worth one line in 01 §3, because
"the split becomes total" currently reads as a 2-way split.

### F3 — Three Paper 2 observation fields are dropped silently

`front_clearance`, `side_clearance_diff` and `local_target_cte` (`src/env.py:341-372`) are
LiDAR-derived local-planner cues, not raw sensor data. 01 §6's five branches do not
include them and the dimension arithmetic (27+7+3+3+51=91) has no room for them.

I will treat them as **deliberately dropped** and proceed. Flagging because
`local_target_cte` is the bypass side-choice cue that Paper 2's `target_side` and
`field_repair` curricula were built to repair — dropping it removes an engineered feature
that a documented failure mode depended on. If that is intended (the position being that
the policy should learn side choice from `c_t` + `boundary` rather than be told), it is
worth one sentence in the methods, because a reviewer comparing observation tables will
notice three features disappearing.

### F4 — Paper 2 simulated 225 beams over 270°, not 720 over 360°

| | Paper 2 sim (`src/lidar.py:28-30`) | 01 §2 |
|---|---|---|
| Swath | 270° | 360° raw |
| Beams | 225 | 720 |
| Resolution | **1.205°** | 0.5° |
| Range | 16.0 m | 1–16 m |

01 §2.2's beams-per-sector table (12 beams in a 6° sector, 45 in a 22.5° sector) presumes
0.5°. So Paper 3 is changing the simulated sensor by 5.4× in angular resolution *and*
1.33× in swath, on top of the re-sectoring. 01 §2.2's implementation note says Algorithm 1
"carries over unchanged — it computes arc width from the per-beam angular resolution θ,
which stays constant at 0.5°". θ does not stay constant: it changes from 1.205° to 0.5°.
That is fine and correct — the real sensor is 0.5° — but it is a change to the published
pooling behaviour, not a no-op, and the methods sentence 01 asks for should say so.

### F5 — The per-sector Φ change is bigger than "pass Φ per sector"

`pool_to_sectors` (`src/lidar.py:130-152`) does two things that assume uniformity:

1. `np.array_split(raw, n_sectors)` splits the beam array **by index**, which equals an
   angular split only on a uniform grid.
2. `neighbour_angle = radians(swath / (raw.size - 1))` is a **global** constant computed
   from full swath and full beam count, passed identically to every sector.

Item 2 is genuinely unchanged under the new design (θ is per-beam and constant at 0.5°).
Item 1 is not: non-uniform sectors require binning beams by bearing against explicit
sector edges. Worth stating because the kickoff frames this as a signature change to
`meyer_feasibility_pool`, whereas that function needs no change at all — the change is
entirely in the beam-to-sector assignment above it.

Sector arithmetic in 01 §2.2 checks out: 15 + 8 + 4 = 27 sectors; 15×6° = 90° (±45°),
8×11.25° = 90° (2×45°), 4×22.5° = 90° (2×45°); total 270° = ±135°. ✓
Beam counts at 0.5°: 12, 22.5, 45 per sector. **The 45–90° band gives 22.5 beams per
sector, not an integer** — 01's table says "22–23", so the allocation must alternate.
That is a real constraint on the implementation and on the Φ bookkeeping; noting it so the
unit test asserts the alternating pattern rather than a constant.

### F6 — Field logs: 720 bins confirmed, but 01 §2.3's stated concern is the wrong one

Analysed all 30 logs under `field_deployment/` (5597 pooled scans from the four largest).
Format is one 720-element integer array per scan, units **decimetres** (values 10–153,
i.e. 1.0–15.3 m — consistent with the C1's 1–16 m spec).

**Item 1 (returns per revolution).** 720 bins in every scan of every log, so the array is
genuinely 0.5°. Mean **506** bins carry a non-zero return (5th–95th pct 424–588). But
**96.5% of the empty bins fall in contiguous runs longer than 3 bins** (median run 1–2,
max run ~260); only ~9 bins per scan look like isolated dropout. So the missing returns are
out-of-range / no-return arcs, **not** angular under-sampling.

01 §2.3 item 1 asks whether the C1 "may deliver closer to ~500 points/rev (≈0.7°) than the
nominal 0.5°". The 506 figure matches that number almost exactly, but the mechanism is
different: the resolution really is 0.5°. **Recommendation: keep 0.5°/720 in simulation
and instead model a no-return process** (probability of no return vs incidence angle and
range, plus a sparse isolated-dropout rate). Simulating 500 uniformly-spaced beams would
reproduce the count while getting the structure wrong. This changes what 01 §2.3 item 1
asks for and is worth resolving before the simulator is frozen.

**Item 2 (aft self-occlusion sector).** No evidence of one in these logs. No bin is zero in
more than 98% of scans in any log. The peak zero-rate bearing is **not stable across logs**
— 186°, 169°, 108°, 216°, 344°, 359°, 4.5°, 24.5°, 355°, 338° — so it tracks the scene,
not the mount. A pooled peak near 185° appears only because the two largest logs dominate.
**Cannot be characterised from this data.** Settling it needs a dedicated static-spin log
with the vessel stationary in a known surround; `2026-07-02/calibration.log` is not that
(its peak sits at 344.5°). Recommend adding it to the next basin session — it is ten
minutes of recording and it gates the overtaking-detection claim.

**Item 3 (motion distortion).** Not assessable from these logs — they contain no per-beam
timestamps, only one wall-clock stamp per revolution. Requires either a raw ROS bag with
`sensor_msgs/LaserScan` `time_increment`, or characterisation by construction from the
odometry. Flagging as blocked on data, not on effort.

**New gap not in 01: the C1 has a 1 m minimum range.** Confirmed — the smallest non-zero
value across all logs is 10 dm = 1.0 m. `src/lidar.py` has no minimum-range dead zone; it
reports ranges down to 0. With the sensor at the bow (`LIDAR_OFFSET_M` = 0.8625 m) and a
1.57 m LBP hull, a target ship alongside within 1 m is **invisible to the real sensor and
fully visible in simulation**. That lands directly on the tracker and on close-quarters
COLREGs behaviour. Recommend a `LIDAR_MIN_RANGE = 1.0` constant applied in both pipelines.

### F7 — The boundary branch carries no information in the current workspace

01 §3.3's redundancy trap is not hypothetical here. Paper 2's basin is a
constant-width 10 × 25 m rectangle and 70% of episodes use a path parallel to the walls
(`VERTICAL_PATH_PROB = 0.70`, straight paths, `PATH_MODE = "straight"`). Under exactly
those conditions the 7 boundary rays are an affine function of `e_y` and heading.

Building `boundary_raycast.py` this session is still correct — the interface is needed and
the field-side gating is a prerequisite for the tracker regardless. But **its value is
gated on O4 and on 04's scenario generator producing varying width, off-centre paths, or
bends.** Until then any ablation of the boundary branch will correctly show no effect,
which would be the wrong conclusion to draw. Noting so it is not measured prematurely.

### F8 — `rl_env_dynamic.py` already in the working directory is a pre-Paper-2 prototype

`static_dynamic_obstacles/rl_env_dynamic.py` (589 lines, dated 2026-04-10) predates
Paper 2: pixel coordinates (`MAP_WIDTH = 400`, `MAP_HEIGHT = 600`), `UPDATE_RATE = 0.5`,
discrete rudder (`PORT/CENTER/STBD` at ±25), `COLLISION_RANGE = 10`. It matches the
description of the stale repo-root `README.md` the kickoff says to ignore. It does contain
an early moving-obstacle TCPA sketch (`DYN_PREDICT_HORIZON`, `DYN_TCPA_EPS`).

Recommend moving it to `archive/` so it is not mistaken for a starting point. **Not moved —
it is outside the read-only boundary but I would rather not relocate a file without a
decision.**

### F9 — Open decisions this session will hit

| | Effect on task 01 |
|---|---|
| **O4** (workspace size) | Blocks final values for `d_scale`, TCPA clip bounds, speed normalisers, and the ship-domain extents (01 §5.2's 4.7 m fore-aft in a 10 m channel). Placeholders will be `TODO(O4)`. |
| **O5** (physical barrier vs software gating) | Does not block — 01 §3.4 requires the gate either way, and the barrier only removes its localisation dependency. |
| **05** (rf2o drift) | Blocks the boundary pose-noise magnitude and the Kalman noise matrices. Hooks exist, defaults 0.0 / placeholder. |
| **O2** (give-way-only vs reciprocity) | Does not block the observation — the 6-way one-hot has the "being overtaken" slot either way — but it decides whether that class is ever populated in training. |

---

## 4. Decisions taken, and the build

All four questions raised at manifest review were answered:

1. `front_clearance`, `side_clearance_diff`, `local_target_cte` — **dropped**.
2. Constants live at **`src/constants.py`**.
3. Keep **720 beams at 0.5°** and model no-returns; downsample later if the model is
   overloaded.
4. `rl_env_dynamic.py` — **deleted** by the user.

Build order followed the kickoff §4 sequence with one reordering: `constants.py` was
written *before* the Bucket A copies rather than after, because every Bucket A module
imports it (`import config as cfg` → `import constants as cfg`).

### 4.1 Delivered

| File | Bucket | Tests |
|---|---|---|
| `src/constants.py` | C | — |
| `src/lidar_pooling.py` | B | `test_lidar_pooling.py` (17) |
| `src/asv_lidar.py` | B | via env tests |
| `src/boundary_raycast.py` | C | `test_boundary_raycast.py` (18) |
| `src/tracking.py` | C | `test_tracking.py` (23) |
| `src/cpa_cri.py` | C | `test_cpa_cri.py` (19) |
| `src/encounter.py` | C | `test_encounter.py` (21) |
| `src/observation.py` | C | `test_observation.py` (35, shared with the extractor) |
| `src/features_extractor.py` | C | ″ |
| `src/env.py` | B/C | `test_env.py` (24) |
| Bucket A copies (12 files) | A | `test_ported_harness.py` (31) |
| `CONSTANTS_AND_SCALES.md`, `OBSERVATION_SPEC.md` | — | — |

**188 tests, all passing, in ~8 s.**

### 4.2 F10 — "Bucket A, copy verbatim" was not viable for the harness

The kickoff put "SB3 training harness, curriculum machinery, logging, evaluation loop"
in Bucket A. Several of those files do not run against the Paper 3 environment, because
their **logging schema is coupled to Paper 2's reward terms and to the three dropped
observation fields**. They import cleanly and fail at runtime, which is the worst
failure mode available — it would have surfaced partway through a training run.

| File | Breakage | Change made |
|---|---|---|
| `rollout.py` | `KeyError: 'lam'` — indexed `info[key]` directly over a hard-coded list containing `r_pf`, `r_oa`, `lam`, `front_clearance`, `block_alpha`, `local_target_cte` | New key list; **missing keys are skipped rather than raising**, so 02 can add reward terms without touching the loop. `lidar.angles` → `.bearings`. `termination_reason` now returns the three collision kinds. `front_clearance_stats` renamed `forward_beam_stats` — it is a raw-beam diagnostic and the old name collided with the dropped observation field. |
| `train.py` | `env.current_lambda` (gone); `mean_r_pf` / `mean_r_oa` / `mean_lambda` / `max_block_alpha` columns; `side_path_guard` read all three dropped fields | Reward columns replaced with perception columns (`min_sector_range`, `max_boundary_closeness`, `min_border_clearance`, `max_tracks`, `target_rate`). **`side_path_guard` removed entirely** — see below. |
| `metrics.py` | `min_front_clearance` derived from a dropped field | Metric removed |
| `train_sac_baseline.py`, `train_ppo_baseline.py` | `cfg.OBS_BORDER_MODE` (deleted) | Logging line removed |
| `render.py` | `env.local_target_cte`, `env.front_clearance`, `env.left/right_clearance`, `from lidar import` | Status line rebuilt; target ships, tracker estimates and the 7 boundary rays added |

**`side_path_guard` deserves its own flag.** It was a test-mode action filter that forced
corrective rudder whenever the path-recovery direction and the clearer side agreed — a
hard-coded side choice, built entirely on `front_clearance`, `side_clearance_diff` and
`local_target_cte`. Dropping those three makes it unimplementable, and Paper 3's position
is that side choice is learned from `lidar` plus `boundary`. It is **not** carried across.
If any Paper 2 result was produced with the guard active, that is worth establishing
before the two are compared; reinstating it would be a decision for 02, not an
inheritance.

`tests/test_ported_harness.py` now asserts that every tracked `info` key is actually
emitted by the environment, so this class of coupling cannot return silently.

### 4.3 Bucket reassignments made during the build

- `evaluate.py`, `evaluate_suite.py` → **D**. Both depend on `eval_layouts.py`
  (Bucket D) or on the frozen suite, which 04 regenerates. Not copied.
- `compare.py` stays **A** — it operates on per-episode CSVs and is schema-agnostic.

### 4.4 F11 — the boundary gate was 85% of step time until vectorised

The first working version ran at **40.7 steps/s (24.6 ms/step)** — 6.8 h per 1M steps on
a single env. Profiling put 85% of that in `gate_beams`, which tested all 720 beam
endpoints against the boundary polygon in a Python loop.

Vectorised — loop over the polygon's four edges instead of over the 720 points, and skip
beams that returned nothing — it runs at **685 steps/s (1.46 ms/step)**, 0.41 h per 1M
steps. A 17× speed-up with no behavioural change; the full suite passes identically.

Worth recording because the pooled scan is now 540 beams over 27 sectors where Paper 2
had 225 over 25, and `feasibility_pool` is O(n²) per sector in the worst case. It is not
the bottleneck today, but it is the next one if the beam count grows.

### 4.5 A correction to F5, found while implementing it

F5 predicted `meyer_feasibility_pool` would need no change. That held. But the second
half of F5 understated the problem: Paper 2's
`neighbour_angle = radians(swath / (n_beams - 1))` is not merely "a global constant" —
it is numerically **1.205°**, where the real sensor is 0.5°. So pooled ranges are not
comparable between the two papers even for an identical scene and identical sector
edges. The algorithm carries over; the numbers do not. Same root cause as F4, but it
lands on the pooling output specifically, which is what the methods sentence has to say.

### 4.6 New findings from the build

**F12 — the environment as shipped is not trainable, by design.** `_reward` is sparse
terminal payoff only. A policy trained against it will not learn the task. 01 ships
perception; 02 ships the reward. Stated here so nobody mistakes a flat learning curve for
a perception bug.

**F13 — target ships exist but nothing spawns them.** `TargetShip` implements constant
velocity (D1) and the full perception chain tracks it correctly — there is an end-to-end
test that spawns a head-on target and confirms it reaches an observation slot. But
`_sample_layout` creates none, because spawning geometry is 03's. Until 03 lands, every
training episode has zero targets and three masked slots.

**F14 — `DYNAMIC_SPEED_ON` is currently below the drift floor it has to clear.**
`tests/test_tracking.py::test_pose_drift_creates_false_velocity_on_a_static_object`
shows 0.02 m/step of pose drift (0.2 m/s apparent) is enough to misclassify a fixed
object as a target ship at the placeholder threshold of 0.15 m/s. That test currently
*asserts the failure*, to document the mechanism. Once 05 supplies the rf2o drift figure,
either the threshold moves above it or the tracker needs drift compensation — and the
test should be inverted to assert the object stays static.

---

## 5. Revision 2 — the two-vessel repositioning

The planning documents were revised on 2026-09-06, repositioning Paper 3 from
multi-vessel to **two-vessel encounters**. This section records what that
changed in the code. Sections 0–4 are left as written.

### 5.1 What the repositioning changed

| | Revision 1 | Revision 2 | Driver |
|---|---|---|---|
| Dynamic targets | up to 3 | **1**, `N_MAX_TARGETS` configurable | S1 |
| Encounter classes | 6 | **5** — crossing collapsed | S3, S4 |
| Observation | 91 dims | **56 dims** | S6 |
| Target branch | 3 × 16 + 3 mask bits | **16, incl. a presence bit** | 01 §6.1 |
| Architecture | shared encoder + DeepSets/attention flag | **plain concatenation** | D3 superseded |
| Ship domain | `TODO`, compressed | **2.0 / 1.0 / 0.75 · Lpp**, provisional | 01 §5.2 |
| Workspace | `TODO(O4)` | **basin-matched, 10 m max** | O4 resolved |
| Boundary gating | `TODO(O5)` | **software gating confirmed** | O5 resolved |

Three previously-open items closed, and their resolutions are now load-bearing
rather than placeholders:

- **O4** — simulation matches the basin, so every swept width is physically
  reproducible. `CORRIDOR_WIDTHS_M` is now a real sweep, not a guess.
- **O5** — software gating, *and* the walls must stay visible to the localiser.
  `env._perceive` is explicitly ordered: localise on the full scan, gate
  afterwards for the tracker only.
- **O6** — no external instrumentation; scan-to-map registration instead. This
  is why the static/dynamic threshold is now documented as a property of
  localisation quality rather than of obstacle behaviour.

### 5.2 Rule 9(b) replaces Rule 18, and that collapses a class

The single most consequential code change. Revision 1 split crossing into
give-way and stand-on classes, following Meyer et al.'s use of Rule 18 — the own
ship is always give-way because it is significantly smaller than the vessels it
meets. **That premise fails here**: own ship and target are similarly sized
model vessels, so the asymmetry Rule 18 requires does not exist, and claiming it
in simulation and then validating against an identical vessel is an
inconsistency a reviewer would find.

Rule 9(b) gives the same conclusion on a premise that holds: a vessel under 20 m
shall not impede a vessel that can safely navigate only within a narrow channel.
The own ship gives way from either side, so the approach side is not a different
obligation and the observation does not carry it.

The geometry is **not** discarded — `encounter.crossing_side()` returns
`"port" | "starboard" | "none"`, and `ObservationBuilder.crossing_sides` exposes
it per track, because 02's passing-side reward term needs it. It is computed and
published, just not observed.

### 5.3 New in the code

| Module | Added |
|---|---|
| `constants.py` | `CORRIDOR_WIDTHS_M` + `widths_in_breadths()`; resolved ship domain; five-class list; Study 2 axes; `EGO_*_NOISE`; `TARGET_*` spawn config; `N_MAX_TARGETS` |
| `asv_lidar.py` | aft self-occlusion mask, per-beam dropout |
| `tracking.py` | detection dropout, velocity-estimate noise, `max_coast` (occlusion duration), `dropped_detections` |
| `encounter.py` | `crossing_side()`, `classification_latency()` |
| `observation.py` | presence bit replacing the mask vector, `crossing_sides` |
| `env.py` | `corridor_width` (Study 1), `_measured_ego` (no-IMU noise), `_sample_target` placeholder spawn, perception metrics |
| `metrics.py` | lateral clearance now measured against the **corridor**, not the basin |
| `rollout.py`, `train.py` | perception metrics in the log schema, `track_uptime` |
| `play.py` | **new** — manual and random driving, named encounter geometries, Study 2 knobs, headless smoke test |
| `render.py` | event pump (the window was never pumping, so the OS reported it unresponsive); `overlay` lines; fixed a hardcoded status offset that only worked at a 25 m basin height |

### 5.4 F15 — two metric bugs the width sweep exposed

Both were found by running the harness across the sweep rather than by a test,
which is worth noting: neither would have shown up at the default width.

**`track_uptime` exceeded 1.0.** The environment counted "any dynamic track
exists" as the target being tracked. A static panel promoted to a dynamic track
by pose drift therefore incremented the tracked counter without incrementing the
visible one. Fixed by attributing tracking to the target by proximity
(`TARGET_MATCH_RADIUS`), so a false positive can no longer inflate the headline
detection statistic for N1.

**`acquisition_range` came back NaN.** It is NaN until first acquisition, and a
plain `max` over the per-step series propagated that for the whole episode.
Fixed with a first-finite aggregation.

**And one that was silently wrong rather than visibly broken:**
`metrics.lateral_border_clearance` measured against `env.map_width`, i.e. the
basin. Once Study 1 narrows the corridor, that reports clearance to a wall the
vessel is not constrained by — every narrow-channel episode would have looked
safer than it was. Now measured against `corridor_bounds_x()`.

### 5.5 F16 — the head-on threshold arithmetic is now a test, not a comment

03 §5 puts the minimum width for a compliant port-to-port head-on at ≈3.66 m
(7.3 B) and asks for the arithmetic to be verified once the ship domain is
finalised. It is encoded as
`tests/test_cpa_cri.py::test_the_width_sweep_brackets_the_head_on_threshold`,
which derives the threshold from `DOMAIN_LATERAL` and
`HEAD_ON_WALL_CLEARANCE` and asserts the sweep brackets it between 4.0 m and
3.5 m.

When 05 replaces the provisional domain with values from the turning circle,
**that test will fail if the sweep levels are not moved with it** — which is the
intended behaviour, because the transition width is itself a reported result.

### 5.6 What is still not built, and whose it is

| | Owner |
|---|---|
| The whole reward, incl. the five COLREGs terms and the scale audit | 02 |
| **The Rule 9 precedence table** — blocking for 01 and 04 | 02 |
| Corridor generator: variable width *along* the path, bends, off-centre paths | 03 |
| Real encounter geometry — `_sample_target` is a head-on placeholder | 03 |
| Reactive and non-compliant target strata | 03 |
| Scenario suite, "Around the Clock", the Study 1/2/3 sweeps | 04 |

Two consequences worth stating plainly:

**The boundary branch still carries no information.** The corridor is a straight
inset rectangle, so port and starboard clearances remain affine functions of
`e_y`. `corridor_width` makes the *width* sweepable, which is what Study 1
needs, but the branch does not earn its place until 03 delivers variable width
along the path, bends, or deliberately off-centre paths. Do not ablate it before
then — a null result would be an artefact of the geometry, not a finding.

**The environment is still not trainable**, by design. `_reward` is sparse
terminal payoff only. A flat learning curve against it is expected, not a
perception bug.

### 5.7 F17 — `02a_REWARD_SPECIFICATION.md` conflicts with the code on two conventions

`planning/02a_REWARD_SPECIFICATION.md` appeared alongside the Revision 2 planning
set. It is 02's deliverable and the reward is not built here, but its §1
conventions table is binding on quantities 01 already produces, and **two of them
disagree with what the code does**. Both verified empirically, not read off.

**1. Cross-track error sign — a genuine conflict.**

| Source | Convention |
|---|---|
| `02a` §1 | `e_y` positive when the OS is to **starboard** of the path |
| `path.py` (Paper 2, Bucket A verbatim) | positive when to **port** |

Measured: a vessel 1 m to starboard of a due-north path reports `e_y = −1.01`.

This is not an oversight in either document. Paper 2 knew its convention was
non-standard — `static_obstacles/src/verify_los_apf.py` opens by noting that
"the environment defines positive cross-track error as *port* of the path, while
the textbook LOS law assumes starboard", and exists partly to check the sign
handling that follows from it. 02a has adopted the textbook convention.

It matters because 02's passing-side and wrong-side terms are conditioned on the
sign of `e_y`, and because `path[0]` is a frozen observation index.

**Recommendation: flip to 02a's convention**, i.e. adopt textbook LOS. It is the
convention the reward is being specified against, the one the classical
comparators use natively, and the one a reader will assume. The cost is
bounded — `path.py` stops being a verbatim Bucket A carry-over, `obs-v2` needs a
version bump, and the sign flip has to be propagated through `obstacles.py`'s
lateral-offset conventions. Paper 2's frozen baseline is unaffected, because it
runs entirely inside `static_obstacles/` and is never imported.

**Not flipped here.** It changes a frozen observation contract on the strength of
a document belonging to a task that has not been kicked off, and the alternative
— 02a adopting the existing convention — is equally consistent. It needs one
decision, not a guess.

**2. Reference speed.**

| Source | Value |
|---|---|
| `02a` §1 | `U_ref = 0.8 m/s` |
| `constants.py` | `U_CRUISE = 0.55 m/s` (`TODO(05)`) |

`SPEED_SCALE = 2 · U_CRUISE = 1.10 m/s` normalises the `ego` branch and both
target speed features, so this is not cosmetic: at `U_ref = 0.8` the scale should
be 1.60 and every speed feature currently reads ~45% high. Both values are
provisional pending 05, but they should be provisional to the *same* number.

**3. What agrees, checked rather than assumed.**

- `r > 0` is a starboard turn — confirmed: positive rudder gives `r = +5.99°/s`
  and a heading rotating +y toward +x.
- `α ∈ (0, 180)` is starboard, `CT = (ψ_TS − ψ_OS) mod 360` — both match
  `cpa_cri.py`.
- The ship-domain formula and `d_req = 2 · d_abeam = 2.36 m` match exactly.
- 02a §3.2 keeps CRI in the observation but **removes it from the reward**, to
  decouple 02 from 01's unfinished CRI constants. No change needed here — the
  CRI feature stays at `target[9]` as specified.

**4. Layout divergence, for whoever starts 02.**

02a §10.2 proposes `colregs/classifier.py`, `colregs/geometry.py`,
`colregs/context.py`. The equivalents here are flat: `encounter.py` (classifier
plus hysteresis) and `cpa_cri.py` (geometry, domain, CRI). The `EncounterContext`
object 02a §10.1 asks for does not exist yet — `ObservationBuilder` currently
computes the class, the crossing side and the risk internally and exposes them
via `encounter_classes` and `crossing_sides`. Rebuilding that as a single context
object shared by observation, reward and metrics is the right move and is 02's to
make; the classifier itself is already a single definition, which is the property
01 §5.3 actually requires.

### 5.8 Quick start

`src/play.py` drives the environment by hand or with random actions.
`python src/env.py` delegates to it, so either entry point works.

```
python src/play.py                                  # manual control
python src/play.py --mode random                    # random actions
python src/play.py --mode random --no-render --episodes 20    # smoke test
python src/play.py --target head_on --corridor-width 4.0
python src/play.py --target being_overtaken --aft-mask 45
```

Manual control is a **held helm** rather than a spring-centred one — arrow keys
move the rudder and it stays where you put it, because holding a steady rate of
turn is the thing worth testing. Throttle starts at 0.0, which is cruise, so the
vessel makes way from step one.

Both modes validate every observation against `observation_space` — shape,
dtype, finiteness and range — and exit non-zero if anything fails. That makes
`--mode random --no-render --episodes N` a usable regression check as well as a
demo.

`--target` places one vessel in a named encounter geometry (`head_on`,
`crossing_stbd`, `crossing_port`, `overtaking`, `being_overtaken`), which is how
the classifier gets eyeballed. It is a convenience, not a scenario generator —
03 still owns that.

**Two things the quick start made visible immediately.**

*The aft mask is not a cosmetic constant.* Running
`--target being_overtaken --aft-mask 45` and comparing against `--aft-mask 0`:
the overtaking vessel goes from tracked on 108 of 152 steps to **never acquired
at all**. That is 01 §2.3's warning made concrete — train the tracker to see
astern when the real mount cannot and Rule 17 behaviour fails in the field for
reasons unrelated to the policy. Locked in as a test.

*The propulsion authority is visibly narrow.* At `RPM_STAGE = 1`, full throttle
down is 9.0 RPM against a 12.0 cruise. Driving it by hand, slackening speed is
barely available as an avoidance action — which is exactly what 02 §4.4 flags,
since Rule 8(e) makes it a lawful manoeuvre and in a narrow channel often the
only admissible one.

### 5.9 Test suite

**231 tests, all passing**, in ~11 s.

| File | Tests | Covers |
|---|---|---|
| `test_lidar_pooling.py` | 17 | sector allocation, Algorithm 1 |
| `test_boundary_raycast.py` | 18 | hand-checked ranges, bends, gating, pose noise |
| `test_tracking.py` | 29 | clustering, Kalman, motion class, Study 2 axes |
| `test_cpa_cri.py` | 21 | CPA, ship domain, CRI, width-threshold arithmetic |
| `test_encounter.py` | 22 | five classes, crossing collapse, side recovery, hysteresis |
| `test_observation.py` | 29 | 56-dim contract, presence gating, slot persistence |
| `test_env.py` | 39 | end-to-end, corridor sweep, degradation axes, SB3 |
| `test_play.py` | 25 | helm behaviour, named geometries, contract check |
| `test_ported_harness.py` | 31 | the Paper 2 harness against the new schema |

Acceptance checks all hold: nothing modified under `static_obstacles/`, no import
reaches into it, `classify` is defined exactly once, and every unresolved
constant carries a `TODO` in `constants.py` and nowhere else.
