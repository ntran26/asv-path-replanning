# Hard-arrangement evaluation

Re-evaluation of every controller on layouts with the **same obstacle counts** as
the standard evaluation set but **harder arrangements**. Kept separate from
`manuscript_results/` because these are not the paper's headline results.

**Result in one line: the classical baseline is essentially unaffected by harder
arrangements, PPO degrades moderately, and SAC degrades severely — reversing the
ordering seen on the standard evaluation set.**

---

## 1. What "harder arrangement" means here

Obstacle count is unchanged — these sets use counts **2, 3 and 4**, exactly the
counts the frozen 500 already contains, so each set pairs against a same-count
subset of the standard evaluation. Only the geometry differs.

Two independent difficulty levers were applied:

**A detour floor.** Every accepted layout has a longer forced detour
(`route_ratio_astar`, A* route ÷ straight line) than **any** same-count layout in
the standard set:

| Count | Standard set (p50 / max) | Hard set (p50 / max) | Floor | Acceptance |
|---|---|---|---|---|
| 2 | 1.041 / 1.065 | 1.083 / 1.088 | 1.065 | 0.5 % |
| 3 | 1.050 / 1.072 | 1.083 / 1.104 | 1.072 | 5.9 % |
| 4 | 1.062 / 1.083 | 1.093 / 1.145 | 1.083 | 3.9 % |

The 2-obstacle set required **20 605 attempts** to find 100 qualifying layouts,
which is itself a measure of how atypical they are.

**Tighter geometry**, applied to the sampler during generation only:

| Parameter | Standard | Hard | Effect |
|---|---|---|---|
| `GATE_GAP_RANGE` | (1.35, 2.25) | (1.00, 1.50) | gap against a 0.80 m hull — 1.00 m leaves 0.10 m each side |
| `OBSTACLE_CENTER_PROB` | 0.30 | 0.75 | obstacles on the path, not beside it |
| `OBSTACLE_LATERAL_OFFSET` | 0.25–0.95 | 0.0–0.45 | blocking rather than passable |
| `TARGET_SIDE_CORRIDOR_OFFSET` | (0.65, 1.05) | (0.50, 0.80) | narrower escape corridor |
| `OBSTACLE_PATH_FRAC` | 0.25–0.70 | 0.20–0.80 | manoeuvres chain, less recovery runway |
| scenario mix | `offpath` 0.05 | `offpath` 0.0, `gate` 0.30 | no pure distractors |

**Why both levers.** `route_ratio` comes from an inflated-grid A* with a *point*
robot, so it measures how far around the vessel must go — not how precisely it
must steer. A narrow gate has a small detour but demands exact threading. The
geometry overrides raise difficulty along that second axis; the floor handles the
first.

**Fairness.** No controller trained or tuned on these layouts, and **none was
re-tuned for them** — same checkpoints, same LOS+APF parameter sets as the
headline results. The shift is equally novel for every method: the learned
policies trained on standard-difficulty arrangements, and the LOS+APF parameters
were selected on them.

The overrides were applied to `config` for generation only and asserted restored
afterwards. Layouts serialise as explicit obstacle polygons, so evaluation
replays the geometry with no config dependency. `env.py` and the reward are
untouched.

---

## 2. Results — success rate (100 episodes per cell)

| Method | 2 obstacles | 3 obstacles | 4 obstacles |
|---|---|---|---|
| SAC (deployed) | 0.700 | 0.480 | 0.330 |
| SAC (3 seeds) | 0.700 (0.68–0.72) | 0.763 (0.72–0.82) | 0.680 (0.60–0.74) |
| PPO (3 seeds) | 0.870 (0.83–0.92) | 0.837 (0.72–0.94) | 0.753 (0.74–0.76) |
| **LOS+APF (3 searches)** | **1.000 (1.00–1.00)** | **0.977 (0.95–0.99)** | **0.873 (0.84–0.91)** |

### Degradation against the standard set at matched obstacle count

| Method | 2 obs | 3 obs | 4 obs | mean |
|---|---|---|---|---|
| SAC (deployed) | −0.250 | −0.470 | −0.560 | **−0.427** |
| SAC (3 seeds) | −0.223 | −0.140 | −0.180 | −0.181 |
| PPO (3 seeds) | −0.043 | −0.050 | −0.110 | −0.068 |
| LOS+APF (3 searches) | +0.047 | +0.064 | −0.060 | **+0.017** |

**The classical controller does not degrade at all on average.** It gains on the
2- and 3-obstacle sets and loses only 6 points at 4.

Paired test, hardest set (4 obstacles), deployed SAC against LOS+APF s1:
success 0.330 vs 0.870, **McNemar p = 6.3e−14**, 57 episodes where only the
classical controller succeeded against 3 the other way.

Full metrics — collisions, RMS CTE, clearances, rudder rate, completion time —
in `tables/hard_arrangement_table.csv`.

---

## 3. Interpretation

**Why the classical controller holds up.** These arrangements are dominated by
narrow gates and centred obstacles. A potential field handles a symmetric gap
natively: two repulsive sources balance and centre the vessel between them, and
the behaviour is unchanged whether the gap is 2.2 m or 1.0 m. There is no
distribution to be outside of.

**Why the learned policies do not.** They never observed gaps this tight or
obstacles this consistently centred. The deployed policy's failure mode is
consistent with its measured behaviour on the standard set — it passes obstacles
at 0.233 m, the closest of any method, which is a viable strategy at standard
spacing and stops being viable when the gap approaches the hull width.

**The deployed policy degrades far more than the retrained seeds** (−0.427 vs
−0.181), despite being *better* on the standard set. That is consistent with it
being the most aggressively tuned to the standard distribution.

---

## 4. What this does and does not mean for the manuscript

It does **not** contradict any published result. Every claim in the paper
concerns the standard evaluation distribution, and within it SAC reproduces at
95.0 % and holds statistical parity with the classical baseline.

It **does** bound the generalisation claim. Performance is not robust to
arrangement difficulty at fixed obstacle count, and the manuscript's sentence
about generalising "across different obstacle densities" should be scoped to the
evaluated distribution — as already recommended in
`manuscript_results/MANUSCRIPT_BRIEF.md` §4.2.

**These results are not in the manuscript package.** Whether to report them is a
judgement call: reviewers did not request a robustness study, and there is no
obligation to publish every exploratory analysis. But if any robustness or
generalisation claim is made, this data is directly relevant and should not sit
unmentioned in a drawer. The one thing that would be indefensible is claiming
robustness to harder layouts while holding this result.

---

## 5. Files

```
tables/hard_arrangement_table.csv     all metrics, all methods, all three sets
tables/compare_obs4_sac_vs_losapf.json  paired McNemar/Wilcoxon, hardest set
per_episode/<set>__<run>/             episodes.csv + summary.json, 30 runs
layouts/hard_arr_obs{2,3,4}.json      the 300 layouts, with route ratios
```

Per-episode CSVs use the standard schema and join on `episode_id`. Layout ids
occupy the 4,000,000+ block and are disjoint from every other set.

Regenerate:

```bash
python src/build_hard_layouts.py
python src/evaluate.py --controller "los_apf:eval_results/baselines/los_apf_best.json" --layouts eval_layouts/hard_arr_obs4.json --tag hard_arr_obs4__los_apf_s1 --workers 6 --deterministic-controller --out-root eval_results/hard_arrangement
```
