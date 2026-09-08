# RENDER_PANEL_SPEC — left telemetry panel for the Paper 3 simulator

**Purpose:** make one frame answer *why did the agent do that*, and make the four failure
modes this project has already hit visible without opening a log.

The panel is a **view on `info`**, never a separate computation. Every field below is a key
emitted by `env.step()` per `02a §10.3`. If the panel needs a number `info` does not carry,
add it to `info` — do not compute it in `render.py`. Same principle as the single
`EncounterContext`: two consumers, one source, or they diverge.

---

## 1. The one idea worth building first

**Show the gate, not just the value.**

When `v_side` reads `0.000` you cannot tell whether that is correct (wrong class for this
term) or a bug (gate stuck, `ρ_t` collapsed, class never latched). Every COLREGs sub-term
should print its value *and* the reason it holds that value. This is the difference between a
panel you glance at and a panel you debug with, and it costs one string per term.

Same idea applies to the speed gate: `g_u = 1.00 [SAT]` immediately says the gate is dead,
which is the exact shape of the F19 bug that pinned the `ego` surge feature at 1.0 for 45% of
a run.

---

## 2. Layout

Monospace, same treatment as the field render. ~52 lines. Blocks toggle with keys so the
panel fits a short window.

```
─── RUN ──────────────────────────────────────────────  [1]
seed=8842   scen=crossing_w06_c02   suite=eval-v3 (a91f2c)
stage=5     mode=policy(SAC-s3-1.2M)   step=220/700  t=22.0s
corridor W=6.0m (12 B)  W_local=6.00   targets=1  spawn=DISPLACED

─── EGO (truth) ─────────────────────────────────────  [2]
x=+2.14 y=+9.83  hdg=+352.1   u=+0.93  v=-0.02  r=+0.81 d/s
U_ref=0.55  U_ref_eff=0.55   g_u=1.00 [SAT]
r_path=+0.00 d/s   r-r_path=+0.81 (STBD)
act rud=+0.31 thr=+0.62 | cmd rudder=-24.8 rpm=12.00
Dact rud=+0.04 (rate limit 0.20)

─── PERCEPTION vs TRUTH ─────────────────────────────  [3]
                 est       true       err
 tgt range      3.412     3.380    +0.032
 tgt bearing   +38.4     +37.9      +0.5
 tgt speed      0.512     0.550    -0.038
 tgt heading  +214.1    +212.8      +1.3
 DCPA           0.94      1.02      -0.08
 TCPA           6.8       7.1       -0.3
 class      CROSSING  CROSSING        ok
 track: age=42 hits=40 misses=0 coast=0.0s  gate: 0 dropped

─── COLREGS ─────────────────────────────────────────  [4]
class=CROSSING(est)  state=ENGAGED  since t=18.8s (+3.2s)
compliant_sense=STBD    A_stbd=NO    A_port=yes
Dy_req=1.42  r_stbd=0.31  r_port=4.10   d_req=2.36
rho=0.73   A_req=0.60  A_t=0.21  urgency=0.55
 v_port  0.000  [turning stbd, compliant]
 v_bow   0.310  [ahead of TS beam, DCPA 0.40 dreq]
 v_side  0.000  [n/a for crossing]
 v_hold  0.000  [wrong class]
 v_r8    0.215  [A_t < A_req, 8(e) only: no stbd room]
 group   0.402  (pre-clip 0.402 / 1.0)

─── REWARD ──────────────────────────────────────────  [5]
                inst      xw      Sep    range(ep)
 path         -0.152  -0.091   -21.4   [-0.44,-0.01]
 progress     +0.983  +0.295   +58.2   [+0.00,+1.00]
 exist        -1.000  -0.050   -11.0   [flat]
 smooth       -0.031  -0.003    -0.9   [-0.19, 0.00]
 obstacle     -0.204  -0.449   -12.1   [-0.71, 0.00]
 boundary      0.000   0.000     0.0   [ 0.00, 0.00]
 domain       -0.118  -0.295    -2.4   [-0.30, 0.00]
 COLREGS      -0.402  -0.724   -18.7   [-0.72, 0.00]
 ───────────────────────────────────────────────────
 step total           -1.317      dominant: COLREGS
 episode                        -12.3  dominant: progress(+58)
 terminal              pending

─── OBS HEALTH ──────────────────────────────────────  [6]
 branch        dim   cur     min     max   clip%
 lidar c_t      27  0.31    0.00    1.00    12%
 boundary        7  0.42    0.19    0.71     0%
 ego             3  0.29    0.00    1.00    45% [!]
 path            3  0.51    0.02    0.98     0%
 target         16  0.44    0.00    1.00     3%

─── CLEARANCE ───────────────────────────────────────  [7]
 boundary 1.82   static obs 2.40   target hull 2.19
 domain margin +1.01   goal in 8.4 m   steps left 480
```

---

## 3. What each block is for

### [1] RUN — provenance
`seed`, scenario name and **suite hash** matter more here than they look. `04 §4.5` requires
the evaluation suite frozen and hashed; putting the hash on screen makes a screenshot
reproducible, which is worth a lot when you are comparing two policies on one geometry three
weeks apart.

`spawn=COMPLIANT|DISPLACED` is the head-on regime from `02a §2.4`. If every frame you ever
look at says `DISPLACED`, `TARGET_COMPLIANT_SPAWN_PROB` is not wired.

### [2] EGO — the speed gate, and `r_path`
`U_ref_eff` next to `U_ref` shows `R-2` firing. `g_u` with a `[SAT]` flag catches a dead gate.

`r_path` and `r − r_path` are `R-8`. On a straight corridor `r_path` is identically zero, so
this line is also the fastest way to confirm 03's bend geometry has actually landed — the day
it becomes non-zero, `R-8` is finally testable.

`Dact` against the rate limit shows whether `κ_δ` is calibrated: if `Dact` never approaches
the limit, `r_smooth` is inert.

### [3] PERCEPTION vs TRUTH — the N1 block
Under `R-1`, safety terms read truth and COLREGs gating reads the estimate. The panel must
show both or that decision is invisible.

**Highlight the class row in red whenever `est ≠ true`.** A misclassification is the failure
`04 §6` names as the one that matters — the agent turns the wrong way — and it is otherwise
almost impossible to spot in a replay. Also print a running count of mismatched steps this
episode.

`coast` is the tracker's time since last detection. Rising `coast` next to a stale class is
the occlusion failure mode.

### [4] COLREGS — the state machine and the admissibility predicate
The three numbers behind `A_stbd` (`Dy_req`, `r_stbd`, `r_port`) explain *why* it flipped,
which is what you actually need when the agent does something odd near a wall.

`A_req` is the health check on my deficit reformulation of Rule 8. **If `A_req` is 1.00 in
every head-on episode you look at, the spawn-DCPA bug is back** and the agent is learning
"always alter" instead of "when to alter".

`compliant_sense` printed explicitly is the guard against the `02 §4.2` trap. Seeing
`sense=PORT` on an overtaking encounter, next to a starboard turn and `v_port` rising, is the
whole bug in one frame.

`state=ENGAGED since` catches hysteresis failures — if the state flickers between IDLE and
ENGAGED, `ENCOUNTER_HOLD_STEPS` is too small and the Rule 8 accumulator is resetting.

### [5] REWARD — four columns, and the fourth is the important one
`inst` (pre-weight) / `xw` (post-weight) / `Sep` (episode integral) / **`range(ep)`**.

The range column is the direct detector for the Paper 2 scale bug. A term whose episode range
is `[-0.44, -0.41]` "varies by less than 10% of its own value" — it is a constant offset
wearing a shaping term's costume, and it is invisible in the other three columns. Print
`[flat]` when the range is below 5% of the term's declared span.

`dominant:` on both the step and episode lines tells you instantly when one term is eating the
signal, which is what happened with the unshifted obstacle exponential.

The panel should also **assert the coefficient ordering live** and flag it if the realised
per-step magnitudes invert — cheap insurance that the `02a §7` hierarchy survives contact.

### [6] OBS HEALTH — `clip%`
Fraction of episode steps at the normaliser's clip, per branch. This is the F19 detector: the
`ego` branch reading `45% [!]` is exactly the bug where the surge feature was pinned at 1.0
and carried no gradient. Flag any branch above ~10%.

Worth a per-dimension drill-down on a keypress, since a single saturating dimension inside a
27-dim branch will not move the branch aggregate much.

### [7] CLEARANCE — how close to which terminal
`domain margin` signed, so negative means intruding. Four numbers that say which termination
is nearest, which is the context you want when a run ends abruptly.

---

## 4. Priority

Build in this order. The first two are what makes the panel worth building now, during T4.

| Tier | Blocks | Why |
|---|---|---|
| **Essential** | [5] REWARD, [4] COLREGS | The reward is being built now and these are its instrument. Without them the scale audit is the only feedback, and it runs after the fact |
| **High** | [3] PERCEPTION, [2] EGO | N1's diagnostic, and the two gates most likely to be silently dead |
| **Useful** | [6] OBS HEALTH, [1] RUN | Catches saturation; makes screenshots reproducible |
| **Later** | [7] CLEARANCE, sparklines | Nice, not diagnostic |

---

## 5. Implementation notes

**Read from `info`, log the same struct.** Emit one CSV row per step containing exactly the
panel's fields. A screenshot and the analysis then agree by construction, and the panel
doubles as the schema for the per-episode record `metrics.py` already writes.

**Colour, sparingly.** The field render is near-monochrome and that is right. Four rules:
red for `est ≠ true` on the class row; red for `clip% > 10`; amber for `state=ENGAGED`; amber
for any COLREGs sub-term above 0.5. Nothing else.

**Toggles.** `1`–`7` show/hide blocks, matching the field render's `M`/`G`/`F` convention.
Default to `[4]` and `[5]` only, so the panel is readable in a short window and you opt into
the rest.

**Add a step-back key.** Hold the last ~200 steps of `info` in a ring buffer and let
`←`/`→` scrub. Nearly every question worth asking about a COLREGs encounter is "what was the
state four seconds ago", and re-running with a breakpoint to find out is the slow way.

**Sparklines, if you want them.** A 40 px polyline per reward term over the last 100 steps,
drawn to the right of the `Sep` column. Cheap in pygame, and a flat line is the same bug the
`range(ep)` column catches — belt and braces on the failure mode that cost Paper 2 the most.

---

## 6. Dropped from the field panel

`UDP local/server`, `RX lines/frames`, `shadow`, `seq`, `hdg_ref`, `S2`, `raw_rudder` — all
transport and hardware plumbing with no simulator analogue. The line budget they free goes to
[4] and [5].

Keep the sector-distance dump, but **as pooled `c_t` in observation units, not metres**. What
matters is what the policy actually receives after pooling and normalisation, not the raw
ranges. Put raw metres behind the drill-down key for when you suspect the pooling itself.
