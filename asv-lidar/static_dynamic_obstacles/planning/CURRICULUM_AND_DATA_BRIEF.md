# Curriculum, scenario generation, and the three data sets — a visualisation brief

**Purpose.** A self-contained description of how scenarios are made, how the
curriculum orders them, and how the training / development / frozen sets are kept
disjoint — written so it can be turned into an explanatory artifact with
equations and animations. Every number is read from the code as of 2026-09-25
(`src/scenario.py`, `src/suite.py`, `src/constants.py`, `src/train_formulation.py`).

Companion documents: `planning/METHODS_BRIEF.md` (the method as frozen),
`configs/baseline_v2.json` (every constant), `PROJECT_STATE.md` (findings).

---

## 1. The one idea to get across

**There is no dataset.** Scenarios are *generated* from seeds, not collected, so
a "data set" here is a **seed range plus a sampling rule**. Everything else
follows: reproducibility is exact, the three sets can be proved disjoint, and
the suite can be regenerated from a hash years later.

    scenario = G(seed, stage, class, geometry)

`G` is deterministic. The same seed always yields the same basin, path,
obstacles, target spawn and target course.

**Animation 1 — "the generator, not the dataset".** A seed integer drops into a
funnel labelled `G`; out falls a complete 10 × 25 m scene. Change the seed, a
different scene falls out. Repeat the same seed, the identical scene falls out.

---

## 2. Seed namespaces: how the sets are kept apart

Five disjoint integer ranges (`constants.SEED_NAMESPACES`):

| Namespace | Range | Used for |
|---|---|---|
| `training` | 0 – 99,999 | every training episode |
| `development` | 200,000 – 209,999 | the 120-episode development set ("validation") |
| `frozen_eval` | 300,000 – 309,999 | the frozen suite ("test") |
| `study1` | 400,000 – 409,999 | the channel-width sweep (R4) |
| `study2` | 500,000 – 509,999 | perception degradation (R5) |

$$\text{seed}(\text{ns}, i) = \mathrm{lo}_{\text{ns}} + \big(i \bmod (\mathrm{hi}_{\text{ns}} - \mathrm{lo}_{\text{ns}} + 1)\big)$$

A test asserts the ranges never overlap. This is the machine-checkable version
of "we did not train on the test set".

**Animation 2 — the number line.** Show 0 … 500,000 as a line with five coloured
bands. Sample dots fall into the training band during a training montage; when
evaluation starts, dots fall only in the development band; when the paper's
tables are produced, only in the frozen band. No dot ever crosses a boundary.

---

## 3. Generating one scenario

### 3.1 Geometry: basin or channel

With probability `p_basin` (per stage, §4) the scene is the **basin** — the
10 × 25 m field site — otherwise a **channel** of sampled width.

Basin: start and goal have fixed along-basin positions and sampled lateral ones,

$$y_{\text{start}} = 2\,\text{m},\quad y_{\text{goal}} = 22\,\text{m},\quad x_{\text{start}}, x_{\text{goal}} \sim \mathcal{U}(2.5, 7.5)\,\text{m}$$

so the reference path is a straight but **slanted** leg, and the navigable
polygon is the basin inset by 0.40 m. Channel: parallel walls at a width drawn
from the stage's range, with the path along the centreline.

**Animation 3 — slanted legs.** Draw the basin rectangle; sample start and goal
dots on the two horizontal lines; connect them. Repeat quickly to show the fan
of legs the policy must follow, and how the *side clearances differ* left and
right — which is why cross-track error is normalised by the local half-width.

### 3.2 The encounter, solved backwards

The generator does **not** place the target and hope an encounter happens. It
samples the encounter's *outcome* and solves backwards for the spawn. For a
class with own speed $U_{OS}$, sample

$$CT \sim p_{\text{class}}(\cdot),\qquad k = \tfrac{U_{TS}}{U_{OS}} \sim \mathcal{U}(k_{\min}, k_{\max}),\qquad T_0 \sim \mathcal{U}(T_{\min}, T_{\max}),\qquad d_0 \sim \mathcal{U}(0, d_{\max})$$

where $CT$ is the crossing angle, $T_0$ the time to CPA and $d_0$ the distance at
CPA. With unit heading vectors $\hat h_{OS}, \hat h_{TS}$, relative velocity and
its unit normal,

$$\vec v_{\text{rel}} = U_{OS}\hat h_{OS} - U_{TS}\hat h_{TS},\qquad \hat n = \pm\frac{(v_{\text{rel},y},\, -v_{\text{rel},x})}{\lVert \vec v_{\text{rel}} \rVert}$$

the CPA point and the spawn follow:

$$p_{\text{CPA}} = p_{OS}(0) + U_{OS} T_0 \hat h_{OS} + d_0\,\hat n, \qquad p_{TS}(0) = p_{\text{CPA}} - U_{TS} T_0 \hat h_{TS}$$

**Why sample $d_0$ explicitly?** Solving backwards *without* it puts the target
on the own ship's projected track every time, so the required alteration is
always maximal and the policy learns "always alter" instead of "when to alter".
The zero branch of the Rule 8 term would never fire.

Per-class ranges as coded:

| Class | $CT$ | $k = U_{TS}/U_{OS}$ | $T_0$ (s) | $d_{\max}$ (m) |
|---|---|---|---|---|
| head-on | reciprocal sector | 0.7 – 1.3 | 12.3 – 17.0 | 2.0 |
| crossing | ±(22.5°, 112.5°) | 0.6 – 1.4 | 7.9 – 14.8 | 2.5 |
| overtaking | narrow astern sector | 0.4 – 0.55 | 19.7 – 31.5 | 2.0 |
| being overtaken | mirror of overtaking | 1.5 – 2.2 | 9.9 – 15.8 | 2.0, floored (§3.3) |
| null | any | 0.85 – 1.15 | — | ≥ 4.0 (no encounter) |

**Animation 4 — the backward solve.** Place the own ship; draw its track; mark
the CPA point at distance $d_0$ perpendicular to the relative velocity at time
$T_0$; then run the target *backwards* along its heading to its spawn. Then play
it forward and watch the two hulls arrive at the CPA exactly as constructed.
This is the single best animation in the brief: it makes the geometry obvious.

### 3.3 Two constraints on the draw

**Contact-free floor (being overtaken).** An overtaker that never gives way will
hit the own ship if the drawn CPA is smaller than the hulls allow, so

$$d_0 \ge d_{\text{contact-free}}(CT, k) + 0.35\,\text{m}$$

and since baseline-v2 **no draw sits below that floor** — the Rule 17(b)
"last-moment" case is out of scope.

**Escapability (crossings).** 20 % of crossings are labelled *unescapable* —
kept and reported, not silently dropped.

### 3.4 Feasibility: every layout admits a route

Obstacles (0–3 squares of 1.0 m) are placed, then the whole layout is checked by
**A\*** on a 0.25 m grid with walls inflated 0.40 m and obstacles 0.45 m. A
layout is accepted only if a route exists no longer than 2.25 × the straight leg:

$$L_{A^*} \le 2.25\,L_{\text{leg}}$$

Up to 20 redraws, then obstacles are thinned. Rejections are counted in a
*rejection ledger*, so "hard" is provable and "impossible" is excluded.

**Animation 5 — the feasibility filter.** Show a candidate layout; flood-fill
the A\* search; if the route exceeds the ratio, flash the layout red and redraw.
A counter ticks up the rejections. It makes the point that difficulty is
*designed*, not accidental.

---

## 4. The curriculum

Five stages, switched on **fraction of the training budget**, not on performance:

| Stage | From | Classes | Obstacles | Channel width (m) | `p_basin` | TCPA draw |
|---|---|---|---|---|---|---|
| 1 | 0 % | no target | 0–1 | 8–10 | 1.00 | full |
| 2 | 8 % | no target | 0–3 | 5–10 | 1.00 | full |
| 3 | 18 % | head-on, crossing, null, no-target | 0–1 | 7–10 | 0.85 | upper half |
| 4 | 32 % | all six | 0–2 | 4.5–10 | 0.75 | full |
| 5 | 50 % | all six | 0–3 | 3.5–10 | 0.75 | full |

With a 2 M-step budget the switches fall at 0, 160 k, 360 k, 640 k and 1.0 M steps.

Two details worth showing:
- **Stage 3 introduces encounters with the *easy half* of the TCPA range** (more
  time to react), then stage 4 opens the full range.
- **Stage 3 teaches head-on and crossing together**, deliberately: teaching
  head-on alone first taught "give way = turn starboard", which then had to be
  unlearned for port crossings.

Within a stage the class is drawn from fixed shares:

$$P(\text{class}) = \{\text{head-on } 0.20,\ \text{crossing } 0.22,\ \text{overtaking } 0.16,\ \text{being overtaken } 0.14,\ \text{null } 0.11,\ \text{no-target } 0.17\}$$

**Animation 6 — the curriculum ladder.** A progress bar across 2 M steps with
five coloured segments. As the bar fills, a small scene panel updates: empty
channel → clutter → first encounters → all classes → narrow water. A live pie
chart shows the class mixture switching on at stage 3.

**Measured side effect worth animating:** the step rate falls as the stages
advance — 17.3 steps/s in stage 1 down to 15.4 in stage 5 — because each added
target and obstacle costs raycasting and collision work.

---

## 5. The three sets

| | Training | Development ("validation") | Frozen suite ("test") |
|---|---|---|---|
| Seeds | `training` | `development` | `frozen_eval` |
| Size | unbounded — a fresh draw every episode | **120** (20 per class × 6) | **780** (39 cells × 20) |
| Sampling | curriculum stage of the moment | stage 5, fixed list | stratified cells |
| Used for | gradient updates | **checkpoint selection** and all diagnostics | the paper's tables |
| Seen how often | continuously | every 200 k steps | **once per policy** |

### 5.1 Development set — the selection instrument

120 fixed episodes, drawn once at stage 5. Every 200 k steps the current policy
is replayed over all of them, with the safety supervisor **off** and **on**, and
the checkpoint is kept if it improves

$$\text{score} = P(\text{goal}) - 2\,P(\text{collision})$$

That factor of 2 is the whole selection rule: a policy that reaches the goal by
taking risks scores worse than a cautious one.

**Animation 7 — best-on-dev selection.** Plot the 10 evaluation points of a real
run as a noisy curve; drop a marker each time a new best appears; show the final
"best" marker being copied to `best_model.zip` while later, worse points are
ignored.

### 5.2 Frozen suite — the evidence

**Tier B, suite 3.1**: 39 cells × 20 episodes = 780 per seed. A cell is

$$\text{cell} = (\text{geometry stratum}) \times (\text{encounter class}) \times (\text{target behaviour})$$

- **Strata (3):** basin; channel-wide 8.75–10 m; channel-intermediate 7.5–8.75 m.
  Channels stop at 7.5 m because below that a two-vessel encounter has no room a
  lawful manoeuvre can use.
- **Classes (5):** head-on, crossing, overtaking, being overtaken, null.
- **Behaviours (3):** `cv` constant velocity, `re` reactive, `nc` non-compliant.

Each run records a **manifest digest** — a hash over every case together with
the constants that generated them — so a table can be traced to the exact
scenarios.

**Animation 8 — the cell grid.** A 3 × 5 × 3 cube of cells, each lighting up
with 20 dots as episodes run, then colouring by success rate. Rotating the cube
to face "stratum" shows the headline finding: performance falls as the channel
narrows.

### 5.3 What is *not* in the paper

**Tier A** (38 named deterministic cases, 35 realisable) exists in the suite but
is out of this paper. **Around the Clock** (24 constellations at equally spaced
bearings) is defined but has no builder yet.

---

## 6. Related subjects worth including

### 6.1 Encounter classification (the sectors)

Class is decided against the **path tangent**, not the instantaneous heading,
with 3° hysteresis and a 2-step (0.8 s) persistence requirement. Sector edges at
**22.5°** (head-on) and **112.5°** (overtaking) are the COLREGs ones.

**Animation 9 — the rose.** A polar diagram around the own ship with the four
sectors shaded; a target orbits and the class label switches at the boundaries;
show the hysteresis band preventing flicker at an edge.

### 6.2 CPA geometry and risk

$$TCPA = -\frac{\vec p_{\text{rel}} \cdot \vec v_{\text{rel}}}{\lVert \vec v_{\text{rel}} \rVert^2}, \qquad DCPA = \lVert \vec p_{\text{rel}} + TCPA\,\vec v_{\text{rel}} \rVert$$

The collision-risk index is 1 inside the ship domain, otherwise the larger of an
exponential decay in DCPA/TCPA (with **different rates before and after** the
CPA) and a Euclidean-distance term.

**Animation 10 — the closing triangle.** Two vessels on converging courses; draw
the relative-velocity vector from the own ship, drop the perpendicular to the
target: its length is DCPA, its time is TCPA. Watch both update live.

### 6.3 The engagement latch

An encounter **engages** when

$$\text{class} \ne \text{none} \quad\wedge\quad 0 < TCPA \le 25\,\text{s} \quad\wedge\quad DCPA < 1.5\,d_{\text{req}},\qquad d_{\text{req}} = 2.5\,\text{m}$$

and then the class, the compliant turn direction, and the heading and speed at
that instant are **frozen** until the target is past and opening. Without the
latch, a policy could turn until the classifier changed its mind and then be
judged against the new class.

**Animation 11 — the latch.** A timeline with three lamps (idle → engaged →
clearing); at engagement, snapshot values freeze into a small panel that stays
fixed while the vessels manoeuvre.

### 6.4 Proximity gate

Every COLREGs penalty is weighted by

$$\rho = \operatorname{clip}\!\left(1 - \frac{DCPA}{1.5\,d_{\text{req}}},\, 0,\, 1\right)$$

so a rule violation far from a real conflict costs nearly nothing, and the same
violation at close quarters costs full price.

### 6.5 Width thresholds — why the strata are not round numbers

The widths at which each manoeuvre stops fitting are **derived** from hull beam,
required separation and wall clearance, not chosen:

$$W_{\text{head-on}} = 3.80\,\text{m}, \qquad W_{\text{overtaking}} = 4.26\,\text{m}, \qquad W_{\text{crossing}} = 7.60\,\text{m}$$

**Animation 12 — the shrinking channel.** Slide a width slider from 10 m down to
3.5 m with two vessels meeting; at each threshold, grey out the manoeuvre that
no longer fits, leaving only speed reduction at the end.

---

## 7. Numbers an artifact can quote safely

- Decision rate **2 Hz**, physics 0.1 s, episode cap **180 steps (90 s)**.
- Cruise **0.558 m/s**; vessel **1.725 m × 0.5 m**.
- Observation **70 values in 6 branches**; action = (rudder, throttle) ∈ [−1, 1]².
- Training: **10 parallel environments**, 2 M steps, **3 seeds** per learner.
- Learners compared: **PPO, RecurrentPPO, SAC, TQC** on the identical formulation.
- Development set **120**; frozen suite **780**; disjoint seed ranges.
- Measured example (SAC seed 0): development score 0.53 at 200 k → **0.92 at 2 M**.
