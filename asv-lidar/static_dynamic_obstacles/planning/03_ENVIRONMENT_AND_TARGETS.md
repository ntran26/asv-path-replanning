# 03 — Environment and Dynamic Target

**Revision 2** — one dynamic target, `N_max` configurable.
**Handover target:** Claude Code
**Depends on:** 01 (tracker and LiDAR interface), 02 (precedence table for width thresholds)
**Consumed by:** 04 (target behaviour models drive the scenario generator)

---

## 1. Purpose

Extend the Gymnasium environment from static-only to static obstacles **plus one dynamic
target vessel**, with corridor geometry rich enough to justify the boundary branch and to
support the channel-width sweep (Study 1).

**Paper 2 environment (baseline):** 10 × 25 m bounded workspace, 20 m reference path,
0–4 static obstacles, 700-step horizon, 0.1 s control period, Bluefin model (64.55 kg,
LOA 1.73 m, LBP 1.57 m, breadth 0.50 m, draft 0.19 m, Iz 10.45 kg·m²).

**Existing Paper 3 work:** `dynamic_obstacles/rl_env_dynamic.py` (~22 KB) already exists in
the repository. **Start by reconciling it against this document rather than building from
scratch.**

---

## 2. Target behaviour models

Behaviour, not presence, is what determines learning. **Confirmed (D1): constant velocity
for training; reactive and non-compliant for evaluation only.**

### 2.1 Constant velocity — training default

Deterministic geometry, reproducible encounters, and consistent with the maritime DRL
literature. Training against a reactive opponent makes the environment non-stationary and
destroys attribution — you cannot tell whether a behaviour change came from the policy or
the opponent.

### 2.2 Compliant reactive — evaluation stratum

**Reuse the encounter-specific VO controller from `PAPER3_BASELINES.md`** (Thyri &
Breivik, 2022). One implementation serves as both comparator and target behaviour policy.

Report as a generalisation stratum, not headline. **Fairness requirement:** run the
classical baselines against reactive targets too.

### 2.3 Non-compliant — evaluation stratum

Retained in Revision 2 even though the Rule 17 release term is gone, because it now tests
something different: whether the policy degrades gracefully when the target violates the
rules it was trained to expect. Implement at least a target that holds course when it
should give way, and one that alters to port in a head-on.

This stratum also feeds the "cases the proposed method also fails" requirement in 04.

---

## 3. Dynamic target implementation

Four details that are cheap now and expensive later.

**Oriented hull, not a circle.** Model the target with a heading and a hull polygon.
Required for ship-domain metrics and for correct aspect-angle computation in the encounter
classifier.

**Ray-cast against the moving polygon.** The target must be genuinely *perceived* through
the simulated LiDAR, not injected as ground truth. Otherwise the tracker gets a free
perfect detection and the N1 claim collapses — the whole contribution is that target state
comes from the sensor.

**Spawn outside LiDAR range.** The target appears beyond `D_max` and is acquired as it
approaches, so detection is part of the task. Track acquisition range is a reported metric.

**Allow occlusion.** A target behind a static obstacle disappears from the scan. Real in a
channel, and the case that would justify recurrence in 01. Log occlusion duration — it is a
Study 2 axis.

The target must respect the channel; no passing through walls.

---

## 4. Corridor geometry

**Requirement inherited from 01 §3.3 and now load-bearing for Study 1.**

The environment must support:

- **Variable channel width** along the path, and across episodes
- **Bends**
- **Deliberately off-centre reference paths**

Without these the boundary branch is redundant (port and starboard clearance become affine
functions of `e_y`), and the width sweep has nothing to sweep.

Width is parameterised in **ship breadths**, not metres, so the sweep and the precedence
thresholds are scale-explicit.

---

## 4a. Suspended static obstacles

Field obstacles are black panels suspended from lines spanning the basin. **Confirmed from
field video: they hang stably and do not swing appreciably while the vessel manoeuvres.**
Treating them as rigidly static in simulation is therefore correct, and no swing
randomisation axis is needed.

What does remain is a **jitter threshold** in the static/dynamic classifier (01 §4), and the
dominant source of jitter is not the obstacle:

- **Ego-pose error dominates.** When the vessel's own position estimate drifts, *every*
  static object in the scan acquires the same apparent velocity. The threshold is therefore
  a property of localisation quality rather than of the obstacles, and it should be set from
  measured pose noise (05 §4) rather than from a nominal value — retightening it as
  scan-to-map registration improves
- A false promotion of a static panel to a target ship is a false positive with COLREGs
  consequences, so bias the threshold and hysteresis toward under-detection of motion
- **Suspension lines** run diagonally across the basin and descend toward their anchors, so
  near the pool edges they may pass through the scan plane. A taut rope returns on one or two
  beams. Set the clustering minimum-points threshold to reject them rather than track them

---

## 5. Workspace dimensions (O4 — RESOLVED)

**Simulation matches the basin.** Maximum corridor width 10 m (20 breadths at B = 0.50 m).
Every simulated width is therefore physically reproducible — a meaningful strengthening of
the field-validation argument.

The unconfined reference case is supplied instead by the **open-water "Around the Clock"
variant** (04 §4.1), which is unbounded by construction. Study 1 then sweeps degrees of
confinement, with a published benchmark anchoring the open-water end. Arguably cleaner than
one continuous sweep, since the wide end is not a case you defined.

With the compressed ship domain from 01 §5.2, the sweep brackets the transition:

| Width | Breadths | Compliant head-on fits? |
|---|---|---|
| 10 m | 20 B | Yes, comfortably |
| 8 m | 16 B | Yes |
| 6 m | 12 B | Yes |
| 5 m | 10 B | Marginal |
| 4 m | 8 B | Tight |
| 3.5 m | 7 B | **No** — below threshold |

Minimum width for a compliant port-to-port head-on is ≈3.66 m (7.3 B): 2.36 m centre-to-centre
lateral separation for non-overlapping domains, plus ≈0.65 m wall clearance each side.
Six levels with the transition bracketed between 4 m and 3.5 m. **Verify this arithmetic once
the ship domain is finalised from the turning-circle data in 05** — the threshold moves with
the domain.

**Related requirement — Froude scaling.** State the model-to-full-scale relationship. A
reviewer will ask whether a 15 s TCPA on a 1.73 m model means anything at full scale, and
the answer belongs in the paper rather than in the rebuttal.

---

## 6. Action space

Structure unchanged (continuous rudder + propulsion). **Propulsion authority widens — resolved
(02 §4.4).** Rule 8(e) speed reduction is now the designated fallback whenever a compliant
course alteration would push the vessel into the boundary, so the agent must be able to slow
substantially and ideally stop.

**Verify reverse thrust capability.** If the vessel can only reduce forward thrust, "take all
way off" is unavailable; model what the platform can actually do and state the limitation.

Rudder saturation and rate limiting stay matched to the real actuator (Paper 2 Table 2).
Revisit once 05 delivers a calibrated actuator model.

---

## 7. Noise and domain randomisation hooks

Build as environment parameters from the start, populated by 05 and swept by Study 2:

| Source | Injected into |
|---|---|
| Pose / odometry drift | boundary raycast, tracker ego-motion compensation |
| LiDAR angular resolution and range noise | raw scan |
| Aft self-occlusion sector | raw scan (masked bearing range) |
| Scan motion distortion | tracker velocity estimates |
| Detection dropout rate | tracker input |
| Ego velocity estimate error (u, v, r differentiated from pose) | `ego` observation branch, and apparent motion of all static objects |
| Vessel model parameters ± identification CI | dynamics |
| Actuator lag and delay | actuator model |

First-class config rather than hardcoded values is what turns "we identified the model"
into "we identified the model and randomised within identification uncertainty" — and it is
what makes Studies 2 and 3 possible at all.

---

## 8. Open items

- Reconcile `dynamic_obstacles/rl_env_dynamic.py` against this specification
- Decide propulsion authority (coupled to 02 §4.4)
- Define the target hull polygon and ship domain at model scale (coupled to 01 §5.2)
- Speed threshold and hysteresis for static/dynamic classification
- Verify the episode horizon: 700 steps at 0.1 s = 70 s. Confirm this is long enough for
  acquisition, classification, manoeuvre and clearing in the longest corridor
