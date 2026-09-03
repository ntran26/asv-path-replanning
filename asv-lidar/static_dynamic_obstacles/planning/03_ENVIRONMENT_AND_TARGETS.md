# 03 — Environment and Dynamic Targets

**Handover target:** Claude Code
**Depends on:** 01 (LiDAR and tracker interface)
**Consumed by:** 04 (target behaviour models are used by the scenario generator)

---

## 1. Purpose

Extend the Gymnasium environment from static obstacles to static **and** dynamic
targets, with channel geometry rich enough to justify the boundary observation branch.

**Paper 2 environment (baseline):** 10 × 25 m bounded workspace, 20 m reference path,
0–4 static obstacles, 700-step horizon, 0.1 s control period (10 Hz), Bluefin model
(64.55 kg, LOA 1.73 m, LBP 1.57 m, breadth 0.50 m, draft 0.19 m, Iz 10.45 kg·m²).

---

## 2. Target behaviour models

**This was the largest gap in the original task list.** "Add dynamic obstacles" specifies
presence but not behaviour, and behaviour determines what the agent actually learns.

**Confirmed (D1): constant-velocity for training; reactive for evaluation only.**

### 2.1 Constant velocity — training default

Deterministic geometry, reproducible encounters, and consistent with the maritime DRL
literature (Waltz & Okhrin, Woo & Kim, Xu et al. all assume linear deterministic
targets).

Training against reactive targets as the primary setting makes the environment
non-stationary and destroys attribution — you cannot tell whether a behaviour change
came from the policy or the opponent.

### 2.2 Compliant reactive — evaluation stratum

**Reuse the encounter-specific VO controller from `PAPER3_BASELINES.md`** (Thyri &
Breivik). The same implementation serves as both a comparator and a target behaviour
policy, saving an entire component.

Report as a generalisation stratum, not as the headline. Outcomes depend on two policies
and are harder to attribute.

**Fairness requirement:** run the classical baselines against reactive targets too, or
the comparison is not like-for-like.

### 2.3 Non-compliant — evaluation stratum

**Required by the Rule 17 contribution.** The stand-on release condition triggers on the
give-way vessel *failing to act*. With only compliant targets, the behaviour that
constitutes the strongest novelty claim can never be exercised.

Implement at least: targets that hold course when they should give way, and targets that
alter to port in a head-on.

---

## 3. Dynamic target implementation

Four details that are easy to get wrong and expensive to fix later.

**Oriented hulls, not circles.** Model targets with a heading and a hull polygon.
Required for ship-domain metrics and for correct aspect-angle computation in the
encounter classifier.

**Ray-cast against the moving polygon.** The target must be genuinely *perceived*
through the simulated LiDAR, not injected as ground truth. Otherwise the tracker gets a
free perfect detection and the sim-to-real gap in perception is hidden.

**Spawn outside LiDAR range.** Targets appear beyond `D_max` and are acquired as they
approach. Detection becomes part of the task rather than a teleport into the observation.

**Allow occlusion.** A target behind a static obstacle disappears from the scan. This is
a real narrow-waterway phenomenon and is the strongest argument for adding memory to the
architecture. Do not special-case it away.

Targets must also respect the channel — no passing through walls.

---

## 4. Channel geometry

**Requirement inherited from 01 §3.3.** The boundary observation branch is redundant if
the path runs down the centreline of a constant-width channel — port and starboard
clearances become affine functions of `e_y`.

The environment must therefore support:

- **Variable channel width** along the path
- **Bends** in the channel
- **Deliberately off-centre reference paths**

These are also exactly the restricted-waterway cases that distinguish this work from
open-water COLREGs DRL, so the requirement is aligned with the positioning rather than
being a tax.

---

## 5. Workspace scaling (O4)

**Open decision.** A 1.73 m vessel in a 10 m-wide channel gives roughly 5.8 breadths of
lateral room. Meaningful COLREGs encounter geometry needs enough range and time for
classification, decision and manoeuvre.

Usable LiDAR range on the RPLidar C1 is roughly 8–10 m — about 5 ship lengths. Tight but
workable for a single encounter; likely inadequate for three simultaneous targets.

Consider whether the simulation workspace grows beyond the physical basin dimensions.
There is a defensible argument either way, but it must be stated: either the simulation
matches the basin for field comparability, or it exceeds it for encounter realism and the
field trials cover a subset.

**Related requirement — Froude scaling.** State the model-to-full-scale relationship
explicitly. A reviewer will ask whether a 15 s TCPA on a 1.73 m model corresponds to
anything meaningful at full scale, and the answer needs to be in the paper rather than
improvised in response.

---

## 6. Action space

Unchanged in structure (continuous rudder + propulsion), but see 02 §4.5: **propulsion
authority may need to widen**, because Rule 8(e) makes slackening speed or stopping a
legal collision-avoidance action. Paper 2's staged curriculum restricted the propulsion
range; keeping that restriction removes a legal manoeuvre.

Rudder saturation and rate limiting stay matched to the real actuator, per Paper 2
Table 2. Revisit once 05 delivers a calibrated actuator model.

---

## 7. Noise and domain randomisation hooks

Build these as environment parameters from the start, populated by 05:

| Source | Injected into |
|---|---|
| Pose / odometry drift | boundary raycast, tracker ego-motion compensation |
| LiDAR angular resolution and range noise | raw scan |
| Aft self-occlusion sector | raw scan (masked bearing range) |
| Scan motion distortion | tracker velocity estimates |
| Vessel model parameters ± identification CI | dynamics |
| Actuator lag and delay | actuator model |

Making these first-class environment config rather than hardcoded values is what turns
"we identified the model" into "we identified the model and randomised within
identification uncertainty" — a substantially stronger claim.

---

## 8. Open items

- **O4** — workspace dimensions; resolve before freezing the scenario generator
- Decide propulsion authority (coupled to 02 §4.5)
- Define the target hull polygon and ship domain at model scale (coupled to 01 §5.2)
- Choose the speed threshold and hysteresis for static/dynamic classification
- Verify episode horizon: 700 steps at 0.1 s = 70 s. Confirm this is long enough for a
  full encounter sequence with three targets in a longer channel.
