# Methods brief — Paper 3, formulation baseline-v1

**Purpose.** One current, paper-ready statement of the method, for drafting the
Methods, Experimental Setup and Limitations sections. It states the formulation
**as frozen** in `configs/baseline_v2.json` (formulation digest
`3d697858e95e5adf`), with each design choice tied to the finding that justifies
it (`F..` = `PROJECT_STATE.md`, `A..` = `OPEN_PROBLEMS.md`).

Written 2026-09-22, revised 2026-09-23 with your three decisions: the Rule 17(b)
below-floor draws leave training **and** the suite (S5), field work is **delayed
within this paper** (not deferred), and the paper is **a formulation plus a
five-learner comparison**. baseline-v1 (`configs/baseline_v1.json`, git tag
`baseline-v1`) differs in exactly one constant and is kept for the record.
Values were read from the frozen config and the source, not from the planning
specs.

> **Precedence.** Where this brief disagrees with an earlier document, this
> brief and `configs/baseline_v1.json` are right. `OBSERVATION_SPEC.md` and
> `CONSTANTS_AND_SCALES.md` were brought up to date on 2026-09-22. Known stale
> documents: `01_PERCEPTION_AND_OBSERVATION.md` (describes the 56-value
> observation, not the current 70); `02a_REWARD_SPECIFICATION.md`
> (predates the A24 stop test, A27 held-heading charge, A29 growing `v_hold`
> and F88 latched risk); `03a`/`04a` (predate basin mode, F74); the draft
> skeleton (7 Sep); `RESULT_TABLES.md` (still names a single "Proposed (SAC)"
> method, see §9). Each stale planning spec now opens with a dated status note
> listing what is superseded.

---

## 1. Problem and scope

A single autonomous surface vessel (ASV) must follow a reference path through
confined water, avoid static obstacles, and resolve an encounter with one
moving vessel in a manner consistent with COLREGs, using only onboard range
sensing (LiDAR) plus its own navigation state. One end-to-end policy produces
rudder and throttle commands at 2 Hz.

**In scope:** Rules 8, 9, 13, 14, 15/16 (as a narrow-channel convention, §5.3)
and 17 for **one** target at a time; static obstacles; basin and channel
geometry. **Out of scope (stated in the paper):** multi-target encounters,
Rule 19 restricted visibility, sound signals, Rule 18 responsibilities, open-water
Rule 15/17 role tables, and any claim of legal compliance: violation terms are
proxies (`CLAIM_LEDGER.md` §2).

## 2. Vessel and simulation

| Item | Value | Source |
|---|---|---|
| Vessel | length 1.725 m, beam 0.5 m (model-scale ASV) | `VESSEL_LENGTH`, `VESSEL_WIDTH` |
| Dynamics | model identified from field logs (05), integrated at 0.1 s | `05_VESSEL_MODEL_AND_SIM2REAL.md` |
| Decision rate | **2 Hz** (0.5 s per action) | `UPDATE_RATE` |
| Cruise | 6 RPM → **0.558 m/s** | F24, `U_NOM` |
| Action | continuous `[rudder, throttle] ∈ [−1, 1]²`; RPM = 6 + 6·throttle, clipped to **[0, 12] RPM** (no astern) | propulsion stage 4 |
| Episode limit | 180 steps (90 s) | `MAX_EPISODE_STEPS` |
| Domain randomisation | hull/dynamics parameters randomised over their identified uncertainty (scale 1.0) | `VESSEL_RANDOMISATION_SCALE` |
| Sensor noise | pose 0.03 m / 0.2°, surge speed 0.05 m/s, yaw rate 1°/s | `BOUNDARY_POSE_NOISE_*`, `EGO_*_NOISE` |

**State this in the model section, not only in Limitations (B5).** The plant
was identified from field logs whose speeds centre on a **median 1.14 m/s at 12
RPM** across 18 logs; the paper operates it at 0.558 m/s (6 RPM), chosen on a
Froude-scaling argument (F24). The identified model is therefore used **below
the band its identification data covers**, which is exactly where the
sim-to-real claim (RQ4) is weakest, so the model section should say so plainly
rather than leave it to Limitations.

## 3. Environment and scenarios

**Basin mode (default; F74, spec 06).** The 10 × 25 m basin of Paper 2's field
site. Start and goal have fixed along-basin positions (y = 2 m and 22 m) and
lateral positions drawn in x ∈ [2.5, 7.5] m, so reference paths are straight
and slanted across the basin. The navigable polygon is the basin inset by
0.40 m. At 10 m the basin is already narrow water.

**Channel mode.** Parallel-walled channels of variable width, used only for the
classes whose rule the width decides — head-on, crossing, overtaking (Rule 9
with 14, 15, 13) — at 15–25 % of their draws in later curriculum stages. Inside
the basin, only head-on traffic keeps a path band (its own side of the
fairway); other targets use the whole basin (F74).

**Feasibility guarantee (F74).** Every layout must admit a route: an A* search
on a 0.25 m grid, walls inflated 0.40 m and obstacles 0.45 m, route at most
2.25 × the leg length; up to 20 redraws, then obstacles are thinned. Layouts are
hard but never impossible, so classical failures are not infeasibility
artefacts (claim C-9).

**Targets.** One moving vessel, constant velocity, **never giving way** (D1):
the policy cannot rely on the other vessel co-operating. Speeds
0.20–0.75 m/s. Encounter classes and training shares: head-on 0.20,
crossing 0.22, overtaking 0.16, being overtaken 0.14, null (target present,
no encounter) 0.11, no target 0.17.

- **Being overtaken (A15, as amended for baseline-v2):** **every** draw passes
  at or above a floor where holding course is safe, so Rule 17(a)(i) is always
  the lawful answer. The labelled 20 % below that floor — the Rule 17(b)
  last-moment case — was removed from training and from the suite, because S5
  put active release out of scope and keeping the draws produced collisions
  with no claim behind them (F96).
- **Crossings (A22):** 80 % escapable, 20 % labelled unescapable. Training
  draws 60 % of crossings from port (F84); evaluation draws are even.

**Static obstacles.** 1.0 m obstacles placed along 25–70 % of the path with
lateral offsets; 0–3 per episode by curriculum stage.

**Curriculum** (fraction of the training budget → stage):

| Stage | From | Classes | Obstacles | Basin share |
|---|---|---|---|---|
| 1 | 0 % | no target | 0–1 | 1.00 |
| 2 | 8 % | no target | 0–3 | 1.00 |
| 3 | 18 % | head-on, crossing (both sides, A27), null, no target | 0–1 | 0.85 |
| 4 | 32 % | all six | 0–2 | 0.75 |
| 5 | 50 % | all six | 0–3 | 0.75 |

15 % of episodes start at low speed (from rest or up to half cruise).

## 4. Perception and observation

**Pipeline.** A 360° LiDAR (720 beams, 0.5°, 16 m range, 1.0 m minimum range)
is split by a free-space motion classifier into static returns and moving
returns. Moving returns feed a tracker (centroid measurement, 0.70 m gate,
confirmed after 2 hits, dropped after 3 misses) from which the target's
bearing, course, speed, DCPA, TCPA and encounter class are computed. **The
policy sees the perceived target, not ground truth.** The channel/basin
boundary is ray-cast from the map at the noisy estimated pose (a virtual
boundary, as used on the vessel).

**Observation: 70 values in six branches** (schema `a25-v3-context`, F72):

| Branch | Values | Contents |
|---|---|---|
| lidar | 27 | closeness per bearing sector over ±135°, static returns only; sectors finest ahead (6°), coarser abeam |
| boundary | 7 | virtual boundary distance at −90…+90° (30° apart) |
| ego | 3 | surge, sway, yaw rate |
| path | 3 | cross-track error **scaled by the local channel half-width on that side** (F72), course error, look-ahead course error |
| target | 16 | distance to domain, bearing (sin, cos), relative course (sin, cos), target speed, relative speed, DCPA, TCPA, collision-risk index, encounter class one-hot (5), presence bit |
| context | 14 | per target: engaged, clearing, **compliant turn sense**, heading change and speed change since engagement, turn admissible, slowdown clears, admissibility known, action required, proximity gate, in extremis, engagement age (12); previous action (2) |

The context branch makes the reward's encounter state observable, so the
policy does not have to infer from a single frame what the reward is judging
(F72).

**Network.** A scene MLP (lidar, boundary, ego, path, previous action →
128) and a target-slot MLP (target + context, 28 → 64 → 64 → 32) whose input is
gated by the presence bit, so an empty slot cannot read as a target on top of
the vessel; the slot encoder has shared weights, so a multi-target extension
is a retrain rather than a redesign. Features (160) feed actor and critic heads
of 256 × 256 ReLU. RecurrentPPO adds a 256-unit LSTM, separately for actor and
critic.

## 5. Reward

**Structure.** `r_t = Σ w_i r_i + r_terminal`, every dense term bounded to
`[−1, 0]` or `[−1, 1]` before weighting, so the weights alone set the
hierarchy.

| Term | Weight | Meaning | Evaluated on |
|---|---|---|---|
| Collision (terminal) | **−300** | any contact | ground truth |
| Goal (terminal) | **+100** | reaching the goal region | ground truth |
| `r_bnd` boundary | 15 | quadratic inside 0.35 m of a wall; the largest dense weight, so slowing always beats touching the boundary | ground truth |
| `r_dom` ship domain | 12.5 | intrusion into the target's asymmetric domain | ground truth |
| `r_obs` obstacles | 11 | shifted exponential, zero beyond 2.0 m | ground truth |
| `r_col` COLREGs group | 9 | §5.1, clipped to [0, 1] before weighting | perceived encounter |
| `r_pf` path following | 3 | cross-track and heading error, **width-normalised**, gated by speed | vessel path state |
| `r_prog` progress | 1.5 | along-path arclength progress (sums to a constant over a completed path) | vessel path state |
| `r_smooth` smoothness | 0.5 | action change relative to the actuator rate limit | — |
| `r_exist` existence | 0.25 | −1 per step | — |

Timeout carries no terminal reward. Boundary, domain and obstacle terms use
ground-truth clearances, because contact is a physical fact (R-1); COLREGs
terms read the encounter as the vessel perceives it (tracked target, latched
state). The observation's path branch is computed from the perceived pose.

### 5.1 COLREGs group

```
v_col = clip(0.55·v_port + 0.55·v_bow + 0.40·v_side + 0.45·v_hold + 0.50·v_r8, 0, 1)
r_col = −v_col
```

Each term is weighted by a proximity gate `ρ = clip(1 − DCPA / (1.5·d_req), 0, 1)`,
with `d_req` = 2.5 m (twice the abeam domain radius).

| Term | Rule | Penalises |
|---|---|---|
| `v_port` | 8, 14, 15/16 | turning **or holding a heading** against the compliant sense while give-way. Held heading counts beyond a 5° dead band, scaled by a 20° alteration (A27). Weighted by the **peak** `ρ` since engagement, so a wrong-way swerve cannot discount its own penalty by opening DCPA (F88) |
| `v_bow` | 15 | passing ahead of the target (crossing), cutting back across its bow (overtaking) |
| `v_side` | 13, 14 | passing on the wrong side (class-dependent sign) |
| `v_hold` | 17(a)(i) | not holding course and speed while being overtaken; the speed part keeps rising past 0.10 m/s to 3× (A29), so fleeing an overtaker is not free |
| `v_r8` | 8 | late or insufficient action: the deficit between the alteration owed and the heading and speed change made, growing as TCPA falls below 15 s |

**Speed carve-outs.** The default path term charges any speed below the
reference, which would make the lawful slow-down unlearnable. Two carve-outs
lower the speed reference while an encounter is engaged:

- **R-2, Rule 8(e).** Give-way, compliant alteration inadmissible (no room),
  and slowing would clear the target: the speed reference drops to 0.4 × cruise,
  so slackening speed costs nothing. The same test gates the speed credit in
  `v_r8`. "Would clear" means the own ship's braking path is simulated and the
  target's DCPA with the own ship stopped must reach 1.76 m (A18, A23, A24).
  Against a reciprocal head-on slowing cannot clear, so there the lateral room is
  the answer; paying for the slow-down there taught the agent to stop in the
  target's path (A18).
- **R-5, narrow overtaking.** Where the port pass does not fit, holding astern
  at the target's speed satisfies the speed reference and the existence cost is
  suspended.

### 5.2 Encounter state

A target is classified against the **path tangent** (A19) into head-on,
crossing (from port or starboard), overtaking, being overtaken or none, with
3° bearing hysteresis. The encounter **engages** when the class is not none,
0 < TCPA ≤ 25 s and DCPA < 1.5 · `d_req`. Class, compliant sense, heading and
speed at engagement are then **latched** (A20) and held until the target is
past and opening (TCPA < 0 or DCPA > 2.5 · `d_req`).

### 5.3 Compliant turn sense (A17)

Starboard (+1) for head-on and for a crossing target from starboard; port (−1)
for a crossing target from port, and for overtaking.

**The own ship gives way to a crossing target from either side, on Rule 9(b):**
a vessel of less than 20 m in length shall not impede the passage of a vessel
that can safely navigate only within a narrow channel. In a channel this
displaces the open-water Rule 15/17 role assignment, so the approach side
decides the *direction* of the give-way turn, not whether there is an
obligation. This is an interpretation of a rule, not a convention invented
here, and it is deliberately preferred to the Rule 18 route used by Meyer et
al., whose premise — own ship much smaller than the vessels it meets — fails
when own ship and target are similarly sized model vessels.

*Corroborating, not the basis:* against a constant-velocity target that never
gives way, holding course as the stand-on vessel reaches the goal in only 0.32
of port crossings (A27), so the interpretation also matters in practice. That
measurement depends on the target model and therefore cannot carry the argument
by itself.

The paper does not claim open-water Rule 15/17 roles.

### 5.4 Verified property (F94)

From the same engagement state, turning the compliant way returns +61
(discounted, 30° alteration) over the wrong way, both for targets from port
and from starboard. The reward prefers compliance in every pair that reaches
the goal either way, and the compliant turn costs nothing on the path terms.
The reward is therefore symmetric and decisive about the crossing direction;
see §8 for what the learner makes of it.

## 6. Runtime safety layer (F68, claim C-7)

> **A deliberate change to decision D2**, which ruled out "a separate safety
> layer". The layer exists because Rule 8(e) has two parts the paper must not
> conflate: slackening speed, which is learned and claimed, and taking all way
> off, which is engineered. It is off during training, evaluated both ways, and
> compliance is reported with it off, so nothing it does is counted as learned
> compliance. Record the change in `00` rather than leaving the documents in
> silent conflict.

An engineered supervisor, separate from the policy, takes all way off when a
collision is imminent and stopping would clear it. It is **off during
training** and evaluated **both off and on**. Compliance metrics are reported
with it off; with it on, the intervention rate is its own column. 8(e) is thus
claimed in two layers: the learned policy slackens speed; the engineered layer
stops. Only the first is a learned-compliance claim.

## 7. Learners and training protocol

Five learners on the identical formulation, differing only in the algorithm
and its original paper's default hyperparameters (Stable-Baselines3 2.3.2,
sb3-contrib 2.3.0):

| Learner | Key settings |
|---|---|
| PPO | lr 3e-4, 512 steps × 10 workers per update, batch 512, 10 epochs, GAE λ 0.95, clip 0.2, **target KL 0.03** (early-stops an update, F51) |
| RecurrentPPO | PPO settings + 256-unit LSTM (actor and critic separate) |
| TD3 | lr 3e-4, buffer 1 M, batch 256, τ 0.005, 10 k learning starts, policy delay 2, target noise 0.2 (clip 0.5), exploration noise 0.1 |
| SAC | lr 3e-4, buffer 1 M, batch 256, τ 0.005, 10 k learning starts, automatic entropy |
| TQC | SAC settings, 2 critics × 25 quantiles, top 2 per critic dropped |

**Common:** discount 0.951 per 0.5 s step (0.99 per 0.1 s physics step, a
~10 s horizon); reward normalisation only (VecNormalize, clip 10); 10 parallel
environments; **2 × 10⁶ environment steps**; off-policy learners at **1.0
gradient step per transition**; **5 seeds** per learner; development-set
evaluation every 2 × 10⁵ steps; each seed represented by its **best
development-set checkpoint** (score = goal rate − 2 × collision rate,
supervisor off). The frozen evaluation suite is never used for selection (A26).

**Development and fairness.** The observation and reward were **specified from
the rule analysis** (02, 02a, 03a, 04a) and are learner-independent: nothing in
the formulation references an algorithm. They were **iterated with PPO as the
development learner** over twelve runs (your decision that PPO is the debug
vehicle), and several of the checks that drove those iterations are
learner-free — the scripted-response studies and the A32 reward-gap measurement
branch scripted manoeuvres with no policy involved (F94). All five learners then
train on the frozen formulation with their original papers' default
hyperparameters and no further tuning. The paper should say this plainly: no
learner is privileged by the design, and the iteration loop that used PPO is
reported.

## 8. Evaluation protocol

| Set | Size | Use |
|---|---|---|
| Development set | 120 episodes (20 per class × 6), development namespace | checkpoint selection, diagnostics |
| Tier A — named cases | 38 defined (35 realised), incl. basin cases and pre-committed expected failures | per-case behaviour |
| Tier B — stratified holdout | 48 cells × 20 = 960 episodes per seed | headline results, touched once per policy |
| **Around the Clock** (O1) | 24 open-water + 24 channel cases × 10 seeds (`suite.around_the_clock`), reported as R8 | the one **externally defined** scenario set in the paper, after Imazu was dropped |

Tier B strata are described by geometry only (basin, channel wide /
intermediate / narrow), never as "hard for classical methods". Every result
reports the seed spread: the pre-registered tables give the mean over 5 seeds
with a 95 % CI (`RESULT_TABLES.md`). Pre-registered metrics: success;
collisions by type (static, boundary, target); RMS cross-track error; path
ratio; intervention rate with the supervisor on (R1); violation rate per class,
with crossings split by side (R2); results by target behaviour (R3), channel
width (R4), perception degradation (R5), ablation (R6), Tier A (R8) and field
(R9). First-alteration compliance by side is a diagnostic added since (F92).

**Measured seed spread** (PPO, two seeds): 0.05 on the development-set goal
rate, 0.20 on crossings (F90). Five seeds separate headline differences, not
crossing-level ones.

## 9. Status: fixed versus pending

**Fixed (can be written now):** §§1–7, the evaluation design in §8, and the
limitations in §10.

**Decided 2026-09-23:** the paper is **a formulation plus a five-learner
comparison** (not a proposed learner plus baselines), which is what was
pre-registered and what the campaign produces, and which makes the classical
comparators essential rather than optional for N2. **Field work is delayed
within this paper, not deferred:** the basin sessions of `PART2_BASIN_PLAN.md`
still happen, so N3, RQ4 and the "physically reproducible" clause that justifies
the two-vessel scope all stand — at the cost of a timeline that depends on pool
access and on deployment (C13).

**Pending:**
- **Results.** The campaign restarts on baseline-v2 (F96); the three runs made on
  baseline-v1 (run 11's two PPO seeds and RecurrentPPO seed 0) do not carry over.
  About 6 days for seed 0 of every learner, ~4 weeks for all 25 (A26, F95).
- **Evaluation machinery.** The Tier B runner and the classical comparators
  (encounter-specific VO, COLREGs-VO, LOS-PID + DWA; C3–C6) are not built. The
  suite freeze, claim ledger and result tables are unsigned drafts.
- **A framing decision.** `RESULT_TABLES.md` still names a single "Proposed
  (SAC, full)" method and pre-registers runs outside the campaign — "SAC, no
  COLREGs terms", the 2 × 2 feature/reward ablation (C-4), randomisation-off
  (C-6) — plus Paper 2's SAC and the three classical comparators. The paper is
  now either **a formulation plus a five-learner comparison**, or a proposed
  learner plus baselines. The ledger and tables must say which, and which
  ablations the compute budget covers, before sign-off.
- **C-4's ablation needs restating.** It was written as 2 × 2 (encounter-class
  feature × reward shaping) before the observation gained the `context` branch
  (F72). "Told or learned" now has **three rungs** — no encounter feature, the
  class one-hot, and the full context branch (latched sense, admissibility,
  action required) — and the third is the strongest form of telling. A 2 × 2
  understates what is handed to the policy.
- **Open decisions carried from the Rev 2 review** (2026-09-23): whether field
  work is deferred to a later paper or delayed within this one (it removes N3
  and the reproducibility clause that justified the two-vessel scope); and
  whether Rule 17(b) returns as a claim or the below-floor draws leave training
  and are reported as a characterised failure mode (S5 put active release out of
  scope; A15 later put those draws in).

**Development-set context, not paper results.** The best PPO formulation run
(run 11, which baseline-v1 reproduces exactly) reached a development goal rate
of 0.92 and 0.88 on two seeds, head-on 1.00 on both, crossing 0.75 and 0.55.

## 10. Limitations (to state in the paper)

1. **Crossing direction is not side-conditioned under PPO.** Six PPO policies
   each pick one opening direction for both sides, although the policy reads
   the compliant-sense input and the reward pays symmetrically for compliance
   (F90–F94). Observability, training exposure and reward magnitude are ruled
   out. Crossing compliance is reported by side for every learner (A32).
2. **Narrow head-ons with no starboard room.** Where the starboard alteration
   does not fit, the policy still turns starboard and collides in 0.36 of such
   cases (F91). A fix tried in run 12 made one seed abandon the starboard
   alteration even in open water (F92) and was withdrawn (A33).
3. **Rule 17(b) is out of scope by design (S5), and now by construction.**
   Being-overtaken draws all sit at or above the contact-free floor, so the
   paper never asks the policy to choose between holding course and acting. A
   waterway where an overtaker does not keep clear at close quarters is outside
   what is claimed (F96).
4. **Rule 9(b) displaces the open-water Rule 15/17 roles** in a narrow channel
   (§5.3): the own ship gives way to a crossing target from either side. An
   interpretation the paper defends, not a role table it claims.
5. **One target, constant velocity, non-co-operating.**
6. **Manoeuvring at 0.55 m/s extrapolated** from the field logs (B5), i.e. the
   identified model is used below its identification band (§2). Field results
   are **pending, not dropped**: two basin sessions (`PART2_BASIN_PLAN.md`) and
   the bridge deployment (C13) deliver N3, RQ4 and claim C-6.
7. **Formulation developed with PPO** (§7).
8. **Proxies, not legal compliance** (§1).

## 11. Where to find more

| Topic | File |
|---|---|
| Every frozen number | `configs/baseline_v1.json` |
| Each constant and why | `CONSTANTS_AND_SCALES.md` |
| Findings F1–F95 (full rationale, chronological) | `PROJECT_STATE.md` |
| Open decisions and known limits | `OPEN_PROBLEMS.md` |
| Basin geometry | `planning/06_BASIN_MODE_SPECIFICATION.md` |
| Claims and pre-committed tables | `planning/CLAIM_LEDGER.md`, `planning/RESULT_TABLES.md` |
| Running the campaign | `TRAINING_GUIDE.md` |
