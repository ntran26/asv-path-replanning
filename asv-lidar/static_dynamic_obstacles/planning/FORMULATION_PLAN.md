# Paper 3 — Section 2 (Formulation): writing plan and verified facts

> **Status note (2026-10-04).** The facts below are baseline-v2's and still hold for baseline-v3. The evaluation now uses the 1,000-episode test set (version 4), and the claims and tables are in `planning/PAPER3_CLAIMS_AND_TABLES.md`. Baseline-v4 is under test (`planning/BASELINE_V4_PLAN.md`).

**Written 2026-09-28** for drafting Section 2 in the six-section structure of
Introduction draft 4 (`planning/Paper3_Introduction_draft4.docx`; skeleton
revision 4). Every number below was read from the frozen config
`configs/baseline_v2.json` (formulation `3d697858e95e5adf`) or the source, not
from the older planning specs. Where this plan and an older document disagree,
this plan is right; section 6 lists the known disagreements.

**Which formulation.** Write Section 2 for **baseline-v2**, the formulation all
four learners are trained on. baseline-v3 (field stages 6–7; SAC seed 0
training since 28 Sep, done about 30 Sep) changes only the curriculum after
stage 4 and the validation set. If v3 is adopted, only section 2.3.5 (curriculum) and
one sentence in section 2.1.4 change; the box in section 2.3.5 holds the v3 text.

**Spelling and voice.** Draft 4 uses American spelling ("maneuver",
"behavior") and "encounter-consistent", "adopted operating convention",
"declared behavioral metrics". Keep them.

---

## 1. What Section 2 must do

Draft 4's roadmap: *"Section 2 presents the formulation: the problem and
workspace, the COLREGs scope and the geometric framework for constrained
encounter responses, and the frozen perception, observation, reward, and
curriculum design."* Section 3 then covers learners, training protocol,
classical comparators and evaluation design — so **network architecture,
hyperparameters, seeds, comparators and test sets are not in Section 2**.

The section has to let a reader reproduce the environment and understand why
each design choice was made, while delivering contributions **C1** (the
formulation) and **C2** (the geometric framework). It must also set up the
claims later sections test: the width sweep (C2, RQ6), the perception study
(C4, RQ3) and the supplied-context-versus-learned-response split (section 1.2.2 of
draft 4).

**Budget:** about 4,000–5,000 words, 5 tables, 3–4 figures, 8–10 numbered
equations. Put full coefficient lists in appendices C and D.

## 2. Structure, content and sources

### 2.1 Problem and workspace

**2.1.1 Task and decision process** (~400 words, Eq. 1–2)

- One own ship follows a reference path from start to goal, avoiding static
  obstacles, the navigable boundary and one moving target, and resolves the
  encounter in an encounter-consistent way. One policy, continuous action, 2 Hz.
- Partially observed MDP: observation `o_t` (70 values, section 2.3.2), action
  `a_t = [δ_cmd, n_cmd] ∈ [−1, 1]²` (rudder, throttle), reward `r_t` (section 2.3.4),
  discount `γ = 0.951` per 0.5 s step (0.99 per 0.1 s physics step, about a
  10 s horizon), episode cap 180 steps (90 s).
- Episode ends: goal region reached (+100), contact with obstacle, boundary or
  target (−300), or the cap (no terminal reward).
- **Why continuous propulsion:** Rule 8(e) makes slackening speed a lawful
  response and, in confined water, often the only admissible one. A policy that
  cannot slow cannot comply (skeleton section 2.1.3).

**2.1.2 Vessel and dynamics** (~350 words, Table 1)

| Item | Value |
|---|---|
| Vessel | model-scale ASV (Bluefin): LOA 1.725 m, beam 0.50 m (skeleton: 64.55 kg, LBP 1.57 m, draft 0.19 m, Iz 10.45 kg·m²) |
| Dynamics | 3-DOF model with actuator dynamics (servo rate limit, lag, delay, thrust map), integrated at 0.1 s |
| Decision rate | 2 Hz (0.5 s per action) |
| Propulsion | RPM = 6 + 6·n_cmd, clipped to [0, 12]; no astern for the policy |
| Cruise | 6 RPM → 0.558 m/s (`U_NOM`) |
| Domain randomization | dynamics parameters over their identified uncertainty (scale 1.0) |

- **State plainly (B5):** the model was identified from field logs centered on
  about 1.14 m/s at 12 RPM; the paper operates at 0.558 m/s on a Froude-scaling
  argument, i.e. below the identification band. Say it here, not only in the
  limitations.
- Cite Paper 2 (Tran et al., 2026) for the platform. The system-identification
  campaign is parked with physical transfer (S13); describe the model as
  identified from field logs (05), not from the planned campaign.

**2.1.3 Workspace** (~450 words, Fig. 1)

- **Basin mode (default):** the 10 × 25 m basin of Paper 2's field site. Start
  at y = 2 m and goal at y = 22 m, each with x drawn in [2.5, 7.5] m, so paths
  are straight or slanted (up to about 14°). The navigable polygon is the basin
  inset by 0.40 m. At 10 m (20 beams) the basin is already narrow water.
- **Channel mode:** parallel-walled channels, used only for the classes whose
  response the width decides (head-on, crossing, overtaking). Training widths by
  stage (section 2.3.5) reach 3.5–10 m. Reference paths are offset from the
  centerline by ±30 % of the local half-width, biased to starboard (mean +12 %,
  Rule 9(a) station), so the boundary branch carries information the
  cross-track error does not.
- **The boundary is from the map, not sensed.** In a restricted waterway the
  navigable limit is a charted contour, buoyed line or regulatory boundary that
  a range sensor cannot detect. The basin reproduces this: the LiDAR sees the
  facility walls 1–2 m beyond the edge. The boundary is ray-cast from the map at
  the noisy estimated pose (skeleton section 1.1 note; draft 4 section 1.1 ¶3).
- **Why two-vessel encounters** (skeleton section 2.1.1): Rules 13–16 are pairwise,
  and confinement makes encounters sequential rather than concurrent. A scope
  decision, stated here, not in the limitations.
- **Figure 1:** basin and a channel to scale, with a slanted leg, three panels,
  a target track, the ship domain and the map boundary versus facility walls.

**2.1.4 Traffic and static obstacles** (~350 words)

- **Target:** one vessel, **constant velocity, never gives way** (D1). Speed
  0.20–0.75 m/s, by class as a ratio of own speed (head-on 0.7–1.3, overtaking
  0.40–0.55, being overtaken 1.5–2.2).
- **Episode types and training shares:** head-on 0.20, crossing 0.22,
  overtaking 0.16, being overtaken 0.14, null (a target that never meets the own
  ship) 0.11, no target 0.17. Training draws 60 % of crossings from port (F84).
- **Being overtaken:** every draw passes at or above a contact-free floor, so
  holding course is always safe and Rule 17(b) is out of scope by construction
  (S5, F96).
- **Generation by class, not by position:** draw the class, then the
  heading-crossing angle, speed and time to CPA, and solve backward for the
  spawn (skeleton section 3.4.1).
- **Static obstacles:** 1.0 m panels, 0–3 per episode by stage, along 25–70 %
  of the path with lateral offsets. Panels are kept clear of ±0.4 T₀ of own-ship
  travel around the CPA so they do not decide the encounter (v3 relaxes this;
  Section 2.3.5).
- **Feasibility:** every layout admits a static route (A* on a 0.25 m grid,
  walls inflated 0.40 m and panels 0.45 m, route ≤ 2.25 × the leg). Draft 4
  says this is **not** a proof that every dynamic encounter is avoidable; keep
  that sentence.

**2.1.5 Scaling** (~150 words) — **[TBC]** a Froude statement so that TCPA,
CPA thresholds and domain sizes read at full scale (skeleton section 2.1.2). If it is
not ready, state the model-scale values and flag the scaling as a limitation.

### 2.2 COLREGs scope and constrained encounter responses

**2.2.1 Rule scope** (~300 words, Table 2) — the rule table of skeleton section 2.2.1.
Selected requirements of Rules 8, 9, 13, 14, 15/16 and 17(a)(i); Rules 2, 5, 6
and 7 acknowledged and operationalized implicitly; Rules 17(a)(ii), 17(b), 18,
19 and signals out of scope. Compliance is measured by **declared behavioral
metrics**, not legal certification.

**2.2.2 Encounter classification** (~450 words, Eq. 3–4, Table 3)

- Inputs are **perceived**: the tracked target (section 2.3.1) and the own ship's
  estimated pose. Relative bearing α (0° ahead, 90° to starboard) and
  heading-crossing angle c (180° reciprocal).
- Bands (after Waltz & Okhrin, 2023, with three changes; `src/encounter.py`),
  tested in Rule 13's order of precedence:
  - **overtaking / being overtaken** first: near-parallel courses (c within
    ±67.5°); overtaking if the own ship is in the target's stern arc
    (112.5–247.5°) and faster by a margin; being overtaken if the target is in
    the own ship's stern arc and faster;
  - **head-on:** |α| ≤ 10° and c within 180 ± 10°;
  - **crossing:** the target on the starboard bow crossing left to right, or on
    the port bow crossing right to left (bearing and crossing-angle bands
    either side of head-on);
  - **none** otherwise.
  **Changes from the source:** port and starboard crossing merged into one
  class (the side is kept for the turn direction and the passing-side term);
  the head-on band widened from ±5° to ±10°; a being-overtaken class added.
- **Classified against the path tangent, not the momentary heading** (A19), so
  the own ship's avoidance turn cannot relabel the encounter. Hysteresis 3° in
  bearing, and a new class must persist 2 steps (1 s).
- **Collision risk:** the maximum of a CPA-based and a distance-based term,
  with distance measured to the target's ship domain. The distance term matters
  in corridors: near-parallel courses give a CPA far away until a small turn
  makes it urgent.
- **Ship domain:** asymmetric, fore 3.14 m (2 L), aft 1.57 m (1 L), abeam
  1.25 m. **Required passing distance `d_req` = 2.5 m** (twice the abeam
  radius).

**2.2.3 Engagement and the latched encounter** (~350 words, Eq. 5, Fig. 2)

- **Engages** when the class is not none, 0 < TCPA ≤ 25 s and DCPA < 1.5·d_req.
  At engagement the class, the crossing side, the compliant turn sense and the
  heading and speed are **latched** (A20) until the target is past and opening
  (TCPA < 0 or DCPA > 2.5·d_req), confirmed over consecutive steps.
- **Why latch:** COLREGs fixes the situation when risk first develops;
  re-deciding it at close range let tracker noise turn a head-on into a port
  crossing and flip the required turn (F56).
- **Compliant turn sense** (Table 4): +1 starboard for head-on and for a
  crossing target from starboard; −1 port for a crossing target from port
  (pass astern of it) and for overtaking; 0 for being overtaken (hold).
- **Narrow-channel convention** (S3, A17): the own ship gives way to a crossing
  target from either side. Use **draft 4's wording** — an explicit operating
  convention motivated by the conditional non-impeding requirement of Rule
  9(b), not the open-water Rule 15/17 role assignment and not a consequence of
  vessel length. Do not use METHODS_BRIEF section 5.3's "a vessel of less than 20 m…
  displaces" phrasing, which claims more than draft 4 does.
- **Figure 2:** the engagement state machine (idle → engaged → clearing → idle)
  with the latched quantities.

**2.2.4 Geometric framework for constrained encounter responses** (~600 words,
Eq. 6–8, Table 5) — **the heart of contribution C2**

- **Lateral deficit:** `Δy_req = max(0, d_req − DCPA)`, the sideways offset
  still owed for a compliant pass.
- **Room:** the usable distance to the boundary on the compliant side, taken as
  the minimum over the passage from the present position to the projected CPA.
  The alteration is **admissible** if the room covers the deficit, with a
  ±0.15 m hysteresis band so the answer does not chatter at the width where the
  maneuver just fits.
- **Speed as the fallback, conditioned on clearance:** slowing is credited only
  if it would let the target pass clear. Simulate the own ship's braking path
  and require the target's DCPA to reach 1.76 m (A18, A23, A24). Against a
  reciprocal head-on, slowing cannot clear, so there the lateral room is the
  answer. Draft 4 section 1.2.3: "stopping in the path of a reciprocally approaching
  vessel may preserve the conflict".
- **Table 5** — the constrained-response table (skeleton section 2.2.3): per encounter,
  the wide-channel and narrow-channel response, governing rules and fallback,
  with the head-on rationale (9(a) channel-keeping satisfies 14 when both keep
  starboard) and the overtaking side (the room lies to the target's port).
- **Width as a geometric variable:** thresholds are an output of the sweep in
  Section 4.4, stated in beams and in the lateral excursion the compliant
  maneuver needs. Frozen predictions (`PREDICTED_THRESHOLDS_M`): head-on 3.8 m
  with a compliant target and 6.3 m with a target on the centerline; overtaking
  4.9 m. **[Reconcile with the claim ledger's 4.26 m overtaking and 7.60 m
  crossing before writing; section 6.]**

### 2.3 Perception, observation, reward, runtime layer and curriculum

**2.3.1 Perception** (~450 words, Fig. 3)

- **LiDAR:** 360°, 720 beams at 0.5°, range 1.0–16 m. Pose noise 0.03 m and
  0.2°, surge-speed noise 0.05 m/s, yaw-rate noise 1°/s.
- **Pipeline:** gate returns outside the map polygon → cluster → compensate for
  ego-motion at the estimated pose → nearest-neighbor association (0.70 m gate;
  confirmed after 2 hits, dropped after 3 misses) → constant-velocity Kalman
  filter → static or moving by **free-space motion evidence** over a 2 s window
  (returns appearing where beams passed through empty water, or vacating where
  the object was), not a speed threshold (F31). A panel's centroid slides as
  the viewpoint changes, and a speed test promoted panels to vessels in 28–33 %
  of frames. Static returns feed the sector branch; the moving track feeds the
  target slot.
- **Information parity:** the classical comparators consume the same tracks
  (Section 3.3).
- **Figure 3:** pipeline block diagram from scan to the observation branches
  and the reward.

**2.3.2 Observation** (~450 words, Table 6) — **70 values in six branches**
(schema `a25-v3-context`; `OBSERVATION_SPEC.md`)

| Branch | Values | Contents |
|---|---|---|
| lidar | 27 | closeness per sector over ±135°, static returns only; sectors finest ahead (6°), coarser abeam |
| boundary | 7 | map ray-cast at −90…+90° (30° apart) from the estimated pose |
| ego | 3 | surge, sway, yaw rate |
| path | 3 | cross-track error scaled by the local half-width on that side, course error, look-ahead course error |
| target | 16 | distance to domain; bearing (sin, cos); crossing angle (sin, cos); target speed; relative speed; DCPA; TCPA; risk index; class one-hot (5); presence bit |
| context | 14 | engaged; clearing; compliant turn sense; heading and speed change since engagement; turn admissible; slowdown clears; admissibility known; action required; proximity gate; in extremis; engagement age (12); previous action (2) |

- **Design rationale:** pooled range sensing is velocity-blind, so target
  kinematics enter through explicit tracking, which also gives parity with the
  comparators. The presence bit, not zero padding, marks an empty slot (zero is
  a valid bearing). Angles as sin/cos.
- **The context branch makes the reward's encounter judgment observable** (F72):
  the same encounter object feeds the observation and the reward ("one module,
  two consumers"), so the agent is never scored on a role it was not shown.
  Draft 4's framing follows: the classifier supplies the encounter regime, and
  the policy learns the rudder-and-propulsion response.
- **Policy inputs are perceived or from the map only** — no ground-truth target
  state.

**2.3.3 Reward** (~700 words, Eq. 9–10, Table 7)

- `r_t = Σ_i w_i r_i + r_terminal`. Every dense term is bounded to [−1, 0]
  (progress [−1, 1]) before weighting, so the **weight is each term's largest
  per-step contribution** and the hierarchy holds by construction. This fixes
  the Paper 2 failure, where a hidden scale made the obstacle term about 49×
  weaker than path following at contact.

| Term | Weight | Meaning | Evaluated on |
|---|---|---|---|
| collision (terminal) | −300 | any contact | ground truth |
| goal (terminal) | +100 | goal region | ground truth |
| `r_bnd` boundary | 15 | quadratic inside 0.35 m of the boundary | ground truth |
| `r_dom` ship domain | 12.5 | intrusion into the target's domain | ground truth |
| `r_obs` obstacles | 11 | shifted exponential, zero beyond 2.0 m | ground truth |
| `r_col` COLREGs group | 9 | Eq. 10, clipped to [0, 1] | perceived encounter |
| `r_pf` path following | 3 | cross-track and heading error, width-normalized, speed-gated | estimated path state |
| `r_prog` progress | 1.5 | along-path progress | estimated path state |
| `r_smooth` smoothness | 0.5 | action change relative to the actuator rate limit | — |
| `r_exist` existence | 0.25 | −1 per step | — |

- **COLREGs group (Eq. 10):**
  `v_col = clip(0.55 v_port + 0.55 v_bow + 0.40 v_side + 0.45 v_hold + 0.50 v_r8, 0, 1)`,
  each term weighted by the proximity gate
  `ρ = clip(1 − DCPA / (1.5 d_req), 0, 1)`:
  - `v_port`: turning or holding a heading against the compliant sense while
    giving way (beyond a 5° dead band, scaled by a 20° alteration), weighted by
    the peak ρ since engagement, so a wrong-way swerve cannot discount its own
    penalty by opening DCPA (F88).
  - `v_bow`: passing ahead of a crossing target, or cutting back across the bow
    after overtaking.
  - `v_side`: passing on the wrong side.
  - `v_hold`: not holding course and speed while being overtaken; the speed
    part keeps rising, so fleeing is not free (A29).
  - `v_r8`: late or insufficient action — the deficit between the alteration
    owed and the heading and speed change made, growing as TCPA falls below
    15 s.
- **Speed carve-outs (Rule 8(e)):** while engaged, (R-2) when the compliant
  turn is inadmissible and slowing would clear, the speed reference drops to
  0.4 × cruise; (R-5) in narrow overtaking, holding astern at the target's
  speed meets the reference and the existence cost is suspended. Without the
  gate the degenerate policy is "always slow".
- **Split of inputs (R-1):** physical terms read ground truth (contact is a
  fact); rule terms read the encounter as perceived. Say explicitly that ground
  truth enters the reward only, which exists only in training, so the policy
  has the same information as the comparators (draft 4 section 1.3).
- **Magnitude hierarchy:** collision ≫ boundary ≫ COLREGs ≫ path ≫ smoothness.
  A compliant collision is worse than a non-compliant near miss.
- **Verified property (F94):** from the same engagement state, the compliant
  crossing turn earns +61 (discounted, 30° alteration) over the wrong way, from
  either side — the reward is symmetric about crossing direction. This is a
  property of the reward, not a result; it belongs here.
- **Implementation notes worth a sentence:** turn direction judged from yaw
  rate, not rudder angle; the head-on port-turn penalty is class-gated because
  overtaking needs a port turn (the compliant-sense lookup removes that trap).

**2.3.4 Runtime layer** (~200 words) — the engineered stop safety layer: off in
training, evaluated separately, interventions not attributed to the policy
(S10, C-7). Rule 8(e) in two layers: the learned policy slackens speed; the
engineered layer takes all way off. **[Open, A38:]** the current safety layer
lowers success slightly (`planning/archive/safety/SAFETY_LAYER_V2_PLAN.md`). Describe the layer
that is actually evaluated; if safety layer v2 is adopted, it goes here instead.

**2.3.5 Curriculum** (~350 words, Table 8)

| Stage | From (of 2 M) | Episodes | Obstacles | Basin share | Channel widths |
|---|---|---|---|---|---|
| 1 | 0 | no target | 0–1 | 1.00 | — |
| 2 | 0.16 M | no target | 0–3 | 1.00 | — |
| 3 | 0.36 M | head-on, crossing (both sides), null, no target; upper TCPA | 0–1 | 0.85 | 7–10 m |
| 4 | 0.64 M | all six types | 0–2 | 0.75 | 4.5–10 m |
| 5 | 1.00 M | all six types | 0–3 | 0.75 | 3.5–10 m |

- 15 % of episodes start slow (from rest or up to half cruise). Targets are
  constant-velocity throughout; reactive and non-compliant targets are
  evaluation-only, because a reactive opponent makes training non-stationary
  and blurs attribution.
- **[Check before writing]** Verify stage 1–2 channel usage and the obstacle
  counts against `CURRICULUM_STAGES` in the frozen config (METHODS_BRIEF section 3 is
  the source of this table).

> **If baseline-v3 is adopted**, replace the table's end with: 2.5 M steps;
> stages 1–4 unchanged at the same step counts; stage 5 (1.0 M) weights three
> panels; **stage 6** (1.5 M) halves the CPA guard and draws 25 % coupled
> layouts; **stage 7** (2.0 M) removes the guard, draws 45 % coupled layouts
> (30 % of their targets change speed once), 30 % of coupled layouts near a
> deployment layout (1–1.5 m from it, never the tested layout); every stage 6–7
> target episode passes a space-time solvability check
> (`src/formulation_v3.py`, F107).

## 3. Figures and tables

| # | Content | Status |
|---|---|---|
| Fig. 1 | Workspace: basin and channel to scale, leg, panels, target track, domain, map boundary vs walls | to make (from `render.py` or the Paper 2 figure style) |
| Fig. 2 | Engagement state machine | to draw |
| Fig. 3 | Perception pipeline and the two consumers of the encounter context | to draw |
| (Fig. 4) | Constrained-response geometry: room vs lateral deficit, head-on and overtaking in a channel | optional; strong for C2 |
| Table 1 | Vessel and simulation | above |
| Table 2 | COLREGs scope | skeleton section 2.2.1 |
| Table 3 | Classification bands | above |
| Table 4 | Compliant turn sense | above |
| Table 5 | Constrained responses | skeleton section 2.2.3 |
| Table 6 | Observation | above |
| Table 7 | Reward | above |
| Table 8 | Curriculum | above |

Fig. 1 can be generated, and Figs. 2–4 drafted, as vector diagrams.

## 4. Equations

1. POMDP tuple and objective (discounted return, γ = 0.951).
2. Action mapping: rudder angle and RPM = 6 + 6·n_cmd clipped to [0, 12].
3. CPA: `TCPA = −(p·v)/|v|²`, `DCPA = |p + v·TCPA|` (relative position and
   velocity from the track).
4. Collision risk: max of the CPA and distance terms.
5. Engagement condition and release condition.
6. Lateral deficit `Δy_req = max(0, d_req − DCPA)`.
7. Admissibility: room over the passage ≥ Δy_req (with hysteresis).
8. Slowdown-clears test: DCPA along the braking path ≥ 1.76 m.
9. Reward sum with bounded terms.
10. COLREGs group with the proximity gate.

Exact forms of 3–10: `src/cpa_cri.py`, `src/colregs/context.py`,
`src/colregs/geometry.py`, `src/reward/terms.py`; appendix C gets the full
coefficient list.

## 5. What Section 2 must not claim

- Legal compliance, legal applicability of Rule 9 from width or operating
  bounds, or that width is a universal legal criterion (draft 4).
- That every dynamic encounter is avoidable (the A* check is static).
- That the policy learns the rule regime — the classifier supplies it; the
  policy learns the response.
- Ground-truth target state in the policy input (it is only in the reward).
- Results. Section 2 states design and properties of the reward (F94), not
  performance.

## 6. Resolve before or while writing

| Item | Where it disagrees | Action |
|---|---|---|
| Abeam domain | skeleton: 0.75 L = 1.18 m; config: 1.25 m (`DOMAIN_LATERAL`), d_req 2.5 m | **done 2026-10-08**: skeleton, `00`, `01`, `02a`, `03a` and `PROJECT_BRIEF` now quote 1.25 m and d_req 2.5 m |
| Head-on width threshold | skeleton: 3.66 m; config: 3.8 m (compliant target), 6.3 m (centerline target) | **done**: the skeleton quotes `PREDICTED_THRESHOLDS_M`; `02a` and `03` note both values |
| Overtaking / crossing thresholds | ledger C-3: 4.26 m / 7.60 m; config: overtaking 4.945 m | check `04a` section 6 and the sweep design; pick one set and restate C-3 |
| Observation table | skeleton section 2.3.2: ~56 values | use Table 6 (70 values) |
| LiDAR rate | skeleton: 10 Hz; decisions at 2 Hz | state what the simulator does per decision step |
| Scaling statement | skeleton section 2.1.2 [TBC] | write, or move to limitations |
| Formulation | v2 vs v3 | decide after the v3 SAC run (section 2.3.5 box) |
| Safety layer | A38 | describe the evaluated version |
| Model provenance | "identified" vs the planned sys-ID campaign (S13) | "identified from field logs (05)"; the campaign is physical transfer |

---

## 7. Drafting inputs

Attach these (in this order of importance):

| # | File | Why | Note |
|---|---|---|---|
| 1 | `planning/FORMULATION_PLAN.md` (this file) | structure, verified numbers, what not to claim | the authority where sources disagree |
| 2 | `planning/Paper3_Introduction_draft4.docx` | voice, terminology, spelling, citation style, the claims Section 2 must support | |
| 3 | `planning/METHODS_BRIEF.md` | rationale for each choice (F/A numbers) | Sections 7–10 are partly stale (five learners, five seeds, "pending" status); Section 2 uses only sections 1–6 |
| 4 | `OBSERVATION_SPEC.md` | exact feature definitions and normalizations | for Table 6 and the observation text |
| 5 | `planning/PAPER3_DRAFT_SKELETON.md` | Section 2 notes, scope table S1–S13, the rule and response tables | its section 2 numbers are superseded by this plan (section 6) |
| 6 | Paper 2 PDF (Tran et al., 2026, *Drones* 10(9):680) | platform, basin, LiDAR pooling, consistent description | |

Optional, for a constant's justification: `CONSTANTS_AND_SCALES.md`
(long). **Do not attach** `02a_REWARD_SPECIFICATION.md`, `01`, `03a` or `04a`
(stale in places; this plan and METHODS_BRIEF supersede them) or
`PROJECT_STATE.md` (too long; the F-numbers are summarized here).

**Drafting brief:**

> Draft Section 2 ("Formulation") of Paper 3, following the structure, numbers
> and exclusions in FORMULATION_PLAN.md exactly; where attached files disagree,
> FORMULATION_PLAN.md wins. Match Introduction draft 4's voice, American
> spelling and terminology ("encounter-consistent", "adopted operating
> convention", "declared behavioral metrics"). Write for baseline-v2. Include
> Tables 1–8 and Equations 1–10, and placeholders for Figures 1–3. Cite only
> works in draft 4's reference list or Paper 2; mark any other needed citation
> [CITE]. Do not state results. Mark every item in the plan's section 6 that could
> not be resolved as [CHECK].
