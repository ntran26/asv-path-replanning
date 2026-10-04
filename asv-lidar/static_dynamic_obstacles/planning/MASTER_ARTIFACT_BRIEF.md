# Paper 3 master artifact — build brief

**Written 2026-09-29.** This file specifies what to build. The facts it
must use are in `MASTER_ARTIFACT_FACTS.md` (vessel model, classical methods,
learners, perception, data generation, field bridge) and
`FORMULATION_EQUATIONS.md` (action, observation, encounter geometry, reward).
The scope, contributions and literature come from
`Paper3_Introduction_draft4.docx`; the platform and field trials from Paper 2
(Tran et al., 2026, *Drones* 10(9):680).

---

## 1. What it is and who it is for

An interactive explainer of Paper 3's **scope, objective and methods** — not
its results, which are pending. A reader with a general engineering degree
(statics, dynamics, basic calculus, some programming) but **no** maritime
engineering, control or machine-learning background should be able to read it
top to bottom and understand what the vessel does, how it senses, how it
decides, how it is trained and how it is tested.

Paper 3 in one sentence: a small autonomous boat follows a planned path through
confined water, avoids fixed obstacles and one other moving vessel in a way
consistent with the collision regulations (COLREGs), using only its own LiDAR
— and four deep reinforcement learning algorithms are trained and compared on
one fixed problem formulation, against classical controllers.

## 2. Global rules

1. **Plain language first, then precision.** Introduce every idea with an
   everyday analogy or picture, then the exact definition. Define every term
   at first use (hover tooltip or glossary link).
2. **Every equation gets four things:** the equation (rendered properly, KaTeX
   if available); one sentence "**In words:**"; a symbol table (symbol,
   meaning, unit, value used here); and where possible a small worked example
   with the real numbers ("at cruise, u = 0.558 m/s, so …").
3. **Every major idea gets a visual**, and the ones marked ▶ below get an
   **interactive** widget (sliders, draggable points, play/pause animation).
   Widgets must work with no server; keep the maths light enough to run live in
   the browser. Label anything illustrative as illustrative.
4. **Numbers come only from the attached facts files.** Do not invent
   parameters. If something needed is missing, show a clearly marked
   placeholder `[TBD]` rather than guessing.
5. **Citations:** cite the method's source where it is used (author, year),
   with a reference list at the end. Prefer Fossen for vessel dynamics and
   guidance. Keep any reference the facts files mark [VERIFY] visibly marked.
6. **Stance on the regulations (from Introduction draft 4):** the COLREGs
   subset is implemented as an *adopted operating convention*; compliance is
   judged by *declared behavioral metrics*, not legal certification; channel
   width is a *geometric experimental variable*, not a legal definition of a
   narrow channel. Use "encounter-consistent" for the desired behavior.
7. **Do not state results.** Where a result will go, say so ("reported in the
   paper").
8. **Two exclusions from the author:** (a) in the vessel-model section, do not
   mention the data files the model was identified from — present the model and
   its identified parameters; (b) do not mention any field-deployment test set;
   describe curriculum stages 6–7 as training on more, and more evenly spread,
   three-obstacle layouts.
9. **The curriculum described is baseline-v3** (seven stages, 2.5 M steps).
10. **Spelling:** American ("maneuver", "behavior"), matching the paper.
11. **Nomenclature** at the end: every symbol (with unit and the section where
    it first appears) and every abbreviation. Group by topic.

## 3. Structure

Tabs or a sticky side navigation; each section opens with a two-line "In this section"
summary and ends with a three-bullet "Key takeaways".

### 0. Start here
One-screen summary: the problem, the four ingredients (vessel, sensing,
decision-making, testing), and a map of the sections. A short "how to read the
equations" box (what bold vectors, hats and subscripts mean).

### 1. Scope and objective
- The problem: route following + static obstacles + one moving vessel in
  confined water; why these interact (a lawful turn may be blocked by a wall or
  a panel, so speed becomes the answer — Rule 8(e)).
- Why onboard LiDAR only (no AIS), and why the navigable boundary comes from a
  map (a channel edge is often a depth contour, not a wall).
- The workspace: 10 × 25 m basin (model scale) and parallel-walled channels.
- Objective, the four contributions and the central questions (draft 4, section 1.3),
  and what is out of scope (multi-target, restricted visibility, Rule 17(b),
  Rule 18, legal compliance).
- ▶ **Overview animation:** top-down basin; the own ship follows a slanted leg
  past three panels; LiDAR rays sweep; a target appears; the encounter is
  classified (label pops up); the own ship alters or slows and passes; reaches
  the goal. Controls: play/pause, encounter type (head-on, crossing from port
  or starboard, overtaking, being overtaken), channel width (10 → 4 m, showing
  the turn stop fitting and speed reduction taking over).

### 2. The collision regulations in brief (added)
- What COLREGs are; Rules 8, 9, 13, 14, 15/16, 17(a)(i) in plain words, each
  with a small diagram.
- The five encounter classes with their bearing/crossing-angle bands (facts:
  `FORMULATION_EQUATIONS.md` section 4.4) and the required turn sense (+1 starboard,
  −1 port, 0 hold).
- The narrow-channel convention (give way to a crossing target from either
  side, motivated by Rule 9(b)) and why it differs from the open-water role
  table.
- Ship domain (asymmetric ellipse, 3.14 / 1.57 / 1.25 m) and required passing
  distance d_req = 2.5 m.
- ▶ **CPA widget:** drag the target and its velocity arrow; live DCPA, TCPA,
  relative bearing, crossing angle, class, CRI, and whether the encounter
  engages (Eq. 3–7 of the equations file).

### 3. The vessel and its dynamics
- Coordinate frames (world north-east vs body surge-sway; heading clockwise
  from north) — diagram.
- From F = ma to three coupled equations: kinematics then kinetics
  (facts section 1). Explain **added mass** (why pushing a hull sideways "feels" twice
  as heavy: m₂₂ ≈ 127 kg vs m = 64.6 kg), **damping** (drag grows with speed),
  **Coriolis/centripetal** terms (why turning couples surge and sway), the
  **Munk moment** (why an elongated hull at an angle wants to turn further), the
  **rudder as a small wing** (lift from angle of attack), the **propeller**
  (thrust ∝ n²), and the **servo** (0.73 s delay + 0.90 s lag).
- Parameter table with meanings; resulting behavior (cruise 0.558 m/s at 6
  rpm-units; about 10.4°/s yaw rate at full rudder; seconds to build a turn).
- Numerical integration (RK4 sub-steps; 5 physics steps per decision).
- Domain randomization: a new physically consistent vessel each episode.
- ▶ **Vessel simulator widget:** sliders for rudder command and rpm-units;
  live top-down trajectory plus time plots of u, v, r and actual vs commanded
  rudder (showing the delay and lag). Buttons for a step rudder, a zig-zag and
  a turning circle. Implement the facts section 1 equations directly (RK4, 0.05 s).
- ▶ **Rudder step response:** commanded vs actual rudder angle with delay/lag
  sliders.
- Sources: Fossen (2021); MMG method (Yasukawa & Yoshimura, 2015 [VERIFY]).

### 4. Classical methods
- Primers, each with one picture: **APF** (attraction/repulsion, the local-
  minimum trap — Paper 2's comparator, context only); **LOS guidance** (aim at
  a point Δ ahead on the path); **PID** (P reacts to error, I to accumulated
  error, D to its rate); **velocity obstacle** (the set of velocities that lead
  to collision within a horizon, drawn as a cone).
- **LOS-PID + DWA** (facts section 2.2): candidate set, closed-loop prediction through
  the vessel model, admissibility, objective G, fallback; "knows obstacles, not
  rules".
- **COLREGs-VO** (facts section 2.3): candidates, own-ship turn model, hard velocity
  obstacle, open-water classification, the starboard constraint
  (cross product), stand-on rule, cost J, the drop-the-rule fallback; and the
  narrow-channel variant.
- ▶ **LOS widget:** drag the vessel; change the look-ahead Δ; see the desired
  heading.
- ▶ **DWA widget:** a vessel, a panel and a target; show all candidate
  trajectories colored by score, crossed out if inadmissible; highlight the
  winner; sliders for the four weights.
- ▶ **VO widget:** own ship and one target; draw the velocity-obstacle cone and
  the starboard half-plane; drag the own velocity; show "allowed / forbidden by
  collision / forbidden by COLREGs".
- Comparison table: what each method knows, assumes and cannot do.

### 5. Reinforcement learning
Build up from zero:
1. The agent–environment loop; state, observation, action, reward; episodes.
2. Return and discount: G_t = Σ γ^k r_{t+k}; γ = 0.951 per 0.5 s ≈ 10 s
   horizon. ▶ **Discount slider** showing how far ahead rewards still count.
3. Policies, value functions V and Q, the Bellman equation — each with
   "in words".
4. Learning by trial: exploration vs exploitation; function approximation with
   neural networks (one diagram of a small network).
5. Policy gradient and actor–critic (the actor proposes, the critic judges).
6. **On-policy vs off-policy** — diagram: on-policy uses fresh experience and
   discards it; off-policy stores experience in a replay buffer and reuses it.
   Trade-offs: stability vs sample efficiency.
7. **PPO:** probability ratio, clipped surrogate objective, advantage via GAE.
   ▶ **Clip widget:** plot the PPO objective vs the ratio for positive and
   negative advantage with an ε slider.
8. **RecurrentPPO:** why memory helps when one frame does not tell the whole
   story (partial observability; e.g. a target briefly occluded); LSTM gates
   in one diagram.
9. **SAC:** maximum-entropy objective (reward + α × randomness), soft Q
   targets, twin critics (minimum to curb over-estimation), target networks,
   automatic α.
10. **TQC:** learn the whole distribution of returns as quantiles, pool
    critics, drop the top quantiles to curb over-estimation. ▶ **Quantile
    widget:** a return distribution as 25 × 2 atoms; slider for how many top
    atoms to drop; show the resulting estimate.
11. Comparison table of the four (on/off-policy, memory, exploration,
    over-estimation control, sample efficiency, cost per step).
12. How this study trains them: facts section 3 (identical network, defaults,
    2.5 M steps, 3 seeds, development-set checkpoint selection, why the test
    suite is touched once). ▶ **Network diagram** of the two-encoder
    architecture with the six observation branches.
- Sources: Sutton & Barto (2018); Schulman et al. (2016, 2017); Haarnoja et
  al. (2018); Kuznetsov et al. (2020); Hochreiter & Schmidhuber (1997);
  Raffin et al. (2021).

### 6. LiDAR perception — **the most important technical section**
The paper's Perception subsection currently lacks the governing equations;
this section must supply them and show how the steps chain together (facts section 4,
equations file section 4):
1. The sensor: 720 beams over 360°, 1–16 m, dead zone; noise model.
   ▶ **Scan viewer:** basin, panels, a moving target, the rays; toggles for
   pose noise and beam dropout.
2. Gating against the map polygon.
3. Segmentation — adaptive breakpoint: the threshold ε + ρΔθ and why it grows
   with range. ▶ slider for ε showing over/under-segmentation.
4. Ego-motion compensation: body scan → world points with the estimated pose
   (and why pose error makes static objects appear to move).
5. Association: nearest neighbour inside a 0.70 m gate; confirm after 2 hits,
   drop after 3 misses. ▶ animation of gates over two scans.
6. **Kalman filter** — predict/update equations, each matrix explained;
   covariance as "uncertainty ellipse". ▶ **KF widget:** noisy centroid
   measurements of a crossing target; the estimate and its velocity converge;
   sliders for σ_a and σ_z.
7. **Static vs moving:** why a speed threshold fails (▶ animation: the visible
   face of a static panel changes as the vessel passes, so its centroid
   "moves"), and the free-space test (appear/vacate over 2 s, thresholds,
   hysteresis). ▶ **Free-space widget:** two scans 2 s apart over a static
   panel and a moving target; mark appear/vacate points; show the verdict.
8. From track to encounter: CPA, domain distance, CRI, classification bands,
   engagement and latching (link to the CPA widget in section 2). Show the state
   machine idle → engaged → clearing → idle.
9. **Sector pooling:** the 27 sectors; max, min and feasibility pooling with
   equations and the algorithm in steps. ▶ **Pooling widget:** one sector with
   an obstacle field; a slider moves a gap narrower/wider than the 0.80 m safe
   width; show all three pooled ranges and the closeness value; explain which
   is right and why.
- Sources: Kalman (1960); Bar-Shalom et al. (2001) [VERIFY]; Borges & Aldon
  (2004) [VERIFY]; Meyer et al. (2020a); Paper 2; Waltz & Okhrin (2023);
  Helgesen et al. (2022) for why tracking quality depends on the environment.

### 7. The learning problem (MDP formulation)
Use `FORMULATION_EQUATIONS.md` exactly.
- Observation: the six branches (70 values) as a labelled diagram, each
  feature with its formula and normalization; the presence bit; the context
  branch and "one module, two consumers". ▶ **Live observation panel:** during
  a replayed scenario, show the 70 values as bars grouped by branch.
- Action: rudder and rpm mapping; why continuous propulsion (Rule 8(e)).
- Termination: goal, collision, timeout.
- Reward: total (Eq. 13), then each term with its own small plot:
  ▶ **Reward explorer (the "ablation" view):** for every term, a plot of the
  term against its driving variable with sliders for its parameters, plus a
  panel "what happens if this is larger/smaller, and why this value" using the
  design rationale below. Include a combined view: pick a scripted maneuver
  (compliant turn, wrong-way turn, slow-down, no action) in a head-on or
  crossing; show the per-step contributions stacked over time and the
  discounted total; changing weights updates the totals. Label it clearly as a
  reward-landscape illustration, not trained-policy results.

  Design rationale to present (all from the project's development findings):

  | Parameter / choice | Why this value | If larger / smaller |
  |---|---|---|
  | Bounded terms, weight = max per-step contribution | Paper 2's obstacle term was ~49× weaker than path following at contact because of a hidden scale | unbounded terms let one term silently dominate |
  | w_bnd = 15, the largest dense weight | slowing must always cost less than touching the boundary | smaller: the policy may graze the wall to keep speed |
  | Collision −300 ≫ goal +100 | a compliant collision must be worse than a non-compliant near miss | smaller: risk-taking pays |
  | Path term width-normalized, exp(−4 ẽ²) | Paper 2's exp(−0.05·e_y) varied < 10 % over a 10 m channel — no gradient | un-normalized: the term becomes a constant offset |
  | Speed gate in r_pf, overspeed gate above 1.2 U_ref | stopping must not score a perfect path; speeding must not pay | no gate: a stationary vessel on the line scores well |
  | Progress telescopes (Σ r_prog constant) | a lawful slowdown costs no progress reward | distance-to-goal form penalizes bends |
  | r_obs shifted exponential, d_oa = 0.6 m, zero beyond 2 m | the unshifted form adds a constant background penalty far from any obstacle (about −53 over an episode in the original design) that carries no gradient | larger decay: obstacles repel from far, the vessel wanders |
  | d_safe = 0.35 m boundary band | penalty only where contact is imminent | larger: narrow channels become uniformly penalized |
  | COLREGs group clipped to [0, 1], w = 9 | two simultaneous violations cannot out-rank safety terms | unclipped: rule terms can exceed w_dom |
  | Proximity gate ρ | rule penalties grow only as the pass gets close | no gate: distant encounters penalized |
  | v_port weighted by peak ρ since engagement | a wrong-way swerve opens DCPA and would otherwise discount its own penalty | current ρ: wrong-way swerves become cheap |
  | v_port held-heading form (5° dead band, 20° scale) | charging only yaw rate let a wrong-way heading, once reached, cost nothing | yaw-rate only: policies hold the wrong heading |
  | v_r8 deficit-based (a_req = Δy_req/d_req) | when the target is already clear (9(a) channel-keeping), no alteration is owed | fixed "always alter": the policy learns to always swerve |
  | v_hold speed part keeps rising (up to 3×) | saturating at 0.10 m/s made fleeing an overtaker free | saturating: the policy outruns overtakers |
  | Smoothness free window (σ = 0.25 for 4 steps after engagement) | Rule 8(b) wants one large alteration, not many small ones | no window: the committed turn is too costly |
  | Carve-outs R-2 (U_eff = 0.4 U_ref) and R-5 (hold astern) | without them the lawful Rule 8(e) slowdown is structurally unlearnable; gated so "always slow" is not rewarded | ungated: the degenerate always-slow policy |
  | Slowdown credited only if stopping clears (≥ 1.76 m) | stopping in a reciprocal target's path preserves the conflict | unconditional: the policy stops in front of head-on targets |
  | Verified property | from one engagement state the compliant 30° crossing turn earns +61 (discounted) over the wrong way, from either side | — |

### 8. Scenarios, curriculum and data sets
Use facts section 6.
- Workspace generation (basin legs, channels, path offset).
- ▶ **Backward-solve widget:** sliders for crossing angle c, speed ratio k,
  time to CPA T₀, miss distance d₀ and side; draw the spawn point, both tracks
  and the CPA; show the validation checks passing or failing.
- Static obstacles and the A* feasibility check (▶ grid view with inflation).
- ▶ **Space-time reachability animation:** the reachable set growing each
  0.5 s while cells near the moving target are removed; solvable or not.
- Curriculum: ▶ **timeline chart** of the seven stages (what is added when).
  Stages 6–7: more, and more evenly spread, three-obstacle layouts in three
  motifs (gallery), CPA guard relaxed then removed, varying-speed targets.
- Data sets: training / development / three-obstacle development / frozen test
  suite / robustness set / width sweep / perception degradation, with a
  seed-namespace diagram showing they never overlap; checkpoint selection
  score.

### 9. Safety layer and evaluation design (added)
- The runtime stop safety layer (facts section 5): trigger, maneuver, hand-back; off in
  training; learned compliance always reported with it off.
- What is measured: success; collisions by type; cross-track error; rule
  metrics per encounter; minimum CPA distribution; intervention rate; by
  target behavior; by width; by perception quality. No values.

### 10. Field trials and the vessel–computer link
- Paper 2's basin trials (platform, basin, panels, procedure) from the Paper 2
  PDF.
- ▶ **System diagram and packet animation:** vessel onboard computer ⇄ UDP ⇄
  shore laptop; START registration; pose/LiDAR lines arriving; frame
  assembly; policy; `$CMD,S1,S2` back.
- Decoding step by step (facts section 7): message formats, frame change (−Y, X,
  −Yaw), range units, beam order and rotation, pose–scan pairing, velocity
  from pose differencing.
- **Observation conversion updated to Paper 3** (facts section 7 table): the same
  perception modules run on the real scan to produce the 70-value
  observation; contrast with Paper 2's observation.
- Action mapping to S1/S2 (sign reversal; S2 = n/24 × 100), shadow mode,
  operator stop latch, logging and replay.
- State clearly that Paper 3's physical encounter trials are planned.

### 11. Nomenclature, abbreviations and references

## 4. Visual style
Clean, light background with a dark-mode toggle; one accent color per
subsystem (vessel, perception, decision-making, testing) used consistently in
diagrams; monospace for code-like items; figures drawn to scale where they show
the basin (10 × 25 m, hull 1.725 × 0.5 m). Accessible contrast; widgets
usable on a laptop screen.
