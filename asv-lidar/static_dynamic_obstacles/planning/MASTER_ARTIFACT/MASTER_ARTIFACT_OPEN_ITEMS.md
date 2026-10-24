# Master artifact — answers to the open items in Sections 0–8 (2026-10-10)

Answers by ID to the list raised after Section 8. F and M answers are checked against the code
(`src/`, `bluefin/`, `configs/baseline_v3.json`); the facts file (`MASTER_ARTIFACT_FACTS.md`) was
updated the same day with every value below, so later sections can rely on it. Q1–Q9 were
confirmed as written (decision, 2026-10-10).

## F — flags

- **F1** Same as own ship: the target uses the same hull outline, 1.725 × 0.50 m (`src/targets.py`).
- **F2** 25 m, the basin's length.
- **F3** Confirmed: channel width is wall to wall, with no inset (the channel polygon is itself the
  navigable boundary; only the basin has the 0.40 m inset). Note for the room examples: in every
  stage that uses channels the width also varies along the channel by up to 1.8 times
  (W_max/W_min ~ U[1.0, 1.8], piecewise linear through 2–4 control points), with a straight path.
- **F4** Confirmed: σ = −1 for a crossing from port (decision A17, `src/colregs/context.py`): give
  way by a port alteration, passing astern.
- **F5** An asymmetric ship domain in the sense of Goodwin (1975), compressed for the basin
  (Szlapczynski & Szlapczynska, 2017, review): fore 2.0 L_pp = 3.14 m, aft 1.0 L_pp = 1.57 m,
  abeam max(0.75 L_pp, 1.25 m) = 1.25 m with L_pp = 1.57 m. The abeam value is floored at the
  LiDAR's 1.0 m minimum range plus half the beam (1.0 + 0.25 m), because a domain smaller than
  the sensor can resolve would be a blind spot. The proportions are provisional, pending the
  turning-circle measurements. d_req = 2 × 1.25 = 2.5 m.
- **F6** Both texts are right in their own convention; the facts file had the mapping wrong. In
  the identified equations δ is positive to port, so the commanded angle is δ_c = −40° × a_δ:
  a positive policy command turns the vessel to starboard, as Section 4's steering laws assume.
  Change δ_c = 40° × a_δ to δ_c = −40° × a_δ and add one sentence on the sign.
- **F7** Confirmed t_b = 0; u_b (0.5 m/s) is then inert and can be dropped. Thrust is
  T = T₁₂ (n/12)².
- **F8** The identified servo rate limit is about 2,985°/s, so it never binds. The field
  bridge's 20°/s command limiter (50 % of full rudder per second) exists on the vessel but is
  off in simulation.
- **F9** 12 rpm-units. With the nominal parameters the steady yaw rate at full rudder is about
  10°/s at 12 rpm-units (1.12 m/s) and about 5°/s at the 6 rpm-unit cruise (speed falls to
  about 0.46 m/s in the turn). Replace "10.4°/s" with "about 10°/s at 12 rpm-units; about 5°/s at
  cruise". (10.4 survives only as the DWA comparator's feed-forward constant, which scales its
  candidate set rather than modelling the plant; checked 2026-10-10,
  `results/classical/dwa_yaw_gain_check.md`.)
- **F10** θ_A and θ_B: two of the 28 bootstrap solutions, picked uniformly at random with
  replacement. λ ~ U(0, 1). Jitter: Gaussian, standard deviation 5 % of each parameter's
  bootstrap standard deviation (not 5 % of its value), scale 1.0; each parameter keeps its sign.
- **F11** Verified: Yasukawa, H. & Yoshimura, Y. (2015). Introduction of MMG standard method for
  ship maneuvering predictions. *J. Mar. Sci. Technol.* 20(1):37–52. Tobin, J. et al. (2017).
  Domain randomization for transferring deep neural networks from simulation to the real world.
  *IROS 2017*, 23–30.
- **F12** smoothness = |r_d − r_d,prev| / (18 °/s) when both the candidate and the last choice are
  yaw-rate set-points; 1 when one of them is plain LOS-PID; plus |speed fraction − last speed
  fraction|.
- **F13** Confirmed: U_pref = U_nom = 0.558 m/s.
- **F14** J_prox = max(0, 0.60 m − c_min) / (0.60 m − 0.15 m), where c_min is the candidate's
  smallest clearance to scan points and the boundary over 10 s: a soft margin from the 0.15 m
  hard limit up to 0.60 m.
- **F15** Correct the first assumption: fixed obstacles and the boundary are judged over the
  first 5.5 m of travel (the same distance for every candidate, not the whole prediction), and
  the tracked target over the first 10 s; the clearance score uses the same windows. The second
  is right: the plain LOS-PID candidates run at full and half U_nom.
- **F16** Remove the narrow-channel VO variant from Section 4: the encounter-specific VO is not a
  Paper 3 comparator (decision 2026-10-08). The comparators are LOS-PID + DWA and COLREGs-VO.
  *Superseded 2026-10-20 (decision S18):* the encounter-specific VO is the reported VO
  comparator and COLREGs-VO only drives the reactive target (facts section 2.4).
- **F17** Verified: Khatib, O. (1986). Real-time obstacle avoidance for manipulators and mobile
  robots. *Int. J. Robotics Research* 5(1):90–98. Fiorini, P. & Shiller, Z. (1998). Motion
  planning in dynamic environments using velocity obstacles. *Int. J. Robotics Research*
  17(7):760–772.
- **F18** Confirmed: target entropy −2 (= −dim of the action), the library default; the config
  sets only the automatic temperature and does not override the target.
- **F19** Scene encoder: 42 inputs (lidar 27, boundary 7, ego 3, path 3, previous action 2) →
  Linear 128 → ReLU → Linear 128 → ReLU, giving the 128 outputs. (Slot encoder: 28 → 64 → 64 →
  32, ReLU, gated by the presence bit; features 128 + 32 = 160.)
- **F20** Confirmed as the library default: one 256-unit LSTM layer for the actor and a separate
  one for the critic (shared_lstm = False, enable_critic_lstm = True), placed after the features
  extractor and before the 256 × 256 heads.
- **F21** None: ranges are exact ray casts with no per-beam range noise (and no isolated beam
  dropout) in training, both left at zero until characterized from field data. Noise enters
  through the pose (0.03 m / 0.2°), surge (0.05 m/s) and yaw-rate (1°/s) measurements.
- **F22** Not a single rate: the planned study sweeps detection dropout from zero up to the
  point of track loss. Keep 20 % in the scan viewer, labeled illustrative.
  *Superseded 2026-10-14:* the perception study is not part of the paper; the 20 % stays,
  labeled illustrative, with no reference to a planned study.
- **F23** Verified: Borges, G. A. & Aldon, M.-J. (2004). Line extraction in 2D range images for
  mobile robotics. *J. Intelligent and Robotic Systems* 40(3):267–297. Bar-Shalom, Y., Li, X. R.
  & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.
  Petrovskaya, A. & Thrun, S. (2009). Model based vehicle detection and tracking for autonomous
  urban driving. *Autonomous Robots* 26(2–3):123–139.
- **F24** Confirmed: engaged only. Every COLREGs term returns zero unless the encounter is
  engaged, so returning to the path after the target has passed (clearing) is never charged.
- **F25** Normal(mean +0.12, standard deviation 0.15), clipped to [−0.30, 0.30], as a fraction of
  the half-width of the narrowest section; positive = starboard (the Rule 9(a) station); drawn
  once per channel and constant along the straight path. The curriculum brief's "along the
  centerline" was wrong and is corrected.
- **F26** The null class has no T₀: it is placed, not solved backward. Range U[8, 15] m at a
  bearing of 20–160° to either side, course within ±10° of the own ship's, speed ratio
  0.85–1.15; accepted only if its CPA is at least 4 m or does not occur within the 90 s episode.
- **F27** The study's function: the smallest centre-to-centre miss distance at which the
  target's hull outline, moving along the straight relative track (relative course c, speed
  ratio k), never touches the own hull drawn as a rectangle inflated by 0.15 m each side
  (2.025 × 0.80 m). Found by bisection between 0 and 4 m (14 halvings, 2 cm steps along the
  track), cached per 0.5° of c and 0.05 of k. Floor = that + 0.35 m; being-overtaken d₀ is
  uniform on [floor, max(2.0, floor + 0.5)] m. A computation from the hull outlines matches if
  it uses these two shapes and the 0.15 m margin.
- **F28** Each crossing draw first decides whether it is to be unescapable (probability 0.20),
  then its kinematics are redrawn until a scripted check agrees. The check: hold course at
  cruise for 1.5 s, then either coast to a stop or commit a 60° alteration in the required
  direction (starboard for a target from starboard, port for one from port); escapable if
  either keeps the hulls (0.15 m margin) apart and the own hull inside the water until 6 s past
  the drawn TCPA.
- **F29** Spawn range R₀: head-on 13.8–19 m (beyond the 12 m effective LiDAR range); crossing
  5–10 m; overtaking 6 m to min(12, 16(1 − k)) m; being overtaken 4–6 m astern (in the basin the
  own ship starts 6.9 m along the leg); null 8–15 m. The target is present from the first step
  and is acquired by the tracker as it comes into range.
- **F30** 25 %: stages 6 and 7 keep stage 5's channel probability (0.25) for head-on, crossing
  and overtaking episodes that are not three-obstacle layouts; the three-obstacle layouts are
  always in the basin.
- **F31** With the own ship at the Rule 9(a) station a quarter-width from the starboard wall, a
  give-way alteration needs one abeam semi-axis of starboard room:
  W_crossing ≥ 4 (a_abeam + w_wall) = 4 (1.25 + 0.65) = 7.60 m. Head-on needs two vessels
  abreast: W_head-on ≥ 2 a_abeam + 2 w_wall = 2.50 + 1.30 = 3.80 m. Overtaking adds a pose-drift
  margin over the time the vessels stay abreast: W_overtaking ≈ 3.80 + 2 ρ_pose t_exp =
  3.80 + 2 × 0.0101 m/s × 22.8 s = 4.26 m, with t_exp = (a_ahead + a_astern + L_pp)/(U_OS − U_TS)
  at a speed ratio of 0.5 (w_wall = 0.65 m).
- **F32** Coupled test episodes come from dedicated blocks at 310,000–349,999, 450,000+ and
  600,000+, used by no training or validation draw. (The coupled validation episodes are at
  360,000+; the facts file said 350,000+, which was an older block.)

## M — mismatches

- **M1** Follow Q4.
- **M2** Agreed: only Sections 9 and 10 remain.
- **M3** Agreed: replace with the new stance (COLREGs-aware; each rule stated as what the reward
  encodes; behavior shown by trajectory snapshots of each encounter type; no legal
  certification claimed).
- **M4** Agreed, all three points. Wording: "3 × 10⁶ steps (seven stages over a 2.5 × 10⁶-step
  base budget, with stage 7 continued to 3 × 10⁶)"; "validation set"; the flag notes that draft 4
  still says two million steps.
- **M5** Agreed: the headline evaluation is test set v4, 1,000 episodes (664 decoupled + 336
  coupled); the width sweep is marked planned.
- **M6** Agreed, all three points. Training targets hold their course and never give way; in
  stage 7, 30 % of the three-obstacle layouts' targets change speed once. Reactive and
  noncompliant targets are in test set v4 as well as the robustness set. Runtime safety layer:
  off in training and in every validation-set evaluation; learned behavior is always reported
  with it off.
- **M7** Agreed: "safety and encounter-consistent behavior".
- **M8** Agreed: reword to safety-layer intervention reported separately from learned
  behavior, and mark the width sweep and perception study as planned.
- **M9** Agreed: shown by trajectory snapshots instead of judged by declared metrics.
- **M10** Agreed, all four. Rule 8: a 20° alteration or a 30 % slowdown counts as one full
  action, passing at 2.5 m, time pressure as TCPA falls below 15 s. Rule 14: port-to-port; no
  alteration needed when keeping to the starboard side already clears the target; "required
  turn" (Q1). Crossing: name the port alteration for a target from port. Rule 17(a)(i): hold
  course and speed, released in extremis (DCPA < 2.5 m and TCPA < 5 s). Close with the
  COLREGs-aware line.
- **M11** Agreed: "validation set" in all three places.
- **M12** Agreed, all five.
- **M13** Use one wording everywhere: the validation set is used to select the learners'
  checkpoints, to tune the classical comparators (coordinate descent, scored by goal −
  2 × collision) and for the learning curves; it is never used for reported results. "Only to
  select checkpoints" was too narrow, because the comparators are tuned on it too. Brief rule
  11, the facts file and the curriculum brief now say this.
- **M14** "noncompliant" throughout.

## Q — decisions (confirmed 2026-10-10)

- **Q1** (b): "required turn sense", "required side", "required turn", and reword the two
  rationale rows. The code name `compliant_turn_sense` stays in code.
- **Q2** "rule-consistent" ("lawful" implies a legal claim the paper does not make).
- **Q3** Follow the brief and facts. Draft 4 is older and will be updated to the same terms.
- **Q4** Rewrite each pair as a short summary under the "In this section" label: the brief asks
  for a summary, and a statement reads better than a list of learning goals.
- **Q5** Use the real values: null as in F26 (no T₀; placed at 8–15 m, CPA ≥ 4 m or none within
  90 s); channels 25 % in stages 6 and 7 (F30); slalom: three panels alternating sides, each
  0.3–3.2 m off the leg, gate panels 1.3–2.7 m either side, side panels 1.2–3.2 m, on-path panels
  within 0.5 m, along 28–85 % of the leg, at least 0.4 m apart; the target is present from the
  first step at the class spawn range (F29); being-overtaken draws are uniform from the floor
  upward, never raised to it (F27).
- **Q6** Keep the facts' generator band and add the sentence: a crossing drawn in the 5° gaps
  either side of the head-on band is re-classified as head-on, fails the re-check and is
  redrawn. Also correct the generator bands for the near-parallel classes: overtaking, being
  overtaken and null are drawn within ±10° of the own ship's course (the classifier accepts up
  to ±67.5° and ±20°).
- **Q7** Keep: they are the code's seed namespaces (training 0–99,999; validation decoupled
  200,000–209,999; frozen benchmark 300,000–309,999; width sweep 400,000–409,999; perception
  study 500,000–509,999). *Superseded 2026-10-14:* the width sweep and the perception study
  are not part of the paper; their blocks stay reserved in the code and are not shown as
  planned data sets (`CURRICULUM_AND_DATA_BRIEF.md`).
- **Q8** First and third assumptions right: the decoupled pool is the frozen benchmark under
  three target behaviors (constant velocity, reactive and, in head-ons, noncompliant: 340 / 258
  / 66 of the 664), and trimming came before the fairness check. The second needs correcting:
  twins were collapsed in both pools (behavior variants in the decoupled pool; fixed- and
  varying-speed twins, and no-target repeats, in the coupled pool), each underlying scenario
  kept once with its variant rotating. Spell out the 8 cells: basin head-on, crossing,
  overtaking, being overtaken and null; channel head-on, crossing and overtaking.
- **Q9** Keep both, now verified: Bengio, Y., Louradour, J., Collobert, R. & Weston, J. (2009).
  Curriculum learning. *ICML 2009*, 41–48. Haarnoja, T. et al. (2018b). Soft actor-critic
  algorithms and applications. arXiv:1812.05905 (the automatic entropy tuning behind F18).
