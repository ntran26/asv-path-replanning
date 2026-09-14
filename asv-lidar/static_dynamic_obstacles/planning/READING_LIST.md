# Reading list — the methods behind Paper 3

**What this is.** The papers and books you need to understand every method this
project implements, in the order that makes each one readable. It has two kinds of
entry:

* **cited** — named in the planning documents or the code, with the venue those
  documents give;
* **foundation** — a method the code implements without naming a source. These
  are the standard references for it.

**Where it's used** points to the file or specification that relies on each
entry, so you can read the paper next to the code.

**Verify** marks a detail — a title, a volume, sometimes a year — that the
project documents do not give and I have not checked against the publication.
Confirm these before citing.

**Priority.** ★ marks the twelve to read first. They cover what a reviewer will
ask about: the rules, the vessel model, the encounter geometry, the RL algorithm,
and the direct prior work.

---

## 1. The rules being learned

| | Reference | Kind | Why, and where it's used |
|---|---|---|---|
| ★ | IMO (1972). *Convention on the International Regulations for Preventing Collisions at Sea (COLREGs)*. Rules 6, 8, 9, 13–17 in particular | foundation | Everything in `src/colregs/`, `src/reward/terms.py` (`v_port`, `v_bow`, `v_side`, `v_hold`, `v_r8`), `src/emergency_stop.py` (Rule 8(e)). Read Rule 9 (narrow channels) and Rule 17 (stand-on vessel) closely: A15, A17 and A18 turned on them |
| ★ | Burmeister, H.-C. & Constapel, M. (2021). Autonomous collision avoidance at sea: a survey. *Frontiers in Robotics and AI* 8 — **verify venue** | cited (`PAPER3_DRAFT_SKELETON.md`) | The gap claim: of 48 surveyed papers, four mention Rule 9 and two address it |
| | Hansen et al. (2022). *IFAC-PapersOnLine* 55(31):222–228 — **verify title** | cited (`PROJECT_BRIEF.md`) | Rule 9 manoeuvrability assessment in confined water |
| | de Vries et al. (2022) — **verify full reference** | cited (`PAPER3_DRAFT_SKELETON.md`) | Autonomous navigation in urban canals |

## 2. The vessel model

| | Reference | Kind | Why, and where it's used |
|---|---|---|---|
| ★ | Fossen, T. I. (2002). *Marine Control Systems: Guidance, Navigation and Control of Ships, Rigs and Underwater Vehicles*. Marine Cybernetics. (Later editions: *Handbook of Marine Craft Hydrodynamics and Motion Control*, Wiley, 2011 and 2021) | cited (`bluefin_modelling/bluefin_model_derivation_explanation.md`, with page references) | Reference frames, the 3-DOF manoeuvring model, added mass, damping, actuator forces. The scaffold for `src/ship.py` and `bluefin/`. Chapters 2, 3.2, 3.5 and 7.5 |
| | Yasukawa, H. & Yoshimura, Y. (2015). Introduction of MMG standard method for ship maneuvering predictions. *Journal of Marine Science and Technology* 20:37–52 | foundation | The MMG hull-force form (`-N_r·u·r`, `-Y_v·u·v`) that `bluefin/REPORT.md` §1 adopted, and the inert MMG block in `Blue02.m` |
| | Skjetne, R., Smogeli, Ø. & Fossen, T. I. (2004). Modeling, identification, and adaptive maneuvering of CyberShip II: a complete design with experiments. *Modeling, Identification and Control* 25(1):3–27 | cited (`PROJECT_BRIEF.md`) | System identification of a model-scale vessel from experiments — the precedent for `bluefin/fit_final.py` and the basin plan (`PART2_BASIN_PLAN.md`) |
| | Fossen, T. I., Breivik, M. & Skjetne, R. (2003). Line-of-sight path following of underactuated marine craft. *IFAC Proceedings* 36(21) | foundation | LOS guidance: the look-ahead course error in `src/path.py` and `r_pf`, the scripted follower in `tools/scale_audit.py`, and the LOS-PID comparator |
| | Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall | foundation | The 28 run-level bootstrap resamples behind the parameter intervals (`bluefin/REPORT.md`), and 04's bootstrap 95 % CIs over seeds |

## 3. Perception: LiDAR to target tracks

| | Reference | Kind | Why, and where it's used |
|---|---|---|---|
| | Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering* 82(1):35–45 | foundation | The constant-velocity Kalman filter per track (`src/tracking.py`, step 5) |
| | Bar-Shalom, Y., Li, X. R. & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley | foundation | Nearest-neighbour data association and why it suffices over JPDA at one or two targets (`src/tracking.py`, `Tracker`), gating, track management |
| | Han, J. et al. (2020). *Journal of Field Robotics* 37(6):987–1002 — **verify title** (Autonomous collision detection and avoidance for the ARAGON USV) | cited (`01_PERCEPTION_AND_OBSERVATION.md` §7) | A field-verified LiDAR/radar track pipeline feeding COLREGs manoeuvres — the benchmark for this project's perception-to-action chain |
| | Kim et al. (2022). *Ocean Engineering* — **verify full reference** | cited (`01` §7) | 2D LiDAR detection on a physical catamaran ASV, simulation and experiment |
| | Villa, J., Aaltonen, J. & Koskinen, K. T. *IEEE/ASME Transactions on Mechatronics* — **verify year and title** | cited (`PROJECT_BRIEF.md`) | LiDAR-based path following in harbour conditions |

## 4. Encounter geometry: CPA, collision risk, domains, classification

| | Reference | Kind | Why, and where it's used |
|---|---|---|---|
| ★ | Waltz, N. & Okhrin, O. (2023). Spatial–temporal recurrent reinforcement learning for autonomous ships. *Neural Networks* 165:634–653 — **verify title** | cited throughout (`01` §5, `04`, `src/cpa_cri.py`, `src/encounter.py`) | **The single most-used source.** CPA/CRI (§3.3), the encounter classification table (§4.3), the "Around the Clock" benchmark (`src/suite.py`), and the observation idea of giving the policy the encounter class. `01` says to read it in full |
| | Xu et al. (2020) — **verify full reference** | cited (via Waltz & Okhrin Table 1; `src/encounter.py`, `src/constants.py`) | Baseline bearing and heading thresholds for the encounter classes |
| ★ | Goodwin, E. M. (1975). A statistical study of ship domains. *Journal of Navigation* 28(3):328–344 | foundation | The ship-domain concept behind `DOMAIN_FORE/AFT/LATERAL`, `r_dom`, `d_req` and the admissibility test (`src/colregs/geometry.py`) |
| | Szlapczynski, R. & Szlapczynska, J. (2017). Review of ship safety domains: models and applications. *Ocean Engineering* 145:277–289 | cited (`src/constants.py`) | Which domain shape to use and why the compressed asymmetric domain was chosen |
| | Chun, D.-H. et al. (2021). Deep reinforcement learning-based collision avoidance for an autonomous ship. *Ocean Engineering* 234 — **verify** | cited (`01` §5, `PAPER3_DRAFT_SKELETON.md`) | The 3·Lpp domain that does not fit a 1.57 m vessel in a 10 m basin; multi-ship handling |

## 5. Reinforcement learning

| | Reference | Kind | Why, and where it's used |
|---|---|---|---|
| ★ | Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. Chapters 3, 9, 13 | foundation | MDPs, discounting (why `discount()` converts γ between 10 Hz and 2 Hz), value functions, policy gradients |
| ★ | Schulman, J., Wolfe, S., Dhariwal, P. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347 | foundation | The algorithm in `src/train_formulation.py`: clipping, `target_kl`, epochs (F51). Explains run 1's KL runaway |
| | Schulman, J., Moritz, P., Levine, S. et al. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR* | foundation | `gae_lambda = 0.95` |
| | Haarnoja, T., Zhou, A., Abbeel, P. & Levine, S. (2018). Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML* | foundation | The SAC baseline (`src/train_sac_baseline.py`) and Papers 1–2's primary algorithm |
| | Raffin, A. et al. (2021). Stable-Baselines3: reliable reinforcement learning implementations. *JMLR* 22(268):1–8 | foundation | The implementation actually run — `VecNormalize`, `SubprocVecEnv`, `MultiInputPolicy` |
| | Ng, A. Y., Harada, D. & Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. *ICML* | foundation | Why a telescoping progress term (`r_prog`, R-9) shapes without changing the optimum — and why the carve-outs (R-2, R-5) are shaping decisions |
| | Bengio, Y., Louradour, J., Collobert, R. & Weston, J. (2009). Curriculum learning. *ICML* | foundation | The scenario and propulsion curricula (`src/curriculum.py`, `STAGE_SCHEDULE`) |

## 6. Deep RL for ship collision avoidance — the direct prior work

| | Reference | Kind | Why, and where it's used |
|---|---|---|---|
| ★ | Meyer, E., Robinson, H., Rasheed, A. & San, O. (2020). Taming an autonomous surface vehicle for path following and collision avoidance using deep reinforcement learning. *IEEE Access* 8 | cited (`02`, draft §2) | The path-relative observation with rangefinder sensing this project descends from, and the Rule 18 route that S3 rejected |
| ★ | Heiberg, A. et al. (2022). Risk-based implementation of COLREGs for autonomous surface vehicles using deep reinforcement learning. *Neural Networks* 152:17–33 | cited (`PROJECT_BRIEF.md`) | Risk-index-based COLREGs rewards — the nearest ancestor of the `r_col` group |
| | Woo, J. & Kim, N. (2020). Collision avoidance for an unmanned surface vehicle using deep reinforcement learning. *Ocean Engineering* 199 | cited (draft §2) | The standard DRL-USV reference |
| | Zhao, L. & Roh, M.-I. (2019). COLREGs-compliant multiship collision avoidance based on deep reinforcement learning. *Ocean Engineering* 191 | cited (draft §2) | Multi-ship handling; slot-based observations |
| | Sawada, R., Sato, K. & Majima, T. (2021). Automatic ship collision avoidance using deep reinforcement learning with LSTM in continuous action spaces. *Journal of Marine Science and Technology* 26 | cited (draft §2) | The Imazu benchmark tradition, dropped here as open-water (D8), and recurrence (`USE_RECURRENCE`) |
| ★ | Waltz, N., Paulig, N. & Okhrin, O. (2025). *Expert Systems with Applications* 274:126933 — **verify title** | cited (`PROJECT_BRIEF.md`, draft §2) | DRL on inland waterways — the closest prior work to confined-water COLREGs |
| | Your Papers 1 (ICMCR 2026) and 2 (MDPI *Drones*) | cited | LiDAR sector pooling, the staged curriculum, sim-to-field validation — the base this code was ported from (`PORTING_MANIFEST.md`) |

## 7. Formalising the rules, and safe RL

| | Reference | Kind | Why, and where it's used |
|---|---|---|---|
| ★ | Krasowski, H. & Althoff, M. (2024). Provable traffic rule compliance in safe reinforcement learning on the open sea. *IEEE Transactions on Intelligent Vehicles* 9(12):7617–7634 | cited (`02`, `PROJECT_BRIEF.md`) | The model for turning COLREGs into checkable predicates — `v_*` terms, the admissibility predicate, the emergency-stop supervisor as a safety layer |

## 8. Classical comparators (and the reactive target model)

| | Reference | Kind | Why, and where it's used |
|---|---|---|---|
| | Fiorini, P. & Shiller, Z. (1998). Motion planning in dynamic environments using velocity obstacles. *International Journal of Robotics Research* 17(7):760–772 | foundation | The velocity-obstacle idea both VO comparators build on |
| ★ | Kuwata, Y., Wolf, M. T., Zarzhitsky, D. & Huntsberger, T. L. (2014). Safe maritime autonomous navigation with COLREGS, using velocity obstacles. *IEEE Journal of Oceanic Engineering* 39(1):110–119 | cited (`04` §5, draft §2) | The COLREGs-VO comparator (C3, not yet built) |
| | Thyri, E. H. & Breivik, M. (2022). Partly COLREGs-compliant collision avoidance for ASVs using encounter-specific velocity obstacles. *IFAC-PapersOnLine* 55(31) — **verify title** | cited (`03a` §5.3, `04`) | The encounter-specific VO comparator, **and** the `T-RE` reactive target model; confined-water domains |
| | Fox, D., Burgard, W. & Thrun, S. (1997). The dynamic window approach to collision avoidance. *IEEE Robotics & Automation Magazine* 4(1):23–33 | foundation | The LOS-PID + DWA comparator |
| | Gonzalez-Garcia et al. (2022) — **verify full reference** | cited (`04`, draft §2) | Optional NMPC comparator with LiDAR-based avoidance and physical experiments |

## 9. Simulation to reality

| | Reference | Kind | Why, and where it's used |
|---|---|---|---|
| | Tobin, J. et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IROS* | foundation | The idea behind Study 3 and hull randomisation |
| | Peng, X. B., Andrychowicz, M., Zaremba, W. & Abbeel, P. (2018). Sim-to-real transfer of robotic control with dynamics randomization. *ICRA* | foundation | Dynamics (not visual) randomisation — what `sample_params()` and `VESSEL_RANDOMISATION_SCALE` actually do |

---

## Suggested order

1. **The rules.** COLREGs Rules 8, 9, 13–17 (§1), then Burmeister & Constapel for the field's gaps.
2. **The vessel.** Fossen chapters 2 and 3.5, then Yasukawa & Yoshimura for MMG. That is enough to read `src/ship.py` and `bluefin/REPORT.md`.
3. **Encounter geometry.** Goodwin, then Waltz & Okhrin (2023) §3.3 and §4.3 in full. With these, `src/cpa_cri.py`, `src/encounter.py` and `src/colregs/` read straightforwardly.
4. **RL.** Sutton & Barto chapters 3 and 13, then PPO and GAE. That covers `train_formulation.py` and the run diagnostics.
5. **The reward design lineage.** Meyer et al. → Heiberg et al. → Krasowski & Althoff, alongside `02a_REWARD_SPECIFICATION.md`.
6. **The closest prior work.** Waltz, Paulig & Okhrin (2025), and the remaining §6 papers for positioning.
7. **Comparators and sim-to-real** (§8, §9) as those parts are built.
