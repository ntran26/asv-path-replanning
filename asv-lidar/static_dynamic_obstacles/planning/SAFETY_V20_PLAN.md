# Safety layer V20: committed stop-terminal contingency over V16

Date: 2026-10-10. Status: **phase 1, design only.** No V20 code exists, no episode has been run, and no controller, environment default, constant, checkpoint, configuration or completed ledger has been changed. Implementation and evaluation (phase 2) start only after approval of this plan.

Starting point: [complete safety record](SAFETY_LAYER_COMPLETE_SUMMARY.md) (sections 11-13 and 15), [evaluation protocol](SAFETY_EVALUATION_PROTOCOL.md), [reachability stage 1](SAFETY_REACHABILITY_STAGE1_PLAN.md) and its [saved-data results](../results/safety_dev/reachability_stage1/README.md). V16 is the reference candidate; V17, V18 and V19 are not inherited.

## 0. Decision in brief

**Method family:** backup-based safety filtering with a committed, recursively rechecked contingency that ends in a verified terminal set (model-predictive shielding, gatekeeper and fail-safe trajectory verification), with the terminal set chosen as *rest with the rudder centred* and the contingency checked against empirically calibrated forecast-error allowances. Braking inevitable-collision states and passive motion safety define what the contingency can and cannot promise around moving targets.

**What V20 changes:** V16 keeps making every decision it can check. V20 additionally certifies, at every decision, a short contingency that brings the vessel to rest clear of all modeled hazards, commits it, and uses certified commands *only in the two V16 branches that currently issue unchecked commands* (`last certificate` and `no escape`). In those branches V20 prefers SAC's command if it is certified, otherwise V16's command if it is certified, otherwise the nearest certified command to SAC, otherwise the committed contingency. V20 never hands an unchecked SAC command to the plant while a certified alternative or a committed contingency exists.

**Why this target:** a new read-only audit of the saved V16 cohorts ([report](../results/safety_dev/v20_development/phase1_saved_cascade_audit/REPORT.md)) finds that **21 of the 22 V16 contacts in the 72 development cases occur immediately after one or more unchecked decisions** (18 `no escape`, 3 `last certificate`; the remaining contact is the known 0.22 m blind-zone start). The same audit shows that the information available at V16's *first* intervention does not separate lost SAC successes from rescues, so V20 deliberately leaves V16's intervention timing unchanged.

**What V20 does not claim:** it does not restore the six lost SAC successes by construction, does not solve target-behavior mismatch, own-dynamics bias or missing obstacle evidence, and provides at most a conditional, contract-based safety statement (section 7). Its expected effect is fewer contacts in unchecked phases; whether goals rise or some contacts become timeouts is an empirical question for phase 2.

## 1. Fixed objects and identity checks performed

| Object | Check on 2026-10-10 | Result |
| --- | --- | --- |
| `runs/sac_formulation_seed0_bl3/kept_best_3M/best_model.zip` | SHA-256 | `993db156...c2bdc8`, equal to summary section 2.3 |
| Same `config.json` | SHA-256 | `a4df64a9...ed44`, equal |
| `configs/baseline_v3.json` | Formulation digest recomputed with `baseline_config.digest` | `03a9c0c22f53ee0b`, equal (the raw file SHA-256 is a different quantity) |
| `src/constants.py` | SHA-256 | `4f7fadaf...ce089`, **differs** from the summary's `3d71c76d...`. Commit `2e427e5` (2026-10-04, after the evidence cutoff) replaced section-sign characters in comments, docstrings and one error string; the abstract syntax tree is identical once string literals are blanked. The summary's hash equals the file at commit `fa5e004`. No runtime value changed. The file is not edited. |
| Runtime | Python 3.13 present; torch and stable-baselines3 absent; PyPI reachable; `download.pytorch.org` refused by the egress proxy; Python 3.11 installed | Phase 2 needs a pinned environment (section 10.1). Historical runs used Python 3.10 on Windows, so fresh OFF and V16 runs in the phase-2 environment are the only valid pairing references. |

## 2. New saved-data evidence that shapes the design

The audit ran no episodes. It reads the V16 traces named by the validated 32-case and 40-case paired reports (72 cases, 3,801 decisions) and hashes every input.

1. **Contacts follow unchecked branches.** V16 records two branches that issue a command whose plan its own checker did not pass at that decision: `last certificate` (171 decisions) and `no escape` (213). Of 22 contact episodes, 21 end after an unchecked run of 1-15 decisions (median 7, about 3.5 s). For the six lost SAC successes the run is 6-15 decisions long. In BAS-NU-CV-070 the last checked decision is `idle` at decision 64, followed by 14 `no escape` decisions toward the end wall, with braking excluded because no target is inside the traffic gate.
2. **Braking and rest do not separate losses from rescues.** Rescues brake (13/19), come to rest (11/19), follow failed stored plans (14/19) and pass through `no escape` (10/19) about as often as the six losses do (5/6, 5/6, 5/6, 6/6). Removing braking or forbidding rest is not supported.
3. **The first intervention is not separable without outcomes.** SAC followed by the proposal's own tail hard-passes the nominal checker at V16's first intervention in 2 of 6 losses (BAS-HO-NC-059, P2-L2-CRS-FIX-19) and in 8 of 19 rescues. A margin-relaxation rule would skip V16's first intervention in those eight rescues in order to address two losses, matching the earlier any-feasible-policy result (3/9 versus 6/9). In that variant P2-L1-CRP-VAR-12 contacts the target after a checked zero-margin decision, i.e. an optimistic target forecast.
4. **A held rudder at rest is not a rest state.** With the identified model, rest (u = v = r = 0, zero propulsion) with zero rudder angle has all derivatives exactly zero for every parameter vector, because thrust, damping, Coriolis and rudder-lift terms each vanish. A 35-degree rudder held at rest still yields lift through the 0.05 m/s minimum inflow floor in `bluefin/dynamics.py` and rotates the hull by about 51 degrees in 30 s (0.09 m drift), enough to swing a corner into a nearby wall. The terminal set must therefore command the rudder to zero.
5. **A stop is short.** At full astern the filter's weak/delayed reverse model gives 0.235 m/s^2 after 0.75 s; the simulator's own model gives 0.471 m/s^2 immediately. Ignoring drag, the stopping distance after the brake command is 1.09 m versus 0.33 m from 0.56 m/s, and 2.39 m versus 0.86 m from 0.9 m/s, taking 1.2-4.6 s. A stop contingency therefore relies on forecasts of 2-5 s rather than the 8 s escape tails, where the summary documents the largest own-ship errors (BAS-NU: 0.17 m and 15 degrees at 2.5 s, 0.65 m at 8 s).

## 3. Answers to the section 12.3 questions

### 3.1 Own dynamics

*Why one-step fits fail on continuous forecasts.* One-step objectives weight the high-frequency residual (gyro noise of 1 deg/s, unobserved servo and FIFO timing, parameter-dependent delay of 12-16 substeps) and are blind to the low-frequency parametric mismatch that dominates multi-step heading error, which position then integrates a second time. A model trained on one-step transitions is also evaluated on its own predicted states during a rollout, a distribution it was never fitted on, so errors compound. These are the arguments of [Abbeel, Ganapathi and Ng (NIPS 2005)](https://proceedings.neurips.cc/paper/2005/hash/09b69adcd7cbae914c6204984097d2da-Abstract.html), who minimize a multi-step criterion, and of [Venkatraman, Hebert and Bagnell (AAAI 2015)](https://doi.org/10.1609/aaai.v29i1.9590), who bound multi-step error under this train/test mismatch. The rejected scalar-gain and whole-parameter-bank selectors (summary sections 10.3-10.4) fitted ten transitions at one operating condition and extrapolated to turning, braking and low speed; their worse eight-second results are consistent with this explanation and remain negative controls.

*Formulation adopted for V20.* Do not adapt the model online. Instead (a) shorten the horizon over which the safety argument needs the forecast (section 2, item 5), and (b) bound the nominal predictor's multi-step error empirically with horizon-indexed split-conformal quantiles calibrated on whole development cases ([Angelopoulos and Bates 2021](https://arxiv.org/abs/2107.07511); multi-horizon use as in [Stankeviciute, Alaa and van der Schaar, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html); planning use as in [Lindemann et al., RA-L 2023](https://arxiv.org/abs/2210.10254)). Calibration scores against the onboard pose estimate, never against truth, so the procedure is deployable from the vessel's own logs; truth is used only to report coverage. Evaluation uses whole held-out development cases and every horizon up to the full contingency duration, not one-step RMSE (section 9.1, gate G1). Deterministic bounds remain the stage-1 reachability route; its interval enclosure (8.7 m at 1 s) is not yet usable.

### 3.2 Target geometry and motion

A joint estimator of partial extent, association and motion mode (extended-object tracking with interacting multiple models, [Granstrom, Baum and Reuter](https://arxiv.org/abs/1604.00970); [Li and Jilkov](https://doi.org/10.1109/TAES.2003.1261132)) is the right long-term answer, and map knowledge can constrain it in confined water. It is **not** part of V20: the saved cases show that partial-hull, stop, turn and corridor-clamp errors are coupled, and a new estimator would change every decision of V16, making the V20 hypothesis untestable in isolation.

V20 keeps V16's perception unchanged, including the motion-axis completion behind the two geometry repairs (CH-HO-CV-073 and DV3-BO-CV-04 are SAC successes that V14/V15 lost and V16 restored). The certificate assumes a **declared target contract**: each tracked target follows constant velocity from V16's estimate within a calibrated allowance rho_tgt(t). Behaviors outside it (non-compliant turns, reactive maneuvers, speed changes, corridor clamping, stop-box stops) are *out of contract by definition*. V20 monitors the contract online by comparing each committed target prediction with the next onboard track estimate and logs violations, in the spirit of set-based prediction that removes a rule assumption once a violation is detected ([Koschi and Althoff, T-IV 2021](https://doi.org/10.1109/TIV.2020.3017385)). V20 does not enlarge the occupancy of a violating target; that is a pre-registered follow-up only if probe results show contract violations preceding contacts.

### 3.3 Policy-aware recoverability

The certificate that can pass the current SAC action is the backup-filter condition: *after executing this command for one decision, a contingency exists that keeps the vessel clear and ends in a verified terminal set* ([Bastani 2019](https://arxiv.org/abs/1905.10691); [Wabersich and Zeilinger, Section 4.1](https://arxiv.org/html/1812.05506v4#S4.SS1); [Agrawal, Chen and Panagou, gatekeeper, IEEE T-RO 2024](https://arxiv.org/abs/2211.14361)). The comparative review by [Kim, Menon, Trivedi and Panagou (2026 preprint)](https://arxiv.org/abs/2604.02401) shows model-predictive shielding as a special case of gatekeeper and attributes much of the conservatism to judging safety by backup feasibility rather than by the nominal policy's own continuation.

The myopic any-feasible regressions are avoided in two ways. First, V20 does not use certification to *remove* V16 interventions (section 2, item 3). Second, the certified contingency is **committed** and is executed when the next check fails, instead of `last certificate` (a plan that fails its recheck) or `no escape` (an unchecked SAC command). SAC is never rolled forward on imagined observations; simulator cloning and recorded future SAC commands remain diagnostic only.

### 3.4 Backup and terminal design

**A terminal set exists for static hazards in the simulated plant.** Rest with zero propulsion and zero rudder is an equilibrium of the identified model for every parameter vector (section 2, item 4), the simulator has no current or wind, and surge cannot become negative. A rest pose whose inflated hull is clear of all static points and inside the navigable polygon therefore stays clear indefinitely. This replaces the old V3 assertion that no invariant set exists; the correct statement is that rest is invariant for static hazards and that no invariant set has been constructed that is safe against *arbitrary* target behavior.

**Targets.** A vessel at rest cannot prevent another vessel from striking it. Following [Bouraine, Fraichard and Salhi (Autonomous Robots 2012)](https://doi.org/10.1007/s10514-011-9258-8), the contingency check excludes *braking inevitable-collision states* with respect to the declared target contract: it requires the contracted target occupancy to stay clear of the resting hull for a hold horizon after rest. Without that check the guarantee degrades to passive safety, which the evaluation would still count as a target contact ([Mitsch et al., IJRR 2017](https://doi.org/10.1177/0278364917733549) distinguish static, passive and passive-friendly safety). The same structure, fail-safe maneuvers to standstill verified against predicted occupancy and executed when a new plan cannot be verified, is used for road vehicles by [Magdici and Althoff (ITSC 2016)](https://doi.org/10.1109/ITSC.2016.7795594) and [Pek et al. (Nature Machine Intelligence 2020)](https://doi.org/10.1038/s42256-020-0225-y); their guarantee likewise holds only while other participants respect stated physical and legal constraints.

**Mission admission and liveness.** Safety and completion are separated. Rest can end in a timeout. V20 records whether a certified contingency exists at the first decision (admission) for every case without removing any case from a denominator. Liveness is delegated to V16 (which still decides wherever it can check) and to SAC (whose certified commands are preferred in the replaced branches), not to the contingency.

### 3.5 Perception safety

V20 inherits V16's static memory, persistent target hypotheses (up to 8 s coast) and exact-source ownership; V19's point transfer is not used. Duplicated target returns therefore remain static points, which makes the certificate conservative rather than unsafe (DV3-CRS-CV-04). Missing evidence remains the main unsafe direction: sparse returns, the 1 m dead zone, occlusion and deleted tracks are not covered by any certificate. Following the known-free-space backup of [Tordesillas et al. (T-RO 2022)](https://doi.org/10.1109/TRO.2021.3100142), V20 logs, for each committed contingency, the fraction of its swept hull samples that lie on current scan rays observed free, inside the dead zone, or beyond any return. This is a reported diagnostic, not a gate, because a hard known-free constraint would also reject most near-wall contingencies in the current sensor model. Phase 2 reports how often committed contingencies rely on unobserved space.

### 3.6 Evaluation

The hypothesis, source and selection are frozen before each episode stage; pre-registered offline gates precede any episode (section 9). All six lost SAC successes, the earlier rescues and the no-intervention controls stay in every denominator. Test sets v3 and v4 are not run and no test-set outcome is read. Reports pair by test ID, episode seed and scenario digest and give goals, rescues, lost SAC successes, timeouts and each contact type, plus certificate metrics (section 9.4).

## 4. V20 design

### 4.1 Architecture

```
SAC action a ──> V16 (unchanged; proposal p, branch label, plan)
                   │
                   ├─ certify(p.command) every decision ──> commit best contingency if certified
                   │
                   └─ if p is an unchecked branch (last certificate / no escape / unknown label):
                         issue the first certified command among
                           1. a (SAC)       2. p.command (V16)
                           3. V16 grid and brake rows, nearest to a
                           4. next command of the committed contingency
                         else: committed contingency's next command (out of contract)
                         else: full astern with rudder centred (out of contract)
```

V16's decisions in checked branches are issued unchanged (they hard-passed V16's own checker). If such a command cannot be certified with a stop contingency, it is still issued, labelled `finite-horizon only`, and the committed contingency is cleared because the vessel has left it. V20's conditional statement (section 7) applies only to certified decisions; the share of certified decisions is reported.

### 4.2 Contingency family

A contingency is the candidate first command for one decision (0.5 s, the executed horizon) followed by a tail:

- **Immediate stop:** full astern from the next decision with rudder -1, 0 or +1 for that decision, then rudder 0 until rest (3 tails).
- **Turn then stop:** rudder in {-1, -0.5, +0.5, +1} at cruise or ceiling propulsion for 1, 2, 3, 4 or 6 decisions (0.5-3 s), then full astern with rudder 0 until rest (40 tails).
- **Hold:** after rest, zero propulsion (policy throttle -1 maps to 0 rpm-units) with rudder 0. The environment already converts a brake request to zero propulsion once observed surge is at most 0.05 m/s.

The 43 tails are fixed and small; they are a finite sample of an infinite control space, so failure to certify is never reported as an inevitable collision.

### 4.3 Certificate predicate

For candidate command c, a tail is **certified** when, for both reverse-thrust models (weak/delayed 0.25 and 0.75 s; strong/immediate 0.5 and 0 s, the existing `safety_prediction` envelope), the rollout from V16's decision snapshot and pre-command actuator history satisfies at every 0.125 s sample until rest, and through the hold:

1. **Static:** inflated-hull clearance to every remembered static point minus the existing 0.05 m gap minus `e_own(t)` is nonnegative.
2. **Boundary (signed):** every corner of the inflated hull lies inside the navigable polygon, and edge clearance minus the existing 0.02 m gap minus `e_own(t)` is nonnegative. The containment test closes the known hole in which unsigned edge distance becomes positive again outside a wall; a unit test checks it is at least as strict as the environment's border predicate for the basin and channel geometries.
3. **Targets:** hull separation to every V16 track view advanced at constant velocity, minus the existing 0.10 m gap, `e_own(t)` and `rho_tgt(t)`, is nonnegative from now until `t_rest + H_hold`, with `H_hold = 8 s` (the existing horizon). Beyond that, safety at rest is rechecked every decision.
4. **Rest:** both models reach |u|, |v| <= 0.02 m/s and |r| <= 1 deg/s with the servo commanded to zero within 10 s; otherwise the tail fails.
5. **Inter-sample allowance:** every sampled clearance is reduced by `0.5 * dt * (s_max + R_h * |r|_max)` over the interval, where `R_h` is the hull half-diagonal; this is a Lipschitz bound on clearance between samples for a rigid rectangle.

Here `e_own(t) = max(0, rho_own(t) - HULL_MARGIN)`: the existing 0.15 m hull margin already acts as a constant allowance, so only the calibrated excess is added. Among certified tails V20 commits the one with the largest minimum slack, breaking ties by shorter duration.

### 4.4 Calibrated allowances

- **Own ship:** for each saved decision with a full snapshot, predict the pose with the nominal predictor from the onboard snapshot, V16's observer state and actuator history, driven by the commands that were actually issued afterwards, using the reverse model closer to the outcome for braking segments. Score `|dp| + R_h * |dpsi|` against the onboard pose estimate at later decisions, for `t` in 0.5-10 s.
- **Targets:** constant-velocity forecasts from V16's track views, scored against the later onboard estimate of the same source, for targets whose scenario behavior label is constant velocity (the contract population; the label is used offline only).
- **Quantile:** split-conformal 95% per horizon bin with the finite-sample rank correction, made nondecreasing in `t`; bins with fewer than 50 pairs are extrapolated linearly from the last two supported bins and never decreased.
- **Data and split, fixed now:** V14, V15 and V16 runs with full snapshots (`persistent_prefix32`, `conditional_prefix32`, `motion_axis_probe5`, `v16_feasible_probe9`, `v16_broader40_paired`), which share V16's predictor and observer. V17 runs are excluded (different yaw observer). Calibration and validation split by scenario: the 32-case cohort scenarios calibrate and the 40-case scenarios validate, then the roles swap. The frozen runtime tables use the pooled data after both coverages are reported.
- **Coverage reporting:** per-decision coverage on the held-out split for each horizon, case-level coverage of the maximum error, and truth-scored coverage. Decisions within a case are correlated and the closed-loop V20 state distribution differs from the calibration runs, so the conformal guarantee is approximate and is stated as such. Recorded future commands are used here to characterize model error offline, never to select an action.

The tables are written once under `results/safety_dev/v20_development/tube_calibration/` and copied into `src/safety_v20_tubes.py` together with the SHA-256 of the calibration artifact. If held-out own-ship coverage falls below 90% at any horizon used by a certified contingency, gate G1 fails (section 9.1).

### 4.5 Consistency with V16's internal state

V20 subclasses `SafetyFilterV16` and overrides `_filter`. When it issues a command other than V16's, it restores the pre-decision actuator copy that V4 already keeps (`_observer_actuators`), reissues its own rudder, sets the brake flag, stores the committed contingency as V16's `plan` (so V16's continuation candidate at the next decision is the certified contingency), sets `mode = "recovery"` and resets `uncertified_steps`. The observer receives the actually issued command from the environment hook, as now. Perception adapters are untouched.

### 4.6 Options, defaults and parity

| Constructor option | Default | Effect when changed |
| --- | --- | --- |
| `certified_fallback` | `True` | `False` returns V16's decision unchanged (parity mode; certificates still logged) |
| `allowance_tables` | `"calibrated"` | `"none"` sets `e_own = rho_tgt = 0` (nominal certificate) |
| `hold_horizon_s` | `8.0` | Hold duration in the target check |
| `out_of_contract` | `"committed_then_stop"` | Pre-registered alternative `"v16"` issues V16's command instead |

Defaults are fixed before any V20 episode. No numerical threshold other than the conformal level, the hold horizon and the contingency grid is introduced, and none is fitted to an outcome.

### 4.7 Logged diagnostics

Each decision records: V16's branch and command; certification status of SAC, V16's command and the issued command; the committed tail, its slack, duration and reverse model margins; V20 level (`v16_unchanged_certified`, `v16_unchanged_finite_only`, `replaced_by_sac`, `replaced_by_v16_certified`, `replaced_by_projection`, `committed_continuation`, `out_of_contract_committed`, `out_of_contract_stop`); recheck survival of the previous commitment; target-contract residuals; the observed-free fraction of the committed swept hull; and computation time.

## 5. Section 11 mechanisms addressed

| Limitation | V20 treatment | Status |
| --- | --- | --- |
| Loss of future feasibility (`last certificate`, `no escape`) | Committed certified contingency replaces both unchecked branches | **Primary target** |
| Physical/time limitations | Rest made a verified terminal set for static hazards; braking allowed without traffic as a contingency only; braking inevitable-collision states with respect to the target contract excluded | Addressed within the contract |
| Policy distribution shift after interventions | In replaced branches the certified command nearest to SAC is issued, preferring SAC itself, so recovery follows SAC's intent rather than a failed stored plan | Partially |
| Own-motion forecast bias | Safety argument rests on 2-5 s forecasts with calibrated allowances; no online refitting | Partially; V16's 8 s decisions keep their bias |
| Margin/preference tradeoff | Not used to remove V16 interventions (evidence in section 2, item 3) | **Deliberately not addressed** |
| Target extent/heading bias | V16 perception retained; geometry repairs preserved by construction | Unchanged |
| Target behavior mismatch | Declared constant-velocity contract, violations monitored and counted | Made explicit, not solved |
| Missing or duplicate obstacle evidence | Duplicates conservative; missing evidence logged | Not solved |
| Limited search | Finite tails; certification failure never called infeasibility | Unchanged |

## 6. Assumptions and whether they hold

| ID | Assumption | Simulator | Physical vessel | How checked |
| --- | --- | --- | --- | --- |
| A1 | Own trajectory lies within `rho_own(t)` of the nominal forecast | Empirical, approximate coverage under randomized parameters | Unknown | G1 coverage; online recheck survival |
| A2 | True braking lies between the weak/delayed and strong/immediate models | Holds: the plant uses the strong end (0.5, no delay) | Reverse thrust not identified (planned crash-stop test S1-C2) | Declared; dual-model check |
| A3 | No environmental forcing | Holds | False; rest is invariant only up to a drift bound | Declared; field use needs a drift bound |
| A4 | Rest with zero rudder and propulsion is an equilibrium | Holds exactly for every parameter vector (section 2, item 4) | Holds only up to A3 | Unit test from model equations |
| A5 | Static memory plus map polygon contain the hazards touched by the contingency | Approximate; sparse points, dead zone, occlusion | Same | Observed-free diagnostic |
| A6 | Targets follow constant velocity within `rho_tgt(t)` over contingency plus hold | False for RE/NC/VS behaviors, clamp and stop-box events | Not established | Contract residual monitor; out-of-contract counts |
| A7 | Map polygon is correct; signed containment is at least as strict as the environment's border test | To be unit-tested | Map accuracy unverified | Unit test |
| A8 | Decision computed before the command is needed | Holds (simulation waits) | 0.5 s deadline not verified | Timing p95/max reported |
| A9 | Calibration cases are representative of V20's closed loop | Approximate (shift) | Unknown | Coverage on V20 traces reported after phase 2 |

## 7. Conditional statement that phase 2 can test

If at decision k the issued command is certified (section 4.3) and A1-A8 hold over the next decision, then at decision k+1 the remainder of the committed contingency is again a certified option, so a certified command exists at k+1; continuing inductively, no contact occurs while the contract holds and every issued command is certified. This is the gatekeeper and model-predictive-shielding argument with an invariant terminal set. Three caveats make it conditional rather than a guarantee: A1 and A6 are empirical with stated miscoverage, so repeated per-decision application carries a cumulative mission risk; the calibrated allowances are not nested from one decision to the next, so recursive survival is measured rather than proved (gate G3); and decisions issued as `finite-horizon only` break the chain. A contact that occurs while V20 reports a certified, in-contract chain falsifies the certificate.

## 8. Differences from rejected approaches

| Rejected or limited earlier approach (summary sections 5-10) | V20 difference |
| --- | --- |
| Longer horizons (V3 16 s ablations) | Safety-critical forecast shortened to the stop contingency |
| Broad course recovery, rejoin references (V3, V5) | Contingencies end at rest; no route is claimed safe |
| Indiscriminate policy priority / any feasible tail (V10, V16 option) | V16's intervention timing is kept; preference for SAC applies only inside replaced unchecked branches and only to certified SAC commands |
| `last certificate` after a failed recheck (V3-V16) | Replaced by certified alternatives or the committed contingency; out-of-contract use is counted |
| Hold-back fallback (V7, removed in V8) | Not reintroduced; no presently failing candidate is selected for delaying contact |
| Braking only with traffic (V2 iteration 6) | Braking is a contingency without traffic, never a preference; rest is the verified terminal |
| Larger target hypothesis unions (V9) | No hypothesis is added to reject SAC; one declared contract is monitored |
| Scalar gain, model bank, raw-yaw substitution (V17, sections 10.3-10.4) | No model adaptation; error is bounded empirically at short horizons |
| Fixed 0.15 m trigger margin as safety buffer | Unchanged for V16's own decisions; the certificate uses calibrated, horizon-indexed allowances instead |
| Stage-1 interval enclosure | Statistical rather than deterministic allowances; stage 1 remains the route to a deterministic bound |
| V18/V19 perception patches | Not inherited; V16 perception only |

## 9. Falsification plan

### 9.1 Offline gates before any V20 episode

All gates use saved development data only and run no episode.

- **G0 Unit and parity tests** (`tests/test_safety_v20_*.py`): equilibrium at rest and rudder drift from the model equations; dual-model contingency rollouts; signed containment versus the environment's border predicate; inter-sample allowance; nondecreasing allowance tables; branch classification, including unknown labels treated as unchecked; actuator rewind and V16 state handoff; `certified_fallback=False` returns V16's exact command and state on synthetic and saved snapshots; no truth, scenario ID or outcome read by runtime code (static import and attribute checks).
- **G1 Allowance calibration.** Pass if held-out own-ship per-decision coverage is at least 90% for every horizon bin used by a certified contingency, in both split directions. Target-contract coverage is reported against truth for constant-velocity targets and for all targets separately; it is not a pass condition because non-constant-velocity targets are out of contract by definition.
- **G2 Saved-state replay at mechanism and cascade states.** Reconstruct V16's decision snapshot and pre-command actuator history from saved full-snapshot records and evaluate the V20 certificate and level selection. V15's `conditional_prefix32` records stand in for V16 in BAS-NU-CV-070, P2-L1-CRP-FIX-05 and BAS-HO-NC-059, where the audit confirmed identical command sequences; `motion_axis_probe5` and `v16_broader40_paired` supply the other V16 states. Measures: certified fraction of V16's issued commands; at each of the 21 contacts after unchecked phases, whether a certified option existed at the last checked decision and at each unchecked decision; the command V20 would issue at each saved `last certificate`/`no escape` decision in rescues and both-goal cases (113 and 84 decisions) and how often it equals V16's.
- **G3 Recursive survival.** Along saved V16 trajectories, for every decision whose issued command is certified, recheck the committed tail at the next saved decision (valid because the vessel executed exactly the contingency's first command). Report survival with Wilson intervals, separately for scenes with and without a target inside the engagement range.

**Advancement to episodes** requires G0 to pass, G1 to pass, G3 survival of at least 0.90 in both strata, and a certified option at the last checked decision in at least 11 of the 21 contact precursors. Otherwise phase 2 stops with a report; the design is revised only with a new dated plan.

### 9.2 Pre-registered predictions for the 12-case probe

| Case | Role | First V16 unchecked decision | Prediction |
| --- | --- | ---: | --- |
| TS2:CH-HO-CV-073 | SAC success, V16 geometry repair | none | Command sequence identical to fresh V16 |
| TS2:CH-HO-CV-043 | No-intervention control | none | Identical to fresh V16 |
| TS2:P2-L1-CRS-FIX-09 | No-intervention control | none | Identical to fresh V16 |
| DV3:DV3-BO-CV-04 | SAC success, V16 geometry repair | 13 | Identical before 13; goal retained; at 13 V20 issues SAC if certified |
| TS2:P2-L1-CRP-VAR-12 | Rescue | 4 | Identical before 4; rescue at risk; reported either way |
| TS2:BAS-NU-CV-070 | Lost SAC success (boundary) | 65 | Identical before 65 (decision-11 intervention unchanged); no boundary contact; timeout expected rather than goal |
| TS2:BAS-HO-NC-059 | Lost SAC success (target) | 23 | Identical before 23; the target's turn and corridor jump are out of contract; outcome uncertain |
| DV3:DV3-CRS-CV-04 | Lost SAC success (boundary) | 53 | Identical before 53; no boundary contact |
| TS2:P2-L2-CRS-FIX-19 | Lost SAC success (boundary) | 28 | Identical before 28; no boundary contact |
| TS2:P2-L3-BO-FIX-09 | Lost SAC success (boundary) | 52 | Identical before 52; no boundary contact |
| TS2:P2-L3-CRP-FIX-03 | Lost SAC success (target) | 6 | Identical before 6; outcome depends on the crossing target's contract |
| TS2:P2-L1-CRP-FIX-05 | Never-triggered target case | 25 | Identical before 25; target stops at 32 (out of contract); outcome uncertain |

"Identical before" means the same issued commands up to that decision, conditional on fresh V16 reproducing its own command sequence in the phase-2 environment. Any earlier divergence is an implementation fault to diagnose before results are interpreted.

### 9.3 Episode stages

| Stage | Cases and controllers | Runs | Purpose |
| --- | --- | ---: | --- |
| E0 | 12-case probe, fresh OFF and fresh V16 | 24 | Environment reproduction of historical outcomes and command sequences, reported but not required for pairing |
| E1 | Same 12 cases, V20 | 12 | Mechanism cases (all six named cases are in the probe), regressions, rescues, controls |
| E2 | 32-case cohort: OFF, V16, V20 | 96 | Development cohort |
| E3 | 40-case cohort: OFF, V16, V20 | 120 | Disjoint development cohort |

**Probe advancement (E1 to E2), fixed now:** (P1) identical command sequences in the three probe cases without unchecked V16 decisions; (P2) no contact while the issued command was certified and in contract; (P3) total V20 contacts strictly fewer than fresh V16's on the probe. Lost V16 goals are reported but do not block advancement. If P1 or P2 fails, phase 2 stops for diagnosis and a dated note; a fix is evaluated only under a new tag and reported with the failed tag. Not more than one pre-registered option change (`allowance_tables="none"` or `out_of_contract="v16"`) is run on the probe, and only if a specific E1 mechanism points to it.

**Promotion over V16** (development only) requires, on the 72 cases combined: fewer contacts than fresh V16, no additional lost SAC successes relative to fresh V16, and no loss of fresh-V16 goals. Otherwise V20 is reported as a tradeoff and V16 remains the reference. No test-set evaluation is implied by promotion.

### 9.4 Reporting

Per stage and combined: goals; rescued SAC failures; lost SAC successes; preservation; timeouts; obstacle, boundary and target contacts; contact-type changes (not counted as rescues); per-case first divergence from fresh V16; and V20 metrics: share of decisions at each level, certified share, recheck survival, out-of-contract events and their outcomes, contract residuals, observed-free fractions and decision time p50/p95/max. Strict validation reuses `report_safety_iteration.py` on the new run directories, pairing by test ID, episode seed and scenario digest with fresh same-environment OFF and V16 references.

## 10. Phase-2 implementation plan

### 10.1 Environment

A Python 3.11 virtual environment outside the repository with the pinned stack from `requirements.txt` (numpy 1.26.4, torch 2.2.1, gymnasium 0.29.1, stable-baselines3 2.3.2, sb3-contrib 2.3.0, pandas 2.2.1, pytest), installed from PyPI because the PyTorch CPU index is blocked; the frozen `pip freeze`, interpreter and platform go into every run manifest. Checkpoint, config and formulation identities are rechecked before every stage.

### 10.2 Files

| Path | Content |
| --- | --- |
| `src/safety_v20_contingency.py` | Contingency family, dual-model rollout, certificate predicate, signed containment, inter-sample allowance, hold check |
| `src/safety_v20_tubes.py` | Frozen allowance tables with calibration-artifact hash |
| `src/safety_v20.py` | `SafetyFilterV20(SafetyFilterV16)`: level selection, commitment, V16 state handoff, diagnostics |
| `tests/test_safety_v20_contingency.py`, `tests/test_safety_v20_filter.py`, `tests/test_safety_v20_tubes.py` | Gate G0 |
| `tools/diagnostics/safety/v20_calibrate_allowances.py` | Gate G1 (saved data only) |
| `tools/diagnostics/safety/v20_replay_saved_decisions.py` | Gates G2 and G3 (saved data only) |
| `tools/diagnostics/safety/v20_iteration.py` | Thin wrapper around `safety_candidate_iteration.py` that writes to `results/safety_dev/v20_development/runs/<tag>/` and adds the environment freeze; the existing runner is not edited |
| `results/safety_dev/v20_development/` | Calibration, replay, runs, reports and README |

`src/constants.py`, `src/env.py`, V16 and its parents, the checkpoint, `configs/baseline_v3.json`, `runs/`, completed ledgers, existing results and `asv-lidar/static_obstacles/` are not edited. Native dispatch in `env.py` is not extended; the runner installs the class explicitly after reset, as for earlier candidates. Existing line endings are preserved and no file over 50 MB is committed.

### 10.3 Compute and run-management rules

At most two evaluation processes, each with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `torch.set_num_threads(1)` (the runner already enforces these). Each tag covers a chunk expected to finish within about 25 minutes (about six cases with three controllers); results are written per episode; tags are never reused, overwritten or silently retried; a STOP marker halts before the next episode. Each completed chunk or document is committed and pushed to `safety-v20`. Estimated total is 252 episode runs plus at most one 12-run option probe.

## 11. Risks that would change the plan

- **Certified share too low:** the weak reverse model needs 2.4 m to stop from 0.9 m/s, and narrow-water target passes may admit no certified contingency. V20 then behaves as V16 (finite-horizon only) and cannot improve safety; G2 and G3 detect this before episodes.
- **Rescues degraded inside replaced branches:** rescues contain 113 unchecked decisions, some of which succeeded because the nominal model was pessimistic. If V20 issues different commands there, rescues may turn into timeouts. G2 quantifies the exposure; E1 measures the effect.
- **Deadlock at rest:** without astern maneuvering, a vessel at rest facing a wall can only turn slowly. Timeouts are reported separately and never counted as goals or as contacts avoided by design.
- **Target contract:** BAS-HO-NC-059, P2-L1-CRP-FIX-05 and reactive targets are expected to leave the contract. Out-of-contract contacts are reported separately; they are not evidence for or against the certificate.
- **Environment drift:** fresh V16 may not reproduce historical command sequences under the new interpreter and library versions. Pairing uses fresh runs; historical reproduction is reported as context only.

## 12. References

Existing R-numbers refer to [section 14 of the complete record](SAFETY_LAYER_COMPLETE_SUMMARY.md#14-method-bibliography-and-exact-scope-of-attribution). Bibliographic identities below were checked on 2026-10-10 through publisher, proceedings or arXiv records; full texts were not all retrievable from this environment, so attributions are limited to the abstract-level claims stated.

- Agrawal, D. R.; Chen, R.; Panagou, D. *gatekeeper: Online Safety Verification and Control for Nonlinear Systems in Dynamic Environments.* IEEE Transactions on Robotics, 2024. [arXiv:2211.14361](https://arxiv.org/abs/2211.14361). Committed trajectory formed from a nominal segment and a backup controller into a controlled-invariant backup set.
- Kim, T.; Menon, A. D.; Trivedi, A.; Panagou, D. *Backup-Based Safety Filters: A Comparative Review of Backup CBF, Model Predictive Shielding, and gatekeeper.* Preprint, 2026. [arXiv:2604.02401](https://arxiv.org/abs/2604.02401).
- Bastani, O. (R3); Wabersich, K. P.; Zeilinger, M. N. (R4); Chen, Y. et al. backup CBF (R5); Hsu, Hu and Fisac (R16); Fraichard and Asama (R22).
- Bouraine, S.; Fraichard, T.; Salhi, H. *Provably safe navigation for mobile robots with limited field-of-views in dynamic environments.* Autonomous Robots 32, 267-283, 2012. [DOI 10.1007/s10514-011-9258-8](https://doi.org/10.1007/s10514-011-9258-8). Passive motion safety and braking inevitable-collision states.
- Mitsch, S.; Ghorbal, K.; Vogelbacher, D.; Platzer, A. *Formal verification of obstacle avoidance and navigation of ground robots.* International Journal of Robotics Research 36(12), 1312-1340, 2017. [DOI 10.1177/0278364917733549](https://doi.org/10.1177/0278364917733549).
- Magdici, S.; Althoff, M. *Fail-safe motion planning of autonomous vehicles.* IEEE ITSC 2016, 452-458. [DOI 10.1109/ITSC.2016.7795594](https://doi.org/10.1109/ITSC.2016.7795594).
- Pek, C.; Manzinger, S.; Koschi, M.; Althoff, M. *Using online verification to prevent autonomous vehicles from causing accidents.* Nature Machine Intelligence 2, 518-528, 2020. [DOI 10.1038/s42256-020-0225-y](https://doi.org/10.1038/s42256-020-0225-y).
- Koschi, M.; Althoff, M. *Set-based prediction of traffic participants considering occlusions and traffic rules.* IEEE Transactions on Intelligent Vehicles 6(2), 249-265, 2021. [DOI 10.1109/TIV.2020.3017385](https://doi.org/10.1109/TIV.2020.3017385).
- Tordesillas, J.; Lopez, B. T.; Everett, M.; How, J. P. *FASTER: Fast and Safe Trajectory Planner for Navigation in Unknown Environments.* IEEE Transactions on Robotics 38(2), 922-938, 2022. [DOI 10.1109/TRO.2021.3100142](https://doi.org/10.1109/TRO.2021.3100142).
- Alsterda, J. P.; Gerdes, J. C. *Contingency model predictive control for automated vehicles.* American Control Conference, 2019. [Stanford record](https://purl.stanford.edu/ty849th6449). Related contingency-plan architecture; not implemented here.
- Lindemann, L.; Cleaveland, M.; Shim, G.; Pappas, G. J. *Safe planning in dynamic environments using conformal prediction.* IEEE Robotics and Automation Letters 8(8), 2023. [arXiv:2210.10254](https://arxiv.org/abs/2210.10254).
- Dixit, A.; Lindemann, L.; Wei, S. X.; Cleaveland, M.; Pappas, G. J.; Burdick, J. W. *Adaptive conformal prediction for motion planning among dynamic agents.* L4DC 2023, PMLR 211, 300-314. [arXiv:2212.00278](https://arxiv.org/abs/2212.00278). Online adaptation is a possible follow-up if closed-loop coverage drifts; not part of V20.
- Gibbs, I.; Candes, E. *Adaptive conformal inference under distribution shift.* NeurIPS 2021. [arXiv:2106.00170](https://arxiv.org/abs/2106.00170).
- Angelopoulos, A. N.; Bates, S. *A gentle introduction to conformal prediction and distribution-free uncertainty quantification.* 2021. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511).
- Stankeviciute, K.; Alaa, A. M.; van der Schaar, M. *Conformal time-series forecasting.* NeurIPS 2021. [Proceedings](https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html).
- Abbeel, P.; Ganapathi, V.; Ng, A. Y. *Learning vehicular dynamics, with application to modeling helicopters.* NIPS 18, 2005. [Proceedings](https://proceedings.neurips.cc/paper/2005/hash/09b69adcd7cbae914c6204984097d2da-Abstract.html).
- Venkatraman, A.; Hebert, M.; Bagnell, J. A. *Improving multi-step prediction of learned time series models.* AAAI 2015, 3024-3030. [DOI 10.1609/aaai.v29i1.9590](https://doi.org/10.1609/aaai.v29i1.9590).
- Sha, L. *Using simplicity to control complexity.* IEEE Software, 2001. Simplex switching between a high-performance and a high-assurance controller; the architectural pattern of V16 as performance layer and the certified contingency as assurance layer.
- Granstrom, Baum and Reuter (R14); Li and Jilkov (R17); Ljung (R15); stage-1 references (Kochdumper et al.; Li et al.; Krasowski and Althoff) as cited in the [stage-1 plan](SAFETY_REACHABILITY_STAGE1_PLAN.md).

## 13. Revision 1, 2026-10-10: offline gate results and redesign before any V20 episode

Status: the phase-1 default configuration fails the pre-registered advancement rule of section 9.1, so no episode was run with it. The pre-registered nominal option (`allowance_tables="none"`) meets the rule, narrowly. This section gives the gate results, corrections to sections 2-4 found during implementation, the diagnosis, the decision, and a revised configuration (revision 1) with pre-registered episode criteria. Sections 0-12 stand as written; where they disagree with this section, this section applies. All evidence is development data; no test-set outcome was read.

### 13.1 Corrections found during implementation

1. **Rest after a braked turn is slow (sections 2 item 4, 3.4, 4.3 item 4, assumption A4).** Rest with zero rudder and zero propulsion is an exact equilibrium, as stated, but the hull does not settle quickly after a turn: at zero surge the identified yaw and sway damping is almost purely quadratic. After a 2 s full-rudder turn at 0.56 m/s followed by full astern (strong reverse model), surge reaches zero at 3.1 s, yet the heading changes by a further 101 degrees by 20 s and the yaw rate is still 5.8 deg/s at 12 s. Predicate 4 (|r| at most 1 deg/s within 10 s) is therefore unattainable for turn tails. The implemented predicate simulates an explicit 20 s window (40 decisions) for both reverse models, spin-down included. A tail qualifies if surge reaches 0.02 m/s with at least the 8 s hold left inside the window and planar speed is at most 0.05 m/s at the window end. The final sample carries a 1 s residual-motion allowance. No pose is frozen at rest.
2. **Tail choice (section 4.3).** The best qualifying immediate-stop tail is committed if one exists, otherwise the best qualifying turn tail (largest minimum slack, then earlier rest).
3. **Computation.** All candidates of one decision are certified in a single vectorised rollout, and clearances are evaluated in column chunks to bound memory. Unit tests show both are identical to single evaluation.
4. **Replay runs.** `g2g3_v1` (too slow) and `g2g3_v2` (memory exhausted) were stopped before any result; their `ABORTED.md` files say why. `g2g3_v3` is the reported run. Its part 2 was resumed once after a memory failure, with an identity note.
5. **Options added after the gates.** These are constructor options, unused by any gate, and each defaults to the phase-1 behavior:
   - `tail_family="extended"`: the 3 stop tails plus 68 turn-cruise-stop tails (rudder in {-1, -0.5, 0, 0.5, 1} for 1, 2, 4 or 6 decisions, then 0-6 s at cruise, then full astern with rudder 0);
   - `enforcement="gatekeeper"`;
   - `commit_margin`.

### 13.2 Gate results

- **Scoring tool:** `tools/diagnostics/safety/v20_gate_report.py`, output [`offline_gates/gates_v1/gates.json`](../results/safety_dev/v20_development/offline_gates/gates_v1/gates.json).
- **Inputs:** the [G1 calibration](../results/safety_dev/v20_development/allowance_calibration/g1_v1/calibration.json) and the merged [G2/G3 replay](../results/safety_dev/v20_development/saved_state_replay/g2g3_v3/replay.json).
- **Replay coverage:** 68 of the 72 development cases have a V16-equivalent full-snapshot trace (3,569 decisions), including all 21 contact precursors.

| Gate | Threshold | Default (`calibrated`) | Option `none` |
| --- | --- | --- | --- |
| G0 unit and parity tests | pass | 41 passed at replay time; 43 now | same |
| G1 held-out own-ship coverage, both split directions, bins used | at least 0.90 | Pass: at least 0.90 in every bin up to 11 s; the bins that fall short (11.5 s and 12 s: 0.896 and 0.892, calibrate-A/validate-B) lie beyond the latest default-family rest time of 7.0 s from the highest saved surge (1.09 m/s) | not used |
| G2 checked or idle V16 commands certified | reported | 1,075/3,216 (33 %) | 2,746/3,216 (85 %) |
| G2 unchecked V16 commands certified | reported | 8/353 | 27/353 |
| G2 unchecked decisions with any certified option (SAC, V16, grid, committed) | reported | 27/353 | 38/353 (11 %) |
| G2 contact precursors with a certified option at the first unchecked decision | reported | 2/21 | 2/21 |
| G3 one-step survival, target in engagement range | at least 0.90 | 157/195 = 0.805 (Wilson 0.744-0.855) | 1,400/1,521 = 0.920 (0.906-0.933) |
| G3 one-step survival, no target | at least 0.90 | 814/859 = 0.948 (0.931-0.961) | 1,137/1,210 = 0.940 (0.925-0.952) |
| G3 contact precursors certified at the last checked decision | at least 11/21 | 3/21 | 11/21 |
| **Advancement** | all of the above | **fails** | **meets the rule** (precursor count exactly at the threshold) |

Target constant-velocity forecasts were scored against truth for reporting only. Coverage is 0.68 at 0.5 s, 0.85 at 2 s, 0.89 at 4 s and 0.96 at 8 s. The frozen target allowance is 0.57 m at 0.5 s, 1.02 m at 2 s, 1.77 m at 5 s, 4.2 m at 10 s and 6.4 m at 20 s.

The replay covers 46 successful V16 episodes (16 rescues and 30 both-goal cases); 25 of them contain unchecked decisions, 166 in all. At those decisions the nominal option would issue:

| Command | Decisions |
| --- | ---: |
| Centred full astern, out of contract | 126 |
| Committed contingency, out of contract | 17 |
| A certified SAC command (2 of them equal to V16's) | 17 |
| A certified grid command | 5 |
| V16's own command | 1 |

### 13.3 Diagnosis

1. **The calibrated allowances remove availability without buying survival.** The target allowance grows to several metres over the 20 s window and excludes most stops near traffic. Survival with a target present is *lower* with the allowances (0.805) than without (0.920). The tables are not nested from one decision to the next, and target-estimate jumps exceed them.
2. **Unchecked-only enforcement acts too late.** Of the 11 precursors with a certified contingency at the last checked decision, 9 lose it at the very next decision: the recheck fails exactly when V16 turns unchecked. The other 2 have a certified SAC command, equal to V16's. Over all 353 unchecked decisions a certified option exists in only 11 %. What makes V16 lose its checked plan is the same thing that breaks the committed contingency.
3. **Why rechecks fail.** The 194 nominal recheck failures were rerun with one input changed at a time ([`recheck_attribution/attribution_v1`](../results/safety_dev/v20_development/recheck_attribution/attribution_v1/attribution.json)):

   | Cause | All failures | The 33 failures that coincide with V16 entering an unchecked run |
   | --- | ---: | ---: |
   | Own state only | 104 | 11 |
   | World update only | 58 (55 of them target estimates) | 16, all target |
   | Both | 12 | 5 |
   | Interaction | 20 | 1 |

   The own-ship hull-point displacement behind own-state failures is small (median 0.077 m, maximum 0.18 m). The committed slack at those failures is also small (median 0.087 m).
4. **Tightening the commit margin helps survival at a known availability cost** ([`offline_variants/variants_v1/margin.json`](../results/safety_dev/v20_development/offline_variants/variants_v1/margin.json)). A new commitment requires slack of at least m; the recheck stays at zero.

   | m | Checked decisions available | Survival, target present | Survival, no target |
   | --- | ---: | ---: | ---: |
   | 0 | 85 % | 0.920 | 0.940 |
   | 0.05 m | 81 % | 0.947 | 0.953 |
   | 0.127 m | 74 % | 0.965 (0.954-0.974) | 0.969 (0.957-0.978) |
   | 0.2 m | 68 % | 0.977 | 0.972 |
   | 0.3 m | 61 % | 0.981 | 0.977 |

5. **Gatekeeper availability.** A scratch screen covered 51 replay episodes, before the replay finished. It took the 371 checked or idle decisions whose V16 command has no nominal stop-family certificate, at margin 0:
   - with the extended family, V16's own command is certified in 124;
   - a grid command is certified in 61;
   - nothing is certified in 186.

   In the 6 lost SAC successes, nothing is certified at 30 of 36 such decisions. The committed tool `v20_offline_variants.py options` repeats this on all 68 episodes with the margin of revision 1, as a reported diagnostic.

### 13.4 Decision: improve within the method family

The method family stays backup-based safety filtering. Phase 1 restricted the backup filter to V16's unchecked branches, to keep V16's intervention timing. The gates show that by then the backup is usually gone. The standard form of the family enforces the backup condition at every decision, so that the system never leaves the set from which a backup is certified. This is gatekeeper (Agrawal, Chen and Panagou 2024) and model-predictive shielding (Bastani). It takes the backup earlier, before the cascade, at the price of more interventions in successful episodes. The literature remedy for small one-step model errors that break recursive feasibility is constraint tightening: new commitments carry a margin, and a committed backup is kept while it remains feasible ([Chisci, Rossiter and Zappa, Automatica 2001](https://doi.org/10.1016/S0005-1098(00)00203-1)). Target-estimate jumps remain out of contract. The calibrated target allowance is not a remedy for them (diagnosis item 1).

Alternatives considered and not taken now:
- Larger, rule-based target occupancy (Koschi and Althoff 2021) would lower availability further.
- Passive safety would count a target striking a vessel at rest as a contact, and the scripted targets do not react.
- Learning-based recoverability would need recorded future SAC commands or simulator cloning at runtime, which is not allowed.

Two configurations go to episodes. Both are frozen before any V20 episode:

| Name (runner mode) | Options | Basis |
| --- | --- | --- |
| Nominal V20 (`v20_nominal`) | `allowance_tables="none"`; otherwise phase-1 defaults (stop family, unchecked-only enforcement, margin 0) | Pre-registered option; meets the section 9.1 rule |
| Revision 1 (`v20_r1`) | `safety_v20.REVISION1_OPTIONS`: `allowance_tables="none"`, `enforcement="gatekeeper"`, `tail_family="extended"`, `commit_margin` = 0.1269 m, which is G1's one-decision own-ship allowance (the 0.5 s row of `OWN_TABLE`) | This section; the margin comes from G1 calibration, not from an outcome |

Under gatekeeper enforcement, a checked V16 command without a certificate at the margin is replaced, in order, by:
1. the certified grid command nearest to V16's command;
2. the committed contingency, if its recheck passes;
3. V16's own command, labelled `gatekeeper_uncertified_v16`, as a last resort.

Unchecked decisions are handled as in section 4.1.

**Disclosure.** Before this section was written, a timing-only check ran revision 1 for the first decisions of TS2:BAS-HO-NC-059: decision time p50 0.22 s, p95 1.4 s. The episode ended at decision 28. The outcome was not read, but early termination of this lost-SAC-success case was seen. No option was changed afterwards.

### 13.5 Pre-registered episode criteria

**E1 (12-case probe; fresh OFF and V16 from `e0_probe12_off_v16` as references):**
- `v20_nominal` is judged by sections 9.2 and 9.3 unchanged: P1 identical command sequences in the three cases without unchecked V16 decisions; P2 no contact while the issued command was certified and in contract; P3 strictly fewer contacts than fresh V16 (7).
- `v20_r1` is judged by:
  - R1-P1: no contact in a decision interval that begins with a certified, in-contract decision (level `v16_unchanged_certified`, `replaced_by_*`, `gatekeeper_replaced` or `committed_continuation`, with no target-contract violation logged at that decision). Gatekeeper enforcement changes checked decisions by design, so identity with V16 is not required.
  - R1-P2: strictly fewer contacts than fresh V16 (7).
- For both, the following are reported and do not block: goals (fresh V16 5, OFF 10), lost fresh-V16 goals, timeouts, and the levels of the last three decisions before every contact.

**E2 and E3 (the other 60 development cases; fresh OFF, fresh V16 and every variant that advanced):** chunked as in section 10.3. Promotion over V16 is unchanged from section 9.3. It is judged on the 72 cases combined: fewer contacts than fresh V16, no additional lost SAC successes, and no lost fresh-V16 goals. A variant that misses any condition is reported as a tradeoff.

**Test set v4** (authorized 2026-10-10; selection [`testset_v4_main/selection.json`](../results/safety_dev/testset_v4_main/selection.json), all 1,000 scenarios in definition order):
- Exactly one V20 variant is frozen after E3: among advancing variants, the one with fewer contacts on the 72 development cases, with ties broken by more goals.
- It runs once with fresh OFF and V16, provided it has fewer contacts than fresh V16 on the development cases; otherwise V20 is not run on the test set and the report says why.
- No code, option or threshold changes after any test-set outcome. Results are reported whatever they show: goals, rescued SAC failures, lost SAC successes, and each contact type, paired by test ID, episode seed and scenario digest.

## 14. E1 result and revision 2, 2026-10-10

### 14.1 E1 on the 12-case probe

- **Runs:** [`runs/e1_probe12_v20`](../results/safety_dev/v20_development/runs/e1_probe12_v20/manifest.json) (`v20_nominal`, `v20_r1`) and the fresh references in [`runs/e0_probe12_off_v16`](../results/safety_dev/v20_development/runs/e0_probe12_off_v16/manifest.json).
- **Strict paired report:** [`reports/e1_probe12`](../results/safety_dev/v20_development/reports/e1_probe12/report.md).

| Controller | Goals | Boundary | Target | Obstacle | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| OFF | 10 | 0 | 1 | 1 | 0 |
| V16 | 5 | 4 | 3 | 0 | 0 |
| V20 nominal | 3 | 3 | 4 | 0 | 2 |
| V20 revision 1 | 2 | 2 | 5 | 0 | 3 |

Against fresh V16:

| Variant | Contacts to timeouts | Goals to target contacts | Other differences |
| --- | --- | --- | --- |
| Nominal | P2-L1-CRP-FIX-05 (target), P2-L3-BO-FIX-09 (boundary) | DV3-BO-CV-04, P2-L1-CRP-VAR-12 | None; the three control cases issue command sequences identical to V16 |
| Revision 1 | Those two plus BAS-NU-CV-070 (boundary) | The same two plus CH-HO-CV-073 | None |

Decision time over 1,822 V20 decisions: p50 0.84 s, p95 1.6 s, maximum 23.6 s. The maximum is revision 1 in CH-HO-CV-073, where extended tails on the whole grid are checked against many remembered points.

**Pre-registered criteria (section 13.5):**

| Variant | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| Nominal | P1 | passes | The three control cases are command-identical to fresh V16 |
| Nominal | P2 | **fails** | In P2-L1-CRP-VAR-12 the target contact follows three certified decisions (target slack 0.29, 0.12 and 0.13 m, with no allowance). The one-decision track residuals there are 0.06-0.12 m, so the online monitor did not flag the variable-speed target as out of contract |
| Nominal | P3 | **fails** | 7 contacts, equal to V16's 7 |
| Revision 1 | R1-P1 | **fails** | Same case: the contact follows `committed_continuation` and `v16_unchanged_certified` decisions |
| Revision 1 | R1-P2 | **fails** | 7 contacts |

Neither variant advances to E2/E3, and no V20 variant was run on E2/E3 or the test set. The fresh OFF and V16 references for the other 60 development cases ([`runs/e2_cohort32_off_v16`](../results/safety_dev/v20_development/runs/e2_cohort32_off_v16/manifest.json), [`runs/e3_broader40_off_v16`](../results/safety_dev/v20_development/runs/e3_broader40_off_v16/manifest.json)) were run while E1 was in progress. They are controller-independent and were not used for any decision here.

### 14.2 Mechanisms

Every outcome difference from V16 traces to decisions where nothing is certified. The last three decision levels before the end of each changed episode are in `paired.csv`.

1. **The uncertified stop helps against static hazards and some stopping targets.** BO-FIX-09 and BAS-NU-CV-070 (revision 1) do not hit the boundary. Each ends at rest clear of it after a long out-of-contract phase (`out_of_contract_stop` or `out_of_contract_committed`), although no stop was certified there. CRP-FIX-05 ends at rest instead of a target contact; its target stops at decision 32.
2. **The same stop is harmful with a crossing target.** In DV3-BO-CV-04 at decision 13 every option is predicted to be struck by the target (SAC -1.02 m, V16 -1.02 m, stop -1.01 m, best grid command -0.98 m, from the saved V16 state). V20 stopped and was struck at decision 17, while V16's unchecked hard turn reached the goal. In CRP-VAR-12 (both variants) and CH-HO-CV-073 (revision 1) the vessel stopped near a target for many decisions and was then struck, or struck it on resuming. Section 3.4 already states the reason: rest is invariant for static hazards, but not against targets.
3. **Without a target allowance the certificate is not reliable at small target slack** (CRP-VAR-12). With the calibrated allowance it is rarely available (section 13.2). The constant-velocity forecast of V16's track estimates covers truth only 68 % of the time at 0.5 s (G1), so neither setting gives a usable target certificate with the current estimates.
4. **Gatekeeper enforcement near targets adds interventions without a safety gain** on the probe: 49 replacements and 55 committed continuations, against 237 decisions where V16's command was issued uncertified. It costs time and contributes to the CH-HO-CV-073 loss.

### 14.3 Decision: restrict the backup to where its terminal set is valid

The evidence does not support a stop-terminal backup around targets. Rest remains a sound terminal set for static hazards. The boundary contacts are 4 of fresh V16's 7 probe contacts, and 4 of the 6 lost SAC successes in the phase-1 audit. Revision 2 therefore restricts V20's active role to scenes without a target inside V2's existing engagement range (7 m, the gate V2 already uses). A different terminal set for targets, for example a moving terminal set that clears the target's swept path, would be a new design. It is not attempted here.

| Option | Revision 2 (`REVISION2_OPTIONS`, runner mode `v20_r2`) | Reason |
| --- | --- | --- |
| `allowance_tables` | `"none"` | As revision 1 (section 13.3) |
| `tail_family` | `"stop"` | Extended tails cost up to 23.6 s per decision; availability matters only without targets, where the stop family suffices |
| `commit_margin` | 0.1269 m (G1 one-decision own-ship allowance) | As revision 1 (section 13.3 item 4) |
| `enforcement` | `"gatekeeper_without_traffic"`: gatekeeper at checked decisions only with no track inside 7 m; otherwise unchecked-only | Section 14.2 items 1 and 4 |
| `out_of_contract` | `"stop_without_traffic"`: committed-then-stop only with no track inside 7 m; otherwise V16's command | Section 14.2 items 1 and 2 |

With a target in range, revision 2 is V16 plus certified replacements of V16's unchecked decisions, so its target behavior is V16's except where a certificate exists. Item 3 still applies to those certified replacements and is reported.

Revision 2 is shaped by the probe outcomes, so the probe is no longer a held-out check for it. The 60 other development cases (E2 and E3) are the development evidence that counts, and test set v4 is the final evaluation.

### 14.4 Pre-registered criteria for revision 2

- **E1b (probe, `v20_r2`)** is a functional check before the cohorts, not evidence of benefit. It must show:
  - (a) identical command sequences to fresh V16 in the three control cases;
  - (b) no fresh-V16 goal turned into a contact;
  - (c) no contact in an interval that begins with a V20 replacement certified in contract (`replaced_by_*`, `gatekeeper_replaced`, `committed_continuation`).
  Contacts after `v16_unchanged_certified` decisions are reported as certificate failures, not criteria, because V20 did not act there.
- **E2/E3 (60 cases)** run only if E1b passes. Promotion over V16 on the 60 cases, judged on these alone and not on the probe:
  - fewer contacts than fresh V16;
  - no additional lost SAC successes relative to fresh V16;
  - no fresh-V16 goal turned into a contact.
  Goals turned into timeouts are reported as a liveness cost. Otherwise revision 2 is reported as a tradeoff and V16 stays the reference.
- **Test set v4** runs once with fresh OFF and V16, and only if revision 2 has fewer contacts than fresh V16 on the 60 cases. The rules of section 13.5 apply: no change after any test-set outcome, and full reporting.

### 14.5 E1b result and deviation note, 2026-10-10

- **Run:** [`runs/e1b_probe12_v20_r2`](../results/safety_dev/v20_development/runs/e1b_probe12_v20_r2/manifest.json).
- **Report:** [`reports/e1b_probe12_r2`](../results/safety_dev/v20_development/reports/e1b_probe12_r2/report.md).

| Controller | Goals | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: |
| V16 | 5 | 4 | 3 | 0 |
| Revision 2 | 6 | 2 | 3 | 1 |

| Transition from fresh V16 | Case |
| --- | --- |
| Boundary contact to goal | BAS-NU-CV-070 |
| Boundary contact to timeout | P2-L3-BO-FIX-09 |

No fresh-V16 goal was lost. The three target contacts end in `out_of_contract_v16` decisions, which are V16's own commands. Decision time: p50 0.67 s, p95 1.09 s, maximum 18.2 s.

| Criterion (section 14.4) | Result | Evidence |
| --- | --- | --- |
| (b) no fresh-V16 goal turned into a contact | passes | No goal lost |
| (c) no contact after an in-contract V20 replacement | passes | Every contact ends after an out-of-contract decision |
| (a) identical command sequences in the three control cases | **fails as written** | CH-HO-CV-043 and P2-L1-CRS-FIX-09 are identical; CH-HO-CV-073 diverges at decision 5 |

The CH-HO-CV-073 divergence is the designed gatekeeper action. No track was within 7 m, and V16's command had certified slack 0.107 m, below the 0.127 m commit margin. V20 issued the nearest certified command (`gatekeeper_replaced`, rudder 0, full ahead), and the episode still reached the goal. Criterion (a) was carried over from the unchecked-only P1 without accounting for the gatekeeper. Section 13.5 had waived identity for revision 1 for this reason, and the same waiver applies here. No code or option is changed. E2/E3 proceed with revision 2 as frozen, and this deviation is recorded before any E2/E3 V20 episode.

The near-V16 options analysis of revision 1 (`offline_variants/variants_v1/options_episodes/`, 36 episode files) was stopped during its part 2 so that compute goes to revision 2. Revision 1 is superseded, so it is not merged.

## 15. Revision 2 development result and revision 3, 2026-10-10

### 15.1 Revision 2 on the 60 held-out development cases

- **Runs:** `runs/e2_cohort32_r2_{a,b}` and `runs/e3_broader40_r2_{a,b,c}`.
- **References:** `runs/e2_cohort32_off_v16` and `runs/e3_broader40_off_v16`.
- **Report:** [`reports/e2e3_cohorts60_r2`](../results/safety_dev/v20_development/reports/e2e3_cohorts60_r2/report.md).

| Controller | Goals | Obstacle | Boundary | Target | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| OFF | 27 | 15 | 3 | 15 | 0 |
| V16 | 45 | 6 | 3 | 6 | 0 |
| Revision 2 | 40 | 6 | 4 | 6 | 4 |

Against fresh V16:

| Change | Cases | Last decisions |
| --- | --- | --- |
| Goal to contact (5) | DV3-CRP-CV-04, DV3-CRP-CV-16, BAS-CR-RE-070, P2-L1-CRP-FIX-09, P2-L1-CRP-VAR-14 | All `out_of_contract_stop` |
| Goal to timeout (2) | BAS-CR-CV-045, CH-CR-CV-073 | - |
| Contact to goal (2) | P2-L2-CRS-VAR-14, P2-L2-HO-VAR-16 | Certified decisions |
| Contact to timeout (2) | BAS-NU-CV-043 (ends in `committed_continuation`), DV3-NT-CV-15 | - |
| Contact type changed | P2-L2-HO-VAR-14 (boundary to obstacle) | - |

Decision time: p50 0.22 s, p95 1.6 s, maximum 19.7 s. CH-CR-CV-073 took 32 min, almost all of it in signed boundary clearance against its 102-edge channel polygon.

Revision 2 fails promotion: 16 contacts against V16's 15, and five V16 goals lost to contacts. Under section 14.4 it is not run on the test set.

**Mechanism.** In every goal-to-contact loss, V20 issued uncertified stops with no target in range while V16 was in `no escape`. The stopped hull then reached the boundary or an obstacle: braked turns keep yawing and drifting at zero surge (section 13.1), and the stop was uncertified because the window already predicted contact. V16's unchecked SAC command had reached the goal in each case. The out-of-contract stop is therefore harmful against static hazards as well as targets (section 14.2). The gains, by contrast, come from certified, in-contract actions: gatekeeper replacements and committed continuations.

### 15.2 Revision 3

Revision 3 (`REVISION3_OPTIONS`, runner mode `v20_r3`) is revision 2 with `out_of_contract="v16"`, the option pre-registered in phase 1. When nothing is certified, V16's command is issued. V20 then changes a command only through a certified, in-contract action: a gatekeeper replacement without a target in range, a certified replacement of an unchecked V16 decision, or a committed continuation that passed its recheck. Revision 3 is the decomposition that isolates what the certified actions contribute.

The certificate code adds exact edge pruning (`candidate_edges` in `src/safety_v20_contingency.py`). Only boundary edges that can be nearest to some hull corner or centre in an evaluation chunk are kept: for every point of the chunk's box the nearest edge is no farther than the best edge's farthest box vertex, by convexity of segment distance. Unit tests on random trajectories in a 120-edge channel and a certificate test show identical results. On the saved CH-CR-CV-073 state a grid certification takes 3.4 s instead of 15.5 s, with identical certificates. All 50 tests pass.

Revision 3 is shaped by the probe and by the revision-2 outcomes on the 60 cases, so no development case is held out for it. The development run is reported in full, and test set v4 is the only held-out evaluation.

### 15.3 Pre-registered rule for revision 3

- **Runs:** all 72 development cases (probe and both cohorts), `v20_r3`, in chunks of 12, paired with the existing fresh OFF and V16 runs.
- **Reported:** goals, each contact type, timeouts, rescued SAC failures, lost SAC successes, every transition against fresh V16, and the last decision levels before each contact.
- **Test set v4:** runs once with fresh OFF, V16 and `v20_r3`, all 1,000 scenarios in definition order, in chunks of 25 with compact gzip traces, if and only if revision 3 has fewer contacts than fresh V16 on the 72 development cases. No change follows any test-set outcome.
