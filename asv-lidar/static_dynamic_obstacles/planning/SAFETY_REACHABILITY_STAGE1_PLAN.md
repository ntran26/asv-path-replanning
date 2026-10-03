# Reachability safety branch: stage 1 and advancement plan

Date: 2026-10-03. Status: offline, conditional model-enclosure prototype. This document does not promote a new controller or report a mission-safety guarantee.

## 1. Decision and scope

Keep the SAC baseline 3 checkpoint and V16 unchanged. Treat their commands, and any CEM/backup search, as proposals. The proposed branch will eventually accept a command only through a separately checked reachable-set argument: a set containing every permitted future state must avoid all forbidden occupancy, and an implementable safe continuation must remain available.

The immediate task is smaller: determine whether a conservative interval calculation can enclose the own-ship transition under explicit uncertainty assumptions, and falsify an incomplete target-motion assumption using existing recordings. No new episodes, backend installations, training changes, environment edits, or changes to `constants.py` are part of stage 1. The Paper 2 project remains read-only.

**Stage 1 provides no action certificate and no mission certificate.** A recorded trajectory lying inside a computed set is a diagnostic observation; it does not establish universal containment or collision avoidance. A useful eventual guarantee will be conditional on admission to a verified recoverable region, the uncertainty contract remaining valid, and correct execution of the certified controller. Mission completion and collision avoidance are separate properties.

## 2. Architecture to work toward

1. **Candidate generator:** SAC, V16, or another planner proposes an action and a continuation. Its prediction and score are not evidence of safety.
2. **State and world contract:** maintain sets for current vessel state, actuator history, fixed plant parameters, disturbances, and obstacle/target occupancy. State what each set is guaranteed to contain and under which assumptions.
3. **Reachable-set checker:** enclose the entire resulting motion, including hull geometry and relevant hybrid events. Reject or return unknown when the enclosure cannot establish the required separation. An enclosure touching an obstacle does not prove the true vessel must collide.
4. **Retained feedback continuation:** before executing a newly accepted action, retain a controller that remains valid for every allowed subsequent observation and state. If a new proposal cannot be verified, execute this retained controller.
5. **Terminal obligation:** the retained controller must lead to a condition with a justified safe continuation for the remaining mission. A finite horizon alone is insufficient. Stopping is insufficient if a moving target can later strike the vessel or if the stopped location is unsafe.

This follows the distinction between finite-horizon action shielding and persistent safety made by [Kochdumper et al.](https://arxiv.org/pdf/2210.10691), and the retained robust-feedback construction of [Li et al.](https://arxiv.org/html/2311.06769v1). It is an intended architecture, not a claim that those theorems already apply here.

## 3. Exact scope of the stage-1 calculation

The prototype evaluates an interval extension of a **real-arithmetic RK4 transition map** corresponding to the own-ship equations. It includes the seven plant coordinates (surge, sway, yaw rate, heading, rudder angle, and position), with the command FIFO and its history treated as additional transition state. The declared input sequence is fixed for a particular enclosure calculation. Interval arithmetic expands each supported operation to contain its real-arithmetic results.

The available `mpmath.iv` backend is suitable for an initial offline experiment, but its documentation calls interval support experimental. Every operation used must be supported and tested. An interval implementation's mathematical inclusion property does not by itself verify the program that assembles those operations. See [mpmath interval-context documentation](https://mpmath.readthedocs.io/en/latest/contexts.html#arbitrary-precision-interval-arithmetic-iv).

The following distinctions are mandatory:

- Enclosing the algebraic RK4 map is not a validated enclosure of the continuous differential-equation flow. RK4 truncation error and motion between grid points have not been bounded.
- Enclosing real-arithmetic operations is not a proof of containment of the original NumPy/libm binary64 implementation. Its rounding, branch, and conversion errors have not been bounded as a whole.
- Decimal strings and binary floating-point inputs describe different initial numbers. Record input conversion semantics, arithmetic precision, and the outward-enclosure convention.
- Plant parameters are fixed within an episode. Reusing their intervals in a natural interval extension loses dependence between repeated occurrences and can enlarge the result. This is conservative for the declared real-arithmetic map if the implementation is correct; it does not create evidence that parameters actually vary every substep.
- Each possible quantized rudder-delay length is a separate, fixed branch throughout a forecast. Union the results of these branches. Do not change delay length between steps as though a different plant were sampled repeatedly. FIFO initialization and any timestep-dependent reinitialization need explicit treatment.
- The raw measurement envelope and the parameter-jitter envelope described below are **conditional hypotheses**, not deterministic support bounds or probability guarantees.

Report stage-1 outputs as conditional containment observed, containment violated, or inconclusive/unsupported, accompanied by the exact assumed sets. Do not label a command safe, unsafe in every possible realization, certified, or mission-completing from these outputs.

## 4. Source and assumption audit

The audit must bind each experiment to source hashes, runtime settings, and saved inputs. These are the current issues the next stages must resolve; a larger interval alone does not resolve an incorrect transition or geometry definition.

| Item | Repository evidence | Consequence and required treatment |
| --- | --- | --- |
| Identified plant and reverse braking | `src/ship.py`, especially `ShipModel.update`; `bluefin/dynamics.py` and `bluefin/ship_model_v3.py` | The plant substeps at no more than 0.05 s. Negative commands add operator-split surge braking and clip surge at zero; this does not model astern motion. Include clipping and braking in the map being checked. Real-vessel reverse performance and delay require independent bounds. Commands such as -24 are simulator rpm-units, not a measured shaft speed. |
| Randomized parameters | `bluefin/ship_model_v3.py:137-170`, `sample_params` | A bootstrap mixture receives Gaussian jitter and a sign clamp. A bootstrap min/max box expanded by three jitter standard deviations is an explicitly assumed box. The code supplies no hard truncation at that box. Preserve fixed parameters across the rollout. |
| Raw ego noise | `src/env.py:1147-1155` | Direct Gaussian additions affect measured surge, sway, and yaw rate. Raw plus/minus three standard deviations is only a chosen conditional envelope. It neither bounds every sample nor directly bounds the observer's processed state, position, or hidden actuator state. |
| Hidden actuator state | `src/ship.py:174-237`; saved BAS-NU initial-state decomposition | Pose and body velocities alone do not determine a future trajectory. Rudder angle, FIFO contents, FIFO timestep, and parameter-dependent delay matter. Saved records do not justify inventing an exact physical servo/FIFO state. |
| Full target transition | `src/targets.py:149-170`, `_react_vo`, `clamp_to_corridor:317-352`; `src/env.py:1365-1373` | Speed profiles, reactive modes, stopping, and confinement are hybrid operations. The 8 degrees/s turn limit in the reactive rule is not a global heading bound. Corridor clamping can snap heading to a tangent and shift position; `stop_box` can stop the target at its previous position after checking its proposed full hull. Enclose the full ordering, including dependence on own-ship motion. |
| Boundary predicate | `src/env.py:208-216,1193-1201`; `src/classical/common.py:352-369` | The environment's border test uses the channel's global x envelope and y range, whereas the filter uses unsigned point-to-segment distances minus hull support. These are not equivalent to each other or to full containment in a bent channel polygon. Unsigned distance can become positive again outside a wall. State the desired property and check its relation to simulator flags explicitly. |
| Hull and world occupancy | `src/env.py:1206-1210`; target hull construction; observation and tracking code | Static-object and target collision predicates use geometry, not only centre distance. Sparse visible LiDAR returns do not automatically cover unseen surfaces or all target occupancy. Position, heading, dimensions, association, occlusion, births, and permitted target behavior need a containment contract. |
| Timing and future feedback | Environment step ordering and saved decision records | The proposal cadence, integration cadence, contact checks, computation deadline, and applied action must be distinguished. A fixed open-loop command sequence does not establish that a future observation-dependent policy remains inside its tube. |

The Gaussian issue is structural: NumPy's `scale` is a standard deviation, not a finite bound. The declared Gaussian noise model is not hard-truncated at three standard deviations. Stage 1 makes **no probability statement** about its chosen envelopes. A future deterministic claim requires an independently justified bounded-error contract; a chance-constrained claim requires a separate probability model and a mission-level risk argument. See [NumPy `Generator.normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html).

For boundary safety, keep both the simulator termination predicate and intended physical channel containment visible in diagnostics. Do not silently repair one by changing the environment during this branch. For deployment, the desired physical property must ultimately be explicit and continuously checked against enclosed hull occupancy.

## 5. Two saved-data experiments

### BAS-NU: conditional own-ship containment

Use the saved own-ship prefix and its declared commands to compare the conditional enclosure against the recorded trajectory. Report horizon, box widths, assumed hidden state, parameter box, delay branches, and the first containment failure or numerical loss of usefulness. A very wide enclosing box is weak evidence of practical usefulness, even if every saved point lies inside it.

The attachment's description of BAS-NU as a no-target case is incorrect. Here NU denotes the null-encounter class, and the existing `results/safety_dev/v10_iterations/audits/bas070_recorded_policy_future/REPORT.md` explicitly scores target **Track 86**. Its recorded policy continuation succeeded, but its sampled states are not a continuous collision-avoidance proof. Stage 1's own-ship calculation does not certify that target interaction. The saved initial-state decomposition also notes missing true plant parameters and actuator state.

### BAS-HO: falsify the global 8 degrees/s target assumption

Use the saved first-divergence target trace to test the proposed yaw-rate envelope. Existing analysis reports an approximately 81.88-degree heading change over 0.5 s, which contradicts a global 8 degrees/s heading bound. The source-level corridor clamp provides a mechanism for this discontinuity; increasing a smooth-motion margin does not fix the omitted hybrid operation.

This test falsifies that candidate assumption. It does not validate a replacement target reachable set. A target reacts to own-ship behavior, so a future recorded under V16 is not automatically the counterfactual future under SAC. Use the observed trace only for the branch and inputs that produced it.

No fresh rollout is required for either experiment. The implementation and result links will accompany the completed stage-1 artifacts; results must retain their conditional labels.

## 6. Method sources and limits of transfer

- **Action-dependent reachable sets:** [Kochdumper et al., *Provably Safe Reinforcement Learning via Action Projection Using Reachability Analysis and Polynomial Zonotopes*](https://doi.org/10.1109/OJCSYS.2023.3256305). Inspiration: a separate action check using conservative sets and explicit uncertainty. Section VII-A distinguishes its basic finite-horizon shield from an infinite-horizon guarantee requiring an additional failsafe or safe final state. The paper computes reachable sets using CORA and uses optimization for projection. The present interval-map prototype does not reproduce its polynomial-zonotope method or inherit its theorem.
- **Retained feedback and terminal safety:** [Li et al., *Learning Predictive Safety Filter via Decomposition of Robust Invariant Set*](https://arxiv.org/html/2311.06769v1). Inspiration: verify a feedback tube, retain the previous verified feedback if verification fails, and use a terminal controller afterward. Theorem 5 requires feasible initial verification and the paper's uncertainty and model assumptions. Its terminal region need not itself be forward invariant: the terminal controller must keep trajectories in the overall allowed state set forever from that region. Do not reduce this requirement to an arbitrary low-speed endpoint.
- **Output feedback:** [van Wijk et al., *Output Feedback Backup Control Barrier Functions*](https://arxiv.org/html/2604.19893v1). Inspiration: couple the true-state and estimated-state flows under bounded measurement error and a verified backup. This is a 2026 preprint. Its formulation explicitly excludes process disturbances to isolate estimation effects. It does not directly prove safety for this uncertain plant plus observer; bounded estimation error and combined disturbance treatment remain obligations.
- **Reach-avoid continuation:** [Hsu et al., *Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning*](https://doi.org/10.15607/RSS.2021.XVII.077), [paper](https://arxiv.org/pdf/2112.12288). Inspiration: treat a learned controller as a candidate and verify the reach-avoid rollout. The paper itself measures false-success predictions by learned value functions. Its rollout argument needs an accurate model or adequate bounded model error and a suitable initial region. A nominal rollout with the errors already observed here does not meet that obligation.
- **Precomputed tracking tubes and failsafe admission:** [Shao et al., *Reachability-based Trajectory Safeguard*](https://arxiv.org/pdf/2011.08421). Inspiration: include tracking error in a trajectory tube and begin with an available safe failsafe maneuver. Its theorem requires a valid error reachable set and its stated sensing/obstacle assumptions. Its trajectory-parameter interface and static-obstacle setup are not a drop-in wrapper for SAC's direct rudder/throttle actions and reactive targets.
- **Marine safety versus mission performance:** [Krasowski and Althoff, *Provable Traffic Rule Compliance in Safe Reinforcement Learning on the Open Sea*](https://doi.org/10.1109/TIV.2024.3400597), [paper](https://arxiv.org/html/2402.08502v2). Inspiration: make encounter assumptions and rule logic explicit. On the paper's 600 handcrafted cases, verification reduced observed collisions from 3.13% to 0%, while goal completion fell from 86.8% to 44.0%. Its open-sea, target-state, and rule assumptions differ from this walled LiDAR problem; the figures are an empirical tradeoff, not proof that this controller satisfies COLREGs or preserves successful SAC missions.

## 7. Stages 2-5 and advancement gates

### Stage 2: validated finite-horizon motion and geometry

Choose the object to certify: the discrete simulator transition, physical continuous motion, or both with an explicit error bridge. Validate each relevant branch, including command conversions, clipping, delay history, parameter reuse, target reaction/clamp/stop, and full-hull swept occupancy. Bound arithmetic and integration errors at the claimed level. Define boundary containment and prove that the checked geometry is conservative for it.

**Gate:** demonstrate nontrivial finite-horizon separation with a defensible enclosure, and reject deliberately violating or unsupported cases. A matching nominal endpoint or a box containing saved samples does not pass this gate. No runtime integration before the transition and geometry contracts are reviewable.

### Stage 3: uncertainty update and implementable feedback

Construct an initial admissible state set and a measurement-update rule that preserves containment. Include hidden actuator state, fixed-parameter dependence, and permitted world evolution. Evaluate observer, controller, and true plant together; the future controller must receive only information available onboard. Reduce interval overexpansion using a justified representation, subdivision, or verified feedback, not by shrinking bounds to fit traces.

**Gate:** repeated updates preserve the stated containment contract, and the same causal feedback implementation can execute the continuation whose tube was checked. Unsupported observations, stale inputs, or deadline misses need an explicit certified response. An out-of-contract event invalidates the conditional guarantee; it is not silently absorbed by a tuned margin.

### Stage 4: terminal continuation and recursive feasibility

Verify a terminal controller or a finite mission-end condition under the permitted future occupancy. Establish admission: some initial states may be outside the region from which collision avoidance can be assured. Preserve a previously verified feedback continuation across failed searches and show that each newly accepted first action leaves a valid continuation. Include action saturation, execution delay, and computation failures in that argument.

**Gate: at least one nontrivial, independently checked safe continuation must be demonstrated under the full declared contract. This has not yet been achieved.** A finite collision-free prefix, a nominal braking plan, or zero observed collisions does not substitute for it. State explicitly whether the theorem is finite-mission or persistent, and what happens after simulated goal termination on a physical vessel.

### Stage 5: runtime integration and evaluation

Only after the earlier gates, integrate the checker as a separate selectable branch while preserving SAC and V16 baselines. Develop with allowed development data, then freeze the method before the main test-set-v3 comparison. Any tuning on a reused evaluation set must be disclosed. Keep evaluation bounded by the current user authorization and protect the existing training process.

**Gate:** paired reporting must show collisions, goals, timeouts, interventions, preserved SAC successes, and rescued SAC failures, along with certificate acceptance, unknown/refusal, out-of-contract events, and latency/deadline failures. Report all starts as well as admitted starts, so rejecting hard missions cannot inflate the apparent guarantee. Statistical performance supports usefulness; it does not replace the earlier proof obligations.

## 8. Reproducibility and bounded tooling

Stage-1 artifacts should record SHA-256 hashes of the plant, dynamics, parameter sampler, target logic, environment, geometry helpers, constants, proposal-controller sources, relevant checkpoint, and saved input files. Record the exact source paths, units, timestep schedule, noise/randomization settings, conditional bounds, initial hidden-state treatment, fixed delay branches, Python/NumPy/mpmath versions, interval precision, and unsupported assumptions. A changed dependency must invalidate a previously associated certificate or trigger a fresh audit.

Offline checks should exercise inclusion through nonlinear operations, actuator/clipping transitions, fixed-delay branches, heading wrap and target hybrid discontinuities, and mismatched geometry predicates. Sample comparisons can falsify an enclosure but cannot prove universal inclusion. Tests and result summaries must make that distinction explicit.

Use the already available Python interval backend for this bounded stage. CORA is a plausible later reachability backend, but it is MATLAB-based and its toolbox requirements need checking; Flow* is another possible validated-flow route with a separate native build/dependency cost. Neither is an installed, validated drop-in solution here. Installing a backend is not itself progress on the uncertainty, target, geometry, or terminal obligations. See the official [CORA repository](https://github.com/TUMcps/CORA) and [Flow* repository](https://github.com/chenxin415/flowstar).

Earlier controller history and saved evaluation evidence: [SAFETY_LAYER_COMPLETE_SUMMARY.md](SAFETY_LAYER_COMPLETE_SUMMARY.md).

## 9. Implemented files and arithmetic choices

- [Conditional numerical-map tube](../src/safety_reachability.py): private 50-decimal-digit interval context, finite explicit state/parameter inputs, a fixed integer delay branch, actuator history reconstruction, 0.05 s plant updates, and transactional rejection on invalid inputs or resource exhaustion. There is no action-selection method or environment hook.
- [Saved-data runner](../tools/diagnostics/safety/reachability_containment.py): completed-trace and source checks; an explicit separation of onboard data from scoring truth; all admissible delay branches; fixed recorded future commands; endpoint containment, width and runtime reporting. It converts the trace's normalized rudder to the plant's percentage units. It excludes terminal post-states because their physical interval can end early at contact.
- [Results and reproduction instructions](../results/safety_dev/reachability_stage1/README.md): exact assumptions, comparisons, limitations, source archives and final validation. The optional arithmetic dependency is pinned in [requirements-reachability.txt](../tools/diagnostics/safety/requirements-reachability.txt); no package installation was needed.

Two algebraic improvements reduce avoidable interval expansion. They alter the interval extension, not the underlying real-valued model or uncertainty bounds:

1. **Monotone servo image.** Let `s` be current servo, `c` the clipped command, `a = 1-exp(-h/tau)` in `[0,1]`, and `L = rate*h >= 0`. Before final angle clipping, the update equals `median(s-L, (1-a)*s+a*c, s+L)`. It is nondecreasing in `s,c`. For fixed `s,c`, it equals `s + sign(c-s)*min(a*abs(c-s), L)` and is monotone in `a,L` in the direction of the command. Its extrema therefore occur among interval endpoints. The implementation encloses all 16 corners using interval endpoint arithmetic. This removes repeated-variable inflation in `s + (c-s)*a` without discarding any point of the declared input box. Temporal parameter correlations remain lost.
2. **Exact force identities.** Evaluate sway squared as `v**2`; evaluate the monotone function `z*abs(z)` at its interval endpoints; combine the two own-ship yaw terms as `(M11-M22-N_uv)*u_eff*v`. These preserve the real-valued equations while respecting square nonnegativity, signed damping and algebraic cancellation. They are not asserted to be bit-identical NumPy evaluations. Binary64 error remains outside the claimed scope.

These are direct algebraic derivations from the identified model, not fitted corrections or demonstrations of the polynomial-zonotope algorithm. The method inspiration and its substantially stronger proof obligations remain [Kochdumper et al.](https://arxiv.org/abs/2210.10691); the arithmetic facility is documented by [mpmath](https://mpmath.readthedocs.io/en/latest/contexts.html#arbitrary-precision-interval-arithmetic-iv). Independent plant comparisons and analytic special cases can detect implementation errors, but do not replace the derivations or validate all numerical-library operations.

Even a nonempty set after a future measurement update will not prove that the true measurement error lies within the declared bound. A usable deterministic contract needs external justification of those bounds, rather than a monitor that merely fails to detect their violation.
