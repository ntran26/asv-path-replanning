# Reachability branch: first saved-data experiment

Date: 2026-10-03. **The offline prototype is implemented and tested; a collision-avoidance certificate is not achieved.** SAC, V16, the environment, constants and the checkpoint remain unchanged. No new policy episodes, test-set-v3 evaluations or backend installations were performed.

The research recommendation led to a separate interval prediction component, not a new numbered safety controller. Its purpose is to check whether explicitly assumed uncertainty can be propagated without excluding possible own-ship states. This experiment also checks an existing target-motion assumption. It cannot select or certify an action.

## Results

Own-ship diagnostic: saved `TS2:BAS-NU-CV-070`, V15 decision 11, using its recorded subsequent commands on that same branch. These future commands are retrospective diagnostic inputs, not commands available to a runtime verifier. Raw onboard measurements and previously issued commands initialize the calculation; scoring truth never initializes or narrows it. NU is a null-encounter label: a target exists in this case.

| Calculation | Horizon | x interval width | y interval width | Heading interval width | Result |
|---|---:|---:|---:|---:|---|
| Initial interval implementation | 0.5 s | 1.082958 m | 1.148524 m | 2.162628 rad | All six recorded state components contained |
| Algebraically tightened implementation | 0.5 s | 0.801952 m | 0.811166 m | 1.381788 rad | All six recorded state components contained |
| Initial implementation | 1.0 s | 12.531187 m | 12.773700 m | 20.096279 rad | Unknown: width guard exceeded |
| Tightened implementation | 1.0 s | 8.682193 m | 9.034630 m | 8.420416 rad | Unknown: heading width guard exceeded |

Both calculations also contain the initial recorded six-component state. There are **two scored endpoints per calculation (0 and 0.5 s), not a successful 8-second containment test**. The requested 8-second horizon stops at 1 second. The width guard is 10 m in either position coordinate or one full rotation in heading; it limits diagnostic usefulness and computation, not physical uncertainty. The 1-second boxes are retained in the artifacts, but are not counted as containment passes. No branches or intervals were pruned to make a result pass.

The algebra changes reduce the 0.5-second position widths by approximately 26% and 29%, without changing the input assumptions. The bounds remain much too broad for close obstacle clearance. This is an arithmetic improvement, **not an improvement in measured mission success**, and it does not establish that BAS-NU's successful SAC continuation is unsafe or unavoidable. The older successful-SAC forecast audit remains separate evidence: [recorded-policy-future report](../v10_iterations/audits/bas070_recorded_policy_future/REPORT.md).

Target diagnostic: saved `TS2:BAS-HO-NC-059`, V16 decisions 20 to 21. Target 0 changes heading from **104.083667 degrees to 185.968311 degrees**, or **81.884644 degrees in 0.5 s**. A global 8 degrees/s model would allow only 4 degrees over that interval and is therefore contradicted by this recording. The simulator's corridor correction can reset heading and position after ordinary target turning. This does not validate a replacement target envelope, nor supply the unrecorded target future on the successful SAC-only branch.

## Assumptions and what the computation means

- The default parameter hypothesis is the coordinate-wise range of the 28 bootstrap vectors, expanded by three times the sampler's jitter standard deviation, then sign-clipped as in the sampler. Scale is 1 and jitter fraction is 0.05. Literal bounds are stored in each `audit.json`.
- The current fresh measurement hypothesis is position +/-0.09 m, heading +/-0.6 degrees, surge/sway +/-0.15 m/s, and yaw rate +/-3 degrees/s, intersected with applicable numerical endpoint velocity limits. These are declared three-sigma hypotheses, **not full Gaussian support, confidence regions or a mission-level probability claim**. The simulator does not truncate the Gaussian draws at these limits.
- Delay is held fixed within each of five branches: 12, 13, 14, 15 or 16 integration steps of 0.05 s. Each branch reconstructs the servo and FIFO from reset's zero servo and all ten past issued commands, including the plant's first-command FIFO fill. The nominal observer's stored servo is not substituted for true actuator state. Reconstructed servo widths are about 7.25-7.85 degrees across the branches.
- The model includes the delay, exponential/rate-clipped servo, midpoint-servo RK4, endpoint velocity clips and immediate reverse-surge split. It uses a private 50-digit `mpmath.iv` context with outward conversion. Parameter and state correlations across time are lost, making the box grow conservatively for the declared real-valued map when implemented correctly.
- This is an experimental interval extension of the **real-arithmetic numerical transition**, not a proof of the continuous vessel motion or complete NumPy/libm binary64 execution. There is no validated swept hull, target occupancy, terminal safe controller, recursive feasibility or runtime deadline guarantee.
- True parameters and true servo/FIFO were not logged. An eventual containment miss could therefore indicate an incorrect conditional hypothesis or an implementation/model problem; the recording alone cannot distinguish them.

## What changed and why

The [new core](../../../src/safety_reachability.py) includes two analytically justified improvements. The servo update is monotone in state and command, with extrema at response/rate-limit endpoints. Evaluating those corners avoids repeated-variable inflation from `s + (c-s)*a`. The body calculation combines the common yaw product before interval evaluation, uses a nonnegative square for cross-flow drag, and evaluates signed-square damping by its monotone endpoint image. No uncertainty bounds, trigger margins or policy actions were tuned.

The architectural inspiration is [Kochdumper et al.'s reachable-set shield](https://arxiv.org/abs/2210.10691). This prototype does not implement its polynomial-zonotope method or inherit its theorem. The elementary algebraic arguments, other method citations and their limits are documented in the [stage-1 plan](../../../planning/SAFETY_REACHABILITY_STAGE1_PLAN.md). In particular, the retained feedback and terminal-controller obligations from [Li et al.](https://arxiv.org/html/2311.06769v1) are planned and remain unimplemented.

## Validation and provenance

**101 focused tests passed.** They cover original-plant numerical parity, forward/reverse commands, rate clipping, delay initialization/history, sampled fixed-parameter containment, analytic servo extrema, 100,000 servo samples in five meaningful interval domains, 1,000 original-RHS samples, exact yaw cancellation, signed damping, finite inputs, transactional failure, private precision, normalized command units, truth separation, periodic heading, full-bank failure handling and terminal-endpoint exclusion. Sampling can falsify an enclosure; it does not prove universal inclusion.

| Artifact | Contents |
|---|---|
| [Initial audit](initial_conditional_v1/audit.json) / [report](initial_conditional_v1/REPORT.md) | Monotone servo with the original body interval expressions; recorded computation time 0.763 s |
| [Tightened audit](algebraic_conditional_v2/audit.json) / [report](algebraic_conditional_v2/REPORT.md) | Same hypotheses plus exact body algebra; recorded computation time 11.796 s |
| `endpoints.csv` in each directory | Component bounds, eligible scoring truth and status, including the rejected wide endpoint |
| `source.zip` in each directory | Exact evaluated core/runner and relevant source snapshots |
| [Final verification](verification.json) | Artifact hashes, unchanged protected-file hashes and validation record |

These timings were collected under shared machine load and are not a controlled runtime comparison. Saved physics/parameter/target/constants hashes match current sources. The saved `env.py` differs from current `env.py`; the original archived bytes match the saved manifest. The audit records this mismatch without assuming behavioral parity, and never imports or executes either environment. Input trace, result, manifest, source archive, scenario, seed, checkpoint and config identities are retained in the JSON. Python 3.10.11, NumPy 1.26.4 and mpmath 1.3.0 were already installed.

From the project root, reproduce the current diagnostic with a **new** tag:

```powershell
python -B tools/diagnostics/safety/reachability_containment.py --tag NEW_TAG
python -B -m pytest -q tests/test_safety_reachability.py tests/test_safety_reachability_servo.py tests/test_safety_reachability_algebra.py tests/test_safety_reachability_containment.py -p no:cacheprovider
```

Existing output directories are not overwritten. Historical source archives preserve the two implementations; the current command uses the tightened implementation. `initial_conditional_v1` and `algebraic_conditional_v2` are experiment tags, **not safety-controller V1/V2**.

## Next advancement condition

The experiment identifies two requirements before online shielding: retain enough parameter/state dependence to obtain useful own-ship bounds, and enclose the complete hybrid target transition. A dependency-preserving reachability backend or justified subdivisions should be evaluated under the same declared hypotheses before changing thresholds. Neither wider margins nor the observed containment of a very wide box establishes safety.

The two V16 geometry rescues, all six lost SAC successes, blind-zone initial feasibility, complete obstacle occupancy, measurement-update containment, inter-sample motion and a verified causal terminal continuation remain outstanding gates. Continue on development data. Keep TS3 for the subsequently frozen comparison. The recommendation's main milestone—one nontrivial family with a verified collision-free continuation—has **not** been reached, and no new controller has been promoted.
