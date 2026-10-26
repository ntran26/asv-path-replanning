# Safety layer versions V1-V21: what each is, and how many distinct methods

Date: 2026-10-25. A catalogue for a possible paper on the design and evaluation of the safety layer. Sources: [complete safety record](SAFETY_LAYER_COMPLETE_SUMMARY.md) (V1-V19), [V20 plan](SAFETY_V20_PLAN.md), [emergency-stop protocol](SAFETY_ESTOP_PROTOCOL.md), [V21 plan](SAFETY_V21_PLAN.md). Code: `src/emergency_stop.py` (V1), `src/safety_v<N>.py` and their helper modules, `src/safety_estop.py`.

## 1. Four distinct methods

The 21 numbered versions implement **four methods that differ in kind**. Most numbered versions are components or variants inside the second.

| Method | Idea | Versions | Related literature |
| --- | --- | --- | --- |
| **A. Rule-triggered stop** | An encounter-rule test (give-way, in extremis, compliant alteration inadmissible, a stop would clear) latches braking, holds, then releases; propulsion only, the policy keeps steering. | V1 | COLREGs Rules 8(e), 17(b) |
| **B. Sampled predictive safety filter with a backup** | Every decision, forward-simulate the policy's command and a bank of alternatives, each followed by a recovery manoeuvre, with the identified vessel model; check the hull against static points, the boundary and target tracks; keep the policy's command if it passes with margin, else issue the passing command closest to it and store its plan. | V2-V19, V21 (R1-R3) | model-predictive shielding, predictive safety filters, backup controllers, dynamic window |
| **C. Certified contingency with a terminal set (gatekeeper)** | Certify at every decision a short contingency that brings the vessel to rest clear of all modelled hazards, under calibrated forecast-error allowances; commit it, and in the branches where B issues unchecked commands use only certified commands or the committed contingency. | V20 (revisions 1-3) | gatekeeper, fail-safe trajectory verification, passive motion safety |
| **D. Predicted-certain-collision stop** | Only where B has no checked plan: if every candidate command and continuation is predicted to make contact within a short window, commit to a permanent full-astern stop with the rudder centred. Layered on B. | V16/V17 + stop (all hazards; boundary and targets), V21 E1 and its static-point trigger | inevitable collision states |

The layer the paper reports is **B (V16) + D (E1)**.

Within method B the versions add or change **six kinds of component**; a paper would present these as the filter's design elements and their ablations, not as separate methods:

| Component of B | Versions |
| --- | --- |
| B1. Backup structure and search | V2 (primitive bank), V3 (stored committed plan, recovery and hand-back), V5 (feedback-course backups), V6 (cross-entropy-method trajectory search), V11 option and V15 (search with the policy's current command fixed) |
| B2. Own-ship state estimation | V4 (model-based ego observer), V17 (raw measured yaw) |
| B3. Static-point memory | V4 (free-space clearing), V19 (exact transfer of target-owned returns), V21 R4 (track-footprint exclusion, not run) |
| B4. Target perception | V6 (motion-qualified provisional tracks), V11 (persistent hypotheses), V12-V14 (source ownership, partial-extent centre), V16 (motion-axis hull completion), V18 (heading memory) |
| B5. Policy-preserving selection | V5 (policy priority, soft ranking), V7 (check the policy's command with the backup tail), V8 (remove hold-back), V9 (recheck and margin comparison), V10 (paired preservation), V15 (conditional prefix search) |
| B6. Unchecked-branch handling and hazard model | V21 R1 (braking fallback), R2 (graded buffers), R3 (scored boundary limits) |

## 2. Version by version

Parent in brackets. "Dev" results are on failure-enriched development cohorts (DV3 = 150 coupled validation episodes; the 32- and 40-case cohorts mix test set v2 and DV3); they are not comparable across rows unless the same cohort is named.

| Version | Method | What it is (change from parent) | Status and key evidence |
| --- | --- | --- | --- |
| V1 | A | Encounter-rule emergency-stop latch: IDLE -> BRAKING -> HOLDING -> policy; fires for a give-way encounter in extremis with no admissible compliant alteration and a stop that clears; propulsion only. | Baseline-v2 SAC: goal 0.892 -> 0.876, target contacts unchanged; false alarms in overtaking, no trigger for most crossings. Superseded. |
| V2 | B | Sampled predictive filter: policy command plus a rudder-throttle grid and braking rows, each followed by a recovery (turns, low propulsion, braking), 6-8 s horizon; selects the passing action closest to the policy. | DV3: 111 -> 115 goals at its best iteration (16 rescues, 12 losses). Core of all B versions. |
| V3 | B (V2) | Stores the selected plan, shifts and rechecks it each decision; recovery mode with hand-back conditions; bounded "last certificate" use of a failed plan (2 s). | DV3 108 goals; boundary contacts rise. Architecture kept. |
| V4 | B (V3) | Model-based ego-motion observer driven by the issued commands; static memory cleared only by free-space ray evidence. | DV3 123 goals (19 rescues, 7 losses). Kept. |
| V5 | B (V4) | Switchable experiments: one-decision commitment, soft infeasibility ranking, feedback-course backups, sideslip compensation, policy priority. | DV3 up to 123 (no gain over V4); quick-24 no gain. Not promoted. |
| V6 | B (V4) | Motion-qualified provisional target views before tracker publication; CEM trajectory search when the bank has no passing plan. | 50-case dev: 23 -> 33 goals against V4. Kept. |
| V7 | B (V6) | Before overriding, checks the policy's current command followed by the proposal's backup tail; preserves the policy if it passes; shadow risk monitor. | Test set v2 reconstruction: 904/1,000 (58 rescues, 26 losses). Kept. |
| V8 | B (V7) | Removes the "hold back" fallback (selecting a failing candidate that only delays contact). | Test set v2: 912/1,000 (57 rescues, 17 losses). Kept. |
| V9 | B (V8) | Rechecks the override against the current snapshot; keeps the policy if its backup has at least the proposal's clearance; optional target-turn ensemble (off). | 32-case dev: 16 = V8's 16 (fewer interventions). Not promoted. |
| V10 | B (V9) | V8's proposal with V9's paired checker: keep the policy if its backup passes and the proposal fails, or both pass with the policy's clearance at least as large. | 32-case dev: 18/32. Kept. |
| V11 | B (V10) | Independently admitted moving-target hypotheses (full-length extent, coherent endpoint motion) with bounded persistence after track loss; optional policy-prefix search. | 32-case dev: 18/32. Kept. |
| V12 | B (V11) | Prefers the base tracker's view over a persistent one for the same source. | 4-case probe only. Not promoted. |
| V13 | B (V11) | Partial-hull centre continuity within the observed extent; fixed source ownership. | 4-case probe only. Base of V14. |
| V14 | B (V13) | Prefers the corrected persistent estimate for a matching source. | 32-case dev: 22/32. Kept. |
| V15 | B (V14) | Searches for a passing plan with the policy's current command fixed, only in branches where the parent fails or passes below the trigger margin. | 32-case dev: 23/32. Kept. |
| V16 | B (V15) | Completes a partially observed target hull along its measured motion axis with the known dimensions. | 32-case dev: 25/32; 40-case: 25/40; test set v4: 903-904 goals, 96-97 contacts. **Reference filter.** |
| V17 | B (V16) | Uses the raw measured yaw rate instead of the observer's on fresh frames. | 9-case probe: = V16. Not promoted. |
| V18 | B (V16) | Bounded memory of a validated hull heading when the fit is lost. | 12-case probe: = V16, no command changed. Not promoted. |
| V19 | B (V16) | Removes from static memory exact current returns owned by a qualified moving target. | 12-case probe and 18-case test-set subset: commands identical to V16. Not promoted. |
| V20 | C (over V16) | Certified stop-terminal contingency with calibrated forecast-error allowances; gatekeeper over V16's unchecked branches; revisions 1-3 (extended tails, traffic gating, no uncertified stops). | 72-case dev: never fewer contacts than V16; test set v4 (revision 2): 881 goals, 96 contacts, 23 timeouts. Stopped (rest is not a safe terminal set here). |
| (V16/V17 + stop) | D (over V16, V17) | Permanent emergency stop when every command and continuation is predicted to make contact within W; all-hazards trigger (W = 1.5 s) and boundary-and-targets trigger (W = 2.5 s). | Test set v4, boundary and targets: 892-893 goals, 77-78 own-ship contacts, 8 target strikes, 21 stops. |
| V21 | B and D (over V16) | R1 braking fallback at "no escape"; R2 graded buffers; R3 scored boundary limits; R4 track-footprint exclusion; E1 stop detector on the collision hull; E1 with static points in the trigger. | Validation: R1-R3 neutral, R4 unsafe, **E1 adopted** (test set v4: own-vessel collisions 135 -> 70, 868 goals, 52 stops). |

## 3. For a paper on the safety layer

- **Designs to compare:** A (V1), B (V16), C (V20 revision 2), B + D (E1), each on the deployed policy and on the same episodes. V1 has been measured only with a baseline-v2 policy, and V5-V15 only on development cohorts, so the comparison needs fresh runs.
- **Ablations of B:** the components of section 1 (observer, free-space memory, CEM search, target-perception corrections, policy-preserving selection), switched off one at a time from V16; most are constructor options already.
- **Evaluation data:** test set v4 has been used for safety-layer evaluation (2026-10-10 and 2026-10-25) and test set v2 for development, so a safety paper needs a fresh held-out set; the validation set and the 72 development cases remain development data.
- **Objective:** own-vessel collisions first (decision 2026-10-25), with stops, lost goals and target strikes reported beside them; and the evidence of sections 11-12 of the safety record on why no version reaches zero (forecast error, target behaviour, stopping distance and drift, no verified invariant set).
- **Offline analyses that support it** but are not controllers: the first-fire counterfactual oracle (V4/V7 branches), the SAC-critic gate study, the reachability stage-1 interval prototype, and the V21 replay probe and audits.
