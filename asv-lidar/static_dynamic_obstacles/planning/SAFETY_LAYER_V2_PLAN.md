# Safety layer v2 — improving the runtime safety layer (plan, 2026-09-28)

> **Naming (2026-10-01).** "Supervisor" is now called the **safety layer** ("safety" for short) everywhere in documents, logs and command-line options (`--safety`, `--train-safety`, `--eval-safety`; the old flags still work). Identifiers frozen in the baseline configs and run records keep their old names, because renaming them would break the baseline-v2/v3 digests and the ability to read past results: `train_supervisor`, `eval_supervisor`, `ESTOP_TRIGGER = "supervisor"`, and the `supervisor` column in episode CSVs and `eval_summary.json`.


**Status (2026-10-01):** first version built — `src/safety_v2.py`, switch
`SAFETY_VERSION = 2` (run time; v1 stays the default), wired into `env.step`, the
frozen suite and the Paper 2 set (`--safety-version 2`); tests
`tests/test_safety_v2.py`. It is step 3 of this plan directly (a predictive filter:
commit the action 1 s, then some recovery — hard turn either way or straight at
the propulsion floor — must keep the hull clear for 6 s; else the nearest safe
grid action), which subsumes steps 1–2's trigger. **First check (kept 3 M policy,
100 L1/L2 fixed-speed episodes; a smoke check, not a tuning set):** goals 58 → 57;
static-obstacle collisions 26 → 15 but boundary collisions 6 → 18, target
collisions 10 → 10; L2 head-on 0.00 → 0.15, L1 head-on 1.00 → 0.85. It steers off
obstacles into walls and does not yet prevent target contacts. **Next iteration:**
a braking recovery (full astern, the one action only the safety layer has, with
the hull's braking model), a longer look-ahead for the boundary, keep the policy's
action when nothing is safe, intervention hysteresis — tuned on the development
sets only (never the Paper 2 set or the frozen suite). Cost: 1.9 → 3.1 s per
episode in evaluation.

**Iterations 2–6 (2026-10-01, evening).** Development set only: the v3 three-panel
development set (150 episodes), kept 3 M SAC policy; `results/safety_v2_dev_*.csv`.

| version | goals | obstacle | boundary | target | timeout | interventions / ep | brake steps / ep |
|---|---|---|---|---|---|---|---|
| safety off | 111 | 26 | 2 | 11 | 0 | – | – |
| iter 2 (braking recovery, look-ahead 8 s, hysteresis) | **115** | 17 | 2 | 13 | 3 | 6.0 | 0.9 |
| iter 3 (+ run-out check, ego smoothing, trigger 0.5, turn preferred) | 107 | 10 | 9 | 10 | 14 | 32.3 | 12.7 |
| iter 5 (trigger 0.15, run-out scaled with speed) | 105 | 15 | 13 | 10 | 7 | 18.8 | 9.0 |
| iter 6 (braking only with a target within 7 m) | 105 | 19 | 17 | 9 | 0 | 12.7 | 2.3 |
| iter 6, no ego smoothing | 94 | 29 | 17 | 10 | 0 | 11.7 | 2.5 |
| iter 6, no run-out against the boundary (**current default**) | 109 | 18 | 14 | 9 | 0 | 11.4 | 2.4 |

What was learned:
* **The dead-ahead pathology (iter 2) is fixed.** Holding straight at an obstacle,
  the margin fell from "fine" to "no escape" in one step, so the filter could only
  brake.  Two causes: the measured yaw rate's noise (1 deg/s) swung the predicted
  margin by ±0.4 m step to step (fixed by smoothing the ego state — removing it
  costs 15 goals), and a fixed 8 s horizon rewarded slowing (less distance run,
  so the obstacle fell past the end; fixed by a terminal run-out check of
  min(2 m, 3 s × speed) along the final heading).  `tests/test_safety_v2.py`
  now checks the outcome: past the obstacle, by turning, speed kept above 0.3 m/s.
* **Braking against static obstacles traps the hull.** With no astern motion in
  the simulator, a hull braked to a stop beside a wall or obstacle has no exit;
  the deadlocks (iter 3: 14 timeouts at ~76 brake steps each) and iter 5's
  boundary contacts were this.  Braking is now kept for traffic only — a target
  moves on, a wall does not — which also leans less on the unidentified reverse
  thrust (see the caveat below).
* **Open: boundary contacts after interventions (14 vs 2).** Traced case
  DV3-BO-CV-04: after a series of slowing overrides the hull ends nearly stopped
  near the goal, heading into the west wall at 0.01 m/s; the policy, never trained
  with an override, keeps the throttle at the floor and drifts in.  The
  minimal-change rule (distance to the policy's action) still favours slowing
  near walls, and every slow state is out of the policy's training distribution.
* **Overall:** no version is clearly better than the policy alone on goals (iter 2's
  +4 / 150 is within the binomial noise, ±5).  The filter reliably removes
  obstacle contacts (26 → 10–19) but trades them for boundary contacts or stalls.

**Reverse-thrust caveat.** Reverse thrust (S2 0 to −100) is not identified: no
field log commands it, and the simulator assumes 0.5 × the forward thrust law
with no reversal delay, no astern motion, no prop walk and no loss of rudder
wash (`ship.REVERSE_THRUST_EFFICIENCY`).  The filter predicts braking
pessimistically (0.25 × the forward law, nothing for the first 0.75 s), but any
result that relies on braking — v1 entirely, v2 against traffic — carries this
caveat until the basin bench check P-8 and crash stops S1-C2 measure it.

**Next options (decision pending):** (a) freeze iter 2 or the current default
and report it honestly as a marginal add-on; (b) train or fine-tune with the
filter in the loop (`--train-safety`, `SAFETY_VERSION = 2`) so the policy learns
the states it leaves behind — the root cause of the remaining losses; (c) keep
tuning the selection rule (e.g. no slowing as an avoidance against static
obstacles), with diminishing returns so far.

**Earlier status:** proposal, not built. Parked while baseline-v3 SAC seed 0 trains (the
evaluations would slow it; building and unit tests would not). Open item A38 in
`OPEN_PROBLEMS.md`.

## 1. Why: the safety layer currently makes things worse

The safety layer (`src/emergency_stop.py`, trigger `stop_required`, latch run by
`env._emergency_stop_override`) is a runtime layer outside the policy. It is off
in training (`train_supervisor: off`), so no policy depends on it and it can be
changed and re-evaluated **without retraining**. The same module runs in the
deployment bridge (`field_deployment/udp_live_rl.py`), so an improvement is also
the field safety layer.

Seed-0 frozen suite (suite 3.4, baseline-v2, constant-velocity targets, 800
episodes per learner), safety layer off -> on:

| Learner | Episodes with a stop | Goal off -> on | Target collisions off -> on |
|---|---|---|---|
| PPO | 2.6 % | 0.807 -> 0.796 | 0.142 -> 0.152 |
| RecurrentPPO | 0.1 % | 0.736 -> 0.736 | 0.198 -> 0.198 |
| SAC | 2.8 % | 0.892 -> 0.876 | 0.064 -> 0.066 |
| TQC | 1.9 % | 0.873 -> 0.863 | 0.054 -> 0.059 |

Episode-level diagnosis (on/off pairs by test ID):

| Learner | Stopped | Lost (goal off, fail on) | Gained | How the lost ones end | Target collisions (on) with no stop |
|---|---|---|---|---|---|
| PPO | 43 | 16 (all overtaking) | 1 | 9 boundary, 7 target | 158 of 167 |
| SAC | 46 | 25 (all overtaking) | 4 | 22 boundary, 3 target | 95 of 98 |
| TQC | 30 | 17 (16 overtaking) | 1 | 10 boundary, 7 target | 73 of 82 |

Three causes:

1. **The trigger is a rule test, not a collision test.** It fires when a
   give-way encounter is `in_extremis` (perceived DCPA inside the ship-domain
   `d_req`, about 2.4 m, with TCPA < 5 s), the compliant alteration is
   inadmissible and `stop_clears`. In practice that is overtaking in narrow
   water, at predicted DCPA 1.5–2.4 m — often a pass that would clear the hull.
   Most stops are false alarms.
2. **The stop itself is unsafe.** Only throttle is overridden (full astern, then
   hold); the policy keeps steering throughout, in states it never trained on
   (reversing, near-zero speed). On release it gets control back mid-channel at
   near-zero speed. Lost episodes end mostly in **boundary** collisions.
3. **It cannot act where collisions happen.** 90–97 % of target collisions with
   it on had no stop at all — mostly crossing and head-on, where a stop cannot
   clear by design (A18: stopping in a reciprocal target's path preserves the
   conflict), and it has no other response.

## 2. Proposed changes (no retraining)

Built as a separately versioned safety layer applied at evaluation only. Its
current parameters (`ESTOP_*`, `EMERGENCY_STOP_ENABLED`) are in `constants.py` and
recorded in `configs/baseline_v2.json`, so editing them there would fail the
baseline-v2 check. v2 lives in its own module/config (e.g. `src/safety_v2.py`,
`configs/safety_v2.json`), selected by a flag.

### Step 1 — a safety-only trigger (~0.5 day)

- Fire on **physical** risk: predicted DCPA below hull contact plus a margin
  (about 1.1–1.4 m; `ESTOP_CLEAR_DCPA_M`-like, not `d_req`) with a
  time-to-collision under a few seconds. Leave rule judgement (passing side,
  admissibility) to the policy, where the reward already handles it.
- Add a cooldown / re-trigger hysteresis (the logs show stop -> release -> stop
  within one episode).
- Expected effect: removes most of the false stops in narrow-channel overtaking.

### Step 2 — a proper stop and hand-back (~0.5 day)

- During BRAKING/HOLDING, the safety layer also owns the rudder: heading hold, or
  steer away from the nearer boundary (map ray-cast is available).
- On release, a short **recovery phase**: LOS guidance back toward the path at low
  speed; return control to the policy only once speed and cross-track error are
  back inside the training distribution.

### Step 3 — a safety filter that can turn, not only stop (~1–2 days; highest payoff)

Each step, check the policy's action by short-horizon forward simulation:

- Own ship: the 3-DOF vessel model (`ship.py`) with actuator dynamics.
- Target: the **tracked** state (perceived, never ground truth) at constant velocity.
- Static: map boundary and LiDAR-detected panels.
- Horizon about 5–8 s at 2 Hz.

If the policy's action is safe, pass it through untouched. If not, choose the safe
action **closest to it** from a sampled rudder x throttle grid, preferring
starboard for head-on and crossing (Rules 17(b) and 2(b) permit departures in
extremis). Stop only if nothing else is safe. This is the dynamic-window idea used
as a shield (simplex / runtime assurance); `src/classical/los_dwa.py`'s window
evaluation can be reused.

### Classical options for step 3

| Method | Fit | Cost |
|---|---|---|
| **Sampled forward simulation (DWA as a filter)** — recommended | Reuses our vessel model and DWA code; handles walls, panels and target together | Lowest |
| Velocity-obstacle override | Reuse `colregs_vo` / `encounter_vo` as the fallback controller; rule-aware by construction | Low; point-mass model, needs margins for turning lag. Mind A35 (side rule per candidate) |
| Control barrier function (CBF-QP) | Minimal change to the policy action with a formal safety condition; "safe RL" framing | Medium; simplified model and careful tuning for an underactuated hull |
| MPC predictive safety filter | Plans a recoverable trajectory; most principled | Highest; 2 Hz online optimisation is feasible but heavy to build and verify |

## 3. Evaluation protocol

- **Tune on the development set only**; the frozen suite and the Paper 2 set are
  run once per safety layer version (each seed-0 policy: about 1 h for the frozen
  suite).
- Report per learner: intervention rate; **false-alarm rate** (interventions in
  episodes that succeed with it off); collisions prevented; episodes made worse;
  collision type after intervention (target / boundary / obstacle).
- Learned compliance stays reported **with the safety layer off** (claim C-7);
  safety-layer-on results are system-level results. Introduction draft 4's C4
  already separates "intervention dependence".
- Unit tests: trigger fires on a contact-course scenario and not on a clear
  overtaking pass; the filter passes safe actions unchanged; rudder ownership
  during a stop; recovery hands back only inside the training envelope.

## 4. Later (needs retraining; not part of this plan)

Training with the shield on (shielded RL), or penalising interventions, would let
the policy learn around the safety layer — but it changes what "the learned policy"
means for the paper's claims. Future work.

## 5. Order and cost

1 and 2 first (about a day together, both cheap to test on the existing seed-0
policies), then 3 as the DWA-style filter. Evaluation waits for a gap in training
(baseline-v3 SAC seed 0 finishes about Wednesday 30 Sep).
