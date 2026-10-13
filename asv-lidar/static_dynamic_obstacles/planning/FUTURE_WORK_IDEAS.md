# Future work ideas

A running list of ideas beyond Paper 3, each with enough detail to judge whether it
makes a paper. Status, date and source are kept per entry.

---

## 1. Closing the sim-to-field gap with logged field runs

**Status:** idea, not started (2026-10-13). **Candidate for a short conference paper**
(about 5 pages): one policy, one question, one comparison.

**Question.** Does a small amount of field data, logged while the deployed policy runs
on the vessel, close the gap between its simulated and its field performance better than
simulation alone?

**Starting point.** The Paper 3 deployed policy (SAC, or TQC if it is chosen on test
set v4), trained only in simulation, and its field trials. The vessel-model
recalibration (`05_VESSEL_MODEL_AND_SIM2REAL.md`, sections 5-6) already plans to replay
the Paper 2 field logs through the calibrated model; this idea goes one step further and
uses the logs to update the policy itself.

**Two routes, compared on the same field data:**

| Route | What changes | How |
|---|---|---|
| A. Fine-tune in a field-calibrated simulator | the simulator | Re-identify the dynamics and sensor noise from the logs (05 sections 5-6), then continue training the deployed policy there for a short budget, warm-started from its checkpoint (a new, recorded formulation variant, since the simulator changes) |
| B. Offline RL from the field logs | the policy, directly from field transitions | Rebuild transitions (observation, action, reward, next observation) from the logs with the same observation builder and reward function, then fine-tune offline with a conservative method that stays close to the logged behaviour, alone or mixed with the simulated replay buffer |

Reference baselines: the simulation-only policy (no update), and route A without the
policy update (the calibrated simulator used for evaluation only).

**Methods to check in the literature first:** conservative offline RL (CQL, Kumar et al.
2020; IQL, Kostrikov et al. 2022; TD3+BC, Fujimoto and Gu 2021), online RL that mixes
in offline data (RLPD, Ball et al. 2023), and sim-to-real fine-tuning or residual
policies for marine vessels. These are pointers, not yet a verified reading list.

**What the logs must carry.** Everything the observation and reward need, on one clock
at the 2 Hz decision rate: pose (position, heading), ego speed and yaw rate (ideally the
raw IMU of 05 section 4.7), the raw LiDAR ranges (pooled offline), the path, target-ship
tracks (Paper 3 trials), and the commanded rudder and throttle. The Paper 2 logs
(`asv-lidar/field_deployment/<date>/*.log`: pose, LiDAR, rudder and thruster commands,
static obstacles only) could serve a static-obstacle pilot of route B before any
target-ship trial.

**Evaluation.** Field success, cross-track error, minimum clearance and rudder rate
before and after the update, on layouts held out from the update data; the gap between
the simulated and field trajectories of the same command sequence (the 05 section 5
replay figure); and how the gain grows with the amount of field data (one ablation).

**Why it fits five pages.** One policy, one table (no update / route A / route B), one
trajectory-overlay figure, one data-amount ablation. The formulation, training and
simulator are Paper 3's and are cited, not re-described.

**Risks.**
- Little data: a 90 s run is 180 decisions, so even 50 runs give under 10,000
  transitions; offline updates on so little data can degrade the policy.
- Offline evaluation is unreliable, so every update needs a field (basin) re-test; an
  updated policy is first checked in the calibrated simulator and the safety layer stays
  available in the field.
- The reward needs quantities the field only estimates (target-ship state from the
  tracker); the reward is computed from what was observed, which differs from training.

**Depends on:** Paper 3's deployed policy and field trials, the logging specification
(05 section 4.7), and the calibrated vessel model.
