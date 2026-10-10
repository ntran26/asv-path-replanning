# Emergency-stop evaluation protocol for V16 and V17

Date: 2026-10-10. Status: protocol fixed before any episode; the detector window is chosen on saved development data by the rule in section 4 before any episode runs.

## 1. Purpose and user decisions

The earlier objective counted every non-goal outcome against a filter. The user relaxed it: a collision that cannot be avoided may instead end in a permanent stop. A stopped vessel counts as better than a collision, like a timeout. The user's rules and decisions (2026-10-10):

1. **Trigger:** if a collision is guaranteed, stop the vessel. "Guaranteed" is decided onboard: every candidate command that the filter checks, each followed by every continuation in a fixed bank (holding the command, braking at once, turning then braking, turning, cruising then braking), is predicted to make physical contact within a short window *W*, under both reverse-thrust models. *W* is fixed from development data (section 4). No simulator truth enters the decision.
2. **Stop manoeuvre:** full astern with the rudder centred; the environment's existing brake-to-zero conversion applies. The stop is permanent and the vessel never resumes.
3. **Targets:** once the own ship commits to the stop, every target ship brakes in a straight line at 0.471 m/s² (the simulator's strongest own-ship braking: the reverse-thrust unit at efficiency 0.5) until it is at rest, ignoring its behaviour model. If a target still strikes the own ship, the outcome is **collision by target ship** (`collision:by_target`).
4. **Episode end:** after the commitment, the episode ends as **stopped** once the own ship is still (true planar speed at most 0.05 m/s and true yaw rate at most 1 deg/s) and every target has finished braking. The target condition keeps rule 3 meaningful, because a braking target could otherwise still reach the stopped ship after the episode was scored; braking from 1 m/s takes about 2 s. Simulator truth is used here only for episode termination and scoring, never for control. Ordinary braking to rest without a commitment does not end an episode.
5. **Scoring:**
   - `stopped` ranks like a timeout: worse than a goal, better than a collision.
   - A boundary or obstacle contact while braking or drifting after the commitment is an ordinary own-ship collision.
   - `collision:by_target` is reported separately and still counted as a contact.
   - A stop in an episode SAC alone completes is a lost SAC success.
6. **Controllers:** V16 and V17, chosen as the version most likely to differ from V16, each with the same detector and stop (`v16_estop`, `v17_estop`). OFF stays plain SAC. Fresh OFF and V16 records already exist for all 72 development cases and all 1,000 test-set-v4 scenarios in the same runtime environment.

## 2. Detector

The detector runs only at decisions where V16 (or V17) has no plan that passed its own checker: `last certificate`, `no escape` or an unknown branch label. A passing plan already predicts clearance with margins, so a collision cannot be predicted as certain there. At such a decision the detector works as follows:

- **First commands:** SAC's command, the filter's issued command, V2's rudder-by-throttle grid and the three full-astern rows.
- **Continuations:**
  - hold the first command;
  - the 43 stop and turn-then-stop tails;
  - the 68 turn-cruise-stop tails of `safety_v20_contingency.py`.

  All are truncated to the window.
- **Rollout:** each sequence is rolled out from the filter's own decision snapshot and pre-command actuator history with the same identified model the filter uses. Sequences that include astern thrust use both the weak (0.25, 0.75 s) and strong (0.5, 0 s) reverse models. Targets move at constant velocity from the filter's track estimates.
- **Contact:** a sample counts as contact if one of the following holds:
  - a true-hull corner is outside the map polygon;
  - the true hull reaches a remembered static point;
  - the true hull overlaps a target's true hull (separating-axis gap below zero).

  No hull margin, gap or allowance is added.
- **Escape time:** the escape time of a sequence is its first contact time, taking the later of the two reverse models. T* is the largest escape time over all sequences.
- **Decision:** a collision is guaranteed when T* is at most W. The vessel then stops at that decision.

## 3. What this protocol can and cannot show

- A stop avoids nothing if contact happens during braking. Such contacts stay collisions.
- The detector is a finite bank of continuations, so "guaranteed" means guaranteed within that bank and model, not a proof.
- Stopping the targets is an evaluation convention that encodes "no safe option for either mode". It does not model real target behaviour.
- Results under this protocol are not comparable with earlier outcome counts without the conversion stated above. Both counts are reported.

## 4. Choosing W (development data only, before any episode)

The detector is replayed on the saved V16-equivalent full-snapshot traces of the development cases (68 of 72, the G2/G3 replay sources). T* is computed at every unchecked decision. Each W in {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0} s is then scored on two counts:
- **False alarms:** V16-successful episodes in which the detector would fire.
- **Caught contacts:** V16 contact episodes in which it would first fire at least one decision before the episode's last decision.

**Rule:** the chosen W maximises caught contacts subject to zero false alarms, with ties going to the smaller W. If no W catches a contact without a false alarm, the result is reported and the choice is returned to the user. Along V16's own trajectory nothing changes before the first firing, so the replay gives exactly where V16 with the stop rule would first commit, up to environment reproduction.

## 5. Evaluation

1. **Development:** `v16_estop` and `v17_estop` on all 72 development cases, paired with the existing fresh OFF and V16 records.
2. **Test set v4:** both modes on all 1,000 scenarios, paired with the existing fresh OFF and V16 records, in chunks with compact gzip traces and at most two processes.

No code, option or W changes after any test-set outcome. Reports give goals, stopped, timeouts, rescued SAC failures, lost SAC successes, and each contact type including `collision:by_target`, paired by test ID, episode seed and scenario digest.
