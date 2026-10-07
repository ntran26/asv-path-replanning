# Hand-back starts — training the policy on the states the safety layer leaves (option, 2026-10-01)

**Status: ON HOLD (decision, 2026-10-01 21:30).** The watcher that would have run it after the
SAC fix-1 fine-tune was stopped before it did anything; to resume, `bash results/handback_run.sh`.
Saved as an option (decision, 2026-10-01), planned as a PPO baseline-v3 run of 3 M steps. Code is built and tested:
`ASVLidarEnv.set_start_pool`, `tools/diagnostics/harvest_handback_starts.py`,
`train_formulation.py --start-pool`, `tests/test_handback_starts.py`, and the chain
`results/handback_run.sh`.

## 1. Why

Safety layer v2 (`SAFETY_LAYER_V2_PLAN.md`) removes obstacle contacts, but on the
development set it gains no goals: it trades obstacle contacts for wall contacts and
stalls. The traced losses share one cause. The filter hands back a boat that is slow,
off its path and near a wall or panel, and the policy has rarely been in that state.
It never trained with an override. In DV3-BO-CV-04 it held the throttle at the floor
and drifted into the west wall.

A policy that has practised recovering from those states should cope with the hand-back.
It should also be more robust with the safety layer off, for example after a near miss
or a perception dropout. The safety layer stays **out of training**. That keeps the
paper's story: the policy is trained alone, and the safety layer is a deployment add-on
evaluated with the same policy, on and off.

The existing recovery starts (`low_speed_start_frac`, `initial_heading_error_deg`,
`initial_lateral_offset_m`) are not enough. They all start at the beginning of the leg,
in open water. The hand-back states occur mid-episode, between panels and walls, often
with a target close by.

## 2. What

1. **Collect the hand-back states.** Run a policy (default: the kept 3 M SAC policy) with
   safety layer v2 on. Record every step where the filter, having overridden, passes the
   policy's action again. Hand-backs closer together than 3 s count once. Each record
   holds:
   - the scenario;
   - the own-ship state before each step of the prefix;
   - the executed actions and the brake flags.
   A hand-back state can be escaped by construction: the filter passed the policy's
   action there, so a safe recovery existed.
2. **Sources (decision: "the dev set and the coupled set"):**
   - `dev`: the v3 coupled development set, 150 scenarios × 2 episode seeds.
   - `field`: 220 fresh layouts near the deployment layouts L1–L3
     (`field_training.sample(near=True)`, training generator, seeds from 380,000).
     These cover every encounter at constant and varying speed, plus no-target.
3. **Leakage, as handled:**
   - The **Paper 2 set's own 630 episodes are not used.** They are the held-out
     deployment test, so training on their states would leak it. The near-deployment
     layouts give "coupled set" states without that. To use the exact Paper 2 episodes
     anyway, add them as a third source; the Paper 2 result for this run then stops
     being a held-out test.
   - The **dev set is used, as decided.** It is also what selects the best
     checkpoint during training (`[EVAL] dev goal`). Training sees only mid-episode
     states from those scenarios, not whole episodes, but the dev-goal selection is
     mildly optimistic for this run. The frozen suite is untouched either way.
4. **Start from them in training.** From stage 6 on (when three-panel layouts and the
   coupled family are in the mix), 25 % of episodes start from a random pool record:
   - The prefix is replayed with the own ship placed on its recorded state each step,
     so the targets, the tracker and the encounter contexts are where they were.
   - The last state is jittered: heading ±10°, speed ±20 %, kept only if the hull is
     clear. This gives a neighbourhood of states, not one exact state.
   - The policy then acts. The rest of the episode, and its reward, are ordinary.
   - Stages 1–5 are unchanged.
5. **Run:** PPO, baseline-v3, seed 0, tag `bl3_hb`. It trains 2.5 M on the v3 schedule,
   then extends to 3 M (as SAC did).
6. **Evaluate:**
   - The frozen suite and the Paper 2 set, safety off and safety v2 on.
   - The question is whether the gap between safety off and safety v2 on closes or
     reverses, i.e. whether the filter now gains goals instead of trading contacts.
   - A second question is whether safety-off performance holds.

## 3. What it needs to answer "does it help"

- **A PPO baseline-v3 control without hand-back starts** (same seed, same 3 M). No PPO
  v3 run exists yet. This control is also the planned PPO v3 seed-0 baseline, so it is
  not extra work if v3 is adopted. **Not queued; a decision is pending** (about 7–8 h of training).
- **The current SAC v3 policy is not a valid control**: it is a different learner.

## 4. Caveats

- **The states depend on the policy and the filter that produced them.** They come from
  the SAC policy with the current v2 default. A PPO policy will be handed back in
  somewhat different places. One re-collection with the new policy (DAgger-style) is the
  usual fix, if the first round helps.
- **It does not fix the filter's own choices.** The filter can still steer into a dead
  end or trade an obstacle contact for a wall contact. Braking only for traffic removed
  part of that.
- **Replay cost:** a hand-back start replays a prefix of about 50–150 steps of simulation
  without policy calls. That is roughly 10–20 % more environment time per episode.
- **Budget:** a 3 M PPO run with evaluations takes about 8 h on this laptop.

## 5. Schedule (as queued)

| step | after | approx. |
|---|---|---|
| SAC fix-1 to 3 M, then its frozen suite + Paper 2 set | running | until ~03:00, 2 Oct |
| collect hand-back states (≈600 episodes, 8 processes) | fix-1 done | ~30 min |
| PPO `bl3_hb` 2.5 M + extend to 3 M | collection | ~8 h |
| frozen suite + Paper 2 set, safety off / v2 | training | ~2 h |
