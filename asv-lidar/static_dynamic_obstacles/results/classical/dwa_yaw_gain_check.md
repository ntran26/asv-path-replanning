# LOS-DWA yaw-rate feed-forward gain — check of 2026-10-10

**Question.** The DWA comparator's yaw-rate loop uses a feed-forward of 10.4 °/s per unit rudder
(`CLASSICAL_YAW_RATE_GAIN_DPS`, pinned in `configs/comparators_v1.json`), described in the code as
the identified steady turn at cruise:

    a_δ = clip(r_d / 10.4 + 0.08 (r_d − r), −1, 1)

The identified hull (`src/ship.py`, nominal parameters) turns more slowly than that at cruise, so
the check asks whether the constant handicaps LOS-DWA.

**Plant.** Steady yaw rate after 90 s at a fixed rudder, from a settled straight run:

| Propeller | 10 % | 20 % | 30 % | 50 % | 100 % rudder | Speed in the full-rudder turn |
|---|---|---|---|---|---|---|
| 6 rpm-units (cruise, 0.558 m/s) | 0.39 | 0.86 | 1.39 | 2.52 | 5.06 °/s | 0.46 m/s |
| 12 rpm-units (1.116 m/s) | 0.78 | 1.71 | 2.75 | 4.98 | 10.00 °/s | 0.92 m/s |

So 10.4 °/s is close to the 12 rpm-unit turn rate; at cruise the plant's gain is about 5.06 °/s
per unit rudder, roughly linear in rudder.

**Candidates at cruise.** Each of DWA's nine set-points r_d ∈ {−9, −6.75, …, +9} °/s was run
closed-loop on the model at cruise with the comparator's speed loop, the command held for each
0.5 s decision (as both the environment and DWA's own rollout do), from a settled straight run.
Heading change over the 3 s and 6 s hold times, for positive set-points (negative ones mirror):

| r_d (°/s) | 2.25 | 4.5 | 6.75 | 9.0 |
|---|---|---|---|---|
| Gain 10.4 (pinned): Δψ in 3 s / 6 s | 6.0° / 8.6° | 11.8° / 17.7° | 15.0° / 27.0° | 15.1° / 35.8° |
| Gain 5.06 (plant-matched): Δψ in 3 s / 6 s | 9.4° / 13.8° | 15.0° / 28.4° | 15.1° / 40.6° | 15.1° / 41.1° |

With the pinned gain the set-points map to a monotonic spread of turns up to about full rudder.
With the plant-matched gain the two largest set-points saturate at full rudder and give nearly the
same trajectory (40.6° and 41.1°), so the candidate set would lose a distinct option. At half speed
both gains give little turn for the small set-points (the hull turns slowly there either way).

**Prediction against execution.** `classical/common.rollout` runs the same law, holds each command
for the 0.5 s decision interval and applies the same 0.73 s rudder delay on the identified dynamics
(nominal parameters; the evaluated hull is randomized, the same mismatch every model-based
comparator has). A gain that differs from the plant therefore does not create a prediction error:
the candidate trajectories DWA scores are the ones the vessel then sails.

**Conclusion.** The constant is mislabeled, not harmful: it acts as a scale on the candidate set,
not as a model of the plant, and the pinned value spreads the candidates better at cruise than the
plant-matched value would. LOS-DWA's test-set results (success 0.770 on test set v4) stand; no
change to the pinned comparator and no re-tuning. The code comments in `src/classical/common.py`
and `src/constant_temp.py` now say what the constant is (values unchanged), and
`planning/MASTER_ARTIFACT_FACTS.md` describes it the same way.
