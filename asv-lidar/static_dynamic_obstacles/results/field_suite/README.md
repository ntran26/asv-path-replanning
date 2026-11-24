# Field test suite (2026-11-20)

Fifteen field scenarios for Paper 3, Section 5: three difficulty modes x the five COLREGs
encounters, named `<mode>_<n>` (`src/field_test_cases.py`, where the numbering is explained).
Built by `tools/tiers/field_suite.py`; every case is executed successfully by the deployed
policy (SAC seed 0, `runs/sac_formulation_seed0_bl3/best_model.zip`) in simulation.

| Mode | Static scene | Plan family |
| --- | --- | --- |
| easy | open basin, straight leg (5, 2) -> (5, 22), no panels | B |
| medium | layout L1: the same leg, Paper 2's three 1 m panels | C on L1 |
| hard | layout L2 or L3: slanted leg across three panels | C on L2/L3 |

| n | Encounter | Rule |
| --- | --- | --- |
| 1 | head-on | 14 |
| 2 | crossing, target to starboard (own ship gives way) | 15, 16 |
| 3 | crossing, target to port (own ship stands on) | 17 |
| 4 | overtaking (target slower) | 13 |
| 5 | being overtaken (target faster; own ship stands on) | 13 |

## How the cases were chosen

- **Draws.** Per cell (mode, layout, encounter), the scenario generator under the Paper 2 field
  rules: constant-velocity target, one heading, speed 0.20-1.10 m/s, starting at least 0.5 m
  inside the walls and stopping short of a wall only 3 s after the closest approach, own hull
  starting 0.30 m clear of every panel (the test set's start check; on L1 a being-overtaken own
  ship starts 8 m along the leg, the test set v2 fix). Seed blocks 860,000 + 1,000 per cell
  (episode seeds 880,000 + 1,000 per cell), unused by any other set.
- **Kept** only if the deployed policy reaches the goal on all 10 sensor-noise seeds on the
  identified plant (no randomisation, the environment's emergency stop off, start at cruise).
- **Robustness** (scored): 10 runs with the plant drawn from the training randomisation, the
  start heading perturbed by up to 5 deg, the start moved sideways by up to 0.20 m, and the
  target launched up to 1 s early or late.
- **Selection** per case: the most robust kept candidate, then one whose nominal run meets the
  Section 3.4.3 rule conditions (`tools/diagnostics/colregs_compliance.py`), then the smallest
  drawn DCPA. Where no kept candidate met the rule conditions, the search was extended over the
  cell's whole seed block requiring them (`--extend`).
- 1,229 candidates drew a valid field scenario; about 3,600 episodes were simulated
  (`candidates.csv`, `candidates_ext.csv`; the chosen rows in `selection.csv`).

## The cases

| Case | Layout | Target speed (m/s) | Nominal | Robust | Rule conditions (nominal run) | Closest approach (m) |
| --- | --- | --- | --- | --- | --- | --- |
| easy_1 | open | 0.57 | 10/10 | 10/10 | met | 1.17 |
| easy_2 | open | 0.52 | 10/10 | 10/10 | met | 2.40 |
| easy_3 | open | 0.57 | 10/10 | 10/10 | met | 1.90 |
| easy_4 | open | 0.25 | 10/10 | 10/10 | met | 2.89 |
| easy_5 | open | 0.93 | 10/10 | 10/10 | met | 1.68 |
| medium_1 | L1 | 0.68 | 10/10 | 10/10 | **not met**: first turn to port, starboard-to-starboard passing | 3.72 |
| medium_2 | L1 | 0.40 | 10/10 | 10/10 | **not met**: first turn to port, crosses ahead of the target | 2.36 |
| medium_3 | L1 | 0.34 | 10/10 | 8/10 | **not met**: first turn to starboard, crosses ahead of the target (re-picked 2026-11-24 for the start from rest; see below) | 2.80 |
| medium_4 | L1 | 0.29 | 10/10 | 10/10 | met | 2.76 |
| medium_5 | L1 | 0.98 | 10/10 | 10/10 | met | 4.50 |
| hard_1 | L3 | 0.57 | 10/10 | 10/10 | met | 3.08 |
| hard_2 | L3 | 0.40 | 10/10 | 10/10 | met | 2.78 |
| hard_3 | L3 | 0.49 | 10/10 | 10/10 | met | 3.10 |
| hard_4 | L2 | 0.23 | 10/10 | 10/10 | met | 3.35 |
| hard_5 | L2 | 0.94 | 10/10 | 10/10 | **not met**: changes speed while being overtaken (stand-on allows 0.15 m/s; lateral departure 0.74 m, inside the 1 m allowed) | 3.33 |

**medium_3 re-picked 2026-11-24** for the start from rest (`planning/FIELD_SESSIONS_PLAN.md`
section 7.1; `tools/tiers/field_suite_repick.py`, `results/field_suite/repick_medium_3/`). With both
ships starting from rest and the target released at the offset that keeps the designed
encounter, the earlier case (generator seed 867137, target 0.45 m/s, rule conditions met) reached
the goal on 2-4 of 10 seeds. The cell's 4 kept candidates and 400 new draws (generator seeds
980,000-980,399, episode seeds 985,000-985,399) were run under that protocol; 2 reached the goal on
10 of 10 seeds from rest, and the selection rule (most robust from rest, then the rule conditions,
then the smallest DCPA) took generator seed 980324: 10/10 and 10/10 robust from rest. The three
rule-consistent candidates reached 1-5 of 10 from rest, so no case in this cell meets both.

## Findings

- **L1 head-on and L1 crossing from starboard have no rule-consistent success in their seed
  blocks.** Of 260 and 226 valid draws, the deployed policy reached the goal while meeting the
  rule conditions in none: it turns to port first (the (5, 8) panel sits on the leg) and passes
  starboard to starboard, or crosses ahead of the give-way target. L1 crossing from port has
  rule-consistent successes, but none above 8/10 under perturbation, and none above 5/10 with
  both ships from rest (re-pick of 2026-11-24).
- **L2 being overtaken** has no rule-consistent success in 244 draws: the successful runs break
  the stand-on condition (lateral departure within 1 m and speed held within 0.15 m/s while
  the target closes); in hard_5 it is the speed. L3 being overtaken drew no
  field-valid scenario in its first 300 seeds; L2 head-on kept none (as test set v4's L2
  head-on, 0/5).
- The open basin is easy for the policy: every cell kept 6 of its first 6-12 valid draws, all
  10/10 under perturbation.

## Why three cases miss the rule conditions (2026-11-20)

`tools/diagnostics/field_suite_rule_feasibility.py` (`rule_feasibility.json`): the oracle's
manoeuvre library (`src/oracle_feasibility.py`: perfect foresight, the true vessel model, the
policy's action space, an A* route around the panels), started at 1 s steps over the 8 s after
the onboard tracker first holds the target, each rollout scored by the same rule conditions;
and the deployed policy on the same encounter with the panels removed. easy_1 is the control:
75 rule-consistent oracle solutions, the policy rule-consistent on 10 of 10 seeds.

| Case | Target tracked | Oracle: reach the goal with 0.20 m margin / of those rule-consistent | Policy, panels removed | Reading |
| --- | --- | --- | --- | --- |
| medium_1 (L1 head-on) | 9.5 s, 7.5 m (the (5, 8) panel hides it) | 337 / 0 (starboard 195, port 79, speed or stand-on 63) | 10/10 goals, 0 rule-consistent, all turn to port | circumstances: no port-to-port passing exists once the target is seen; without the panels it is tracked at 15.6 m and 1 of 549 rollouts (a 70 deg starboard turn) passes port to port |
| medium_2 (L1 crossing from starboard) | 2.0 s, 5.7 m | 81 / 0 | 9/10 goals, 0 rule-consistent | circumstances: the target is close when first seen; without the panels no oracle manoeuvre reaches the goal with margin at all, rule-consistent or not |
| hard_5 (L2 being overtaken) | 1.0 s, 3.9 m | 224 / 194; holding the route at cruise (the stand-on) 9 / 9 | 10/10 goals, 0 rule-consistent | the policy: standing on is possible and safe, and the policy does not hold course and speed, with or without the panels |

Six further L1 head-on draws on which the deployed policy turned to port (generator seeds
865017, 865155, 865299, 865462, 865625, 865763): no rule-consistent oracle solution in any; in
two no library manoeuvre reaches the goal with margin and in two more only 2-5 port turns do,
while the policy reaches the goal in all six.

Across the test set (constant-velocity targets, episodes reaching the goal, track-based rule
conditions; `results/test_set/v4/<run>/compliance.csv`), being overtaken is rule-consistent in
15 % of the deployed policy's open-basin episodes, 17 % for its 3 M-step final checkpoint,
0-21 % for the other SAC and TQC seeds, and 100 % for the rule-based comparator
(`encounter_vo_v2`). The stand-on is therefore a learned behaviour common to every learner and
not improved by the longer training, so more training of the same formulation is not expected to
change it; the deployed seed is also weaker at head-on in the open basin (52 %, the other seeds
and TQC 62-82 %).

## Use

```bash
python src/field_test_cases.py list
python src/field_test_cases.py sheet medium_2
python src/field_test_cases.py run all --seeds 10
python src/field_test_cases.py run hard_3 --render
python src/test_asv.py --algo sac --scenario hard_3            # the same, through setups.py
python ../field_deployment/udp_live_p3.py --scenario easy_1 --server-ip 10.201.205.110 --run-id T1-easy_1-r1
```

The field sheet gives the own ship's point and heading at t = 0 (it passes there at cruise; it
runs up along the leg from the leg start first, and for being overtaken starts further along),
and the target's start, heading and speed, launched so it is at its start at t = 0.
