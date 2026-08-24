"""Hard-arrangement evaluation layouts: same obstacle counts, harder geometry.

    python src/build_hard_layouts.py            # build + report acceptance rates
    python src/build_hard_layouts.py --probe    # acceptance rates only, no write

The obstacle *count* is deliberately unchanged -- these sets use counts 2, 3 and
4, exactly the counts the frozen 500 already contains, so each hard set pairs
against a same-count subset of the standard evaluation.  What changes is the
arrangement: obstacles sit on the path rather than beside it, gates are narrowed
towards the vessel's own footprint, the bypass corridor in the `target_side`
family is tightened, obstacles are spread over more of the path so avoidance
manoeuvres chain, and the pure-distractor `offpath` family is removed.

How the difficulty is applied
-----------------------------
The sampler reads its geometry from `config` at call time, so this script
temporarily rebinds those module constants **during generation only**.  Layouts
are serialised as explicit obstacle polygons, and the evaluation harness replays
that geometry directly, so nothing at evaluation time depends on these values --
`config` is restored before the script exits.

`env.py` and the reward are untouched, and the standard evaluation, tuning and
OOD sets are unaffected.

A route-ratio floor is applied on top, so every accepted layout demonstrably
forces a detour rather than merely looking cluttered.  The A* feasibility filter
inherited from `generate_suite` still guarantees every layout is solvable.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import config as cfg
from env import ASVLidarEnv

OUT_DIR = "eval_layouts"

# Seed base and case-id block disjoint from every other set:
# evaluation 675_974..1_076_073, tuning 5_000_000+, OOD 7_000_000+ (ids
# 2_000_000+), hard-count ladder 8_000_000+ (ids 3_000_000+).
HARD_ARR_BASE_SEED = 9_000_000
HARD_ARR_CASE_ID_OFFSET = 4_000_000

# Obstacle counts are NOT raised -- these are counts the frozen 500 already has.
HARD_ARR_COUNTS = [2, 3, 4]
HARD_ARR_N = 100

# Per-count detour floors: the MAXIMUM route ratio present in the standard
# evaluation set at that obstacle count.  Every accepted hard layout is
# therefore strictly harder, by this measure, than any same-count layout the
# methods were evaluated on before.
#
# A single global floor does not work.  At 1.09 the 2-obstacle case accepted
# 0/10 in 4000 attempts: two 1 m squares in a 10 m basin cannot force a >9 %
# detour, there is always a way round.
#
# Note what this measure does and does not capture.  `route_ratio` comes from an
# inflated-grid A* with a POINT robot, so it measures how far around the vessel
# must go -- not how precisely it must steer.  A narrow gate has a small detour
# but demands exact threading.  The gate/corridor narrowing in HARD_OVERRIDES is
# what raises difficulty along that second axis; the floor below handles the
# first.
MIN_ROUTE_RATIO_BY_COUNT: Dict[int, float] = {2: 1.065, 3: 1.072, 4: 1.083}

# Geometry overrides applied to `config` for generation only.
#
# Reference scales: the inflated hull is 0.80 m wide (VESSEL_WIDTH 0.5 +
# 2 x HULL_MARGIN 0.15), obstacles are 1.0 m squares, the basin is 10 m wide.
# A 1.00 m gate therefore leaves ~0.10 m either side of the hull.
HARD_OVERRIDES: Dict[str, Any] = {
    # Drop the pure-distractor family; weight towards genuine blocking.
    "TRAIN_SCENARIO_MODES": ["normal", "target_side", "field_repair", "gate", "offpath"],
    "TRAIN_SCENARIO_PROBS": [0.30, 0.35, 0.05, 0.30, 0.00],

    # Put obstacles on the path rather than beside it.
    "OBSTACLE_CENTER_PROB": 0.75,
    "OBSTACLE_LATERAL_OFFSET_MIN": 0.0,
    "OBSTACLE_LATERAL_OFFSET_MAX": 0.45,

    # Spread them over more of the path so manoeuvres chain.
    "OBSTACLE_PATH_START_FRAC": 0.20,
    "OBSTACLE_PATH_END_FRAC": 0.80,

    # Narrow the gates towards the hull width.
    "GATE_GAP_RANGE": (1.00, 1.50),
    "GATE_PATH_FRAC_RANGE": (0.30, 0.75),
    "GATE_LATERAL_EXTRA": (0.0, 0.15),

    # Tighten the passable corridor in the side-choice family.
    "TARGET_SIDE_CORRIDOR_OFFSET_RANGE": (0.50, 0.80),
    "TARGET_SIDE_BLOCKED_OFFSET_RANGE": (1.20, 2.10),
    "TARGET_SIDE_PATH_FRAC_RANGE": (0.32, 0.72),
}


class _Overrides:
    """Apply the hard geometry to `config` for the duration of generation."""

    def __init__(self, overrides: Dict[str, Any]) -> None:
        self.overrides = overrides
        self.saved: Dict[str, Any] = {}

    def __enter__(self):
        for k, v in self.overrides.items():
            self.saved[k] = getattr(cfg, k)
            setattr(cfg, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(cfg, k, v)
        return False


def build_set(obs_count: int, n: int, block: int, min_ratio: float,
              probe: bool = False, verbose: bool = True) -> Tuple[List[Dict[str, Any]], int]:
    import generate_suite as gs

    env = ASVLidarEnv(map_width=gs.MAP_WIDTH, map_height=gs.MAP_HEIGHT,
                      max_obs=obs_count, path_mode=gs.PATH_MODE)
    env.reset(seed=HARD_ARR_BASE_SEED + 1000 * block)

    scenarios: List[Dict[str, Any]] = []
    attempts = 0
    # The inherited cap (n x 400) is too tight here: at 2 obstacles the
    # acceptance rate is ~0.4 %, so 100 layouts needs ~25 000 attempts and would
    # sit uncomfortably close to a 40 000 ceiling.  Raised so a shortfall is a
    # real failure rather than an artefact of the budget.
    limit = n * 1000
    while len(scenarios) < n and attempts < limit:
        attempts += 1
        seed = HARD_ARR_BASE_SEED + 1_000_000 * block + obs_count * 10_000 + attempts
        scenario = gs.build_scenario(env, obs_count, len(scenarios), seed)
        if scenario is None:
            continue
        # Extra difficulty gate on top of the inherited feasibility filter.
        if scenario["route_ratio_astar"] < min_ratio:
            continue
        scenario["case_id"] = (HARD_ARR_CASE_ID_OFFSET + 100_000 * block
                               + obs_count * 1000 + len(scenarios))
        scenario["hard_arrangement"] = True
        scenarios.append(scenario)

    env.close()
    if verbose:
        rate = len(scenarios) / max(attempts, 1)
        rr = [s["route_ratio_astar"] for s in scenarios]
        detail = (f"route_ratio p50={np.percentile(rr, 50):.3f} "
                  f"max={max(rr):.3f}" if rr else "none accepted")
        print(f"    obs={obs_count}: {len(scenarios)}/{n} in {attempts} attempts "
              f"(accept {rate:.1%})  {detail}")
    return scenarios, attempts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true",
                    help="report acceptance rates on a small sample, write nothing")
    ap.add_argument("--min-ratio", type=float, default=None,
                    help="override the per-count detour floor with one value")
    args = ap.parse_args()

    n = 10 if args.probe else HARD_ARR_N
    floors = ({c: args.min_ratio for c in HARD_ARR_COUNTS} if args.min_ratio
              else dict(MIN_ROUTE_RATIO_BY_COUNT))
    print(f"Hard-arrangement layouts (counts {HARD_ARR_COUNTS})")
    print(f"Per-count detour floors (= max route ratio in the standard set): {floors}")
    print("Geometry overrides applied to config for generation only:")
    for k, v in HARD_OVERRIDES.items():
        print(f"    {k} = {v}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    with _Overrides(HARD_OVERRIDES):
        for block, count in enumerate(HARD_ARR_COUNTS):
            name = f"hard_arr_obs{count}"
            print(f"  {name}")
            scenarios, attempts = build_set(count, n, block, floors[count],
                                            probe=args.probe)
            if args.probe:
                continue
            if len(scenarios) < n:
                raise SystemExit(
                    f"{name}: only {len(scenarios)}/{n} layouts met the difficulty "
                    f"floor after {attempts} attempts -- lower --min-ratio")
            payload = {
                "metadata": {
                    "name": name,
                    "role": "hard_arrangement_evaluation",
                    "n_scenarios": len(scenarios),
                    "obstacle_counts": [count],
                    "path_mode": "straight",
                    "base_seed": HARD_ARR_BASE_SEED,
                    "case_id_offset": HARD_ARR_CASE_ID_OFFSET + 100_000 * block,
                    "min_route_ratio": floors[count],
                    "min_route_ratio_rationale": (
                        "maximum route_ratio_astar present at this obstacle count in "
                        "eval_layouts_v1.json, so every layout here is strictly harder "
                        "by that measure than any same-count layout in the standard set"),
                    "generator": "generate_suite.build_scenario with hard geometry overrides",
                    "geometry_overrides": {k: list(v) if isinstance(v, tuple) else v
                                           for k, v in HARD_OVERRIDES.items()},
                    "note": (
                        "Obstacle COUNT is unchanged from the standard evaluation set "
                        "-- only the arrangement is harder. Pairs against the same-count "
                        "subset of eval_layouts_v1.json. Equally out-of-distribution for "
                        "every method: no controller was trained or tuned on these "
                        "layouts, and none was re-tuned for them."
                    ),
                },
                "scenarios": scenarios,
            }
            path = os.path.join(OUT_DIR, f"{name}.json")
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"    wrote {path}")

    # config is restored on exit; confirm rather than assume.
    assert cfg.GATE_GAP_RANGE == (1.35, 2.25), "config was not restored"
    assert cfg.OBSTACLE_CENTER_PROB == 0.30, "config was not restored"
    print("\nconfig restored to standard values")


if __name__ == "__main__":
    main()
