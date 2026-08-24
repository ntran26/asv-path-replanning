"""Frozen layout sets for baseline comparison, and deterministic env construction.

Two sets, both serialised to `eval_layouts/`:

`eval_layouts_v1.json`   the 500-episode frozen evaluation set.  This is a
                         verbatim copy of the shipped holdout in
                         `eval_suite/asv_eval_suite.json` -- it is NOT
                         regenerated.  The suite is a fixed holdout that
                         published results already reference, so rebuilding it
                         would make every existing number incomparable.  The
                         copy exists only so that every method in this study
                         reads one canonical file.

`tune_layouts_v1.json`   a 100-episode tuning set, 20 per obstacle count, built
                         with the *same* generator (`generate_suite`) but from a
                         disjoint seed base.  Baseline tuning uses this set and
                         only this set.

Usage:

    python src/eval_layouts.py --build          # write both files
    python src/eval_layouts.py --check          # verify disjointness + replay

    from eval_layouts import load_layouts, reset_to_layout
    for rec in load_layouts("eval_layouts/eval_layouts_v1.json"):
        obs, info = reset_to_layout(env, rec)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from env import ASVLidarEnv

# Where the shipped holdout lives.  Read-only as far as this module is concerned.
FROZEN_SOURCE = "eval_suite/asv_eval_suite.json"

OUT_DIR = "eval_layouts"
EVAL_LAYOUTS = os.path.join(OUT_DIR, "eval_layouts_v1.json")
TUNE_LAYOUTS = os.path.join(OUT_DIR, "tune_layouts_v1.json")

# generate_suite.py uses BASE_SEED = 675973 and derives per-case seeds as
# BASE_SEED + obs_count * 100_000 + attempts, which lands in [675_974, ~1_100_000].
# Starting the tuning set at 5_000_000 keeps the two sets disjoint by a wide
# margin no matter how many rejection attempts either generator needs.
TUNE_BASE_SEED = 5_000_000
TUNE_PER_OBS_COUNT = 20
TUNE_OBSTACLE_COUNTS = [0, 1, 2, 3, 4]

# Tuning case ids are offset so they can never collide with evaluation case ids
# (which run 0..499).  A stray join on episode_id then fails loudly.
TUNE_CASE_ID_OFFSET = 900_000

# --- Out-of-distribution generalisation sets (see OOD_PROTOCOL.md) -----------
# Conditions neither the learned policies nor the tuned classical controller has
# seen: obstacle count 5 is outside TRAIN_OBS_COUNTS, and curved reference paths
# are outside PATH_MODE="straight".  Seed base is disjoint from both the
# evaluation set (675_974..1_076_073) and the tuning set (5_000_000+).
OOD_BASE_SEED = 7_000_000
# 2_000_000 rather than 800_000: with a 100_000-per-block stride the latter put
# block 1 at exactly 900_000, which is TUNE_CASE_ID_OFFSET, and 20 tuning ids
# collided.  Case ids are join keys for the paired tests, so a collision would
# silently pair unrelated episodes.
OOD_CASE_ID_OFFSET = 2_000_000
OOD_N_PER_COUNT = 100

# --- Hard evaluation ladder -------------------------------------------------
# Obstacle densities above the training range (config.TRAIN_OBS_COUNTS = 0..4)
# and above the frozen evaluation set (0..4).  Straight reference paths
# throughout, so obstacle density is the ONLY variable that changes -- the point
# is a clean degradation curve, not a mixed shift.
#
# Equally out-of-distribution for every method: the learned policies never
# trained above 4 obstacles, and the LOS+APF parameters were selected on
# 0-4 obstacle layouts.  No method is re-tuned for these sets.
#
# Seed base and case-id block are disjoint from the evaluation set
# (675_974..1_076_073), the tuning set (5_000_000+) and the OOD sets
# (7_000_000+ / case ids 2_000_000+).
HARD_BASE_SEED = 8_000_000
HARD_CASE_ID_OFFSET = 3_000_000
HARD_N_PER_COUNT = 100

HARD_SETS: Dict[str, Any] = {
    "hard_obs5": {"counts": [5], "path_mode": "straight", "n": HARD_N_PER_COUNT, "block": 0},
    "hard_obs6": {"counts": [6], "path_mode": "straight", "n": HARD_N_PER_COUNT, "block": 1},
    "hard_obs7": {"counts": [7], "path_mode": "straight", "n": HARD_N_PER_COUNT, "block": 2},
}

# name -> (obstacle counts, path mode, n per count, case-id block)
OOD_SETS: Dict[str, Any] = {
    "ood_obs5":       {"counts": [5],             "path_mode": "straight", "n": 100, "block": 0},
    "ood_curve":      {"counts": [0, 1, 2, 3, 4], "path_mode": "curve",    "n": 20,  "block": 1},
    "ood_curve_obs5": {"counts": [5],             "path_mode": "curve",    "n": 100, "block": 2},
}


# ---------------------------------------------------------------------------
# Loading and deterministic replay
# ---------------------------------------------------------------------------
def load_layouts(path: str) -> List[Dict[str, Any]]:
    """Read a layout set.  Accepts both this module's format and the suite's."""
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload["scenarios"]
    return payload


def reset_to_layout(env: ASVLidarEnv, record: Dict[str, Any]):
    """Reset `env` onto a layout record, deterministically.

    Mirrors exactly what `evaluate_suite.py` does, so results from the new
    harness are directly comparable with the existing suite evaluator:
    the seed is passed *and* the layout is pinned through `options["scenario"]`.

    The scenario path bypasses every random draw in `reset` -- `_load_scenario`
    reads start/goal/obstacles/path straight from the record, and
    `_sample_obs_border_mode` returns without drawing while
    `OBS_BORDER_MODE != "mixed"`.  The seed is therefore belt-and-braces, but it
    is what the existing evaluator passes and dropping it would be a silent
    divergence.
    """
    return env.reset(seed=int(record.get("seed", 0)),
                     options={"scenario": record})


def make_env_from_layout(record: Dict[str, Any], **env_kwargs) -> ASVLidarEnv:
    """Construct a fresh env sized to a layout record and reset onto it."""
    env = ASVLidarEnv(
        map_width=float(record.get("map_width", 10.0)),
        map_height=float(record.get("map_height", 25.0)),
        max_obs=5,
        path_mode=str(record.get("path_mode", "straight")),
        **env_kwargs,
    )
    reset_to_layout(env, record)
    return env


def episode_id(record: Dict[str, Any]) -> int:
    """Stable identifier used to pair episodes across methods."""
    return int(record["case_id"])


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def copy_frozen_set() -> Dict[str, Any]:
    """Copy the shipped holdout into the canonical evaluation-set path."""
    with open(FROZEN_SOURCE, "r") as f:
        payload = json.load(f)

    scenarios = payload["scenarios"]
    out = {
        "metadata": {
            "name": "eval_layouts_v1",
            "role": "evaluation",
            "n_scenarios": len(scenarios),
            "source": FROZEN_SOURCE,
            "note": (
                "Verbatim copy of the shipped fixed holdout. Not regenerated: "
                "published results reference these exact layouts. The source "
                "metadata below is carried through unchanged, including its "
                "inaccurate '600-case ... 0..5' description -- the file "
                "actually holds 500 cases over obstacle counts 0..4."
            ),
            "source_metadata": payload.get("metadata", {}),
        },
        "scenarios": scenarios,
    }
    return out


def build_tuning_set(verbose: bool = True) -> Dict[str, Any]:
    """Generate the tuning layouts with the suite generator and disjoint seeds."""
    # Imported here so that `load_layouts` users do not pay for it.
    import generate_suite as gs

    env = ASVLidarEnv(map_width=gs.MAP_WIDTH, map_height=gs.MAP_HEIGHT,
                      max_obs=max(TUNE_OBSTACLE_COUNTS), path_mode=gs.PATH_MODE)
    env.reset(seed=TUNE_BASE_SEED)

    scenarios: List[Dict[str, Any]] = []
    for obs_count in TUNE_OBSTACLE_COUNTS:
        made = 0
        attempts = 0
        while made < TUNE_PER_OBS_COUNT:
            attempts += 1
            if attempts > TUNE_PER_OBS_COUNT * gs.MAX_ATTEMPTS_PER_CASE:
                raise RuntimeError(
                    f"Could not generate enough feasible tuning scenarios for "
                    f"{obs_count} obstacles ({made}/{TUNE_PER_OBS_COUNT}).")

            seed = TUNE_BASE_SEED + obs_count * 100_000 + attempts
            scenario = gs.build_scenario(env, obs_count, made, seed)
            if scenario is None:
                continue

            # Re-key so tuning ids can never be confused with evaluation ids.
            scenario["case_id"] = TUNE_CASE_ID_OFFSET + obs_count * 100 + made
            scenarios.append(scenario)
            made += 1

        if verbose:
            print(f"  obstacle count {obs_count}: {made} scenarios "
                  f"({attempts} attempts)")

    env.close()
    return {
        "metadata": {
            "name": "tune_layouts_v1",
            "role": "tuning",
            "n_scenarios": len(scenarios),
            "n_per_obstacle_count": TUNE_PER_OBS_COUNT,
            "obstacle_counts": TUNE_OBSTACLE_COUNTS,
            "base_seed": TUNE_BASE_SEED,
            "case_id_offset": TUNE_CASE_ID_OFFSET,
            "generator": "generate_suite.build_scenario",
            "map_width": gs.MAP_WIDTH,
            "map_height": gs.MAP_HEIGHT,
            "path_mode": gs.PATH_MODE,
            "note": (
                "Disjoint-seed tuning set. The 500-episode evaluation set is "
                "never used for tuning."
            ),
        },
        "scenarios": scenarios,
    }


def build_hard_sets(verbose: bool = True) -> None:
    """Generate the hard obstacle-density ladder (5, 6, 7 obstacles)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in HARD_SETS:
        if verbose:
            print(f"Generating {name} ...")
        payload = build_ood_set(name, verbose=verbose, spec=HARD_SETS[name],
                                base_seed=HARD_BASE_SEED,
                                case_offset=HARD_CASE_ID_OFFSET,
                                role="hard_evaluation")
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        if verbose:
            print(f"  wrote {path} ({payload['metadata']['n_scenarios']} scenarios)")


def build_ood_set(name: str, verbose: bool = True, spec=None,
                  base_seed: Optional[int] = None,
                  case_offset: Optional[int] = None,
                  role: str = "out_of_distribution_evaluation") -> Dict[str, Any]:
    """Generate one out-of-distribution layout set.

    Uses the same `generate_suite.build_scenario` and the same A* feasibility
    filter as the evaluation suite, so the only thing that differs is the
    condition being varied.  See OOD_PROTOCOL.md, which was written before any
    of these layouts existed.

    Note on the feasibility filter for curved sets: `route_ratio` compares the
    A* route against the *straight-line* start-goal distance, so on a curved
    reference path the ratio runs higher than it would for a straight one.  The
    filter still guarantees the goal is reachable, which is its purpose here;
    the bounds are simply less tight.  Recorded rather than silently accepted.
    """
    import generate_suite as gs

    spec = OOD_SETS[name] if spec is None else spec
    base_seed = OOD_BASE_SEED if base_seed is None else base_seed
    case_offset = OOD_CASE_ID_OFFSET if case_offset is None else case_offset
    counts = spec["counts"]
    path_mode = spec["path_mode"]
    n_each = spec["n"]
    block = spec["block"]

    env = ASVLidarEnv(map_width=gs.MAP_WIDTH, map_height=gs.MAP_HEIGHT,
                      max_obs=max(max(counts), 1), path_mode=path_mode)
    env.reset(seed=base_seed + 1000 * block)

    scenarios: List[Dict[str, Any]] = []
    for obs_count in counts:
        made = 0
        attempts = 0
        while made < n_each:
            attempts += 1
            if attempts > n_each * gs.MAX_ATTEMPTS_PER_CASE:
                raise RuntimeError(
                    f"{name}: only {made}/{n_each} feasible scenarios for "
                    f"{obs_count} obstacles after {attempts} attempts")

            seed = (base_seed + 1_000_000 * block
                    + obs_count * 10_000 + attempts)
            scenario = gs.build_scenario(env, obs_count, made, seed)
            if scenario is None:
                continue

            scenario["case_id"] = (case_offset + 100_000 * block
                                   + obs_count * 1000 + made)
            scenario["path_mode"] = path_mode          # build_scenario hardcodes "straight"
            scenario["ood_set"] = name
            scenarios.append(scenario)
            made += 1

        if verbose:
            print(f"    {obs_count} obstacles: {made} scenarios ({attempts} attempts)")

    env.close()
    return {
        "metadata": {
            "name": name,
            "role": role,
            "n_scenarios": len(scenarios),
            "obstacle_counts": counts,
            "path_mode": path_mode,
            "base_seed": base_seed,
            "case_id_offset": case_offset + 100_000 * block,
            "generator": "generate_suite.build_scenario",
            "protocol": "OOD_PROTOCOL.md",
            "note": (
                "Out-of-distribution relative to BOTH families: obstacle count 5 "
                "is outside config.TRAIN_OBS_COUNTS = [0,1,2,3,4], and curved "
                "reference paths are outside PATH_MODE='straight'. The LOS+APF "
                "parameters were tuned on straight-path 0-4 obstacle layouts, so "
                "the shift is equally novel for the classical baseline."
            ),
        },
        "scenarios": scenarios,
    }


def build_ood(verbose: bool = True) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in OOD_SETS:
        if verbose:
            print(f"Generating {name} ...")
        payload = build_ood_set(name, verbose=verbose)
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        if verbose:
            print(f"  wrote {path} ({payload['metadata']['n_scenarios']} scenarios)")


def build(verbose: bool = True) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if verbose:
        print(f"Copying frozen evaluation set from {FROZEN_SOURCE} ...")
    eval_set = copy_frozen_set()
    with open(EVAL_LAYOUTS, "w") as f:
        json.dump(eval_set, f, indent=2)
    if verbose:
        print(f"  wrote {EVAL_LAYOUTS} ({eval_set['metadata']['n_scenarios']} scenarios)")

    if verbose:
        print("Generating tuning set ...")
    tune_set = build_tuning_set(verbose=verbose)
    with open(TUNE_LAYOUTS, "w") as f:
        json.dump(tune_set, f, indent=2)
    if verbose:
        print(f"  wrote {TUNE_LAYOUTS} ({tune_set['metadata']['n_scenarios']} scenarios)")


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------
def check(verbose: bool = True) -> bool:
    """Verify the two sets are disjoint and that replay is deterministic."""
    ok = True
    ev = load_layouts(EVAL_LAYOUTS)
    tu = load_layouts(TUNE_LAYOUTS)

    print(f"evaluation set : {len(ev)} scenarios")
    print(f"tuning set     : {len(tu)} scenarios")

    ev_seeds = {int(r["seed"]) for r in ev}
    tu_seeds = {int(r["seed"]) for r in tu}
    seed_overlap = ev_seeds & tu_seeds
    print(f"seed overlap   : {len(seed_overlap)}  (must be 0)")
    ok &= not seed_overlap

    ev_ids = {int(r["case_id"]) for r in ev}
    tu_ids = {int(r["case_id"]) for r in tu}
    id_overlap = ev_ids & tu_ids
    print(f"case_id overlap: {len(id_overlap)}  (must be 0)")
    ok &= not id_overlap
    ok &= len(ev_ids) == len(ev) and len(tu_ids) == len(tu)

    # A layout is only usable if identical geometry comes back on every reset.
    env = ASVLidarEnv(map_width=10.0, map_height=25.0, max_obs=5, path_mode="straight")
    mismatches = 0
    for rec in [ev[0], ev[250], ev[499], tu[0], tu[50], tu[99]]:
        snaps = []
        for _ in range(2):
            reset_to_layout(env, rec)
            snaps.append((
                round(env.start_x, 12), round(env.start_y, 12),
                round(env.goal_x, 12), round(env.goal_y, 12),
                tuple(tuple(tuple(np.round(p, 12)) for p in o) for o in env.obstacles),
                tuple(map(tuple, np.round(env.path.points, 12).tolist())),
            ))
        if snaps[0] != snaps[1]:
            mismatches += 1
            print(f"  !! case {rec['case_id']} replayed differently")
    print(f"replay mismatches: {mismatches}  (must be 0)")
    ok &= mismatches == 0
    env.close()

    print("\nOK" if ok else "\nFAILED")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="write both layout files")
    ap.add_argument("--build-hard", action="store_true",
                    help="write the hard obstacle-density ladder (5, 6, 7 obstacles)")
    ap.add_argument("--build-ood", action="store_true",
                    help="write the out-of-distribution sets (see OOD_PROTOCOL.md)")
    ap.add_argument("--check", action="store_true", help="verify disjointness and replay")
    args = ap.parse_args()

    if not (args.build or args.check or args.build_ood or args.build_hard):
        ap.error("pass --build, --build-hard, --build-ood and/or --check")
    if args.build:
        build()
    if args.build_hard:
        build_hard_sets()
    if args.build_ood:
        build_ood()
    if args.check:
        raise SystemExit(0 if check() else 1)


if __name__ == "__main__":
    main()
