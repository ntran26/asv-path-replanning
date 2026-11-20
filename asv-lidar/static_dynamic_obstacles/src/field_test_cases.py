"""
Field test cases for Paper 3 (Section 5): three difficulty modes x the five COLREGs encounters.

Every case is a constant-velocity target encounter that a second model vessel can sail in the
10 x 25 m basin (the field rules of `paper2_set.py`: the target starts at least 0.5 m inside the
walls, holds one heading, its speed lies in 0.20-1.10 m/s, and it stops short of the wall only
after the encounter is over), and every case is executed successfully by the deployed policy in
simulation (`tools/tiers/field_suite.py`, which selected them; the record is in `CASES` and in
`results/field_suite/`).

Naming: <mode>_<number>, for example easy_1, medium_5, hard_3.

Modes (the static scene; the target encounter is the same kind of draw in all three):
    easy    open basin: the straight leg (5, 2) -> (5, 22), no panels
            (FIELD_TRIALS_PLAN.md family B, open-basin encounters)
    medium  layout L1 of Paper 2: the straight leg (5, 2) -> (5, 22) with its three 1 m panels
            (family C on L1, coupled encounters)
    hard    layout L2 or L3 of Paper 2: a slanted leg across three 1 m panels
            (family C on L2/L3; the layout of each case is in CASES)

Numbers (the COLREGs encounter; the same number is the same encounter in every mode):
    1  head-on                       Rule 14: both alter to starboard
    2  crossing, target to starboard Rules 15 and 16: the own ship gives way
    3  crossing, target to port      Rule 17: the own ship stands on
    4  overtaking                    Rule 13: the own ship overtakes a slower target
    5  being overtaken               Rule 13: a faster target overtakes; the own ship stands on

So easy_1 is the open-basin head-on, medium_5 being overtaken on L1, hard_3 the crossing from
port on L2 or L3.

Use:
    from field_test_cases import TestCase
    case = TestCase()
    built, seed = case.scenario("medium_2")          # for ASVLidarEnv.reset(seed=seed, options={"generated": built})
    sx, sy, gx, gy = case.position("medium_2")       # the leg
    panels = case.obstacles("medium_2")              # panel polygons
    sheet = case.field_sheet("medium_2")             # where and how to launch the target

    python src/field_test_cases.py list
    python src/field_test_cases.py sheet hard_4
    python src/field_test_cases.py run easy_1 [--seeds 10] [--randomised] [--render]
    python src/test_asv.py --algo sac --scenario easy_1 [--save-mp4]      # through setups.py
    python ../field_deployment/udp_live_p3.py --scenario easy_1 --server-ip ... --run-id ...

Coordinates are the basin's map frame in metres (x across, 0-10; y along, 0-25); headings are
compass degrees (0 = +y, 90 = +x).
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import constant_temp as ct
import paper2_set as p2
import scenario as scn
import targets as tgt

MODES = ("easy", "medium", "hard")
ENCOUNTERS = {1: "HO", 2: "CRS", 3: "CRP", 4: "OT", 5: "BO"}            # number -> paper2_set code
ENCOUNTER_NAMES = {"HO": "head-on", "CRS": "crossing, target to starboard (give way)",
                   "CRP": "crossing, target to port (stand on)", "OT": "overtaking",
                   "BO": "being overtaken (stand on)"}
MODE_LAYOUTS = {"easy": ("OPEN",), "medium": ("L1",), "hard": ("L2", "L3")}
OPEN_LEG = ((5.0, 2.0), (5.0, 22.0))
# Test set v2's fix (tools/tiers/test_set.py CELL_FIXES): on L1 a being-overtaken own ship
# would otherwise start on the (5, 8) panel.
CELL_FIXES = {("L1", "BO"): {"own_start_s": 8.0}}
GENERATOR_STAGE = 5
GENERATOR_NAMESPACE = "frozen_eval"


def case_name(mode: str, code: str) -> str:
    number = next(n for n, c in ENCOUNTERS.items() if c == code)
    return f"{mode}_{number}"


def leg(layout: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if layout == "OPEN":
        return OPEN_LEG
    lay = p2.LAYOUTS[layout]
    return tuple(lay["start"]), tuple(lay["goal"])


def panels(layout: str) -> List[List[Tuple[float, float]]]:
    return [] if layout == "OPEN" else p2.panel_polygons(layout)


def flags_for(layout: str, code: str) -> Dict:
    """The generator flags of a case: the leg, the panels, the field stop box, the crossing side."""
    side = p2.ENCOUNTERS[code][1]
    start, goal = leg(layout)
    flags = {"basin_leg": [list(start), list(goal)], "set": "field_suite", "layout": layout,
             "target_stop_box": p2.stop_box(), **CELL_FIXES.get((layout, code), {})}
    if layout != "OPEN":
        flags["fixed_obstacles"] = [[list(p) for p in poly] for poly in panels(layout)]
    if side:
        flags["side"] = side
    return flags


def build(name: str, layout: str, code: str, generator_seed: int) -> Optional[scn.Scenario]:
    """The scenario generator's draw for a case (None if the draw caps out).

    The same call builds the candidates (`tools/tiers/field_suite.py`) and the cases here, so a
    case is rebuilt exactly; `TestCase.scenario` checks the result against the recorded digest.
    """
    generator = scn.ScenarioGenerator(stage=GENERATOR_STAGE, seed_namespace=GENERATOR_NAMESPACE)
    built = generator.sample(int(generator_seed), case_id=name, encounter_class=p2.ENCOUNTERS[code][0],
                             behaviour=tgt.T_CV, geometry_mode="basin", flags=flags_for(layout, code))
    if built is not None and layout == "OPEN":
        built.n_obstacles, built.obstacles = 0, ()       # the open basin: no drawn clutter either
    return built


# ---------------------------------------------------------------------------
# The cases (written by tools/tiers/field_suite.py --write; do not edit by hand)
# ---------------------------------------------------------------------------
# Per case: the draw (layout, encounter, generator seed), the episode seed and the scenario's
# digest; the field set-up (own ship at t = 0 at cruise, target start, heading and speed; times
# from the moment the own ship passes `own_at_t0`); and the simulated verification of the
# deployed policy: `nominal` on the identified plant over sensor-noise seeds, `robust` with the
# plant drawn from the training randomisation and the start and the target's launch perturbed.
# BEGIN CASES
CASES: Dict[str, Dict] = {
    "easy_1": {
        "mode": "easy",
        "encounter": "HO",
        "layout": "OPEN",
        "generator_seed": 860000,
        "episode_seed": 880000,
        "digest": "db6073c07b0e833001dd29c661bed8bb67621bde96db114b7a81baae8f255619",
        "own_at_t0": [5.0, 2.0],
        "own_heading_deg": 0.0,
        "target_start": [4.69, 16.01],
        "target_heading_deg": 181.1,
        "target_speed_mps": 0.57,
        "tcpa_s": 12.4,
        "dcpa_m": 0.45,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 24.5,
        "nominal_min_target_range_m": 1.17},
    "easy_2": {
        "mode": "easy",
        "encounter": "CRS",
        "layout": "OPEN",
        "generator_seed": 861010,
        "episode_seed": 881010,
        "digest": "d38071479b02e43e523c1f428f5895d4c3fa5c85eead10052d33a7ad04408549",
        "own_at_t0": [5.0, 2.0],
        "own_heading_deg": 0.0,
        "target_start": [8.63, 6.29],
        "target_heading_deg": 292.1,
        "target_speed_mps": 0.52,
        "tcpa_s": 9.0,
        "dcpa_m": 1.28,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 28.0,
        "nominal_min_target_range_m": 2.4},
    "easy_3": {
        "mode": "easy",
        "encounter": "CRP",
        "layout": "OPEN",
        "generator_seed": 862005,
        "episode_seed": 882005,
        "digest": "2419b131470ae7f1e4d59ae1626eae640e95789c0200f601964b5e8c70e58011",
        "own_at_t0": [5.0, 2.0],
        "own_heading_deg": 0.0,
        "target_start": [1.7, 10.03],
        "target_heading_deg": 119.4,
        "target_speed_mps": 0.57,
        "tcpa_s": 8.8,
        "dcpa_m": 1.28,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 29.0,
        "nominal_min_target_range_m": 1.9},
    "easy_4": {
        "mode": "easy",
        "encounter": "OT",
        "layout": "OPEN",
        "generator_seed": 863003,
        "episode_seed": 883003,
        "digest": "c6fad8b0bd98273d5e2cd0cd006e5bd8a2128a6ef12c7f40061af36b9737c96d",
        "own_at_t0": [5.0, 2.0],
        "own_heading_deg": 0.0,
        "target_start": [7.16, 8.07],
        "target_heading_deg": 354.6,
        "target_speed_mps": 0.25,
        "tcpa_s": 20.3,
        "dcpa_m": 1.67,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 24.5,
        "nominal_min_target_range_m": 2.89},
    "easy_5": {
        "mode": "easy",
        "encounter": "BO",
        "layout": "OPEN",
        "generator_seed": 864023,
        "episode_seed": 884023,
        "digest": "78971be39e61755c34779418970c62369f676a0b4de5fff112aa68e512753db8",
        "own_at_t0": [5.0, 8.9],
        "own_heading_deg": 0.0,
        "target_start": [3.94, 4.62],
        "target_heading_deg": 358.9,
        "target_speed_mps": 0.93,
        "tcpa_s": 11.3,
        "dcpa_m": 1.25,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 16.5,
        "nominal_min_target_range_m": 1.68},
    "medium_1": {
        "mode": "medium",
        "encounter": "HO",
        "layout": "L1",
        "generator_seed": 865010,
        "episode_seed": 885010,
        "digest": "2d0b4eddbc88e8381c1ef214a1577e1a898f3e3e5392c3740cf3a3c510bf6bd0",
        "own_at_t0": [5.0, 2.0],
        "own_heading_deg": 0.0,
        "target_start": [5.18, 20.64],
        "target_heading_deg": 174.0,
        "target_speed_mps": 0.68,
        "tcpa_s": 15.1,
        "dcpa_m": 1.26,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": False,
        "nominal_time_s": 28.0,
        "nominal_min_target_range_m": 3.72},
    "medium_2": {
        "mode": "medium",
        "encounter": "CRS",
        "layout": "L1",
        "generator_seed": 866037,
        "episode_seed": 886037,
        "digest": "11783dd8188ddc2e99b8cac309f57c0b5c85b26b3d45f71c7fcffc4c1ef1b41d",
        "own_at_t0": [5.0, 2.0],
        "own_heading_deg": 0.0,
        "target_start": [7.73, 8.95],
        "target_heading_deg": 221.1,
        "target_speed_mps": 0.4,
        "tcpa_s": 8.3,
        "dcpa_m": 0.58,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": False,
        "nominal_time_s": 29.0,
        "nominal_min_target_range_m": 2.36},
    "medium_3": {
        "mode": "medium",
        "encounter": "CRP",
        "layout": "L1",
        "generator_seed": 867137,
        "episode_seed": 887137,
        "digest": "7e7952670443a49567b7e76cef01f6acdb3dbca2082815f51ce0473f6501dad5",
        "own_at_t0": [5.0, 2.0],
        "own_heading_deg": 0.0,
        "target_start": [1.56, 6.32],
        "target_heading_deg": 85.1,
        "target_speed_mps": 0.45,
        "tcpa_s": 8.0,
        "dcpa_m": 0.23,
        "nominal": "10/10",
        "robust": "8/10",
        "rule_conditions_met": True,
        "nominal_time_s": 34.0,
        "nominal_min_target_range_m": 1.74},
    "medium_4": {
        "mode": "medium",
        "encounter": "OT",
        "layout": "L1",
        "generator_seed": 868009,
        "episode_seed": 888009,
        "digest": "846e84fee4b010c3162f0d01ca67be8d63057a4552c0a1a2ee7b0119542ef7a6",
        "own_at_t0": [5.0, 2.0],
        "own_heading_deg": 0.0,
        "target_start": [6.87, 7.88],
        "target_heading_deg": 352.6,
        "target_speed_mps": 0.29,
        "tcpa_s": 22.2,
        "dcpa_m": 1.05,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 30.0,
        "nominal_min_target_range_m": 2.76},
    "medium_5": {
        "mode": "medium",
        "encounter": "BO",
        "layout": "L1",
        "generator_seed": 869144,
        "episode_seed": 889144,
        "digest": "03127dda1c954005e7df494438f047610189623f457ed5e4248bcbd18a00b190",
        "own_at_t0": [5.0, 10.0],
        "own_heading_deg": 0.0,
        "target_start": [4.03, 4.11],
        "target_heading_deg": 357.3,
        "target_speed_mps": 0.98,
        "tcpa_s": 13.6,
        "dcpa_m": 1.62,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 13.0,
        "nominal_min_target_range_m": 4.5},
    "hard_1": {
        "mode": "hard",
        "encounter": "HO",
        "layout": "L3",
        "generator_seed": 875011,
        "episode_seed": 895011,
        "digest": "4d20f1d4fb400ba6ed4908ebd8c922cfe957e45aefc90c2b70290b7f9491afee",
        "own_at_t0": [7.0, 2.0],
        "own_heading_deg": 346.0,
        "target_start": [3.97, 16.3],
        "target_heading_deg": 171.5,
        "target_speed_mps": 0.57,
        "tcpa_s": 13.0,
        "dcpa_m": 0.19,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 24.0,
        "nominal_min_target_range_m": 3.08},
    "hard_2": {
        "mode": "hard",
        "encounter": "CRS",
        "layout": "L3",
        "generator_seed": 876122,
        "episode_seed": 896122,
        "digest": "cf4e6e0031d4e8b5bea2a87bc02149b32f6b999f8aec386ebacc3109838b5de9",
        "own_at_t0": [7.0, 2.0],
        "own_heading_deg": 346.0,
        "target_start": [6.58, 11.65],
        "target_heading_deg": 207.1,
        "target_speed_mps": 0.4,
        "tcpa_s": 10.7,
        "dcpa_m": 0.93,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 24.5,
        "nominal_min_target_range_m": 2.78},
    "hard_3": {
        "mode": "hard",
        "encounter": "CRP",
        "layout": "L3",
        "generator_seed": 877020,
        "episode_seed": 897020,
        "digest": "cfea06a691f657c8cccd13a6abdb1aacb5c6cc19ea68b469193a0fee1ad62e1f",
        "own_at_t0": [7.0, 2.0],
        "own_heading_deg": 346.0,
        "target_start": [2.87, 9.7],
        "target_heading_deg": 153.9,
        "target_speed_mps": 0.49,
        "tcpa_s": 8.3,
        "dcpa_m": 1.29,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 23.5,
        "nominal_min_target_range_m": 3.1},
    "hard_4": {
        "mode": "hard",
        "encounter": "OT",
        "layout": "L2",
        "generator_seed": 873004,
        "episode_seed": 893004,
        "digest": "81d4abb8c45f8abe289c083bc86d7758f91555e7e458d02a4975f303fc54ba4d",
        "own_at_t0": [3.0, 2.0],
        "own_heading_deg": 14.0,
        "target_start": [4.77, 10.65],
        "target_heading_deg": 23.5,
        "target_speed_mps": 0.23,
        "tcpa_s": 26.3,
        "dcpa_m": 0.62,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": True,
        "nominal_time_s": 27.0,
        "nominal_min_target_range_m": 3.35},
    "hard_5": {
        "mode": "hard",
        "encounter": "BO",
        "layout": "L2",
        "generator_seed": 874037,
        "episode_seed": 894037,
        "digest": "367e0c339ec1a70adae79876ca0d09a25b7403704fb725f96e5ab746e3e7904c",
        "own_at_t0": [4.67, 8.7],
        "own_heading_deg": 14.0,
        "target_start": [3.69, 4.55],
        "target_heading_deg": 21.5,
        "target_speed_mps": 0.94,
        "tcpa_s": 10.3,
        "dcpa_m": 1.37,
        "nominal": "10/10",
        "robust": "10/10",
        "rule_conditions_met": False,
        "nominal_time_s": 14.5,
        "nominal_min_target_range_m": 3.33},
}
# END CASES


class TestCase:
    """The field test cases, loadable by name (the interface of the July bridge's `test_run.TestCase`)."""

    def names(self, mode: Optional[str] = None) -> List[str]:
        return [n for n in CASES if mode is None or n.startswith(mode + "_")]

    def info(self, name: str) -> Dict:
        if name not in CASES:
            raise KeyError(f"{name}: not a field test case ({', '.join(CASES) or 'none written yet'})")
        return CASES[name]

    def position(self, name: str) -> Tuple[float, float, float, float]:
        """Start and goal of the leg: sx, sy, gx, gy."""
        (sx, sy), (gx, gy) = leg(self.info(name)["layout"])
        return sx, sy, gx, gy

    def obstacles(self, name: str) -> List[List[Tuple[float, float]]]:
        """The panel polygons (1 m squares; none in the easy mode)."""
        return panels(self.info(name)["layout"])

    def target(self, name: str) -> Dict:
        c = self.info(name)
        return {"start": tuple(c["target_start"]), "heading_deg": c["target_heading_deg"],
                "speed_mps": c["target_speed_mps"]}

    def scenario(self, name: str) -> Tuple[scn.Scenario, int]:
        """The simulator scenario and its episode seed, rebuilt and checked against the record."""
        c = self.info(name)
        built = build(name, c["layout"], c["encounter"], c["generator_seed"])
        if built is None or built.digest() != c["digest"]:
            raise ValueError(f"{name}: the rebuilt scenario differs from the recorded one "
                             "(the generator or a constant changed since the suite was built)")
        return built, int(c["episode_seed"])

    def field_sheet(self, name: str) -> Dict:
        """What the field crew sets up for the case."""
        c = self.info(name)
        sx, sy, gx, gy = self.position(name)
        return {"case": name, "mode": c["mode"], "encounter": ENCOUNTER_NAMES[c["encounter"]],
                "layout": c["layout"], "leg_start": (sx, sy), "leg_goal": (gx, gy),
                "panel_centres": [tuple(map(float, xy)) for xy in
                                  ([] if c["layout"] == "OPEN" else p2.LAYOUTS[c["layout"]]["panels"])],
                "own_at_t0": tuple(c["own_at_t0"]), "own_heading_deg": c["own_heading_deg"],
                "target_start": tuple(c["target_start"]), "target_heading_deg": c["target_heading_deg"],
                "target_speed_mps": c["target_speed_mps"], "target_stops_short_of_wall_m": ct.FIELD_WALL_MARGIN_M,
                "closest_approach_s": c["tcpa_s"], "note": c.get("note", "")}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "runs" / "sac_formulation_seed0_bl3" / "best_model.zip"


def simulate(name: str, *, seed: Optional[int] = None, randomised: bool = False, render: bool = False,
             model_path: Path = DEFAULT_MODEL, model=None) -> Dict:
    """One episode of a case with the deployed policy (deterministic actions), as the suite was
    verified: the identified plant unless `randomised`, the emergency stop of the environment
    off, the start at cruise.  Returns the outcome and a few metrics."""
    import numpy as np
    import curriculum
    import train_formulation as tf
    from env import ASVLidarEnv
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    if model is None:
        import json
        config = Path(model_path).parent / "config.json"
        algo = json.loads(config.read_text()).get("algo", "sac") if config.exists() else "sac"
        model = tf.ALGORITHMS[algo].load(str(model_path), device="cpu")
    import constants as cfg
    built, episode_seed = TestCase().scenario(name)
    env = ASVLidarEnv(render_mode="human" if render else None, emergency_stop=False, low_speed_start_frac=0.0,
                      vessel_randomisation=cfg.VESSEL_RANDOMISATION_SCALE if randomised else None)
    actor = tf.EpisodeActor(model) if hasattr(tf, "EpisodeActor") else None
    obs, _ = env.reset(seed=episode_seed if seed is None else int(seed), options={"generated": copy.deepcopy(built)})
    outcome, steps, min_range = "timeout", 0, math.inf
    while True:
        if actor is not None:
            action = actor(obs)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(np.asarray(action, dtype=np.float32).reshape(2))
        steps += 1
        if env.targets:
            t = env.targets[0]
            min_range = min(min_range, math.hypot(t.x - env.asv_x, t.y - env.asv_y))
        if render:
            env.render()
        if term or trunc:
            if info.get("collided"):
                outcome = f"collision:{info.get('collision_kind')}"
            elif info.get("reached_goal"):
                outcome = "goal"
            break
    env.close()
    return {"case": name, "seed": episode_seed if seed is None else int(seed), "outcome": outcome,
            "time_s": round(steps * cfg.UPDATE_RATE, 1), "min_target_range_m": round(min_range, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Field test cases: list, show the field sheet, or simulate.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("sheet")
    s.add_argument("name")
    r = sub.add_parser("run")
    r.add_argument("names", nargs="+", help="case names, or easy / medium / hard / all")
    r.add_argument("--seeds", type=int, default=1, help="sensor-noise seeds from the case's episode seed")
    r.add_argument("--randomised", action="store_true", help="draw the plant from the training randomisation")
    r.add_argument("--render", action="store_true")
    r.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    case = TestCase()
    if args.cmd == "list":
        for n in case.names():
            c = CASES[n]
            print(f"{n:<9} {c['layout']:<5} {ENCOUNTER_NAMES[c['encounter']]:<42} target {c['target_speed_mps']:.2f} m/s"
                  f"  nominal {c['nominal']}  robust {c['robust']}")
        return 0
    if args.cmd == "sheet":
        for k, v in case.field_sheet(args.name).items():
            print(f"{k:<28} {v}")
        return 0
    names = []
    for n in args.names:
        names += case.names() if n == "all" else case.names(n) if n in MODES else [n]
    failed = 0
    for n in names:
        _, seed0 = case.scenario(n)
        for j in range(args.seeds):
            row = simulate(n, seed=seed0 + 10_000 * j, randomised=args.randomised, render=args.render,
                           model_path=args.model)
            failed += row["outcome"] != "goal"
            print(row, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
