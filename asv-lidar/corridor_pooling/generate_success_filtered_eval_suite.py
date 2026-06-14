"""
Generate a success-filtered ASV evaluation/demo suite for one SAC policy.

What this does
--------------
1. Optionally reads an existing saved evaluation suite.
2. Evaluates each candidate scenario with the supplied SAC policy.
3. Saves ONLY scenarios where the policy reaches the goal.
4. If an obstacle group has fewer than N successful scenarios, it generates
   more feasible candidates and keeps testing until that group reaches N.

Default output:
  data/env_setup/eval_suite_success_filtered/asv_success_suite.json
  data/env_setup/eval_suite_success_filtered/cases/obs*_success*.json
  data/env_setup/eval_suite_success_filtered/success_filter_summary.json

Important
---------
This is NOT an unbiased evaluation set. It is a policy-passing scenario pack.
Use the original fixed holdout suite to report generalisation success rate.
Use this success-filtered suite for demonstrations, field-test setup selection,
regression tests on known-pass layouts, or curriculum collection.

Example
-------
python generate_success_filtered_eval_suite.py \
  --model-path sac_old_fixed_rpm.zip \
  --source-suite data/env_setup/eval_suite/asv_eval_suite.json \
  --obstacle-counts 0 1 2 3 4 \
  --n-per-count 100 \
  --border-mode mixed

For a more deployment-like deterministic filter, use:

python generate_success_filtered_eval_suite.py \
  --model-path sac_old_fixed_rpm.zip \
  --obstacle-counts 0 1 2 3 4 \
  --n-per-count 100 \
  --border-mode asymmetric
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3 import SAC

import rl_env
from rl_env import ASVLidarEnv, DEFAULT_EVAL_LAMBDA

# Reuse your existing, already-tested helper functions.
import generate_eval_suite as gen
import evaluate_sac_suite as ev


Point = Tuple[float, float]
Scenario = Dict[str, Any]
Row = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create a fixed-size ASV scenario suite containing only layouts that one SAC policy succeeds on."
    )
    ap.add_argument("--model-path", required=True, help="Path to the SAC .zip policy to filter scenarios with.")
    ap.add_argument(
        "--source-suite",
        default="data/env_setup/eval_suite/asv_eval_suite.json",
        help="Optional existing suite to mine successful scenarios from first. Use '' to skip.",
    )
    ap.add_argument(
        "--out-dir",
        default="data/env_setup/eval_suite_success_filtered",
        help="Output directory for the success-filtered suite.",
    )
    ap.add_argument("--n-per-count", type=int, default=100, help="Number of successful saved scenarios per obstacle count.")
    ap.add_argument(
        "--obstacle-counts",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Obstacle-count groups to save. Use 0 1 2 3 4 5 if you want obs_5 too.",
    )
    ap.add_argument("--base-seed", type=int, default=675973, help="Base seed for generated filler candidates.")
    ap.add_argument("--max-steps", type=int, default=2000, help="Maximum simulation steps per candidate evaluation.")
    ap.add_argument(
        "--max-generated-policy-attempts-per-count",
        type=int,
        default=20000,
        help="Maximum generated feasible candidates to evaluate per obstacle count before failing.",
    )
    ap.add_argument(
        "--max-layout-attempts-per-count",
        type=int,
        default=200000,
        help="Maximum raw layout-generation attempts per obstacle count before failing.",
    )
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic model prediction. Default: true.",
    )
    ap.add_argument(
        "--border-mode",
        choices=["current", "none", "asymmetric", "both", "mixed"],
        default="mixed",
        help=(
            "Observation LiDAR border mode used during filtering. "
            "Use 'current' to leave rl_env.py unchanged."
        ),
    )
    ap.add_argument(
        "--mixed-border-probs",
        nargs=3,
        type=float,
        default=None,
        metavar=("P_NONE", "P_ASYM", "P_BOTH"),
        help="Optional probabilities when --border-mode mixed is used. Example: 0.25 0.50 0.25",
    )
    ap.add_argument("--lambda-value", type=float, default=DEFAULT_EVAL_LAMBDA, help="Lambda override for ASVLidarEnv.")
    ap.add_argument("--map-width", type=float, default=10.0)
    ap.add_argument("--map-height", type=float, default=25.0)
    ap.add_argument("--path-mode", default="straight")
    ap.add_argument("--vertical-prob", type=float, default=0.70, help="Vertical-path probability for generated filler scenarios.")
    ap.add_argument("--slant-max-dx", type=float, default=2.0, help="Maximum start/goal x offset for slanted generated paths.")
    ap.add_argument(
        "--save-attempt-details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save CSV/JSON rows for all policy-evaluated candidates, including failures. Default: true.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return ap.parse_args()


def configure_modules(args: argparse.Namespace) -> None:
    """Push CLI settings into the existing helper modules' global constants."""
    # Evaluation helper settings.
    ev.MAX_STEPS = int(args.max_steps)
    ev.DETERMINISTIC = bool(args.deterministic)
    ev.MAP_WIDTH = float(args.map_width)
    ev.MAP_HEIGHT = float(args.map_height)
    ev.PATH_MODE = str(args.path_mode)
    ev.EVAL_LAMBDA = float(args.lambda_value)

    # Generation helper settings.
    gen.MAP_WIDTH = float(args.map_width)
    gen.MAP_HEIGHT = float(args.map_height)
    gen.PATH_MODE = str(args.path_mode)
    gen.LAMBDA_VALUE = float(args.lambda_value)
    gen.VERTICAL_PROB = float(args.vertical_prob)
    gen.SLANT_MAX_DX = float(args.slant_max_dx)
    gen.BASE_SEED = int(args.base_seed)
    gen.EXACT_OBSTACLE_COUNT = True

    # Border-visibility mode lives in rl_env.py.
    if args.border_mode != "current":
        rl_env.OBS_BORDER_MODE = str(args.border_mode)
    if args.mixed_border_probs is not None:
        p_none, p_asym, p_both = [float(x) for x in args.mixed_border_probs]
        total = p_none + p_asym + p_both
        if total <= 0.0:
            raise ValueError("--mixed-border-probs must sum to a positive value.")
        rl_env.OBS_BORDER_P_NONE = p_none / total
        rl_env.OBS_BORDER_P_ASYMMETRIC = p_asym / total
        rl_env.OBS_BORDER_P_BOTH = p_both / total


def ensure_output_dirs(out_dir: str, overwrite: bool) -> Tuple[str, str, str, str, str, str]:
    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {out_dir}\n"
            f"Use --overwrite or choose a new --out-dir."
        )
    os.makedirs(out_dir, exist_ok=True)
    individual_dir = os.path.join(out_dir, "cases")
    os.makedirs(individual_dir, exist_ok=True)
    return (
        os.path.join(out_dir, "asv_success_suite.json"),
        individual_dir,
        os.path.join(out_dir, "success_filter_summary.json"),
        os.path.join(out_dir, "success_filter_attempts.json"),
        os.path.join(out_dir, "success_filter_attempts.csv"),
        os.path.join(out_dir, "success_filter_README.txt"),
    )


def path_length(points: Sequence[Sequence[float]]) -> float:
    if points is None or len(points) < 2:
        return 0.0
    p = np.asarray(points, dtype=np.float32)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def load_source_scenarios(path: str) -> List[Scenario]:
    if not path:
        return []
    if not os.path.exists(path):
        print(f"[WARN] Source suite not found, skipping: {path}")
        return []
    with open(path, "r") as f:
        payload = json.load(f)
    scenarios = payload.get("scenarios", payload if isinstance(payload, list) else [])
    if not isinstance(scenarios, list):
        raise ValueError(f"Could not read scenarios from source suite: {path}")
    print(f"Loaded {len(scenarios)} candidate scenarios from source suite: {path}")
    return scenarios


def normalize_source_scenario(sc: Scenario, source_suite: str) -> Scenario:
    out = copy.deepcopy(sc)
    out.setdefault("obstacle_count", len(out.get("obstacles", [])))
    out.setdefault("group", f"obs_{int(out['obstacle_count'])}")
    out.setdefault("map_width", float(gen.MAP_WIDTH))
    out.setdefault("map_height", float(gen.MAP_HEIGHT))
    out.setdefault("path_mode", str(gen.PATH_MODE))
    out["candidate_source"] = "source_suite"
    out["source_suite"] = source_suite
    out["source_case_id"] = out.get("case_id")
    return out


def make_generated_candidate(env: ASVLidarEnv, obs_count: int, seed: int, serial: int) -> Optional[Scenario]:
    rng = np.random.default_rng(int(seed))
    sx, sy, gx, gy = gen.sample_start_goal(rng)
    path = gen.build_path_for_env(env, sx, sy, gx, gy)
    obstacles = gen.generate_obstacles_exact(env, obs_count, seed + 17)

    if len(obstacles) != obs_count:
        return None

    ok, route_ratio = gen.feasible((sx, sy), (gx, gy), obstacles)
    if not ok:
        return None

    return {
        "case_id": int(serial),
        "group": f"obs_{obs_count}",
        "obstacle_count": int(obs_count),
        "seed": int(seed),
        "start": [float(sx), float(sy)],
        "goal": [float(gx), float(gy)],
        "obstacles": [[list(p) for p in obs] for obs in obstacles],
        "path": path.tolist(),
        "map_width": float(gen.MAP_WIDTH),
        "map_height": float(gen.MAP_HEIGHT),
        "path_mode": str(gen.PATH_MODE),
        "route_ratio_astar": None if route_ratio is None else float(route_ratio),
        "candidate_source": "generated",
        "source_case_id": None,
    }


def evaluate_candidate(model: SAC, env: ASVLidarEnv, scenario: Scenario, candidate_index: int) -> Row:
    # evaluate_sac_suite.evaluate_one returns the core metrics and termination reason.
    row = ev.evaluate_one(model, env, scenario)
    row["candidate_index"] = int(candidate_index)
    row["candidate_source"] = str(scenario.get("candidate_source", "unknown"))
    row["candidate_seed"] = int(scenario.get("seed", -1))
    row["source_case_id"] = scenario.get("source_case_id")
    row["filter_obs_border_mode"] = str(getattr(env, "obs_border_mode_used", "unknown"))
    row["filter_lambda"] = float(getattr(env, "current_lambda", np.nan))
    return row


def is_success(row: Row) -> bool:
    return int(row.get("success", 0)) == 1 and str(row.get("term_reason")) == "goal"


def final_case_id(obs_count: int, local_idx: int, n_per_count: int) -> int:
    return int(obs_count * n_per_count + local_idx)


def accept_scenario(
    scenario: Scenario,
    row: Row,
    *,
    obs_count: int,
    local_idx: int,
    n_per_count: int,
    model_path: str,
    individual_dir: str,
    suite_json_path: str,
) -> Scenario:
    accepted = copy.deepcopy(scenario)
    accepted["source_case_id"] = scenario.get("source_case_id")
    accepted["case_id"] = final_case_id(obs_count, local_idx, n_per_count)
    accepted["group"] = f"obs_{obs_count}"
    accepted["obstacle_count"] = int(obs_count)
    accepted["policy_filter"] = {
        "model_path": model_path,
        "success": int(row.get("success", 0)),
        "term_reason": str(row.get("term_reason", "unknown")),
        "ep_len": int(row.get("ep_len", 0)),
        "ep_reward": float(row.get("ep_reward", 0.0)),
        "mean_abs_cte": float(row.get("mean_abs_cte", 0.0)),
        "mean_abs_course_error": float(row.get("mean_abs_course_error", 0.0)),
        "mean_speed": float(row.get("mean_speed", 0.0)),
        "mean_rpm": float(row.get("mean_rpm", 0.0)),
        "mean_abs_rudder_deg": float(row.get("mean_abs_rudder_deg", 0.0)),
        "min_front_clearance": float(row.get("min_front_clearance", np.nan)),
        "min_border_clearance": float(row.get("min_border_clearance", np.nan)),
        "path_efficiency": float(row.get("path_efficiency", np.nan)),
        "obs_border_mode": str(row.get("filter_obs_border_mode", "unknown")),
        "lambda": float(row.get("filter_lambda", np.nan)),
        "candidate_source": str(row.get("candidate_source", "unknown")),
        "candidate_index": int(row.get("candidate_index", -1)),
    }

    individual_path = os.path.join(individual_dir, f"obs{obs_count}_success{local_idx:03d}.json")
    with open(individual_path, "w") as f:
        json.dump(
            {
                "start": accepted["start"],
                "goal": accepted["goal"],
                "obstacles": accepted.get("obstacles", []),
                "map_width": accepted.get("map_width"),
                "map_height": accepted.get("map_height"),
                "path_mode": accepted.get("path_mode"),
                "path": accepted.get("path", []),
                "source_suite": suite_json_path,
                "case_id": accepted["case_id"],
                "source_case_id": accepted.get("source_case_id"),
                "obstacle_count": int(obs_count),
                "policy_filter": accepted["policy_filter"],
            },
            f,
            indent=2,
        )
    return accepted


def write_csv(path: str, rows: List[Row]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.n_per_count <= 0:
        raise ValueError("--n-per-count must be positive.")
    if not args.obstacle_counts:
        raise ValueError("--obstacle-counts cannot be empty.")

    configure_modules(args)
    suite_json_path, individual_dir, summary_json_path, attempts_json_path, attempts_csv_path, readme_path = ensure_output_dirs(
        args.out_dir, args.overwrite
    )

    print(f"Loading SAC model: {args.model_path}")
    model = SAC.load(args.model_path)

    env = ASVLidarEnv(
        render_mode=None,
        map_width=float(args.map_width),
        map_height=float(args.map_height),
        max_obs=max(args.obstacle_counts),
        path_mode=str(args.path_mode),
        lambda_override=float(args.lambda_value),
        test_case=None,
        record_video=False,
    )
    env.reset(seed=int(args.base_seed))

    wanted = {int(n): int(args.n_per_count) for n in args.obstacle_counts}
    accepted_counts = {int(n): 0 for n in args.obstacle_counts}
    stats: Dict[int, Dict[str, int]] = {
        int(n): {
            "source_policy_attempts": 0,
            "source_accepted": 0,
            "generated_layout_attempts": 0,
            "generated_policy_attempts": 0,
            "generated_accepted": 0,
        }
        for n in args.obstacle_counts
    }

    accepted_scenarios: List[Scenario] = []
    attempt_rows: List[Row] = []
    candidate_index = 0
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1) Mine successes from existing suite first.
    # ------------------------------------------------------------------
    source_scenarios = load_source_scenarios(args.source_suite)
    for raw_sc in source_scenarios:
        sc = normalize_source_scenario(raw_sc, args.source_suite)
        obs_count = int(sc.get("obstacle_count", len(sc.get("obstacles", []))))
        if obs_count not in wanted:
            continue
        if accepted_counts[obs_count] >= wanted[obs_count]:
            continue

        # Ensure evaluate_one has a case_id even if the source did not.
        sc.setdefault("case_id", int(candidate_index))
        row = evaluate_candidate(model, env, sc, candidate_index)
        candidate_index += 1
        stats[obs_count]["source_policy_attempts"] += 1
        attempt_rows.append(row)

        if is_success(row):
            local_idx = accepted_counts[obs_count]
            accepted = accept_scenario(
                sc,
                row,
                obs_count=obs_count,
                local_idx=local_idx,
                n_per_count=args.n_per_count,
                model_path=args.model_path,
                individual_dir=individual_dir,
                suite_json_path=suite_json_path,
            )
            accepted_scenarios.append(accepted)
            accepted_counts[obs_count] += 1
            stats[obs_count]["source_accepted"] += 1

    print("\nAfter mining source suite:")
    for obs_count in args.obstacle_counts:
        print(
            f"  obs_{obs_count}: {accepted_counts[obs_count]:3d}/{wanted[obs_count]} "
            f"from source attempts={stats[obs_count]['source_policy_attempts']}"
        )

    # ------------------------------------------------------------------
    # 2) Generate and evaluate filler candidates until each group has N.
    # ------------------------------------------------------------------
    for obs_count in args.obstacle_counts:
        layout_attempts = 0
        generated_policy_attempts = 0
        next_progress_print = max(10, args.n_per_count // 10)

        while accepted_counts[obs_count] < wanted[obs_count]:
            if layout_attempts >= args.max_layout_attempts_per_count:
                raise RuntimeError(
                    f"Hit --max-layout-attempts-per-count for obs_{obs_count}. "
                    f"Accepted {accepted_counts[obs_count]}/{wanted[obs_count]}."
                )
            if generated_policy_attempts >= args.max_generated_policy_attempts_per_count:
                raise RuntimeError(
                    f"Hit --max-generated-policy-attempts-per-count for obs_{obs_count}. "
                    f"Accepted {accepted_counts[obs_count]}/{wanted[obs_count]}."
                )

            layout_attempts += 1
            stats[obs_count]["generated_layout_attempts"] += 1
            seed = int(args.base_seed + 10_000_000 + obs_count * 1_000_000 + layout_attempts)
            sc = make_generated_candidate(env, obs_count, seed, candidate_index)
            if sc is None:
                continue

            row = evaluate_candidate(model, env, sc, candidate_index)
            candidate_index += 1
            generated_policy_attempts += 1
            stats[obs_count]["generated_policy_attempts"] += 1
            attempt_rows.append(row)

            if is_success(row):
                local_idx = accepted_counts[obs_count]
                accepted = accept_scenario(
                    sc,
                    row,
                    obs_count=obs_count,
                    local_idx=local_idx,
                    n_per_count=args.n_per_count,
                    model_path=args.model_path,
                    individual_dir=individual_dir,
                    suite_json_path=suite_json_path,
                )
                accepted_scenarios.append(accepted)
                accepted_counts[obs_count] += 1
                stats[obs_count]["generated_accepted"] += 1

                if accepted_counts[obs_count] % next_progress_print == 0 or accepted_counts[obs_count] == wanted[obs_count]:
                    print(
                        f"obs_{obs_count}: saved {accepted_counts[obs_count]:3d}/{wanted[obs_count]} successes "
                        f"(generated policy attempts={generated_policy_attempts}, layout attempts={layout_attempts})"
                    )

    # Keep deterministic ordering: obs_0 cases first, then obs_1, ...
    accepted_scenarios.sort(key=lambda s: (int(s["obstacle_count"]), int(s["case_id"])))

    # ------------------------------------------------------------------
    # 3) Write suite and summaries.
    # ------------------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    for obs_count in args.obstacle_counts:
        st = stats[obs_count]
        policy_attempts = st["source_policy_attempts"] + st["generated_policy_attempts"]
        accepted = accepted_counts[obs_count]
        summary_rows.append(
            {
                "group": f"obs_{obs_count}",
                "obstacle_count": int(obs_count),
                "saved_successful_scenarios": int(accepted),
                "requested_successful_scenarios": int(wanted[obs_count]),
                "source_policy_attempts": int(st["source_policy_attempts"]),
                "source_accepted": int(st["source_accepted"]),
                "generated_layout_attempts": int(st["generated_layout_attempts"]),
                "generated_policy_attempts": int(st["generated_policy_attempts"]),
                "generated_accepted": int(st["generated_accepted"]),
                "total_policy_attempts": int(policy_attempts),
                "acceptance_rate_among_policy_attempts": float(accepted / policy_attempts) if policy_attempts > 0 else float("nan"),
            }
        )

    suite = {
        "metadata": {
            "description": (
                "Success-filtered ASV suite: each saved scenario was evaluated with the specified policy "
                "and kept only if the policy reached the goal. This is not an unbiased evaluation set."
            ),
            "created_by": os.path.basename(__file__),
            "model_path": args.model_path,
            "source_suite": args.source_suite,
            "suite_json": suite_json_path,
            "map_width": float(args.map_width),
            "map_height": float(args.map_height),
            "path_mode": str(args.path_mode),
            "n_per_obstacle_count": int(args.n_per_count),
            "obstacle_counts": [int(n) for n in args.obstacle_counts],
            "base_seed": int(args.base_seed),
            "vertical_probability_for_generated_fillers": float(args.vertical_prob),
            "slant_max_dx_for_generated_fillers": float(args.slant_max_dx),
            "max_steps": int(args.max_steps),
            "deterministic_policy": bool(args.deterministic),
            "lambda_value": float(args.lambda_value),
            "obs_border_mode": str(rl_env.OBS_BORDER_MODE),
            "obs_border_probs": {
                "none": float(getattr(rl_env, "OBS_BORDER_P_NONE", np.nan)),
                "asymmetric": float(getattr(rl_env, "OBS_BORDER_P_ASYMMETRIC", np.nan)),
                "both": float(getattr(rl_env, "OBS_BORDER_P_BOTH", np.nan)),
            },
            "fixed_rpm_in_rl_env": bool(getattr(rl_env, "FIXED_RPM", False)),
            "summary": summary_rows,
        },
        "scenarios": accepted_scenarios,
    }

    with open(suite_json_path, "w") as f:
        json.dump(suite, f, indent=2)
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    if args.save_attempt_details:
        with open(attempts_json_path, "w") as f:
            json.dump(attempt_rows, f, indent=2)
        write_csv(attempts_csv_path, attempt_rows)

    with open(readme_path, "w") as f:
        f.write(
            "ASV success-filtered scenario suite\n"
            "===================================\n\n"
            "This directory contains scenarios that were kept only because the supplied SAC policy reached the goal.\n"
            "Do not use this as an unbiased success-rate benchmark. Use the original holdout suite for reporting.\n\n"
            f"Model: {args.model_path}\n"
            f"Suite: {suite_json_path}\n"
            f"Obstacle counts: {args.obstacle_counts}\n"
            f"Successful scenarios per count: {args.n_per_count}\n"
            f"Border mode during filtering: {rl_env.OBS_BORDER_MODE}\n"
            f"Deterministic policy: {args.deterministic}\n"
        )

    elapsed = time.time() - t0
    print("\nDone. Success-filtered suite created.")
    for row in summary_rows:
        print(
            f"{row['group']:>6s}: saved={row['saved_successful_scenarios']:3d} "
            f"policy_attempts={row['total_policy_attempts']:4d} "
            f"acceptance={row['acceptance_rate_among_policy_attempts']:.3f} "
            f"source_kept={row['source_accepted']:3d} generated_kept={row['generated_accepted']:3d}"
        )
    print(f"\nSaved suite: {suite_json_path}")
    print(f"Saved individual cases: {individual_dir}")
    print(f"Saved summary: {summary_json_path}")
    if args.save_attempt_details:
        print(f"Saved attempt details: {attempts_csv_path}")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
