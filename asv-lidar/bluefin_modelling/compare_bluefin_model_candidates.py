from __future__ import annotations

import importlib.util
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from bluefin_test_utils import (
    extract_motion_metrics,
    extract_turn_metrics,
    plot_open_loop_response,
    plot_path,
    save_json,
)

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "model_candidate_comparison"
DT = 0.1
STRAIGHT_DURATION = 40.0
TURN_DURATION = 50.0

MOTION_KEYS = [
    "peak_u_body_mps",
    "initial_accel_0_2_after_motion_mps2",
    "initial_accel_0_5_after_motion_mps2",
    "distance_at_10s_after_motion_m",
    "time_to_50pct_peak_u_after_motion_s",
    "time_to_90pct_peak_u_after_motion_s",
]
TURN_KEYS = [
    "peak_abs_yaw_rate_degps",
    "time_to_90deg_after_turn_s",
    "time_to_180deg_after_turn_s",
    "radius_first_90deg_m",
    "radius_first_180deg_m",
    "u_body_10s_after_turn_mps",
]
MOTION_WEIGHTS = {
    "peak_u_body_mps": 2.0,
    "initial_accel_0_2_after_motion_mps2": 2.0,
    "initial_accel_0_5_after_motion_mps2": 1.5,
    "distance_at_10s_after_motion_m": 2.0,
    "time_to_50pct_peak_u_after_motion_s": 1.5,
    "time_to_90pct_peak_u_after_motion_s": 1.0,
}
TURN_WEIGHTS = {
    "peak_abs_yaw_rate_degps": 2.5,
    "time_to_90deg_after_turn_s": 2.0,
    "time_to_180deg_after_turn_s": 1.5,
    "radius_first_90deg_m": 1.5,
    "radius_first_180deg_m": 1.0,
    "u_body_10s_after_turn_mps": 1.0,
}

V1_TUNED_OVERRIDES = {
    "MASS": 64.55,
    "THRUST_COEF": 0.07,
    "DRAG_COEF": 1.5,
    "LINEAR_SURGE_DAMP": 2.0,
    "TURN_COEF": 5.0,
    "RUDDER_FORCE_SCALE": 0.1,
    "LINEAR_YAW_DAMP": 4.0,
}
V2_TUNED_OVERRIDES = {
    "MASS": 64.55,
    "THRUST_COEF": 0.06,
    "DRAG_COEF": 1.5,
    "THRUST_LOW_SPEED_BOOST": 1.6,
    "THRUST_HIGH_SPEED_DECAY": 0.26,
    "LINEAR_SURGE_DAMP": 2.0,
    "TURN_COEF": 3.0,
    "RUDDER_FORCE_SCALE": 0.32,
    "RUDDER_YAW_SCALE": 2.6,
    "RUDDER_X_DRAG_SCALE": 0.02,
    "LINEAR_YAW_DAMP": 1.5,
}


@dataclass
class Candidate:
    name: str
    path: Path
    straight_rpms: List[float]
    turn_rpms: List[float]
    turn_rudders_deg: List[float]
    overrides: Dict[str, float] = field(default_factory=dict)
    override_grid: Dict[str, List[float]] = field(default_factory=dict)
    notes: str = ""


def rel_error(sim: Optional[float], real: Optional[float], floor: float = 1e-6) -> float:
    if sim is None or real is None:
        return 10.0
    return abs(sim - real) / max(abs(real), floor)


def score_section(
    sim_metrics: Dict[str, Any],
    real_targets: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    parts: Dict[str, float] = {}
    total = 0.0
    for key, wt in weights.items():
        err = rel_error(sim_metrics.get(key), real_targets.get(key))
        parts[key] = err
        total += wt * err
    return {"score_total": total, "parts": parts}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_real_targets() -> tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    test3 = load_json(THIS_DIR / "test_3_metrics.json")
    test4 = load_json(THIS_DIR / "test_4_metrics.json")
    motion_targets = {key: test3["straight_metrics"].get(key) for key in MOTION_KEYS}
    turn_targets = {key: test4["turn_metrics"].get(key) for key in TURN_KEYS}
    return motion_targets, turn_targets


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_overrides(module: Any, overrides: Dict[str, float]) -> None:
    for key, value in overrides.items():
        if hasattr(module, key):
            setattr(module, key, float(value))


def call_update(model: Any, rpm: float, rudder_percent: float, dt: float, thruster_rpm: float = 0.0):
    try:
        return model.update(rpm, rudder_percent, dt, thruster_rpm=thruster_rpm)
    except TypeError:
        return model.update(rpm, rudder_percent, dt)


def simulate_open_loop(
    module: Any,
    *,
    duration_s: float,
    dt: float,
    rpm: float,
    rudder_deg: float,
    overrides: Dict[str, float],
    thruster_rpm: float = 0.0,
) -> Dict[str, np.ndarray]:
    apply_overrides(module, overrides)
    model = module.ShipModel()

    n = int(np.floor(duration_s / dt)) + 1
    t = np.arange(n, dtype=float) * dt

    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    heading_deg = np.zeros(n, dtype=float)
    yaw_rate_degps = np.zeros(n, dtype=float)
    u_body = np.zeros(n, dtype=float)
    v_body = np.zeros(n, dtype=float)
    rudder_deg_cmd = np.full(n, float(rudder_deg), dtype=float)
    rudder_percent_cmd = np.full(n, (rudder_deg / 40.0) * 100.0, dtype=float)
    rpm_cmd = np.full(n, float(rpm), dtype=float)

    xk = 0.0
    yk = 0.0
    rudder_percent = (rudder_deg / 40.0) * 100.0

    for i in range(n):
        dx, dy, hdg_deg, yawrate_deg = call_update(model, rpm, rudder_percent, dt, thruster_rpm=thruster_rpm)
        xk += dx
        yk += dy

        x[i] = xk
        y[i] = yk
        heading_deg[i] = hdg_deg
        yaw_rate_degps[i] = yawrate_deg
        u_body[i] = float(getattr(model, "_u", getattr(model, "_v", 0.0)))
        v_body[i] = float(getattr(model, "_v_sway", 0.0))

    return {
        "t_sec": t,
        "x_m": x,
        "y_m": y,
        "heading_deg": heading_deg,
        "yaw_rate_degps": yaw_rate_degps,
        "u_body_mps": u_body,
        "v_body_mps": v_body,
        "rudder_deg_cmd": rudder_deg_cmd,
        "rudder_percent_cmd": rudder_percent_cmd,
        "rpm_cmd": rpm_cmd,
    }


def iter_override_sets(override_grid: Dict[str, List[float]]) -> Iterable[Dict[str, float]]:
    if not override_grid:
        yield {}
        return
    keys = list(override_grid.keys())
    for values in itertools.product(*(override_grid[key] for key in keys)):
        yield {key: float(value) for key, value in zip(keys, values)}


def build_comparison(keys: List[str], sim_metrics: Dict[str, Any], real_targets: Dict[str, Optional[float]]) -> Dict[str, Any]:
    return {
        key: {
            "real": real_targets.get(key),
            "sim": sim_metrics.get(key),
            "rel_error": rel_error(sim_metrics.get(key), real_targets.get(key)),
        }
        for key in keys
    }


def evaluate_candidate(candidate: Candidate, motion_targets: Dict[str, Optional[float]], turn_targets: Dict[str, Optional[float]]) -> Dict[str, Any]:
    module = load_module_from_path(f"candidate_{candidate.name}", candidate.path)
    candidate_dir = OUT_DIR / candidate.name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    best_motion: Optional[Dict[str, Any]] = None
    best_turn: Optional[Dict[str, Any]] = None

    for extra_overrides in iter_override_sets(candidate.override_grid):
        active_overrides = dict(candidate.overrides)
        active_overrides.update(extra_overrides)

        for rpm in candidate.straight_rpms:
            sim = simulate_open_loop(
                module,
                duration_s=STRAIGHT_DURATION,
                dt=DT,
                rpm=rpm,
                rudder_deg=0.0,
                overrides=active_overrides,
            )
            metrics = extract_motion_metrics(sim)
            score = score_section(metrics, motion_targets, MOTION_WEIGHTS)
            row = {
                "rpm": rpm,
                "overrides": active_overrides,
                "score_total": score["score_total"],
                "parts": score["parts"],
                "metrics": metrics,
                "sim": sim,
            }
            if best_motion is None or row["score_total"] < best_motion["score_total"]:
                best_motion = row

        for rpm in candidate.turn_rpms:
            for rudder_deg in candidate.turn_rudders_deg:
                sim = simulate_open_loop(
                    module,
                    duration_s=TURN_DURATION,
                    dt=DT,
                    rpm=rpm,
                    rudder_deg=rudder_deg,
                    overrides=active_overrides,
                )
                metrics = extract_turn_metrics(sim)
                score = score_section(metrics, turn_targets, TURN_WEIGHTS)
                row = {
                    "rpm": rpm,
                    "rudder_deg": rudder_deg,
                    "overrides": active_overrides,
                    "score_total": score["score_total"],
                    "parts": score["parts"],
                    "metrics": metrics,
                    "sim": sim,
                }
                if best_turn is None or row["score_total"] < best_turn["score_total"]:
                    best_turn = row

    if best_motion is None or best_turn is None:
        raise RuntimeError(f"Candidate {candidate.name} produced no simulations")

    plot_open_loop_response(candidate_dir / "best_straight_response.png", best_motion["sim"], f"{candidate.name} straight")
    plot_path(candidate_dir / "best_straight_path.png", best_motion["sim"], f"{candidate.name} straight path")
    plot_open_loop_response(candidate_dir / "best_turn_response.png", best_turn["sim"], f"{candidate.name} turn")
    plot_path(candidate_dir / "best_turn_path.png", best_turn["sim"], f"{candidate.name} turn path")

    motion_report = build_comparison(MOTION_KEYS, best_motion["metrics"], motion_targets)
    turn_report = build_comparison(TURN_KEYS, best_turn["metrics"], turn_targets)

    save_json(candidate_dir / "best_motion_metrics.json", best_motion["metrics"])
    save_json(candidate_dir / "best_motion_comparison.json", {"motion": motion_report})
    save_json(candidate_dir / "best_turn_metrics.json", best_turn["metrics"])
    save_json(candidate_dir / "best_turn_comparison.json", {"turn": turn_report})

    result = {
        "candidate": candidate.name,
        "path": str(candidate.path),
        "notes": candidate.notes,
        "best_motion": {
            "rpm": best_motion["rpm"],
            "overrides": best_motion["overrides"],
            "score_total": best_motion["score_total"],
            "metrics": best_motion["metrics"],
            "errors": best_motion["parts"],
        },
        "best_turn": {
            "rpm": best_turn["rpm"],
            "rudder_deg": best_turn["rudder_deg"],
            "overrides": best_turn["overrides"],
            "score_total": best_turn["score_total"],
            "metrics": best_turn["metrics"],
            "errors": best_turn["parts"],
        },
    }
    result["joint_score"] = best_motion["score_total"] + best_turn["score_total"]
    save_json(candidate_dir / "summary.json", result)
    return result


def build_candidates() -> List[Candidate]:
    return [
        Candidate(
            name="simple_baseline",
            path=THIS_DIR / "old_models" / "ship_model.py",
            straight_rpms=[20, 30, 40, 50, 60, 80, 100],
            turn_rpms=[20, 30, 40, 50, 60, 80, 100],
            turn_rudders_deg=[10, 15, 20, 25, 30],
            notes="Original lumped ship model still used by the RL environment in this checkout.",
        ),
        Candidate(
            name="blue02_lineage_v1_tuned",
            path=THIS_DIR / "replay_results" / "ship_model_bluefin_matlab_style.py",
            straight_rpms=[14.0],
            turn_rpms=[20.0],
            turn_rudders_deg=[25.0],
            overrides=V1_TUNED_OVERRIDES,
            notes="Existing Blue02-inspired Python model with the repo's best known shared-output tuning.",
        ),
        Candidate(
            name="blue02_lineage_v2_tuned",
            path=THIS_DIR / "ship_model_bluefin_v2.py",
            straight_rpms=[15.0],
            turn_rpms=[24.0],
            turn_rudders_deg=[30.0],
            overrides=V2_TUNED_OVERRIDES,
            notes="Current Blue02-lineage model used for Bluefin-focused fitting.",
        ),
        Candidate(
            name="bluefin4dof_direct_port",
            path=THIS_DIR / "ship_model_bluefin_4dof.py",
            straight_rpms=[8.0, 10.0, 12.0, 15.0, 18.0],
            turn_rpms=[8.0, 10.0, 12.0, 15.0, 18.0],
            turn_rudders_deg=[20.0, 25.0, 30.0, 35.0],
            override_grid={"RPM_COMMAND_SCALE": [40.0, 60.0, 80.0, 100.0]},
            notes="Direct runnable port of Bluefin4DOFModel02.m with only command-scale alignment and numerical guards.",
        ),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    motion_targets, turn_targets = build_real_targets()

    results = [evaluate_candidate(candidate, motion_targets, turn_targets) for candidate in build_candidates()]
    results.sort(key=lambda row: row["joint_score"])
    save_json(OUT_DIR / "summary_ranked.json", {"results": results})

    print("Saved ranked comparison to", OUT_DIR / "summary_ranked.json")
    for row in results:
        print(
            f"{row['candidate']}: joint={row['joint_score']:.3f}, "
            f"motion={row['best_motion']['score_total']:.3f}, "
            f"turn={row['best_turn']['score_total']:.3f}"
        )


if __name__ == "__main__":
    main()
