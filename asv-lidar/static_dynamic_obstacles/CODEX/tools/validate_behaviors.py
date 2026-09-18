"""Deterministic end-to-end reference-controller acceptance, one CPU process.

Run: python tools/validate_behaviors.py --noise both --output results/behaviors.json
Exit status is nonzero if any requested case fails. These are development
contracts, not evidence that a PPO policy learned the behaviours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as cfg
from corridor import rectangle
from env import ASVLidarEnv, _polygon_gap
from reference_controller import ReferenceController


def box(x, y, width=1.3, length=1.0):
    return [[x-width/2, y-length/2], [x+width/2, y-length/2], [x+width/2, y+length/2], [x-width/2, y+length/2]]


def cases():
    """Hand-specified feasible geometries, never filtered by controller outcome."""
    target = lambda x, y, h, speed, cls: dict(x=x, y=y, heading=h, speed=speed, encounter_class=cls, confined=False)
    base = dict(start=[5.0, 4.0], goal=[5.0, 22.0], path=[[5.0, y] for y in np.linspace(4.0, 22.0, 91)], obstacles=[], targets=[])
    return {
        "empty": dict(base),
        "recovery": dict(base, start=[6.5, 4.0]),
        "static": dict(base, obstacles=[box(5.0, 12.0)]),
        "head_on": dict(base, targets=[target(5.0, 21.0, 180.0, cfg.U_REF, "head_on")]),
        "crossing_starboard": dict(base, targets=[target(11.0, 12.0, 270.0, 0.42, "crossing")]),
        "crossing_port": dict(base, targets=[target(-1.0, 12.0, 90.0, 0.42, "crossing")]),
        "overtaking": dict(base, targets=[target(5.0, 10.0, 0.0, 0.20, "overtaking")]),
        "being_overtaken": dict(base, start=[5.0, 7.0], targets=[target(2.4, 3.0, 0.0, 0.90, "being_overtaken")]),
        "null": dict(base, targets=[target(2.0, 14.0, 180.0, cfg.U_REF, "null")]),
        "combined": dict(base, obstacles=[box(5.0, 9.0)], targets=[target(5.0, 23.0, 180.0, 0.35, "head_on")]),
    }


def stand_on_speed_metrics(records):
    """Measure actual speed during the pre-action perceived stand-on interval."""
    active = [row for row in records if any(ctx["cls"] == "being_overtaken"
        and ctx["state"] in ("engaged", "clearing")
        for ctx in row["perception"]["contexts"])]
    errors = [abs(row["speed"] - cfg.U_REF) / cfg.U_REF for row in active]
    return dict(active_steps=len(active),
        max_speed_error_fraction=max(errors, default=None),
        mean_speed_error_fraction=float(np.mean(errors)) if errors else None,
        cruise_command_fraction=float(np.mean([abs(row["throttle"]) < 1e-6 for row in active])) if active else None,
        override_steps=sum(row.get("stand_on_override", False) for row in active))


def run_case(name, noise=False, seed=812):
    # The final propulsion stage provides zero-thrust coasting, no reverse.
    cfg.FIXED_RPM = False
    cfg.RPM_DELTA, cfg.RPM_FLOOR, cfg.RPM_CEIL = cfg.RPM_STAGES[4]
    env = ASVLidarEnv(channel=rectangle(10.0), pose_noise=noise,
        pose_stale_prob=cfg.POSE_STALE_PROB if noise else 0.0,
        ego_speed_noise=cfg.EGO_SPEED_NOISE if noise else 0.0,
        ego_yaw_rate_noise_dps=cfg.EGO_YAW_RATE_NOISE_DPS if noise else 0.0,
        vessel_randomisation=None, emergency_stop=False)
    scenario = cases()[name]
    obs, _ = env.reset(seed=seed, options={"scenario": scenario, "initial_speed": cfg.U_REF})
    controller = ReferenceController()
    cte, records, target_distance = [], [], []
    static_clearance, target_clearance, boundary_clearance = [], [], []
    first_alteration = None
    result = {}
    for step in range(300):
        perception = dict(pose=list(env.estimated_pose()), ego=list(env._measured_ego()),
            tracks=[dict(id=int(t.id), position=t.position.tolist(), velocity=t.velocity.tolist()) for t in env.tracks],
            contexts=[dict(id=int(ctx.track_id), cls=str(ctx.cls), state=str(ctx.state), tcpa=float(ctx.tcpa), dcpa=float(ctx.dcpa)) for ctx in env.encounter_contexts.values()])
        action = controller.action(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        cte.append(abs(float(env.cross_track_error)))
        distance = min((math.hypot(t.x-env.asv_x, t.y-env.asv_y) for t in env.targets), default=float("inf"))
        target_distance.append(distance)
        hull = env.hull_polygon()
        static_clearance.extend(_polygon_gap(hull, obstacle)[0] for obstacle in env.obstacles)
        target_clearance.extend(_polygon_gap(hull, target.hull())[0] for target in env.targets)
        boundary_clearance.append(float(info["true_border_clearance"]))
        if first_alteration is None and abs(controller.offset) >= 0.75:
            first_alteration = math.copysign(1.0, controller.offset)
        records.append(dict(t=round(env.elapsed_time, 2), x=env.asv_x, y=env.asv_y, heading=env.asv_h,
            speed=env.speed_mps, offset=controller.offset, rudder=float(action[0]), throttle=float(action[1]),
            target_x=env.targets[0].x if env.targets else None, target_y=env.targets[0].y if env.targets else None,
            perceived_classes=[str(ctx.cls) for ctx in env.encounter_contexts.values()],
            obligations=list(controller.obligations.values()), cost=controller.last_diagnostics["predicted_cost"], perception=perception,
            fallback=bool(controller.last_diagnostics.get("fallback", False)),
            rule_relaxed=bool(controller.last_diagnostics.get("rule_relaxed", False)),
            stand_on_override=bool(controller.last_diagnostics.get("stand_on_override", False))))
        if terminated or truncated:
            break
    checks = {"goal": bool(info.get("reached_goal")), "no_collision": not bool(info.get("collided")),
        "returned_to_path": float(cte[-1]) < 0.65}
    checks["boundary_margin"] = min(boundary_clearance) >= 0.05
    if static_clearance:
        checks["static_hull_margin"] = min(static_clearance) >= 0.10
    if target_clearance:
        checks["target_hull_margin"] = min(target_clearance) >= 0.10
    if name in ("empty", "null", "being_overtaken"):
        checks["holds_path"] = max(cte) < 0.65
    stand_on_metrics = stand_on_speed_metrics(records)
    if name == "being_overtaken":
        checks["stand_on_observed"] = stand_on_metrics["active_steps"] > 0
        checks["stand_on_speed"] = stand_on_metrics["max_speed_error_fraction"] is not None and stand_on_metrics["max_speed_error_fraction"] <= 0.05
        checks["stand_on_cruise_command"] = stand_on_metrics["cruise_command_fraction"] == 1.0
        checks["stand_on_no_override"] = stand_on_metrics["override_steps"] == 0
    if name == "head_on":
        checks["first_alteration_starboard"] = first_alteration == 1.0
        passing = next((r for r in records if r["y"] >= r["target_y"]), None)
        checks["starboard_pass"] = passing is not None and passing["x"] > passing["target_x"] + 0.75
    if name == "overtaking":
        passing = next((r for r in records if r["y"] >= r["target_y"]), None)
        checks["completed_port_pass"] = passing is not None and passing["x"] < passing["target_x"] - 0.75
    if name in ("crossing_port", "crossing_starboard"):
        # Slowing to pass astern is valid; a mandatory turn would reject it.
        # Evaluate actual crossing geometry independently of the reward.
        crossing = next((r for r in records if r["y"] >= 12.0), None)
        travel_sign = 1 if name == "crossing_port" else -1
        checks["passes_astern"] = crossing is not None and travel_sign * (crossing["x"] - crossing["target_x"]) < -0.75
    result.update(case=name, noise="nominal" if noise else "off", seed=seed, checks=checks, passed=all(checks.values()),
        steps=len(records), outcome="goal" if info.get("reached_goal") else info.get("collision_kind") or "timeout",
        max_cte_m=max(cte), final_cte_m=cte[-1], mean_cte_m=float(np.mean(cte)),
        min_target_centre_distance_m=min(target_distance) if env.targets else None,
        min_static_hull_clearance_m=min(static_clearance) if static_clearance else None,
        min_target_hull_clearance_m=min(target_clearance) if target_clearance else None,
        min_boundary_clearance_m=min(boundary_clearance),
        prediction_fallback_steps=sum(r["fallback"] for r in records),
        rule_relaxation_steps=sum(r["rule_relaxed"] for r in records),
        stand_on_speed=stand_on_metrics,
        first_alteration=first_alteration, supervisor_events=info.get("estop/events", 0), trajectory=records)
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(cases()), default=list(cases()))
    parser.add_argument("--noise", choices=["off", "nominal", "both"], default="both")
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "behaviors.json")
    parser.add_argument("--plot", action="store_true", help="Save a trajectory overview next to the JSON")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT.resolve()):
        parser.error("--output must remain inside the CODEX directory")
    started = time.time()
    fingerprint_paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]
    fingerprints = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in fingerprint_paths}
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for noise in ([False, True] if args.noise == "both" else [args.noise == "nominal"]):
        for name in args.cases:
            row = run_case(name, noise, args.seed)
            results.append(row)
            print(f"{'PASS' if row['passed'] else 'FAIL'} {name:20s} noise={row['noise']:7s} {row['outcome']:9s} cte_final={row['final_cte_m']:.2f} checks={row['checks']}", flush=True)
            args.output.write_text(json.dumps(dict(complete=False, controller="perception-only predictive LOS reference; no PPO weights",
                scope="narrow-channel research give-way assumptions", seconds=time.time()-started, source_sha256=fingerprints, results=results), indent=2, allow_nan=False), encoding="utf-8")
    payload = dict(controller="perception-only predictive LOS reference; no PPO weights", scope="narrow-channel research give-way assumptions", seconds=time.time()-started,
        source_sha256=fingerprints, complete=True, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    if args.plot:
        plot_results(results, args.output.with_suffix(".png"))
    return 0 if all(row["passed"] for row in results) else 1


def plot_results(results, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig, axes = plt.subplots(2, 5, figsize=(14, 9), constrained_layout=True)
    for ax, (name, scenario) in zip(axes.flat, cases().items()):
        path = np.array(scenario["path"])
        ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1, label="Reference path")
        for obstacle in scenario["obstacles"]:
            ax.add_patch(Polygon(obstacle, color="0.4"))
        for row in results:
            if row["case"] != name:
                continue
            trajectory = row["trajectory"]
            ax.plot([r["x"] for r in trajectory], [r["y"] for r in trajectory], label=row["noise"])
            if trajectory[0]["target_x"] is not None:
                ax.plot([r["target_x"] for r in trajectory], [r["target_y"] for r in trajectory], ":", color="tab:red", alpha=0.5)
        ax.set(title=name, xlim=(-0.5, 10.5), ylim=(0, 25), xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Perception-only reference controller: deterministic development scenes")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
