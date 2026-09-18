"""Additional read-only trajectory checks for a completed behaviour report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as cfg
from validate_behaviors import stand_on_speed_metrics


def summarize(report):
    if not report.get("complete", False):
        raise ValueError("the behaviour report is not complete")
    rows = []
    for episode in report["results"]:
        if episode["case"] not in ("being_overtaken", "combined"):
            continue
        trajectory = episode["trajectory"]
        row = {"case": episode["case"], "noise": episode["noise"], "seed": episode["seed"]}
        if episode["case"] == "being_overtaken":
            speeds = np.asarray([frame["speed"] for frame in trajectory])
            errors = np.abs(speeds - cfg.U_REF)
            row.update(reference_speed_mps=float(cfg.U_REF),
                mean_speed_mps=float(np.mean(speeds)), min_speed_mps=float(np.min(speeds)), max_speed_mps=float(np.max(speeds)),
                mean_absolute_speed_error_mps=float(np.mean(errors)), max_absolute_speed_error_mps=float(np.max(errors)),
                mean_absolute_speed_error_fraction=float(np.mean(errors) / cfg.U_REF),
                max_absolute_speed_error_fraction=float(np.max(errors) / cfg.U_REF),
                cruise_command_fraction=float(np.mean([abs(frame["throttle"]) < 1e-6 for frame in trajectory])),
                max_absolute_throttle_action=max(abs(frame["throttle"]) for frame in trajectory),
                active_encounter=stand_on_speed_metrics(trajectory))
        else:
            passing = next((frame for frame in trajectory if frame["y"] >= frame["target_y"]), None)
            offset = None if passing is None else passing["x"] - passing["target_x"]
            row.update(passing_time_s=None if passing is None else passing["t"],
                starboard_offset_m=offset, starboard_pass=offset is not None and offset > 0.75)
        rows.append(row)
    return {"interpretation": "Post-hoc trajectory measurements, not independent trials or COLREG certification.", "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "behaviors.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "behavior_additional_checks.json")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT.resolve()):
        parser.error("--output must remain inside CODEX")
    result = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
