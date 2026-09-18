"""C15: tracker error against ground truth, centroid vs hull-fitted measurement.

Replays fixed scenarios with a path follower (so both modes see the same own
ship motion until the encounter differs), obstacles off, supervisor off, and on
every step pairs the dynamic track nearest the true target with the truth.
Reported by true range bin:

* position error of the track against the target's centre;
* course error of the track's velocity against the target's heading;
* speed error;
* error of the perceived `dcpa_if_stopped` (the A18 stop test) against the
  same quantity computed from the truth, and how often the two disagree about
  `stop_clears`.

    python tools/diagnostics/c15_track_error.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "tiers"))

import common  # noqa: E402
import constants as cfg  # noqa: E402
import curriculum  # noqa: E402
import train_formulation as tf  # noqa: E402
from env import ASVLidarEnv  # noqa: E402

OUT = ROOT / "results" / "c15_track_error"
RANGE_BINS = [0.0, 2.0, 3.0, 4.0, 6.0, 9.0, 16.0]


def wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def dcpa_if_stopped(rng, alpha_deg, ct_deg):
    a, c = math.radians(alpha_deg), math.radians(ct_deg)
    along = math.sin(a) * math.sin(c) + math.cos(a) * math.cos(c)
    return rng if along >= 0.0 else rng * abs(math.sin(a - c))


def run(mode, scenarios, stop_fit=False):
    cfg.STOP_TEST_USES_HULL_FIT = bool(stop_fit)
    label = mode + ("+stop_fit" if stop_fit else "")
    env = ASVLidarEnv(render_mode=None)
    env.tracker.measurement = mode
    rows = []
    for i, (set_name, b) in enumerate(scenarios):
        env.forced_num_obs = 0
        env.reset(seed=700_000 + i, options={"generated": b})
        env.tracker.measurement = mode
        env.estop_enabled = False
        while True:
            _, _, term, trunc, info = env.step(common.follower_action(env))
            t = env.targets[0]
            true_rng = math.hypot(t.x - env.asv_x, t.y - env.asv_y)
            tracks = env.tracker.dynamic_tracks()
            if tracks:
                tr = min(tracks, key=lambda k: math.hypot(*(k.position - np.array([t.x, t.y]))))
                pos_err = float(math.hypot(*(tr.position - np.array([t.x, t.y]))))
                if pos_err < 2.0:
                    course_err = abs(wrap180(tr.course_deg - t.heading)) if tr.speed > 0.1 else np.nan
                    # truth and perception of the A18 stop test, in the own-ship frame
                    alpha_t = math.degrees(math.atan2(t.x - env.asv_x, t.y - env.asv_y)) - env.asv_h
                    ct_t = t.heading - env.asv_h
                    true_dis = dcpa_if_stopped(true_rng, alpha_t, ct_t)
                    ctx = env.encounter_contexts.get(tr.id)
                    perc_dis = float(ctx.dcpa_if_stopped) if ctx is not None else np.nan
                    rows.append(dict(mode=label, set=set_name, scenario=i, true_range=true_rng,
                                     pos_err=pos_err, course_err=course_err,
                                     speed_err=abs(tr.speed - t.speed),
                                     dis_err=abs(perc_dis - true_dis) if np.isfinite(perc_dis) else np.nan,
                                     clears_disagree=(bool(perc_dis >= cfg.ESTOP_CLEAR_DCPA_M)
                                                      != bool(true_dis >= cfg.ESTOP_CLEAR_DCPA_M))
                                     if np.isfinite(perc_dis) else np.nan))
            if term or trunc:
                break
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curriculum.apply_stage(tf.PROPULSION_STAGE)
    scenarios = [("head_on_width", b) for b in common.head_on_width_set(8)]
    scenarios += [(b.encounter_class, b) for b in common.development_set(
        8, classes=("crossing", "overtaking", "being_overtaken"))]
    # F66: the adopted configuration is the centroid track with the stop test reading the fit.
    d = pd.DataFrame(run("centroid", scenarios) + run("centroid", scenarios, stop_fit=True))
    d.to_csv(OUT / "frames.csv", index=False)
    d["range_bin"] = pd.cut(d.true_range, RANGE_BINS)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    agg = dict(frames=("pos_err", "size"), pos_err=("pos_err", "median"),
               course_err=("course_err", "median"), course_err_p90=("course_err", lambda s: s.quantile(0.9)),
               speed_err=("speed_err", "median"), stop_dcpa_err=("dis_err", "median"),
               stop_test_disagrees=("clears_disagree", "mean"))
    by_range = d.groupby(["range_bin", "mode"], observed=True).agg(**agg).round(3)
    by_set = d.groupby(["set", "mode"]).agg(**agg).round(3)
    overall = d.groupby("mode").agg(**agg).round(3)
    text = "\n".join(["== by true range (medians unless noted)", by_range.to_string(), "",
                      "== by scenario set", by_set.to_string(), "", "== overall", overall.to_string()])
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
