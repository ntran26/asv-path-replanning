"""Tier 0 — does the reward point the right way?  (minutes, no training)

Three scripted policies over the same fixed development scenarios, obstacles
off so the encounter is the only thing being scored:

* `follower`   — keeps the path, ignores the target;
* `compliant`  — one committed 30 deg alteration in the sense the reward calls
                 compliant, held until the encounter clears;
* `wrong_way`  — the same alteration in the opposite sense.

Checks, each PASS/FAIL:

T0.1  for each give-way class, the COLREGs penalty of `compliant` is smaller
      than `wrong_way`'s, and `v_port` fires less;
T0.2  being overtaken: holding course (`compliant`, which holds for this
      class) accrues less `v_hold` than leaving the stand-on role
      (`wrong_way`, a 30 deg alteration);
T0.3  null encounters carry (almost) no COLREGs penalty for a path follower;
T0.4  head-on encounters do not latch a port sense (A19/A20), follower <= 0.10;
T0.5  every reference path is straight (F59): `r_path` is zero throughout;
T0.6  supervisor stops are rarely followed by a target collision (A18): <= 0.25
      of stopped episodes, per policy.

    python tools/tiers/tier0_scripted.py [--per-class 12] [--tag name]
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from common import RESULTS, development_set, run_pool

GIVE_WAY = ("head_on", "crossing", "overtaking")
POLICIES = ("follower", "compliant", "wrong_way")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--tag", default="current")
    args = ap.parse_args()

    out = RESULTS / f"tier0_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    scen = development_set(args.per_class, classes=GIVE_WAY + ("being_overtaken", "null"))
    jobs = [(b, 900_000 + i, p, 0, {"scenario": i}) for p in POLICIES for i, b in enumerate(scen)]
    d = pd.DataFrame(run_pool(jobs))
    d.to_csv(out / "episodes.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    summary = d.groupby(["class", "policy"]).agg(
        n=("steps", "size"), goal=("outcome", lambda s: (s == "goal").mean()),
        collision=("collided", "mean"), colregs=("colregs_integral", "mean"),
        v_port=("frames_v_port", "mean"), v_bow=("frames_v_bow", "mean"),
        v_side=("frames_v_side", "mean"), v_hold=("frames_v_hold", "mean"),
        v_r8=("frames_v_r8", "mean"), estops=("estops", "mean"),
        port_sense=("ever_port_sense_non_overtaking", "mean")).round(2)

    checks = {}
    g = d.groupby(["class", "policy"])
    for cls in GIVE_WAY:
        c, w = g.get_group((cls, "compliant")), g.get_group((cls, "wrong_way"))
        ok = (c.colregs_integral.mean() > w.colregs_integral.mean()
              and c.frames_v_port.mean() < w.frames_v_port.mean())
        checks[f"T0.1 {cls}: compliant colregs {c.colregs_integral.mean():.1f} > wrong-way "
               f"{w.colregs_integral.mean():.1f}; v_port {c.frames_v_port.mean():.1f} < "
               f"{w.frames_v_port.mean():.1f}"] = bool(ok)
    bo_f, bo_w = g.get_group(("being_overtaken", "compliant")), g.get_group(("being_overtaken", "wrong_way"))
    checks[f"T0.2 being overtaken: holding v_hold integral {bo_f.integral_v_hold.mean():.2f} "
           f"< leaving the role {bo_w.integral_v_hold.mean():.2f}"] = bool(
        bo_f.integral_v_hold.mean() < bo_w.integral_v_hold.mean())
    nf = g.get_group(("null", "follower"))
    checks[f"T0.3 null: follower COLREGs penalty {nf.colregs_integral.mean():.1f} >= -5"] = bool(
        nf.colregs_integral.mean() >= -5.0)
    hf = g.get_group(("head_on", "follower"))
    checks[f"T0.4 head-on: follower port-sense episodes {hf.ever_port_sense_non_overtaking.mean():.2f} <= 0.10"] = bool(
        hf.ever_port_sense_non_overtaking.mean() <= 0.10)
    checks[f"T0.5 straight paths: max |r_path| {d.r_path_max.max():.2e} == 0"] = bool(d.r_path_max.max() <= 1e-9)
    for pol in POLICIES:
        s = d[(d.policy == pol) & (d.estops > 0)]
        rate = float(s.estop_then_target_collision.mean()) if len(s) else 0.0
        checks[f"T0.6 {pol}: stopped episodes {len(s)}, then target collision {rate:.2f} <= 0.25"] = bool(rate <= 0.25)

    lines = [f"Tier 0 ({args.tag}) — {len(d)} episodes, {time.time() - started:.0f} s", "",
             summary.to_string(), "", "Checks:"]
    lines += [f"  [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks.items()]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    (out / "checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    print(text)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
