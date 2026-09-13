"""Acceptance tests — run these before any training run.

Every check here corresponds to a failure that actually occurred while
identifying this model, and each one is silent: the simulator keeps producing
numbers, the training loop keeps running, and the policy quietly learns the
artefact. They are cheap, so they belong in CI and in the pre-flight of any
long training job.

  A1  boundedness across the action envelope
  A2  steady-turn behaviour is physical (monotonic, right sign, sane radius)
  A3  timestep consistency
  A4  determinism
  A5  domain-randomisation safety — every sampled parameter set passes A1-A3
  A6  field-data regression guard — holdout metrics must stay better than v2
  A7  actuator parity with the deployment bridge

Exit code is non-zero if any test fails.
"""

from __future__ import annotations

import json
import sys
from typing import List, Tuple

import numpy as np

import dynamics as dyn
from ship_model_v3 import ShipModel, IDENTIFIED, sample_params
from vessel_sim import VesselSim, turning_circle

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
def a1_bounded(params=None, quiet=False) -> bool:
    """States stay finite and inside physical limits over the action envelope."""
    ok = True
    worst = ""
    for helm in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for rpm in (6.0, 12.0, 18.0, 24.0):
            sim = VesselSim(params=params, rpm=rpm)
            sim.reset()
            for _ in range(240):          # 120 s
                sim.step(helm)
            st = sim.truth()
            bad = (not np.isfinite([st.u, st.v, st.yaw_rate_degps, st.x, st.y]).all()
                   or st.u > 4.0 or abs(st.v) > 2.5 or abs(st.yaw_rate_degps) > 120.0)
            if bad:
                ok = False
                worst = f"helm={helm} rpm={rpm}: u={st.u:.2f} v={st.v:.2f} r={st.yaw_rate_degps:.1f}"
    if not quiet:
        check("A1 bounded over action envelope", ok, worst)
    return ok


def a2_steady_turn(params=None, quiet=False) -> bool:
    """Turn rate grows with helm, keeps one sign, and the circle is plausible."""
    rows = []
    for helm in (0.25, 0.5, 0.75, 1.0):
        sim = VesselSim(params=params)
        rows.append(turning_circle(sim, helm=helm, duration=60.0))
    r = np.array([abs(x["yaw_rate_degps"]) for x in rows])
    sgn = np.sign([x["yaw_rate_degps"] for x in rows])

    mono = bool(np.all(np.diff(r) > -0.5))
    same_sign = bool(np.all(sgn == sgn[-1]))
    full = rows[-1]
    radius_ok = 1.0 <= full["radius_L"] <= 6.0
    speed_ok = 0.4 <= full["speed_ratio"] <= 0.95
    ok = mono and same_sign and radius_ok and speed_ok
    detail = (f"|r| {np.round(r,1).tolist()} deg/s, full helm radius "
              f"{full['radius_L']:.2f} L, speed ratio {full['speed_ratio']:.2f}")
    if not quiet:
        check("A2 steady turn physical", ok, detail)
    return ok


def a3_timestep(params=None, quiet=False) -> bool:
    """Trajectory must not depend materially on the integration substep."""
    ref = None
    ok = True
    spread = 0.0
    for sub in (0.01, 0.02, 0.05):   # the supported range; 0.1 drifts ~0.5 m
        sim = VesselSim(params=params, sub_dt=sub)
        sim.reset()
        for k in range(60):
            sim.step(np.sin(k * 0.4))
        st = sim.truth()
        vec = np.array([st.x, st.y, st.heading_deg])
        if ref is None:
            ref = vec
        else:
            d = float(np.max(np.abs(vec - ref)))
            spread = max(spread, d)
    ok = spread < 0.25
    if not quiet:
        check("A3 timestep consistency", ok, f"max deviation {spread:.3f} (x/y in m, heading in deg)")
    return ok


def a4_determinism() -> bool:
    outs = []
    for _ in range(2):
        sim = VesselSim()
        sim.reset()
        for k in range(60):
            sim.step(np.sin(k * 0.3))
        st = sim.truth()
        outs.append((st.x, st.y, st.heading_deg))
    ok = outs[0] == outs[1]
    check("A4 determinism", ok, "" if ok else f"{outs[0]} vs {outs[1]}")
    return ok


def a5_randomisation(n: int = 12, scale: float = 1.0) -> bool:
    """Every domain-randomisation draw must itself be a usable simulator."""
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(n):
        p = sample_params(rng, scale=scale)
        if not (a1_bounded(p, quiet=True) and a2_steady_turn(p, quiet=True)
                and a3_timestep(p, quiet=True)):
            bad += 1
    ok = bad == 0
    check("A5 domain-randomisation safety", ok, f"{n - bad}/{n} draws usable")
    return ok


def a6_field_regression() -> bool:
    """Guard against edits that quietly undo the identification."""
    import os
    path = next((q for q in ("validation.json", "out/validation.json")
                 if os.path.exists(q)), None)
    if path is None:
        check("A6 field regression guard", False, "validation.json missing — run validate.py")
        return False
    val = json.load(open(path))
    rows = {r["horizon"]: r for r in val["v1_holdout"]}
    checks = [
        ("free-run heading", rows["free"]["v3_psi"] < rows["free"]["v2_psi"]),
        ("10 s heading", rows["10 s"]["v3_psi"] < rows["10 s"]["v2_psi"]),
        ("free-run position", rows["free"]["v3_pos"] < rows["free"]["v2_pos"]),
        ("turn-rate KS", val["v4_holdout"]["ks_v3"] < val["v4_holdout"]["ks_v2"]),
        ("path length", abs(np.mean(val["v5_holdout"]["v3_err"]))
         < abs(np.mean(val["v5_holdout"]["v2_err"]))),
    ]
    ok = all(c[1] for c in checks)
    failed = [c[0] for c in checks if not c[1]]
    check("A6 field regression guard", ok,
          "all holdout metrics still beat v2" if ok else f"regressed: {failed}")
    return ok


def a7_actuator_parity() -> bool:
    """The bridge's command rate limit must be reproduced in training."""
    sim = VesselSim(command_rate_pct_s=50.0)
    sim.reset()
    sim.step(1.0)
    first = sim._last_cmd_pct
    limited = abs(first - 0.0) <= 50.0 * sim.control_dt + 1e-6

    free = VesselSim(command_rate_pct_s=None)
    free.reset()
    free.step(1.0)
    unlimited = abs(free._last_cmd_pct) > 99.0

    # transport delay present in the model
    delay_ok = IDENTIFIED["rud_delay"] > 0.0
    # observation staleness present
    s = VesselSim(obs_delay_steps=1)
    s.reset()
    s.step(1.0)
    stale_ok = s.observe().t < s.truth().t

    ok = limited and unlimited and delay_ok and stale_ok
    check("A7 actuator parity", ok,
          f"rate limit {limited}, ablation {unlimited}, "
          f"delay {IDENTIFIED['rud_delay']:.2f}s, obs staleness {stale_ok}")
    return ok


def main() -> int:
    print("Simulator acceptance tests\n")
    a1_bounded()
    a2_steady_turn()
    a3_timestep()
    a4_determinism()
    a5_randomisation()
    a6_field_regression()
    a7_actuator_parity()

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
