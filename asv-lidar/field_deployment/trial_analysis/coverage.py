"""Coverage of the training randomisation (FIELD_CALIBRATION_PLAN.md section 6, output 5;
claim C-19; table F4).

The identified parameter set (and, if given, its bootstrap or day-to-day sets) is turned into
manoeuvring quantities and placed in the distribution of the same quantities over the plant
randomisation the policy trained on (`ship.sample_params` at `VESSEL_RANDOMISATION_SCALE`,
blends of the 28 July bootstrap vectors):

* turn rate and turning radius in a steady turn (`bluefin/simulate.steady_turn`) at the turning
  circles of S1-D: rpm 12, helm 25, 50, 75 and 100 %, and the cruise setting rpm 6, helm 50 and
  100 %, both signs;
* the thrust curve: steady straight speed at rpm 6, 9, 12, 18 and 24;
* actuator lag: transport delay, first-order lag and their sum (the 63 % time of a step
  without the rate limit), and the servo rate.

A quantity is inside when the identified value lies within the central `--envelope` percent of
the training distribution (default 95: the 2.5th to 97.5th percentiles).  C-19 is met when every
quantity of the four registered groups (turn rate, turning radius, thrust curve, actuator lag)
is inside; the reverse response carries no prediction and is not part of it.

    python coverage.py --identified params.json [--intervals sets.json] [--draws 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import trial_io as tio

import dynamics as dyn  # noqa: E402  (bluefin)
import ship as shipmod  # noqa: E402
from simulate import steady_turn  # noqa: E402

# (rpm, helms) of the S1-D turning circles; rpm 6 is the deployed cruise setting
TURN_POINTS = ((12.0, (-100.0, -75.0, -50.0, -25.0, 25.0, 50.0, 75.0, 100.0)),
               (6.0, (-100.0, -50.0, 50.0, 100.0)))
THRUST_RPMS = (6.0, 9.0, 12.0, 18.0, 24.0)
GROUPS = ("turn rate", "turning radius", "thrust curve", "actuator lag")


def quantities(theta: np.ndarray) -> Dict[str, np.ndarray]:
    """Manoeuvring quantities for each parameter vector (rows of `theta`)."""
    theta = np.atleast_2d(theta)
    out: Dict[str, np.ndarray] = {}
    for rpm, helms in TURN_POINTS:
        u, r = steady_turn(theta, rudders=helms, rpm=rpm)
        for j, helm in enumerate(helms):
            rate = np.abs(r[:, j])
            out[f"turn rate | helm {helm:+.0f} %, rpm {rpm:.0f} (deg/s)"] = rate
            out[f"turning radius | helm {helm:+.0f} %, rpm {rpm:.0f} (m)"] = u[:, j] / np.maximum(np.radians(rate), 1e-9)
    for rpm in THRUST_RPMS:
        u0, _ = steady_turn(theta, rudders=(0.0,), rpm=rpm)
        out[f"thrust curve | steady speed at rpm {rpm:.0f} (m/s)"] = u0[:, 0]
    names = list(dyn.PARAM_NAMES)
    delay, tau = theta[:, names.index("rud_delay")], theta[:, names.index("rud_tau")]
    out["actuator lag | transport delay (s)"] = delay
    out["actuator lag | first-order lag (s)"] = tau
    out["actuator lag | delay + lag, 63 % step time (s)"] = delay + tau
    out["servo | rate limit (deg/s)"] = theta[:, names.index("rud_rate")]
    return out


def training_draws(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = float(tio.cfg.VESSEL_RANDOMISATION_SCALE)
    return np.array([dyn.as_vector(shipmod.sample_params(rng, scale=scale)) for _ in range(n)])


def read_params(path: Path) -> List[Dict[str, float]]:
    """One parameter set (`{"params": {...}}` or a plain dict) or a list of them."""
    data = json.loads(Path(path).read_text())
    items = data if isinstance(data, list) else [data]
    return [{k: float(v) for k, v in item.get("params", item).items()} for item in items]


def coverage(identified: Dict[str, float], intervals: Optional[List[Dict[str, float]]] = None,
             *, draws: int = 2000, envelope: float = 95.0, seed: int = 0) -> List[Dict]:
    lo_q, hi_q = 50.0 - envelope / 2.0, 50.0 + envelope / 2.0
    train = quantities(training_draws(draws, seed))
    point = quantities(dyn.as_vector(identified)[None, :])
    spread = quantities(np.array([dyn.as_vector(p) for p in intervals])) if intervals else None
    rows = []
    for name, values in train.items():
        value = float(point[name][0])
        lo, hi = np.percentile(values, [lo_q, hi_q])
        group = name.split(" | ")[0]
        rows.append({"group": group, "quantity": name, "identified": value,
                     "interval_low": float(np.min(spread[name])) if spread else float("nan"),
                     "interval_high": float(np.max(spread[name])) if spread else float("nan"),
                     "training_low": float(lo), "training_median": float(np.median(values)),
                     "training_high": float(hi),
                     "training_percentile": float(100.0 * np.mean(values < value)),
                     "inside": bool(lo <= value <= hi), "registered": group in GROUPS})
    return rows


def c19_verdict(rows: List[Dict]) -> Dict:
    per = {g: all(r["inside"] for r in rows if r["group"] == g) for g in GROUPS}
    return {"per_group": per, "met": all(per.values()),
            "rule": "every quantity of turn rate, turning radius, thrust curve and actuator lag inside the envelope"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--identified", type=Path, required=True, help="the session-1 parameter set")
    ap.add_argument("--intervals", type=Path, help="bootstrap or day-to-day parameter sets (a JSON list)")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--envelope", type=float, default=95.0)
    ap.add_argument("--out", type=Path, default=Path("coverage"))
    args = ap.parse_args()
    identified = read_params(args.identified)[0]
    intervals = read_params(args.intervals) if args.intervals else None
    rows = coverage(identified, intervals, draws=args.draws, envelope=args.envelope)
    verdict = c19_verdict(rows)
    import pandas as pd
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "f4_coverage.csv", index=False)
    (args.out / "c19.json").write_text(json.dumps({"envelope_pct": args.envelope, "draws": args.draws, **verdict}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
