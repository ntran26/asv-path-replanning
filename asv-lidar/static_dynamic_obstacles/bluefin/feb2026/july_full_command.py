"""July 2026: the yaw rate reached while the transmitted `$CMD` rudder is held near +-100 %.

    python bluefin/feb2026/july_full_command.py > bluefin/feb2026/july_full_command.txt

The comparison point for the February radio-control turns: if a full receiver deflection and a
`$CMD` of +-100 % moved the rudder equally, the July full-command holds at a similar speed should
reach yaw rates of the February size, allowing for the July holds being short (the policy
oscillated, and on most July runs the bridge ramped the command at 50 %/s).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUEFIN = os.path.dirname(HERE)
sys.path.insert(0, BLUEFIN)

from build_dataset import classify, load_all  # noqa: E402
from simulate import _smooth_derivative  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(BLUEFIN, "..", "..", "field_deployment"))


def main():
    runs = [r for r in load_all(DATA_DIR) if classify(r) == "moving"]
    print(f"{len(runs)} moving July runs (2026-07-02 and 2026-07-03)")
    allr, allc, allu, holds = [], [], [], []
    for run in runs:
        r = np.rad2deg(_smooth_derivative(run.t, np.deg2rad(run.yaw)))
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(run.x), np.diff(run.y)))])
        u = _smooth_derivative(run.t, s)
        allr.append(r); allc.append(run.rudder_at(run.t)); allu.append(u)
        big = np.abs(run.rudder_cmd) >= 95
        i = 0
        while i < len(big):
            if not big[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(big) and big[j + 1] and np.sign(run.rudder_cmd[j + 1]) == np.sign(run.rudder_cmd[i]):
                j += 1
            dur = run.tc[j] - run.tc[i] + 0.5
            if dur >= 1.5:
                # response window: from the hold's start plus the effective delay to 1.5 s after it ends
                m = (run.t >= run.tc[i] + 0.73) & (run.t <= run.tc[j] + 1.5)
                if m.any():
                    expect = -np.sign(run.rudder_cmd[i])          # +$CMD gives r < 0
                    k = np.argmax(expect * r[m])
                    holds.append((run.name, run.tc[i], dur, int(np.sign(run.rudder_cmd[i])), r[m][k], float(u[m].mean())))
            i = j + 1
    allr, allc, allu = np.concatenate(allr), np.concatenate(allc), np.concatenate(allu)
    print("pooled |r| (deg/s) p50 / p90 / p99: %.1f / %.1f / %.1f" % tuple(np.percentile(np.abs(allr), [50, 90, 99])))
    print(f"share of pose samples with |$CMD| >= 95 %: {np.mean(np.abs(allc) >= 95):.2f}")
    print(f"\nholds of |$CMD| >= 95 % for >= 1.5 s: {len(holds)}; largest yaw rate in the commanded direction "
          f"from 0.73 s after the hold began to 1.5 s after it ended")
    print("run                        t (s)  hold (s)  sign  peak r (deg/s)  mean U (m/s)")
    for h in sorted(holds, key=lambda z: -z[5]):
        print("  %-26s %5.1f  %6.1f   %+d    %+8.1f       %.2f" % h)
    fast = [h for h in holds if h[5] >= 0.9]
    if fast:
        pk = np.array([abs(h[4]) for h in fast])
        print(f"\nholds at mean U >= 0.9 m/s: {len(fast)}; peak |r| {pk.min():.1f} .. {pk.max():.1f} deg/s (median {np.median(pk):.1f})")


if __name__ == "__main__":
    main()
