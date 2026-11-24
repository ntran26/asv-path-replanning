"""Predictions of the current simulator plant for the calibration manoeuvres.

The expected results of `planning/FIELD_SESSIONS_PLAN.md`: the training plant (`src/ship.py`, the
v3 parameters identified from the July 2026 logs, no randomisation, 0.05 s steps) driven through
each manoeuvre of the calibration sessions, open loop, from steady straight motion unless stated.
These are the model's predictions that the sessions test, not measurements.

    python tools/diagnostics/manoeuvre_predictions.py > results/field_sessions/manoeuvre_predictions.txt

Conventions: helm in percent of the 40 deg maximum (simulator sign); "RPM" is the vessel's command
unit (0-24); turn geometry relative to the heading at the moment the helm is applied (advance
along it, transfer across it); the lateral room of 8 m stands for the 10 m wide navigable area
with 1 m kept clear of each wall.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "bluefin")]
import ship  # noqa: E402

DT = 0.05
LATERAL_ROOM_M = 8.0


def state(m):
    return m.state_dict()


def settle(m, rpm, seconds=60.0):
    for _ in range(int(seconds / DT)):
        m.update(rpm, 0.0, DT)


def frame(s0):
    h0 = math.radians(s0["heading_deg"])
    fwd = (math.sin(h0), math.cos(h0))
    side = (fwd[1], -fwd[0])
    return fwd, side


def along(s, s0, axis):
    return (s["x_m"] - s0["x_m"]) * axis[0] + (s["y_m"] - s0["y_m"]) * axis[1]


def wrap180(deg):
    return (deg + 540.0) % 360.0 - 180.0


def straight(rpm):
    """Acceleration from rest: steady speed, time and distance to 90 % of it."""
    m = ship.ShipModel()
    speeds, dists = [], []
    for _ in range(int(80 / DT)):
        m.update(rpm, 0.0, DT)
        s = state(m)
        speeds.append(s["u_body_mps"])
        dists.append(math.hypot(s["x_m"], s["y_m"]))
    i90 = next(i for i, u in enumerate(speeds) if u >= 0.9 * speeds[-1])
    return speeds[-1], (i90 + 1) * DT, dists[i90]


def coast(rpm):
    """Coast-down from steady speed: time to half speed; time and distance to 0.1 m/s."""
    m = ship.ShipModel()
    settle(m, rpm)
    s0 = state(m)
    t, t_half = 0.0, None
    while t < 120:
        m.update(0.0, 0.0, DT)
        t += DT
        s = state(m)
        if t_half is None and s["u_body_mps"] <= 0.5 * s0["u_body_mps"]:
            t_half = t
        if s["u_body_mps"] <= 0.1:
            return s0["u_body_mps"], t_half, t, math.hypot(s["x_m"] - s0["x_m"], s["y_m"] - s0["y_m"])
    return s0["u_body_mps"], t_half, None, None


def turn(rpm, helm):
    """Steady turn (over the second full turn) and the first turn's geometry."""
    m = ship.ShipModel()
    settle(m, rpm)
    s0 = state(m)
    fwd, side = frame(s0)
    t, dpsi, prev = 0.0, 0.0, s0["heading_deg"]
    out = {"U0": s0["speed_mps"], "adv": None, "trans": None, "tact": None, "t360": None, "psi_room": None,
           "fwd_extent": 0.0, "side_extent": 0.0}
    rates, speeds = [], []
    while t < 400 and dpsi < 720:
        m.update(rpm, helm, DT)
        t += DT
        s = state(m)
        dpsi += abs(wrap180(s["heading_deg"] - prev))
        prev = s["heading_deg"]
        ax, lat = along(s, s0, fwd), abs(along(s, s0, side))
        if dpsi < 360:
            out["fwd_extent"] = max(out["fwd_extent"], ax)
            out["side_extent"] = max(out["side_extent"], lat)
        if out["adv"] is None and dpsi >= 90:
            out["adv"], out["trans"] = ax, lat
        if out["tact"] is None and dpsi >= 180:
            out["tact"] = lat
        if out["psi_room"] is None and lat >= LATERAL_ROOM_M:
            out["psi_room"] = dpsi
        if out["t360"] is None and dpsi >= 360:
            out["t360"] = t
        if 360 <= dpsi < 720:
            rates.append(abs(s["yaw_rate_degps"]))
            speeds.append(s["speed_mps"])
    out["rate"] = sum(rates) / len(rates)
    out["U"] = sum(speeds) / len(speeds)
    out["R"] = out["U"] / math.radians(out["rate"])
    out["ratio"] = out["U"] / out["U0"]
    return out


def zigzag(rpm, delta_deg, psi_deg):
    """Z(delta/psi): first and second overshoot, period, lateral excursion, run length to the second overshoot."""
    helm = delta_deg / 40.0 * 100.0
    probe = ship.ShipModel()
    settle(probe, rpm)
    h = state(probe)["heading_deg"]
    for _ in range(60):
        probe.update(rpm, helm, DT)
    sgn = 1 if wrap180(state(probe)["heading_deg"] - h) > 0 else -1

    m = ship.ShipModel()
    settle(m, rpm)
    s0 = state(m)
    fwd, side = frame(s0)
    phase, cmd, psi, prev, t = +1, helm, 0.0, s0["heading_deg"], 0.0
    hist, switches = [], []
    while len(switches) < 5 and t < 400:
        m.update(rpm, cmd, DT)
        t += DT
        s = state(m)
        psi += wrap180(s["heading_deg"] - prev) * sgn
        prev = s["heading_deg"]
        hist.append((t, psi, along(s, s0, side), along(s, s0, fwd)))
        if phase > 0 and psi >= psi_deg:
            switches.append(t)
            phase, cmd = -1, -helm
        elif phase < 0 and psi <= -psi_deg:
            switches.append(t)
            phase, cmd = +1, helm
    overs, t_second = [], None
    for i in range(2):
        seg = [x for x in hist if switches[i] <= x[0] <= switches[i + 1]]
        peak = max(seg, key=lambda x: x[1]) if i % 2 == 0 else min(seg, key=lambda x: x[1])
        overs.append(abs(peak[1]) - psi_deg)
        if i == 1:
            t_second = peak
    period = switches[3] - switches[1]
    lateral = max(abs(x[2]) for x in hist if x[0] <= t_second[0])
    return overs, period, lateral, t_second[3]


FULL_ASTERN_RPM = -24.0     # S2 = -100, the emergency stop's command (src/emergency_stop.py)


def crash(rpm, helm=0.0, efficiency=0.5, onset_s=0.0):
    """Full astern (S2 = -100, rudder centred) from steady straight motion or a steady turn, to rest."""
    m = ship.ShipModel(reverse_efficiency=efficiency)
    settle(m, rpm)
    if helm:
        for _ in range(int(40 / DT)):
            m.update(rpm, helm, DT)
    s0 = state(m)
    fwd, side = frame(s0)
    t, prev, dpsi, s = 0.0, s0["heading_deg"], 0.0, s0
    t_surge0 = None
    while t < 90:
        m.update(0.0 if t < onset_s else FULL_ASTERN_RPM, 0.0, DT)
        t += DT
        s = state(m)
        dpsi += wrap180(s["heading_deg"] - prev)
        prev = s["heading_deg"]
        if t_surge0 is None and s["u_body_mps"] <= 1e-3:
            t_surge0 = t
            reach_at_stop = along(s, s0, fwd)
        if t_surge0 is not None and abs(s["v_body_mps"]) < 0.02 and abs(s["yaw_rate_degps"]) < 0.5:
            break
    return {"u0": s0["speed_mps"], "t_surge0": t_surge0, "reach": reach_at_stop, "t_rest": t,
            "lateral": abs(along(s, s0, side)), "dpsi": abs(dpsi)}


def main():
    print("Plant: src/ship.py with the identified v3 parameters (no randomisation), dt = %.2f s" % DT)
    p = ship.IDENTIFIED
    print("Actuator in the model: transport delay %.2f s, first-order lag %.2f s, rate limit %.0f deg/s "
          "(effectively none)" % (p["rud_delay"], p["rud_tau"], p["rud_rate"]))
    print()
    print("STRAIGHT, from rest: steady speed; time and distance to 90 % of it")
    for rpm in (6, 9, 12, 15, 18, 21, 24):
        u, t90, d90 = straight(rpm)
        print("  RPM %2d: %.2f m/s; %4.1f s, %.1f m" % (rpm, u, t90, d90))
    print()
    print("COAST-DOWN from steady speed: time to half speed; time and distance to 0.1 m/s")
    for rpm in (9, 12, 15, 21):
        u0, th, t01, d01 = coast(rpm)
        print("  from RPM %2d (%.2f m/s): half speed %.1f s; 0.1 m/s after %.1f s and %.1f m" % (rpm, u0, th, t01, d01))
    print()
    print("TURNS: steady rate, radius, speed in the turn (ratio to straight), advance and transfer at 90 deg,")
    print("       tactical diameter, time for 360 deg, footprint of the first full turn (along x across the")
    print("       initial heading), heading change before the transfer reaches %.0f m" % LATERAL_ROOM_M)
    for rpm, helms in ((6, (50, 100)), (12, (25, 50, 75, 100)), (18, (50, 100))):
        for helm in helms:
            r = turn(rpm, helm)
            print("  RPM %2d helm %3d %%: %4.1f deg/s, R %4.1f m, %.2f m/s (%.2f), adv %4.1f, trans %4.1f, TD %4.1f m, "
                  "360 deg in %3.0f s, footprint %4.1f x %4.1f m, %3.0f deg in %.0f m"
                  % (rpm, helm, r["rate"], r["R"], r["U"], r["ratio"], r["adv"], r["trans"], r["tact"], r["t360"],
                     r["fwd_extent"], r["side_extent"], r["psi_room"] or 720, LATERAL_ROOM_M))
        r = turn(rpm, -100)
        print("  RPM %2d helm -100 %%: %4.1f deg/s, R %4.1f m (the model is symmetric)" % (rpm, r["rate"], r["R"]))
    print()
    print("ZIG-ZAGS: first and second overshoot, period, largest lateral excursion and distance run")
    print("          up to the second overshoot (from the first helm order)")
    for rpm, d, ps in ((12, 20, 20), (12, 10, 10), (6, 20, 20)):
        overs, period, lat, run = zigzag(rpm, d, ps)
        print("  RPM %2d %d/%d: overshoots %.1f and %.1f deg, period %.1f s, lateral %.1f m, %.1f m run"
              % (rpm, d, ps, overs[0], overs[1], period, lat, run))
    print()
    print("CRASH STOPS, full astern (S2 = -100, -24 rpm units) with the rudder centred; strong reverse model (efficiency 0.5, no")
    print("onset delay) and weak (0.25, 0.75 s onset), the safety layer's bracket: time and head reach to")
    print("zero surge; then time to rest (sway < 0.02 m/s, yaw < 0.5 deg/s), lateral drift, heading change")
    for rpm, helm in ((9, 0), (12, 0), (15, 0), (12, 50), (12, 100)):
        for name, eff, on in (("strong", 0.5, 0.0), ("weak", 0.25, 0.75)):
            c = crash(rpm, helm, eff, on)
            where = "straight" if not helm else "turn, helm %d %%" % helm
            print("  RPM %2d %-16s %-6s from %.2f m/s: surge 0 after %4.1f s, head reach %.2f m; rest after %4.1f s, "
                  "lateral %.2f m, heading change %3.0f deg"
                  % (rpm, where, name, c["u0"], c["t_surge0"], c["reach"], c["t_rest"], c["lateral"], c["dpsi"]))


if __name__ == "__main__":
    main()
