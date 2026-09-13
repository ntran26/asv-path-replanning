"""Vectorised 3-DOF manoeuvring dynamics for the Bluefin model vessel.

One implementation, used two ways: the fitter integrates a whole batch of runs
at once, and `ship_model_v3.ShipModel` wraps a batch of size one so the
simulator keeps the existing scalar interface.

Structural changes relative to `ship_model.py` (v2)
---------------------------------------------------
1. **The MMG hull block is removed, not re-tuned.** Evaluated at model scale
   (U = 1.2 m/s, v' = 0.15, r' = 0.35) the whole nondimensional block from
   `Blue02.m` produces ~0.09 N of sway force and ~0.07 N.m of yaw moment,
   against ~10 N and ~5 N.m from the tunable linear terms that sit beside it.
   The derivatives are three orders of magnitude too small to influence the
   solution, so v2's behaviour was set entirely by `TURN_COEF`,
   `LINEAR_*_DAMP` and the rudder scales. The sign inconsistency between
   `Blue02.m` and `ship_model.py` (the drift-dependent terms Yv, Yvv, Yvr, Nv,
   Nvv, Nvr enter with opposite sign once the global negation and the stripped
   minus signs on Yr/Nr/Yrr/Nrr are combined) is therefore real but
   numerically inert. v3 replaces the block with damping terms that carry
   physical units and can actually be identified.

2. **Explicit actuator model.** Transport delay, first-order servo lag and
   rate limit are separate, identifiable parameters. v2 wrote
   `delta_dot = clip(delta_cmd - delta, +/-rate)`, which is a first-order lag
   with an implicit 1 s time constant, not a rate limiter.

3. **Drift-induced yaw moment** (`N_uv`) is retained explicitly: it is what
   makes a turn tighten or widen and is the term the Paper 2 field results
   pointed at.

4. **Damping is speed-dependent, in MMG form.** An earlier v3 draft used
   `-N_r*r` and `-Y_v*v`, i.e. damping independent of forward speed. Every fit
   then drove `N_r` to its lower bound and routed all yaw damping through the
   drift term, because `N_uv*u*v` was the only damping that scaled with speed.
   That was the model telling us the structure was wrong: the MMG linear terms
   are `0.5*rho*L^2*d*U^2 * Nr' * (rL/U)`, which is proportional to `U*r`, not
   to `r`. The linear terms now carry that `u` factor, which removes the
   degeneracy — `N_r*u*r` and `N_uv*u*v` are the same order and physically
   distinct (one in yaw rate, one in drift).

5. **Speed loss in a turn comes from cross-flow drag** (`X_vv*v^2`,
   `X_rr*(L r)^2`), the physical mechanism, rather than from rudder drag alone. With only
   `X_delta` available, the fit drove rudder drag to absurd values (20 N,
   comparable to thrust) to reproduce the measured slowdown while turning.

6. **The Coriolis terms are complete.** v2 carried `+m22*v*r` in surge and
   `-m11*u*r` in sway but no yaw counterpart. The missing term is the Munk
   moment `(m_x - m_y)*u*v` = `-59.1*u*v`, which is destabilising — that is
   real physics, and it is why the identified yaw damping has to exceed it. That triple is only
   energy-conserving if the yaw Munk moment `(m11 - m22)*u*v` is present too,
   and with m22 = 127 kg against m11 = 68 kg the omission injects energy: held
   at full rudder the model accelerates without limit and pins every state at
   its numerical clip, at every timestep tested. The field runs never exposed
   this because they are 20 s of oscillating rudder, never a sustained turn —
   but training episodes do hold rudder, so this had to be fixed before the
   model could be used. `N_uv` now carries only the additional hydrodynamic
   drift moment on top of the Munk term.

7. **The rudder is integrated outside the RK4 stages**, exactly (exponential
   approach plus a rate clamp). Carrying a 0.01 s servo lag as an RK4 state at
   a 0.05 s step is stiff, and it showed: the rudder settled at 32.6 deg for a
   40 deg command.

8. **Rudder dynamic pressure uses the axial inflow only.** v2 scaled the normal
   force by `(u_R^2 + v_R^2)` with `v_R = v + l_R*r`, so rudder force grew with
   the square of the yaw rate — a positive feedback (r up -> v_R up -> force up
   -> r up) that diverges in a sustained turn. At full rudder the v2 form
   reaches a rudder normal force of ~200 N on a 64 kg vessel. `Blue02.m` uses
   the axial form, which is the classical one, and it is bounded. The flow
   angle still enters through the angle of attack, where it belongs.

Added masses and inertia are kept from `Blue02.m` — those are correctly scaled
and physical.

State vector (per vessel): [u, v, r, psi, delta, x, y]
    u, v   body-frame surge/sway velocity (m/s)
    r      yaw rate (rad/s)
    psi    heading (rad), measured as in log_parser: x = East-ish, y = North-ish,
           with dx = u sin(psi) + v cos(psi), dy = u cos(psi) - v sin(psi)
    delta  actual rudder angle (rad)
    x, y   position (m)

Rudder sign convention — the chain matters and is easy to get backwards:

    policy action a0  --(rudder_scale)-->  simulator rudder percent  = 100*a0
    policy action a0  --(rudder_sign=-1)-> transmitted $CMD percent  = -100*a0

so the simulator rudder that corresponds to a logged command is
`rud_sim = -cmd_transmitted`. `cmd_percent_to_angle` below takes the
*transmitted* command and already accounts for this, which is why it carries no
minus sign even though `ship_model.py` (which takes simulator percent) does.

Checked against the July logs: with this convention a positive transmitted
command produces a negative yaw rate in both the model and the measurement
(correlation +0.67 between effective rudder angle and measured yaw rate). The
opposite convention makes the model turn the wrong way, which is worse than a
model that does not turn at all — that is the failure mode that made the
baseline replay look hopeless before the sign was settled.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# Geometry (unchanged — used by the simulator for collision/LiDAR geometry)
# ---------------------------------------------------------------------------
VESSEL_LENGTH = 1.725
VESSEL_WIDTH = 0.50
HULL_MARGIN = 0.15
HULL_FORWARD_SHIFT = 0.0
LIDAR_OFFSET_M = VESSEL_LENGTH / 2.0

# ---------------------------------------------------------------------------
# Physical constants kept from Blue02.m
# ---------------------------------------------------------------------------
RHO = 1000.0
MASS = 64.55
M_X = 3.662             # surge added mass
M_Y = 62.7366           # sway added mass
IZ_JZ = 10.2347         # yaw inertia + added inertia
L = VESSEL_LENGTH
DRAFT = 0.193

M11 = MASS + M_X
M22 = MASS + M_Y
M33 = IZ_JZ

# Rudder geometry from Blue02.m
A_R = 0.0091            # rudder area (m^2)
F_ALPHA = 2.69279       # lift slope
W_R = 0.22              # wake fraction at the rudder
K_X = 0.6177            # propeller race factor
L_R = -0.77735          # rudder longitudinal position for inflow (m)

# Rudder-to-hull geometry, held FIXED at physical values so that rudder
# effectiveness is carried by a single identifiable gain (k_R). Letting the
# sway gain, the yaw arm and the lift scale all float made the fit degenerate:
# the optimiser drove the direct yaw moment to zero and produced yaw
# indirectly through sway, which is not the physics.
A_H = 0.443853          # hull sway-force amplification
X_H = -0.776            # hull force application point (m)
X_RUD = -0.8625         # rudder position, stern (m) = -L/2
RUD_ARM = abs(X_RUD + A_H * X_H)   # = 1.207 m

MAX_RUD_ANGLE_DEG = 40.0

# Numerical guards
MIN_FLOW_SPEED = 0.05
MAX_SURGE_SPEED = 5.0
MAX_SWAY_SPEED = 3.0
MAX_YAW_RATE_RAD = np.deg2rad(160.0)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Identified on 2026-07-02 (see fit_model.py); values below are the defaults
# written back by the fitter.
PARAM_NAMES = (
    "T12",          # thrust at 12 rpm, high-speed limit             [N]
    "t_boost",      # low-speed thrust boost amplitude              [-]
    "u_boost",      # low-speed thrust boost e-folding speed        [m/s]
    "X_uu",         # quadratic surge damping                       [N/(m/s)^2]
    "X_vv",         # cross-flow surge drag from sway               [N/(m/s)^2]
    "X_rr",         # cross-flow surge drag from yaw                [N/(m/s)^2]
    "X_delta",      # rudder-induced surge drag                     [-]
    "Y_v",          # sway damping, speed-dependent                 [N/(m^2/s^2)]
    "Y_vv",         # quadratic sway damping                        [N/(m/s)^2]
    "k_R",          # rudder effectiveness scale                     [-]
    "k_race",       # propeller race inflow at 12 rpm-units           [m/s]
    "N_r",          # yaw damping, speed-dependent                  [N.m/(m/s)/(rad/s)]
    "N_rr",         # quadratic yaw damping                         [N.m/(rad/s)^2]
    "N_uv",         # drift-induced yaw moment                      [N.m/(m^2/s^2)]
    "rud_rate",     # servo rate limit                              [deg/s]
    "rud_tau",      # servo first-order lag                         [s]
    "rud_delay",    # transport delay, command -> deflection        [s]
)

# v2 behaviour reproduced in the v3 structure is not meaningful (v2's hull block
# is inert), so the "prior" below is only a starting point for the optimiser.
DEFAULT_PARAMS: Dict[str, float] = {
    "T12": 11.0,
    "t_boost": 1.0,
    "u_boost": 0.5,
    "X_uu": 5.0,
    "X_vv": 30.0,
    "X_rr": 10.0,
    "X_delta": 0.2,
    "Y_v": 55.0,
    "Y_vv": 40.0,
    "k_R": 1.0,
    "k_race": 1.0,
    "N_r": 15.0,
    "N_rr": 10.0,
    "N_uv": 5.0,
    "rud_rate": 45.0,
    "rud_tau": 0.20,
    "rud_delay": 0.30,
}


def as_vector(p: Dict[str, float]) -> np.ndarray:
    return np.array([p[k] for k in PARAM_NAMES], dtype=float)


def as_dict(v: np.ndarray) -> Dict[str, float]:
    return {k: float(x) for k, x in zip(PARAM_NAMES, v)}


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
def rudder_normal_force(u: np.ndarray, v: np.ndarray, r: np.ndarray,
                        delta: np.ndarray, rpm: np.ndarray,
                        p: Dict[str, float]) -> np.ndarray:
    """Rudder normal force (N), positive for positive angle of attack.

    Inflow keeps v2's propeller-race term so the rudder retains authority near
    zero speed — the field runs all start from rest with the rudder hard over,
    so this part of the model is exercised by the data.
    """
    u_eff = np.maximum(u, 0.0)
    # Propeller race. The `rpm` field is a command unit on a 0-24 scale, not
    # shaft RPM: treating it as rev/s makes the race contribute 0.07 m/s, i.e.
    # nothing, and the rudder then loses all authority as the vessel slows in a
    # turn. `k_race` is the race inflow at the nominal 12 rpm-units, scaled as
    # sqrt of the command ratio (momentum theory: race speed ~ sqrt(thrust)).
    race = p["k_race"] * np.sqrt(np.maximum(rpm, 0.0) / 12.0)
    u_r = np.maximum(MIN_FLOW_SPEED, (1.0 - W_R) * u_eff + race)
    v_r = v + L_R * r
    # Angle of attack uses the flow direction; dynamic pressure uses the axial
    # inflow only. Scaling pressure by (u_r^2 + v_r^2) makes rudder force grow
    # with yaw rate squared and diverges in a sustained turn.
    alpha = delta - np.arctan2(v_r, u_r)
    return 0.5 * RHO * A_R * F_ALPHA * (u_r * u_r) * np.sin(alpha)


def thrust(rpm: np.ndarray, u: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """Propeller thrust (N).

    T = T12 * (rpm/12)^2 * (1 + t_boost * exp(-u / u_boost))

    The low-speed boost shape is retained from v2. `T12` is anchored at 12 rpm
    because 18 of the 20 usable runs held exactly that setting; the square-law
    RPM exponent is carried over unverified and is NOT identifiable from this
    dataset (see the basin test list, test B1).
    """
    u_eff = np.maximum(u, 0.0)
    boost = 1.0 + p["t_boost"] * np.exp(
        -u_eff / np.maximum(np.asarray(p["u_boost"], dtype=float), 1e-6))
    ratio = np.maximum(rpm, 0.0) / 12.0
    return p["T12"] * ratio * ratio * boost


def derivatives(s: np.ndarray, rpm: np.ndarray, delta_cmd: np.ndarray,
                p: Dict[str, float]) -> np.ndarray:
    """State derivative. `s` has shape (7, N); returns the same shape."""
    u, v, r, psi, delta, x, y = s

    u_eff = np.maximum(u, 0.0)

    # --- forces ------------------------------------------------------------
    f_n = rudder_normal_force(u, v, r, delta, rpm, p)
    cos_d = np.cos(delta)

    x_prop = thrust(rpm, u, p)
    lr = L * r
    # Cross-flow drag must resist forward motion whatever the sign of sway or
    # yaw rate, so it takes even powers (the MMG form). Writing it as
    # -X_vv*v*|v| makes it odd in v, which turns it into a 100 N *thrust*
    # whenever the vessel drifts to port and diverges in a sustained turn.
    x_damp = (-p["X_uu"] * u_eff * np.abs(u_eff)
              - p["X_vv"] * v * v
              - p["X_rr"] * lr * lr)
    x_rud = -p["X_delta"] * np.abs(f_n) * np.abs(np.sin(delta))
    x_total = x_prop + x_damp + x_rud + M22 * v * r

    y_damp = -p["Y_v"] * u_eff * v - p["Y_vv"] * v * np.abs(v)
    y_rud = -(1.0 + A_H) * p["k_R"] * f_n * cos_d
    y_total = y_damp + y_rud - M11 * u_eff * r

    n_damp = -p["N_r"] * u_eff * r - p["N_rr"] * r * np.abs(r)
    n_drift = -p["N_uv"] * u_eff * v
    n_munk = (M11 - M22) * u_eff * v          # completes the Coriolis triple
    n_rud = -RUD_ARM * p["k_R"] * f_n * cos_d
    n_total = n_damp + n_drift + n_munk + n_rud

    du = x_total / M11
    dv = y_total / M22
    dr = n_total / M33

    dx = u_eff * np.sin(psi) + v * np.cos(psi)
    dy = u_eff * np.cos(psi) - v * np.sin(psi)

    # The rudder is advanced outside the RK4 stages (see rk4_step), so its
    # derivative is zero here and `delta` is held frozen across the stages.
    return np.stack([du, dv, dr, r, np.zeros_like(u), dx, dy])


def advance_rudder(delta: np.ndarray, delta_cmd: np.ndarray,
                   p: Dict[str, float], dt: float) -> np.ndarray:
    """Exact first-order approach to the command, clamped by the rate limit.

    Solved analytically rather than as an RK4 state: with a servo time constant
    at or below the integration step the ODE form is stiff and settles short of
    the commanded angle.
    """
    max_rud = np.deg2rad(MAX_RUD_ANGLE_DEG)
    tau = np.maximum(np.asarray(p["rud_tau"], dtype=float), 1e-4)
    rate = np.deg2rad(np.asarray(p["rud_rate"], dtype=float))
    step = (np.clip(delta_cmd, -max_rud, max_rud) - delta) * (1.0 - np.exp(-dt / tau))
    step = np.clip(step, -rate * dt, rate * dt)
    return np.clip(delta + step, -max_rud, max_rud)


def rk4_step(s: np.ndarray, rpm: np.ndarray, delta_cmd: np.ndarray,
             p: Dict[str, float], dt: float) -> np.ndarray:
    delta_new = advance_rudder(s[4], delta_cmd, p, dt)

    # body dynamics see the rudder at its mid-step value
    s_mid = s.copy()
    s_mid[4] = 0.5 * (s[4] + delta_new)

    k1 = derivatives(s_mid, rpm, delta_cmd, p)
    k2 = derivatives(s_mid + 0.5 * dt * k1, rpm, delta_cmd, p)
    k3 = derivatives(s_mid + 0.5 * dt * k2, rpm, delta_cmd, p)
    k4 = derivatives(s_mid + dt * k3, rpm, delta_cmd, p)
    s1 = s_mid + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    s1[0] = np.clip(s1[0], 0.0, MAX_SURGE_SPEED)
    s1[1] = np.clip(s1[1], -MAX_SWAY_SPEED, MAX_SWAY_SPEED)
    s1[2] = np.clip(s1[2], -MAX_YAW_RATE_RAD, MAX_YAW_RATE_RAD)
    s1[4] = delta_new
    return s1


def cmd_percent_to_angle(cmd_percent: np.ndarray) -> np.ndarray:
    """Map a *transmitted* `$CMD` rudder percent to commanded rudder angle (rad).

    See the sign discussion in the module docstring: the transmitted command is
    the negative of the simulator's rudder percent, and `ship_model.py` negates
    its input, so the two negations cancel and this mapping is positive.
    """
    c = np.clip(cmd_percent, -100.0, 100.0)
    return c / 100.0 * np.deg2rad(MAX_RUD_ANGLE_DEG)
