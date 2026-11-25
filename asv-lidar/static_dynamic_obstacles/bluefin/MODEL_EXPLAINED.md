# The Bluefin vessel model (v3), explained

Date: 2026-11-24. Status: description of the frozen training plant. Nothing here changes the model.

This note describes the vessel model that the Paper 3 training simulator uses, for a reader who
knows ship manoeuvring but not this code. It covers the equations in Fossen's matrix form and in
the MMG modular form, each force term, the actuator and the integration, the identified parameters
and their uncertainty, the training randomisation and the reverse braking, and what is measured,
assumed or inherited. Every number is taken from the code, from `REPORT.md` and `REVIEW.md`, or
from `explain/explain_numbers.py`, which derives the remaining ones from the frozen code
(`explain/explain_numbers.txt` holds its output).

| File | Role |
| --- | --- |
| `bluefin/dynamics.py` | the equations of motion, the actuator and the RK4 step (shared by the fitter and the simulator) |
| `bluefin/ship_model_v3.py` | the identified parameters, their bootstrap intervals and vectors, `sample_params` |
| `src/ship.py` | the training plant: imports the two above, adds sub-stepping and reverse braking |
| `src/env.py` | calls `ShipModel.update` at the physics step and draws a new parameter set per episode |
| `bluefin/REPORT.md`, `REVIEW.md`, `refit_v4/` | the identification record, its independent review, and two refits not adopted |

## 1. In conversation

**One sentence.** A three-degree-of-freedom manoeuvring model of the 1.7 m Bluefin, in Fossen's
matrix form with MMG-style speed-scaled hull damping and a lifting-surface rudder, whose fifteen
free parameters were fitted to 14 policy-driven runs of the July 2026 field session and checked on
a second day, with a transport delay and servo lag in front of the rudder.

**Three sentences.** The plant is a 3-DOF surge-sway-yaw model, M nu_dot + C(nu) nu + D(nu) nu =
tau, with the added masses and rudder geometry of an earlier MMG model of the same hull, damping
that scales with forward speed as in the MMG form, and a rudder force computed from its angle of
attack in the propeller-and-hull inflow. Its hull, rudder and actuator parameters were identified
from 14 runs of oscillating rudder at one propeller setting (2 July 2026), validated on 6 runs of
another day (3 July 2026, heading error 7.9 deg at 3 s against 9.4 deg for the previous model),
and the domain randomisation draws whole bootstrap parameter vectors so that correlated
parameters stay consistent. The sustained-turn behaviour, the thrust at other propeller settings
and the reverse thrust are not measured but set by priors, and they are what the 2026 calibration
sessions measure.

## 2. Vessel, frames and states

**Vessel** (`dynamics.py`): length L = 1.725 m, beam B = 0.50 m, draught d = 0.193 m, mass
m = 64.55 kg, one propeller, one rudder of area A_R = 0.0091 m^2 (2.73 % of L d) deflecting up to
40 deg. The LiDAR sits at x = +L/2 = +0.8625 m in the simulator's geometry.

**Frames.** The earth frame is the pose convention of `log_parser.py`: x and y in the basin plane,
heading psi measured from +y towards +x. The body frame has u forward, v to starboard and r
positive when turning to starboard. This is Fossen's body frame {b} (x forward, y starboard,
z down) and his NED frame with north and east relabelled: Fossen's north is the model's y, his east
the model's x. The kinematics are

    x_dot   = u sin(psi) + v cos(psi)
    y_dot   = u cos(psi) - v sin(psi)
    psi_dot = r

**State** (`dynamics.py`, per vessel): nu = [u, v, r], eta = [x, y, psi], and the actual rudder
angle delta; `ShipModel` adds the command delay line.

**Inputs.**
- Rudder: the simulator's `rudder_percent` in [-100, 100]; the commanded angle is
  delta_cmd = -(rudder_percent / 100) x 40 deg. The minus sign reproduces the field chain:
  policy action a0 gives simulator percent +100 a0 and a transmitted `$CMD` rudder of -100 a0
  (`REPORT.md` 1.2), and `cmd_percent_to_angle` maps the transmitted value to the angle.
- Propulsion: a command in "rpm-units" on the bridge's 0-24 scale, not shaft speed. July ran at 12;
  the policy of baseline-v3 commands 4.5-7.5 around a cruise of 6 (`constants.py`, stage 1).

## 3. Equations of motion in Fossen's matrix form

    M nu_dot + C(nu) nu + D(nu) nu = tau_R + tau_P

**Mass matrix**, rigid body plus added mass, diagonal (no x_G, no sway-yaw added-mass coupling):

    M = diag(m - X_udot, m - Y_vdot, I_z - N_rdot) = diag(m11, m22, m33)
      = diag(64.55 + 3.662, 64.55 + 62.7366, 10.2347) = diag(68.212 kg, 127.287 kg, 10.2347 kg m^2)

The added masses and the yaw inertia (I_z + J_z = 9.6038 + 0.6309) are those of `Blue02.m`.

**Coriolis and centripetal matrix**, rigid body and added mass together, skew-symmetric:

    C(nu) = [ 0         0          -m22 v ]        C(nu) nu = [ -m22 v r        ]
            [ 0         0           m11 u ]                   [  m11 u r        ]
            [ m22 v    -m11 u       0     ]                   [ (m22 - m11) u v ]

Moved to the right-hand side these are the three terms in `derivatives`: `+M22*v*r` in surge,
`-M11*u*r` in sway, and the Munk moment `(M11 - M22)*u*v` = -59.075 u v in yaw. The Munk moment is
destabilising. `REPORT.md` section 2 (defect 4) found it missing in the previous model; without it
the triple is not energy-conserving and the model accelerated without limit at held rudder.

**Damping**, nonlinear, written so that D(nu) nu reproduces the code:

    D(nu) = [ X_uu |u|   X_vv v           X_rr L^2 r      ]     D(nu) nu = [ X_uu u|u| + X_vv v^2 + X_rr (L r)^2 ]
            [ 0          Y_v u + Y_vv |v|  0               ]                [ Y_v u v + Y_vv v|v|                ]
            [ 0          N_uv u           N_r u + N_rr |r| ]                [ N_uv u v + N_r u r + N_rr r|r|     ]

D is not a dissipation matrix in Fossen's sense: the N_uv u v term is the circulatory (lift) yaw
moment of the hull in drift, which can feed energy into yaw. With N_uv = -83.97 it acts on the
force side as +83.97 u v; together with the Munk moment the net drift moment is
(-N_uv + m11 - m22) u v = +24.89 u v, so in a turn, where the drift v has the opposite sign to r,
it opposes the turn.

**Rudder forces** tau_R, from the rudder normal force F_N:

    F_N   = 0.5 rho A_R f_alpha u_R^2 sin(alpha_R)
    u_R   = max(0.05, (1 - w_R) max(u, 0) + k_race sqrt(n / 12))
    alpha_R = delta - atan2(v + l_R r, u_R)

    tau_R = [ -X_delta |F_N| |sin delta|              ]
            [ -(1 + a_H) k_R F_N cos delta           ]
            [ -|x_R + a_H x_H| k_R F_N cos delta     ]

with rho = 1000, f_alpha = 2.69279, w_R = 0.22, l_R = -0.77735 m, a_H = 0.443853,
x_H = -0.776 m and x_R = -0.8625 m, so |x_R + a_H x_H| = 1.2069 m. k_R is the single identified
rudder effectiveness; the geometry is held fixed so that rudder authority has one identifiable
gain (`dynamics.py`, "Rudder-to-hull geometry").

**Propeller thrust** tau_P, surge only:

    T(n) = T12 (n / 12)^2 (1 + t_boost exp(-u / u_boost)),   t_boost = 0, so T(n) = 16.0623 (n / 12)^2 N

**The resulting scalar equations** (`dynamics.derivatives`, with u read as max(u, 0) in every
speed-scaled term):

    m11 u_dot = T(n) - X_uu u|u| - X_vv v^2 - X_rr (L r)^2 - X_delta |F_N| |sin delta| + m22 v r
    m22 v_dot = -Y_v u v - Y_vv v|v| - (1 + a_H) k_R F_N cos delta - m11 u r
    m33 r_dot = -N_r u r - N_rr r|r| - N_uv u v + (m11 - m22) u v - |x_R + a_H x_H| k_R F_N cos delta

## 4. The same model in the MMG modular view

The MMG standard method (Yasukawa and Yoshimura 2015, J. Mar. Sci. Technol. 20:37-52) writes the
equations about midship with separate hull (H), propeller (P) and rudder (R) forces:

    (m + m_x) u_dot - (m + m_y) v r - x_G m r^2         = X_H + X_P + X_R
    (m + m_y) v_dot + (m + m_x) u r + x_G m r_dot       = Y_H + Y_R
    (I_zG + x_G^2 m + J_z) r_dot + x_G m (v_dot + u r)  = N_H + N_R

With x_G = 0 this is the v3 system, and the module boundaries map as follows.

| Module | MMG standard | v3 |
| --- | --- | --- |
| Hull, surge | 0.5 rho L d U^2 (-R'_0 u'^2 + X'_vv v'^2 + X'_vr v' r' + X'_rr r'^2 + X'_vvvv v'^4) | -X_uu u\|u\| - X_vv v^2 - X_rr (L r)^2: resistance and cross-flow drag, no v r term |
| Hull, sway | 0.5 rho L d U^2 (Y'_v v' + Y'_R r' + cubic terms) | -Y_v u v - Y_vv v\|v\|: a linear term in U v and a cross-flow drag; no hull sway force from yaw rate |
| Hull, yaw | 0.5 rho L^2 d U^2 (N'_v v' + N'_R r' + cubic terms); the Munk moment is inside N'_v | -N_r u r - N_rr r\|r\| - N_uv u v, with the Munk moment written separately in C(nu) |
| Propeller | (1 - t_P) rho n_P^2 D_P^4 K_T(J_P) | T12 (n/12)^2: no thrust deduction, no advance-ratio dependence, n in command units |
| Rudder, force | F_N = 0.5 rho A_R U_R^2 f_alpha sin alpha_R, with U_R^2 = u_R^2 + v_R^2 | u_R^2 only (axial inflow); `REPORT.md` defect 3: the total-velocity form fed back yaw rate into rudder force and diverged |
| Rudder, inflow | u_R from the propeller slipstream (K_T, J_P, eta, kappa, epsilon) | (1 - w_R) u plus a lumped race k_race sqrt(n/12) (`REPORT.md` defect 5) |
| Rudder, flow angle | alpha_R = delta - atan(v_R / u_R), v_R = U gamma_R beta_R' | alpha_R = delta - atan2(v + l_R r, u_R): no flow-straightening factor gamma_R |
| Rudder, forces | X_R = -(1 - t_R) F_N sin delta; Y_R = -(1 + a_H) F_N cos delta; N_R = -(x_R + a_H x_H) F_N cos delta | X_R = -X_delta \|F_N\| \|sin delta\|; Y_R as MMG times k_R; N_R = -\|x_R + a_H x_H\| k_R F_N cos delta (sign, below) |

The linear hull terms carry the MMG speed scaling: 0.5 rho L d U^2 Y'_v (v / U) is proportional to
U v, and 0.5 rho L^2 d U^2 N'_R (r L / U) to U r. `REPORT.md` defect 1 records that the
speed-independent forms (-Y_v v, -N_r r) made every fit collapse; with the u factor the yaw damping
N_r u r and the drift moment N_uv u v are of the same order and separable.

**Equivalent nondimensional values** (`explain/explain_numbers.txt`), dividing by 0.5 rho L d = 166.46 kg/m,
0.5 rho L^2 d = 287.15 kg and 0.5 rho L^3 d = 495.33 kg m:

| v3 term | Equivalent | Reading |
| --- | --- | --- |
| Y_v = 0.924 | Y'_v = -0.0055 | linear sway damping almost absent |
| Y_vv = 220.98 | cross-flow drag coefficient C_D = 1.33 | the sway damping is carried by cross-flow drag, of a plausible size |
| N_r = 1.04 | N'_r = -0.0021 | linear yaw damping almost absent |
| -N_uv + m11 - m22 = +24.89 | N'_v = +0.087 (net, Munk included) | the drift moment carries the yaw balance in a turn |
| X_uu = 12.89 | C_T = 0.034 on the wetted surface 0.7614 m^2 of `Blue02.m` | total resistance coefficient at model scale |

The small linear coefficients are a property of the fit, not hull derivatives: `REPORT.md` section 8
warns that k_R, N_r and N_uv trade off and must not be quoted as physical measurements.

**The sign of the rudder yaw moment.** In MMG, x_R < 0 (the rudder is aft), x_R + a_H x_H is
negative, and N_R = -(x_R + a_H x_H) F_N cos delta has the opposite sign to Y_R: a rudder pushing
the stern to port turns the bow to starboard. `dynamics.py` takes the absolute value of the lever
arm, so Y_R and N_R have the same sign, as if the rudder force acted at x = +0.836 m, forward of
midship. The same sign is in `Blue02.m` (line 117,
`NR = -(0.646875 + 0.443853*0.7569)*FN*cos(del)` beside line 116, `YR = -(1+0.443853)*FN*cos(del)`).
The yaw direction is right, because the command sign was fitted to the logs; what is inverted is
the direction of the rudder's sway force relative to the turn. In the steady full-helm turn at
RPM 12 the rudder pushes the hull towards the centre of the turn with +4.72 N against the
centripetal requirement of 10.92 N, so the hull's sway damping has to supply 6.20 N instead of
the 15.65 N a stern rudder would leave to it, and the drift at midship settles at 10.2 deg. The fit
absorbed this through k_R, N_uv and the sway damping; it is not a reason to change the frozen
plant, but a structural point for the calibration fit (`feb2026/REPORT_FEB2026.md`).

## 5. Every force term

| Term | Expression | Value (v3) | Physical meaning | Identified? |
| --- | --- | --- | --- | --- |
| Thrust | T12 (n/12)^2 | T12 = 16.06 N | propeller thrust at the July setting of 12 | yes, at n = 12 only; the square law is assumed |
| Low-speed thrust boost | 1 + t_boost exp(-u/u_boost) | t_boost = 0 (u_boost = 0.5 inactive) | bollard-pull increase at low speed, kept from v2 | fixed at zero (the fit drove it there) |
| Resistance | -X_uu u\|u\| | 12.89 N s^2/m^2 | hull resistance, quadratic | yes; steady speed 1.116 m/s at n = 12 against 1.14 m/s mined from the logs (`REVIEW.md`) |
| Cross-flow surge drag, sway | -X_vv v^2 | 50.64 | speed loss while drifting | yes |
| Cross-flow surge drag, yaw | -X_rr (L r)^2 | 0.058 | speed loss while turning | no (interval 0-13.4) |
| Rudder drag | -X_delta \|F_N\| \|sin delta\| | 0.116 | drag of the deflected rudder | no (interval 0-0.55) |
| Coriolis, surge | +m22 v r | m22 = 127.29 kg | centripetal and added-mass coupling: the main speed loss in a turn (3.68 N of the 16.06 N at RPM 12, full helm) | from the masses |
| Linear sway damping | -Y_v u v | 0.924 | lift of the hull in drift | no (interval 0-5.96) |
| Cross-flow sway drag | -Y_vv v\|v\| | 220.98 | separated cross-flow drag, C_D = 1.33 | yes (206-236) |
| Coriolis, sway | -m11 u r | m11 = 68.21 kg | centripetal term | from the masses |
| Rudder sway force | -(1 + a_H) k_R F_N cos delta | k_R = 2.18 | rudder lift plus the hull force it induces (a_H) | k_R yes (1.14-2.55), geometry fixed |
| Linear yaw damping | -N_r u r | 1.04 | hull yaw damping, MMG-scaled | no (0.014-8.69) |
| Quadratic yaw damping | -N_rr r\|r\| | 0.017 | cross-flow yaw drag; the only yaw damping left at zero surge | no (0-5.85); `fit_final.py` meant to fix it at zero |
| Drift moment | -N_uv u v | N_uv = -83.97 | hull lift moment in drift, on top of the Munk moment | yes (-86.8 to -55.3), 1.8 % from its bound of -90 (`REVIEW.md` section 4) |
| Munk moment | (m11 - m22) u v | -59.08 u v | ideal-fluid destabilising moment | from the masses |
| Rudder yaw moment | -\|x_R + a_H x_H\| k_R F_N cos delta | arm 1.207 m | rudder and induced hull moment | through k_R |
| Rudder inflow race | k_race sqrt(n/12) | 0.0135 m/s | propeller slipstream over the rudder | no (0-0.21) |

**The balance in a steady full-helm turn** at RPM 12 (`explain/explain_numbers.txt`): u = 0.917 m/s
(0.82 of straight running), v = -0.166 m/s, r = 10.0 deg/s, radius 5.34 m. Yaw: rudder +3.95 N m
and Munk +8.97 N m against the drift moment -12.75 N m; the yaw-rate damping N_r contributes
-0.17 N m. The turn is held by the drift moment, not by yaw-rate damping. Sway: rudder +4.72 N and
cross-flow drag +6.06 N against the centripetal -10.92 N. Surge: thrust 16.06 N against resistance
10.84 N, the Coriolis term 3.68 N and cross-flow drag 1.39 N. The rudder's angle of attack falls
from the 40 deg deflection to 17.5 deg once the turn is established, because the flow at the
rudder arrives at 22.5 deg from the side.

## 6. Actuator and integration

**Rudder path** (`ship.py` and `dynamics.py`), in order:
1. Command: delta_cmd = -(rudder_percent / 100) x 40 deg, clipped to +-40 deg.
2. Transport delay: a first-in first-out line of round(rud_delay / h) sub-steps. At the 0.05 s
   sub-step it holds 15 entries, so the simulator's delay is 0.75 s for the identified 0.730 s.
   It is an effective delay: transport latency, servo response and estimator lag lumped together
   (`REPORT.md` section 8).
3. Rate limit: rud_rate = 2985 deg/s, in effect none. The identification could not see a servo
   rate limit faster than the 20 deg/s the bridge's limiter imposed on most July runs.
4. Servo lag: the exact first-order step
   delta_{k+1} = delta_k + clip((delta_cmd - delta_k)(1 - exp(-h / tau)), +-rate h), tau = 0.895 s.
   It is advanced outside the RK4 stages and the stages see its mid-step value; carried as an RK4
   state the lag was stiff and settled at 32.6 deg for a 40 deg command (`REPORT.md` section 2).

**Propulsion path**: none. Thrust follows the command instantly; no delay, lag or acceleration
limit of the motor is modelled.

**Integration.** Classical RK4 on (u, v, r, psi, x, y) with delta frozen across the stages. The
training environment steps the plant at `PHYSICS_DT` = 0.1 s and `ShipModel.update` splits each
step into sub-steps of at most `SUB_DT` = 0.05 s (`REPORT.md` section 9: 0.1 s drifts about 0.5 m
over 30 s). Decisions come at 2 Hz (`UPDATE_RATE` = 0.5 s), so each decision holds for five
physics steps and ten RK4 sub-steps, and the collision test runs at every physics step. After
each sub-step u is clipped to [0, 5] m/s, |v| to 3 m/s and |r| to 160 deg/s.

**Bridge command limiter.** The 50 %/s rudder limiter of the July bridge is available in the
environment (`RUDDER_COMMAND_LIMIT`) and is off in baseline-v3, as it is in the bridge since
revision 7 (`REVIEW.md` section 8.4).

**Linear response.** A 10 % helm step at RPM 6 settles at a yaw rate equivalent to a Nomoto gain
K = r_ss / delta = 0.099 1/s (0.195 1/s at RPM 12; K L / U = 0.31 at both, the expected speed
scaling) and reaches 63 % of it 1.30 s after the step (1.15 s at RPM 12), of which 0.75 s is the
transport delay. With a 0.90 s servo lag in that path, the response time is set by the actuator,
not by the hull (`explain/explain_numbers.txt`).

## 7. Identification and the identified parameters

**Data and protocol** (`REPORT.md` sections 3, 4). Fit on 14 runs of 2026-07-02, validation on
6 runs of 2026-07-03, held back until the parameters were frozen: about 11 minutes of motion at
2 Hz, pose quantised to 0.1 m and 0.1 deg, RPM 12 in 18 of 20 runs, rudder oscillating throughout
(no sustained turn). Objective: heading and position residuals pooled over 3 s and 10 s windows and
the free run, a penalty on the p90 of |yaw rate| (a plain least-squares fit under-turned:
3.5 deg/s against 8.7 deg/s measured), a drift-angle spread term, and the steady-turn priors of
section 9 as a constraint. Optimiser: vectorised differential evolution from three starts.
Uncertainty: 28 run-level bootstrap resamples, each re-optimised locally from the full-data
optimum.

**Parameters**, fit and 90 % bootstrap interval (`REPORT.md` section 8, `ship_model_v3.CI_LOW/HIGH`):

| Parameter | Fit | 5 % | 95 % | Status |
| --- | --- | --- | --- | --- |
| T12 (N) | 16.06 | 14.74 | 18.61 | identified at n = 12 |
| X_uu | 12.89 | 11.31 | 15.11 | identified |
| X_vv | 50.64 | 47.67 | 54.78 | identified |
| X_rr | 0.058 | 0 | 13.38 | not identified |
| X_delta | 0.116 | 0 | 0.555 | not identified |
| Y_v | 0.924 | 0 | 5.96 | not identified |
| Y_vv | 220.98 | 205.67 | 236.02 | identified |
| k_R | 2.18 | 1.14 | 2.55 | correlated with N_r, N_uv |
| k_race (m/s) | 0.0135 | 0 | 0.211 | not identified |
| N_r | 1.04 | 0.014 | 8.69 | not identified; correlated |
| N_rr | 0.017 | 0 | 5.85 | not identified |
| N_uv | -83.97 | -86.80 | -55.34 | correlated; near its bound of -90 |
| rud_rate (deg/s) | 2985 | 2676 | 2998 | saturated: no rate limit below 20 deg/s seen |
| rud_tau (s) | 0.895 | 0.872 | 0.982 | identified as part of the lumped actuator |
| rud_delay (s) | 0.730 | 0.670 | 0.780 | effective delay, not a servo specification |

Fixed: t_boost = 0, u_boost = 0.5. Seven of the fifteen free parameters sit within 3 % of a bound
(`REVIEW.md` section 4). The intervals come from one day's runs and are a lower bound on the real
uncertainty (`REPORT.md` section 8). Removing 41 samples of hauled motion from the training runs
moved the yaw and actuator parameters by more than their intervals (refit v4, `REVIEW.md`
section 8.2), so the identification is not robust to data selection; neither refit was adopted.
The `ship_model_v3.py` docstring line "`rud_tau` sits at its lower bound" is stale
(`REVIEW.md` section 3.6).

**Holdout result** (`REPORT.md` section 6, with the corrections of `REVIEW.md` section 3): heading
error 1.31 / 5.19 / 7.90 / 10.17 / 13.82 / 23.53 deg at 1 / 2 / 3 / 5 / 10 s and free run, better
than the previous model at every horizon; a constant-yaw-rate extrapolation is better at 1 and 2 s,
and holding the heading constant is better over the free run (22.94 deg).

## 8. Training randomisation and reverse braking

**Randomisation** (`ship_model_v3.sample_params`, `env.reset`). At every episode reset of
baseline-v3 (`VESSEL_RANDOMISATION_SCALE` = 1.0), on a random stream separate from the scenario's:
1. two of the 28 bootstrap vectors are picked at random and blended with a uniform weight w,
   vec = w B_i + (1 - w) B_k;
2. the blend is scaled about the bootstrap mean by `scale` (1.0 in baseline-v3);
3. Gaussian jitter of 5 % of each parameter's bootstrap standard deviation is added;
4. each parameter is clamped to the sign of its bootstrap mean.

Blending whole vectors keeps the correlation of k_R, N_r and N_uv: independent draws from the
intervals land off the solution manifold and give unstable or wrong-way plants (`REPORT.md`
section 9). Over 2000 draws (`explain/explain_numbers.txt`), the 5-95 % ranges are T12 14.9-18.1 N,
X_uu 11.6-15.1, k_R 1.20-2.50, N_r 0.15-7.97, N_uv -86.3 to -60.9, rud_delay 0.67-0.77 s and
rud_tau 0.82-0.98 s; steady speed at RPM 6 is 0.532-0.582 m/s. Over 200 draws the full-helm
steady turn at RPM 6 spans 4.97-8.08 deg/s and radius 2.29-5.40 m (5-95 %: 3.14-5.29 m), speed
ratio 0.64-0.82. The randomisation spans one day's run-to-run variation; every draw starts from a
vector that satisfied the same steady-turn priors.

**Reverse braking** (`ship.py`). The identified model has no reverse thrust: `dynamics.thrust`
clamps a negative command to zero and the integrator clips surge at zero, because no July run
contains a reverse command. For the emergency stop, a negative command is applied by operator
splitting: the identified dynamics advance with zero thrust, then a braking force
`REVERSE_THRUST_EFFICIENCY` x T(|n|) removes surge momentum over the sub-step, floored at zero.
The efficiency is 0.5, an assumed value (`ship.py` TODO(05)). Full astern is S2 = -100, which the
environment maps to -24 rpm-units (`emergency_stop.s2_to_rpm`), so the braking force is
0.5 x 64.25 = 32.1 N. Reverse thrust enters only the surge equation, so every forward-thrust
trajectory is bit-identical to `ship_model_v3` (`tests/test_vessel_model.py`). Astern motion is not
modelled; the impulse the clip at zero discards is reported (`last_astern_impulse`).

## 9. Measured, assumed, inherited

**Measured in July 2026** (one propeller setting, oscillating rudder):
- the straight-line surge balance at RPM 12: steady speed 1.116 m/s against 1.14 m/s mined
  independently from the same logs (`REVIEW.md` section 2);
- the transient yaw and drift response to an oscillating rudder at that speed, through the
  windowed prediction errors;
- the lumped actuator: 0.73 s effective delay and 0.90 s lag.

**Assumed (priors, not measurements):**
- Sustained turns. The July runs contain none, and an unconstrained fit extrapolated badly
  (turn rate falling from 7.4 to 2.2 deg/s as helm went from 25 to 100 %). The fit therefore
  carries bands as a constraint: turn rate growing with helm, every helm turning the same way,
  radius 1.5-4 L, full-helm speed 0.55-0.85 of straight running. The result is 2.2 / 5.0 / 7.7 /
  10.0 deg/s at 25-100 % helm, radius 3.05 L (5.3 m), speed ratio 0.82 (`REPORT.md` section 7).
- Thrust at other settings: T12 is anchored at RPM 12 and the square law is carried over, so the
  steady speed is linear in the command (0.0930 m/s per rpm-unit) at every other setting.
- Reverse thrust: efficiency 0.5 of forward thrust, no onset delay.
- Low-speed yaw: the speed-scaled damping vanishes with surge and N_rr is 0.017, so a yaw rate
  present when surge reaches zero hardly decays (0.5 mN m of damping at 10 deg/s against an inertia
  of 10.23 kg m^2). This is what makes a stop from a full-helm turn keep turning in
  `results/field_sessions/manoeuvre_predictions.txt`.

**Inherited from `Blue02.m`:** the mass, the added masses m_x and m_y, the yaw inertia, the rudder
area, lift slope, wake fraction at the rudder, inflow lever l_R, a_H and x_H, and the sign of the
rudder yaw moment (section 4). The rudder position is x_R = -L/2 = -0.8625 m, not `Blue02.m`'s
-1.05309 m. The nondimensional hull block of `Blue02.m` is removed: evaluated at model scale it
produced about 0.09 N of sway force and 0.07 N m of yaw moment against about 10 N and 5 N m from
the terms beside it, so the previous model's behaviour came entirely from its tuning gains
(`REPORT.md` 1.3). Its derivative table must not be presented as governing either model.

**Against Fossen's standard models.**
- The linear manoeuvring model (Davidson-Schiff form, M nu_dot + N(u_0) nu = b delta, with
  constant derivatives at one speed, or its Nomoto reduction) is the small-perturbation limit of
  this one. v3 keeps the nonlinear Coriolis terms, quadratic cross-flow drag, speed-scaled linear
  terms and a rudder that stalls through its angle of attack, so it stays valid over the full
  helm range and from rest to cruise.
- There are no off-diagonal sway-yaw terms: no x_G, no Y_rdot / N_vdot added-mass coupling, no hull
  Y_r or N_v derivative other than the drift moment. The yaw-sway coupling runs only through the
  Coriolis terms, the drift moment and the rudder inflow angle.
- The damping is not linear-plus-quadratic in the usual sense of Fossen's Chapter 6 models
  (D = D_L + D_NL(nu), constant coefficients): its linear part scales with u, as in MMG, and
  vanishes at rest. Fossen's low-speed cross-flow drag integrals, which couple v and r and act at
  zero surge, are not included (section 10).
- The propeller is not a K_T(J) model: thrust does not fall with advance speed (t_boost = 0), and
  the command unit is not shaft speed.
- The actuator (delay, rate limit, lag) is explicit, which Fossen's models of ship dynamics usually
  leave to the control layer.

## 10. Known limits, in one place

| Limit | Consequence | What settles it |
| --- | --- | --- |
| Sustained turns from priors | full-helm radius 5.3 m and 10 deg/s at RPM 12 are assumptions; the February radio-control turns measured 2.4 m and 3.9 m (`feb2026/REPORT_FEB2026.md`) | turning circles, S1-D |
| Square-law thrust | speeds at RPM other than 12 are extrapolated | RPM sweep, S1-C |
| Reverse efficiency 0.5, no onset delay | the emergency stop's distance | crash stops, S1-C2 |
| No yaw damping at zero surge | residual spin after a stop from a turn | crash stops from turns, S1-C2 |
| Rudder sway force forward of midship (sign from `Blue02.m`) | drift in turns tuned through other terms | a refit with the MMG sign on S1 data |
| Pose compared at midship | if the pose is the LiDAR's, 0.86 m forward, the identified sway is that of a point near the bow | the pose reference point, O-9 |
| One day's bootstrap | randomisation narrower than the real uncertainty | S2 repeatability gives a second day |
| Lumped actuator | delay, lag and rate trade off | servo steps with an independent rudder angle, S1-B |
