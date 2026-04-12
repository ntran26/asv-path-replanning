# Bluefin 4DOF model: journal subsection rewrite and Codex handoff

## Part 1 — Journal subsection draft using the Bluefin 4DOF model

### X.X Vessel Dynamic Model and Control

The autonomous surface vessel is represented by a physics-informed manoeuvring model with four degrees of freedom (4-DOF), namely surge, sway, roll, and yaw. This formulation is appropriate for low-speed surface-vessel navigation because it preserves the dominant planar manoeuvring behavior while also retaining the coupling between lateral motion, roll motion, and turning response. The model is derived from a Japanese-style manoeuvring formulation adapted for the Bluefin platform and includes separate contributions from the hull, propeller, rudder, and bow thruster.

Two coordinate systems are used. The vessel pose is expressed in an earth-fixed frame as

\[
\eta = [x,\; y,\; \phi,\; \psi]^{\top},
\]

where \(x\) and \(y\) denote the planar position, \(\phi\) is the roll angle, and \(\psi\) is the heading angle. The body-fixed velocity vector is defined as

\[
\nu = [u,\; v,\; p,\; r]^{\top},
\]

where \(u\) is surge velocity, \(v\) is sway velocity, \(p\) is roll rate, and \(r\) is yaw rate.

The vessel state vector is written as

\[
x = [u,\; v,\; p,\; r,\; x,\; y,\; \phi,\; \psi,\; \delta,\; n_1,\; n_2]^{\top},
\tag{1}
\]

where \(\delta\) is the actual rudder angle, \(n_1\) is the main propeller rotational speed, and \(n_2\) is the bow-thruster rotational speed. The control input vector is

\[
u_c = [\delta_c,\; n_{1c},\; n_{2c}]^{\top},
\tag{2}
\]

where \(\delta_c\) is the commanded rudder angle, \(n_{1c}\) is the commanded propeller speed, and \(n_{2c}\) is the commanded bow-thruster speed.

The kinematics are expressed as

\[
\dot{x} = u\cos\psi - v\sin\psi \cos\phi,
\tag{3}
\]

\[
\dot{y} = u\sin\psi + v\cos\psi \cos\phi,
\tag{4}
\]

\[
\dot{\phi} = p,
\tag{5}
\]

\[
\dot{\psi} = r\cos\phi.
\tag{6}
\]

These relations convert the body-fixed velocities into earth-fixed motion and account for the effect of roll on the projected sway and yaw motion.

The dynamics are modeled by balancing the total hull, propeller, rudder, and bow-thruster contributions. The effective inertial terms are written as

\[
m_{11}=m+m_x,\qquad m_{22}=m+m_y,\qquad m_{33}=I_x+J_x,\qquad m_{44}=I_z+J_z,
\tag{7}
\]

where \(m\) is the rigid-body mass, \(m_x\) and \(m_y\) are added-mass terms in surge and sway, \(I_x\) and \(I_z\) are rigid-body roll and yaw inertias, and \(J_x\) and \(J_z\) are added inertias.

The total dimensionless forces and moments are decomposed as

\[
X_d = X_{dH}+X_{dP}+X_{dR},
\tag{8}
\]

\[
Y_d = Y_{dH}+Y_{dR}+Y_{dB},
\tag{9}
\]

\[
K_d = K_{dH}+K_{dH2}+K_{dR}+K_{dB},
\tag{10}
\]

\[
N_d = N_{dH}+N_{dR}-x_GY_d+N_{dB},
\tag{11}
\]

where the subscripts \(H\), \(P\), \(R\), and \(B\) denote the hull, propeller, rudder, and bow-thruster contributions, respectively.

The surge, sway, roll, and yaw accelerations are then written as

\[
\dot{u}=\frac{X_d\left(\tfrac12\rho L d U^2\right)+m_{22}vr}{m_{11}},
\tag{12}
\]

\[
\dot{v}=\frac{Y_d\left(\tfrac12\rho L d U^2\right)-m_{11}ur}{m_{22}},
\tag{13}
\]

\[
\dot{p}=\frac{K_d\left(\tfrac12\rho L d^2 U^2\right)+(z_H-z_G)(m_y\dot{v}+m_xur)}{m_{33}},
\tag{14}
\]

\[
\dot{r}=\frac{N_d\left(\tfrac12\rho L^2 d U^2\right)}{m_{44}},
\tag{15}
\]

where \(U=\sqrt{u^2+v^2}\) is the resultant body-frame speed, \(\rho\) is water density, \(L\) is vessel length, \(d\) is draft, and \(x_G\), \(z_G\), and \(z_H\) are geometric offsets used in the roll-yaw coupling terms.

The propeller thrust is modeled through an advance-ratio formulation. The advance ratio is given by

\[
J = \frac{(1-w)u}{n_1 D_P},
\tag{16}
\]

and the thrust coefficient is written as a quadratic polynomial

\[
K_T = a_0 + a_1J + a_2J^2.
\tag{17}
\]

The corresponding propeller surge contribution is

\[
X_{dP} \propto |n_1|n_1 (1-t) K_T D_P^4.
\tag{18}
\]

This form allows the propeller force to vary naturally with both shaft speed and vessel forward speed through the advance ratio.

The rudder force is generated from the local inflow velocity and effective angle of attack. The non-dimensional rudder inflow components are

\[
u_{dR} = \epsilon (1-w)\sqrt{\eta\left(1+\kappa\sqrt{1+\frac{8K_T}{\pi J^2}}-1\right)^2 + (1-\eta)},
\tag{19}
\]

\[
v_{dR} = -\gamma_R\left(\beta-l_R r_d+\frac{p(z_R-z_G)}{U}\right),
\tag{20}
\]

with resultant rudder inflow

\[
U_{dR} = \sqrt{u_{dR}^2 + v_{dR}^2}.
\tag{21}
\]

The effective rudder angle of attack is

\[
\alpha_R = \delta - \tan^{-1}\left(\frac{-v_{dR}}{u_{dR}}\right),
\tag{22}
\]

and the rudder normal force is approximated as

\[
F_{dN}= -\left(\frac{A_R}{Ld}\right)\left(\frac{6.13\lambda}{2.25+\lambda}\right)U_{dR}^2\sin\alpha_R.
\tag{23}
\]

The rudder contributions are then resolved into surge, sway, roll, and yaw as

\[
X_{dR} = (1-t_R)F_{dN}\sin\delta \cos\phi,
\tag{24}
\]

\[
Y_{dR} = (1+a_H)F_{dN}\cos\delta \cos\phi,
\tag{25}
\]

\[
K_{dR} = z_R Y_{dR}/L,
\tag{26}
\]

\[
N_{dR} = (x_R + a_H x_H)F_{dN}\cos\delta \cos\phi.
\tag{27}
\]

The hull contribution is described by nonlinear polynomial terms in drift angle, yaw rate, and roll angle. In practice, these terms act as hydrodynamic damping and restoring effects that shape the vessel response in sway, roll, and yaw. The roll equation also includes an explicit hydrostatic restoring moment and a damping term,

\[
K_{dH2}=z_GY_{dH}-B_{44}p-C_{44}\phi-(z_R-z_G)Y_{dR},
\tag{28}
\]

where \(C_{44}=g m GM\) is the roll restoring coefficient and \(B_{44}\) is the roll damping term.

The bow thruster is represented as a lateral force applied at an offset from the vessel centre. Its contribution is

\[
F_{BT}\propto |n_2|n_2 K_{BT},
\tag{29}
\]

which produces sway, roll, and yaw effects through

\[
Y_{dB}=F_{BT},\qquad K_{dB}=z_BF_{BT}/L,\qquad N_{dB}=x_BF_{BT}/L.
\tag{30}
\]

The rudder and shaft states are dynamic rather than instantaneous. The actuator dynamics are modeled as

\[
\dot{\delta} = \mathrm{sat}_{\dot{\delta}_{\max}}(\delta_c-\delta),
\tag{31}
\]

\[
\dot{n}_1 = \mathrm{sat}_{\dot{n}_{\max}}(n_{1c}-n_1),
\qquad
\dot{n}_2 = \mathrm{sat}_{\dot{n}_{\max}}(n_{2c}-n_2),
\tag{32}
\]

where the saturation operator enforces realistic rudder-rate and shaft-rate limits. This is important because the reinforcement-learning agent acts on commanded controls, while the vessel responds according to the actual actuator states.

The full state vector is integrated numerically with a fixed-step Runge–Kutta method in the simulation environment. Internally, the model uses SI units and radians, while the reinforcement-learning environment maps normalized actions to commanded rudder and shaft-speed values. The calibrated parameter set is then identified by comparing simulated straight-line and turning-circle tests against the measured Bluefin trials.

---

## Part 2 — Codex handoff: everything needed to construct and validate the new Bluefin 4DOF model

This section is a practical handoff note for another coding agent.

### 1. Which MATLAB files matter

Use these files as the main source:

- `bluefin_matlab/Bluefin 2022/Bluefin4DOFModel02.m`
- `bluefin_matlab/Bluefin 2022/Bluefin4DOFModel02_Solver.m`

Fallback / comparison files:
- `bluefin_matlab/Bluefin 2022/Bluefin4DOFModel.m`
- `bluefin_matlab/Bluefin 2022/JapaneseModelBluefin01.m`

Do **not** use `Blue02.m` as the main source. It is incomplete/inconsistent:
- comments and state indexing do not agree,
- it claims a 9-state model but returns only 7 derivatives,
- it does not provide a clean full 3-DOF/4-DOF state evolution.

### 2. Why Bluefin4DOFModel02.m is the best source

It is the strongest candidate because it contains:
- surge, sway, roll, and yaw dynamics,
- actual vessel position and heading,
- rudder actuator state,
- main propeller state,
- bow-thruster state,
- separate hull, propeller, rudder, and thruster forces,
- a solver file with example setup.

The model state is:

\[
x = [u,\; v,\; p,\; r,\; x,\; y,\; \phi,\; \psi,\; \delta,\; n_1,\; n_2]^{\top}
\]

The input is:

\[
u_c = [\delta_c,\; n_{1c},\; n_{2c}]^{\top}
\]

This is richer and more internally consistent than `Blue02.m`.

### 3. What the Python implementation must preserve

A Python implementation should preserve:

1. **State ordering**
   - `u, v, p, r, x, y, phi, psi, delta, n1, n2`

2. **Inputs**
   - commanded rudder angle
   - commanded main propeller rpm
   - commanded bow-thruster rpm

3. **Physics blocks**
   - hull forces and moments
   - propeller thrust via advance ratio \(J\)
   - rudder force from effective inflow
   - roll restoring / damping
   - bow-thruster side force and yaw moment
   - actuator dynamics for rudder and shafts

4. **Units**
   - SI internally
   - radians for angles inside the model
   - rpm/rps mapped carefully

5. **Earth-fixed kinematics**
   - `x_dot, y_dot, phi_dot, psi_dot`

### 4. Numerical issues that must be handled in Python

The MATLAB file assumes some quantities are nonzero. In Python, these must be guarded.

#### A. Zero-speed protection
The model computes:
\[
U = \sqrt{u^2+v^2}
\]
and also uses:
\[
\beta = -\sin^{-1}(v/U)
\]

If \(U \to 0\), the implementation will blow up.

In Python:
- use `U_eff = max(U, U_MIN)`
- clip the argument of `asin` to `[-1, 1]`

#### B. Propeller advance ratio
The model uses:
\[
J = \frac{(1-w)u}{n_1 D_P}
\]

If \(n_1\) is zero or nearly zero, this becomes singular.

In Python:
- use `n1_eff = sign(n1)*max(abs(n1), n_min)`
- or switch to a static-thrust approximation near zero shaft speed

#### C. Rudder-flow model
The rudder inflow equations contain \(J^2\) in the denominator inside the square-root term. This must be protected numerically at low speed / low rpm.

#### D. Roll/yaw growth
The model includes polynomial hull terms and tentative coefficients. Add safe clipping to avoid numerical explosion:
- clip roll angle
- clip yaw rate
- clip sway velocity
if needed during debugging

### 5. Recommended Python class design

Use a class with the same external feel as the old model:

```python
model = ShipModel()
dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rudder_percent, dt, thruster_rpm=0.0)
```

Internally:
- keep full state vector
- map `rudder_percent -> delta_c`
- map `rpm -> n1_c`
- optional `thruster_rpm -> n2_c`

The class should also expose:
- `self._u`
- `self._v`
- `self._p`
- `self._r`
- `self._phi`
- `self._psi`
- `self._delta`
- `self._n1`
- `self._n2`

for logging and debugging.

### 6. Integration method

Use fixed-step RK4, not Euler.

Why:
- the model includes actuator states,
- nonlinear force polynomials,
- roll-yaw coupling,
- and speed-dependent inflow.

Euler is likely to be too unstable or too inaccurate at the RL timestep.

Recommended:
- `dt = 0.1 s` or whatever the simulator uses
- RK4 over the full 11-state vector

### 7. Sign conventions

Before training or sweeping, decide and freeze the sign convention.

Recommended standard:
- positive rudder = starboard
- positive yaw rate = starboard

But the final convention must match the real vessel logs and controller interpretation.

If the real vessel / controller uses the opposite sign, flip the mapping once at the interface rather than changing signs inconsistently inside the dynamics.

### 8. What files from the current workflow should be reused

Keep and adapt:
- `log_parser.py`
- `run_open_loop_tests.py`
- `bluefin_test_utils.py`
- `focused_v2_sweep.py` (only as a template)
- the real benchmark JSON/log files:
  - `test_3.log`
  - `test_4.log`
  - `test_3_metrics.json`
  - `test_4_metrics.json`
  - comparison JSON files already created

The new 4DOF candidate should be validated with the **same benchmark metrics** as the v2 model so results are directly comparable.

### 9. How to validate the 4DOF candidate

Use the same two experiment classes:

#### A. Straight-line speed test
Simulate zero rudder, constant shaft command.

Compare against real metrics:
- peak forward speed
- initial acceleration over 0–2 s
- initial acceleration over 0–5 s
- distance at 10 s
- time to 50% peak speed
- time to 90% peak speed

#### B. Turning-circle test
Simulate constant shaft command and constant rudder.

Compare against real metrics:
- peak yaw rate
- time to 90° after turn start
- time to 180° after turn start
- first-90° radius
- first-180° radius
- speed 10 s into the turn

### 10. What to sweep first

Do **not** try to sweep every coefficient in `Bluefin4DOFModel02.m`. There are too many.

Start by keeping the MATLAB hydrodynamic coefficients fixed and sweeping only a small set of scale factors around them.

Recommended first sweep parameters:
- `PROPELLER_THRUST_SCALE`
- `RUDDER_FORCE_SCALE`
- `RUDDER_YAW_SCALE`
- `RUDDER_X_DRAG_SCALE`
- `LINEAR_SURGE_DAMP`
- `LINEAR_YAW_DAMP`
- optionally:
  - `ROLL_DAMP_SCALE`
  - `BOW_THRUSTER_SCALE`

This gives a manageable identification problem.

### 11. What success looks like

A good new 4DOF model should improve over the current best v2 fit in the areas where v2 still struggles:

- better peak yaw rate
- better early-turn and 180° turn timing
- better turn-speed retention
- ideally a better full surge transient shape

The current best v2 fit already matches straight-line peak speed, 0–2 s acceleration, and 10 s distance quite well, but still has:
- too-slow 50%/90% rise timing,
- too-low peak yaw rate,
- too-tight early turn,
- and too-low speed in the turn.

The 4DOF candidate is worth adopting only if it improves those weaknesses.

### 12. Practical implementation order for Codex

1. Read and port `Bluefin4DOFModel02.m`
2. Build a stable Python class with RK4
3. Add numerical protections for low speed / low rpm
4. Match the external API of the old Python ship model
5. Run simple open-loop straight and turn tests
6. Export the same benchmark metrics as the current tooling
7. Compare against the existing v2 baseline
8. Only then run a focused sweep

### 13. Current status summary for the handoff

- `Blue02.m` should **not** be the main source.
- `Bluefin4DOFModel02.m` is the most promising MATLAB source.
- The new Python candidate should preserve the 4DOF structure and actuator states.
- Validation must use the same straight and turning metrics already established in the project.
- The first comparison target is not “looks plausible”, but “beats the current v2 baseline on the remaining known errors”.

