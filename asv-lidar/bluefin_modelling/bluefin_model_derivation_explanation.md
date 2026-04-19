# Bluefin vessel model derivation and Python implementation notes

**Purpose.** This note documents the workflow:

```text
MATLAB Bluefin model -> marine craft modelling theory -> Python ship model
```

It is written to support the journal/thesis documentation and to make the modelling process repeatable. It focuses on:

1. the older `Blue02.m`-style model and how it led to `ship_model_bluefin_v2.py`;
2. the more complete `Bluefin4DOFModel02.m` model and how it leads to `bluefin_4dof_final.py`;
3. the page locations in Fossen's *Marine Control Systems* where the theory can be checked.

The final Python file prepared with the latest refined faithful-4DOF configuration is:

```text
bluefin_4dof_final.py
```

## 1. Key references in *Marine Control Systems*

The exact Bluefin coefficients are not in Fossen's book. They come from the MATLAB files. Fossen's book provides the general marine-craft modelling framework used to interpret and structure those coefficients.

| Modelling idea used here | Where to check in Fossen's book | How it relates to the Bluefin model |
|---|---|---|
| Body-fixed and earth-fixed coordinate frames | Chapter 2, Reference Frames, p. 19-21 | The Bluefin models use body velocities such as surge $u$, sway $v$, roll rate $p$, yaw rate $r$, and earth-fixed pose $x,y,\psi$. Fossen states that positions/orientations are described in an inertial or NED frame, while velocities are expressed in the body-fixed frame. |
| General marine craft equation | Chapter 3, p. 50; p. 63-64| Fossen writes the marine vessel model in the form $M\dot{\nu}+C(\nu)\nu+D(\nu)\nu+g(\eta)=\tau+g_0+w$. The MATLAB Bluefin models are component-wise versions of this idea. |
| Added mass and hydrodynamic inertia | Chapter 3.2.1, p. 63-67 | The Bluefin models use $m_x$, $m_y$, $J_x$, and $J_z$ as added-mass / added-inertia terms. Fossen explains that added mass represents inertia of the surrounding fluid. |
| Hydrodynamic damping | Chapter 3.2.2, p. 71-75 | The polynomial hull force terms in the MATLAB models are empirical hydrodynamic damping / manoeuvring derivatives. Fossen explains linear and nonlinear damping, including skin friction, vortex shedding, and velocity-dependent terms. |
| 3-DOF horizontal model | Chapter 3.5.1, p. 104-108 | `Blue02.m` is best understood as a 3-DOF horizontal manoeuvring model in surge, sway, and yaw. Fossen defines this reduction as $\nu=[u,v,r]^T$ and $\eta=[x,y,\psi]^T$. |
| Forward-speed model | Chapter 3.5.2, p. 107 | Fossen's forward-speed equation includes both linear and quadratic damping. This motivated using speed-dependent surge resistance in Python. |
| Rudder sign and yaw convention | Chapter 3.5.2, p. 108 | Fossen states that a positive rudder angle may be defined to yield positive yaw rate in a given model convention. This justifies explicitly documenting and fixing the sign bridge in Python. |
| Actuator forces and moment arms | Chapter 7.5.1, p. 288-291 | Fossen explains that propellers, tunnel thrusters, rudders, fins, etc. generate forces and moments through actuator geometry. This supports the 4DOF separation into propeller, rudder, and bow-thruster contributions. |
| 4-DOF actuator layout | Chapter 7.5.1, p. 291 | Fossen explicitly discusses actuator configuration columns in 4 DOF: surge, sway, roll, and yaw. This matches the Bluefin 4DOF model structure. |
| Restoring forces / roll restoring | Chapter 3.2.3, p. 75-76 | The 4DOF model includes roll restoring through $C_{44}=g\,m\,GM$, which follows the general idea that gravity and buoyancy generate restoring forces and moments. |

## 2. The general theory behind both MATLAB models

Fossen's general marine craft equation is

$$
M\dot{\nu}+C(\nu)\nu+D(\nu)\nu+g(\eta)=\tau+g_0+w
$$

For the Bluefin work, the terms are interpreted as follows:

- $M\dot{\nu}$: rigid-body and added-mass inertia;
- $C(\nu)\nu$: centripetal/Coriolis-like velocity coupling;
- $D(\nu)\nu$: hydrodynamic damping and manoeuvring derivatives;
- $g(\eta)$: restoring effects, especially important for roll in 4DOF;
- $\tau$: propeller, rudder, and thruster forces/moments;
- $w$: disturbances such as wind, waves, and current, not included in the current calm-water calibration.

The Bluefin MATLAB scripts do not necessarily write the equations in matrix form. Instead, they expand the model into component equations for $X$, $Y$, $K$, and $N$, where:

- $X$: surge force
- $Y$: sway force
- $K$: roll moment
- $N$: yaw moment

This is still the same modelling logic as Fossen's equation. The MATLAB model simply writes each DOF explicitly.

## 3. `Blue02.m`: what it is and how it maps to theory

### 3.1 Model type

`Blue02.m` is best understood as a **3-DOF horizontal manoeuvring model**. Its intended dynamic states are surge $u$, sway $v$, yaw rate $r$, heading $\psi$, rudder angle $\delta$, and planar position $x,y$.

This matches Fossen's 3-DOF horizontal reduction:

$$
\nu = [u,\ v,\ r]^T,\qquad \eta = [x,\ y,\ \psi]^T
$$

Fossen explains this model class under "3 DOF Horizontal Model", where he states that horizontal ship motion is usually described by surge, sway, and yaw.

### 3.2 Inertia and added mass

The MATLAB file uses:

$$
m_{11}=m+m_x,\qquad m_{22}=m+m_y,\qquad m_{33}=I_z+J_z
$$

This follows Fossen's distinction between rigid-body mass/inertia and hydrodynamic added mass. In the book, the total inertia is written as $M=M_{RB}+M_A$, and added mass is described as the inertia of the surrounding fluid.

### 3.3 Hull force and damping terms

`Blue02.m` forms hull contributions such as

$$
X_H,\quad Y_H,\quad N_H
$$

from polynomial terms in $v$, $r$, $v^2$, $r^2$, and $vr$. These are empirical manoeuvring derivatives. They correspond to Fossen's $D(\nu)\nu$ and nonlinear hydrodynamic force terms.

A simplified interpretation is:

$$
X = X_H + X_P + X_R
$$
$$
Y = Y_H + Y_R
$$
$$
N = N_H + N_R
$$

Fossen explains that damping may include linear and nonlinear components, including skin friction, vortex shedding, and other velocity-dependent effects.

### 3.4 Propeller model

In `Blue02.m`, the propeller contributes a surge force $X_P$. This follows the same actuator idea as Fossen's actuator section: a main propeller produces a force in the longitudinal $x$-direction.

### 3.5 Rudder model

`Blue02.m` computes an effective rudder inflow, then forms a normal force $F_N$. This force is resolved into surge, sway, and yaw contributions:

$$
X_R \sim F_N\sin\delta
$$
$$
Y_R \sim F_N\cos\delta
$$
$$
N_R \sim x_R F_N\cos\delta
$$

This matches Fossen's actuator explanation: an aft rudder produces a lateral force as a function of rudder deflection, and that lateral force creates a yaw moment for steering. 

### 3.6 Why `Blue02.m` alone was not enough

`Blue02.m` was useful, but it had several limitations:

1. The state comments and implementation are not fully consistent.
2. Roll is mentioned but not actually included.
3. It does not include a bow-thruster state.
4. It is less complete than `Bluefin4DOFModel02.m`.
5. It does not provide enough independent tuning freedom to match both straight-line and turning performance.

This is why the Python path first used a simplified 3DOF-inspired model (`ship_model_bluefin_v2.py`) and then moved to a 4DOF candidate.

## 4. `ship_model_bluefin_v2.py`: how it was constructed from the 3DOF theory

`ship_model_bluefin_v2.py` is not a literal copy of `Blue02.m`. It is a **control-oriented 3DOF model** inspired by Bluefin constants and by the 3DOF theory.

### 4.1 State and public interface

The Python state is:

$$
x = [u,\ v,\ r,\ \psi,\ \delta,\ x,\ y]^T
$$

In the code, this is implemented by:

- `reset()` and stored fields: lines 118-125;
- `state_dict()`: lines 127-139;
- `_state_vector()` and `_set_state_vector()`: lines 166-179.

The public interface is:

```python
dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)
```

This is implemented at lines 141-164. It keeps the same input/output format as the older Gym environment while hiding the internal dynamics.

### 4.2 RK4 integration

The model uses Runge-Kutta integration at lines 148-152:

$$
x_{k+1}=x_k+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4)
$$

This is a numerical improvement over a simple Euler step. It was used because the nonlinear rudder and thrust equations can be stiff enough that Euler integration gives unstable or inaccurate results at the 10 Hz simulation rate.

### 4.3 Speed-shaped propeller law

In v2, the propeller force is implemented at lines 196-208 as:

$$
X_P = (1-t_P)K_T n|n|
\frac{1+k_b e^{-u/U_b}}{1+k_d u^2}
$$

This was added because a simple constant $rpm^2$ thrust law could not match both early acceleration and the final speed envelope. The idea is consistent with Fossen's forward-speed equation, where surge dynamics include both control force and linear/quadratic resistance. 

### 4.4 Hull damping and manoeuvring derivatives

The v2 model computes hull surge damping at lines 221-228 and sway/yaw damping at lines 230-240. These terms are the Python equivalent of the MATLAB hull-polynomial terms, but simplified and tuned.

The form follows the idea that hull resistance and manoeuvring forces are velocity-dependent, as discussed in Fossen's hydrodynamic damping section.

### 4.5 Rudder split: axial loss, sway force, yaw moment

The v2 rudder block is implemented at lines 249-265:

- rudder inflow $u_R,v_R$
- angle of attack $\alpha_R$
- rudder normal force $F_N$
- axial drag $X_R$
- sway force $Y_R$
- yaw moment $N_R$

The key v2 modelling decision was to separate:

$$
X_R,\quad Y_R,\quad N_R
$$

instead of forcing all rudder effects to share one coefficient. This follows the physical idea that rudder forces create both lateral force and yaw moment, while also reducing forward speed. Fossen's actuator section supports this interpretation by describing rudders as lateral-force devices that generate steering yaw moments.

## 5. `Bluefin4DOFModel02.m`: why it is the better MATLAB source

`Bluefin4DOFModel02.m` is more complete than `Blue02.m`.

It uses the state vector:

$$
x = [u,\ v,\ p,\ r,\ x,\ y,\ \phi,\ \psi,\ \delta,\ n_1,\ n_2]^T
$$

The inputs are:

$$
u_c=[\delta_c,\ n_{1c},\ n_{2c}]^T
$$

This means the MATLAB file models:

- surge $u$
- sway $v$
- roll rate $p$
- yaw rate $r$
- position $x,y$
- roll angle $\phi$
- heading $\psi$
- actual rudder angle $\delta$
- actual propeller state $n_1$
- actual bow-thruster state $n_2$

This matches the 4DOF idea in Fossen's actuator/allocation section, where 4DOF is listed as surge, sway, roll, and yaw.

### 5.1 4DOF force and moment balance

The MATLAB file constructs:

$$
X_d = X_{dH}+X_{dP}+X_{dR}
$$

$$
Y_d = Y_{dH}+Y_{dR}+Y_{dB}
$$

$$
K_d = K_{dH}+K_{dH2}+K_{dR}+K_{dB}
$$

$$
N_d = N_{dH}+N_{dR}-x_GY_d+N_{dB}
$$

These are simply component-wise versions of the generalized force vector $\tau$ in Fossen's general equation.

### 5.2 Surge and sway acceleration

The MATLAB model uses:

$$
\dot{u} = \frac{X_d(0.5\rho LdU^2)+m_{22}vr}{m_{11}}
$$

$$
\dot{v} = \frac{Y_d(0.5\rho LdU^2)-m_{11}ur}{m_{22}}
$$

These equations come from expanding the inertia and Coriolis-like coupling terms in component form. They are analogous to Fossen's rigid-body component equations and the general matrix equation. 

### 5.3 Roll equation

The 4DOF model adds roll:

$$
\dot{p} = \frac{K_d(0.5\rho Ld^2U^2)+(z_H-z_G)(m_y\dot{v}+m_xur)}{m_{33}}
$$

The roll restoring term is based on:

$$
C_{44}=g\,m\,GM
$$

This follows Fossen's discussion of restoring forces and moments from gravity and buoyancy. 

### 5.4 Yaw equation

The yaw equation is:

$$
\dot{r} = \frac{N_d(0.5\rho L^2dU^2)}{m_{44}}
$$

This is the yaw component of the generalized moment equation.

### 5.5 Propeller advance-ratio model

The 4DOF MATLAB file uses an advance-ratio style propeller law:

$$
J=\frac{(1-w)u}{n_1D_P}
$$

$$
K_T=a_0+a_1J+a_2J^2
$$

Then $K_T$ is used in the propeller surge contribution $X_{dP}$. This is more physically structured than the simple $rpm^2$ law in the earliest Python model.

Fossen's actuator section says the main propeller produces a longitudinal force for transit, while the exact propeller coefficient law comes from the MATLAB Bluefin file itself.

### 5.6 Rudder inflow and normal force

The 4DOF MATLAB file computes a local rudder inflow and a rudder normal force:

$$
U_{dR}=\sqrt{u_{dR}^2+v_{dR}^2},
$$

$$
\alpha_R=\delta-\arctan\left(\frac{-v_{dR}}{u_{dR}}\right)
$$

$$
F_{dN}=-
\left(\frac{A_R}{Ld}\right)
\left(\frac{6.13\lambda}{2.25+\lambda}\right)
U_{dR}^2\sin\alpha_R
$$

The force is then resolved into surge, sway, roll, and yaw contributions. This is directly aligned with Fossen's explanation that actuator forces generate forces and moments through moment arms.

### 5.7 Bow thruster

The MATLAB file adds a bow-thruster force $F_{BT}$, contributing to sway, roll, and yaw through its moment arms. This is consistent with Fossen's description of tunnel thrusters as transverse actuators producing $F_y$ for low-speed manoeuvring and dynamic positioning.

## 6. `bluefin_4dof_final.py`: how the Python file maps to the 4DOF theory

The final Python file keeps the same public interface as the v2 model:

```python
dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)
```

This makes model switching easy in the Gym environment.

### 6.1 Constants and calibration

The calibrated constants are at the top of `bluefin_4dof_final.py`:

- `RPM_COMMAND_SCALE = 85.0`
- `PROPELLER_THRUST_SCALE = 1.7`
- `RUDDER_FORCE_SCALE = 0.6`
- `RUDDER_X_DRAG_SCALE = 0.6`
- `ROLL_DAMP_SCALE = 1.0`
- `ROLL_RESTORE_SCALE = 1.4`

These values are based on the latest refined faithful-4DOF sweep. The validation commands were:

- straight speed test: `rpm = 15`
- turning test: `turn_rpm = 18`
- turning rudder: `turn_rudder_deg = 25`

The commands are not constants of the ship. They are test inputs used to compare simulation and real trials.

### 6.2 State and compatibility interface

The 4DOF Python state is:

$$
x=[u,\ v,\ p,\ r,\ x,\ y,\ \phi,\ \psi,\ \delta,\ n_1,\ n_2]^T
$$

In the Python file this is implemented in:

- `reset()`: lines 76-110
- `state_dict()`: lines 112-128
- `_state_vector()`: lines 169-185
- `_set_state_vector()`: lines 187-202

The compatibility fields `_v_sway`, `_w`, and `_h` make the 4DOF model compatible with scripts written for older models.

### 6.3 Input mapping and sign convention

The Python `update()` method maps the public control command into MATLAB-style internal inputs:

- line 146: rudder percentage becomes internal rudder angle;
- line 147: repo-facing rpm becomes MATLAB command rpm;
- line 148: optional bow-thruster command becomes MATLAB command rpm.

The sign bridge is explicit:

```python
delta_cmd = -rud / 100 * MAX_RUD_ANGLE
```

This means the public repo convention can stay consistent even if the MATLAB internal sign is opposite. This is important because Fossen notes that rudder sign convention is model-defined; what matters is consistency between rudder and yaw rate.

### 6.4 RK4 integration and guards

The Python code uses RK4 at lines 150-154. It clips extreme states at lines 156-162. These protections are not part of the theory; they are practical numerical safeguards to make the model usable from rest and in repeated RL/sweep simulations.

### 6.5 Derivative function

The physics is implemented in `_derivatives()` from line 204 onward.

The block structure is:

| Python block | Lines | Theory / MATLAB source |
|---|---:|---|
| Unpack state | 211 | $x=[u,v,p,r,x,y,\phi,\psi,\delta,n_1,n_2]^T$ |
| Speed and drift angle | 213-215 | $U=\sqrt{u^2+v^2}$, $\beta=-\arcsin(v/U)$ |
| Rudder actuator | 217-223 | $\dot{\delta}=\mathrm{sat}(\delta_c-\delta)$ |
| Shaft dynamics | 225-228 | $\dot{n}_1,\dot{n}_2$ saturation |
| Physical constants | 230-259 | Bluefin4DOFModel02 coefficients |
| Hull coefficients | 261-306 | MATLAB polynomial hull derivatives |
| Inertia terms | 308-321 | $m_{11}=m+m_x$, $m_{22}=m+m_y$, $m_{33}=I_x+J_x$, $m_{44}=I_z+J_z$ |
| Propeller model | 327-354 | advance-ratio $J$, thrust coefficient $K_T$, propeller surge force |
| Rudder model | 356-364 | rudder inflow, $\alpha_R$, $F_N$, $X_R,Y_R,K_R,N_R$ |
| Hull forces/moments | 366-408 | $X_{dH},Y_{dH},K_{dH},N_{dH}$ |
| Roll restoring/damping | 411-416, 445-451 | $C_{44}=g\,m\,GM$, roll damping/restoring moments |
| Bow thruster | 418-430 | $Y_B,K_B,N_B$ |
| Total forces/moments | 432-435 | $X,Y,K,N$ sums |
| Accelerations | 441-453 | $\dot{u},\dot{v},\dot{p},\dot{r}$ |
| Kinematics | 454-457 | earth-fixed $\dot{x},\dot{y},\dot{\phi},\dot{\psi}$ |
| Return derivative vector | 459 onward | full state derivative |

### 6.6 Why the 4DOF model is useful even if v2 remains competitive

The v2 model gave better straight-line matching in the latest calibration, while the faithful 4DOF model gave better turn geometry and speed retention in a turn. This is expected:

- v2 is more empirically shaped for the two real tests;
- 4DOF is more physically structured and closer to a conventional manoeuvring model.

Therefore, the 4DOF model is more attractive for long-term modelling, while v2 may remain the short-term RL training model if straight-line fidelity is more important.

## 7. Practical workflow for future modelling problems

The general modelling process should be:

1. **Start from theory.** Use Fossen's equation $M\dot{\nu}+C(\nu)\nu+D(\nu)\nu+g(\eta)=\tau+w$ as the modelling scaffold.
2. **Identify states and inputs.** Decide whether the vessel needs 3DOF or 4DOF.
3. **Separate force sources.** Hull, propeller, rudder, and thruster should be separate blocks.
4. **Use MATLAB or literature coefficients as a starting point.**
5. **Add numerical guards only where needed.**
6. **Run output-only benchmarks.** Use straight-line and turning-circle tests.
7. **Tune small scale factors first.** Avoid sweeping every hydrodynamic derivative.
8. **Compare against real metrics, not only visual plots.**
9. **Prefer a simpler model if it is more reliable for RL.** A more detailed model is not automatically better unless it improves the validation metrics.

## 8. Main takeaway

The workflow is:

```text
Blue02.m
  -> 3DOF surge-sway-yaw idea
  -> ship_model_bluefin_v2.py
  -> strong practical RL model after empirical thrust/rudder shaping

Bluefin4DOFModel02.m
  -> 4DOF surge-sway-roll-yaw structure
  -> bluefin_4dof_final.py
  -> more physically complete model, stronger turning-shape reference
```

Both models are valid, but they serve different purposes. `ship_model_bluefin_v2.py` is currently the better practical RL model if straight-line speed fidelity is critical. `bluefin_4dof_final.py` is the better physics-based candidate for future refinement, especially for turning, roll, and actuator-state modelling.

