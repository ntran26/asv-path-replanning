
# Bluefin Vessel Model Derivation

**Purpose.** This note documents the workflow:

```text
MATLAB Bluefin model -> marine craft modelling theory -> Python ship model
```

It is written to support the journal/thesis documentation and to make the modelling process repeatable. It focuses on:

1. the older `Blue02.m`-style model and how it led to `ship_model_bluefin_v2.py`;
2. the more complete `Bluefin4DOFModel02.m` model and how it leads to `bluefin_4dof_final.py`;
3. the page locations in Fossen's *Marine Control Systems* where the theory can be checked.

The final Python file prepared with the latest refined 4-DOF configuration is:

```text
ship_model_bluefin_4dof.py
```

## 1. Key references in *Marine Control Systems: Guidance, Navigation, and Control of Ships, Rigs and Underwater Vehicles*

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

- $M\dot{\nu}$: rigid-body and added-mass inertia
- $C(\nu)\nu$: centripetal/Coriolis-like velocity coupling
- $D(\nu)\nu$: hydrodynamic damping and manoeuvring derivatives
- $g(\eta)$: restoring effects, especially important for roll in 4DOF
- $\tau$: propeller, rudder, and thruster forces/moments
- $w$: disturbances such as wind, waves, and current, not included in the current calm-water calibration

The Bluefin MATLAB scripts do not necessarily write the equations in matrix form. Instead, they expand the model into component equations for $X$, $Y$, $K$, and $N$, where:

- $X$: surge force
- $Y$: sway force
- $K$: roll moment
- $N$: yaw moment

This is still the same modelling logic as Fossen's equation. The MATLAB model simply writes each DOF explicitly.

## 3. `Blue02.m`: MATLAB code, equations, and theory mapping

`Blue02.m` is best understood as a compact **3-DOF horizontal manoeuvring model** in surge, sway, and yaw, with rudder and propeller effects added explicitly. It is not a fully consistent 9-state implementation, because the file comments list a larger state vector than the code actually uses. Nevertheless, the force construction follows the same marine-craft modelling idea described by Fossen: inertia and added mass are balanced against hydrodynamic damping and actuator forces.

### 3.1 Function, intended states, and inputs

The MATLAB file starts with the function definition and the intended state/input comments:
 
```matlab
function xdot = Blue02(x,ui)
% State vector:
% x(1) = u [m]
% x(2) = v [m]
% x(3) = p [rad/s]
% x(4) = r [rad/s]
% x(5) = x [m]
% x(6) = y [m]
% x(7) = phi [rad]
% x(8) = psi [rad]
% x(9) = del [rad]
% Input vector:
% u(1) = rudder [rad]
% u(2) = rpm of prop [rpm]
% u(3) = rpm of thruster [rpm]
```

The intended 3-DOF horizontal theory is:

$$
\nu=[u,\ v,\ r]^T,
\qquad
\eta=[x,\ y,\ \psi]^T.
$$

This is the standard horizontal-plane vessel model described by Fossen under the 3-DOF reduction. Surge $u$, sway $v$, and yaw rate $r$ are body-fixed velocities, while $x$, $y$, and $\psi$ describe earth-fixed position and heading.

However, the implementation later unpacks the state differently:

```matlab
U = sqrt(0.6^2+0.4^2);
u = x(1)/U;
v = x(2)/U;
r = x(3)*L/U;
psi = x(4);
del = x(5);
phi = x(6);
del_c = ui(1);
n1 = ui(2)/60;     % rps of prop
n2 = ui(3)/60;     % rps of thruster
```

This means the actual implemented state is closer to:

$$
x \approx [u,\ v,\ r,\ \psi,\ \delta,\ \phi]^T
$$

with the returned derivative vector later containing seven derivatives. This mismatch is one reason why `Blue02.m` was treated as a useful reference rather than a direct final model.

### 3.2 Physical constants and Bluefin vessel parameters

The MATLAB model defines water density, rudder limits, vessel mass, added-mass terms, yaw inertia, length, draft, and wetted surface quantities:

```matlab
rho = 1000;
rhoA= 0.1250;
g   = 9.81;
del_max  = 40*pi/180;
deld_max = 20*pi/180;

m    = 64.55;
mx   = 3.662;
my   = 62.7366;
Iz   = 9.6038;
Jz   = 0.6309;
IzJz = 10.2347;
L  = 1.725;
d  = 0.193;
Sw = 0.7614;
```

These values define the physical scale of the vessel. In Fossen's notation, the total inertia is interpreted as the sum of rigid-body inertia and hydrodynamic added inertia:

$$
M = M_{RB}+M_A.
$$

The file then constructs the simplified diagonal inertia terms:

```matlab
m11 = (m+mx);
m22 = (m+my);
m33 = IzJz;
```

which correspond to:

$$
m_{11}=m+m_x,
\qquad
m_{22}=m+m_y,
\qquad
m_{33}=I_z+J_z.
$$

Here $m_{11}$ is effective surge inertia, $m_{22}$ is effective sway inertia, and $m_{33}$ is effective yaw inertia. This follows the same physical interpretation as Fossen's added-mass discussion: the vessel must accelerate both its own mass and some surrounding water.

### 3.3 Rudder saturation and actuator dynamics

The MATLAB rudder command is first saturated to the maximum rudder angle:

```matlab
if abs(del_c) >= del_max,
   del_c = sign(del_c)*del_max;
end
```

This corresponds to:

$$
|\delta_c| \le \delta_{\max}.
$$

The rudder angle then follows the command through a rate-limited first-order actuator:

```matlab
del_dot = del_c - del;

if abs(del_dot) >= deld_max,
   del_dot = sign(del_dot)*deld_max;
end
```

The corresponding equation is:

$$
\dot{\delta} = \operatorname{sat}_{\dot{\delta}_{\max}}(\delta_c-\delta).
$$

This matches the actuator modelling idea in Fossen's actuator chapters: the control system commands an actuator, but the vessel responds to the actual actuator state, not an ideal instantaneous command.

### 3.4 Non-dimensional velocity variables

The model computes the resultant speed, then forms non-dimensional sway and yaw variables:

```matlab
U = sqrt(u*u+v*v);
vd = v/U;
rd = r*L/U;
```

The corresponding definitions are:

$$
U = \sqrt{u^2+v^2},
$$

$$
v_d = \frac{v}{U},
\qquad
r_d = \frac{rL}{U}.
$$

These non-dimensional variables are common in manoeuvring models because they express lateral velocity and yaw rate relative to the vessel speed and length.

### 3.5 Hull surge, sway, and yaw contributions

The MATLAB hull surge force is:

```matlab
XH = m*v*r-1/2*rho*U^2*L*d*Sw/(L*d)*0.4631/(log(4*10^7)^2.6)*u^2+...
     Xvv*v^2+Xvr*v*r+Xrr*r^2;
```

This can be interpreted as:

$$
X_H = mvr + X_{resistance}(u,U) + X_{vv}v^2 + X_{vr}vr + X_{rr}r^2.
$$

The first term $mvr$ is a velocity-coupling term. The resistance term is a speed-dependent hull drag approximation. The remaining polynomial terms are empirical manoeuvring derivatives.

The sway force is computed as:

```matlab
YH = 0.5*rho*L*d*U^2*(Yv*vd+Yvr*vd*abs(rd)+Yr*rd+...
     Yvv*abs(vd)*vd+Yrr*rd*abs(rd));
```

which corresponds to:

$$
Y_H = \frac{1}{2} \rho L d U^2
\left(
\begin{aligned}
Y_{vv\_d} + Y_{vr} v d |r_d| + Y_{rr\_d} + Y_{vv} v_d |v_d| + Y_{rr} r_d |r_d|
\end{aligned}
\right)
$$

The yaw moment is computed as:

```matlab
NH = 0.5*rho*(L^2)*d*U^2*(Nv*vd+Nvr*abs(vd)*rd+Nr*rd+Nvv*vd*abs(vd)...
     +Nrr*rd*abs(rd));
```

which corresponds to:

$$
N_H = \frac{1}{2}\rho L^2 d U^2
\left(
\begin{aligned}
N_vv_d + N_{vr}|v_d|r_d + N_rr_d + N_{vv}v_d|v_d| + N_{rr}r_d|r_d|
\end{aligned}
\right)
$$

These are component-wise versions of the hydrodynamic damping and manoeuvring-derivative terms in Fossen's general marine craft equation:

$$
M\dot{\nu}+C(\nu)\nu+D(\nu)\nu+g(\eta)=\tau
$$

In `Blue02.m`, the damping is not written as a matrix $D(\nu)$, but as explicit empirical force and moment expressions.

### 3.6 Propeller surge force

The propeller block is:

```matlab
KT = 0.25; J=0.072;
XP = (1-tp)*rho*(n1^2)*Dp^4*KT*J;
YP = 0;
NP = 0;
```

This gives the propeller contribution:

$$
X_P = (1-t_P)\rho n_1^2D_P^4K_TJ
$$

The model assumes the propeller mainly contributes surge force:

$$
Y_P = 0,
\qquad
N_P = 0
$$

This is consistent with the actuator interpretation in Fossen: a main propeller is primarily a longitudinal actuator, producing thrust in the vessel's body-fixed surge direction.

### 3.7 Rudder inflow and rudder normal force

The MATLAB rudder inflow approximation is:

```matlab
uR = 0.856113*u*sqrt(1+6.3*(1-(0.856113*u)/(0.00717*1000))^1.5);
```

This defines an effective rudder inflow speed:

$$
u_R = 0.856113u
\sqrt{1+6.3\left(1-\frac{0.856113u}{0.00717\cdot1000}\right)^{1.5}}
$$

The rudder normal force is then:

```matlab
FN = -1/2*10^3*0.0091*2.6927*u^2*sin(del+(0.603463/uR)*(v-1.5525*r));
```

which corresponds to:

$$
F_N = -\frac{1}{2}\rho A_R f_{\alpha}u^2
\sin\left(
\delta + \frac{0.603463}{u_R}(v-1.5525r)
\right)
$$

The term inside the sine is an effective rudder angle of attack: it combines actual rudder deflection with lateral inflow due to sway and yaw.

### 3.8 Rudder force resolution into surge, sway, and yaw

The MATLAB code resolves the rudder normal force into body-force and yaw-moment components:

```matlab
XR = -(1-0.449821)*FN*sin(del);
YR = -(1+0.443853)*FN*cos(del);
NR = -(0.646875+0.443853*0.7569)*FN*cos(del);
```

These correspond to:

$$
X_R = -(1-t_R)F_N\sin\delta
$$

$$
Y_R = -(1+a_H)F_N\cos\delta
$$

$$
N_R = -(x_R+a_Hx_H)F_N\cos\delta
$$

This is exactly the actuator-moment-arm concept from Fossen's actuator discussion: a rudder generates a lateral force and the moment arm between that force and the centre of gravity generates a yaw moment.

### 3.9 Total forces and moments

The MATLAB code sums the force contributions:

```matlab
X = XH + XP + XR;
Y = YH + YP + YR;
N = NH + NP + NR;
```

which corresponds to:

$$
X = X_H+X_P+X_R
$$

$$
Y = Y_H+Y_P+Y_R
$$

$$
N = N_H+N_P+N_R
$$

This maps directly to the generalized force vector $\tau$ in Fossen's notation.

### 3.10 Returned derivatives

The MATLAB derivative vector is:

```matlab
xdot = [X*2/(m11*(rho*L*L*U*U))
        Y*2/(m22*(rho*L*L*U*U))
        1/(Iz+Jz)*(N)
        r/L*U
        del_dot
        u*cos(psi)-v*cos(phi)*sin(psi)
        u*sin(psi)-v*cos(phi)*cos(phi)];
```

The first three entries are surge, sway, and yaw acceleration-like terms:

$$
\dot{u} = \frac{2X}{m_{11}\rho L^2U^2}
$$

$$
\dot{v} = \frac{2Y}{m_{22}\rho L^2U^2}
$$

$$
\dot{r} = \frac{N}{I_z+J_z}
$$

The heading derivative is:

$$
\dot{\psi}=\frac{r}{L}U
$$

The rudder derivative is:

$$
\dot{\delta}=\operatorname{sat}_{\dot{\delta}_{\max}}(\delta_c-\delta)
$$

The planar kinematics are:

$$
\dot{x}=u\cos\psi-v\cos\phi\sin\psi
$$

$$
\dot{y}=u\sin\psi-v\cos\phi\cos\phi
$$

The final $\dot{y}$ expression appears unusual because it contains $\cos(\phi)\cos(\phi)$. This is another sign that `Blue02.m` should be treated as an intermediate reference rather than the cleanest final source.

### 3.11 Why `Blue02.m` alone was not enough

`Blue02.m` was useful, but it had several limitations:

1. The state comments and implementation are not fully consistent.
2. Roll is mentioned but not actually included as a proper dynamic state.
3. It does not include a consistent bow-thruster state.
4. It is less complete than `Bluefin4DOFModel02.m`.
5. It does not provide enough independent tuning freedom to match both straight-line and turning performance.

## 4. `ship_model_bluefin_v2.py`: how it was constructed from the 3-DOF theory

`ship_model_bluefin_v2.py` is not a literal copy of `Blue02.m`. It is a **control-oriented 3-DOF model** inspired by Bluefin constants and by the 3-DOF theory.

### 4.1 State and public interface

The Python state is:

$$
x = [u,\ v,\ r,\ \psi,\ \delta,\ x,\ y]^T
$$

In the code, this is implemented by:

- `reset()` and stored fields
```python
def reset(self) -> None:
    self._v = 0.0
    self._v_sway = 0.0
    self._w = 0.0
    self._h = 0.0
    self._delta = 0.0
    self._x = 0.0
    self._y = 0.0
```
- `state_dict()`
```python
def state_dict(self) -> Dict[str, float]:
    return {
        "u_body_mps": float(self._v),
        "v_body_mps": float(self._v_sway),
        "yaw_rate_radps": float(self._w),
        "yaw_rate_degps": float(math.degrees(self._w)),
        "heading_rad": float(self._h),
        "heading_deg": float(math.degrees(self._h) % 360.0),
        "rudder_deg": float(math.degrees(self._delta)),
        "x_m": float(self._x),
        "y_m": float(self._y),
        "speed_mps": float(math.hypot(self._v, self._v_sway)),
    }
```
- `_state_vector()` and `_set_state_vector()`
```python
def _state_vector(self) -> np.ndarray:
    return np.array([
        self._v, self._v_sway, self._w, self._h, self._delta, self._x, self._y
    ], dtype=float)

def _set_state_vector(self, s: np.ndarray) -> None:
    self._v = float(s[0])
    self._v_sway = float(s[1])
    self._w = float(s[2])
    self._h = float(s[3])
    self._delta = float(s[4])
    self._x = float(s[5])
    self._y = float(s[6])
```

The public interface is:

```python
dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)
```

It keeps the same input/output format as the older Gym environment while hiding the internal dynamics.

### 4.2 RK4 integration

The model uses Runge-Kutta integration:

$$
x_{k+1}=x_k+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4)
$$

```python
k1 = self._derivatives(s0, rpm, rud, thruster_rpm)
k2 = self._derivatives(s0 + 0.5 * dt * k1, rpm, rud, thruster_rpm)
k3 = self._derivatives(s0 + 0.5 * dt * k2, rpm, rud, thruster_rpm)
k4 = self._derivatives(s0 + dt * k3, rpm, rud, thruster_rpm)
s1 = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
```

This is a numerical improvement over a simple Euler step. It was used because the nonlinear rudder and thrust equations can be stiff enough that Euler integration gives unstable or inaccurate results at the 10 Hz simulation rate.

### 4.3 Speed-shaped propeller law

In v2, the propeller force is implemented as:

$$
X_P = (1-t_P)K_T n|n|
\frac{1+k_b e^{-u/U_b}}{1+k_d u^2}
$$

```python
def _propeller_force(self, rpm: float, u_eff: float) -> float:
    n = max(rpm, 0.0)
    static_term = THRUST_COEF * n * abs(n)
    low_speed_boost = 1.0 + THRUST_LOW_SPEED_BOOST * math.exp(-u_eff / max(THRUST_BOOST_U0, 1e-6))
    high_speed_decay = 1.0 / (1.0 + THRUST_HIGH_SPEED_DECAY * u_eff * u_eff)
    return (1.0 - TP) * static_term * low_speed_boost * high_speed_decay
```

This was added because a simple constant $rpm^2$ thrust law could not match both early acceleration and the final speed envelope. The idea is consistent with Fossen's forward-speed equation, where surge dynamics include both control force and linear/quadratic resistance. 

### 4.4 Hull damping and manoeuvring derivatives

The v2 model computes hull surge damping and sway/yaw damping. 

```python
# Hull surge force
cf = self._safe_log_cf()
x_visc = -DRAG_COEF * 0.5 * rho * SW * cf * u_eff * abs(u_eff)
x_cross = -DRAG_COEF * 0.5 * rho * L * DRAFT * U * U * (
    XVV * (math.sin(beta) ** 2) + XVR * abs(math.sin(beta)) * abs(r_nd) + XRR * (r_nd ** 2)
)
x_lin = -LINEAR_SURGE_DAMP * u_eff
x_hull = x_visc + x_cross + x_lin

# Hull sway / yaw damping
y_hull = -TURN_COEF * (
    0.5 * rho * L * DRAFT * U * U * (
        YV * beta + YVV * abs(beta) * beta + YR * r_nd + YRR * abs(r_nd) * r_nd + YVR * beta * abs(r_nd)
    ) + LINEAR_SWAY_DAMP * v
)
n_hull = -TURN_COEF * (
    0.5 * rho * (L ** 2) * DRAFT * U * U * (
        NV * beta + NVV * abs(beta) * beta + NR * r_nd + NRR * abs(r_nd) * r_nd + NVR * abs(beta) * r_nd
    ) + LINEAR_YAW_DAMP * r
)
```

These terms are the Python equivalent of the MATLAB hull-polynomial terms, but simplified and tuned.

The form follows the idea that hull resistance and manoeuvring forces are velocity-dependent, as discussed in Fossen's hydrodynamic damping section.

### 4.5 Rudder split: axial loss, sway force, yaw moment

The v2 rudder block implemented:

- rudder inflow $u_R,v_R$
- angle of attack $\alpha_R$
- rudder normal force $F_N$
- axial drag $X_R$
- sway force $Y_R$
- yaw moment $N_R$

```python
# Rudder inflow and normal force
n_prop = max(rpm, 0.0) / 60.0
u_r = max(MIN_FLOW_SPEED, (1.0 - WR) * u_eff + 0.6 * KX * n_prop)
v_r = v + L_R * r
alpha_r = delta - math.atan2(v_r, u_r)

f_n = RUDDER_FORCE_SCALE * 0.5 * rho * AR * FALP * (u_r * u_r + v_r * v_r) * math.sin(alpha_r)

# Split rudder effect into axial loss, sway, and yaw separately.
x_rud = -RUDDER_X_DRAG_SCALE * abs(f_n) * abs(math.sin(delta))
y_rud = -(1.0 + AH) * f_n * math.cos(delta)
rudder_arm = abs(X_RUDDER + AH * X_HULL)
n_rud = -RUDDER_YAW_SCALE * rudder_arm * f_n * math.cos(delta)

x_total = x_hull + x_prop + x_rud
y_total = y_hull + y_rud
n_total = n_hull + n_rud + n_thr_moment
```

The key v2 modelling decision was to separate:

$$
X_R,\quad Y_R,\quad N_R
$$

instead of forcing all rudder effects to share one coefficient. This follows the physical idea that rudder forces create both lateral force and yaw moment, while also reducing forward speed. Fossen's actuator section supports this interpretation by describing rudders as lateral-force devices that generate steering yaw moments.

## 5. `Bluefin4DOFModel02.m`: MATLAB code, equations, and theory mapping

`Bluefin4DOFModel02.m` is a more complete model than `Blue02.m`. It is based on a Japanese-style 4DOF manoeuvring model that includes surge, sway, roll, and yaw. It also includes actuator states for rudder angle, main propeller speed, and bow-thruster speed.

### 5.1 State vector and input vector

The MATLAB file documents the full state vector:

```matlab
% x(1)=u        = surge velocity          (m/s)
% x(2)=v        = sway velocity           (m/s)
% x(3)=p        = roll rate               (rad/s)
% x(4)=r        = yaw velocity            (rad/s)
% x(5)=x        = position in x-direction (m)
% x(6)=y        = position in y-direction (m)
% x(7)=phi      = roll angle              (rad)
% x(8)=psi      = yaw angle               (rad)
% x(9)=delta    = actual rudder angle     (rad)
% x(10)=n1      = propeller               (rps)
% x(11)=n2      = bow thruster            (rps)
```

The corresponding state vector is:

$$
x=[u,\ v,\ p,\ r,\ x,\ y,\ \phi,\ \psi,\ \delta,\ n_1,\ n_2]^T
$$

The input vector is documented as:

```matlab
% ui      = [ delta_c n1_c n2_c]'
% delta_c = commanded rudder angle          (rad)
% n1_c    = commanded shaft velocity vector (rpm)
% n2_c    = commanded thruster velocity     (rpm)
```

The corresponding input vector is:

$$u_c=[\delta_c,\ n_{1c},\ n_{2c}]^T
$$

The file verifies these dimensions explicitly:

```matlab
if (length(x) ~= 11),error('x-vector must have dimension 11 !');end
if (length(ui) ~= 3),error('u-vector must have dimension  3 !');end
```

This structure is much closer to Fossen's general marine-craft formulation, because it keeps actuator states as part of the dynamic system rather than treating commands as instantaneous force inputs.

### 5.2 Speed, drift angle, and non-dimensional states

The MATLAB file computes vessel length, resultant speed, and drift angle:

```matlab
L = 1.725;                     % length of ship (m)
U = sqrt(x(1)^2 + x(2)^2);     % service speed (m/s)

b = -asin(x(2)/U);
```

This corresponds to:

$$
U=\sqrt{u^2+v^2}
$$

$$
\beta=-\sin^{-1}\left(\frac{v}{U}\right)
$$

The code then forms dimensional and non-dimensional variables:

```matlab
ud   = x(1)/U;     u = x(1);
vd   = x(2)/U;     v = x(2);
pd   = x(3)*L/U;   p = x(3);
rd   = x(4)*L/U;   r = x(4);
phi = x(7);
psi = x(8);
delta = x(9);
n1   = x(10);
n2   = x(11);
```

The corresponding definitions are:

$$
u_d=\frac{u}{U},\qquad v_d=\frac{v}{U},\qquad
p_d=\frac{pL}{U},\qquad r_d=\frac{rL}{U}
$$

These non-dimensional quantities are used in the empirical hull, rudder, and roll terms.

### 5.3 Inputs and actuator commands

The MATLAB input conversion is:

```matlab
delta_c = ui(1);
n1_c    = ui(2)/60;  % n1_c in rps
n2_c    = ui(3)/60;  % n2_c in rps
```

This corresponds to:

$$
\delta_c=ui_1,
\qquad
n_{1c}=\frac{ui_2}{60},
\qquad
n_{2c}=\frac{ui_3}{60}
$$

The conversion by 60 changes rpm into revolutions per second.

### 5.4 Effective inertia terms

The MATLAB file defines rigid-body mass, added-mass values, and moments of inertia:

```matlab
m = 64.55;
mx = 3.662;
my = 62.7366;
Ix = 0.567;
Iz = 9.6038;
Jx = 0.6309;
Jz = 10.2347;

m11 = (m+mx);
m22 = (m+my);
m33 = (Ix+Jx);
m44 = (Iz+Jz);
```

These correspond to:

$$
m_{11}=m+m_x,
\qquad
m_{22}=m+m_y,
\qquad
m_{33}=I_x+J_x,
\qquad
m_{44}=I_z+J_z
$$

This is the component-wise equivalent of Fossen's $M=M_{RB}+M_A$, extended from 3DOF to surge, sway, roll, and yaw.

### 5.5 Rudder and shaft actuator dynamics

The MATLAB file limits the commanded rudder angle:

```matlab
if abs(delta_c) >= delta_max*pi/180,
   delta_c = sign(delta_c)*delta_max*pi/180;
end
```

which corresponds to:

$$
|\delta_c|\le \delta_{\max}
$$

It then computes the rate-limited actual rudder derivative:

```matlab
delta_dot = delta_c - delta;

if abs(delta_dot) >= Ddelta_max*pi/180,
   delta_dot = sign(delta_dot)*Ddelta_max*pi/180;
end
```

which corresponds to:

$$
\dot{\delta}=\operatorname{sat}_{\dot{\delta}_{\max}}(\delta_c-\delta)
$$

The shaft-speed dynamics are:

```matlab
n1_dot = n1s - n1;
n2_dot = n2s - n2;

if abs(n1_dot) >= Nc_max,
   n1_dot = sign(n1_dot)*Nc_max;
end
if abs(n2_dot) >= Nc_max,
   n2_dot = sign(n2_dot)*Nc_max;
end
```

which correspond to:

$$
\dot{n}_1=\operatorname{sat}_{N_{c\max}}(n_{1c}-n_1),
\qquad
\dot{n}_2=\operatorname{sat}_{N_{c\max}}(n_{2c}-n_2).
$$

This actuator-state formulation is important because it gives the vessel a finite response time to commanded rudder and shaft-speed changes.

### 5.6 Propeller advance-ratio model

The propeller block is:

```matlab
J = onew*u/(n1s*DPs);
a0 = 0.3267; a1 = -0.2297; a2 = -0.1607;
KT = a0+a1*J+a2*J^2;
XdP = abs(n1s)*n1s*onet*KT*(DPs^4)/(0.5*L*d*U^2);
```

The corresponding advance ratio is:

$$
J=\frac{(1-w)u}{n_1D_P}
$$

The thrust coefficient is:

$$
K_T=a_0+a_1J+a_2J^2.
$$

The non-dimensional propeller surge contribution is:

$$
X_{dP}=\frac{|n_1|n_1(1-t)K_TD_P^4}{0.5LdU^2}
$$

This is more physically structured than a simple $rpm^2$ thrust law because the propeller force depends on both shaft speed and the vessel's forward speed through $J$.

### 5.7 Rudder inflow, angle of attack, and normal force

The MATLAB rudder inflow block is:

```matlab
udR = epsi*(onew)*sqrt(eta*((1+kappa*sqrt(1+8*KT/(pi*J^2))-1)^2)+(1-eta));
vdR = -gR*(b-ldR*rd+(p*(zR-zG)/U));
UdR = sqrt(udR^2+vdR^2);
alphaR  = delta-atan2(-vdR,udR);
FdN = -(ARpLd)*(6.13*lambda/(2.25+lambda))*UdR^2*sin(alphaR);
```

The corresponding equations are:

$$u_{dR}=\epsilon(1-w)
\sqrt{\eta\left(1+\kappa\sqrt{1+\frac{8K_T}{\pi J^2}}-1\right)^2+(1-\eta)}
$$

$$
v_{dR}=-\gamma_R\left(\beta-l_{dR}r_d+\frac{p(z_R-z_G)}{U}\right)
$$

$$
U_{dR}=\sqrt{u_{dR}^2+v_{dR}^2}
$$

$$
\alpha_R=\delta-\tan^{-1}\left(\frac{-v_{dR}}{u_{dR}}\right)
$$

$$
F_{dN}=-\left(\frac{A_R}{Ld}\right)
\left(\frac{6.13\lambda}{2.25+\lambda}\right)
U_{dR}^2\sin\alpha_R
$$

This is the rudder-force equivalent of Fossen's actuator-force explanation. The rudder produces a hydrodynamic force based on inflow and deflection; that force is then resolved into forces and moments.

### 5.8 Rudder force and moment resolution

The MATLAB code resolves the rudder normal force as:

```matlab
XdR = (onetR)*FdN*sin(delta)*cos(phi);
YdR = (1+aH)*FdN*cos(delta)*cos(phi);
KdR = zR*YdR/L;
NdR = (xR+aH*xH)*FdN*cos(delta)*cos(phi);
```

The corresponding equations are:

$$
X_{dR}=(1-t_R)F_{dN}\sin\delta\cos\phi
$$

$$
Y_{dR}=(1+a_H)F_{dN}\cos\delta\cos\phi
$$

$$
K_{dR}=\frac{z_RY_{dR}}{L}
$$

$$
N_{dR}=(x_R+a_Hx_H)F_{dN}\cos\delta\cos\phi
$$

The surge term represents rudder-induced drag/thrust effect, the sway term is lateral rudder force, the roll term is caused by vertical offset, and the yaw moment comes from the rudder moment arm.

### 5.9 Hull hydrodynamic forces and moments

The MATLAB hull force block is:

```matlab
XdH = Xd0*(1+cx0*abs(phi))+Xdrph*rd*phi+Xdbb*(1+cxbb*abs(phi))*b^2+...
      Xdbrmdy*b*rd+Xdrr*(1+cxrr*abs(phi))*rd^2+Xdbbbb*b^4;
YdH = Ydph*phi+Ydb*(1+cyb*abs(phi))*b+Ydrmdx*(1+cyr*abs(phi))*rd+...
      Ydbbph*b^2*phi+Ydbrph*b*rd*phi+Ydrrph*rd^2*phi+Ydbbb*b^3+...
      Ydbbr*b^2*rd+Ydbrr*b*rd^2+Ydrrr*rd^3;
KdH = Kdph*phi+Kdb*b+Kdr*rd+Kdbbph*b^2*phi+Kdbrph*b*rd*phi+...
      Kdrrph*rd^2*phi+Kdbbb*b^3+Kdbbr*b^2*rd+Kdbrr*b*rd^2+Kdrrr*rd^3;
NdH = Ndph*phi+Ndb*(1+cnb*abs(phi))*b+Ndr*(1+cnr*abs(phi))*rd+...
      Ndbbph*b^2*phi+Ndbrph*b*rd*phi+Ndrrph*rd^2*phi+Ndbbb*b^3+...
      Ndbbr*b^2*rd+Ndbrr*b*rd^2+Ndrrr*rd^3;
```

This corresponds to nonlinear polynomial manoeuvring derivatives:

$$
X_{dH}=f_X(\beta,r_d,\phi)
$$

$$
Y_{dH}=f_Y(\beta,r_d,\phi)
$$

$$
K_{dH}=f_K(\beta,r_d,\phi)
$$

$$
N_{dH}=f_N(\beta,r_d,\phi)
$$

In Fossen's compact notation, these terms belong to the hydrodynamic damping and restoring structure. In this MATLAB file they are expanded explicitly using empirical coefficients from the Japanese manoeuvring model.

### 5.10 Roll restoring and roll damping

The MATLAB roll restoring and damping block is:

```matlab
C44  = g*m*GM;
a    = 0.5;
B44  = 2*a/pi*sqrt(g*m*GM*(Ix+Jx));
KdH2 = zG*YdH-B44*p-C44*phi-(zR-zG)*YdR;
```

The restoring coefficient is:

$$
C_{44}=g m GM
$$

The roll damping approximation is:

$$
B_{44}=\frac{2a}{\pi}\sqrt{gmGM(I_x+J_x)}
$$

The additional roll moment is:

$$
K_{dH2}=z_GY_{dH}-B_{44}p-C_{44}\phi-(z_R-z_G)Y_{dR}
$$

This follows Fossen's general restoring-force concept: gravity and buoyancy generate restoring moments that depend on displacement from equilibrium. Here the restoring effect is applied to roll.

### 5.11 Bow-thruster force and moments

The MATLAB bow-thruster block is:

```matlab
Dbt = 0.033;
xB = 0.45;
zB = -0.05;
KBT = 0.026;
FBT = abs(n2s)*n2s*KBT/(0.5*rho*L*d*U^2);
YdB = FBT;
KdB = zB*FBT/L;
NbB = xB*FBT/L;
```

The corresponding bow-thruster force is:

$$
F_{BT}=\frac{|n_2|n_2K_{BT}}{0.5\rho LdU^2}
$$

It contributes to sway, roll, and yaw as:

$$
Y_{dB}=F_{BT}
$$

$$
K_{dB}=\frac{z_BF_{BT}}{L}
$$

$$
N_{dB}=\frac{x_BF_{BT}}{L}
$$

This corresponds to Fossen's actuator-allocation idea: a transverse thruster produces a lateral force, and that force creates roll and yaw moments through its moment arms.

### 5.12 Total forces and moments

The MATLAB model sums all components:

```matlab
Xd = XdH + XdP + XdR;
Yd = YdH + YdR + YdB;
Kd = KdH+KdH2+KdR+KdB;
Nd = NdH + NdR -xG*Yd + NbB;
```

which correspond to:

$$
X_d=X_{dH}+X_{dP}+X_{dR}
$$

$$
Y_d=Y_{dH}+Y_{dR}+Y_{dB}
$$

$$
K_d=K_{dH}+K_{dH2}+K_{dR}+K_{dB}
$$

$$
N_d=N_{dH}+N_{dR}-x_GY_d+N_{dB}
$$

This is the explicit component-wise version of the generalized force vector $\tau$ in Fossen's marine-craft equation.

### 5.13 Surge, sway, roll, and yaw accelerations

The MATLAB file first computes sway acceleration because it is reused in the roll equation:

```matlab
vdot = (Yd*(0.5*rho*L*d*U^2)-m11*u*r)/m22;
```

This corresponds to:

$$
\dot{v}=\frac{Y_d(0.5\rho LdU^2)-m_{11}ur}{m_{22}}
$$

The final derivative vector is:

```matlab
xdot = [(Xd*(0.5*rho*L*d*U^2)+m22*v*r)/m11
        (Yd*(0.5*rho*L*d*U^2)-m11*u*r)/m22
        (Kd*(0.5*rho*L*d^2*U^2)+(zH-zG)*(my*vdot+mx*u*r))/m33
        Nd*(0.5*rho*L^2*d*U^2)/m44
        cos(psi)*u-sin(psi)*v*cos(phi)
        sin(psi)*u+cos(psi)*v*cos(phi)
        p
        r*cos(phi)
        delta_dot
        n1_dot
        n2_dot];
```

The first four equations are:

$$
\dot{u}=\frac{X_d(0.5\rho LdU^2)+m_{22}vr}{m_{11}}
$$

$$
\dot{v}=\frac{Y_d(0.5\rho LdU^2)-m_{11}ur}{m_{22}}
$$

$$
\dot{p}=\frac{K_d(0.5\rho Ld^2U^2)+(z_H-z_G)(m_y\dot{v}+m_xur)}{m_{33}}
$$

$$
\dot{r}=\frac{N_d(0.5\rho L^2dU^2)}{m_{44}}
$$

The earth-fixed kinematics are:

$$
\dot{x}=u\cos\psi-v\sin\psi\cos\phi
$$

$$
\dot{y}=u\sin\psi+v\cos\psi\cos\phi
$$

$$
\dot{\phi}=p
$$

$$
\dot{\psi}=r\cos\phi
$$

The remaining actuator-state derivatives are:

$$
\dot{\delta},\qquad \dot{n}_1,\qquad \dot{n}_2
$$

This is the main reason `Bluefin4DOFModel02.m` is a stronger modelling source than `Blue02.m`: it is a complete dynamic system with surge, sway, roll, yaw, position, heading, rudder state, propeller state, and bow-thruster state.

## 6. `bluefin_4dof_final.py`: how the Python file maps to the 4-DOF theory

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

- `reset()`
```python
def reset(self) -> None:
    self._u = 0.0
    self._v = 0.0
    self._p = 0.0
    self._r = 0.0
    self._x = 0.0
    self._y = 0.0
    self._phi = 0.0
    self._psi = 0.0
    self._delta = 0.0
    self._n1 = 0.0
    self._n2 = 0.0

    self._v_sway = 0.0
    self._w = 0.0
    self._h = 0.0
```
- `state_dict()`
```python
def state_dict(self) -> Dict[str, float]:
    return {
        "u_body_mps": float(self._u),
        "v_body_mps": float(self._v),
        "roll_rate_radps": float(self._p),
        "yaw_rate_radps": float(self._r),
        "yaw_rate_degps": float(math.degrees(self._r)),
        "roll_deg": float(math.degrees(self._phi)),
        "heading_rad": float(self._psi),
        "heading_deg": float(math.degrees(self._psi) % 360.0),
        # Public convention: negative rudder = port, positive = starboard.
        "rudder_deg": float(-math.degrees(self._delta)),
        "prop_rps": float(self._n1),
        "thruster_rps": float(self._n2),
        "x_m": float(self._x),
        "y_m": float(self._y),
    }
```
- `_state_vector()`
```python
def _state_vector(self) -> np.ndarray:
    return np.array(
        [
            self._u,
            self._v,
            self._p,
            self._r,
            self._x,
            self._y,
            self._phi,
            self._psi,
            self._delta,
            self._n1,
            self._n2,
        ],
        dtype=float,
    )
```
- `_set_state_vector()`
```python
def _set_state_vector(self, s: np.ndarray) -> None:
    self._u = float(s[0])
    self._v = float(s[1])
    self._p = float(s[2])
    self._r = float(s[3])
    self._x = float(s[4])
    self._y = float(s[5])
    self._phi = float(s[6])
    self._psi = float(s[7])
    self._delta = float(s[8])
    self._n1 = float(s[9])
    self._n2 = float(s[10])

    self._v_sway = self._v
    self._w = self._r
    self._h = self._psi
```

The compatibility fields `_v_sway`, `_w`, and `_h` make the 4DOF model compatible with scripts written for older models.

### 6.3 Input mapping and sign convention

The Python `update()` method maps the public control command into MATLAB-style internal inputs:

- Rudder percentage becomes internal rudder angle;
- Repo-facing rpm becomes MATLAB command rpm;
- Optional bow-thruster command becomes MATLAB command rpm.

```python
delta_cmd = float(-np.clip(rud, -100.0, 100.0)) / 100.0 * math.radians(MAX_RUD_ANGLE)
n1_cmd_rpm = max(float(rpm), 0.0) * RPM_COMMAND_SCALE
n2_cmd_rpm = float(thruster_rpm) * THRUSTER_COMMAND_SCALE
```

This means the public repo convention can stay consistent even if the MATLAB internal sign is opposite. This is important because Fossen notes that rudder sign convention is model-defined; what matters is consistency between rudder and yaw rate.

### 6.4 RK4 integration and guards

The Python code uses RK4.

```python
k1 = self._derivatives(s0, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
k2 = self._derivatives(s0 + 0.5 * dt * k1, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
k3 = self._derivatives(s0 + 0.5 * dt * k2, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
k4 = self._derivatives(s0 + dt * k3, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
s1 = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
```

It also clips extreme states.

```python
s1[0] = float(np.clip(s1[0], -1.0, MAX_SURGE_SPEED))
s1[1] = float(np.clip(s1[1], -MAX_SWAY_SPEED, MAX_SWAY_SPEED))
s1[2] = float(np.clip(s1[2], -MAX_ROLL_RATE_RAD, MAX_ROLL_RATE_RAD))
s1[3] = float(np.clip(s1[3], -MAX_YAW_RATE_RAD, MAX_YAW_RATE_RAD))
s1[6] = float(np.clip(s1[6], -MAX_ROLL_ANGLE_RAD, MAX_ROLL_ANGLE_RAD))
s1[8] = float(np.clip(s1[8], -math.radians(MAX_RUD_ANGLE), math.radians(MAX_RUD_ANGLE)))
self._set_state_vector(s1)
```

These protections are not part of the theory; they are practical numerical safeguards to make the model usable from rest and in repeated RL/sweep simulations.

### 6.5 Derivative function

The physics is implemented in `_derivatives()`. The block structure is:

**Unpack state**
```python
u, v, p, r, xpos, ypos, phi, psi, delta, n1, n2 = [float(z) for z in s]
```
$$ x=[u,v,p,r,x,y,\phi,\psi,\delta,n_1,n_2]^T $$

**Speed and drift angle**
```python
l_ship = 1.725
u_mag = max(math.hypot(u, v), MIN_FLOW_SPEED)
drift = -math.asin(float(np.clip(v / u_mag, -1.0, 1.0)))
```
$$ U=\sqrt{u^2+v^2},\qquad \beta=-\arcsin(v/U) $$

**Rudder actuator**
```python
delta_dot = float(
    np.clip(
        delta_cmd - delta,
        -math.radians(MAX_RUD_RATE_DPS),
        math.radians(MAX_RUD_RATE_DPS),
    )
)
```
$$ \dot{\delta}=\mathrm{sat}(\delta_c-\delta) $$

**Shaft dynamics**
```python
n1_target = n1_cmd_rpm / 60.0
n2_target = n2_cmd_rpm / 60.0
n1_dot = float(np.clip(n1_target - n1, -MAX_SHAFT_RATE_RPSPS, MAX_SHAFT_RATE_RPSPS))
n2_dot = float(np.clip(n2_target - n2, -MAX_SHAFT_RATE_RPSPS, MAX_SHAFT_RATE_RPSPS))
```
$$ \dot{n}_1,\qquad\dot{n}_2 $$

**Physical constants**
```python
# Parameters from the MATLAB file.
beam = 0.5
draft = 0.193
disp = 0.06455
x_g = -0.1
d_prop = 0.1
lambda_r = 1.4697
eta = 0.879
area_r = 0.0091
area_r_over_ld = area_r / (l_ship * draft)
x_r = -1.05309
gm = 1.87
z_g = 0.005
z_r = -0.01
z_h = 0.02
cf = 1.0

onet = 0.859 * cf
onew = 0.806 * cf
onet_r = 0.857 * cf
one_a_h = 1.403 * cf
a_h = one_a_h - 1.0
x_h = -0.646 * cf
g_r0 = 0.394 * cf
c_g = -0.53 * cf
g_r = g_r0 * (1.0 + c_g * abs(phi)) * cf
ld_r = -0.795 * cf
epsi = 0.740 * cf
kappa = 0.810 * cf
eta_r = 0.140 * cf
```

**Hull coefficients**
```python
xd0 = -0.0212 * cf
cx0 = -0.02 * cf
xdrph = 0.0092 * cf
xdbb = -0.0348 * cf
cxbb = 2.10 * cf
xdbrmdy = -0.0957 * cf
xdrr = -0.0070 * cf
cxrr = 3.74 * cf
xdbbbb = -0.0018 * cf

ydph = 0.0053 * cf
ydb = 0.2501 * cf
cyb = -0.14 * cf
ydrmdx = 0.0346 * cf
cyr = -0.61 * cf
ydbbph = -0.2979 * cf
ydbrph = 0.6308 * cf
ydrrph = -0.0854 * cf
ydbbb = 2.6087 * cf
ydbbr = -1.7091 * cf
ydbrr = 1.1682 * cf
ydrrr = -0.0461 * cf

kdph = -0.0185 * cf
kdb = -0.2586 * cf
kdr = 0.0532 * cf
kdbbph = 0.2229 * cf
kdbrph = 0.5374 * cf
kdrrph = -0.0928 * cf
kdbbb = -0.7293 * cf
kdbbr = 1.1474 * cf
kdbrr = -0.3351 * cf
kdrrr = -0.0132 * cf

ndph = -0.0086 * cf
ndb = 0.0966 * cf
cnb = 0.22 * cf
ndr = -0.0513 * cf
cnr = -0.62 * cf
ndbbph = -0.2510 * cf
ndbrph = 0.0722 * cf
ndrrph = -0.0172 * cf
ndbbb = 0.4218 * cf
ndbbr = -0.8629 * cf
ndbrr = 0.1459 * cf
ndrrr = -0.0439 * cf
```

**Inertia terms**
```python
rho = 1000.0
g = 9.81
m = 64.55
mx = 3.662
my = 62.7366
i_x = 0.567
i_z = 9.6038
j_x = 0.6309
j_z = 10.2347

m11 = m + mx
m22 = m + my
m33 = i_x + j_x
m44 = i_z + j_z
```

$$ m_{11}=m+m_x,\qquad m_{22}=m+m_y, \qquad m_{33}=I_x+J_x, \qquad m_{44}=I_z+J_z $$

**Propeller model**
```python
n1_force = float(np.sign(n1) * max(abs(n1), MIN_ADVANCE_RATIO)) if abs(n1) >= MIN_ADVANCE_RATIO else 0.0
n2_force = float(np.sign(n2) * max(abs(n2), MIN_ADVANCE_RATIO)) if abs(n2) >= MIN_ADVANCE_RATIO else 0.0

if abs(n1_force) < MIN_ADVANCE_RATIO:
    j_adv = 0.0
    kt = 0.0
    xd_p = 0.0
    ud_r = RUDDER_INFLOW_SCALE * epsi * onew
else:
    j_adv = PROPELLER_ADVANCE_SCALE * onew * u / max(abs(n1_force) * d_prop, MIN_ADVANCE_RATIO)
    a0, a1, a2 = 0.3267, -0.2297, -0.1607
    kt = a0 + a1 * j_adv + a2 * j_adv * j_adv
    xd_p = (
        PROPELLER_THRUST_SCALE
        * abs(n1_force)
        * n1_force
        * onet
        * kt
        * (d_prop**4)
        / (0.5 * l_ship * draft * u_mag * u_mag)
    )
    j_sq = max(j_adv * j_adv, MIN_ADVANCE_RATIO * MIN_ADVANCE_RATIO)
    prop_term = max(1.0 + 8.0 * kt / (math.pi * j_sq), 0.0)
    ud_r = RUDDER_INFLOW_SCALE * epsi * onew * math.sqrt(
        eta_r * ((1.0 + kappa * math.sqrt(prop_term) - 1.0) ** 2) + (1.0 - eta_r)
    )
```
Advance-ratio $J$, thrust coefficient $K_T$, propeller surge force

**Rudder model**
```python
vd_r = -g_r * (drift - ld_r * rd + (p * (z_r - z_g) / u_mag))
ud_total_r = math.hypot(ud_r, vd_r)
alpha_r = delta - math.atan2(-vd_r, ud_r)
fd_n = -(area_r_over_ld) * (6.13 * lambda_r / (2.25 + lambda_r)) * (ud_total_r**2) * math.sin(alpha_r)

xd_r = onet_r * fd_n * math.sin(delta) * math.cos(phi)
yd_r = RUDDER_FORCE_SCALE * (1.0 + a_h) * fd_n * math.cos(delta) * math.cos(phi)
kd_r = z_r * yd_r / l_ship
nd_r = RUDDER_FORCE_SCALE * RUDDER_YAW_SCALE * (x_r + a_h * x_h) * fd_n * math.cos(delta) * math.cos(phi)
```
$$ \alpha_R,F_N,X_R,Y_R,K_R,N_R $$

**Hull forces/moments**
```python
xd_h = (
    xd0 * (1.0 + cx0 * abs(phi))
    + xdrph * rd * phi
    + xdbb * (1.0 + cxbb * abs(phi)) * drift * drift
    + xdbrmdy * drift * rd
    + xdrr * (1.0 + cxrr * abs(phi)) * rd * rd
    + xdbbbb * drift**4
)
yd_h = (
    ydph * phi
    + ydb * (1.0 + cyb * abs(phi)) * drift
    + ydrmdx * (1.0 + cyr * abs(phi)) * rd
    + ydbbph * drift * drift * phi
    + ydbrph * drift * rd * phi
    + ydrrph * rd * rd * phi
    + ydbbb * drift**3
    + ydbbr * drift * drift * rd
    + ydbrr * drift * rd * rd
    + ydrrr * rd**3
)
kd_h = (
    kdph * phi
    + kdb * drift
    + kdr * rd
    + kdbbph * drift * drift * phi
    + kdbrph * drift * rd * phi
    + kdrrph * rd * rd * phi
    + kdbbb * drift**3
    + kdbbr * drift * drift * rd
    + kdbrr * drift * rd * rd
    + kdrrr * rd**3
)
nd_h = (
    ndph * phi
    + ndb * (1.0 + cnb * abs(phi)) * drift
    + ndr * (1.0 + cnr * abs(phi)) * rd
    + ndbbph * drift * drift * phi
    + ndbrph * drift * rd * phi
    + ndrrph * rd * rd * phi
    + ndbbb * drift**3
    + ndbbr * drift * drift * rd
    + ndbrr * drift * rd * rd
    + ndrrr * rd**3
)
```
$$ X_{dH},Y_{dH},K_{dH},N_{dH} $$

**Roll restoring/damping**
```python
c44 = g * m * gm
damping_a = 0.5
b44 = 2.0 * damping_a / math.pi * math.sqrt(max(g * m * gm * (i_x + j_x), 0.0))
kd_h2 = z_g * yd_h - (z_r - z_g) * yd_r

roll_damping_moment = -ROLL_DAMP_SCALE * b44 * p
roll_restoring_moment = -ROLL_RESTORE_SCALE * c44 * phi
pdot = (
    kd * force_scale_k
    + roll_damping_moment
    + roll_restoring_moment
    + (z_h - z_g) * (my * vdot + mx * u * r)) / m33
```
$$ C_{44}=g\,m\,GM $$

**Bow thruster**
```python
x_b = 0.45
z_b = -0.05
k_bt = 0.026
f_bt = (
    BOW_THRUSTER_SCALE
    * abs(n2_force)
    * n2_force
    * k_bt
    / (0.5 * rho * l_ship * draft * u_mag * u_mag)
)
yd_b = f_bt
kd_b = z_b * f_bt / l_ship
nb_b = x_b * f_bt / l_ship
```
$$ Y_B,K_B,N_B $$

**Total forces/moments**
```python
xd = xd_h + xd_p + RUDDER_X_DRAG_SCALE * xd_r
yd = yd_h + yd_r + yd_b
kd = kd_h + kd_h2 + kd_r + kd_b
nd = nd_h + nd_r - x_g * yd + nb_b
```
$$ X,Y,K,N $$

**Accelerations**
```python
vdot = (yd * force_scale_x - m11 * u * r) / m22
surge_linear_force = -LINEAR_SURGE_DAMP * u
yaw_linear_moment = -LINEAR_YAW_DAMP * r
udot = (xd * force_scale_x + surge_linear_force + m22 * v * r) / m11
roll_damping_moment = -ROLL_DAMP_SCALE * b44 * p
roll_restoring_moment = -ROLL_RESTORE_SCALE * c44 * phi
pdot = (
    kd * force_scale_k
    + roll_damping_moment
    + roll_restoring_moment
    + (z_h - z_g) * (my * vdot + mx * u * r)
) / m33
rdot = (nd * force_scale_n + yaw_linear_moment) / m44
```
$$ \dot{u},\dot{v},\dot{p},\dot{r} $$

**Kinematics**
```python
xdot = math.cos(psi) * u - math.sin(psi) * v * math.cos(phi)
ydot = math.sin(psi) * u + math.cos(psi) * v * math.cos(phi)
phidot = p
psidot = r * math.cos(phi)
```
Earth fixed $$ \dot{x},\dot{y},\dot{\phi},\dot{\psi} $$

**Return derivative vector**
```python
return np.array(
    [
        udot,
        vdot,
        pdot,
        rdot,
        xdot,
        ydot,
        phidot,
        psidot,
        delta_dot,
        n1_dot,
        n2_dot,
    ],
    dtype=float,
)
```
Full state derivative

### 6.6 Why the 4-DOF model is useful even if v2 remains competitive

The v2 model gave better straight-line matching in the latest calibration, while the faithful 4DOF model gave better turn geometry and speed retention in a turn. This is expected:

- v2 is more empirically shaped for the two real tests;
- 4-DOF is more physically structured and closer to a conventional manoeuvring model.

Therefore, the 4-DOF model is more attractive for long-term modelling, while v2 may remain the short-term RL training model if straight-line fidelity is more important.

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

