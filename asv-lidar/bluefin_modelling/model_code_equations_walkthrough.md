# Bluefin Model Code And Equations Walkthrough

## Purpose

This note explains the main code sections and matching equations for the Bluefin modelling path in this repo:

1. `old_models/ship_model.py`
2. `old_models/Blue02.m`
3. `old_models/ship_model_bluefin.py`
4. `old_models/ship_model_bluefin_v2.py`
5. `bluefin_4dof_report_bundle/matlab/Bluefin4DOFModel02.m`
6. `ship_model_bluefin_4dof.py`

The aim is to help a fresher understand both:

- what each block of code is doing, and
- what equation or modelling idea it corresponds to.

## Reading Guide

Throughout this note:

- `u` = surge velocity, forward speed
- `v` = sway velocity, lateral speed
- `p` = roll rate
- `r` = yaw rate
- `phi` = roll angle
- `psi` = yaw/heading angle
- `delta` = rudder angle
- `n1` = propeller speed
- `n2` = bow-thruster speed
- `m` = mass
- `I` = moment of inertia

When equations are written in compact form, they are there to explain the intent of the code, not to reproduce every coefficient exactly.

---

## 1. `old_models/ship_model.py`: the original simple ship model

This is the smallest and easiest model in the repo. It is useful because it is fast and stable, but it is also the least realistic.

### 1.1 Parameter block

**Code**

```python
MASS = 64.55
THRUST_COEF = 0.04
DRAG_COEF = 10
TURN_COEF = 100

MAX_RUD_ANGLE = 30
RUDDEROFFSET = 3
MOMINERTIA = 0.5 * MASS * RUDDEROFFSET**2
```

**Equation**

$$
\begin{aligned}
m &= 64.55 \\
T &= k_T \,\mathrm{rpm}^2 \\
D &= k_D \, v^2 \\
I_z &= 0.5 \, m \, l_r^2
\end{aligned}
$$

**Explanation**

- `THRUST_COEF` converts propeller command into forward thrust.
- `DRAG_COEF` gives a simple quadratic drag law.
- `TURN_COEF` is a lumped yaw damping term.
- `RUDDEROFFSET` is the lever arm used to turn a rudder side force into a yaw moment.
- `MOMINERTIA` is not measured from geometry in detail. It is a simple approximate yaw inertia.

### 1.2 Force and moment calculation

**Code**

```python
def _calc_forces(self, rpm, rud):
    thrust = THRUST_COEF * rpm**2
    rud_angle = np.radians(MAX_RUD_ANGLE * rud / 100)
    fwd_thrust = thrust * np.cos(rud_angle) - DRAG_COEF * self._v**2

    rud_moment = thrust * np.sin(rud_angle) * RUDDEROFFSET
    moment = rud_moment - (TURN_COEF * self._w)
    return fwd_thrust, moment
```

**Equation**

$$
\begin{aligned}
T &= k_T \,\mathrm{rpm}^2 \\
\delta &= \delta_{\max}\,\frac{\mathrm{rud}}{100} \\
X &= T \cos(\delta) - k_D v^2 \\
N &= T \sin(\delta)\, l_r - k_N w
\end{aligned}
$$

**Explanation**

- The propeller thrust is split into a forward part `T cos(delta)` and a turning part `T sin(delta)`.
- Drag depends only on forward speed.
- Turning is produced directly by the rudder deflecting the thrust.
- There is no separate sway velocity.
- Hull, propeller, and rudder are not modelled separately.

### 1.3 Integration and kinematics

**Code**

```python
def update(self, rpm, rud, dt):
    d = self._v * dt + self._a * dt * dt * 0.5
    self._h = self._h + self._w * dt + self._dw * dt * dt * 0.5

    dx = d * np.sin(self._h)
    dy = d * np.cos(self._h)

    thrust, moment = self._calc_forces(rpm, rud)
    a = thrust / MASS
    dw = moment / MOMINERTIA
```

**Equation**

$$
\begin{aligned}
d &= v\,dt + 0.5\,a\,dt^2 \\
h_{k+1} &= h_k + w\,dt + 0.5\,dw\,dt^2 \\
dx &= d \sin(h) \\
dy &= d \cos(h) \\
\frac{dv}{dt} &= \frac{X}{m} \\
\frac{dw}{dt} &= \frac{N}{I_z}
\end{aligned}
$$

**Explanation**

- This is a simple second-order kinematic update.
- The boat always moves in the heading direction.
- No lateral slip exists in the model.
- That is why this model is attractive for RL, but also why it cannot capture real Bluefin manoeuvring properly.

---

## 2. `Blue02.m`: the first MATLAB-based Bluefin model

This file is the first attempt at a more classical manoeuvring model. It introduces separate surge, sway, and yaw terms plus rudder dynamics. However, it is not internally clean, which is one reason it was not used as the final truth source.

### 2.1 Stated state and input definition

**Code**

```matlab
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

**Equation**

$$
\begin{aligned}
x &= [u, v, p, r, x, y, \phi, \psi, \delta]^T \\
u_{\mathrm{in}} &= [\delta_c, \mathrm{rpm}_{\mathrm{prop}}, \mathrm{rpm}_{\mathrm{thr}}]^T
\end{aligned}
$$

**Explanation**

- This comment block says the model is effectively a 4DOF state description.
- But later code only uses part of this state definition consistently.
- This mismatch is one of the warning signs in `Blue02.m`.

### 2.2 Actual variable extraction in code

**Code**

```matlab
U = sqrt(0.6^2+0.4^2);
u = x(1)/U;
v = x(2)/U;
r = x(3)*L/U;
psi = x(4);
del = x(5);
phi = x(6);
```

**Equation**

$$
\begin{aligned}
u_d &= \frac{x_1}{U_{\mathrm{ref}}} \\
v_d &= \frac{x_2}{U_{\mathrm{ref}}} \\
r_d &= \frac{x_3 L}{U_{\mathrm{ref}}}
\end{aligned}
$$

**Explanation**

- The code comments and the actual indexing do not agree.
- The code is effectively treating the state vector differently from the comment header.
- It also starts by dividing states by a fixed reference speed `U = sqrt(0.6^2 + 0.4^2)`.
- That makes the file harder to port directly without interpretation.

### 2.3 Non-dimensionalization

**Code**

```matlab
U = sqrt(u*u+v*v);
vd = v/U;
rd = r*L/U;
```

**Equation**

$$
\begin{aligned}
U &= \sqrt{u^2 + v^2} \\
v_d &= \frac{v}{U} \\
r_d &= \frac{rL}{U}
\end{aligned}
$$

**Explanation**

- This is a standard manoeuvring-model trick.
- Instead of writing all forces in dimensional form directly, the code uses non-dimensional sway and yaw variables.
- The price is that the equations become harder to read for newcomers.

### 2.4 Hull force block

**Code**

```matlab
XH = m*v*r - 1/2*rho*U^2*L*d*Sw/(L*d)*0.4631/(log(4*10^7)^2.6)*u^2 + ...
     Xvv*v^2 + Xvr*v*r + Xrr*r^2;

YH = 0.5*rho*L*d*U^2*(Yv*vd + Yvr*vd*abs(rd) + Yr*rd + ...
     Yvv*abs(vd)*vd + Yrr*rd*abs(rd));

NH = 0.5*rho*(L^2)*d*U^2*(Nv*vd + Nvr*abs(vd)*rd + Nr*rd + ...
     Nvv*vd*abs(vd) + Nrr*rd*abs(rd));
```

**Equation**

$$
\begin{aligned}
X_H &= mvr + X_{\mathrm{viscous}} + X_{\mathrm{crossflow}} \\
Y_H &= 0.5\,\rho\,L\,d\,U^2\,f_Y(v_d, r_d) \\
N_H &= 0.5\,\rho\,L^2\,d\,U^2\,f_N(v_d, r_d)
\end{aligned}
$$

**Explanation**

- `XH` is the hull surge force.
- `YH` is the hull sway force.
- `NH` is the hull yaw moment.
- These are no longer simple constants; they depend on velocity combinations such as `v^2`, `v*r`, and `r^2`.
- This is the first point where the model starts resembling a real manoeuvring model instead of a toy RL model.

### 2.5 Propeller and rudder force blocks

**Code**

```matlab
KT = 0.25; J = 0.072;
XP = (1-tp)*rho*(n1^2)*Dp^4*KT*J;

uR = 0.856113*u*sqrt(1+6.3*(1-(0.856113*u)/(0.00717*1000))^1.5);
FN = -1/2*10^3*0.0091*2.6927*u^2*sin(del + (0.603463/uR)*(v-1.5525*r));

XR = -(1-0.449821)*FN*sin(del);
YR = -(1+0.443853)*FN*cos(del);
NR = -(0.646875+0.443853*0.7569)*FN*cos(del);
```

**Equation**

$$
\begin{aligned}
X_P &= (1 - t_p)\,\rho\,n_1^2\,D_p^4\,K_T\,J \\
F_N &= -0.5\,\rho\,A_R\,C_L\,U_R^2 \sin(\alpha_R) \\
X_R &= -c_x F_N \sin(\delta) \\
Y_R &= -c_y F_N \cos(\delta) \\
N_R &= -c_n F_N \cos(\delta)
\end{aligned}
$$

**Explanation**

- The propeller contributes mainly surge thrust.
- The rudder contributes a normal force `FN`, which is then resolved into surge loss, sway force, and yaw moment.
- This is a much more realistic picture than the original `ship_model.py`, where the rudder just bends the thrust.

### 2.6 Final state derivative

**Code**

```matlab
xdot = [X*2/(m11*(rho*L*L*U*U))
        Y*2/(m22*(rho*L*L*U*U))
        1/(Iz+Jz)*(N)
        r/L*U
        del_dot
        u*cos(psi)-v*cos(phi)*sin(psi)
        u*sin(psi)-v*cos(phi)*cos(phi)];
```

**Equation**

$$
\begin{aligned}
\frac{du}{dt} &= \frac{X}{m_{11}} \\
\frac{dv}{dt} &= \frac{Y}{m_{22}} \\
\frac{dr}{dt} &= \frac{N}{I_z + J_z} \\
\frac{d\psi}{dt} &= r \\
\frac{d\delta}{dt} &= \dot{\delta} \\
\frac{dx}{dt} &= u \cos(\psi) - v \sin(\psi) \\
\frac{dy}{dt} &= u \sin(\psi) + v \cos(\psi)
\end{aligned}
$$

**Explanation**

- This is the ODE block that makes the vessel move.
- The important idea is correct: forces and moments are converted into accelerations.
- But the code only returns seven derivatives even though the comment header suggests a bigger state.
- That inconsistency is a major reason `Blue02.m` was not a good final source for direct transfer.

---

## 3. `ship_model_bluefin.py`: the first practical Python 3DOF adaptation

This file tries to keep the structure of the MATLAB model while fitting the existing Python interface used by the RL code.

### 3.1 State and parameter design

**Code**

```python
MASS = 64.55
MX = 3.662
MY = 62.7366
IZ = 9.6038
JZ = 0.6309

THRUST_COEF = 0.07
DRAG_COEF = 1.5
TURN_COEF = 5.0

RUDDER_FORCE_SCALE = 0.10
LINEAR_SURGE_DAMP = 2.0
LINEAR_SWAY_DAMP = 20.0
LINEAR_YAW_DAMP = 4.0
```

**Equation**

$$
\begin{aligned}
m_{11} &= m + m_x \\
m_{22} &= m + m_y \\
m_{33} &= I_z + J_z \\
X_{\mathrm{total}} &= X_{\mathrm{hull}} + X_{\mathrm{prop}} + X_{\mathrm{rudder}} \\
Y_{\mathrm{total}} &= Y_{\mathrm{hull}} + Y_{\mathrm{rudder}} \\
N_{\mathrm{total}} &= N_{\mathrm{hull}} + N_{\mathrm{rudder}}
\end{aligned}
$$

**Explanation**

- `ship_model_bluefin.py` is the first Python file to really keep surge, sway, and yaw as separate dynamics.
- It still exposes the old Python interface, but internally it is already a 3DOF manoeuvring model.
- The extra linear damping terms were added to keep it numerically robust in the Python environment.

### 3.2 State vector and RK4 integration

**Code**

```python
def _state_vector(self) -> np.ndarray:
    return np.array([
        self._v, self._v_sway, self._w, self._h, self._delta, self._x, self._y
    ], dtype=float)

def update(self, rpm, rud, dt, *, thruster_rpm=0.0):
    k1 = self._derivatives(s0, rpm, rud, thruster_rpm)
    k2 = self._derivatives(s0 + 0.5 * dt * k1, rpm, rud, thruster_rpm)
    k3 = self._derivatives(s0 + 0.5 * dt * k2, rpm, rud, thruster_rpm)
    k4 = self._derivatives(s0 + dt * k3, rpm, rud, thruster_rpm)
    s1 = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
```

**Equation**

$$
\begin{aligned}
s &= [u, v, r, \psi, \delta, x, y]^T \\
s_{k+1} &= s_k + \frac{dt}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
\end{aligned}
$$

**Explanation**

- This was a major implementation improvement over the simpler model.
- RK4 gives much better numerical stability than a crude Euler step.
- The state now explicitly contains sway, heading, rudder angle, and position.

### 3.3 Force blocks in v1

**Code**

```python
x_prop = (1.0 - TP) * THRUST_COEF * rpm * abs(rpm)

f_n = TURN_COEF * RUDDER_FORCE_SCALE * 0.5 * rho * AR * FALP * (
    u_r * u_r + v_r * v_r
) * math.sin(alpha_r)

x_rud = -(1.0 - TR) * f_n * math.sin(delta)
y_rud = -(1.0 + AH) * f_n * math.cos(delta)
n_rud = -(X_RUDDER + AH * X_HULL) * f_n * math.cos(delta)
```

**Equation**

$$
\begin{aligned}
X_P &= (1 - T_P)\,k_T\,\mathrm{rpm}\,\lvert \mathrm{rpm} \rvert \\
F_N &= k_{\mathrm{turn}}\,k_{\mathrm{rud}}\,0.5\,\rho\,A_R\,C_L\,U_R^2 \sin(\alpha_R) \\
X_R &= -(1 - T_R)\,F_N \sin(\delta) \\
Y_R &= -(1 + A_H)\,F_N \cos(\delta) \\
N_R &= -(x_R + A_H x_H)\,F_N \cos(\delta)
\end{aligned}
$$

**Explanation**

- Compared with `ship_model.py`, this is a large jump in physical detail.
- The propeller is now a dedicated thrust term.
- Rudder forces are built from inflow velocity and attack angle.
- Hull, propeller, and rudder all contribute separately.
- But this first Python 3DOF version still ties many behaviours together through `TURN_COEF`, so it is not yet easy to calibrate cleanly.

---

## 4. `ship_model_bluefin_v2.py`: what was added in v2

Version 2 is not just a clean-up. It adds new modelling structure specifically to match real-vessel logs better.

### 4.1 New thrust-shaping block

**Code**

```python
THRUST_LOW_SPEED_BOOST = 1.6
THRUST_BOOST_U0 = 0.7
THRUST_HIGH_SPEED_DECAY = 0.26

def _propeller_force(self, rpm: float, u_eff: float) -> float:
    n = max(rpm, 0.0)
    static_term = THRUST_COEF * n * abs(n)
    low_speed_boost = 1.0 + THRUST_LOW_SPEED_BOOST * math.exp(
        -u_eff / max(THRUST_BOOST_U0, 1e-6)
    )
    high_speed_decay = 1.0 / (1.0 + THRUST_HIGH_SPEED_DECAY * u_eff * u_eff)
    return (1.0 - TP) * static_term * low_speed_boost * high_speed_decay
```

**Equation**

$$
X_P = (1 - T_P)\,k_T\,n\,\lvert n \rvert
\left(1 + k_{\mathrm{boost}} e^{-u/u_0}\right)
\frac{1}{1 + k_{\mathrm{decay}} u^2}
$$

**Explanation**

- In v1, thrust was basically proportional to `rpm * |rpm|`.
- In v2, thrust is stronger at low speed and weaker at high speed.
- This was introduced because the earlier model could not match the real acceleration transient well.

### 4.2 New rudder split in v2

**Code**

```python
RUDDER_FORCE_SCALE = 0.32
RUDDER_YAW_SCALE = 2.60
RUDDER_X_DRAG_SCALE = 0.02

x_prop = self._propeller_force(rpm, u_eff)

f_n = RUDDER_FORCE_SCALE * 0.5 * rho * AR * FALP * (
    u_r * u_r + v_r * v_r
) * math.sin(alpha_r)

x_rud = -RUDDER_X_DRAG_SCALE * abs(f_n) * abs(math.sin(delta))
y_rud = -(1.0 + AH) * f_n * math.cos(delta)
n_rud = -RUDDER_YAW_SCALE * rudder_arm * f_n * math.cos(delta)
```

**Equation**

$$
\begin{aligned}
F_N &= k_F\,0.5\,\rho\,A_R\,C_L\,U_R^2 \sin(\alpha_R) \\
X_R &= -k_X \lvert F_N \rvert \lvert \sin(\delta) \rvert \\
Y_R &= -(1 + A_H)\,F_N \cos(\delta) \\
N_R &= -k_N\,l_R\,F_N \cos(\delta)
\end{aligned}
$$

**Explanation**

- In v1, one rudder-related scale strongly influenced sway force, yaw moment, and speed loss together.
- In v2, these effects are partially separated.
- That made it possible to improve yaw-rate fitting without destroying forward-speed fitting.

### 4.3 What v2 adds compared with v1


- `ship_model_bluefin.py` brings Blue02-style 3DOF dynamics into Python
- `ship_model_bluefin_v2.py` keeps that structure, but add empirical calibration knobs where the logs show the model is still wrong

So v2 adds:

1. speed-shaped propeller thrust
2. split rudder authority
3. improved calibration flexibility
4. better practical fit to real-vessel benchmarks

It is therefore a better simulator than v1, but less of a strict MATLAB transfer.

---

## 5. `Bluefin4DOFModel02.m`: the richer MATLAB 4DOF model

This file is the main MATLAB source behind the current 4DOF Python model.

### 5.1 State and input definition

**Code**

```matlab
% x(1)=u
% x(2)=v
% x(3)=p
% x(4)=r
% x(5)=x
% x(6)=y
% x(7)=phi
% x(8)=psi
% x(9)=delta
% x(10)=n1
% x(11)=n2

% ui = [ delta_c n1_c n2_c ]'
```

**Equation**

$$
\begin{aligned}
x &= [u, v, p, r, x, y, \phi, \psi, \delta, n_1, n_2]^T \\
u_{\mathrm{in}} &= [\delta_c, n_{1c}, n_{2c}]^T
\end{aligned}
$$

**Explanation**

- This is already a much cleaner and fuller vessel model than `Blue02.m`.
- It tracks roll, yaw, rudder state, propeller state, and bow-thruster state.
- That is why it became the stronger source for the later Python work.

### 5.2 Normalization and drift angle

**Code**

```matlab
L = 1.725;
U = sqrt(x(1)^2 + x(2)^2);
b = -asin(x(2)/U);

ud = x(1)/U;
vd = x(2)/U;
pd = x(3)*L/U;
rd = x(4)*L/U;
```

**Equation**

$$
\begin{aligned}
U &= \sqrt{u^2 + v^2} \\
\beta &= -\arcsin\left(\frac{v}{U}\right) \\
u_d &= \frac{u}{U} \\
v_d &= \frac{v}{U} \\
p_d &= \frac{pL}{U} \\
r_d &= \frac{rL}{U}
\end{aligned}
$$

**Explanation**

- This model is built in the style of a non-dimensional manoeuvring model.
- The drift angle `b` describes how much the hull is moving sideways relative to its forward direction.
- Roll and yaw are both included in normalized form.

### 5.3 Actuator dynamics

**Code**

```matlab
delta_dot = delta_c - delta;
n1_dot = n1s - n1;
n2_dot = n2s - n2;

if abs(delta_dot) >= Ddelta_max*pi/180
   delta_dot = sign(delta_dot)*Ddelta_max*pi/180;
end
if abs(n1_dot) >= Nc_max
   n1_dot = sign(n1_dot)*Nc_max;
end
if abs(n2_dot) >= Nc_max
   n2_dot = sign(n2_dot)*Nc_max;
end
```

**Equation**

$$
\begin{aligned}
\frac{d\delta}{dt} &= \mathrm{sat}(\delta_c - \delta) \\
\frac{dn_1}{dt} &= \mathrm{sat}(n_{1c} - n_1) \\
\frac{dn_2}{dt} &= \mathrm{sat}(n_{2c} - n_2)
\end{aligned}
$$

**Explanation**

- The actuators do not jump instantly to the command.
- Rudder and shaft speed move toward their commands with rate limits.
- This is important because a real vessel does not respond instantaneously.

### 5.4 Propeller block

**Code**

```matlab
J = onew*u/(n1s*DPs);
a0 = 0.3267; a1 = -0.2297; a2 = -0.1607;
KT = a0 + a1*J + a2*J^2;
XdP = abs(n1s)*n1s*onet*KT*(DPs^4)/(0.5*L*d*U^2);
```

**Equation**

$$
\begin{aligned}
J &= \frac{(1 - w)u}{n_1 D_p} \\
K_T &= a_0 + a_1 J + a_2 J^2 \\
X_P &\propto n_1 \lvert n_1 \rvert K_T D_p^4
\end{aligned}
$$

**Explanation**

- Unlike the earlier models, thrust is no longer a fixed `rpm^2` law.
- It depends on the propeller advance ratio `J`, which reflects how the inflow speed changes the propeller loading.
- This is much more physically realistic.

### 5.5 Rudder block

**Code**

```matlab
udR = epsi*(onew)*sqrt(eta*((1+kappa*sqrt(1+8*KT/(pi*J^2))-1)^2)+(1-eta));
vdR = -gR*(b-ldR*rd+(p*(zR-zG)/U));
UdR = sqrt(udR^2+vdR^2);
alphaR = delta-atan2(-vdR,udR);
FdN = -(ARpLd)*(6.13*lambda/(2.25+lambda))*UdR^2*sin(alphaR);

XdR = (onetR)*FdN*sin(delta)*cos(phi);
YdR = (1+aH)*FdN*cos(delta)*cos(phi);
KdR = zR*YdR/L;
NdR = (xR+aH*xH)*FdN*cos(delta)*cos(phi);
```

**Equation**

$$
\begin{aligned}
U_R &= \sqrt{u_R^2 + v_R^2} \\
\alpha_R &= \delta - \mathrm{atan2}(-v_R, u_R) \\
F_N &\propto U_R^2 \sin(\alpha_R) \\
X_R &= c_X F_N \sin(\delta) \\
Y_R &= c_Y F_N \cos(\delta) \\
K_R &= \frac{z_R Y_R}{L} \\
N_R &= c_N F_N \cos(\delta)
\end{aligned}
$$

**Explanation**

- The rudder force depends on its own local inflow velocity, not just vessel speed.
- Roll angle `phi` already affects the rudder terms.
- The rudder contributes not only surge loss, sway force, and yaw moment, but also a roll moment.

### 5.6 Hull, roll, and bow-thruster blocks

**Code**

```matlab
XdH = ...
YdH = ...
KdH = ...
NdH = ...

C44  = g*m*GM;
B44  = 2*a/pi*sqrt(g*m*GM*(Ix+Jx));
KdH2 = zG*YdH-B44*p-C44*phi-(zR-zG)*YdR;

FBT = abs(n2s)*n2s*KBT/(0.5*rho*L*d*U^2);
YdB = FBT;
KdB = zB*FBT/L;
NbB = xB*FBT/L;
```

**Equation**

$$
\begin{aligned}
X_H,\;Y_H,\;K_H,\;N_H &= \text{nonlinear hull terms} \\
K_{\mathrm{restore}} &= -C_{44}\phi \\
K_{\mathrm{damp}} &= -B_{44}p \\
Y_B &\propto n_2 \lvert n_2 \rvert \\
K_B &= \frac{z_B Y_B}{L} \\
N_B &= \frac{x_B Y_B}{L}
\end{aligned}
$$

**Explanation**

- `Bluefin4DOFModel02.m` is the first file in this chain to model roll properly.
- It includes hydrostatic restoring, roll damping, and bow-thruster effects.
- This is a major step beyond the earlier 3DOF models.

### 5.7 Final ODE block

**Code**

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

**Equation**

$$
\begin{aligned}
\frac{du}{dt} &= \frac{X + m_{22}vr}{m_{11}} \\
\frac{dv}{dt} &= \frac{Y - m_{11}ur}{m_{22}} \\
\frac{dp}{dt} &= \frac{K + \text{coupling}}{m_{33}} \\
\frac{dr}{dt} &= \frac{N}{m_{44}} \\
\frac{dx}{dt} &= u \cos(\psi) - v \sin(\psi)\cos(\phi) \\
\frac{dy}{dt} &= u \sin(\psi) + v \cos(\psi)\cos(\phi) \\
\frac{d\phi}{dt} &= p \\
\frac{d\psi}{dt} &= r \cos(\phi)
\end{aligned}
$$

**Explanation**

- This is the full 4DOF free-running vessel model.
- It includes dynamic coupling between translational and rotational motion.
- In theory, this is much closer to a real small vessel than the earlier models.

---

## 6. `ship_model_bluefin_4dof.py`: the current Python 4DOF model

This file is the current practical Python version used in the repo. It follows `Bluefin4DOFModel02.m` closely, but adds runtime guards and calibration knobs.

### 6.1 Command bridge and calibration constants

**Code**

```python
RPM_COMMAND_SCALE = 90.0
THRUSTER_COMMAND_SCALE = 60.0

RECOMMENDED_COMMAND_RPM_MAX = 18.0
RECOMMENDED_PROP_RPM_MAX = RPM_COMMAND_SCALE * RECOMMENDED_COMMAND_RPM_MAX

PROPELLER_THRUST_SCALE = 2.2
PROPELLER_ADVANCE_SCALE = 1.0
RUDDER_FORCE_SCALE = 0.10
RUDDER_YAW_SCALE = 1.7
RUDDER_INFLOW_SCALE = 1.0
RUDDER_X_DRAG_SCALE = 0.01
LINEAR_SURGE_DAMP = 1.5
LINEAR_YAW_DAMP = 0.0
ROLL_DAMP_SCALE = 4.0
ROLL_RESTORE_SCALE = 1.2
```

**Equation**

$$
\mathrm{rpm}_{MATLAB} = \mathrm{RPM\_COMMAND\_SCALE} \cdot \mathrm{rpm}_{repo}
$$

$$
n_{1c} = \frac{\mathrm{rpm}_{MATLAB}}{60}
$$

$$
X_P^{new} = k_{prop} X_P^{MATLAB}
$$

$$
Y_R^{new} = k_{rud} Y_R^{MATLAB}
$$

$$
N_R^{new} = k_{yaw} N_R^{MATLAB}
$$

$$
K_{roll}^{new} = k_{damp} K_{damp} + k_{restore} K_{restore}
$$

**Explanation**

- The repo uses a simplified command scale, while the MATLAB model expects propeller rpm in a different convention.
- These scale constants bridge that mismatch.
- The tuning constants let the Python model stay close to the MATLAB structure while still matching the real logs better.

### 6.2 State vector and update interface

**Code**

```python
def _state_vector(self) -> np.ndarray:
    return np.array([
        self._u, self._v, self._p, self._r,
        self._x, self._y, self._phi, self._psi,
        self._delta, self._n1, self._n2
    ], dtype=float)

def update(self, rpm, rud, dt, *, thruster_rpm=0.0):
    delta_cmd = float(np.clip(rud, -100.0, 100.0)) / 100.0 * math.radians(MAX_RUD_ANGLE)
    n1_cmd_rpm = max(float(rpm), 0.0) * RPM_COMMAND_SCALE
    n2_cmd_rpm = float(thruster_rpm) * THRUSTER_COMMAND_SCALE

    k1 = self._derivatives(s0, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
    k2 = self._derivatives(s0 + 0.5 * dt * k1, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
    k3 = self._derivatives(s0 + 0.5 * dt * k2, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
    k4 = self._derivatives(s0 + dt * k3, delta_cmd, n1_cmd_rpm, n2_cmd_rpm)
```

**Equation**

$$
\begin{aligned}
s &= [u, v, p, r, x, y, \phi, \psi, \delta, n_1, n_2]^T \\
\delta_c &= \mathrm{sat}(\mathrm{rud})\,\delta_{\max} \\
n_{1c} &= \mathrm{rpm}_{\mathrm{repo}}\,k_{\mathrm{rpm}} \\
n_{2c} &= \mathrm{rpm}_{\mathrm{thr}}\,k_{\mathrm{thr}} \\
s_{k+1} &= \mathrm{RK4}\left(s_k, f(s, u_{\mathrm{in}})\right)
\end{aligned}
$$

**Explanation**

- The state vector now matches the 4DOF MATLAB file directly.
- The interface still looks like the old Python simulator so the rest of the repo can use it.
- This is an important engineering step: the model became richer without breaking the surrounding code.

### 6.3 Numerical guards

**Code**

```python
MIN_FLOW_SPEED = 0.05
MIN_ADVANCE_RATIO = 1e-4
MAX_SURGE_SPEED = 5.0
MAX_SWAY_SPEED = 3.0
MAX_ROLL_RATE_RAD = math.radians(180.0)
MAX_YAW_RATE_RAD = math.radians(180.0)

s1[0] = float(np.clip(s1[0], -1.0, MAX_SURGE_SPEED))
s1[1] = float(np.clip(s1[1], -MAX_SWAY_SPEED, MAX_SWAY_SPEED))
s1[2] = float(np.clip(s1[2], -MAX_ROLL_RATE_RAD, MAX_ROLL_RATE_RAD))
s1[3] = float(np.clip(s1[3], -MAX_YAW_RATE_RAD, MAX_YAW_RATE_RAD))
```

**Equation**

$$
\begin{aligned}
U &= \max(U, U_{\min}) \\
J &= \max(J, J_{\min}) \\
s_i &= \mathrm{clip}(s_i, s_{i,\min}, s_{i,\max})
\end{aligned}
$$

**Explanation**

- The MATLAB equations assume physically meaningful speeds and inflow states.
- In Python, during startup or aggressive tuning sweeps, those assumptions can fail.
- These guards prevent divide-by-zero and explosive states.

### 6.4 Propeller and rudder sections in the Python port

**Code**

```python
j_adv = PROPELLER_ADVANCE_SCALE * onew * u / max(abs(n1_force) * d_prop, MIN_ADVANCE_RATIO)
kt = a0 + a1 * j_adv + a2 * j_adv * j_adv
xd_p = (
    PROPELLER_THRUST_SCALE
    * abs(n1_force) * n1_force
    * onet * kt * (d_prop**4)
    / (0.5 * l_ship * draft * u_mag * u_mag)
)

yd_r = RUDDER_FORCE_SCALE * (1.0 + a_h) * fd_n * math.cos(delta) * math.cos(phi)
nd_r = RUDDER_FORCE_SCALE * RUDDER_YAW_SCALE * (
    x_r + a_h * x_h
) * fd_n * math.cos(delta) * math.cos(phi)
```

**Equation**

$$
\begin{aligned}
J &= k_J \frac{(1 - w)u}{n_1 D_p} \\
K_T &= a_0 + a_1 J + a_2 J^2 \\
X_P &= k_{\mathrm{prop}} X_P^{\mathrm{MATLAB}} \\
Y_R &= k_F (1 + a_H) F_N \cos(\delta)\cos(\phi) \\
N_R &= k_F k_N (x_R + a_H x_H) F_N \cos(\delta)\cos(\phi)
\end{aligned}
$$

**Explanation**

- The Python port keeps the same basic propeller and rudder structure as the MATLAB file.
- But it introduces explicit scaling hooks for calibration.
- This is one reason the Python 4DOF model can fit the logs better than a literal untuned port.

### 6.5 Roll dynamics in the Python port

**Code**

```python
roll_damping_moment = -ROLL_DAMP_SCALE * b44 * p
roll_restoring_moment = -ROLL_RESTORE_SCALE * c44 * phi
pdot = (
    kd * force_scale_k
    + roll_damping_moment
    + roll_restoring_moment
    + (z_h - z_g) * (my * vdot + mx * u * r)
) / m33
```

**Equation**

$$
\begin{aligned}
K_{\mathrm{total}} &= K_{\mathrm{hydro}} + K_{\mathrm{damp}} + K_{\mathrm{restore}} + K_{\mathrm{coupling}} \\
K_{\mathrm{damp}} &= -k_{\mathrm{damp}} B_{44} p \\
K_{\mathrm{restore}} &= -k_{\mathrm{restore}} C_{44} \phi \\
\frac{dp}{dt} &= \frac{K_{\mathrm{total}}}{m_{33}}
\end{aligned}
$$

**Explanation**

- This is where roll becomes explicit in the Python implementation.
- The model includes hydrostatic restoring and roll damping, then divides by the effective roll inertia.
- This section is a key reason the current model is more faithful to the richer MATLAB source than the earlier 3DOF models.

---

## 7. Big-picture comparison

### 7.1 What changes from one model to the next?

To keep GitHub printing clean on A4, the comparison is written as a stacked list instead of a wide table:

1. `ship_model.py`
   - State idea: forward speed + heading + yaw rate.
   - Force idea: one thrust, one drag, one turning moment.
   - Advantage: very fast and simple.
   - Limitation: not physically rich.

2. `Blue02.m`
   - State idea: surge, sway, yaw, rudder.
   - Force idea: separate hull, propeller, rudder terms.
   - Advantage: introduces real manoeuvring structure.
   - Limitation: internally inconsistent.

3. `ship_model_bluefin.py`
   - State idea: Python 3DOF.
   - Force idea: Blue02-inspired hull/prop/rudder split.
   - Advantage: practical Python structure.
   - Limitation: hard to tune cleanly.

4. `ship_model_bluefin_v2.py`
   - State idea: Python 3DOF with empirical shaping.
   - Force idea: speed-shaped thrust + split rudder gains.
   - Advantage: better match to logs.
   - Limitation: less literal as a MATLAB transfer.

5. `Bluefin4DOFModel02.m`
   - State idea: surge, sway, roll, yaw + actuators.
   - Force idea: nonlinear hull + prop + rudder + thruster + roll.
   - Advantage: richest vessel physics source.
   - Limitation: not directly repo-ready.

6. `ship_model_bluefin_4dof.py`
   - State idea: Python 4DOF.
   - Force idea: guarded/tunable port of MATLAB 4DOF.
   - Advantage: best current calibrated model.
   - Limitation: still needs validation margin for field truth.

### 7.2 Final takeaway

- the original model is easy because it hides most of the physics,
- the Blue02 family introduces real manoeuvring structure,
- `v2` improves fit by adding empirical shaping,
- the 4DOF MATLAB model is the richer theory source,
- the current Python 4DOF model is the practical compromise between theory, code stability, and calibration to real logs.
