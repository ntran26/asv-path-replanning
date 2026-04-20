# Bluefin vessel model derivation and Python implementation notes

**Purpose.** This note documents the workflow:

```text
MATLAB Bluefin model files -> marine craft modelling theory -> Python ship model files
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
| Body-fixed and earth-fixed coordinate frames | Chapter 2, Reference Frames, printed pp. 19-21 / PDF pp. 33-35 | The Bluefin models use body velocities such as surge \(u\), sway \(v\), roll rate \(p\), yaw rate \(r\), and earth-fixed pose \(x,y,\psi\). Fossen states that positions/orientations are described in an inertial or NED frame, while velocities are expressed in the body-fixed frame. fileciteturn41file17 |
| General marine craft equation | Chapter 3, printed p. 50 / PDF p. 62; also printed pp. 63-64 / PDF pp. 75-77 | Fossen writes the marine vessel model in the form \(M\dot{\nu}+C(\nu)\nu+D(\nu)\nu+g(\eta)=\tau+g_0+w\). The MATLAB Bluefin models are component-wise versions of this idea. fileciteturn42file6 fileciteturn42file4 |
| Added mass and hydrodynamic inertia | Chapter 3.2.1, printed pp. 63-67 / PDF pp. 75-79 | The Bluefin models use \(m_x\), \(m_y\), \(J_x\), and \(J_z\) as added-mass / added-inertia terms. Fossen explains that added mass represents inertia of the surrounding fluid. fileciteturn42file4 fileciteturn42file16 |
| Hydrodynamic damping | Chapter 3.2.2, printed pp. 71-75 / PDF pp. 83-88 | The polynomial hull force terms in the MATLAB models are empirical hydrodynamic damping / manoeuvring derivatives. Fossen explains linear and nonlinear damping, including skin friction, vortex shedding, and velocity-dependent terms. fileciteturn42file2 |
| 3-DOF horizontal model | Chapter 3.5.1, printed pp. 104-108 / PDF pp. 117-121 | `Blue02.m` is best understood as a 3-DOF horizontal manoeuvring model in surge, sway, and yaw. Fossen defines this reduction as \(\nu=[u,v,r]^T\) and \(\eta=[x,y,\psi]^T\). fileciteturn41file1 |
| Forward-speed model | Chapter 3.5.2, printed p. 107 / PDF p. 120 | Fossen's forward-speed equation includes both linear and quadratic damping. This motivated using speed-dependent surge resistance in Python. fileciteturn41file4 |
| Rudder sign and yaw convention | Chapter 3.5.2, printed p. 108 / PDF p. 121 | Fossen states that a positive rudder angle may be defined to yield positive yaw rate in a given model convention. This justifies explicitly documenting and fixing the sign bridge in Python. fileciteturn41file9 |
| Actuator forces and moment arms | Chapter 7.5.1, printed pp. 288-291 / PDF pp. 298-301 | Fossen explains that propellers, tunnel thrusters, rudders, fins, etc. generate forces and moments through actuator geometry. This supports the 4DOF separation into propeller, rudder, and bow-thruster contributions. fileciteturn41file6 fileciteturn41file7 fileciteturn41file0 |
| 4-DOF actuator layout | Chapter 7.5.1, printed p. 291 / PDF p. 301 | Fossen explicitly discusses actuator configuration columns in 4 DOF: surge, sway, roll, and yaw. This matches the Bluefin 4DOF model structure. fileciteturn41file0 |
| Restoring forces / roll restoring | Chapter 3.2.3, printed pp. 75-76 / PDF pp. 87-89 | The 4DOF model includes roll restoring through \(C_{44}=g\,m\,GM\), which follows the general idea that gravity and buoyancy generate restoring forces and moments. fileciteturn42file9 |

## 2. The general theory behind both MATLAB models

Fossen's general marine craft equation is

\[
M\dot{\nu}+C(\nu)\nu+D(\nu)\nu+g(\eta)=\tau+g_0+w .
\]

For the Bluefin work, the terms are interpreted as follows:

- \(M\dot{\nu}\): rigid-body and added-mass inertia;
- \(C(\nu)\nu\): centripetal/Coriolis-like velocity coupling;
- \(D(\nu)\nu\): hydrodynamic damping and manoeuvring derivatives;
- \(g(\eta)\): restoring effects, especially important for roll in 4DOF;
- \(\tau\): propeller, rudder, and thruster forces/moments;
- \(w\): disturbances such as wind, waves, and current, not included in the current calm-water calibration.

The Bluefin MATLAB scripts do not necessarily write the equations in matrix form. Instead, they expand the model into component equations for \(X\), \(Y\), \(K\), and \(N\), where:

- \(X\): surge force;
- \(Y\): sway force;
- \(K\): roll moment;
- \(N\): yaw moment.

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
Y_H = \frac{1}{2}\rho LdU^2
\left(
Y_vv_d + Y_{vr}v_d|r_d| + Y_rr_d
+ Y_{vv}|v_d|v_d + Y_{rr}r_d|r_d|
\right).
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
N_vv_d + N_{vr}|v_d|r_d + N_rr_d
+ N_{vv}v_d|v_d| + N_{rr}r_d|r_d|
\right).
$$

These are component-wise versions of the hydrodynamic damping and manoeuvring-derivative terms in Fossen's general marine craft equation:

$$
M\dot{\nu}+C(\nu)\nu+D(\nu)\nu+g(\eta)=\tau.
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
X_P = (1-t_P)\rho n_1^2D_P^4K_TJ.
$$

The model assumes the propeller mainly contributes surge force:

$$
Y_P = 0,
\qquad
N_P = 0.
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
\sqrt{1+6.3\left(1-\frac{0.856113u}{0.00717\cdot1000}\right)^{1.5}}.
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
\right).
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
X_R = -(1-t_R)F_N\sin\delta,
$$

$$
Y_R = -(1+a_H)F_N\cos\delta,
$$

$$
N_R = -(x_R+a_Hx_H)F_N\cos\delta.
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
X = X_H+X_P+X_R,
$$

$$
Y = Y_H+Y_P+Y_R,
$$

$$
N = N_H+N_P+N_R.
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
\dot{u} = \frac{2X}{m_{11}\rho L^2U^2},
$$

$$
\dot{v} = \frac{2Y}{m_{22}\rho L^2U^2},
$$

$$
\dot{r} = \frac{N}{I_z+J_z}.
$$

The heading derivative is:

$$
\dot{\psi}=\frac{r}{L}U.
$$

The rudder derivative is:

$$
\dot{\delta}=\operatorname{sat}_{\dot{\delta}_{\max}}(\delta_c-\delta).
$$

The planar kinematics are:

$$
\dot{x}=u\cos\psi-v\cos\phi\sin\psi,
$$

$$
\dot{y}=u\sin\psi-v\cos\phi\cos\phi.
$$

The final $\dot{y}$ expression appears unusual because it contains $\cos(\phi)\cos(\phi)$. This is another sign that `Blue02.m` should be treated as an intermediate reference rather than the cleanest final source.

### 3.11 Why `Blue02.m` alone was not enough

`Blue02.m` was useful, but it had several limitations:

1. The state comments and implementation are not fully consistent.
2. Roll is mentioned but not actually included as a proper dynamic state.
3. It does not include a consistent bow-thruster state.
4. It is less complete than `Bluefin4DOFModel02.m`.
5. It does not provide enough independent tuning freedom to match both straight-line and turning performance.

This is why the Python path first used a simplified 3DOF-inspired model (`ship_model_bluefin_v2.py`) and then moved to a 4DOF candidate.

## 4. `ship_model_bluefin_v2.py`: how it was constructed from the 3DOF theory

`ship_model_bluefin_v2.py` is not a literal copy of `Blue02.m`. It is a **control-oriented 3DOF model** inspired by Bluefin constants and by the 3DOF theory.

### 4.1 State and public interface

The Python state is:

\[
x = [u,\ v,\ r,\ \psi,\ \delta,\ x,\ y]^T.
\]

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

\[
x_{k+1}=x_k+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4).
\]

This is a numerical improvement over a simple Euler step. It was used because the nonlinear rudder and thrust equations can be stiff enough that Euler integration gives unstable or inaccurate results at the 10 Hz simulation rate.

### 4.3 Speed-shaped propeller law

In v2, the propeller force is implemented at lines 196-208 as:

\[
X_P = (1-t_P)K_T n|n|
\frac{1+k_b e^{-u/U_b}}{1+k_d u^2}.
\]

This was added because a simple constant \(rpm^2\) thrust law could not match both early acceleration and the final speed envelope. The idea is consistent with Fossen's forward-speed equation, where surge dynamics include both control force and linear/quadratic resistance. fileciteturn41file4

### 4.4 Hull damping and manoeuvring derivatives

The v2 model computes hull surge damping at lines 221-228 and sway/yaw damping at lines 230-240. These terms are the Python equivalent of the MATLAB hull-polynomial terms, but simplified and tuned.

The form follows the idea that hull resistance and manoeuvring forces are velocity-dependent, as discussed in Fossen's hydrodynamic damping section. fileciteturn42file2

### 4.5 Rudder split: axial loss, sway force, yaw moment

The v2 rudder block is implemented at lines 249-265:

- rudder inflow \(u_R,v_R\);
- angle of attack \(\alpha_R\);
- rudder normal force \(F_N\);
- axial drag \(X_R\);
- sway force \(Y_R\);
- yaw moment \(N_R\).

The key v2 modelling decision was to separate:

\[
X_R,\quad Y_R,\quad N_R
\]

instead of forcing all rudder effects to share one coefficient. This follows the physical idea that rudder forces create both lateral force and yaw moment, while also reducing forward speed. Fossen's actuator section supports this interpretation by describing rudders as lateral-force devices that generate steering yaw moments. fileciteturn41file7

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
x=[u,\ v,\ p,\ r,\ x,\ y,\ \phi,\ \psi,\ \delta,\ n_1,\ n_2]^T.
$$

The input vector is documented as:

```matlab
% ui      = [ delta_c n1_c n2_c]'
% delta_c = commanded rudder angle          (rad)
% n1_c    = commanded shaft velocity vector (rpm)
% n2_c    = commanded thruster velocity     (rpm)
```

The corresponding input vector is:

$$u_c=[\delta_c,\ n_{1c},\ n_{2c}]^T.
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
U=\sqrt{u^2+v^2},
$$

$$
\beta=-\sin^{-1}\left(\frac{v}{U}\right).
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
u_d=\frac{u}{U},\qquad v_d=\frac{v}{U},
$$

$$
p_d=\frac{pL}{U},\qquad r_d=\frac{rL}{U}.
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
n_{2c}=\frac{ui_3}{60}.
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
m_{44}=I_z+J_z.
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
|\delta_c|\le \delta_{\max}.
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
\dot{\delta}=\operatorname{sat}_{\dot{\delta}_{\max}}(\delta_c-\delta).
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
J=\frac{(1-w)u}{n_1D_P}.
$$

The thrust coefficient is:

$$
K_T=a_0+a_1J+a_2J^2.
$$

The non-dimensional propeller surge contribution is:

$$
X_{dP}=\frac{|n_1|n_1(1-t)K_TD_P^4}{0.5LdU^2}.
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
\sqrt{\eta\left(1+\kappa\sqrt{1+\frac{8K_T}{\pi J^2}}-1\right)^2+(1-\eta)},
$$

$$
v_{dR}=-\gamma_R\left(\beta-l_{dR}r_d+\frac{p(z_R-z_G)}{U}\right),
$$

$$
U_{dR}=\sqrt{u_{dR}^2+v_{dR}^2},
$$

$$
\alpha_R=\delta-\tan^{-1}\left(\frac{-v_{dR}}{u_{dR}}\right),
$$

$$
F_{dN}=-\left(\frac{A_R}{Ld}\right)
\left(\frac{6.13\lambda}{2.25+\lambda}\right)
U_{dR}^2\sin\alpha_R.
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
X_{dR}=(1-t_R)F_{dN}\sin\delta\cos\phi,
$$

$$
Y_{dR}=(1+a_H)F_{dN}\cos\delta\cos\phi,
$$

$$
K_{dR}=\frac{z_RY_{dR}}{L},
$$

$$
N_{dR}=(x_R+a_Hx_H)F_{dN}\cos\delta\cos\phi.
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
X_{dH}=f_X(\beta,r_d,\phi),
$$

$$
Y_{dH}=f_Y(\beta,r_d,\phi),
$$

$$
K_{dH}=f_K(\beta,r_d,\phi),
$$

$$
N_{dH}=f_N(\beta,r_d,\phi).
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
C_{44}=g m GM.
$$

The roll damping approximation is:

$$
B_{44}=\frac{2a}{\pi}\sqrt{gmGM(I_x+J_x)}.
$$

The additional roll moment is:

$$
K_{dH2}=z_GY_{dH}-B_{44}p-C_{44}\phi-(z_R-z_G)Y_{dR}.
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
F_{BT}=\frac{|n_2|n_2K_{BT}}{0.5\rho LdU^2}.
$$

It contributes to sway, roll, and yaw as:

$$
Y_{dB}=F_{BT},
$$

$$
K_{dB}=\frac{z_BF_{BT}}{L},
$$

$$
N_{dB}=\frac{x_BF_{BT}}{L}.
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
X_d=X_{dH}+X_{dP}+X_{dR},
$$

$$
Y_d=Y_{dH}+Y_{dR}+Y_{dB},
$$

$$
K_d=K_{dH}+K_{dH2}+K_{dR}+K_{dB},
$$

$$
N_d=N_{dH}+N_{dR}-x_GY_d+N_{dB}.
$$

This is the explicit component-wise version of the generalized force vector $\tau$ in Fossen's marine-craft equation.

### 5.13 Surge, sway, roll, and yaw accelerations

The MATLAB file first computes sway acceleration because it is reused in the roll equation:

```matlab
vdot = (Yd*(0.5*rho*L*d*U^2)-m11*u*r)/m22;
```

This corresponds to:

$$
\dot{v}=\frac{Y_d(0.5\rho LdU^2)-m_{11}ur}{m_{22}}.
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
\dot{u}=\frac{X_d(0.5\rho LdU^2)+m_{22}vr}{m_{11}},
$$

$$
\dot{v}=\frac{Y_d(0.5\rho LdU^2)-m_{11}ur}{m_{22}},
$$

$$
\dot{p}=\frac{K_d(0.5\rho Ld^2U^2)+(z_H-z_G)(m_y\dot{v}+m_xur)}{m_{33}},
$$

$$
\dot{r}=\frac{N_d(0.5\rho L^2dU^2)}{m_{44}}.
$$

The earth-fixed kinematics are:

$$
\dot{x}=u\cos\psi-v\sin\psi\cos\phi,
$$

$$
\dot{y}=u\sin\psi+v\cos\psi\cos\phi,
$$

$$
\dot{\phi}=p,
$$

$$
\dot{\psi}=r\cos\phi.
$$

The remaining actuator-state derivatives are:

$$
\dot{\delta},\qquad \dot{n}_1,\qquad \dot{n}_2.
$$

This is the main reason `Bluefin4DOFModel02.m` is a stronger modelling source than `Blue02.m`: it is a complete dynamic system with surge, sway, roll, yaw, position, heading, rudder state, propeller state, and bow-thruster state.

## 6. `bluefin_4dof_final.py`: how the Python file maps to the 4DOF theory

The final Python file keeps the same public interface as the v2 model:

```python
dx, dy, heading_deg, yaw_rate_degps = model.update(rpm, rud, dt)
```

This makes model switching easy in the Gym environment.

### 6.1 Constants and calibration

The calibrated constants are at the top of `bluefin_4dof_final.py`:

- `RPM_COMMAND_SCALE = 85.0`;
- `PROPELLER_THRUST_SCALE = 1.7`;
- `RUDDER_FORCE_SCALE = 0.6`;
- `RUDDER_X_DRAG_SCALE = 0.6`;
- `ROLL_DAMP_SCALE = 1.0`;
- `ROLL_RESTORE_SCALE = 1.4`.

These values are based on the latest refined faithful-4DOF sweep. The validation commands were:

- straight speed test: `rpm = 15`;
- turning test: `turn_rpm = 18`;
- turning rudder: `turn_rudder_deg = 25`.

The commands are not constants of the ship. They are test inputs used to compare simulation and real trials.

### 6.2 State and compatibility interface

The 4DOF Python state is:

\[
x=[u,\ v,\ p,\ r,\ x,\ y,\ \phi,\ \psi,\ \delta,\ n_1,\ n_2]^T.
\]

In the Python file this is implemented in:

- `reset()`: lines 76-110;
- `state_dict()`: lines 112-128;
- `_state_vector()`: lines 169-185;
- `_set_state_vector()`: lines 187-202.

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

This means the public repo convention can stay consistent even if the MATLAB internal sign is opposite. This is important because Fossen notes that rudder sign convention is model-defined; what matters is consistency between rudder and yaw rate. fileciteturn41file9

### 6.4 RK4 integration and guards

The Python code uses RK4 at lines 150-154. It clips extreme states at lines 156-162. These protections are not part of the theory; they are practical numerical safeguards to make the model usable from rest and in repeated RL/sweep simulations.

### 6.5 Derivative function

The physics is implemented in `_derivatives()` from line 204 onward.

The block structure is:

| Python block | Lines | Theory / MATLAB source |
|---|---:|---|
| Unpack state | 211 | \(x=[u,v,p,r,x,y,\phi,\psi,\delta,n_1,n_2]^T\) |
| Speed and drift angle | 213-215 | \(U=\sqrt{u^2+v^2}\), \(\beta=-\sin^{-1}(v/U)\) |
| Rudder actuator | 217-223 | \(\dot{\delta}=\mathrm{sat}(\delta_c-\delta)\) |
| Shaft dynamics | 225-228 | \(\dot{n}_1,\dot{n}_2\) saturation |
| Physical constants | 230-259 | Bluefin4DOFModel02 coefficients |
| Hull coefficients | 261-306 | MATLAB polynomial hull derivatives |
| Inertia terms | 308-321 | \(m_{11}=m+m_x\), \(m_{22}=m+m_y\), \(m_{33}=I_x+J_x\), \(m_{44}=I_z+J_z\) |
| Propeller model | 327-354 | advance-ratio \(J\), thrust coefficient \(K_T\), propeller surge force |
| Rudder model | 356-364 | rudder inflow, \(\alpha_R\), \(F_N\), \(X_R,Y_R,K_R,N_R\) |
| Hull forces/moments | 366-408 | \(X_{dH},Y_{dH},K_{dH},N_{dH}\) |
| Roll restoring/damping | 411-416, 445-451 | \(C_{44}=g\,m\,GM\), roll damping/restoring moments |
| Bow thruster | 418-430 | \(Y_B,K_B,N_B\) |
| Total forces/moments | 432-435 | \(X,Y,K,N\) sums |
| Accelerations | 441-453 | \(\dot{u},\dot{v},\dot{p},\dot{r}\) |
| Kinematics | 454-457 | earth-fixed \(\dot{x},\dot{y},\dot{\phi},\dot{\psi}\) |
| Return derivative vector | 459 onward | full state derivative |

### 6.6 Why the 4DOF model is useful even if v2 remains competitive

The v2 model gave better straight-line matching in the latest calibration, while the faithful 4DOF model gave better turn geometry and speed retention in a turn. This is expected:

- v2 is more empirically shaped for the two real tests;
- 4DOF is more physically structured and closer to a conventional manoeuvring model.

Therefore, the 4DOF model is more attractive for long-term modelling, while v2 may remain the short-term RL training model if straight-line fidelity is more important.

## 7. Practical workflow for future modelling problems

The general modelling process should be:

1. **Start from theory.** Use Fossen's equation \(M\dot{\nu}+C(\nu)\nu+D(\nu)\nu+g(\eta)=\tau+w\) as the modelling scaffold. fileciteturn42file6
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
