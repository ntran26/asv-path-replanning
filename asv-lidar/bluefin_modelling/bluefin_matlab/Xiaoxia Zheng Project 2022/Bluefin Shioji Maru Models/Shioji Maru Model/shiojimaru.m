% Filename: shiojimaru.m
% Mathematical model for Shioji Maru, the training vessel of
% Tokyo University of Marine Science and Technology

function xdot = shiojimaru(x,ui)
% State variables:
% u     = surge velocity          (m/s)      > x1
% v     = sway velocity           (m/s)      > x2
% r     = yaw rate                (rad/s)    > x3
% psi   = yaw angle               (rad)      > x4
% delta = actual rudder angle     (rad)      > x5
% theta = pitch angle        (deg)           > x6
% x     = position in x-direction (m)        > x7
% y     = position in y-direction (m)        > x8
%
% The input vector is:
% ui      = [delta_c cpp_c BT ST Xcc Ycc Wx Wy]' where
% delta_c = commanded rudder angle   (rad)
% cpp_c  = commanded pitch angle (deg)  
% BT = Bow Thruster (notch/10)
% ST = Stern Thruster (notch/10)
% Xcc = Current X [m/s]
% Ycc = Current Y [m/s]
% Wx  = Wind X    [m/s]
% Wy  = Wind Y    [m/s]

% Ver 1.0
% First created on 6/9/2014 by Hung Nguyen
% Last modified on 6/9/2014 by Hung Nguyen
% Reference:
% Shioji Maru (Nguyen, 2007)
% 

% Check of input and state dimensions
if (length(x) ~= 8),error('x-vector must have dimension 8!');end
if (length(ui) ~= 8),error('u-vector must have dimension 8!');end

% System parameters:
% Physical constants:
    rho = 104.61;       % divided by g = 9.81 m/s^2?
    rhoA= 0.1250;       % divided by g = 9.81 m/s^2?
    g   = 9.81;
    del_max  = 40*pi/180;
    deld_max = 5*pi/180;
    cpp_max  = 15;
    rad = asin(1.0)/90.0;
% Ship
    m  = 6.7418e4;
    mx = 2.671e3;
    my = 5.4032e4;
    IzJz=1.5184e7;
    L  = 46.00;
    d  =  2.85;
    Sw =480.00;
%  Hull parameters:
    ct = 4.32750e-3;
    Xv =-0.111418e-1;      Yv =-0.28689;        Nv =-0.140762;
    Xvv= 0.955500e-2;      Yvv=-0.554027;       Nvv=-0.974819e-1;
    Xr = 0.137031e-2;      Yr =-0.355849e-2;    Nr =-0.612311e-1;
    Xrr= 0.181790e-3;      Yrr= 0.128952;       Nrr= 0.117956e-1;
    Xvr= 0.176915;         Yvr=-0.136844;       Nvr=-0.169032;
% Propeller parameters:
    tp = 0.193;
    wp = 0.28989;
    rps= 5.0;
    Dp = 2.2;
    ctp0= 20.86;
    c0= 3.509177e-1; c1= 2.291329e-2; c2=-3.064803e-1;
    c3=-5.197373e-3; c4= 3.784367e-4; c5= 1.681795e-2;
    c6=-3.186904e-6; c7= 2.015531e-3; c8= 2.272834e-6;
    c9=-1.803216e-1;
% Rudder parameters:
    tr  = 0.215;
    aH  = 0.219;
    xR  =-22.649;
    xH  =-14.4;
    kx  = 0.6177;
    wr  = 0.22;
    AR  =  4.25;
    falp=  1.85;
    gamR= 0.4998;
    lR  =-0.77735;
    eps =(1.0-wr)/(1.0-wp);
    a  = 1;
% Thrututer
    Thb = 0.85e3;
    Ths = 0.65e3;
    Tlb = 17.5;
    Tls =-18.8;
% Wind:
    Aof =58.10;
    Aos =275.0;
    CX  =-0.322453;
    CY  =-0.951717;
    CN  =-0.108088;
% Rudder and CPP:
    Td  = 11.9;
    Te  = 2.8571;
% Masses and moments of inertia:
    m11 = (m+mx);
    m22 = (m+my);
    m33 = IzJz;    
 
% States and inputs:
ua    = x(1);  va    = x(2); r = x(3); psi = x(4);
del   = x(5);  cpp   = x(6);
del_c = ui(1); cpp_c = ui(2);
BT    = ui(3); ST    = ui(4);
Xcc   = ui(5); Ycc   = ui(6);
Wx    = ui(7); Wy    = ui(8);

% Rudder & cpp saturations and dynamics:
if abs(del_c) >= del_max,
   del_c = sign(del_c)*del_max;
end

if abs(cpp_c) >= cpp_max,
   cpp_c = sign(cpp_c)*cpp_max;
end

del_dot = (del_c-del)/(abs(del_c-del)*Td+a);

if abs(del_dot) >= deld_max,
   del_dot = sign(del_dot)*deld_max;
end

% Disturbances:
% Tidal current & water against velocities:
uc = Xcc*cos(psi) + Ycc*sin(psi);
vc = -Xcc*sin(psi) + Ycc*cos(psi);
u = ua - uc;
v = va - vc;

% Wind:
uW = Wx*cos(psi)+Wy*sin(psi)-ua;
vW = -Wx*sin(psi)+Wx*cos(psi)-va;
UW = sqrt(uW^2+vW^2);
XW = -0.5*rhoA*Aof*UW*CX*uW;
YW = -0.5*rhoA*Aos*UW*CY*vW;
NW = 0.5*rhoA*Aos*CN*uW*vW;

% Non-dimentional states:
U = sqrt(u*u+v*v);
vd = v/U;
rd = r*L/U;

% Hull:
XH = m*va*r+mx*vc*r-0.5*rho*Sw*ct*abs(u)*u+0.5*rho*L*d*U^2*(Xv*abs(vd)+...
     Xvr*vd*rd+Xv*abs(rd)+Xvv*vd*vd+Xrr*rd*rd);
YH = -m*ua*r-my*uc*r+0.5*rho*L*d*U^2*(Yv*vd+Yvr*vd*rd+Yr*rd+...
     Yvv*abs(vd)*vd+Yrr*rd*abs(rd));
NH = 0.5*rho*L^2*d*U^2*(Nv*vd+Nvr*abs(vd)*rd+Nr*rd+Nvv*vd*abs(vd)...
     +Nrr*rd*abs(rd));

% Propeller:
Jp = (1-wp)/(rps*Dp)*u;
cppd = cpp-ctp0;
XP = (1-tp)*rho*rps^2*Dp^4*(c0+c1*cppd+c2*Jp+c3*cppd*Jp+c4*cppd^2+...
      c5*Jp^2+c6*cppd^2*Jp^2+c7*cppd*Jp^2+c8*cppd^3+c9*Jp^3);
YP = 0;
NP = 0;
  
% Thrusters:
XT = 0;
YT = Thb*BT/10 + Ths*ST/10;
NT = Thb*Tlb*BT/10 + Ths*Tls*ST/10;

% Rudder:
UR = (eps-kx)*(1-wp)*u+kx*0.7*pi*Dp*rps*tan(cpp*rad);
%URd = UR/U;
%alR = del+atan(gamR*(vd+lR*rd)/URd);
%FN = 0.5*rho*AR*falp*UR^2*sin(alR);
FN = 0.5*rho*AR*falp*UR*(UR*sin(del)+(v+lR*L*r)*gamR*cos(del));
XR = -(1-tr)*FN*sin(del);
YR = -(1+aH)*FN*cos(del);
NR = -(xR+aH*xH)*FN*cos(del);

% Forces and moments:
X = XH + XP + XR + XT + XW;
Y = YH + YP + YR + YT + YW;
N = NH + NP + NR + NT + NW;

% Return derivatives:ps
xdot = [X/m11
        Y/m22
        N/m33
        r
        del_dot
        (cpp_c-cpp)/(abs(cpp_c-cpp)*Te+a)
        ua*cos(psi)-va*sin(psi)
        ua*sin(psi)+va*cos(psi)];
    
% End of function

