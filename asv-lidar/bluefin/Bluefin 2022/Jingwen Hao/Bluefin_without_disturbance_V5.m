% Filename: Bluefin_with_disturbance.m
% Mathematical model for Shioji Maru, the training vessel of
% Tokyo University of Marine Science and Technology

function xdot = Bluefin_without_disturbance_V5(x,ui)
% State variables:
% u     = surge velocity          (m/s)      > x1
% v     = sway velocity           (m/s)      > x2
% r     = yaw rate                (rad/s)    > x3
% psi   = yaw angle               (rad)      > x4
% delta = actual rudder angle     (rad)      > x5
% x     = position in x-direction (m)        > x6
% y     = position in y-direction (m)        > x7
%
% The input vector is:
% ui      = [delta_c n1 BT Xcc Ycc Wx Wy]' where
% delta_c = commanded rudder angle   (rad)
% n1  = propeller shaft speed (rps)
% BT = Bow Thruster (notch/10)
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
%if (length(x) ~= 7),error('x-vector must have dimension 7!');end
%if (length(ui) ~= 7),error('u-vector must have dimension 7!');end

% System parameters:
% Physical constants:
    rho = 1046.1;       % divided by g = 9.81 m/s^2?
    rhoA= 0.1250;       % divided by g = 9.81 m/s^2?
    g   = 9.81;
    del_max  = 40*pi/180;
    deld_max = 5*pi/180;
    rad = asin(1.0)/90.0;
% Ship
    m  = 64.55;
   % m = m*2/(rho*L*L*L);
    mx = 2.557374;
    my = 5.173345;
    Iz = 10.17;
   %Iz = Iz*2/(rho*L*L*L*L*L);
    L  = 1.569;
   %L = L/L;
    d  =  0.193;
    %d = d/d;
    Sw = 480.00;
%  Hull parameters:
    ct = 4.32750e-3;
    Xv = 0.9610;         Yv = 0;              Nv = -0.0000;  % by 10 zigzag data
    Xvv = -1.1143;       Yvv = 0.0068;        Nvv = 0.0030;
    Xr = -0.0005;        Yr = -0.0000;        Nr = -0.0000;
    Xrr = -0.0000;       Yrr = -0.0000;       Nrr = -0.0000;
    Xvr = -0.00021;      Yvr = -0.0000;       Nvr = -0.0000;
    
    % Xv = -0.0278;      Yv = -0.0049;        Nv = 5.2042*(10^-17);
  % Xvv = 0.0133;     Yvv = 0.0095;       Nvv = 0.0030;
   % Xr = -0.0005;       Yr = -0.0001;        Nr = -2.7529*(10^-21);
  % Xrr = -0.0000;      Yrr = -0.0000;       Nrr = -1.4863*(10^-25);
   % Xvr = -0.0000;      Yvr = -0.0000;       Nvr = 7.6233*(10^-21);
% Propeller parameters:
    tp = 0.193;
    wp = 0.28989;
    Dp = 2.2;
    Fp = 8.1928;
% Rudder parameters:
    tr  = 0.215;
    aH  = 0.219;
    xR  =-22.649;
    xH  =-14.4;
    kx  = 0.6177;
    wr  = 0.22;
    AR  =  0.162;
    falp=  1.85;
    gamR= 0.4998;
    lR  =-0.77735;
   eps =(1.0-wr)/(1.0-wp);
    a  = 1;
% Thrututer
   % Thb = 0.65e3;
   %Tlb =-18.8;
   %Ft = 10;
   Ft = 1.20834;
   Tlb = 0.38;
   
% Rudder:
    Td  = 11.9;          % this is not applicable for Bluefin!
    Te  = 2.8571;
% Masses and moments of inertia:
    %m11 = (m+mx);
   % m22 = (m+my);
    %m33 = Iz; 
    
    m11 = (m+mx);
    m22 = (m+my);
    m33 = Iz;
    
% Non-dimentional states:
%U = sqrt(u*u+v*v);
U = sqrt(0.06369^2+0.00493^2);

% States and inputs:
u = x(1)/U;  
v = x(2)/U; 
r = x(3)*L/U; 
psi = x(4);
del = x(5); 
del_c = ui(1); 
n1F = ui(2); 
BF = ui(3); 
%Xcc   = ui(4); Ycc = ui(5);
%Wx = ui(6); Wy = ui(7);

% Non-dimentional states:
vd = v/U;
rd = r*L/U;    

% Rudder & cpp saturations and dynamics:
if abs(del_c) >= del_max,
   del_c = sign(del_c)*del_max;
end


del_dot = (del_c-del)/(abs(del_c-del)*Td+a);

if abs(del_dot) >= deld_max,
   del_dot = sign(del_dot)*deld_max;
end

% Non-dimentional states:
U = sqrt(u*u+v*v);
vd = v/U;
rd = r*L/U;

% Hull:
XH = m*v*r-0.5*rho*Sw*ct*abs(u)*u+0.5*rho*L*d*U^2*(Xv*abs(vd)+...
     Xvr*vd*rd+Xv*abs(rd)+Xvv*vd*vd+Xrr*rd*rd);
YH = -m*u+0.5*rho*L*d*U^2*(Yv*vd+Yvr*vd*rd+Yr*rd+...
     Yvv*abs(vd)*vd+Yrr*rd*abs(rd));
NH = 0.5*rho*L^2*d*U^2*(Nv*vd+Nvr*abs(vd)*rd+Nr*rd+Nvv*vd*abs(vd)...
     +Nrr*rd*abs(rd));

% Propeller:
%Jp = (1-wp)/(n1*Dp)*u;
%XP = (1-tp)*rho*n1^2*Dp^4;
XP = Fp*2*n1F;
YP = 0;
NP = 0;
  
% Thrusters:
XT = 0;
%YT = Thb*BT/10;
YT = Ft*4*BF;
NT = Tlb*YT;
%YT = 1.20834;
%NT = 1.20834*0.38;

% Rudder:
%UR = (eps-kx)*(1-wp)*u+kx*0.7*pi*Dp*n1;
%URd = UR/U;
%alR = del+atan(gamR*(vd+lR*rd)/URd);
%FN = 0.5*rho*AR*falp*UR^2*sin(alR);
%FN = 0.5*rho*AR*falp*UR*(UR*sin(del)+(v+lR*L*r)*gamR*cos(del));
%XR = -(1-tr)*FN*sin(del);
%YR = -(1+aH)*FN*cos(del);
%NR = -(xR+aH*xH)*FN*cos(del);

%FN = 576.2*0.475*(u^2);
FN = 576.2*0.0612*(u^2);
XR = FN*sin(del);
YR = FN*cos(del);
NR = 0.7*FN*cos(del);


% Forces and moments:
X = XH + XP + XR + XT;
Y = YH + YP + YR + YT;
N = NH + NP + NR + NT;

% Return derivatives:ps
xdot = [X*2/(m11*(rho*L*L*U*U))
        Y*2/(m22*(rho*L*L*U*U))
        N*2/(m33*(rho*L*L*L*U*U))
        r*L/U
        del_dot
        (u*cos(psi)-v*sin(psi))/U
        (u*sin(psi)+v*cos(psi))/U];
    
% End of function

