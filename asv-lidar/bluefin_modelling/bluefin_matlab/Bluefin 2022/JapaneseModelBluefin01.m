% function [sys,x0,str,ts] = JapaneseModelBluefin01(t,x,u,flag)
% This function is based on the following paper:
% Simple Simulation Model for Manoeuvring Ship Motion
% of single-propeller and Single-rudder ship
% Masatoshi KONDO, Hiroyuki YANO, Yo FUKUI and Yasuo YOSHIMURA (2015).
% Simple simulation model for manoeuvring ship motion of twin-propeller
% and single-rudder ships. The Journal of JIN.
% URL: https://www.jstage.jst.go.jp/article/jin/133/0/133_28/_article/-char/ja/
% Yoshimura, Y., Kondo, M., Nakano, T and Yamashita, R. (2017).
% Equivalent simple mathematical model for the manoeuvrability of
% twin-propeller ships under the same propeller-rps
% First Created on 17/11/2020 by Hung Nguyen
% Last modified for Bluefin by Hung Nguyen on 23/6/2022
% Some coefficients mx, my, Ix, Iy, Ix, Jx, Jz etc. are not available in
% the papers > deduce their numerical values.

function xdot = JapaneseModelBluefin01(x,ui)

%function sys=mdlDerivatives(t,x,u)

% [xdot,U] = container(x,ui) returns the speed U in m/s (optionally) and the 
% time derivative of the state vector: x = [ u v r x y psi p phi delta n ]'  for
% a free running model Bluefin L = 1.747 m, where
%
% x(1)=u        = surge velocity          (m/s)
% x(2)=v        = sway velocity           (m/s)
% x(3)=r        = yaw velocity            (rad/s)
% x(4)=x        = position in x-direction (m)
% x(5)=y        = position in y-direction (m)
% x(6)=psi      = yaw angle               (rad)
% x(7)=delta    = actual rudder angle     (rad)
% x(8)=n1       = propeller               (rpm)
% x(9)=n2       = bow thruster            (rpm)

%
% The input vector is :
% ui      = [ delta_c n1_c n2]'  where
% delta_c = commanded rudder angle          (rad)
% n1_c     = commanded shaft velocity vector (rpm)  
% n2_c   = commanded thruster shaft velocity [rpm]


% Check of input and state dimensions
if (length(x) ~= 9),error('x-vector must have dimension 9 !');end
if (length(ui) ~= 3),error('u-vector must have dimension  3 !');end

% Normalization variables
L = 2.344;                     % length of ship (m)
U = sqrt(x(1)^2 + x(2)^2);     % service speed (m/s)

% Drift angle, to be assumed to be betaR:
b = -asin(x(2)/U);
%b = -atan2(x(2),x(1));

% Check service speed
%if U <= 0,error('The ship must have speed greater than zero');end
%if x(10) <= 0,error('The propeller rpm must be greater than zero');end

delta_max  = 40;             % max rudder angle (deg)
Ddelta_max = 20;             % max rudder rate (deg/s)
Nc_max     = 1000/60;        % max rps

% Inputs and non-dimensional state variables:
delta_c = ui(1); 
n1_c    = ui(2)/60;  % n1_c in rps
n2_c    = ui(3)/60;  % n2_c in rps

ud   = x(1)/U;     u = x(1);
vd   = x(2)/U;     v = x(2);
rd   = x(3)*L/U;   r = x(3);
% x(4) xpos
% x(5) ypos
psi = x(6); 
delta = x(7);     
n1   = x(8); % rps
n2   = x(9);


% Parameters (Japanese Model):
% Some information on the boat:
    Lpp = 1.725; L = Lpp;
    B   = 0.50;
% draft
    dm = 0.10; d = dm;
    trim = 0.0;
% displacement 
    disp = 0.06455;
    xG = -0.1;
    Dp = 0.0756;
    PpDp = 1.103;
    dpDp = 1.32;
% rudder aspect ratio 
    lambda = 1.470;
% eta (=Dp/rudder height) 
    eta   = 0.879;
    ARpLd = 1/46.7;
    AR    = L*d*ARpLd;
    xR    = -0.5;   % of Bluefin!
   
% (Hoorn's: aH = 0.237;xR = -0.5;xH = -0.45;zR = 0.033;cRX = 0.6175;)
% Propellers' nominate rpm and diameter:
    n1s  = n1_c/sqrt(2);
    n2s  = n2_c/sqrt(2);
    DPs  = sqrt(2)*Dp;

% ratios of interaction coefficients:
    onet    =  0.797;
    onew    =  0.856;
    onetR   =  0.853;
    aH      =  0.402;
    xH      = -0.646;
    gR      =  0.393;
    ldR     = -0.994;           % just from the Yoshimura (2017)
    epsi    =  0.737;
    kappa   =  0.168;
    kx      =  0.124;
% Japanese Model's coefficients:
    Xd0     = -0.0212;
    Xdbb    = -0.0348;
    Xdbrmdy = -0.0957;
    Xdrr    = -0.0070;
    Xdbbbb  = -0.0018;
    Ydb     =  0.2501;
    Ydrmdx  =  0.0346;
    Ydbbb   =  2.6087;
    Ydbbr   = -1.7091;
    Ydbrr   =  1.1682;
    Ydrrr   = -0.0461;
    Ndb     =  0.0966;
    Ndr     = -0.0513;
    Ndbbb   =  0.4218;
    Ndbbr   = -0.8629;
    Ndbrr   =  0.1459;
    Ndrrr   = -0.0439;

% Conversion Factor:
    cf = 0.8923; % = ratio fo L/B of to that of Japanese model
%    cf = 1.0;
% ratios of interaction coefficients:
    onet    =  0.797*cf;
    onew    =  0.856*cf;
    onetR   =  0.853*cf;
    oneaH   =  1.403*cf; aH = oneaH-1;
%    aH      =  0.402*cf;
    xH      = -0.646*cf;
%    gR      =  0.393*cf;
    gR0     =  0.394*cf; cg = -0.53*cf;
%    gR      = gR0*(1+cg*abs(phi));    
    ldR     = -0.994*cf;           
    epsi    =  0.737*cf;
    kappa   =  0.168*cf;
%    kx      =  0.124*cf;
% Japanese Model's coefficients:
% Surge terms:
    Xd0     = -0.0212*cf;  cx0  = -0.02*cf;
    Xdrph  =  0.0092*cf;
    Xdbb    = -0.0348*cf;  cxbb = 2.10*cf;
    Xdbrmdy = -0.0957*cf;
    Xdrr    = -0.0070*cf;  cxrr = 3.74*cf;
    Xdbbbb  = -0.0018*cf;
% coeff01 = [Xd0 Xdrph Xdbb Xdbrmdy Xdrr Xdbbbb]'
% coeff02 = [cx0 cxbb cxrr]'
 
% Sway terms:    
    Ydph    =  0.0053*cf;
    Ydb     =  0.2501*cf; cyb  = -0.14*cf;
    Ydrmdx  =  0.0346*cf; cyr  = -0.61*cf;
    Ydbbph  = -0.2979*cf;
    Ydbrph  =  0.6308*cf;
    Ydrrph  = -0.0854*cf;
    Ydbbb   =  2.6087*cf;
    Ydbbr   = -1.7091*cf;
    Ydbrr   =  1.1682*cf;
    Ydrrr   = -0.0461*cf;
%  coeff03 = [Ydph Ydb Ydrmdx Ydbbph Ydbrph Ydrrph Ydbbb Ydbbr Ydbrr Ydrrr]'
%  coeff04 = [cyb cyr]'
  
% Roll (K) moments:
    Kdph    = -0.0185*cf;
    Kdb     = -0.2586*cf;
    Kdr     =  0.0532*cf;
    Kdbbph  =  0.2229*cf;
    Kdbrph   = 0.5374*cf;
    Kdrrph  = -0.0928*cf;
    Kdbbb   = -0.7293*cf;
    Kdbbr   =  1.1474*cf;
    Kdbrr   = -0.3351*cf;
    Kdrrr   = -0.0132*cf;
%  coeff05 = [Kdph Kdb Kdb Kdbbph Kdbrph Kdbbb Kdbbr Kdbrr Kdrrr]'
  
% Yaw (N) moments:    
    Ndph    = -0.0086*cf;
    Ndb     =  0.0966*cf; cnb = 0.22*cf;
    Ndr     = -0.0513*cf; cnr = -0.62*cf;
    Ndbbph  = -0.2510*cf;
    Ndbrph  =  0.0722*cf;
    Ndrrph  = -0.0172*cf;
    Ndbbb   =  0.4218*cf;
    Ndbbr   = -0.8629*cf;
    Ndbrr   =  0.1459*cf;
    Ndrrr   = -0.0439*cf;
%  coeff06 = [Ndph Ndb Ndr Ndbbph Ndbrph Ndrrph Ndbbb Ndbbr Ndbrr Ndbrr]'
%  coeff = [cnb cnr]'
  
% Other parameters which are missing in papers:

m = 64.55;
mx = 0.0375*m;my = 0.8929*m;
Ix = 2.5*0.6987; Iz = 22.5*0.6987;
Jx = 0.35*0.6987;
Jz = 22.5*0.6987;
g = 9.81; 
Cb = 0.950;   % Bluefin's
weights = 64.55*0.6987; rho = 1000;
GMd = 1.87/L;
W     = weights*g/(rho*L^2*U^2/2);

m11 = (m+mx);
m22 = (m+my);
m44 = (Iz+Jz);

% Rudder saturation and dynamics
if abs(delta_c) >= delta_max*pi/180,
   delta_c = sign(delta_c)*delta_max*pi/180;
end

delta_dot = delta_c - delta;

if abs(delta_dot) >= Ddelta_max*pi/180,
   delta_dot = sign(delta_dot)*Ddelta_max*pi/180;
end

n1_dot = n1s - n1;
n2_dot = n2s - n2;

if abs(n1_dot) >= Nc_max,
   n1_dot = sign(n1_dot)*Nc_max;
end
if abs(n2_dot) >= Nc_max,
   n2_dot = sign(n2_dot)*Nc_max;
end


% Propeller:
J = onew*u/(n1s*DPs);
a0 = 0.3267; a1 = -0.2297; a2 = -0.1607;  % From Yoshimura (1988)
KT = a0+a1*J+a2*J^2;
XdP = (abs(n1s)*n1s)*onet*KT*(DPs^4)/(L*d*U^2);

% Rudder forces and moments:
udR = epsi*(onew)*sqrt(eta*((1+kappa*sqrt(1+8*KT/(pi*J^2))-1)^2)+(1-eta));
vdR = gR*(b-ldR*rd);
UdR = sqrt(udR^2+vdR^2);
alphaR  = delta+atan2(vdR,udR);
FdN = -(ARpLd)*(6.13*lambda/(2.25+lambda))*UdR^2*sin(alphaR);

% Rudder model by Bluefin:
XdR = (onetR)*FdN*sin(delta);
YdR = (1+aH)*FdN*cos(delta);
NdR = (xR+aH*xH)*FdN*cos(delta);

% Bow thruster:
zG = 0.152;z0 = 0.0365;zST = zG-z0;zBT = zST;
xB = 0.785;
xS = -0.455;

% Thrust is proportional to n^2
KT3 = 0.0360;
TbB = KT3*abs(n2)*n2/(0.5*U^2*L^2*rho);
NbB = xB*TbB;

% Hydrodynamic forces and moments:
XdH = Xd0 + Xdbb*b^2+Xdbrmdy*b*rd+Xdrr*rd^2+Xdbbbb*b^4;
YdH = Ydb*b+Ydrmdx*rd+Ydbbb*b^3+Ydbbr*b^2*rd+Ydbrr*b*rd^2+Ydrrr*rd^3;
NdH = Ndb*b+Ndr*rd+Ndbbb*b^3+Ndbbr*b^2*rd+Ndbrr*b*rd^2+Ndrrr*rd^3;

% Overall forces and moments:
Xd = XdH + XdP + XdR;
Yd = YdH + YdR + TbB;
Nd = NdH + NdR -xG*Yd + NbB;

% Calculation of state derivatives

% Return derivatives:
% The following equations are for Japanese Model (twin prop, single rudder)
xdot = [(Xd*(0.5*rho*L*d*U^2)+m22*v*r)/m11
        (Yd*(0.5*rho*L*d*U^2)-m11*u*r)/m22
        Nd*(0.5*rho*L^2*d*U^2)/m44
        cos(psi)*u-sin(psi)*v
        sin(psi)*u+cos(psi)*v
        r
        delta_dot 
        n1_dot                 
        n2_dot];
   
end
       
% End of function


