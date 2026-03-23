% function [sys,x0,str,ts] = JapaneseModel(t,x,u,flag)
% This function is based on the following paper:
% Fukui, Y., Yokota, H., Yano, H., Kondo, M., Nakano, T., and Yoshimura, Y. (2017). 
% 4-DOF mathematical model for manoeuvring simulation including roll motion. The Journal of JIN.
% First Created on 17/11/2020 by Hung Nguyen
% Last Modified on 26/06/2022 by Hung Nguyen for Bluefin
% 
function xdot = Bluefin4DOFModel(x,ui)

%function sys=mdlDerivatives(t,x,u)

% [xdot,U] = container(x,ui) returns the speed U in m/s (optionally) and the 
% time derivative of the state vector: x = [ u v r x y psi p phi delta n ]'  for
% a free running model (Japanese) L = 1.725 m, where
%
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

%
% The input vector is :
% ui      = [ delta_c n1_c n2_c]'  where
% delta_c = commanded rudder angle          (rad)
% n1_c     = commanded shaft velocity vector (rpm)  
% n2_c    = commanded thruster velocity      (rpm)

% Check of input and state dimensions
if (length(x) ~= 11),error('x-vector must have dimension 11 !');end
if (length(ui) ~= 3),error('u-vector must have dimension  3 !');end

% Normalization variables
L = 1.725;                     % length of ship (m)
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
pd   = x(3)*L/U;   p = x(3);
rd   = x(4)*L/U;   r = x(4);
% x(5) xpos
% x(6) ypos
phi = x(7);
psi = x(8); 
delta = x(9);     
n1   = x(10); % rps
n2   = x(11); % rps

% Parameters (Japanese Model):
% Some information on the boat (Fukui et al 2017):
    Lpp = 1.725; L = Lpp;   % Length
    B   = 0.5;              % Breadth
% draft
    dm = 0.193; d = dm;
    trim = 0.0;
    
% displacement 
    disp = 0.06455;
    xG = -0.1;
    Dp = 0.1;
    Dpph = 0.8791;
%    rudaratio = 1.4697; % lambda
    PpDp = 1.103;
    dpDp = 1.32;
% rudder aspect ratio 
    lambda = 1.4697;
% eta (=Dp/rudder height)
    eta   = 0.879;
%    ARpLd = 1/46.7;
    %AR    = L*d*ARpLd
    AR  =  0.0091;
    ARpLd = AR/(L*d);
    xR    = -1.05309;   % of Bluefin!
    GM    = 1.87;
    zG    = 0.005;  % a tentative value!
    zR    = -0.01;
    zH    = 0.02;   % a tentative value!
   
% (Hoorn's: aH = 0.237;xR = -0.5;xH = -0.45;zR = 0.033;cRX = 0.6175;)
% Propellers' nominate rpm and diameter: using 1 for single prop single
% rudder ship:
    n1s  = n1_c;
    n2s  = n2_c;
    DPs  = Dp;

% ratios of interaction coefficients (passenger ferry):
    onet    =  0.859;     % 1-t
    onew    =  0.806;     % 1-w
    onetR   =  0.857;     % 1-tR
    oneaH   =  1.403; aH = oneaH-1;
    xH      = -0.646;
    gR0     =  0.394; cg = -0.53;
    gR      = gR0*(1+cg*abs(phi));
    ldR     = -0.795;           % just from the Yoshimura (2017)
    epsi    =  0.740;
    kappa   =  0.810;
    eta     =  0.140;
  % kx      =  0.140;
% Japanese Model's coefficients (Passenger Vessel):
% Surge (X) forces:
    Xd0     = -0.0212; cx0  = -0.02;
    Xdrph  =  0.0092;
    Xdbb    = -0.0348; cxbb = 2.10;
    Xdbrmdy = -0.0957;
    Xdrr    = -0.0070; cxrr = 3.74;
    Xdbbbb  = -0.0018;
% Sway (Y) forces:    
    Ydph    =  0.0053;
    Ydb     =  0.2501; cyb  = -0.14;
    Ydrmdx  =  0.0346; cyr  = -0.61;
    Ydbbph  = -0.2979;
    Ydbrph  =  0.6308;
    Ydrrph  = -0.0854;
    Ydbbb   =  2.6087;
    Ydbbr   = -1.7091;
    Ydbrr   =  1.1682;
    Ydrrr   = -0.0461;
% Roll (K) moments:
    Kdph    = -0.0185;
    Kdb     = -0.2586;
    Kdr     =  0.0532;
    Kdbbph  =  0.2229;
    Kdbrph   = 0.5374;
    Kdrrph  = -0.0928;
    Kdbbb   = -0.7293;
    Kdbbr   =  1.1474;
    Kdbrr   = -0.3351;
    Kdrrr   = -0.0132;
% Yaw (N) moments:    
    Ndph    = -0.0086;
    Ndb     =  0.0966; cnb = 0.22;
    Ndr     = -0.0513; cnr = -0.62;
    Ndbbph  = -0.2510;
    Ndbrph  =  0.0722;
    Ndrrph  = -0.0172;
    Ndbbb   =  0.4218;
    Ndbbr   = -0.8629;
    Ndbrr   =  0.1459;
    Ndrrr   = -0.0439;
    
% All above hydrodynamic coefficients were double-checked!

% Other parameters which are missing in papers:
g = 9.81;rho = 1000;
m = 64.55;
mx = 3.662;
my = 62.7366;
Ix = 0.567;    % a tentative value
Iz = 9.6038;
Jx = 0.6309;
Jz = 10.2347;
%Iz = 9.6038;
%Jz = 0.6309;
%IzJz=10.2347;
%g = 9.81; 
Cb = 0.7878;  % Bluefin's
weights = 64.55; 
GMd = 0.087/L;
% 
W     = weights*g/(rho*L^2*U^2/2);

m11 = (m+mx);
m22 = (m+my);
m33 = (Ix+Jx);
m44 = (Iz+Jz);

% Rudder saturation and dynamics
if abs(delta_c) >= delta_max*pi/180,
   delta_c = sign(delta_c)*delta_max*pi/180;
end

delta_dot = delta_c - delta;

if abs(delta_dot) >= Ddelta_max*pi/180,
   delta_dot = sign(delta_dot)*Ddelta_max*pi/180;
end

%n_dot = 0;

n1_dot = n1s - n1;
n2_dot = n2s - n2;

if abs(n1_dot) >= Nc_max,
   n1_dot = sign(n1_dot)*Nc_max;
end
if abs(n2_dot) >= Nc_max,
   n2_dot = sign(n2_dot)*Nc_max;
end

%n3_dot = n3s - n3;
%n4_dot = n4s - n4;

%if abs(n3_dot) >= Nc_max,
%   n3_dot = sign(n3_dot)*Nc_max;
%end
%if abs(n4_dot) >= Nc_max,
%   n4_dot = sign(n4_dot)*Nc_max;
%end

% Propeller:
% Propeller forces and moments:
J = onew*u/(n1s*DPs);
a0 = 0.3267; a1 = -0.2297; a2 = -0.1607;  % From Yoshimura (1988)
KT = a0+a1*J+a2*J^2;
% The following is for a single propeller ship (equivalent to 2-1 ship):
XdP = abs(n1s)*n1s*onet*KT*(DPs^4)/(0.5*L*d*U^2);
%XdP = 2*n1s^2*onet*KT*(DPs^4)/(L*d*U^2);

% Rudder forces and moments:
udR = epsi*(onew)*sqrt(eta*((1+kappa*sqrt(1+8*KT/(pi*J^2))-1)^2)+(1-eta));
vdR = -gR*(b-ldR*rd+(p*(zR-zG)/U));  % equation (3.8)
UdR = sqrt(udR^2+vdR^2);
%UdR  = (epsi-kx)*onew*x(1)/U;
%gamR = gR*(1+cg*abs(phi));
alphaR  = delta-atan2(-vdR,udR);
%bR = gR(b-rd*ldR);
FdN = -(ARpLd)*(6.13*lambda/(2.25+lambda))*UdR^2*sin(alphaR);

% The sign (-) was included in FdN:
XdR = (onetR)*FdN*sin(delta)*cos(phi);
YdR = (1+aH)*FdN*cos(delta)*cos(phi);
KdR = zR*YdR/L;
%NdR = (xR+aH*xH)*FdN*cos(delta)*cos(d)*cos(phi);
NdR = (xR+aH*xH)*FdN*cos(delta)*cos(phi);

% Hydrodynamic forces and moments:
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

C44  = g*m*GM;
a    = 0.5;    % unknown value, just a tentative value
B44  = 2*a/pi*sqrt(g*m*GM*(Ix+Jx));
KdH2 = zG*YdH-B44*p-C44*phi-(zR-zG)*YdR;

% Bow thruster's forces and moments:
Dbt = 0.033;
xB = 0.45;   % tentative value
zB = -0.05;  % tentative value
KBT = 0.026;
FBT = abs(n2s)*n2s*KBT/(0.5*rho*L*d*U^2);
YdB = FBT;
KdB = zB*FBT/L; 
NbB = xB*FBT/L;

% Overall forces and moments:
Xd = XdH + XdP + XdR;
Yd = YdH + YdR + YdB;
Kd = KdH+KdH2+KdR+KdB;
Nd = NdH + NdR -xG*Yd + NbB;

% vdot term:
        vdot = (Yd*(0.5*rho*L*d*U^2)-m11*u*r)/m22;

        %vdot = 0;
% Return derivatives:
% The following equations are for Japanese Model (twin prop, single rudder)
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
 end      
% End of function
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

