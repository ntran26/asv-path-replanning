function [sys,x0] = ShiojiMaruModel(t,x,u,flag)

% S-Function: ShiojiMaruModel.m
% 
% MMG model of Shioji Maru without effect of environmental disturbances
% 
% x = [u v r psi delta pitch xpos ypos];
% 
% u        = surge velocity           [m/s]
% v        = sway velocity            [m/s]
% r        = yaw velocity             [rad/s]
% psi      = yaw angle                [rad]
% delta    = actual rudder angle      [rad]
% pitch    = actual pitch angle       [rad]
% xpos     = position in x-axis       [m]
% ypos     = position in y-axis       [m]
%
% The inputs are:
%
% ui      = [delta_c pitch_c], where
% delta_c = commanded rudder angle [rad]
% pitch_c = commmaded pitch angle  [deg]
%
% Written by Hung Nguyen, Australian Maritime College
% on 29 May 2004;
%
% References:
% 1. Guidance and Control of Ocean Vehicles
% 2. Unpublished Papers (Tokyo University of Mercantile Marine)
% 3. Self-tuning Pole Assignment and Optimal Control Systems for Ships
%

% Derivative coefficients:

m      =    6.7418e+4;   mx    =    2.6710e+3;   my     =    5.4032e+4;
IzzJzz =    1.5184e+7;   Lpp   =         46.0;   d      =         2.85;
Sw     =        480.0;   Ct    =   4.32750e-3;   tR     =        0.215;
WR     =         0.22;   AR    =         4.25;   xR     =      -22.649;
xH     =        -14.4;   aH    =        0.219;   fa     =         1.85;
gamR   =       0.4998;   lR    =     -0.77735;   kx     =       0.6177;
Dp     =          2.2;   n     =          5.0;   tP     =        0.193;
WP     =      0.28989;   CT0   =        20.86;   C0     =  3.509177e-1;
C1     =  2.291329e-2;   C2    = -3.064803e-1;   C3     = -5.197373e-3;
C4     =  3.784367e-4;   C5    =  1.681795e-2;   C6     = -3.186904e-6;
C7     =  2.015531e-3;   C8    =  2.272834e-6;   C9     = -1.803216e-1;

Xv     = -0.111418e-1;   Yv    =    -0.286890;   Nv     =    -0.140762;
Xvr    =     0.176915;   Yvr   =    -0.136844;   Nvr    =    -0.169032;
Xr     =  0.137031e-2;   Yr    = -0.355849e-2;   Nr     = -0.612311e-1;
Xvv    =  0.955500e-2;   Yvv   =    -0.554027;   Nvv    = -0.974819e-1;
Xrr    =  0.181790e-3;   Yrr   =     0.128952;   Nrr    =  0.117956e-1;

TRUD   =         11.9;   TCPP  =       2.8571;   Aof    =        58.10;
Aos    =        275.0;   a     =            1;   CN     =    -0.108088;  
CX     =    -0.322456;   CY    =    -0.951717;   
m11    =       m + mx;   m12   =       m + my;   %URR    =       1.056; what for? 

% Coefficients for Forces and Moments induced by thrusters

%BT  =       0;     % not use bow thruster
%ST  =       0;     % not use stern thruster
xB  =    17.5;
xS  =   -18.8;
TB  =   850.0;
TS  =   650.0;

% Physical const.

rho = 104.61;
rhoA= 0.1250;
g   = 9.81;
rad = pi/180; deg = 180/pi;

% Initial surge velocity:

cpp = 14.5;                        % pitch angl in degrees
UU0 = 2.058+0.884*cpp-0.014*cpp^2; % surge velocity in knots
u0  = UU0*1852/3600;               % surge velocity in m/s


% S-function:

if flag == 1,
   
   % return state derivatives

   delta_c = u(1);
   pitch_c = u(2);
   BT      = u(3);
   ST      = u(4);

U      = sqrt(x(1)^2+x(2)^2);

uu     = x(1); v     = x(2); r      = x(3); psi   = x(4);
delta  = x(5); pitch = x(6); xpos   = x(7); ypos  = x(8); 

vv     = v/U; rr  = r*Lpp/U;


% Forces and moments induced by environmental disturbances:

% Forces and moments:

pitchPP = pitch-CT0;
JP  = (1-WP)*uu/(n*Dp);
eps = (1-WR)/(1-WP);
UR  = (eps-kx)*(1-WP)*uu+kx*0.7*pi*Dp*n*tan(pitch*rad);

FN  = 0.5*rho*AR*fa*(UR^2*sin(delta)+UR*gamR*(v+lR*Lpp*r)*cos(delta));

X   = m*v*r-0.5*rho*Sw*Ct*abs(uu)*uu+0.5*rho*Lpp*d*U^2*(Xv*abs(v)+...
      Xvr*vv*rr+Xr*abs(rr)+Xvv*vv*vv+Xrr*rr*rr)+...
      (1-tP)*rho*Lpp*Dp^2*(C0+C1*pitchPP+C2*JP+C3*pitchPP*JP+...
      C4*pitchPP^2+C5*JP^2+C6*pitchPP^2*JP^2+C7*pitchPP*JP^2+C8*pitchPP^3+...
      C9*JP^3)-(1-tR)*FN*sin(delta);
   
Y   = -m*uu*r+0.5*rho*Lpp*d*U^2*(Yv*vv+Yvr*vv*abs(rr)+Yr*rr+Yvv*vv*abs(vv)+...
      Yrr*rr*abs(rr))+TB*BT/10+TS*ST/10-(1+aH)*FN*cos(delta);
   
N   = 0.5*rho*Lpp^2*d*U^2*(Nv*vv+Nvr*abs(vv)*rr+Nr*rr+Nvv*vv*abs(vv)+...
      Nrr*rr*abs(rr))+TB*xB*BT/10+TS*xS*ST/10-(xR+aH*xH)*FN*cos(delta);

% Return derivatives:

sys(1) = X/m11;
sys(2) = Y/m12;
sys(3) = N/IzzJzz;
sys(4) = r;
sys(5) = (delta_c-delta)/(abs(delta_c-delta)*TRUD+a);
sys(6) = (pitch_c-pitch)/(abs(pitch_c-pitch)*TCPP+a);
sys(7) = uu*cos(psi)-v*sin(psi);
sys(8) = uu*sin(psi)+v*cos(psi);

      
      
elseif flag == 0,
    % return initial conditions
    sys=[8;0;8;-1;0;0];
    x0=[u0;0;0;0;0;0;0;0];
     
 
elseif flag == 3,
   
   % return outputs:
   sys=[x(1) x(2) x(3) x(4) x(5) x(6) x(7) x(8)];

else
    sys = [];

end
      
      
% end of function mmgmodel01

