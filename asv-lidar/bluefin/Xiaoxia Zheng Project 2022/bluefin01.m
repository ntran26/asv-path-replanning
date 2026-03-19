% Filename: bluefin01.m
% Ship Hull Dynamics and Position
% 

function xdot = bluefin01(x,ui)
 %
 %state vector
 %x = [psi yaw angle [rad]
 % r yaw rate [rad/s]
 % x position in x-axis [m]
 % y] position in y-axis [m]
 %Input vector
 %ui = [del_c
 %      0];
 %
 %System parameters

 T = 4.942; %seconds
 K = 0.9221;
 V = 0.5; % speed in m/s
 u = V; v = 0;
 
 psi = x(1);
 r = x(2);
 del = ui(1);

% Return Derivatives;

xdot = [r
       -r/T + (K/T) * del
       u*cos(psi) - v*sin(psi)
       u*sin(psi) + v*cos(psi)];

%End of function 
