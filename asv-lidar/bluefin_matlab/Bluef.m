% Filename: Bluef.m
% Mathematical model for Bluefin, the training vessel model of
% University of Tasmania - Australian Maritime College
% By learning shiojimaru.sim1 by Hung Nguyen, the following fucntion can be
% created
clear
% State variables:
% u     = surge velocity          (m/s)      > x1
% v     = sway velocity           (m/s)      > x2
% r     = yaw velocity            (rad/s)    > x3
% psi   = yaw angle               (rad)      > x4
% delta = actual rudder angle     (rad)      > x5
% theta = pitch angle        (deg)           > x6
% x     = position in x-direction (m)        > x7
% y     = position in y-direction (m)        > x8
% The input vector is:
% ui      = [delta_c Xcc Ycc]' where
% delta_c = commanded rudder angle(rad)
% cpp_c  = commanded pitch angle (deg)  
% Xcc = Current X [m/s]
% Ycc = Current Y [m/s]

% Initial conditions:
x = [0.6 0 0 0 0 0 0]';

% Inputs:
ui = [-10*pi/180 0.6 0.25]';

% Step size and time span:
ss = 0.1;
st = 200;
index = 0;

for ii = 0:ss:st
    index = index + 1;
 
 % Euler method:
    xdot = Blue(x,ui);
    x    = x + ss*xdot;
  
  % Store data:
  data(index,1) = ii;
  data(index,2) = x(1);
  data(index,3) = x(2);
  data(index,4) = x(3);
  data(index,5) = x(4);    
  data(index,6) = x(5);
  data(index,7) = x(6);
  data(index,8) = x(7);  
end

plot(data(:,8),data(:,7));
grid on; grid minor;
title('Trajectory at 10 degree')
xlabel('Position in x-axis [m]')
ylabel('Position in y-axis [m]')
axis('equal');


    
    
    