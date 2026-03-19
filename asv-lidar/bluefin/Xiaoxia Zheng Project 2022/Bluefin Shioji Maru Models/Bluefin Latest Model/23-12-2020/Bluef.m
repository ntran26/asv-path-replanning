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
% x     = position in x-direction (m)        > x6
% y     = position in y-direction (m)        > x7
% The input vector is:
% ui      = [delta_c Xcc Ycc]' where
% delta_c = commanded rudder angle(rad) 
% Xcc = Current X [m/s]
% Ycc = Current Y [m/s]

% Initial conditions:
x = [0.35 0 0 0 0 0 0]';

% Inputs:

del = -35*pi/180;
del2 = del/2;
ui = [del 0.6 0.4]';

% Step size and time span:
ss = 0.1;
st = 200;
index = 0;
% Zigzag Tests %Flow is Zigzag test 20-20: rudder is 20 deg, change in yaw 
% is 20 deg
% for ii = 0:ss:st
%     index = index + 1;
%     
%       psi = x(4)-0;
%     if (psi >= del2)
%         ui(1) = -del;      
%     elseif (psi <= -del2)
%         ui(1) = del;
%     end

% Turning Circle test
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
% x8 = data(:,8);
% y8 = data(:,7);
% save run8
grid on; grid minor;
title('Trajectory at 20 Degree')
xlabel('Position in x-axis [m]')
ylabel('Position in y-axis [m]')
axis('equal');


    
    
    