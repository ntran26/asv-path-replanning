% Filename: Bluefin_with_disturbance_sim1.m
%
clear

% State variables:
% u     = surge velocity          (m/s)      > x1
% v     = sway velocity           (m/s)      > x2
% r     = yaw velocity            (rad/s)    > x3
% psi   = yaw angle               (rad)      > x4
% delta = actual rudder angle     (rad)      > x5
% x     = position in x-direction (m)        > x6
% y     = position in y-direction (m)        > x7
%
% The input vector is:
% ui      = [delta_n1 n1 BT Xcc Ycc Wx Wy]' where
% delta_c = commanded rudder angle   (rad)
% n1F  = propeller shaft speed (rps)  
% BT = Bow Thruster (notch/10)
% or BF = Bow Thruster (notch/10)
% Xcc = Current X [m/s]
% Ycc = Current Y [m/s]
% Wx  = Wind X    [m/s]
% Wy  = Wind Y    [m/s]

% Initial conditions:
%x = [-0.00399 0.00152 0 0 0 0 0]';
x = [0.00399 0 0 0 0 0 0]';

% Inputs:
ui = [-35*pi/180 0.5 0.25]';

% Step size and time span:
ss = 0.01;
st = 50;

index = 0;

for ii = 0:ss:st
    index = index + 1;
 
 % Euler method:
    xdot = Bluefin_without_disturbance_V5(x,ui);
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
plot(data(:,7),data(:,8));grid
axis('equal');
title('Turning circle trajectory');
xlabel('X-position [m]');
ylabel('Y-position [m]]');


    
    
    