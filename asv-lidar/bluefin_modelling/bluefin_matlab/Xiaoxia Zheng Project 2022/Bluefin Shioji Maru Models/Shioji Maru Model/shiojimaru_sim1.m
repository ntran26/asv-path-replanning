% Filename: shiojimaru_sim1.m
%
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

% Initial conditions:
x = [6.627 0 0 0 0 13 0 0]';

% Inputs:
ui = [20*pi/180 13 0 0 0 0 0 0]';

% Step size and time span:
ss = 0.01;
st = 600;

index = 0;

for ii = 0:ss:st
    index = index + 1;
 
 % Euler method:
    xdot = shiojimaru(x,ui);
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
  data(index,9) = x(8);  
end

plot(data(:,9),data(:,8));grid
axis('equal');

    
    
    