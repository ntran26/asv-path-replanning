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
% n1  = propeller shaft speed (rps)  
% BT = Bow Thruster (notch/10)
% Xcc = Current X [m/s]
% Ycc = Current Y [m/s]
% Wx  = Wind X    [m/s]
% Wy  = Wind Y    [m/s]

% Initial conditions:
%x = [-0.00399 0.00152 0 0 0 0 0]';
%x = [u v r psi/rad del x y]';
x = [-0.06369 0.00493 20.34*pi/180 50.945*pi/180 10*pi/180 0 0]';

% Inputs:
ui = [10*pi/180 0.5 0]';

% Type of zigzag test:
zigzag = [10,10];

% Step size and time span:
%ss = 0.01;
ss = 0.01;
st = 500;

index = 0;

for ii = 0:ss:st
index = index + 1;
 
 % Euler method:
   % xdot = Bluefin_with_disturbance_V5_v2(x,ui);
  %  x    = x + ss*xdot;
% 2RK method
k1 = ss*Bluefin_without_disturbance_V6(x,ui);
k2 = ss*Bluefin_without_disturbance_V6(x+k1,ui);
x = x + 0.5*(k1+k2);
    
% Boundary of yaw angle:
%if x(4) > 2*pi
% x(4) = x(1) - 2*pi;
%elseif x(4) <= 0
% x(4) = x(4) + 2*pi;
%else x(4) = x(4);

%end
if x(4) >= zigzag(2)*pi/180
ui(1) = -zigzag(1)*pi/180;
elseif x(4) <= -zigzag(2)*pi/180
ui(1) = zigzag(1)*pi/180;
else
ui(1) = ui(1);
end   
    
  
  % Store data:
  data(index,1) = ii;
  data(index,2) = x(1);% surge velocity     (m/s)   > x1
  data(index,3) = x(2);% sway velocity      (m/s)   > x2
  data(index,4) = x(3);% yaw velocity       (rad/s) > x3
  data(index,5) = x(4);% yaw angle          (rad)   > x4    
  data(index,6) = x(5);% actual rudder angle(rad)   > x5
  data(index,7) = x(6);% position in x-direction (m)> x6
  data(index,8) = x(7);% position in y-direction (m)> x7   
  
end
%plot(data(:,7),data(:,8));grid
%axis('equal');

% Visualize simulated system;
figure(1)
subplot(311);
plot(data(:,1),data(:,5)*180/pi);grid
title('Ploting yaw, yawrate, rudder');
ylabel('Yaw [deg]');

subplot(312);
plot(data(:,1),data(:,4)*180/pi);grid
ylabel('Yaw rate [deg/s]');

subplot(313);
plot(data(:,1),data(:,6)*180/pi);grid
ylabel('Rudder [deg]');
xlabel('Time [sec]');

figure(2)
plot(data(:,7),data(:,8)*180/pi);grid
%axis('equal')1

figure(3)
plot(data(:,6)*(180/pi),data(:,5)*(180/pi));grid
%axis('equal')


    
    
    