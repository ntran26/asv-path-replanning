% Filename: bluefin01_sim.m
% Using 2nd order runge-kutta (2RK) method to solve differential equations
% to simulate a surface vessel described by Nomoto's manoeuvring model
% Trdot + r = K*delta
%

clear

% Initial conditions;

x = [0 0 0 0]';

% Input vector;
 ui = [20*pi/180          % Rudder angle in radian
       0];                % other input!

 % simulation parameters
 dt = 0.01;
 ST = 100;
 index = 0;

for ii = 0:dt:ST
 index = index + 1;
 % 2RK method
 k1 = dt*bluefin01(x,ui);
 k2 = dt*bluefin01(x+k1,ui);
 x = x + 0.5*(k1+k2);

 % Boundary of yaw angle:
 if x(1) > 2*pi
 x(1) = x(1) - 2*pi;
 elseif x(1) <= 0
 x(1) = x(1) + 2*pi;
 else x(1) = x(1);
 end 

 % Store data
 data(index,1) = ii; % time [sec]
 data(index,2) = x(1); % yaw angle [rad]
 data(index,3) = x(2); % yaw rate [rad/sec] 
 data(index,4) = ui(1); % rudder angle [rad]
 data(index,5) = x(3); % x-pos
 data(index,6) = x(4); % y-pos

 end

% Visualize simulated system;
 figure(1)
 subplot(311);
 plot(data(:,1),data(:,2)*180/pi);grid
 title('Ploting yaw, yawrate, rudder');
 ylabel('Yaw [deg]');
 subplot(312);
 plot(data(:,1),data(:,3)*180/pi);grid
 ylabel('Yaw rate [deg/s]');
 subplot(313);
 plot(data(:,1),data(:,4)*180/pi);grid
 ylabel('Rudder [deg]');
 xlabel('Time [sec]');

figure(2)
 plot(data(:,6),data(:,5));grid;title('Trajectory');
 xlabel('Position in y-axis [m]');
 ylabel('Position in x-axis [m]');
 axis('equal')
