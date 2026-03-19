% Filename: bluefin03_sim.m
% Using 2nd order runge-kutta (2RK) method to solve differential equations
% to simulate a surface vessel described by Nomoto's manoeuvring model
% Trdot + r = K*delta
% Desing PID autopilot!

clear

% Initial conditions;

x = [0 0 0 0]';

% Input vector;
%ui = [40*pi/180          % Rudder angle in radian
%      0];                % other input!

% Setcouse
psid = 60*pi/180;        % Setcourse in rad;
errormax = pi;
errormin = -errormax;

% Control gains:
K = [0.25 5 0.75]';

% initial error vector:
e = [0 0 0]';
ei0 = e(2);ed0 = e(3);

 % simulation parameters
 dt = 0.1;
 ST = 400;
 index = 0;

for ii = 0:dt:ST
 index = index + 1;
 % 2RK method
 ui = pidcontrol(e,K);
 delc = ui(1);
 k1 = dt*bluefin01(x,delc);
 k2 = dt*bluefin01(x+k1,delc);
 x = x + 0.5*(k1+k2);

 % Boundary of yaw angle:
 if x(1) > 2*pi
 x(1) = x(1) - 2*pi;
 elseif x(1) <= 0
 x(1) = x(1) + 2*pi;
 else x(1) = x(1);
 end 
 
 % Error:
 psi = x(1);
 err = psid - psi;
 if err > errormax
     err = 2*pi-errormax;
 elseif err < errormin
     err = 2*pi+errormin;
 else
     err = err;
 end
 
 % update the error vector:
    e(1) = err;
    ei = ei0 + e(1)*dt;
    e(2) = ei;
    ed = (e(1)-ed0)/dt;
    e(3) = ed;
    ed0 = e(1)

  % Store data
 data(index,1) = ii; % time [sec]
 data(index,2) = x(1); % yaw angle [rad]
 data(index,3) = x(2); % yaw rate [rad/sec] 
 data(index,4) = delc(1); % rudder angle [rad]
 data(index,5) = x(3); % x-pos
 data(index,6) = x(4); % y-pos
 data(index,7) = psid; % desired course [rad]

 end

% Visualize simulated system;
 figure(1)
 subplot(311);
 plot(data(:,1),data(:,2)*180/pi,data(:,1),data(:,7),'-k');grid
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
