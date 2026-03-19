% Filename: bluefin02_sim.m
 % Using 2nd order runge-kutta (2RK) method to solve differential equations
 % to simulate a surface vessel described by Nomoto's manoeuvring model
 % Trdot + r = K*delta

clear

% Initial conditions;

x = [0 0 0 0]';

% Input vector;
 ui = [20*pi/180 % rudder angle in radi
       0]; % other input!

% Zigzag type:
 zigzag = [20,20];

% simulation parameters
 dt = 0.1;
 ST = 400;
 index = 0;

for ii = 0:dt:ST
 index = index + 1;
 % 2RK method
 k1 = dt*bluefin01(x,ui);
 k2 = dt*bluefin01(x+k1,ui);
 x = x + 0.5*(k1+k2);

 % for zig-zag test:
 if x(1) >= zigzag(2)*pi/180
    ui(1) = -zigzag(1)*pi/180;
 elseif x(1) <= -zigzag(2)*pi/180
    ui(1) = zigzag(1)*pi/180;
 else
    ui(1) = ui(1);
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
 plot(data(:,6),data(:,5));grid
 title('Trajectory');
 xlabel('Position in y-axis [m]');
 ylabel('Position in x-axis [m]');
 axis('equal');
