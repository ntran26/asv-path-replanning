% JapaneseModelBluefin_Solver.m

clear

% Initial conditions:
% x(1)=u        = surge velocity          (m/s)
% x(2)=v        = sway velocity           (m/s)
% x(3)=r        = yaw velocity            (rad/s)
% x(4)=x        = position in x-direction (m)
% x(5)=y        = position in y-direction (m)
% x(6)=psi      = yaw angle               (rad)
% x(7)=delta    = actual rudder angle     (rad)
% x(8)=n1       = propeller 1   (rpm)
% x(9)=n2       = propeller 2   (rpm)

x = [0.5 0 0 0 0 0 0 1 1]';
u0 = [20*pi/180 1000 0]';

% simulation parameters
ss = 0.1;
%FT = 1.38;
FT = 200;
index = 0;

for k = 0:ss:FT
    index = index + 1;
    ui = u0;
% Euler's method:    
    xdot = JapaneseModelBluefin01(x,ui);
    x    = x + ss*xdot;

    if x(6) >= 2*pi,
        x(6) = x(6)-2*pi;
    elseif x(6) <= 0,
        x(6) = x(6) + 2*pi;
    else
        x(6) = x(6);
    end
    
% RK2 or RK4 method?

% store data:
    data(index,1) = ui(1);
    data(index,2) = ui(2);
    data(index,3) = x(1);
    data(index,4) = x(2);
    data(index,5) = x(3);
    data(index,6) = x(4);  % x-position
    data(index,7) = x(5);  % y-position
    data(index,8) = x(6);
    data(index,9) = x(7);
    data(index,10) = x(8);
    data(index,11) = k;   % time
end

% plotting
figure(1)
subplot(211);plot(data(:,11),data(:,3));grid
ylabel('u');
subplot(212);plot(data(:,11),data(:,4));grid
ylabel('v');

% to plot the trajectory:
figure(2)
plot(data(:,7),data(:,6));grid;axis('equal')


