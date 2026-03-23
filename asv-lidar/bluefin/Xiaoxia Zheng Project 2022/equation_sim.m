% solver for equation

clear;

%initial condition:
x = [0;0];
% input:
u0 = [1];
ome =0.5;

% solver:
index = 0;
dt = 0.1;

for ii = 0:dt:20

    index = index + 1;
    u = u0;
    k1 = dt*equation(x,u,ii);
    k2 = dt*equation(x+k1,u,ii);
    x = x + 0.5*(k1+k2);
    
% store data:
    data(index,1) = ii;     % time
    data(index,2) = x(1);   % y
    data(index,3) = x(2);   % ydot
    
end

plot(data(:,1),data(:,2));grid

