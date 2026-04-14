%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Vehicle guidance for obstacle avoidance example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Kevin Passino
%   Version: 1/25/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear                % Initialize memory

xmin=[0; 0];         % Set edges of region want to search in
xmax=[30;30];

Nsteps=500;            % Maximum number of steps to produce

% Next set the parameters of the vehicle:

lambda=0.1;  % Step size to take in chosen direction at each move
Ns=16;        % Number of points on circular pattern to sense
r=1;          % Sensing radius
xs=0*ones(2,Ns); % Initialize
Jo(:,1)=0*ones(Ns,1);
Jg(:,1)=0*ones(Ns,1);
J(:,1)=0*ones(Ns,1);
theta(:,1)=0*ones(Ns,1);
for m=2:Ns  % Compute the angles to be used around the circle
	theta(m,1)=theta(m-1,1)+(pi/180)*(360/Ns); 
end

% Goal position of vehicle
xgoal=[25; 25];

% Initial vehicle position
x=[5; 5];

% Weighting parameters for planning (sets priority for being aggresive
% in the direction of the goal vs. avoiding obstacles
w1=1;
w2=1.0000e-04;

% Allocate memory 

x(:,2:Nsteps)=0*ones(2,Nsteps-1);

% The obstacles:

figure(1)
clf
% Plot initial and final positions
plot(5,5,'s',25,25,'x')
axis([0 30 0 30])
hold on
xlabel('x');
ylabel('y');
title('Obstacles (o), initial vehicle (square) and goal (x) positions');
hold on
% Plot obstacle positions (sets obstaclefunction)
plot(20,15,'o',8,10,'o',10,10,'o',12,10,'o',24,20,'o',18,20,'o')
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the functions:

xx=0:31/100:30;   % For our function the range of values we are considering
yy=xx;

% Compute the obstacle and goal functions

for jj=1:length(xx)
	for ii=1:length(yy)
		zz(ii,jj)=obstaclefunction([xx(jj);yy(ii)],w1);
	end
end
for jj=1:length(xx)
	for ii=1:length(yy)
		zzz(ii,jj)=goalfunction([xx(jj);yy(ii)],xgoal,w2);
	end
end

figure(2)
clf
surf(xx,yy,zz);
%colormap(jet)
% Use next line for generating plots to put in black and white documents.
colormap(white);
xlabel('x');
ylabel('y');
zlabel('w_1J_o');
title('Function w_1J_o showing (scaled) obstacle function values');


figure(3)
clf
contour(xx,yy,zz,25)
colormap(jet)
% Use next line for generating plots to put in black and white documents.
%colormap(white);
xlabel('x');
ylabel('y');
title('Contour map of w_1J_o and initial (square) and goal (x) positions');
hold on
% Plot initial and final positions
plot(5,5,'s',25,25,'x')
hold off

figure(4)
clf
surf(xx,yy,zzz);
view(82,26);
%colormap(jet)
% Use next line for generating plots to put in black and white documents.
colormap(white);
xlabel('x');
ylabel('y');
zlabel('w_2J_g');
title('Goal function (scaled)');
%rotate3d

figure(5)
clf
contour(xx,yy,zzz,25)
colormap(jet)
% Use next line for generating plots to put in black and white documents.
%colormap(gray);
xlabel('x');
ylabel('y');
title('Contour function of w_2J_g and initial (square) and goal (x) positions');
hold on
% Plot initial and final positions
plot(5,5,'s',25,25,'x')
hold off


figure(6)
clf
contour(xx,yy,zz+zzz,50)
colormap(jet)
% Use next line for generating plots to put in black and white documents.
%colormap(gray);
xlabel('x');
ylabel('y');
title('J=w_1J_o + w_2J_g and initial (square) and goal (x) positions');
hold on

% Plot initial and final positions
plot(5,5,'s',25,25,'x')

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start the simulation loop 

for k=1:Nsteps

	% Use projection to keep in boundaries (like hitting a wall and staying at it)
	
	x(:,k)=min(x(:,k),xmax);
	x(:,k)=max(x(:,k),xmin);

	% Sense points on circular pattern

	for m=1:Ns
		xs(:,m)=[x(1,k)+r*cos(theta(m,1)); x(2,k)+r*sin(theta(m,1))]; % Point on circular pattern
		Jo(m,1)=obstaclefunction(xs(:,m),w1); % Compute the obstace function (what is
											% sensed at each sensed point
		Jg(m,1)=goalfunction(xs(:,m),xgoal,w2); % Compute how well each point 
												% moves toward the goal
		J(m,1)=Jo(m,1)+Jg(m,1); % Compute function for opt. in planning
	end

	% Next pick the best direction
	
	[val,bestone]=min(J);
	
	% Then, update the vehicle position (pick best direction and move step of lambda that way)
		
	x(:,k+1)=[x(1,k)+lambda*cos(theta(bestone,1)); x(2,k)+lambda*sin(theta(bestone,1))];
	
	% But the vehicle is in a real environment so when it tries to move to that point it
	% only gets to near that point.  To simulate this we perterb the final position.
	
	Deltalambda=0.1*lambda*(2*rand-1); % Set the length perturbation to be up to 10% of the step size
	Deltatheta=2*pi*(2*rand-1); % Set to be 360deg variation from chosen direction
	x(:,k+1)=[x(1,k+1)+Deltalambda*cos(theta(bestone,1)+Deltatheta); ...
	           x(2,k+1)+Deltalambda*sin(theta(bestone,1)+Deltatheta)];
		
end % End main loop...



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Next, provide some plots of the results of the simulation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t=0:Nsteps;  % For use in plotting

figure(7) 
clf
plot(t,x(1,:),'k-',t,x(2,:),'k--')
ylabel('x, y')
xlabel('Iteration, k')
title('Vehicle trajectory (x solid, y dashed)')


figure(8) 
clf
contour(xx,yy,zz,25)
colormap(jet)
% Use next line for generating plots to put in black and white documents.
%colormap(gray);
xlabel('x');
ylabel('y');
title('Vehicle path to avoid obstacles and reach goal');

hold on

plot(x(1,:),x(2,:),'r-')

plot(5,5,'s',25,25,'x')

hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

