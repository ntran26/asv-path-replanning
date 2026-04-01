%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Radial Basis Function Neural Controller for Tanker Ship Heading Regulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% By: Kevin Passino 
% Version: 1/21/00
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear		% Clear all variables in memory
pause off

% Initialize ship parameters 
% (can test two conditions, "ballast" or "full"):

ell=350;			% Length of the ship (in meters)
u=5;				% Nominal speed (in meters/sec)
abar=1;             % Parameters for nonlinearity
bbar=1;

% Define the reference model (we use a first order transfer function 
% k_r/(s+a_r)):

a_r=1/150;
k_r=1/150;

% Adaptation gain:

eta=1;

% Parameters for reinforcement function:

eta_e=1;
eta_c=20;

% Parameters for the radial basis function neural network

% Define parameters of the approximator

nG=11;   % The number of partitions on each edge of the grid
nR=nG^2;  % The number of receptive field units in the RBF

n=2; % The number of inputs 

tempe=(-pi/2):(pi)/(nG-1):pi/2;  % Defines a uniformly spaced vector roughly on the input domain
			             % that is used to form the uniform grid on the (e,c) space
tempc=(-0.01):(0.02)/(nG-1):0.01;

k=0; % Counter for centers below

% Place the centers on a grid

for i=1:length(tempe)
	for j=1:length(tempc)
	  k=k+1;
	  center(1,k)=tempe(i);
	  center(2,k)=tempc(j);
	end
end

% Define spreads of Gaussian functions

sigmae=0.7*((pi/nG)); % Use same value for all on e domain
sigmac=0.7*((0.02)/nG); 

% Next, pick the *initial* strengths for the receptive field units (these are what will
% later be adjusted by the reinforcement learning method): 

% First, you could use the approach from the neural networks chapter:

temp=(-((nG-1)/2)):1:((nG-1)/2);

for i=1:length(temp) % Across the e dimension
	for j=1:length(temp) % Across the c dimension
	thetamat(i,j)=-((1/10)*(200*(pi/180))*temp(i)+(1/10)*(200*(pi/180))*temp(j));
	% Saturate it between max and min possible inputs to the plant
	thetamat(i,j)=max([-80*(pi/180), min([80*(pi/180), thetamat(i,j)])]);
						% Note that there are only nR "stregths" to adjust - here we choose them
	                    % according to this mathematical formula to get an appropriately shaped surface
	end
end

% And, put them in a vector

k=0; % Counter for centers below

for i=1:length(temp)
	for j=1:length(temp)
	  k=k+1;
	  theta(k,1)=thetamat(i,j);
	end
end

% Another choice is just to use all zero strengths - to test how good it is at synthesizing the
% initial controller.

thetaold=0*theta;

% phi for the RBF NN is initialized below

% Compute vectors with points over the whole range of 
% the neural controller inputs - for use below

e_input=(-pi/2):(pi)/50:(pi/2); 
c_input=(-0.01):(0.02)/50:(0.01); 

% Convert from radians to degrees:

e_inputd=e_input*(180/pi);
c_inputd=c_input*(180/pi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate the RBF regulating the ship heading 
	
% Next, we initialize the simulation:

t=0; 		% Reset time to zero
index=1;	% This is time's index (not time, its index).  
tstop=20000;	% Stopping time for the simulation (in seconds) - normally 20000
step=1;     % Integration step size
T=10;		% The controller is implemented in discrete time and
			% this is the sampling time for the controller.
			% Note that the integration step size and the sampling
			% time are not the same.  In this way we seek to simulate
			% the continuous time system via the Runge-Kutta method and
			% the discrete time controller as if it were
			% implemented by a digital computer.  Hence, we sample
			% the plant output every T seconds and at that time
			% output a new value of the controller output.
counter=10;	% This counter will be used to count the number of integration
			% steps that have been taken in the current sampling interval.
			% Set it to 10 to begin so that it will compute a controller
			% output at the first step.
			% For our example, when 10 integration steps have been
			% taken we will then we will sample the ship heading
			% and the reference heading and compute a new output
			% for the controller.  
eold=0;     % Initialize the past value of the error (for use
            % in computing the change of the error, c).  Notice
            % that this is somewhat of an arbitrary choice since 
            % there is no last time step.  The same problem is
            % encountered in implementation.  
cold=0;     % Need this to initialize phiold below

psi_r_old=0; % Initialize the reference trajectory
yeold=0; 	 % Intial condition used to calculate yc
ymold=0; 	 % Initial condition for the first order reference model

x=[0;0;0];	% First, set the state to be a vector            
x(1)=0;		% Set the initial heading to be zero
x(2)=0;		% Set the initial heading rate to be zero.  
			% We would also like to set x(3) initially but this
			% must be done after we have computed the output
			% of the controller.  In this case, by
			% choosing the reference trajectory to be 
			% zero at the beginning and the other initial conditions
			% as they are, and the controller as designed,
			% we will know that the output of the controller
			% will start out at zero so we could have set 
			% x(3)=0 here.  To keep things more general, however, 
			% we set the intial condition immediately after 
			% we compute the first controller output in the 
			% loop below.

% Need to initialize phi

for i=1:nR
	phiold(i,1)=exp(-(((eold-center(1,i))^2)/sigmae^2)-(((cold-center(2,i))^2)/sigmac^2));
end


% Next, we start the simulation of the system.  This is the main 
% loop for the simulation of the control system.

psi_r=0*ones(1,tstop+1);
psi=0*ones(1,tstop+1);
e=0*ones(1,tstop+1);
c=0*ones(1,tstop+1);
s=0*ones(1,tstop+1);
w=0*ones(1,tstop+1);
delta=0*ones(1,tstop+1);
ym=0*ones(1,tstop+1);
J_R=0*ones(1,tstop+1);
ye=0*ones(1,tstop+1);
yc=0*ones(1,tstop+1);

while t <= tstop

% First, we define the reference input psi_r  (desired heading).

if t>=0, psi_r(index)=0; end			    % Request heading of 0 deg
if t>=100, psi_r(index)=45*(pi/180); end     % Request heading of 45 deg
if t>=1500, psi_r(index)=0; end    			% Request heading of 0 deg
if t>=3000, psi_r(index)=45*(pi/180); end    % Request heading of -45 deg
if t>=4500, psi_r(index)=0; end    			% Request heading of 0 deg
if t>=6000, psi_r(index)=45*(pi/180); end     % Request heading of 45 deg
if t>=7500, psi_r(index)=0; end    			% Request heading of 0 deg
if t>=9000, psi_r(index)=45*(pi/180); end     % Request heading of 45 deg
if t>=10500, psi_r(index)=0; end    			% Request heading of 0 deg
if t>=12000, psi_r(index)=45*(pi/180); end    % Request heading of -45 deg
if t>=13500, psi_r(index)=0; end    			% Request heading of 0 deg
if t>=15000, psi_r(index)=45*(pi/180); end     % Request heading of 45 deg
if t>=16500, psi_r(index)=0; end    			% Request heading of 0 deg
if t>=18000, psi_r(index)=45*(pi/180); end     % Request heading of 45 deg
if t>=19500, psi_r(index)=0; end    			% Request heading of 0 deg

% Next, suppose that there is sensor noise for the heading sensor with that is
% additive, with a uniform distribution on [- 0.01,+0.01] deg.
%s(index)=0.01*(pi/180)*(2*rand-1);
s(index)=0;					  % This allows us to remove the noise.

psi(index)=x(1)+s(index);     % Heading of the ship (possibly with sensor noise).

if counter == 10,  % When the counter reaches 10 then execute the 
				   % controller

counter=0; 			% First, reset the counter

% Reference model calculations:
% The reference model is part of the controller and to simulate it
% we take the discrete equivalent of the
% reference model to compute psi_m from psi_r (if you use
% a continuous-time reference model you will have to augment 
% the state of the closed-loop system with the state(s) of the 
% reference model and hence update the state in the Runge-Kutta 
% equations).
%
% For the reference model we use a first order transfer function 
% k_r/(s+a_r) but we use the bilinear transformation where we 
% replace s by (2/step)(z-1)/(z+1), then find the z-domain 
% representation of the reference model, then convert this 
% to a difference equation:

ym(index)=(1/(2+a_r*T))*((2-a_r*T)*ymold+...
                                    k_r*T*(psi_r(index)+psi_r_old));

ymold=ym(index);  
psi_r_old=psi_r(index);

	% This saves the past value of the ym and psi_r so that we can use it
	% the next time around the loop
	

% Radial basis function neural network controller calculations:

e(index)=psi_r(index)-psi(index); % Computes error (first layer of perceptron)
c(index)=(e(index)-eold)/T; % Sets the value of c

eold=e(index);   % Save the past value of e for use in the above
				 % computation the next time around the loop


% Next, perform calculations for reinforcement signal

ye(index)=ym(index)-psi(index);		    % Calculates ye
yc(index)=(ye(index)-yeold)/T;			% Calculates yc
yeold=ye(index);					    % Saves the value of ye for use the 
							            % next time

% Compute the reinforcement signal:

J_R(index)=eta*(-eta_e*ye(index)-eta_c*yc(index));

% When reinforcement signal is very small, simply make it zero (in
% this way it will not over-react to small deviations in adjusting
% the controller - it will only make adjustments when they are really needed)

if abs(J_R(index))<0.005
	J_R(index)=0;
end

% Compute the adjustments to the strengths

for i=1:nR
	theta(i,1)=thetaold(i,1)+J_R(index)*phiold(i,1);
end

% Next, compute the phi vector for the next time around the loop

for i=1:nR
	phi(i,1)=exp(-(((e(index)-center(1,i))^2)/sigmae^2)-(((c(index)-center(2,i))^2)/sigmac^2));
end

thetaold=theta(:,1); % Save this for next time around the loop
phiold=phi(:,1); % Save this for next time so that in the above formula the indices
                 % for thetaold and phiold are the same

% Compute the RBF output

delta(index)=theta(:,1)'*phi(:,1); % Performs summing and scaling of receptive field units


else % This goes with the "if" statement to check if the counter=10
     % so the next lines up to the next "end" statement are executed
     % whenever counter is not equal to 10

% Now, even though we do not compute the neural controller at each
% time instant, we do want to save the data at its inputs and output at
% each time instant for the sake of plotting it.  Hence, we need to 
% compute these here (note that we simply hold the values constant):

e(index)=e(index-1);	
c(index)=c(index-1); 
delta(index)=delta(index-1);
ye(index)=ye(index-1);
yc(index)=yc(index-1);
J_R(index)=J_R(index-1);
ym(index)=ym(index-1);

end % This is the end statement for the "if counter=10" statement

% Next, the Runge-Kutta equations are used to find the next state. 
% Clearly, it would be better to use a Matlab "function" for
% F (but here we do not, so we can have only one program).
  
	time(index)=t;

% First, we define a wind disturbance against the body of the ship
% that has the effect of pressing water against the rudder

%w(index)=0.5*(pi/180)*sin(2*pi*0.001*t);  % This is an additive sine disturbance to 
										% the rudder input.  It is of amplitude of
										% 0.5 deg. and its period is 1000sec.
%delta(index)=delta(index)+w(index);


% Next, implement the nonlinearity where the rudder angle is saturated
% at +-80 degrees

if delta(index) >= 80*(pi/180), delta(index)=80*(pi/180); end
if delta(index) <= -80*(pi/180), delta(index)=-80*(pi/180); end

% The next line is used in place of the line following it to
% change the speed of the ship
if t>=1000000,
%if t>=9000,      % This switches the ship speed (unrealistically fast)
u=3; % A lower speed

else

u=5;

end

% Next, we change the parameters of the ship to tanker to reflect
% changing loading conditions (note that we simulate as if
% the ship is loaded while moving, but we only change the parameters
% while the heading is zero so that it is then similar to re-running
% the simulation, i.e., starting the tanker operation at different 
% times after loading/unloading has occurred).

% The next line is used in place of the line following it to keep
% "ballast" conditions throughout the simulation
%if t>=1000000,
if t>=9000,      % This switches the parameters in the middle of the simulation
K_0=0.83;  		% These are the parameters under "full" conditions
tau_10=-2.88;
tau_20=0.38;
tau_30=1.07;

else

K_0=5.88;		% These are the parameters under "ballast" conditions
tau_10=-16.91;
tau_20=0.45;
tau_30=1.43;

end

% The following parameters are used in the definition of the tanker model:

K=K_0*(u/ell);
tau_1=tau_10*(ell/u);
tau_2=tau_20*(ell/u);
tau_3=tau_30*(ell/u);


% Next, comes the plant:
% Now, for the first step, we set the initial condition for the
% third state x(3).

if t==0, x(3)=-(K*tau_3/(tau_1*tau_2))*delta(index); end

% Next, we use the formulas to implement the Runge-Kutta method
% (note that here only an approximation to the method is implemented where
% we do not compute the function at multiple points in the integration step size).

F=[ x(2) ;
    x(3)+ (K*tau_3/(tau_1*tau_2))*delta(index) ;
    -((1/tau_1)+(1/tau_2))*(x(3)+ (K*tau_3/(tau_1*tau_2))*delta(index))-...
        (1/(tau_1*tau_2))*(abar*x(2)^3 + bbar*x(2)) + (K/(tau_1*tau_2))*delta(index) ];
        
	k1=step*F;
	xnew=x+k1/2;

F=[ xnew(2) ;
    xnew(3)+ (K*tau_3/(tau_1*tau_2))*delta(index) ;
    -((1/tau_1)+(1/tau_2))*(xnew(3)+ (K*tau_3/(tau_1*tau_2))*delta(index))-...
        (1/(tau_1*tau_2))*(abar*xnew(2)^3 + bbar*xnew(2)) + (K/(tau_1*tau_2))*delta(index) ];
   
	k2=step*F;
	xnew=x+k2/2;

F=[ xnew(2) ;
    xnew(3)+ (K*tau_3/(tau_1*tau_2))*delta(index) ;
    -((1/tau_1)+(1/tau_2))*(xnew(3)+ (K*tau_3/(tau_1*tau_2))*delta(index))-...
        (1/(tau_1*tau_2))*(abar*xnew(2)^3 + bbar*xnew(2)) + (K/(tau_1*tau_2))*delta(index) ];
   
	k3=step*F;
	xnew=x+k3;

F=[ xnew(2) ;
    xnew(3)+ (K*tau_3/(tau_1*tau_2))*delta(index) ;
    -((1/tau_1)+(1/tau_2))*(xnew(3)+ (K*tau_3/(tau_1*tau_2))*delta(index))-...
        (1/(tau_1*tau_2))*(abar*xnew(2)^3 + bbar*xnew(2)) + (K/(tau_1*tau_2))*delta(index) ];
   
	k4=step*F;
	x=x+(1/6)*(k1+2*k2+2*k3+k4); % Calculated next state


t=t+step;  			% Increments time
index=index+1;	 	% Increments the indexing term so that 
					% index=1 corresponds to time t=0.
counter=counter+1;	% Indicates that we computed one more integration step

% Plot the mapping in the middle of the simulation:

if t==8999
	
for jj=1:length(e_input) 
	for ii=1:length(c_input)
		
	for i=1:nR
		phit(i,1)=exp(-(((e_input(jj)-center(1,i))^2)/sigmae^2)-(((c_input(ii)-center(2,i))^2)/sigmac^2));
	end

	delta_output(ii,jj)=theta'*phit(:,1); % Performs summing and scaling of receptive field units

	end
end

% Plot the controller map

delta_outputd1=delta_output*(180/pi);

figure(1)
clf
surf(e_inputd,c_inputd,delta_outputd1);
view(145,30);
colormap(white);
xlabel('Heading error (e), deg.');
ylabel('Change in heading error (c), deg.');
zlabel('Controller output (\delta), deg.');
title('Radial basis function neural network controller mapping between inputs and output');
rotate3d
zoom

end


end % This end statement goes with the first "while" statement 
    % in the program so when this is complete the simulation is done.

%
% Next, we provide plots of the input and output of the ship 
% along with the reference heading that we want to track.
%

% Next, we provide plots of data from the simulation

% First, we convert from rad. to degrees
psi_r=psi_r*(180/pi);
psi=psi*(180/pi);
delta=delta*(180/pi);
e=e*(180/pi);
c=c*(180/pi);
ym=ym*(180/pi);
ye=ye*(180/pi);

% Next, we provide plots 

figure(2)
clf
subplot(311)
plot(time,psi,'k-',time,ym,'k--',time,psi_r,'k-.')
zoom
grid on
title('Ship heading (solid) and desired ship heading (dashed), deg.')
subplot(312)
plot(time,delta,'k-')
zoom
grid on
title('Rudder angle, output of neural controller (input to the ship), deg.')
subplot(313)
plot(time,J_R,'k-')
zoom
grid on
xlabel('Time (sec)')
title('Reinforcement signal (nonzero values indicate adaptation)')

figure(3)
clf
subplot(211)
plot(time,e,'k-')
zoom
grid on
title('Ship heading error between ship heading and desired heading, deg.')
subplot(212)
plot(time,c,'k-')
zoom
grid on
xlabel('Time (sec)')
title('Change in ship heading error, deg./sec')

figure(4)
clf
subplot(211)
plot(time,ye,'k-')
zoom
grid on
title('Ship heading error between ship heading and reference model heading, deg.')
subplot(212)
plot(time,yc,'k-')
zoom
grid on
xlabel('Time (sec)')
title('Change in heading error between output and reference model, deg./sec')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Next, provide a plot of the *final* RBF neural controller surface:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%flag1=1;
%if flag1==0
	

for jj=1:length(e_input) 
	for ii=1:length(c_input)
		
	for i=1:nR
		phit(i,1)=exp(-(((e_input(jj)-center(1,i))^2)/sigmae^2)-(((c_input(ii)-center(2,i))^2)/sigmac^2));
	end

	delta_output(ii,jj)=theta'*phit(:,1); % Performs summing and scaling of receptive field units

	end
end

% Plot the final controller map

delta_outputd=delta_output*(180/pi);

figure(5)
clf
surf(e_inputd,c_inputd,delta_outputd);
view(145,30);
colormap(white);
xlabel('Heading error (e), deg.');
ylabel('Change in heading error (c), deg.');
zlabel('Controller output (\delta), deg.');
title('Radial basis function neural network controller mapping between inputs and output');
rotate3d
zoom

%end

% Plot the difference between the two plots at the middle and end of the simulation


figure(6)
clf
surf(e_inputd,c_inputd,delta_outputd-delta_outputd1);
view(145,30);
colormap(white);
xlabel('Heading error (e), deg.');
ylabel('Change in heading error (c), deg.');
zlabel('Controller output (\delta), deg.');
title('Difference between mapping shapes');
rotate3d

% To view it better, create a contour plot

figure(7) 
% Neg indicates decreased size of map by end, pos indicates that increased
% Jet option below will give red for increases, blue for decreases
% Gray will give light for positive, dark for negative (use this to be consistent
% with GA plots)
clf
contour(e_inputd,c_inputd,delta_outputd-delta_outputd1,20)
%colormap(jet)
colormap(gray);
xlabel('Heading error (e), deg.');
ylabel('Change in heading error (c), deg.');
title('Difference between mapping shapes, contour map');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of program %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
