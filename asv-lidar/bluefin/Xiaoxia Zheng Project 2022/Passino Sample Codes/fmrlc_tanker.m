%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fuzzy Model Reference Learning Control (FMRLC) System for a Tanker Ship
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% By: Kevin Passino 
% Version: 4/26/01
%
% Notes: This program has evolved over time and uses programming 
% ideas of Andrew Kwong, Jeff Layne, and Brian Klinehoffer.
%
% This program simulates an FMRLC for a tanker
% ship.  It has a fuzzy controller with two inputs, the error
% in the ship heading (e) and the change in that error (c).  The output
% of the fuzzy controller is the rudder input (delta).  The FMRLC 
% adjusts the fuzzy controller to try to get tanker ship heading (psi) 
% to track the output of a "reference model" (psi_m) that has as an 
% input the reference input heading (psi_r). We simulate the tanker 
% as a continuous time system that is controlled by an FMRLC that 
% is implemented on a digital computer with a sampling interval of T=1 sec.  
%
% This program can be used to illustrate:
%  - How to code an FMRLC (for two inputs and one output, 
%    for the controller and "fuzzy inverse model").
%  - How to tune the input and output gains of an FMRLC.
%  - How changes in plant conditions ("ballast" and "full" and speed) 
%    can affect performance and how the FMRLC can adapt to improve
%    performance when there are such changes.
%  - How the effects of sensor noise (heading sensor noise) and plant 
%    disturbances (wind hitting the side of the ship) can be reduced
%    over the case of non-adaptive fuzzy control.
%  - The shape of the nonlinearity synthesized by the FMRLC
%    by plotting the input-output map of the fuzzy controller at the
%    end of the simulation (and providing the output membership function
%    centers).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear		% Clear all variables in memory

% Initialize ship parameters 

ell=350;			% Length of the ship (in meters)
abar=1;             % Parameters for nonlinearity
bbar=1;

% Define the reference model (we use a first order transfer function 
% k_r/(s+a_r)):

a_r=1/150;
k_r=1/150;

% Initialize parameters for the fuzzy controller

nume=11; 	% Number of input membership functions for the e 
			% universe of discourse (can change this but must also
			% change some variables below if you make such a change)
numc=11; 	% Number of input membership functions for the c 
			% universe of discourse (can change this but must also
			% change some variables below if you make such a change)

% Next, we define the scaling gains for tuning membership functions for 
% universes of discourse for e, change in e (what we call c) and 
% delta.  These are g1, g2, and g0, respectively
% These can be tuned to try to improve the performance. 

% First guess: 
g1=1/pi;,g2=100;,g0=8*pi/18; % Chosen since: 
		% g1: The heading error is at most 180 deg (pi rad) 
		% g2: Just a guess - that ship heading will change at most
		%     by 0.01 rad/sec (0.57 deg/sec)
		% g0: Since the rudder is constrained to move between +-80 deg
% "Good" tuned values:
g1=2/pi;,g2=250;,g0=8*pi/18; 

% Next, define some parameters for the membership functions

we=0.2*(1/g1);	  
	% we is half the width of the triangular input membership 
	% function bases (note that if you change g0, the base width
	% will correspondingly change so that we always end
	% up with uniformly distributed input membership functions)
	% Note that if you change nume you will need to adjust the 
	% "0.2" factor if you want membership functions that 
	% overlap in the same way.
wc=0.2*(1/g2);        
	% Similar to we but for the c universe of discourse
base=0.4*g0;      
	% Base width of output membership fuctions of the fuzzy 
	% controller

% Place centers of membership functions of the fuzzy controller:

% Centers of input membership functions for the e universe of
% discourse of fuzzy controller (a vector of centers)
ce=[-1 -0.8 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1]*(1/g1);

% Centers of input membership functions for the c universe of
% discourse of fuzzy controller (a vector of centers)
cc=[-1 -0.8 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1]*(1/g2);

% This next matrix specifies the rules of the fuzzy controller.  
% The entries are the centers of the output membership functions.  
% This choice represents just one guess on how to synthesize 
% the fuzzy controller.  Notice the regularity 
% of the pattern of rules (basiscally we are using a type of 
% saturated index adding).  Notice that it is scaled by g0, the 
% output scaling factor, since it is a normalized rule base.
% The rule base can be tuned to try to improve performance.

% The gain gf is a gain that can be set to one to initialize 
% the rule base with an initial guess at the fuzzy controller to be
% synthesized or to zero to initialize the rule base to all zeros
% (i.e., all centers at zero so the rules all say to put zero
% into the plant).

gf=1;

rules=[1    1     1     1    1     1    0.8   0.6   0.3    0.1     0;
  	   1    1     1     1    1   0.8    0.6   0.3   0.1    0     -0.1;
	   1    1     1     1   0.8  0.6    0.3   0.1   0     -0.1   -0.3;
	   1    1     1   0.8   0.6  0.3    0.1    0    -0.1   -0.3  -0.6;
	   1    1   0.8   0.6   0.3  0.1     0   -0.1   -0.3   -0.6  -0.8;
	   1  0.8   0.6   0.3   0.1   0    -0.1  -0.3   -0.6   -0.8   -1;
	 0.8  0.6   0.3   0.1    0   -0.1  -0.3  -0.6   -0.8   -1     -1;
	 0.6  0.3   0.1    0   -0.1  -0.3  -0.6  -0.8   -1     -1     -1;
	 0.3  0.1    0   -0.1  -0.3  -0.6  -0.8   -1    -1     -1     -1;
	 0.1   0   -0.1  -0.3  -0.6  -0.8   -1    -1    -1     -1     -1;
	   0 -0.1  -0.3  -0.6  -0.8  -1     -1    -1    -1     -1     -1]*gf*g0;

% Next, we define some parameters for the fuzzy inverse model

gye=2/pi;,gyc=10;    % Scaling gains for the error and change in error for 
					% the inverse model
					% These are tuned to improve the performance of the FMRLC
gp=0.4;	

% Scaling gain for the output of inverse model.  If you let gp=0 then
% you turn off the learning mechanism and hence the FMRLC reduces to 
% a direct fuzzy controller that is not tuned.  Hence, from this 
% program you can also see how to code a non-adaptive fuzzy controller.
% If gp is large then generally large updates will be made to the 
% fuzzy controller.  You should think of gp as an "adaptation gain."

numye=11; 	% Number of input membership functions for the ye 
			% universe of discourse
numyc=11; 	% Number of input membership functions for the yc 
			% universe of discourse

wye=0.2*(1/gye);	% Sets the width of the membership functions for 
					% ye from center to extremes
wyc=0.2*(1/gyc);	% Sets the width of the membership functions for 
					% yc from center to extremes
invbase=0.4*gp; % Sets the base of the output membership functions
				% for the inverse model

% Place centers of inverse model membership functions
% For error input for learning mechanism
cye=[-1 -0.8 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1]*(1/gye);

% For change in error input for learning mechanism
cyc=[-1 -0.8 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1]*(1/gyc);

% The next matrix contains the rule-base matrix for the fuzzy 
% inverse model.  Notice that for simplicity we choose it to have 
% the same structure as the rule base for the fuzzy controller.  
% While this will work for the control of the tanker 
% for many nonlinear systems a different structure 
% will be needed for the rule base.  Again, the entries are 
% the centers of the output membership functions, but now for 
% the fuzzy inverse model.

inverrules=[1   1   1    1    1    1    0.8  0.6   0.4   0.2   0;
    	    1   1   1    1    1   0.8   0.6  0.4   0.2    0   -0.2;
	        1   1   1    1   0.8  0.6   0.4  0.2    0    -0.2 -0.4;
	        1   1   1   0.8  0.6  0.4   0.2   0    -0.2  -0.4 -0.6;
	        1   1  0.8  0.6  0.4  0.2   0    -0.2  -0.4  -0.6 -0.8;
	        1  0.8 0.6  0.4  0.2   0   -0.2  -0.4  -0.6  -0.8  -1;
	      0.8  0.6 0.4  0.2   0   -0.2 -0.4  -0.6  -0.8   -1   -1;
	      0.6  0.4 0.2   0   -0.2 -0.4 -0.6  -0.8   -1    -1   -1;
	      0.4  0.2  0  -0.2  -0.4 -0.6 -0.8   -1    -1    -1   -1;
	      0.2   0  -0.2 -0.4 -0.6 -0.8  -1    -1    -1    -1   -1;
	       0  -0.2 -0.4 -0.6 -0.8  -1   -1    -1    -1    -1   -1]*gp;

% Next, we set up some parameters/variables for the 
% knowledge-base modifier

d=1;   

% This sets the number of steps the knowledge-base modifier looks
% back in time. For this program it must be an integer
% less than or equal to 10 (but this is easy to make larger)

% The next four vectors are used to store the information about 
% which rules were on 1 step in the past, 2 steps in the past, ...., 
% 10 steps in the past (so that picking 0<= d <= 10 can be used).
	   
meme_int=[0 0 0 0 0 0 0 0 0 0];  
	% sets up the vector to store up to 10 values of e_int
meme_count=[0 0 0 0 0 0 0 0 0 0];  
	% sets up the vector to store up to 10 values of e_count
memc_int=[0 0 0 0 0 0 0 0 0 0];  
	% sets up the vector to store up to 10 values of c_int
memc_count=[0 0 0 0 0 0 0 0 0 0];  
	% sets up the vector to store up to 10 values of c_count

% Now, you can proceed to do the simulation or simply view the nonlinear
% surface generated by the fuzzy controller that is now fully defined.

flag1=input('Do you want to simulate the \n FMRLC for the tanker?  \n (type 1 for yes and 0 for no) ');

if flag1==1, 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Next, we initialize the simulation:

t=0; 		% Reset time to zero
index=1;	% This is time's index (not time, its index).  
tstop=20000;	% Stopping time for the simulation (in seconds)
step=1;     % Integration step size
T=10;		% The controller is implemented in discrete time and
			% this is the sampling time for the controller.
			% Note that the integration step size and the sampling
			% time are not the same.  In this way we seek to simulate
			% the continuous time system via the Runge-Kutta method and
			% the discrete time fuzzy controller as if it were
			% implemented by a digital computer.  Hence, we sample
			% the plant output every T seconds and at that time
			% output a new value of the controller output.
counter=10;	% This counter will be used to count the number of integration
			% steps that have been taken in the current sampling interval.
			% Set it to 10 to begin so that it will compute a fuzzy controller
			% output at the first step.
			% For our example, when 10 integration steps have been
			% taken we will then we will sample the ship heading
			% and the reference heading and compute a new output
			% for the fuzzy controller.  
eold=0;     % Initialize the past value of the error (for use
            % in computing the change of the error, c).  Notice
            % that this is somewhat of an arbitrary choice since 
            % there is no last time step.  The same problem is
            % encountered in implementation.  
			
psi_r_old=0; % Initialize the reference trajectory
yeold=0; 	 % Intial condition used to calculate yc
ymold=0; 	 % Initial condition for the first order reference model

x=[0;0;0];	% First, set the state to be a vector            
x(1)=0;		% Set the initial heading to be zero
x(2)=0;		% Set the initial heading rate to be zero.  
			% We would also like to set x(3) initially but this
			% must be done after we have computed the output
			% of the fuzzy controller.  In this case, by
			% choosing the reference trajectory to be 
			% zero at the beginning and the other initial conditions
			% as they are, and the fuzzy controller as designed
			% we will know that the output of the fuzzy controller
			% will start out at zero so we could have set 
			% x(3)=0 here.  To keep things more general, however, 
			% we set the intial condition immediately after 
			% we compute the first controller output in the 
			% loop below.
			
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Next, we start the simulation of the system.  This is the main 
% loop for the simulation of fuzzy control system.

while t <= tstop

% First, we define the reference input psi_r (desired heading).

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
% additive, with a uniform distribution on +- 0.01 deg.
%s(index)=0.01*(pi/180)*(2*rand-1);
s(index)=0;					  % This allows us to remove the noise.

psi(index)=x(1)+s(index);     % Heading of the ship (possibly with sensor noise).

if counter == 10,  % When the counter reaches 10 then execute the 
				   % FMRLC

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
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fuzzy controller calculations:
% First, for the given fuzzy controller inputs we determine
% the extent at which the error membership functions 
% of the fuzzy controller are on (this is the fuzzification part).

c_count=0;,e_count=0;   % These are used to count the number of 
						% non-zero mf certainities of e and c
e(index)=psi_r(index)-psi(index);	
			% Calculates the error input for the fuzzy controller
			
c(index)=(e(index)-eold)/T; % Sets the value of c

eold=e(index);   % Save the past value of e for use in the above
				 % computation the next time around the loop
				
% The following if-then structure fills the vector mfe 
% with the certainty of each membership fucntion of e for the 
% current input e.  We use triangular membership functions.

	if e(index)<=ce(1)		% Takes care of saturation of the left-most 
					        % membership function
         mfe=[1 0 0 0 0 0 0 0 0 0 0]; % i.e., the only one on is the 
         							  % left-most one
	 e_count=e_count+1;,e_int=1; 	  %  One mf on, it is the 
	 								  % left-most one.
	elseif e(index)>=ce(nume)				  % Takes care of saturation 
									  % of the right-most mf
	 mfe=[0 0 0 0 0 0 0 0 0 0 1];
	 e_count=e_count+1;,e_int=nume; % One mf on, it is the 
	 								% right-most one
	else      % In this case the input is on the middle part of the 
			  % universe of discourse for e
			  % Next, we are going to cycle through the mfs to 
			  % find all that are on
	   for i=1:nume
		 if e(index)<=ce(i)
		  mfe(i)=max([0 1+(e(index)-ce(i))/we]);   
		  				% In this case the input is to the 
		  				% left of the center ce(i) and we compute
						% the value of the mf centered at ce(i)
						% for this input e
			if mfe(i)~=0	
				% If the certainty is not equal to zero then say 
				% that have one mf on by incrementing our count
			 e_count=e_count+1;
			 e_int=i;	% This term holds the index last entry 
			 			% with a non-zero term
			end
		 else 
		  mfe(i)=max([0,1+(ce(i)-e(index))/we]); 
		  						% In this case the input is to the
		  						% right of the center ce(i)
			if mfe(i)~=0
			 e_count=e_count+1;   
			 e_int=i;  % This term holds the index of the 
			 		   % last entry with a non-zero term
			end
		 end
		end
	end


% The following if-then structure fills the vector mfc with the 
% certainty of each membership fucntion of the c
% for its current value (to understand this part of the code see the above 
% similar code for computing mfe).  Clearly, it could be more efficient to
% make a subroutine that performs these computations for each of 
% the fuzzy system inputs.

	if c(index)<=cc(1)	% Takes care of saturation of left-most mf
         mfc=[1 0 0 0 0 0 0 0 0 0 0];
	 c_count=c_count+1;
	 c_int=1;   
	elseif c(index)>=cc(numc)	
				        % Takes care of saturation of the right-most mf
	 mfc=[0 0 0 0 0 0 0 0 0 0 1];
	 c_count=c_count+1; 
	 c_int=numc;
	else
		for i=1:numc
		 if c(index)<=cc(i)  
		  mfc(i)=max([0,1+(c(index)-cc(i))/wc]);
			if mfc(i)~=0
			 c_count=c_count+1;
			 c_int=i;  	    % This term holds last entry 
			 		     	% with a non-zero term
			end
		 else 
		  mfc(i)=max([0,1+(cc(i)-c(index))/wc]);
			if mfc(i)~=0
			 c_count=c_count+1;
			 c_int=i;	% This term holds last entry 
			 			% with a non-zero term
			end
		 end
		end
	end


% The next two loops calculate the crisp output using only the non-
% zero premise of error,e, and c.  This cuts down computation time 
% since we will only compute the contribution from the rules that 
% are on (i.e., a maximum of four rules for our case).  The minimum
% is used for the premise (and implication for the center-of-gravity
% defuzzification case).

num=0;
den=0;     
	for k=(e_int-e_count+1):e_int  
						% Scan over e indices of mfs that are on
		for l=(c_int-c_count+1):c_int	
						% Scan over c indices of mfs that are on
		  prem=min([mfe(k) mfc(l)]); 
		  				% Value of premise membership function
% This next calculation of num adds up the numerator for the 
% center of gravity defuzzification formula. rules(k,l) is the output center 
% for the rule.  base*(prem-(prem)^2/2 is the area of a symmetric
% triangle with base width "base" and that is chopped off at 
% a height of prem (since we use minimum to represent the 
% implication). Computation of den is similar but without 
% rules(k,l).
		  num=num+rules(k,l)*base*(prem-(prem)^2/2);
		  den=den+base*(prem-(prem)^2/2);		
% To do the same computations, but for center-average defuzzification,
% use the following lines of code rather than the two above (notice
% that in this case we did not use any information about the output
% membership function shapes, just their centers; also, note that
% the computations are slightly simpler for the center-average defuzzificaton):
%		num=num+rules(k,l)*prem;
%		den=den+prem;
		end
	end

   delta(index)=num/den;   
   			% Crisp output of fuzzy controller that is the input
   			% to the plant.
			
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Next, we perform computations for the fuzzy inverse model.

ye(index)=ym(index)-psi(index);		    % Calculates ye
yc(index)=(ye(index)-yeold)/T;			% Calculates yc
yeold=ye(index);					    % Saves the value of ye for use the 
							            % next time

ye_count=0;,yc_count=0;    	% Counts the non-zero certainties 
							% of ye and yc similar to how we did 
							% for the fuzzy controller

% The following if-then structure fills the vector mfye with the 
% certainty of each membership fucntion of ye (similar to the 
% fuzzy controller).  Notice that we use the same number of 
% input membership functions as in the fuzzy controller - 
% just for convenience - it does not have to be this way

	if ye(index)<=cye(1)			% Takes care of saturation
         mfye=[1 0 0 0 0 0 0 0 0 0 0];
	 ye_count=ye_count+1;,ye_int=1;
	elseif ye(index)>=cye(numye)	% Takes care of saturation
	 mfye=[0 0 0 0 0 0 0 0 0 0 1];
	 ye_count=ye_count+1;,ye_int=numye;
	else
		for i=1:numye
		 if ye(index)<=cye(i)
		  mfye(i)=max([0 1+(ye(index)-cye(i))/wye]);
			if mfye(i)~=0
			 ye_count=ye_count+1;
			 ye_int=i;	% This term holds last entry with 
			 			% a non-zero term
			end
		 else 
		  mfye(i)=max([0,1+(cye(i)-ye(index))/wye]);
			if mfye(i)~=0
			 ye_count=ye_count+1;
			 ye_int=i;	% This term holds last entry with 
			 			% a non-zero term
			end
		 end
		end
	end

% The following if-then structure fills the vector mfyc with the 
% certainty of each membership fucntion of yc

	if yc(index)<=cyc(1)
         mfyc=[1 0 0 0 0 0 0 0 0 0 0];
	 yc_count=yc_count+1;,yc_int=1;
	elseif yc(index)>=cyc(numyc)
	 mfyc=[0 0 0 0 0 0 0 0 0 0 1];
	 yc_count=yc_count+1;,yc_int=numyc;
	else
		for i=1:numyc
		 if yc(index)<=cyc(i)
		  mfyc(i)=max([0 1+(yc(index)-cyc(i))/wyc]);
			if mfyc(i)~=0
			 yc_count=yc_count+1;
			 yc_int=i;
			end
		 else 
		  mfyc(i)=max([0,1+(cyc(i)-yc(index))/wyc]);
			if mfyc(i)~=0
			 yc_count=yc_count+1;
			 yc_int=i;
			end
		 end
		end
	end

% These for loops calculate the crisp output of the inverse model 
% using only the non-zero premise of error,ye, and change in 
% error,yc.  This cuts down computation time (similar to the 
% fuzzy controller). Minimum is used for the premise and the 
% implication.  To understand this part of the code see the 
% similar code for the fuzzy controller (of course you could use
% center-average defuzzification).

invnum=0;
invden=0;     
	for k=(ye_int-ye_count+1):ye_int
		for l=(yc_int-yc_count+1):yc_int			
		  prem=min([mfye(k) mfyc(l)]);
		  invnum=invnum+inverrules(k,l)*invbase*(prem-(prem)^2/2);
		  invden=invden+invbase*(prem-(prem)^2/2);
		end
	end
	
% Next we compute the output of the fuzzy inverse model.

if gp==0, 
	p(index)=0;				% Allow us to turn off the learning
else
  p(index)=invnum/invden;   % Crisp output of inverse model
  if abs(p(index))<0.01*gp, % Robustification term which is used
      p(index)=0;			% to stop the adaptation when the fuzzy
  end						% controller is close to where it should
							% be.
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Next, we modify the centers of ouput membership functions 
% associated with the rules that were on d steps ago
% The indexing sheme is similar to the one used in the 
% computation of the outputs of the fuzzy controller and 
% fuzzy inverse model. Notice that in the early steps these
% two loops would not be executed since we initialized
% meme_int, meme_count, memc_int, and memc_count with all zeros.
% This implies that there are no updates until d executions of 
% the controller (e.g., for our case it executes the controller 
% at the first step and then on the next controller execution which
% occurs at 10 sec. the rules will be modified).

	for k=(meme_int(d)-meme_count(d)+1):meme_int(d)
		for l=(memc_int(d)-memc_count(d)+1):memc_int(d)			
		  rules(k,l)=rules(k,l)+p(index);
		end
	end

% Note: Here, do not save the entire rule base for the past d steps
% so this implies that, e.g., for a large d value, between d steps ago 
% and the present time the rules could be modified and so the way this 
% is coded it would modify the one that was modified in this case, less
% than d steps ago.  To avoid this you would have to save all the rules 
% over the past d steps and when you modify you may be undoing what was
% done on a previous step - and it is not clear that would be good either.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Store values of pointers/counters for rules that were on at this step

% Next we will save the number of mfs that are on and the pointer 
% e_int as to which rules were on.  This vector of length 
% 10 saves the last 10 values of e_count and e_int as time 
% progresses (hence, it is a moving window).
% These will be used by the FMRLC knowledge-base modifier.

meme_count=[e_count meme_count(1:9)];
meme_int=[e_int meme_int(1:9)];

% Next we will save the number of mfs that are on and the pointer 
% c_int as to which rules were on.  This vector of length 10 
% saves the last 10 values of c_count and e_int as time progresses 
% (hence, it is a moving window). These will be used by the FMRLC 
% knowledge-base modifier.

memc_count=[c_count memc_count(1:9)];
memc_int=[c_int memc_int(1:9)];

else % This goes with the "if" statement to check if the counter=10
     % so the next lines up to the next "end" statement are executed
     % whenever counter is not equal to 10

% Now, even though we do not compute the FMRLC at each
% time instant, we do want to save the data at its inputs and output at
% each time instant for the sake of plotting.  Hence, we need to 
% compute these here (note that we simply hold the values):

e(index)=e(index-1);	
c(index)=c(index-1); 
delta(index)=delta(index-1);
ye(index)=ye(index-1);
yc(index)=yc(index-1);
p(index)=p(index-1);
ym(index)=ym(index-1);

end % This is the end statement for the "if counter=10" statement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Next, the Runge-Kutta equations are used to find the next state. 
% Clearly, it would be better to use a Matlab "function" for
% F (but here we do not, so we can have only one program).
% (This does not implement the exact equations in the book but
% if you think about the problem a bit you will see that these
% really should be accurate enough. Here, we are not calculating
% intermediate values of the controller output, only
% one per integration step; we do this simply to save some 
% computations.  Similarly, for the reference input.)
  
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
if t>=1000000,
%if t>=9000,      % This switches the parameters in the middle of the simulation
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

% Now, for the first step, we set the initial condition for the
% third state x(3).

if t==0, x(3)=-(K*tau_3/(tau_1*tau_2))*delta(index); end

% Next, we use the formulas to implement the Runge-Kutta method

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

end % This end statement goes with the first "while" statement 
    % in the program so when this is complete the simulation is done.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Next, we provide plots of data from the simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% First, we convert from rad. to degrees
psi_r=psi_r*(180/pi);
psi=psi*(180/pi);
delta=delta*(180/pi);
e=e*(180/pi);
c=c*(180/pi);
ym=ym*(180/pi);
ye=ye*(180/pi);

% Next, we provide plots 

figure(1)
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
title('Rudder angle, output of fuzzy controller (input to the ship), deg.')
subplot(313)
plot(time,p,'k-')
zoom
grid on
xlabel('Time (sec)')
title('Fuzzy inverse model output (nonzero values indicate adaptation)')

figure(2)
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

figure(3)
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

end % This ends the if statement (on flag1) on whether you want to do a simulation
    % or just see the control surface
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Next, provide a plot of the fuzzy controller surface:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Request input from the user to see if they want to see the tuned
% controller mapping:

flag2=input('Do you want to see the nonlinear \n mapping implemented by the tuned fuzzy \n controller? \n (type 1 for yes and 0 for no) ');

if flag2==1,
	
% First, compute vectors with points over the whole range of 
% the fuzzy controller inputs plus 20% over the end of the range
% and put 100 points in each vector

e_input=(-(1/g1)-0.2*(1/g1)):(1/100)*(((1/g1)+0.2*(1/g1))-(-(1/g1)-...
     0.2*(1/g1))):((1/g1)+0.2*(1/g1)); 
ce_input=(-(1/g2)-0.2*(1/g2)):(1/100)*(((1/g2)+0.2*(1/g2))-(-(1/g2)-...
     0.2*(1/g2))):((1/g2)+0.2*(1/g2));

% Next, compute the fuzzy controller output for all these inputs

for jj=1:length(e_input) 
	for ii=1:length(ce_input) 

c_count=0;,e_count=0;   % These are used to count the number of 
						% non-zero mf certainities of e and c

% The following if-then structure fills the vector mfe 
% with the certainty of each membership fucntion of e for the 
% current input e.  We use triangular membership functions.

	if e_input(jj)<=ce(1)		% Takes care of saturation of the left-most 
					            % membership function
         mfe=[1 0 0 0 0 0 0 0 0 0 0]; % i.e., the only one on is the 
         							  % left-most one
	 e_count=e_count+1;,e_int=1; 	  %  One mf on, it is the 
	 								  % left-most one.
	elseif e_input(jj)>=ce(nume)				  % Takes care of saturation 
									  % of the right-most mf
	 mfe=[0 0 0 0 0 0 0 0 0 0 1];
	 e_count=e_count+1;,e_int=nume; % One mf on, it is the 
	 								% right-most one
	else      % In this case the input is on the middle part of the 
			  % universe of discourse for e
			  % Next, we are going to cycle through the mfs to 
			  % find all that are on
	   for i=1:nume
		 if e_input(jj)<=ce(i)
		  mfe(i)=max([0 1+(e_input(jj)-ce(i))/we]);   
		  				% In this case the input is to the 
		  				% left of the center ce(i) and we compute
						% the value of the mf centered at ce(i)
						% for this input e
			if mfe(i)~=0	
				% If the certainty is not equal to zero then say 
				% that have one mf on by incrementing our count
			 e_count=e_count+1;
			 e_int=i;	% This term holds the index last entry 
			 			% with a non-zero term
			end
		 else 
		  mfe(i)=max([0,1+(ce(i)-e_input(jj))/we]); 
		  						% In this case the input is to the
		  						% right of the center ce(i)
			if mfe(i)~=0
			 e_count=e_count+1;   
			 e_int=i;  % This term holds the index of the 
			 		   % last entry with a non-zero term
			end
		 end
		end
	end

% The following if-then structure fills the vector mfc with the 
% certainty of each membership fucntion of the c
% for its current value.

	if ce_input(ii)<=cc(1)	% Takes care of saturation of left-most mf
         mfc=[1 0 0 0 0 0 0 0 0 0 0];
	 c_count=c_count+1;
	 c_int=1;   
	elseif ce_input(ii)>=cc(numc)	
				        % Takes care of saturation of the right-most mf
	 mfc=[0 0 0 0 0 0 0 0 0 0 1];
	 c_count=c_count+1; 
	 c_int=numc;
	else
		for i=1:numc
		 if ce_input(ii)<=cc(i)  
		  mfc(i)=max([0,1+(ce_input(ii)-cc(i))/wc]);
			if mfc(i)~=0
			 c_count=c_count+1;
			 c_int=i;  	    % This term holds last entry 
			 		     	% with a non-zero term
			end
		 else 
		  mfc(i)=max([0,1+(cc(i)-ce_input(ii))/wc]);
			if mfc(i)~=0
			 c_count=c_count+1;
			 c_int=i;	% This term holds last entry 
			 			% with a non-zero term
			end
		 end
		end
	end

% The next loops calculate the crisp output using only the non-
% zero premise of error,e, and c.  

num=0;
den=0;     
	for k=(e_int-e_count+1):e_int  
						% Scan over e indices of mfs that are on
		for l=(c_int-c_count+1):c_int	
						% Scan over c indices of mfs that are on
		  prem=min([mfe(k) mfc(l)]); 
		  				% Value of premise membership function
% This next calculation of num adds up the numerator for the 
% center of gravity defuzzification formula. 
		  num=num+rules(k,l)*base*(prem-(prem)^2/2);
		  den=den+base*(prem-(prem)^2/2);		
% To do the same computations, but for center-average defuzzification,
% use the following lines of code rather than the two above:
%		num=num+rules(k,l)*prem;
%		den=den+prem;
		end
	end

   delta_output(ii,jj)=num/den;   
   			% Crisp output of fuzzy controller that is the input
   			% to the plant.
			
	end
end

% Convert from radians to degrees:

delta_output=delta_output*(180/pi);
e_input=e_input*(180/pi);
ce_input=ce_input*(180/pi);

% Plot the data

figure(4)
clf
surf(e_input,ce_input,delta_output);
view(145,30);
colormap(white);
xlabel('Heading error (e), deg.');
ylabel('Change in heading error (c), deg.');
zlabel('Fuzzy controller output (\delta), deg.');
title('FMRLC-tuned fuzzy controller mapping between inputs and output');

% Next, print out to the screen the rule base (output centers)

rules

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of program %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
