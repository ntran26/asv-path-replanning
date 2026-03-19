%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For training Takagi-Sugeno fuzzy systems using recursive least squares
%
% By: Kevin Passino
% Version: 3/12/99
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
figure(1)
clf             % Clear the figure that will be used

% First set the number of steps for the simulation

Nrls=301; % One more than the number of iterations you want
time=0:Nrls-1; % For use in plotting (notice that time starts at zero in the plots but
                         % and the index 1 corresponds to a time of zero
lambda=1; % Use standard (nonweighted) least squares

% Define the parameters for the phi function of the fuzzy system

R=20;
c(1,:)=-5.4:0.6:6;
sigma(1,:)=0.1*ones(1,R);


% Next, pick the initial conditions (the index 1 corresponds to k=0)

theta(:,1)=0*ones(2*R,1);

% As another option you can run the batch least squares program and generate the
% best guess at theta then use it to initialize the RLS in the following manner:
%load variables theta40
%theta(:,1)=theta40;  % If you use this then you will get good approximation accuracy
                               % with very little adjustment from RLS
% Add some noise to the initialization so that it is not as good
%for i=1:2*R
%	theta(i,1)=theta(i,1)+0.25*theta(i,1);
%end

alpha=100;

P(:,:,1)=alpha*eye(2*R);

% Note that we do not need to initialize K

% Next, define the error that results from the initial choice of parameters:

x(1)=6*(-1+2*rand);  % Input data uniformly distributed on (-6,6)

z(1)=0.15*(rand-0.5)*2;  % Define the auxiliary variable
G(1)=exp(-50*(x(1)-1)^2)-0.5*exp(-100*(x(1)-1.2)^2)+atan(2*x(1))+2.15+...
0.2*exp(-10*(x(1)+1)^2)-0.25*exp(-20*(x(1)+1.5)^2)+0.1*exp(-10*(x(1)+2)^2)-0.2*exp(-10*(x(1)+3)^2);
if x(1) >= 0
	G(1)=G(1)+0.1*(x(1)-2)^2-0.4;
end
y(1)=G(1)+z(1); % Adds in the influence of the auxiliary variable

% Next, compute the estimator output

for j=1:R
	mu(j,1)=exp(-0.5*((x(1)-c(1,j))/sigma(1,j))^2);
end
	
denominator(1)=sum(mu(:,1)); 
	
for j=1:R
	xi(j,1)=mu(j,1)/denominator(1);
end

phi(:,1)=[xi(:,1)', x(1)*xi(:,1)']';

% The estimator output is:

yhat(1)=theta(:,1)'*phi(:,1); 

epsilon(1,1)=y(1)-yhat(1); % Define the estimation error 


% Next, start the estimator
	
for k=2:Nrls
	
	% First, generate data from the "unknown" function

	x(k)=6*(-1+2*rand);  % Input data uniformly distributed on (-6,6)

	z(k)=0.15*(rand-0.5)*2;  % Define the auxiliary variable
	G(k)=exp(-50*(x(k)-1)^2)-0.5*exp(-100*(x(k)-1.2)^2)+atan(2*x(k))+2.15+...
	0.2*exp(-10*(x(k)+1)^2)-0.25*exp(-20*(x(k)+1.5)^2)+0.1*exp(-10*(x(k)+2)^2)-0.2*exp(-10*(x(k)+3)^2);
	if x(k) >= 0
		G(k)=G(k)+0.1*(x(k)-2)^2-0.4;
	end
	y(k)=G(k)+z(k); % Adds in the influence of the auxiliary variable

	% Compute the phi vector

	for j=1:R
		mu(j,k)=exp(-0.5*((x(k)-c(1,j))/sigma(1,j))^2);
	end
	
	denominator(k)=sum(mu(:,k)); 
	
	for j=1:R
		xi(j,k)=mu(j,k)/denominator(k);
	end

	phi(:,k)=[xi(:,k)', x(k)*xi(:,k)']';
	

	% Next, compute the RLS update
		
	K(:,k)=P(:,:,k-1)*phi(:,k)/(lambda+phi(:,k)'*P(:,:,k-1)*phi(:,k));
	theta(:,k)=theta(:,k-1)+K(:,k)*(y(k)-phi(:,k)'*theta(:,k-1));
	P(:,:,k)=(1/lambda)*(eye(size(P(:,:,k-1)))-K(:,k)*phi(:,k)')*P(:,:,k-1);
	
	yhat(k)=theta(:,k)'*phi(:,k);  % The current guess of the estimator
	epsilon(k,1)=y(k)-yhat(k);  % Compute the estimation error  (for plotting if you want)
	
	
if k <=11  % For the first 10 iterations plot the approximator mapping
	
% Next, compute the estimator mapping and plot it on the data

xt=-6:0.05:6;

for i=1:length(xt),
	for j=1:R
		mut(j,i)=exp(-0.5*((xt(i)-c(1,j))/sigma(1,j))^2);
	end
	
	denominatort(i)=sum(mut(:,i)); 
	
	for j=1:R
		xit(j,i)=mut(j,i)/denominatort(i);
	end

	phit(:,i)=[xit(:,i)', xt(i)*xit(:,i)']';

	Fts(i)=theta(:,k)'*phit(:,i);
end

% Plot the estimator mapping after having k-1 pieces of training data

figure(1)
subplot(5,2,k-1)
plot(x,y,'ko',xt,Fts,'k')
xlabel('x')
T=num2str(k-1);
T=strcat('y, k=',T);
ylabel(T)
grid
axis([-6 6 0 4.8])
hold on

end  % Ends the test to see if 10 data pairs have been used

end

% Next, plot the output of the last approximator obtained and the unknown function

% First, compute the approximator 

for i=1:length(xt),
	for j=1:R
		mut(j,i)=exp(-0.5*((xt(i)-c(1,j))/sigma(1,j))^2);
	end
	
	denominatort(i)=sum(mut(:,i)); 
	
	for j=1:R
		xit(j,i)=mut(j,i)/denominatort(i);
	end

	phit(:,i)=[xit(:,i)', xt(i)*xit(:,i)']';

	Fts(i)=theta(:,k)'*phit(:,i);
end

% Next, plot

figure(2)
plot(x,y,'ko',xt,Fts,'k')
xlabel('x')
T=num2str(k-1);
T=strcat('y, k=',T);
ylabel(T)
title('Takagi-Sugeno fuzzy system trained with RLS')
grid
axis([-6 6 0 4.8])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
