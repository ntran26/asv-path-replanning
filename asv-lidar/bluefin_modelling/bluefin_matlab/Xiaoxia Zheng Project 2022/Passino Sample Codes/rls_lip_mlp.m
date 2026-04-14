%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For training two-layer perceptrons using recursive least squares
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
lambda=1; % Forgetting factor value

% Define the parameters for the phi function of the neural network
n1=25;
w(1,:)=ones(1,n1);
b=-6:0.5:6;

% Next, pick the initial conditions (the index 1 corresponds to k=0)

theta(:,1)=0*ones(n1+1,1);

% As another option you can run the batch least squares program and generate the
% best guess at theta then use it to initialize the RLS in the following manner:
%load variables theta26
%theta(:,1)=theta26;  % If you use this then you will get good approximation accuracy
                               % with very little adjustment from RLS
% Add some "noise" to the initialization so that it is not as good
%for i=1:n1+1
%	theta(i,1)=theta(i,1)+0.25*theta(i,1);
%end

alpha=100;

P(:,:,1)=alpha*eye(n1+1);

% Note that there is no need to initialize K

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

for j=1:n1
	phi(j,1)=inv(1+exp(-b(j)-w(1,j)*x(1)));
end

phi(n1+1,1)=1;

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
	
	for j=1:n1
		phi(j,k)=inv(1+exp(-b(j)-w(1,j)*x(k)));
	end

	phi(n1+1,k)=1;

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
	for j=1:n1
		phit(j,i)=inv(1+exp(-b(j)-w(1,j)*xt(i)));
	end

	phit(n1+1,i)=1;

	Fmlp25(i)=theta(:,k)'*phit(:,i);
end

% Plot the estimator mapping after having k-1 pieces of training data

figure(1)
subplot(5,2,k-1)
plot(x,y,'ko',xt,Fmlp25,'k')
xlabel('x')
T=num2str(k-1);
T=strcat('y, k=',T);
ylabel(T)
grid
axis([-6 6 0 4.8])
hold on

end

end

% Next, plot the output of the last approximator obtained and the unknown function

% First, compute the approximator 

for i=1:length(xt),
        
        for j=1:n1
		phit(j,i)=inv(1+exp(-b(j)-w(1,j)*xt(i)));
	end

	phit(n1+1,i)=1;

	Fmlp25(i)=theta(:,k)'*phit(:,i);
end

% Next, plot

figure(2)
plot(x,y,'ko',xt,Fmlp25,'k')
xlabel('x')
T=num2str(k-1);
T=strcat('y, k=',T);
ylabel(T)
title('Neural network trained with RLS')
grid
axis([-6 6 0 4.8])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
