clc
clear all
close all
%% 
%%Prior information%%%%%%%%%%%%%%%%%%
Lpp=46; %length of ship
Ro=2*Lpp; %circle of acceptance radius

wpt=2*[900 900 1800 900 2700 0 1800 -900 900 -900 0 0];
Ux=[6 6 6 6 6 6];

wpt_number=1;

Ud=Ux(wpt_number);   %surge velocity


ts=0;
ss=0.1;
tf=4000;


%% PID parameters
K=[8.5 0.0001 78];%autopilot heading controller gains

ei=0;ed0=0;
e=[0 ei ed0];
%% 

x=[0 0 0 0 0];
index=0;

%% guidance system loop
for i=0:ss:tf
index=index+1;
yd=wpt(wpt_number*2-1); %fixed earth frame y-cordinates
xd=wpt(wpt_number*2);  %fixed earth frame x-cordinates 
Ud=Ux(wpt_number);   %velocity
%% LOS algorithm
psi_d=atan2(yd-x(5),xd-x(4));

%% error computation
psi_e=psi_d-x(1);
%error limit for
if psi_e>=pi/2
    psi_e=psi_e-2*pi;
elseif psi_e<=-pi/2
    psi_e=psi_e+2*pi;
end
del_g=guid_pidheadingcontroller(K,e); % recalling the PID algorithm for guidance tracker

% PID computation for autopilot heading
e(1)=psi_e;%proportional term of the error
%integral of the error
k11=ss*psi_e;
k12=ss*(psi_e+k11);
ei=ei+0.5*(k11+k12);
e(2)=ei;
%derivative of the error
ed=(psi_e-ed0)/ss;
ed0=psi_e;
e(3)=ed;

%% calling the referencemodel for the guidance system
%returning the derivatives of the function


k1 = 0.1*model(x,Ud,del_g);
k2 = 0.1*model(x+k1,Ud,del_g);
x = x + 0.5*(k1+k2);

%boundary of desired heading angle
if x(1)>2*pi
    x(1)=x(1)-2*pi;
elseif x(1)<0
    x(1)=x(1)+2*pi;
else
    x(1)=x(1);
end
    

%% waypoint indexing
M=abs(xd)-abs(x(4));
N=abs(yd)-abs(x(5));
if abs(M)<Ro && abs(N)<Ro
    %disp('waypoint found');
wpt_number=wpt_number+1;
end

    if wpt_number> 6 
    break;
        
    end

 %% storing data
%%%%%%%%%%%%autopilot data

global data
data(index,1)= x(1); % yaw
data(index,2)= x(2); % yaw rate
data(index,3)= x(3); %rudder angle
data(index,4)= x(4); %xpos
data(index,5)= x(5); %ypos
data(index,6)=i; %time
data(index,7)=Ud*ones(size(i)); % velocity
data(index,8)=x(1)*180/pi; %yaw in degrees

%% computing learning rate for NN

del_cg=data(index,1); %heading

lratedata=lr(del_cg);
data(index,9)=lratedata;
end


plot(data(:,5),data(:,4),'r','linewidth',1.05);grid minor 
xlabel('y-positions[m]')
ylabel('x-positions[m]')
title('Reference trajectory')
