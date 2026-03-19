% GUIInitialization.m

 clear
 global waypointx waypointy L1 index
 %m1  
 m1   = 0.0084;
 mx1  = 0.00031514;
 rho1 = 1000;
 Cd1  = 0.075/(log(10^6)-2)^2;
 A1   = 1.226;
 
 L1 = 2.47;        % length of ship
 S1 = L1;
 
 index = 2;
 
x2 = [0  0 30 30 ];
y2 = [0 40*S1  60*S1  120*S1];

waypointx = x2';   waypointy = y2';


Ts=1.127;
Ks=6.385;
Tu=5.892;
Ku=0.002;
%opitmal input calculation
A1=[-1/Ts 0 0;1 0 0;0 1852/3600 0 ];
B1=[Ks/Ts;0;0];
Q1=[1e+8 0 0;0 1e+6 0;0 0 1e+6];
R1=(180/pi)^2*10;
K1 = lqr(A1,B1,Q1,R1);

A2=[0 1;0 -1/Tu];
B2=[0;Ku/Tu];
Q2=[1e+6 0; 0 6e+7];
R2=1e-5;
K2= lqr(A2,B2,Q2,R2);


