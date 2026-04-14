function [sys,x0,str,ts] = guidance(t,x,u,flag)

% GUI.m
%

switch flag
   
case 0
   [sys,x0,str,ts] = mdlInitializeSizes;
   
%case 2
%   [sys,x0,str,ts] = mdlUpdate(t,x,u,lambda,P,xi);

case 3
   sys = mdlOutputs(t,x,u);
 
case { 1, 2, 4, 9}
   sys = [];
   
otherwise
   error(['Unhandled flag = ',num2str(flag)]);
   
end

% End of function myfun

function [sys,x0,str,ts] = mdlInitializeSizes

sizes = simsizes;

sizes.NumContStates = 0;
sizes.NumDiscStates = 0;
sizes.NumOutputs = 1;
sizes.NumInputs = -1;
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;

sys = simsizes(sizes);
x0 = [];
str = [];
ts = [-1 0];

% end of mdlInitializeSizes

 function sys = mdlOutputs(t,x,u)

  global waypointx waypointy L1 index 
  
 xxpos = u(1);
 yypos = u(2);
 
 xx = waypointx(index);yy = waypointy(index);
 
 R0 = sqrt((xx-xxpos)^2 + (yy-yypos)^2);
 
 if R0 <=2.5*L1
    index = index+1;
    yy = waypointy(index);
    xx = waypointx(index);
    
 end
 
 Coref = atan2((yy-yypos),(xx-xxpos));

 sys = [Coref];    % Reference cource
    
 %     end