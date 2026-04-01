% Filename: Blue.m
% Mathematical model for Bluefin, the scaled vessel model of
% University of Tasmania - Australian Maritime College
% By learning shiojimaru.m by Hung Nguyen, the following fucntion can be
% created
function xdot = Blue(x,ui)
    rho = 1000;       
    rhoA = 0.1250;       
    g   = 9.81;
    del_max  = 40*pi/180;
    deld_max = 20*pi/180;
    rad = asin(1.0)/90.0;
% Ship
    m  = 64.55;
    mx = 3.662;
    my = 62.7366;
    Iz = 9.6038;
    Jz = 0.6309;
    IzJz=10.2347;
    L  = 1.725;
    d  =  0.193;
    Sw = 0.7614;
%  Hull parameters:
     ct = 0.004868;
     Xv = 398.715321509066e-003;     Yv = 2.47781051381700e-003;        Nv = 1.10546039494704e-003; 
     Xr = 105.779551566758e-006;     Yr = -94.5956792789195e-009;       Nr = -42.2032985948020e-009;
     Xvv = 0.0623;                   Yvv = 1.08140832998334e-003;       Nvv = 482.463882083071e-006;
     Xrr = 0.0027;                   Yrr = -22.7583008858493e-012;      Nrr = -10.1534803187344e-012;
     Xvr = 1.1415;                   Yvr = 262.214901533461e-009;       Nvr = 116.985615725573e-009;
% Propeller parameters:
    tp = 0.193;
    wp = 0.163065;
    rps= 5;
    Dp = 0.1;
% Rudder parameters:
    tr  = 0.256311;
    aH  = 0.443853;
    xR  =-1.05309;
    xH  =-0.733125;
    kx  = 0.6177;
    wr  = 0.22;
    AR  =  0.0091;
    falp=  2.69279;
    gamR= 0.4998;
    lR  =-0.77735;
    eps =(1.0-wr)/(1.0-wp);
    a  = 1;
% Masses and moments of inertia:
    m11 = (m+mx);
    m22 = (m+my);
    m33 = IzJz;    
% States and inputs:
    U = sqrt(0.6^2+0.4^2);
    u = x(1)/U;  
    v = x(2)/U; 
    r = x(3)*L/U; 
    psi = x(4);
    del = x(5); 
    phi = x(6);
    del_c = ui(1); 
    Xcc = ui(2); 
    Ycc = ui(3);
% Rudder saturations and dynamics:
if abs(del_c) >= del_max,
   del_c = sign(del_c)*del_max;
end

del_dot = del_c - del;

if abs(del_dot) >= deld_max,
   del_dot = sign(del_dot)*deld_max;
end

% Non-dimentional states:
U = sqrt(u*u+v*v);
vd = v/U;
rd = r*L/U;

% Hull:

 XH = m*v*r-1/2*rho*U^2*L*d*Sw/(L*d)*0.4631/(log(4*10^7)^2.6)*u^2+...
     Xvv*v^2+Xvr*v*r+Xrr*r^2;
 YH = 0.5*rho*L*d*U^2*(Yv*vd+Yvr*vd*abs(rd)+Yr*rd+...
     Yvv*abs(vd)*vd+Yrr*rd*abs(rd));
 % to add the model for Roll motion here!
 NH = 0.5*rho*L*d*U^2*(Nv*vd+Nvr*abs(vd)*rd+Nr*rd+Nvv*vd*abs(vd)...
     +Nrr*rd*abs(rd));
% Propeller:
% Kt = 0.25;
XP = (1-tp)*1000*5^2*0.1^4*0.25*0.072;
YP = 0;
NP = 0;
% Rudder:
uR = 0.856113*u*sqrt(1+6.3*(1-(0.856113*u)/(0.00717*1000))^1.5);
FN = -1/2*10^3*0.0091*2.6927*u^2*sin(del+(0.603463/uR)*(v-1.5525*r));
XR = -(1-0.449821)*FN*sin(del);
YR = -(1+0.443853)*FN*cos(del);
NR = -(0.646875+0.443853*0.7569)*FN*cos(del);
% Forces and moments:
X = XH + XP + XR ;
Y = YH + YP + YR ;
N = NH + NP + NR ;
% Return derivatives:ps
xdot = [X*2/(m11*(rho*L*L*U*U))
        Y*2/(m22*(rho*L*L*U*U))
        1/(Iz+Jz)*(N)
        r/L*U
        del_dot
        u*cos(psi)-v*cos(phi)*sin(psi)
        u*sin(psi)-v*cos(phi)*cos(phi)];   
% End of function

