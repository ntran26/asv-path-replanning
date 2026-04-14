function xdot = model(x,Ud,del_g)

k=0.11;
T=7.5;
a=1;
Trud=11.9;


psi_d=x(1);
r=x(2);
del_c=x(3);
x_pos=x(4);
y_pos=x(5);


xdot=     [r
      -r/T+(k*del_c/T)
       (del_g-del_c)/(abs(del_g-del_c)*Trud+a)
       Ud*cos(psi_d)
       Ud*sin(psi_d)];
%end