% function for the ODE

function xdot = equation(x,u,t)

A = [0 1;-7/2 -3/2];
B = [0;5/2];

xdot = A*x + B*u;

