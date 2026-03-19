function [del_cg] = pidheadingcontroller(K,e)

%kp=proportional gain
%ki=integral gain
%kd=derivative gain

Kp=K(1);
Ki=K(2);
Kd=K(3);


limit=25*pi/180; %limits in radian

%computing the PID of the error
pid_psi=e(1)*Kp+e(2)*Ki+e(3)*Kd;
%limit
if pid_psi>=limit
    pid_psi=limit;
elseif pid_psi<=-limit
    pid_psi=-limit;
else
    pid_psi=pid_psi;
end
del_cg=pid_psi;

    
%end

