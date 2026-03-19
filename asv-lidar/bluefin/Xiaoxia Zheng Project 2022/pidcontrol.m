function ui = pidcontrol(e,K)

Kp = K(1);
Ki = K(2);
Kd = K(3);

delc = Kp*e(1) + Ki*e(2) + Kd*e(3);

% Limits for delc:
delmax = 40*pi/180;
delmin = -delmax;

if delc > delmax
    delc = delmax;
elseif delc < delmin
    delc = delmin;
else
    delc = delc;
end

ui = [delc
      0];

end

