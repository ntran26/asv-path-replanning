% Gaol function:
% Author: K. Passino, Version: 1/25/01
function Jg=goalfunction(x,xgoal,w2)

% An example gaol function:

	Jg=w2*(x-xgoal)'*(x-xgoal);
%	Jg=0.01*(x-xgoal)'*(x-xgoal)/(1-0.01*(x-xgoal)'*(x-xgoal));
	

