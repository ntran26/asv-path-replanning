function[lratedata]=lr(del_cg)
if del_cg<10*pi/180
    eta=0.000022;
end
if del_cg>10*pi/180
    eta=0.000029;
end
if del_cg>15*pi/180
    eta=0.000034;
end
if del_cg>25*pi/180
    eta=0.000039;
end
if del_cg>35*pi/180
    eta=0.000045;
end
if del_cg>45*pi/180
    eta=0.000056;
end
if del_cg>55*pi/180
    eta=0.000066;
end
if del_cg>65*pi/180
    eta=0.000079;
end
if del_cg>80*pi/180
    eta=0.0086;
end

if del_cg>83*pi/180
    eta=-0.001;
end

if del_cg>85*pi/180
    eta=-0.0005;
end
if del_cg>87*pi/180
    eta=-0.0006;
end
if del_cg>89*pi/180
    eta=-0.0003;
end

if del_cg>90*pi/180
    eta=-0.5;
end

%% right turning point
if del_cg>=91*pi/180
    eta=0.43;
end
if del_cg>=92*pi/180
    eta=0.00009;
end

if del_cg>=94*pi/180
    eta=0.00001;
end
if del_cg>=96*pi/180
    eta=-0.00015;
end

%% turning point
if del_cg>=136*pi/180
  eta=0.003;
end
if del_cg>=137*pi/180
  eta=0.004;
end

if del_cg>=138*pi/180
  eta=0.02;
end
if del_cg>=139*pi/180
  eta=0.029;
end
if del_cg>=140*pi/180
  eta=0.012;
end
if del_cg>=145*pi/180
  eta=0.00001;
end

if del_cg>=150*pi/180
  eta=0.000001;
end
if del_cg>=160*pi/180
  eta=-0.00015;
end
if del_cg>=170*pi/180
  eta=-0.00098;
end
if del_cg>=172*pi/180
  eta=-0.00025;
end
if del_cg>=174*pi/180
  eta=-0.00008;
end

if del_cg>=230*pi/180
  eta=0.06;
end

if del_cg>=235*pi/180
  eta=0.08;
end
if del_cg>=240*pi/180
  eta=0.02;
end
if del_cg>=245*pi/180
  eta=-0.05;
end
if del_cg>=245*pi/180
  eta=-0.03;
end
if del_cg>=250*pi/180
  eta=-0.06;
end
if del_cg>=254*pi/180
  eta=0.0008;
end
if del_cg>=265*pi/180
  eta=0.001;
end
if del_cg>=267*pi/180
  eta=0.0005;
end

if del_cg>=269*pi/180
  eta=0.00035;
end

if del_cg>=271*pi/180
  eta=0.099;
end
if del_cg>=272*pi/180
  eta=-0.009;
end
if del_cg>=273*pi/180
  eta=-0.00012;
end
if del_cg>=275*pi/180
  eta=-0.000001;
end
if del_cg>=285*pi/180
  eta=-0.00014;
end
lratedata=eta;