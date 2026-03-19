% test random number
 ss = 0.01;
 st = 100;
 t = 0:ss:st;
 r = 0.025 + 0.0025*randn(st/ss+1,1);
 plot(t,r);