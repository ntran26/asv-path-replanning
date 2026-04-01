Bluef_zigzag_test
subplot(211);plot(data(:,1),data(:,6)*180/pi,data(:,1),data(:,5)*180/pi);grid
legend('rudder','yaw')
subplot(212);plot(data(:,1),data(:,4)*180/pi);grid;
legend('yawrate')
