import socket

ROBOT_ADDR = ('10.201.208.152', 5050)

# create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# configure it to listen on any IP address, port 5050
sock.bind(('0.0.0.0', 5050))

# send a msg to the robot to register as a listener
sock.sendto(b'START\n', ROBOT_ADDR)

# main program loop
while True:
    # wait until a new message arrives
    msg, addr = sock.recvfrom(4096)

    # Do something with the message
    #   e.g. parse data, update agent, etc.
    print(msg)

    # Send a command (if necessary) to the robot
    sock.sendto(b'SOME COMMAND', ROBOT_ADDR)