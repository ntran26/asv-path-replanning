import socket

ROBOT_ADDR = ('10.201.208.152', 5050)

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Configure to listen on any IP address, port 5050
sock.bind(('0.0.0.0', 5050))

# Send a message to the robot to register as a listener
sock.sendto(b'START\n', ROBOT_ADDR)

# Main loop
while True:
    # wait until new message arrives
    msg, addr = sock.recvfrom(4096)

    # process the message


    # send a command
    sock.sendto(b'COMMAND', ROBOT_ADDR)
