import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0',5000))
sock.sendto(b'HEY\n', ("10.201.208.152",5050))


while True:
    msg, addr = sock.recvfrom(4096)
    print(msg.decode())
