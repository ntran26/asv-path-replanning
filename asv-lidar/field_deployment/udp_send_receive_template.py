import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MESSAGE = b"Hello, world!"

print("UDP target IP:", UDP_IP)
print("UDP target port:", UDP_PORT)
print("Message:", MESSAGE)

sock = socket.socket(socket.AF_INET,    # internet
                     socket.SOCK_DGRAM) # UDP

# Sending message
sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

# Receiving message
sock.bind((UDP_IP, UDP_PORT))

while True:
   data, addr = sock.recvfrom(1024)     # buffer size 1024 bytes
   print("Receive message:", data)