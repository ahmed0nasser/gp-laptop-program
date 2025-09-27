import socket

# 1) Bind to all interfaces, port 5005
UDP_IP = ""           # empty = all local interfaces
UDP_PORT = 5006
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Listening on UDP port {UDP_PORT} …")

# 2) Receive loop
while True:
    data, addr = sock.recvfrom(1024)  # buffer size
    text = data.decode('utf-8', errors='ignore')
    print(f"Received from {addr}: {text}")