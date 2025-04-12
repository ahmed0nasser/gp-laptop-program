import socket

# First ESP32's IP address and port
esp32_ip = "192.168.22.164"  # Replace with the actual IP address of the first ESP32
esp32_port = 1234

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_command(command):
    """
    Send a command to the ESP32 over the persistent TCP connection.
    """
    # Send data to the first ESP32
    sock.sendto(command.encode(), (esp32_ip, esp32_port))
