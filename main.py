# main.py
import sys
import os

# Get the absolute path to the parent directory of project_scripts
project_scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_scripts'))

# Add it to sys.path if it's not already there
if project_scripts_path not in sys.path:
    sys.path.append(project_scripts_path)

import asyncio
import websockets
import socket
import subprocess
import transmitter

# Get local IP address
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
port = 8080

client_processes = {}  # Dictionary to store processes per client

async def echo(websocket):
    client_ip = websocket.remote_address[0]
    print(f"Client {client_ip} connected")
    client_processes[websocket] = None  # Initialize process for this client
    try:
        async for message in websocket:
            print(f"Received from {client_ip}: {message}")

            if client_processes[websocket]:
                print(f"Terminating previous process for {client_ip}")
                client_processes[websocket].terminate()
                client_processes[websocket].wait()

            if message == "reset":
                transmitter.send_command("STOP|REMOTE")  # Stop movement
                print("Mode selection Reseted")
                return

            parsed_message = message.split(":")

            if parsed_message[0] == "select_mode":
                mode = parsed_message[1]
                if mode == "face":
                    process = subprocess.Popen(["python", "./project_scripts/face.py"])
                elif mode == "eye":
                    process = subprocess.Popen(["python", "./project_scripts/eye.py"])
                elif mode == "voice":
                    process = subprocess.Popen(["python", "./project_scripts/speech_vosk2.py"])
                elif mode == "hand":
                    process = subprocess.Popen(["python", "./project_scripts/hand.py"])
                else:
                    print(f"Unknown mode: {mode}")
                    return
                print(f"Execute {mode} mode")
                client_processes[websocket] = process
            elif parsed_message[0] == "remote":
                movement_command = parsed_message[1].upper()
                transmitter.send_command(f"{movement_command}|REMOTE")
                print(f"Send {movement_command}|REMOTE to ESP32")
            else:
                print(f"Unknown command: {message}")
    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_ip} disconnected")

async def main():
    print(f"WebSocket server is running on ws://{local_ip}:{port}")
    async with websockets.serve(echo, local_ip, port):
        await asyncio.Future()  # Keeps the server running

if __name__ == "__main__":
    asyncio.run(main())