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
from access_token import set_access_token
import logging
import json
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get local IP address
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
port = 8083

client_processes = {}  # Dictionary to store processes per client
client_data_tasks = {}  # Dictionary to store data streaming tasks per client

async def simulate_data_producer(queue):
    """Simulate asynchronous data production with varied health data."""
    while True:
        # Simulate realistic health data with wider variation
        health_data = {
            "temperature": round(random.uniform(36.0, 39.0), 1),  # Body temperature range (normal to fever)
            "bloodOxygen": random.randint(90, 100),  # Blood oxygen range (normal to slightly low)
            "heartRate": random.randint(60, 120)  # Heart rate range (resting to elevated)
        }
        await queue.put(health_data)
        logger.info(f"Produced new health data: {health_data}")
        await asyncio.sleep(random.uniform(3, 7))  # Random interval between 3-7 seconds

async def send_real_time_data(websocket, queue):
    """Send real-time health data to the client."""
    try:
        while True:
            data = await queue.get()
            await websocket.send(json.dumps(data))
            logger.info(f"Sent data to {websocket.remote_address[0]}: {data}")
            queue.task_done()
    except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
        logger.info(f"Stopped sending real-time data to {websocket.remote_address[0]}")
        raise

async def wss(websocket):
    client_ip = websocket.remote_address[0]
    print(f"Client {client_ip} connected")
    client_processes[websocket] = None  # Initialize process for this client
    client_data_tasks[websocket] = None  # Initialize data task for this client
    data_queue = asyncio.Queue()  # Create a queue for this client's data

    try:
        async for message in websocket:
            print(f"Received from {client_ip}: {message}")

            if message == "reset":
                transmitter.send_command("STOP|REMOTE")  # Stop movement
                print("Mode selection reseted")
            elif message == "send_real_time":
                if not client_data_tasks[websocket]:
                    # Start the data producer if not already running
                    asyncio.create_task(simulate_data_producer(data_queue))
                    # Start sending data to client
                    client_data_tasks[websocket] = asyncio.create_task(
                        send_real_time_data(websocket, data_queue)
                    )
                    print(f"Started sending real-time data to {client_ip}")
                else:
                    print(f"Real-time data already streaming to {client_ip}")
            elif message == "stop_real_time":
                if client_data_tasks[websocket]:
                    client_data_tasks[websocket].cancel()
                    client_data_tasks[websocket] = None
                    print(f"Stopped sending real-time data to {client_ip}")
                else:
                    print(f"No real-time data streaming to stop for {client_ip}")
            else:
                parsed_message = message.split(":")

                if parsed_message[0] == "select_mode":  # Mode selection
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
                elif parsed_message[0] == "remote":  # Remote control
                    movement_command = parsed_message[1].upper()
                    transmitter.send_command(f"{movement_command}|REMOTE")
                    print(f"Send {movement_command}|REMOTE to ESP32")
                elif parsed_message[0] == "access_token":  # Access token
                    set_access_token(parsed_message[1])
                else:
                    print(f"Unknown command: {message}")
    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_ip} disconnected")

async def main():
    print(f"WebSocket server is running on ws://{local_ip}:{port}")
    async with websockets.serve(wss, local_ip, port):
        await asyncio.Future()  # Keeps the server running

if __name__ == "__main__":
    asyncio.run(main())