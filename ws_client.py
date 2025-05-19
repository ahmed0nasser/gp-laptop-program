import asyncio
import websockets
import json
import logging
import random
from asyncio import Queue
from access_token import get_access_token

# Configure server
ip = "127.0.0.1"
port = 8081
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        await asyncio.sleep(random.uniform(3, 5))  # Random interval between 3-7 seconds

async def connect_to_server():
    # WebSocket server details
    server_url = f"ws://{ip}:{port}"  # Replace with actual server IP and port
    data_queue = Queue()
    access_token = None

    # Delay connection by 5 seconds until it gets access token
    while not access_token:
        logger.info("No access_token yet, waiting 5 seconds before getting access token again...")
        await asyncio.sleep(5)
        access_token = get_access_token()

    logger.info("Initiating connection to server")

    # Start data producer
    asyncio.create_task(simulate_data_producer(data_queue))

    try:
        async with websockets.connect(server_url) as websocket:
            # Send authentication token
            auth_message = json.dumps({"accessToken": access_token})
            await websocket.send(auth_message)
            logger.info("Sent authentication token")

            # Wait for authentication response
            auth_response = await websocket.recv()
            auth_response = json.loads(auth_response)
            logger.info(f"Authentication response: {auth_response}")

            if auth_response.get("status") == "success":
                # Continuously process and send data from queue
                while True:
                    health_data = await data_queue.get()
                    await websocket.send(json.dumps(health_data))
                    logger.info(f"Sent health data: {health_data}")

                    # Wait for server acknowledgment
                    acknowledgment = await websocket.recv()
                    acknowledgment = json.loads(acknowledgment)
                    logger.info(f"Server acknowledgment: {acknowledgment}")
                    data_queue.task_done()
            else:
                logger.error("Authentication failed")

    except websockets.exceptions.ConnectionClosed:
        logger.error("Connection to server closed")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

# Run the WebSocket client
if __name__ == "__main__":
    asyncio.run(connect_to_server())