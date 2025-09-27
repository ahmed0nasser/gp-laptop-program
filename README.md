# Smart Wheelchair Control System

This repository contains the core laptop program for a smart wheelchair system. It acts as a central hub, integrating with a mobile application and the wheelchair's hardware to provide advanced control and monitoring capabilities.

---

## 📋 Table of Contents

- [About The Project](#about-the-project)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🚀 About The Project

The Smart Wheelchair Laptop Program is designed to run on a patient's laptop, providing a bridge between the user, a mobile application, and the wheelchair's microcontroller. It processes commands, relays sensor data, and enables multiple modes of control to enhance the user's mobility and independence.

### Built With

- [Python](https://www.python.org/)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- [VOSK](https://alphacephei.com/vosk/) for voice recognition
- [WebSockets](https://pypi.org/project/websockets/) for mobile app communication

---

## ✨ Features

- **Voice Control**: Utilizes the VOSK speech recognition model to control wheelchair movement with voice commands.
- **Mobile App Integration**: A WebSocket server receives commands and settings from a companion mobile app.
- **Direct Hardware Communication**: A UDP client sends movement instructions directly to the wheelchair's microcontroller.
- **Real-time Sensor Data**: A WebSocket client forwards sensor readings from the wheelchair to a backend for processing and monitoring.
- **Multiple Control Modes**: The system is designed to handle various control scripts for different operational modes.

---

## 🏗️ System Architecture

This program serves as the central nervous system for the project, coordinating communication between three main components:

1.  **Mobile App**: The user can control the wheelchair and view its status through a mobile app. The app communicates with this laptop program via a WebSocket server (`ws_server.py`).
2.  **Laptop Program (This Repository)**: It receives commands from the mobile app or through direct input (like voice commands), processes them, and sends corresponding movement signals to the wheelchair.
3.  **Wheelchair Microcontroller**: The microcontroller receives movement commands from the laptop program via a UDP socket (`udp_connection.py` and `transmitter.py`) and controls the wheelchair's motors and actuators accordingly.

```
+----------------+      +-------------------------+      +---------------------------+
|  Mobile App    |      |  Laptop Program (Core)  |      | Wheelchair Microcontroller|
| (WebSocket)    | <--> |   (ws_server/client)    | <--> |      (UDP Socket)         |
+----------------+      +-------------------------+      +---------------------------+
                          |         ^
                          |         |
                          v         |
                      +-----------------+
                      | Voice Commands  |
                      |    (VOSK)       |
                      +-----------------+
```

---

## 🏁 Getting Started

Follow these steps to get the local development environment running.

### Prerequisites

Make sure you have Python 3.10 and pip installed on your system.

- **Python 3**
  ```sh
  python --version
  ```

### Installation

1.  **Clone the repository**
    ```sh
    git clone https://github.com/your_username/gp-laptop-program.git
    cd gp-laptop-program
    ```
2.  **Create and activate a virtual environment**
    - On macOS and Linux:
      ```sh
      python3.10 -m venv venv
      source venv/bin/activate
      ```
    - On Windows:
      ```sh
      py -3.10 -m venv venv
      .\venv\Scripts\activate
      ```
3.  **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

Once the installation is complete, run the main program from the root directory:

```sh
python main.py
```

This will start the WebSocket server to listen for mobile app connections and initialize the UDP socket for wheelchair communication.

---

## 📂 Project Structure

```
.
├── control_scripts/    # Scripts for different control modes and UDP transmission
├── models/             # VOSK voice recognition model
├── venv/               # Virtual environment directory
├── access_token.py     # Manages access token storage
├── main.py             # Main entry point of the program
├── udp_connection.py   # Handles the UDP socket connection to the wheelchair
├── ws_client.py        # WebSocket client to send sensor data to a backend
├── ws_server.py        # WebSocket server to receive commands from the mobile app
└── requirements.txt    # Project dependencies
```

---
