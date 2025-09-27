import cv2
import mediapipe as mp
import numpy as np
import time
import socket
import transmitter

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
mode = "HAND"

# Initialize video capture
cap = cv2.VideoCapture(0)

# Parameters for direction confirmation
threshold_frames = 10  # Number of frames to confirm direction change
direction_counter = 0  # Counter to confirm direction change
last_direction = None  # To store the last confirmed direction
current_direction = None  # Current detected direction
last_stop_direction = False  # Track if STOP was last sent (False = can move, True = can move after UP)

# Global socket for persistent ESP32 connection
client_socket = None

def calculate_direction(landmarks, image_shape):
    """Determine the hand movement direction."""
    wrist = [landmarks[0].x * image_shape[1], landmarks[0].y * image_shape[0]]
    middle_tip = [landmarks[12].x * image_shape[1], landmarks[12].y * image_shape[0]]
    vector = [middle_tip[0] - wrist[0], middle_tip[1] - wrist[1]]

    # Debug output for troubleshooting
    # print(f"Vector: X={vector[0]:.2f}, Y={vector[1]:.2f}")

    if abs(vector[0]) > abs(vector[1]):  # Horizontal movement
        if vector[0] > 0:
            return "RIGHT"
        else:
            return "LEFT"
    else:  # Vertical movement
        if vector[1] > 0:
            return "DOWN"
        else:
            return "UP"

def direction_map_func(detected_dir):
    """Map detected hand direction to commands."""
    global last_stop_direction
    if detected_dir == "RIGHT" and last_stop_direction:
        return "RIGHT"  # Send RIGHT as is
    elif detected_dir == "LEFT" and last_stop_direction:
        return "LEFT"   # Send LEFT as is
    elif detected_dir == "FORWARD" and last_stop_direction:
        return "FORWARD"   # No hand detected, move forward
    elif detected_dir == "UP":
        last_stop_direction = True  # Enable movement after STOP
        return "FORWARD"   # Start from rest
    else:  # DOWN
        last_stop_direction = False  # Disable movement until UP
        return "STOP"   # Stop and wait for UP

# Connect to ESP32 at startup
#connect_to_esp32()

# Main loop
try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally and convert it to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image with MediaPipe Hands
        results = hands.process(image)

        # Convert the image back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        # Default direction when no hand is detected
        detected_direction = "FORWARD"

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # Calculate the hand direction
                detected_direction = calculate_direction(hand_landmarks.landmark, image.shape)

        # Update the direction after confirming consistency
        if detected_direction == current_direction:
            direction_counter += 1
        else:
            current_direction = detected_direction
            direction_counter = 1

        # Confirm direction change after threshold frames
        if (direction_counter >= threshold_frames) or (current_direction != last_direction): #and current_direction != last_direction:
            
            mapped_direction = direction_map_func(current_direction)
            
            transmitter.send_command(f"{mapped_direction}|{mode}")
            cv2.putText(image, f"mapped_direction: {mapped_direction}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Detected: {current_direction}, Mapped: {mapped_direction}")
           # send_command(mapped_direction, "HAND")  # Send mapped direction to ESP32
            last_direction = current_direction  # Update last confirmed direction
            direction_counter=0
        # Display the direction on the image
        cv2.putText(image, f"Direction: {current_direction}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Hand Direction Control', image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program terminated by user")

# Release resources
cap.release()
cv2.destroyAllWindows()
if client_socket:
    client_socket.close()
    print("Socket connection closed")