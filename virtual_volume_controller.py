import cv2
import mediapipe as mp
import os
import numpy as np
import math
import subprocess

# Define the minimum and maximum distances between the index finger and thumb for the volume control gesture
VOLUME_MIN_DIST = 50
VOLUME_MAX_DIST = 150

# Define the minimum and maximum values for the system volume
VOLUME_MIN = 0
VOLUME_MAX = 100

# Initialize the MediaPipe hands module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Initialize the MediaPipe hands model
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    # Initialize the window for displaying the video and volume bar
    cv2.namedWindow('MediaPipe Hands')

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB format for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe hands
        results = hands.process(image)

        # Draw the hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the index finger and thumb landmarks
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_x, thumb_y = thumb.x * image.shape[1], thumb.y * image.shape[0]
                index_x, index_y = index.x * image.shape[1], index.y * image.shape[0]

                # Calculate the distance between the index finger and thumb landmarks
                distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

                # Map the distance to a volume value between VOLUME_MIN and VOLUME_MAX
                volume_value = np.interp(distance, [VOLUME_MIN_DIST, VOLUME_MAX_DIST], [VOLUME_MIN, VOLUME_MAX])
                volume_value = int(volume_value)

                # Set the system volume to the mapped value
                subprocess.run(["osascript", "-e", f"set volume output volume {volume_value}"])

                # Draw the volume bar on the image
                bar_x = int(frame.shape[1] * 0.8)
                bar_y = int(frame.shape[0] * 0.2)
                bar_w = 20
                bar_h = int(frame.shape[0] * 0.6)
                bar_filled_h = int(volume_value / VOLUME_MAX * bar_h)
                cv2.rectangle(frame, (bar_x, bar_y + bar_h), (bar_x + bar_w, bar_y + bar_h - bar_filled_h), (0, 255, 0),
                              -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                cv2.putText(frame, f"{volume_value}%", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)

                # Display the frame with the volume bar
                cv2.imshow('MediaPipe Hands', frame)

                # Exit the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()


