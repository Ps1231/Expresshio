import cv2
import os
import numpy as np
import mediapipe as mp
from key_points import mediapipe_detection , draw_styled_landmarks

# Mediapipe modules
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Number of sequences (videos) to collect per action
no_sequences = 30

# Length of each sequence (video)
sequence_length = 30

# Folder start index
start_folder = 30

# Function to extract keypoints from the detected results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Setup directories for collecting data
for action in actions:
    try:
        dirmax = np.max([int(d) for d in os.listdir(os.path.join(DATA_PATH, action)) if d.isdigit()])
    except ValueError:  # In case there are no numeric directories
        dirmax = 0
    # dirmax = np.max([int(d) for d in os.listdir(os.path.join(DATA_PATH, action)) if d.isdigit()], default=0)
    for sequence in range(1, no_sequences + 1):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(dirmax + sequence)))
        except:
            pass

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up the Mediapipe Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    action_idx = 0  # Start with the first action
    collecting = False  # Flag to check if data collection has started

    while True:
        action = actions[action_idx]
        print(f"Ready to collect data for: {action}")

        # Wait for 'S' key to start collecting for current action
        while not collecting:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"Press 'S' to start collecting for {action}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # Press 'S' to start collecting data
                collecting = True
                print(f"Collecting data for {action}...")
                break

            if key == ord('q'):  # Press 'Q' to quit
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Collect data for the current action
        for sequence in range(start_folder, start_folder + no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Display instructions during collection
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # Extract keypoints
                keypoints = extract_keypoints(results)

                # Save the keypoints in the appropriate folder
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Handle key press for switching actions or stopping the collection
                key = cv2.waitKey(10) & 0xFF

                if key == ord('q'):  # Press 'q' to quit
                    print("Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                if key == ord(' '):  # Press space to move to the next action
                    print(f"Switching to next action: {actions[(action_idx + 1) % len(actions)]}")
                    action_idx = (action_idx + 1) % len(actions)  # Switch to next action
                    collecting = False  # Stop collecting for this action and wait for 'S' key to start the next action
                    break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Function to detect mediapipe landmarks
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make predictions
    image.flags.writeable = True  # Image is now writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results

# Function to draw landmarks
def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,  # Face mesh connection
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
