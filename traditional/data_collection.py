# data_collection.py
import cv2
import os

# Define the gesture name
gesture_name = input("Enter the name of the gesture (e.g., 'attention', 'stop', etc.): ")

# Create a directory to save the gesture images
gesture_dir = os.path.join('data', gesture_name)
os.makedirs(gesture_dir, exist_ok=True)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
start_index = 0  # Start index for naming images
max_images = 100  # Number of images to capture for each gesture

print("Press 's' to start capturing frames. Press 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the live video feed
    cv2.imshow('Gesture Collection', frame)

    # Wait for the 's' key to start capturing frames
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        while start_index < max_images:
            ret, frame = cap.read()
            if not ret:
                break

            # Display the live video feed
            cv2.imshow('Gesture Collection', frame)

            # Save the frame in the gesture directory
            img_name = os.path.join(gesture_dir, f'{gesture_name}_{start_index}.jpg')
            cv2.imwrite(img_name, frame)
            print(f"Picture {start_index} saved")

            start_index += 1

            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()