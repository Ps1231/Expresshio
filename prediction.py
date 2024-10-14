# real_time_prediction.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/best_model.keras')

# Define the labels for each gesture
gesture_labels = {
    0: 'A',
    1: 'b',
    2: 'C',
    3: 'D'
}

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, grayscale, normalize)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    normalized_frame = np.reshape(normalized_frame, (1, 64, 64, 1))

    # Make predictions using the trained model
    predictions = model.predict(normalized_frame)
    predicted_label = np.argmax(predictions)

    # Display the predicted gesture on the screen
    cv2.putText(frame, gesture_labels[predicted_label], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Gesture Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()