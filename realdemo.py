import cv2
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

# Load the trained model
model = load_model('my_model.h5')

# Re-compile the model to ensure metrics are built
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define a dictionary to map predicted labels to characters
map_characters = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 
    25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'
}

# Try using a different camera index if 0 doesn't work
cap = cv2.VideoCapture(0)  # Change to cap = cv2.VideoCapture(1) if needed

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the region of interest (ROI) dimensions
roi_top, roi_bottom, roi_right, roi_left = 100, 400, 350, 650  # Customize these values

# Set the model input size (should match the training input size)
imageSize = 50  # Change this to 50 as the model expects 50x50 images

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip the frame horizontally (optional)
    frame = cv2.flip(frame, 1)

    # Define the Region of Interest (ROI) on the frame
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    
    # Draw a rectangle around the ROI (for visualizing)
    cv2.rectangle(frame, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)

    # Preprocess the ROI: resize, convert to array, and normalize
    roi_resized = cv2.resize(roi, (imageSize, imageSize))  # Resize to 50x50
    roi_array = np.array(roi_resized, dtype='float32')
    roi_array = np.expand_dims(roi_array, axis=0)
    roi_array = preprocess_input(roi_array)  # Preprocessing for VGG16

    # Make prediction
    # Predict the class for the region of interest (roi)
    pred = model.predict(roi_array)
    pred_class = np.argmax(pred)  # Get the index of the class with the highest probability

    # Map the predicted class index to the corresponding character
    if pred_class in map_characters:
        pred_character = map_characters[pred_class]
        print(f"Predicted character: {pred_character}")
    else:
        print("Unknown prediction.")
    # Map the predicted index to the corresponding character

    # Annotate the frame with the prediction
    cv2.putText(frame, pred_character, (roi_right, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame with the ROI and the prediction
    cv2.imshow('Sign Language Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
