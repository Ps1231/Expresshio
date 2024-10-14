# preprocess_data.py
import cv2
import os
import numpy as np

def preprocess_image(img_path, target_size):
    """
    Preprocesses a single image.

    Args:
        img_path (str): Path to the image file.
        target_size (tuple): Target size for resizing the image.

    Returns:
        preprocessed_img (numpy array): Preprocessed image.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    return img_normalized

def preprocess_data(data_dir, preprocessed_dir, target_size):
    """
    Preprocesses image data in the given directory.

    Args:
        data_dir (str): Directory containing gesture image data.
        preprocessed_dir (str): Directory to save preprocessed data.
        target_size (tuple): Target size for resizing images.
    """
    for gesture_folder in os.listdir(data_dir):
        gesture_path = os.path.join(data_dir, gesture_folder)
        preprocessed_gesture_dir = os.path.join(preprocessed_dir, gesture_folder)
        os.makedirs(preprocessed_gesture_dir, exist_ok=True)

        for img_file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_file)
            preprocessed_img = preprocess_image(img_path, target_size)
            preprocessed_img_name = os.path.join(preprocessed_gesture_dir, img_file.replace('.jpg', '.npy'))
            np.save(preprocessed_img_name, preprocessed_img)
            print(f"Preprocessed and saved {preprocessed_img_name}")

if __name__ == "__main__":
    data_dir = 'data'
    preprocessed_dir = 'preprocessed_data'
    target_size = (64, 64)
    os.makedirs(preprocessed_dir, exist_ok=True)
    preprocess_data(data_dir, preprocessed_dir, target_size)