import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

def load_data(preprocessed_dir):
    """
    Loads preprocessed data from the given directory.

    Args:
        preprocessed_dir (str): Directory containing preprocessed data.

    Returns:
        data (numpy array): Preprocessed data.
        labels (numpy array): Corresponding labels.
    """
    data = []
    labels = []

    for idx, gesture_folder in enumerate(os.listdir(preprocessed_dir)):
        gesture_path = os.path.join(preprocessed_dir, gesture_folder)
        for img_file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_file)
            img = np.load(img_path)
            data.append(img)
            labels.append(idx)

    data = np.array(data)
    data = data.reshape(data.shape[0], 64, 64, 1)
    labels = np.array(labels)
    return data, labels

def build_model(num_classes):
    """
    Builds a deeper CNN model with dropout to prevent overfitting.

    Args:
        num_classes (int): Number of gesture classes.

    Returns:
        model (Sequential): Built CNN model.
    """
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Added dropout layer
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, X_train, y_train, X_test, y_test, class_weights):
    """
    Trains the model with early stopping and class weighting.

    Args:
        model (Sequential): Built CNN model.
        X_train (numpy array): Training data.
        y_train (numpy array): Training labels.
        X_test (numpy array): Testing data.
        y_test (numpy array): Testing labels.
        class_weights (dict): Class weights to handle class imbalance.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.compile(optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), 
              class_weight=class_weights, callbacks=[early_stopping])
    
    model.save('models/best_model.keras')

    # Evaluate the model
    predictions = model.predict(X_test)
    y_pred = predictions.argmax(axis=1)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    preprocessed_dir = 'preprocessed_data'
    data, labels = load_data(preprocessed_dir)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)
    
    # Determine number of classes
    num_classes = len(np.unique(labels))
    
    # Calculate class weights to handle class imbalance
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))
    
    # Build and train the model
    model = build_model(num_classes)
    train_model(model, X_train, y_train, X_test, y_test, class_weights)
