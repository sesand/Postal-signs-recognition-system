# Mount Google Drive to access dataset
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout

# Function to load and preprocess video data
def load_video_data(directory, categories, frames_per_video=16, image_size=(64, 64)):
    X = []
    y = []

    for category in categories:
        path = os.path.join(directory, category)
        class_label = categories.index(category)

        for video in os.listdir(path):
            video_path = os.path.join(path, video)
            cap = cv2.VideoCapture(video_path)
            frames = []

            for _ in range(frames_per_video):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, image_size)
                    frames.append(frame)
                else:
                    break

            cap.release()

            if len(frames) == frames_per_video:
                X.append(frames)
                y.append(class_label)

    X = np.array(X)
    y = np.array(y)
    return X, y

# Define categories and paths
categories = ["STD", "advertisement", "airmail","courier","email","money_order","postcard","stamp","telegram","telephone"]  # Update with your categories
data_directory = '/content/drive/MyDrive/Sesan_Project_02/lucas kanade'  # Update with your dataset directory

# Load and preprocess data
X, y = load_video_data(data_directory, categories)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define 3D CNN model
model = Sequential([
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
