import cv2
import numpy as np
from tensorflow.keras.models import load_model
from time import sleep
from Adafruit_CharLCD import Adafruit_CharLCD

# Load the trained model
model = load_model('/content/drive/MyDrive/Postal_sign/3dcnn_model/model_for_13Signs.h5')

# Define categories
categories = ["STD", "advertisement", "airmail", "courier", "email", "envelope", "fax", "letter", "money_order", "postcard", "stamp", "telegram", "telephone"]  # Update with your categories

# Function to preprocess single video for prediction
def preprocess_video(video_path, frames_per_video=16, image_size=(64, 64)):
    frames = []
    cap = cv2.VideoCapture(video_path)

    for _ in range(frames_per_video):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, image_size)
            frames.append(frame)
        else:
            break

    cap.release()

    if len(frames) < frames_per_video:
        # Pad frames if video is shorter than frames_per_video
        frames.extend([frames[-1]] * (frames_per_video - len(frames)))

    return np.array(frames)

# Function to predict label of single video
def predict_video_label(video_path):
    preprocessed_video = preprocess_video(video_path)
    preprocessed_video = np.expand_dims(preprocessed_video, axis=0)  # Add batch dimension
    prediction = model.predict(preprocessed_video)
    predicted_label = categories[np.argmax(prediction)]
    return predicted_label

# Initialize LCD
lcd = Adafruit_CharLCD(rs=26, en=19, d4=13, d5=6, d6=5, d7=21, cols=16, lines=2)

# Path to the video file you want to visualize
video_path = '/content/drive/MyDrive/Postal_sign/object_tracking/lucas_kanade/courier/sesan.mp4'  # Update with your video path

# Predict label of the video
predicted_label = predict_video_label(video_path)

# Display predicted label on LCD
lcd.message(predicted_label)
sleep(5)  # Display for 5 seconds
lcd.clear()
