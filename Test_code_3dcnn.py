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

# Path to the video file you want to predict
video_path = '/content/drive/MyDrive/Sesan_Project_02/lucas kanade/courier/hema.mp4'  # Update with your video path

# Predict label of the video
predicted_label = predict_video_label(video_path)
print("Predicted Label:", predicted_label)
