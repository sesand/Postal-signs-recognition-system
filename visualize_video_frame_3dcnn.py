import matplotlib.pyplot as plt

# Function to visualize frames with predicted label
def visualize_frames_with_label(video_path, predicted_label):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Display frames with predicted label
    plt.figure(figsize=(10, 8))
    for i, frame in enumerate(frames):
        plt.subplot(len(frames)//5 + 1, 5, i+1)
        plt.imshow(frame)
        plt.axis('off')
        plt.title(predicted_label)
        plt.tight_layout()

    plt.show()

# Path to the video file you want to visualize
video_path = '/content/drive/MyDrive/Sesan/fax/aathi.mp4'  # Update with your video path

# Predict label of the video
predicted_label = predict_video_label(video_path)

# Visualize the frames with predicted label
visualize_frames_with_label(video_path, predicted_label)
