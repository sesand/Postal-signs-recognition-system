import cv2
import matplotlib.pyplot as plt

# Function to create a video summary with predicted labels
def create_video_summary(video_path, predicted_label):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    key_frames_indices = range(0, frame_count, frame_count // 10)  # Adjust the step size as needed

    for i in key_frames_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Display key frames with predicted label
    plt.figure(figsize=(12, 8))
    for i, frame in enumerate(frames):
        plt.subplot(2, 5, i+1)
        plt.imshow(frame)
        plt.axis('off')
        plt.title(predicted_label)
        plt.tight_layout()

    plt.show()

# Path to the video file you want to visualize
video_path = '/content/drive/MyDrive/Sesan/letter/Komal_letter.mp4'  # Update with your video path

# Predict label of the video
predicted_label = predict_video_label(video_path)

# Create the video summary with predicted label
create_video_summary(video_path, predicted_label)
