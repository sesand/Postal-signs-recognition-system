import cv2
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error
from google.colab.patches import cv2_imshow  # Import cv2_imshow for Colab

def calculate_tracking_metrics(pred, threshold=5):
    precision = np.sum(pred < threshold) / len(pred)
    recall = np.sum(pred < threshold) / len(pred)
    return precision, recall

def calculate_rmse(pred, gt):
    return np.sqrt(mean_squared_error(gt, pred))

def sparse_optical_flow(video_path, output_folder, output_video_path):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)

    lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    frame_count = 0
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)

    frame_height, frame_width, _ = old_frame.shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

    fps_list = []

    while True:
        start_time = datetime.now()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.medianBlur(frame, 5)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:
            p1, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                for i, (new, old) in enumerate(zip(p1, p0)):
                    a, b = new.ravel()
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

                frame_count += 1
                frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
                cv2.imwrite(frame_filename, frame)

                out.write(frame)

                cv2_imshow(frame)  # Use cv2_imshow for displaying images in Colab

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)
        old_gray = frame_gray.copy()

        elapsed_time = (datetime.now() - start_time).total_seconds()
        fps = 1 / elapsed_time
        fps_list.append(fps)
        print(f'FPS: {fps}')

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    average_fps = sum(fps_list) / len(fps_list)

    print(f"Average FPS: {average_fps}")

if __name__ == "__main__":
    input_video_path = "/content/drive/MyDrive/Sesan_Project_02/videos/BS & Original videos/sesan/BS videos/letter_video001.mp4"
    output_folder = "/content/drive/MyDrive/Sesan-Project01/Object Tracking/Sparse Flow/KOMAL/letter/frames"
    output_video_path = "/content/drive/MyDrive/Sesan-Project01/Object Tracking/Sparse Flow/sesan/letter/video.mp4"
    sparse_optical_flow(input_video_path, output_folder, output_video_path)





