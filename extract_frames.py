import cv2
import os

video_folder = r"E:\Siddy\Object Detection\Surveillance  Videos"
output_folder = r"E:\Siddy\Object Detection\GymFrames"

frames_per_second = 1  # adjustable

for filename in os.listdir(video_folder):
    if filename.endswith((".mp4", ".avi", ".mov")):
        video_path = os.path.join(video_folder, filename)
        cap = cv2.VideoCapture(video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = int(fps / frames_per_second) if fps > 0 else 1

        frame_count, saved_count = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_filename = f"{os.path.splitext(filename)[0]}_frame{frame_count}.jpg"
                cv2.imwrite(os.path.join(output_folder, frame_filename), frame)
                saved_count += 1
            frame_count += 1

        cap.release()
        print(f"Extracted {saved_count} frames from {filename}")

print("✅ Frame extraction complete! Frames saved in GymFrames.")