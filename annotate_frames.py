import cv2
import os

# Directory containing extracted frames
frames_dir = r"E:\Siddy\Object Detection\GymFrames"

# Directory to save annotated frames
output_dir = r"E:\Siddy\Object Detection\AnnotatedFrames"
os.makedirs(output_dir, exist_ok=True)

# Gym equipment classes (from auto_annotate.py reference)
gym_classes = {
    0: 'treadmill',
    1: 'bench',
    2: 'dumbbell',
    3: 'barbell',
    4: 'exercise_bike',
}

# Set to collect unique labels
unique_labels = set()

# Process each frame
for filename in os.listdir(frames_dir):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(frames_dir, filename)
        frame = cv2.imread(image_path)

        if frame is None:
            continue

        h, w = frame.shape[:2]

        # Corresponding label file
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(frames_dir, label_filename)

        if not os.path.exists(label_path):
            print(f"No label file for {filename}, skipping annotation.")
            continue

        # Read and parse labels
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Draw bounding boxes from labels
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            box_w = float(parts[3])
            box_h = float(parts[4])

            # Convert to pixel coordinates
            x_center = int(center_x * w)
            y_center = int(center_y * h)
            box_width = int(box_w * w)
            box_height = int(box_h * h)

            x0 = x_center - box_width // 2
            y0 = y_center - box_height // 2
            x1 = x_center + box_width // 2
            y1 = y_center + box_height // 2

            # Get label
            label = gym_classes.get(class_id, f'class_{class_id}')
            unique_labels.add(label)

            # Draw bounding box with corner accents
            color = (0, 255, 255)
            thickness = 2
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
            corner_len = int(min(box_width, box_height) * 0.15)
            # Top-left
            cv2.line(frame, (x0, y0), (x0 + corner_len, y0), color, thickness)
            cv2.line(frame, (x0, y0), (x0, y0 + corner_len), color, thickness)
            # Top-right
            cv2.line(frame, (x1, y0), (x1 - corner_len, y0), color, thickness)
            cv2.line(frame, (x1, y0), (x1, y0 + corner_len), color, thickness)
            # Bottom-left
            cv2.line(frame, (x0, y1), (x0 + corner_len, y1), color, thickness)
            cv2.line(frame, (x0, y1), (x0, y1 - corner_len), color, thickness)
            # Bottom-right
            cv2.line(frame, (x1, y1), (x1 - corner_len, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 - corner_len), color, thickness)

            # Label
            cv2.putText(frame, label, (x0, max(0, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save annotated frame
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)
        print(f"Annotated {filename}")

print("✅ Annotation complete! Annotated frames saved in AnnotatedFrames.")
print(f"Unique labels detected: {sorted(unique_labels)}")