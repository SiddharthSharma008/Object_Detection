"""
Gym Equipment Detection Script
Detects and draws bounding boxes around gym equipment in images and videos.
"""
import cv2
import torch
from ultralytics import YOLO
import os
from pathlib import Path

# Configuration
MODEL_PATH = 'best.pt'
CONFIDENCE_THRESHOLD = 0.15  # Lowered for better recall
IOU_THRESHOLD = 0.45  # NMS threshold for duplicate suppression
INPUT_SOURCE = 'GymFrames'  # Can be image file, video file, or directory
OUTPUT_DIR = 'DetectedEquipment'
SAVE_ANNOTATED_FRAMES = True

# Preprocessing options
ENHANCE_CONTRAST = True
CLARITY_BOOST = True
MULTI_SCALE_DETECTION = True

# Gym equipment classes (from gym.yaml)
GYM_CLASSES = {
    0: 'treadmill',
    1: 'bench',
    2: 'dumbbell',
    3: 'barbell',
    4: 'exercise_bike'
}

# Color for bounding boxes (BGR format)
BOX_COLOR = (0, 255, 255)  # Yellow
TEXT_COLOR = (0, 255, 255)  # Yellow
THICKNESS = 2

def enhance_image(frame):
    """Enhance image for better detection."""
    if ENHANCE_CONTRAST:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if CLARITY_BOOST:
        # Unsharp masking for edge enhancement
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        frame = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        frame = cv2.convertScaleAbs(frame)
    
    return frame

def draw_bounding_box(frame, x1, y1, x2, y2, label, confidence):
    """Draw bounding box with label on frame."""
    h, w = frame.shape[:2]
    
    # Ensure coordinates are within frame
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    
    # Draw main rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)
    
    # Draw corner accents
    corner_len = int(min(x2 - x1, y2 - y1) * 0.15)
    corners = [
        ((x1, y1), (x1 + corner_len, y1), (x1, y1 + corner_len)),  # Top-left
        ((x2, y1), (x2 - corner_len, y1), (x2, y1 + corner_len)),  # Top-right
        ((x1, y2), (x1 + corner_len, y2), (x1, y2 - corner_len)),  # Bottom-left
        ((x2, y2), (x2 - corner_len, y2), (x2, y2 - corner_len))   # Bottom-right
    ]
    
    for corner_point, line1_end, line2_end in corners:
        cv2.line(frame, corner_point, line1_end, BOX_COLOR, THICKNESS)
        cv2.line(frame, corner_point, line2_end, BOX_COLOR, THICKNESS)
    
    # Draw label
    label_text = f"{label} ({confidence:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
    
    # Background rectangle for label
    bg_x1 = x1
    bg_y1 = max(0, y1 - text_size[1] - 10)
    bg_x2 = x1 + text_size[0] + 5
    bg_y2 = bg_y1 + text_size[1] + 5
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), BOX_COLOR, -1)
    cv2.putText(frame, label_text, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)

def detect_in_image(image_path, model, output_path):
    """Detect gym equipment in a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error reading image: {image_path}")
        return
    
    h, w = frame.shape[:2]
    
    # Enhance image for better detection
    enhanced_frame = enhance_image(frame.copy())
    
    # Run detection with optimization
    results = model(enhanced_frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    
    # Process detections
    detection_count = 0
    detected_boxes = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            detected_boxes.append((x1, y1, x2, y2, confidence, class_id))
    
    # Multi-scale detection: run detection at different scales for better accuracy
    if MULTI_SCALE_DETECTION:
        for scale in [0.8, 1.2]:
            scaled = cv2.resize(enhanced_frame, (int(w * scale), int(h * scale)))
            scale_results = model(scaled, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            
            for result in scale_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0] / scale)
                    confidence = float(box.conf[0]) * 0.95  # Slightly lower weight for scaled results
                    class_id = int(box.cls[0])
                    detected_boxes.append((x1, y1, x2, y2, confidence, class_id))
    
    # Draw all detected boxes on original frame
    for x1, y1, x2, y2, confidence, class_id in detected_boxes:
        detection_count += 1
        label = GYM_CLASSES.get(class_id, f'Unknown ({class_id})')
        draw_bounding_box(frame, x1, y1, x2, y2, label, confidence)
    
    # Save result
    cv2.imwrite(output_path, frame)
    return detection_count

def detect_in_video(video_path, model, output_dir):
    """Detect gym equipment in a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_name = Path(video_path).stem
    output_video = os.path.join(output_dir, f"{video_name}_detected.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_count = 0
    
    print(f"Processing video: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Enhance frame for better detection
        enhanced = enhance_image(frame.copy())
        
        # Run detection
        results = model(enhanced, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = GYM_CLASSES.get(class_id, f'Unknown ({class_id})')
                
                draw_bounding_box(frame, x1, y1, x2, y2, label, confidence)
        
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    
    print(f"Video saved: {output_video}")
    return detection_count

def process_directory(directory, model, output_dir):
    """Process all images in a directory."""
    image_files = list(Path(directory).glob('*.jpg')) + list(Path(directory).glob('*.png'))
    
    total_detections = 0
    
    for idx, image_path in enumerate(image_files, 1):
        output_path = os.path.join(output_dir, image_path.name)
        detections = detect_in_image(str(image_path), model, output_path)
        if detections:
            total_detections += detections
            print(f"[{idx}/{len(image_files)}] {image_path.name} - {detections} detections")
        else:
            print(f"[{idx}/{len(image_files)}] {image_path.name} - No detections")
    
    return total_detections

def main():
    """Main detection pipeline."""
    print("=" * 60)
    print("GYM EQUIPMENT DETECTION")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully!")
    print(f"Classes: {GYM_CLASSES}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    # Determine input type and process
    if os.path.isfile(INPUT_SOURCE):
        print(f"\nProcessing file: {INPUT_SOURCE}")
        if INPUT_SOURCE.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            detect_in_video(INPUT_SOURCE, model, OUTPUT_DIR)
        else:
            detect_in_image(INPUT_SOURCE, model, os.path.join(OUTPUT_DIR, Path(INPUT_SOURCE).name))
    
    elif os.path.isdir(INPUT_SOURCE):
        print(f"\nProcessing directory: {INPUT_SOURCE}")
        total = process_directory(INPUT_SOURCE, model, OUTPUT_DIR)
        print(f"\n✅ Total detections: {total}")
    
    else:
        print(f"Error: {INPUT_SOURCE} not found!")
    
    print(f"\n✅ Detection complete! Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
