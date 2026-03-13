import cv2
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def auto_annotate_gym_equipment(images_dir, output_labels_dir):
    """
    Auto-generate bounding box annotations using YOLOv5 pre-trained model.
    
    This creates initial annotations that can be reviewed/refined in LabelImg.
    
    Args:
        images_dir: Directory containing extracted .jpg frames
        output_labels_dir: Directory to save YOLO format .txt annotation files
    """
    
    print("🤖 Loading YOLOv5 pre-trained model...")
    model = YOLO('yolov5s.pt')
    
    # Map COCO classes to gym equipment
    # This maps detected objects to gym equipment categories
    coco_to_gym = {
        'person': None,  # Skip people
        'sports ball': 'dumbbell',
        'baseball bat': 'barbell',
        'tennis racket': None,
        'bicycle': 'exercise_bike',
        'bench': 'bench',
        'backpack': None,
        'handbag': None,
    }
    
    gym_classes = {
        'treadmill': 0,
        'bench': 1,
        'dumbbell': 2,
        'barbell': 3,
        'exercise_bike': 4,
    }
    
    images_path = Path(images_dir)
    output_path = Path(output_labels_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    print(f"📊 Found {len(image_files)} images to annotate\n")
    
    annotated_count = 0
    
    for img_path in tqdm(image_files, desc="Annotating images"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Run inference
        results = model.predict(img, conf=0.5, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None:
            # Create empty annotation file
            txt_path = output_path / img_path.with_suffix('.txt').name
            txt_path.touch()
            continue
        
        # Extract detections
        boxes = results[0].boxes
        
        # Convert to YOLO format and filter
        yolo_annotations = []
        
        for box in boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Filter low confidence detections
            if conf < 0.5:
                continue
            
            # Get class name from COCO
            class_name = model.names[cls]
            
            # Skip non-gym objects
            if class_name not in coco_to_gym or coco_to_gym[class_name] is None:
                continue
            
            # Map to gym class
            gym_class = coco_to_gym[class_name]
            class_id = gym_classes[gym_class]
            
            # Convert to YOLO format (normalized center_x, center_y, width, height)
            center_x = ((x1 + x2) / 2) / w
            center_y = ((y1 + y2) / 2) / h
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            
            # Clamp values to [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            box_w = max(0, min(1, box_w))
            box_h = max(0, min(1, box_h))
            
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
        
        # Save annotations
        if yolo_annotations:
            txt_path = output_path / img_path.with_suffix('.txt').name
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            annotated_count += 1
        else:
            # Create empty annotation file
            txt_path = output_path / img_path.with_suffix('.txt').name
            txt_path.touch()
    
    print(f"\n✅ Auto-annotation complete!")
    print(f"   Images with detected equipment: {annotated_count}/{len(image_files)}")
    print(f"   Labels saved to: {output_labels_dir}")
    print("\n💡 TIP: Review and refine annotations in LabelImg before training!")

if __name__ == "__main__":
    images_dir = r"E:\Siddy\Object Detection\GymFrames"
    labels_dir = r"E:\Siddy\Object Detection\GymFrames\Labels"  # Save .txt files alongside images
    
    auto_annotate_gym_equipment(images_dir, labels_dir)
