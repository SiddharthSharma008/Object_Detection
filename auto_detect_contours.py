import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def auto_annotate_with_contours(images_dir, output_labels_dir, conf_threshold=0.5):
    """
    Auto-generate bounding box annotations using contour detection.
    
    This identifies potential equipment by detecting edges and contours,
    which can then be manually reviewed/refined in LabelImg.
    
    Args:
        images_dir: Directory containing extracted .jpg frames
        output_labels_dir: Directory to save YOLO format .txt annotation files
        conf_threshold: Minimum contour area ratio to consider as equipment
    """
    
    print("🤖 Detecting equipment using contour analysis...\n")
    
    images_path = Path(images_dir)
    output_path = Path(output_labels_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    print(f"📊 Found {len(image_files)} images to process\n")
    
    annotated_count = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_annotations = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Filter out very small or very large objects
            if (bw * bh) < 500 or (bw * bh) > (w * h * 0.5):
                continue
            
            # Filter by aspect ratio (avoid very thin objects)
            aspect_ratio = float(bw) / bh if bh != 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
            
            # Convert to YOLO format (normalized center_x, center_y, width, height)
            center_x = (x + bw / 2) / w
            center_y = (y + bh / 2) / h
            box_w = bw / w
            box_h = bh / h
            
            # Clamp values to [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            box_w = max(0, min(1, box_w))
            box_h = max(0, min(1, box_h))
            
            # Assign a default class (0 = generic equipment)
            # User will manually assign proper classes in LabelImg
            class_id = 0
            
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
    
    print(f"\n✅ Auto-detection complete!")
    print(f"   Images with detected objects: {annotated_count}/{len(image_files)}")
    print(f"   Labels saved to: {output_labels_dir}")
    print(f"\n📋 Next steps:")
    print(f"   1. Review annotations in LabelImg")
    print(f"   2. Delete false detections (background clutter)")
    print(f"   3. Assign correct class labels (treadmill, bench, dumbbell, etc.)")
    print(f"   4. Run: python organize_annotations.py")
    print(f"\n💡 TIP: This creates baseline boxes - manual refinement is important for quality!")

if __name__ == "__main__":
    images_dir = r"E:\Siddy\Object Detection\GymFrames"
    labels_dir = r"E:\Siddy\Object Detection\GymFrames"  # Save .txt files alongside images
    
    auto_annotate_with_contours(images_dir, labels_dir)
