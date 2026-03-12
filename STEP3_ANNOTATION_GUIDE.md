# 🖊️ Step 3: Annotation — Detailed Guide

## 🔹 3.1 Install and Launch LabelImg
- In your VS Code terminal (with venv activated):
  ```bash
  pip install labelImg
  ```
- Start the tool:
  ```bash
  labelImg
  ```
- A GUI window opens where you can load images and draw bounding boxes.

---

## 🔹 3.2 Load Your Frames
- In LabelImg, click **Open Dir**.
- Select your folder:  
  `E:\Siddy\Object Detection\GymFrames`
- You'll see your extracted frames listed.

---

## 🔹 3.3 Define Classes
- Click **Change Save Dir** → point it to the same folder (`GymFrames`) or directly into `gym_dataset/labels/train`.
- Click **Create RectBox** → draw a box around an object (e.g., treadmill).
- When prompted for a label, type the class name (e.g., `treadmill`).
- Repeat for other equipment: `bench`, `dumbbell`, `barbell`, `exercise bike`.

💡 Tip: Keep class names consistent with your `gym.yaml`.

---

## 🔹 3.4 Save in YOLO Format
- In the menu, select **YOLO** as the output format.
- Each image will now have a `.txt` file with annotations.
- Example `frame123.txt`:
  ```
  0 0.45 0.60 0.20 0.30
  2 0.70 0.50 0.15 0.25
  ```
  - `0` = treadmill (class ID)
  - `2` = dumbbell (class ID)
  - Numbers = normalized coordinates (center_x, center_y, width, height).

---

## 🔹 3.5 Annotation Best Practices
- **Accuracy matters**: boxes should tightly fit the equipment.
- **Diversity**: annotate equipment in different angles, lighting, and usage scenarios.
- **Balance classes**: try to have similar numbers of images per class.  
  - Example: 500 treadmill, 500 dumbbell, 500 bench, etc.
- **Skip clutter**: don't annotate irrelevant background objects.

---

## 🔹 3.6 Organize Annotations
- Once you finish annotating:
  - Move images + `.txt` files into `gym_dataset/images/train` and `gym_dataset/images/val`.
  - Labels go into `gym_dataset/labels/train` and `gym_dataset/labels/val`.

---

## ✅ Why This Step Is Critical
- YOLOv5 doesn't "know" what a treadmill looks like until you teach it with bounding boxes.
- Poor annotations = poor detection (e.g., boxes too loose, wrong class names).
- Good annotations = strong, reliable detection in real surveillance videos.

---

## 📊 Next: Dataset Balance Checker
You can use the script below to analyze your annotations and ensure balanced class distribution before training:

```python
import os
from collections import defaultdict

def check_dataset_balance(labels_dir):
    """Check the balance of classes in your YOLO format annotations."""
    class_counts = defaultdict.count()
    
    for txt_file in os.listdir(labels_dir):
        if txt_file.endswith('.txt'):
            with open(os.path.join(labels_dir, txt_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    
    print("📊 Dataset Class Distribution:")
    print("-" * 40)
    for class_id in sorted(class_counts.keys()):
        print(f"Class {class_id}: {class_counts[class_id]} annotations")
    print("-" * 40)
    
    total = sum(class_counts.values())
    print(f"Total annotations: {total}")
    
    # Check balance
    if not class_counts:
        print("⚠️ No annotations found!")
        return
    
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    balance_ratio = max_count / min_count if min_count > 0 else 0
    
    if balance_ratio > 2:
        print(f"⚠️ Dataset is imbalanced! Max/Min ratio: {balance_ratio:.2f}")
        print("💡 Consider collecting more data for underrepresented classes.")
    else:
        print(f"✅ Dataset is well-balanced! Max/Min ratio: {balance_ratio:.2f}")

# Usage:
# check_dataset_balance(r"E:\Siddy\Object Detection\gym_dataset\labels\train")
```

Save this as `check_dataset_balance.py` and run it after annotating to ensure your data is ready for training!
