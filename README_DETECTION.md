# Gym Equipment Detection System

## Overview
Detects and draws bounding boxes around gym equipment in images and videos using a custom-trained YOLOv5 model.

## Setup & Usage

### Quick Start
```bash
python detect_gym_equipment.py
```

This processes all images in `GymFrames/` and saves annotated results to `DetectedEquipment/`.

### Configuration
Edit `detect_gym_equipment.py` to customize:
- `MODEL_PATH`: Path to the trained model (default: `best.pt`)
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.25)
- `INPUT_SOURCE`: Input file/folder (default: `GymFrames`)
- `OUTPUT_DIR`: Output directory (default: `DetectedEquipment`)

### Supported Equipment Classes
- treadmill (class 0)
- bench (class 1)
- dumbbell (class 2)
- barbell (class 3)
- exercise_bike (class 4)

### Input Options

**Process directory of images:**
```python
INPUT_SOURCE = 'GymFrames'
```

**Process single image:**
```python
INPUT_SOURCE = 'path/to/image.jpg'
```

**Process video:**
```python
INPUT_SOURCE = 'path/to/video.mp4'
```

### Output
- **Images**: Annotated frames saved with yellow bounding boxes
- **Videos**: Processed video with detections saved as `.mp4`
- **Terminal**: Detection summary with counts per frame

### Model Files
- `best.pt`: Custom-trained model (primary)
- `yolov5su.pt`: Alternative model weights
- `gym.yaml`: Dataset configuration (5 equipment classes)

## Features
✅ Real-time equipment detection
✅ Multi-format support (JPG, PNG, MP4, AVI, etc.)
✅ Yellow bounding boxes with corner accents
✅ Confidence scores displayed
✅ Batch processing for directories
✅ Video processing with frame-by-frame detection

## Example Output
```
============================================================
GYM EQUIPMENT DETECTION
============================================================

Loading model: best.pt
Model loaded successfully!
Classes: {0: 'treadmill', 1: 'bench', 2: 'dumbbell', 3: 'barbell', 4: 'exercise_bike'}
Confidence threshold: 0.25

Processing directory: GymFrames
[1/258] V2_CAM 2_main_20260309105316_frame0.jpg - 12 detections
[2/258] V2_CAM 2_main_20260309105316_frame11.jpg - 8 detections
...
✅ Total detections: 350

✅ Detection complete! Results saved to: DetectedEquipment
============================================================
```

## Advanced Usage

### Adjust Detection Sensitivity
Lower confidence threshold = more detections (more false positives)
Higher confidence threshold = fewer detections (fewer false negatives)

Edit in `detect_gym_equipment.py`:
```python
CONFIDENCE_THRESHOLD = 0.15  # More sensitive
CONFIDENCE_THRESHOLD = 0.50  # More strict
```

### Process Custom Video
```python
INPUT_SOURCE = r'Surveillance Videos/your_video.mp4'
```

## Troubleshooting
- **No detections?** Lower `CONFIDENCE_THRESHOLD` or check model file path
- **Slow processing?** Reduce image resolution or use GPU (requires CUDA)
- **Wrong detections?** Model may need retraining with more varied data

## Files Reference
- `detect_gym_equipment.py`: Main detection script (PRIMARY - USE THIS)
- `gym.yaml`: Equipment class definitions
- `best.pt`: Trained model weights
- `GymFrames/`: Input directory with extracted frames
- `DetectedEquipment/`: Output directory with annotated results
