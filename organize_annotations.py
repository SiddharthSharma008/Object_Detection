import os
import shutil
from pathlib import Path
import random

def organize_annotations(gymframes_dir, dataset_dir, train_split=0.8):
    """
    Organize annotated images and labels into train/val split.
    
    Args:
        gymframes_dir: Directory containing annotated .jpg files and .txt label files
        dataset_dir: Destination gym_dataset directory
        train_split: Proportion of data for training (default 80/20)
    """
    
    images_dir = Path(dataset_dir) / "images"
    labels_dir = Path(dataset_dir) / "labels"
    
    train_img_dir = images_dir / "train"
    val_img_dir = images_dir / "val"
    train_lbl_dir = labels_dir / "train"
    val_lbl_dir = labels_dir / "val"
    
    # Get all image files
    gymframes_path = Path(gymframes_dir)
    image_files = [f for f in gymframes_path.glob("*.jpg")]
    
    if not image_files:
        print(f"❌ No images found in {gymframes_dir}")
        return
    
    print(f"📊 Found {len(image_files)} annotated images")
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"📈 Splitting: {len(train_files)} train, {len(val_files)} val\n")
    
    # Copy train files
    print("Copying train files...")
    for img_file in train_files:
        txt_file = img_file.with_suffix('.txt')
        if txt_file.exists():
            shutil.copy2(img_file, train_img_dir / img_file.name)
            shutil.copy2(txt_file, train_lbl_dir / txt_file.name)
        else:
            print(f"⚠️  Warning: {txt_file.name} not found (image might not be annotated)")
    
    # Copy val files
    print("Copying validation files...")
    for img_file in val_files:
        txt_file = img_file.with_suffix('.txt')
        if txt_file.exists():
            shutil.copy2(img_file, val_img_dir / img_file.name)
            shutil.copy2(txt_file, val_lbl_dir / txt_file.name)
        else:
            print(f"⚠️  Warning: {txt_file.name} not found (image might not be annotated)")
    
    print("\n✅ Dataset organized successfully!")
    print(f"   Train: {len(list(train_img_dir.glob('*.jpg')))} images")
    print(f"   Val:   {len(list(val_img_dir.glob('*.jpg')))} images")

if __name__ == "__main__":
    gymframes_dir = r"E:\Siddy\Object Detection\GymFrames"
    dataset_dir = r"E:\Siddy\Object Detection\gym_dataset"
    
    organize_annotations(gymframes_dir, dataset_dir, train_split=0.8)
