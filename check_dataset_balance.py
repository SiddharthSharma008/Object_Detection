import os
from collections import defaultdict

def check_dataset_balance(labels_dir):
    """Check the balance of classes in your YOLO format annotations."""
    class_counts = defaultdict(int)
    
    if not os.path.exists(labels_dir):
        print(f"❌ Directory not found: {labels_dir}")
        return
    
    for txt_file in os.listdir(labels_dir):
        if txt_file.endswith('.txt'):
            with open(os.path.join(labels_dir, txt_file), 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
    
    if not class_counts:
        print("⚠️ No annotations found!")
        return
    
    print("\n" + "="*50)
    print("📊 Dataset Class Distribution:")
    print("="*50)
    for class_id in sorted(class_counts.keys()):
        print(f"Class {class_id}: {class_counts[class_id]:>6} annotations")
    print("="*50)
    
    total = sum(class_counts.values())
    print(f"Total annotations: {total}")
    print("="*50 + "\n")
    
    # Check balance
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    balance_ratio = max_count / min_count if min_count > 0 else 0
    
    if balance_ratio > 2:
        print(f"⚠️  Dataset is IMBALANCED!")
        print(f"   Max/Min ratio: {balance_ratio:.2f}")
        print("   💡 Consider collecting more data for underrepresented classes.\n")
    else:
        print(f"✅ Dataset is WELL-BALANCED!")
        print(f"   Max/Min ratio: {balance_ratio:.2f}\n")

if __name__ == "__main__":
    # Check your train set
    train_labels = r"E:\Siddy\Object Detection\gym_dataset\labels\train"
    check_dataset_balance(train_labels)
    
    # Check your val set
    val_labels = r"E:\Siddy\Object Detection\gym_dataset\labels\val"
    check_dataset_balance(val_labels)
