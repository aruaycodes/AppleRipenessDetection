import os
import shutil
from pathlib import Path
import random

def create_yolo_labels(image_path, ripeness_level):
    """Create YOLO format label for an image"""
    # In YOLO format, we'll place the apple in the center of the image
    # with a reasonable size (80% of image width and height)
    return f"{ripeness_level} 0.5 0.5 0.8 0.8\n"

def prepare_dataset():
    # Create necessary directories
    os.makedirs('data/train/images', exist_ok=True)
    os.makedirs('data/train/labels', exist_ok=True)
    os.makedirs('data/val/images', exist_ok=True)
    os.makedirs('data/val/labels', exist_ok=True)

    # Define ripeness levels and their corresponding class indices
    ripeness_levels = {
        '20%': 0,
        '40%': 1,
        '60%': 2,
        '80%': 3,
        '100%': 4
    }

    # Source directory
    source_dir = 'data/APPLE RIPENESS LEVELS IMAGE DATASET'

    # Process each ripeness level
    for ripeness, class_idx in ripeness_levels.items():
        source_path = os.path.join(source_dir, ripeness)
        if not os.path.exists(source_path):
            print(f"Warning: Directory {source_path} not found")
            continue

        # Get all images in the directory
        images = [f for f in os.listdir(source_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly split into train and validation sets (80% train, 20% val)
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Process training images
        for img in train_images:
            # Copy image
            src_img = os.path.join(source_path, img)
            dst_img = os.path.join('data/train/images', f"{ripeness}_{img}")
            shutil.copy2(src_img, dst_img)

            # Create label
            label = create_yolo_labels(dst_img, class_idx)
            label_path = os.path.join('data/train/labels', f"{ripeness}_{os.path.splitext(img)[0]}.txt")
            with open(label_path, 'w') as f:
                f.write(label)

        # Process validation images
        for img in val_images:
            # Copy image
            src_img = os.path.join(source_path, img)
            dst_img = os.path.join('data/val/images', f"{ripeness}_{img}")
            shutil.copy2(src_img, dst_img)

            # Create label
            label = create_yolo_labels(dst_img, class_idx)
            label_path = os.path.join('data/val/labels', f"{ripeness}_{os.path.splitext(img)[0]}.txt")
            with open(label_path, 'w') as f:
                f.write(label)

    print("Dataset preparation completed!")

if __name__ == '__main__':
    prepare_dataset() 