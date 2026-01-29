
import cv2
import yaml
import numpy as np
import random
from pathlib import Path
import os

# Config
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "uecfoodpix_yolo"
OUTPUT_DIR = ROOT_DIR / "outputs" / "visualization"
NUM_SAMPLES = 10

def visualize():
    # Load config
    yaml_path = DATA_DIR / "data.yaml"
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    
    # Get images
    img_dir = DATA_DIR / "train" / "images"
    label_dir = DATA_DIR / "train" / "labels"
    
    all_imgs = list(img_dir.glob("*.jpg"))
    if not all_imgs:
        print("No images found!")
        return
        
    # Sample
    samples = random.sample(all_imgs, min(len(all_imgs), NUM_SAMPLES))
    
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Visualizing {len(samples)} images to {OUTPUT_DIR}...")
    
    for img_path in samples:
        # Load image
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Load label
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Draw
        for line in lines:
            parts = line.strip().split()
            cls_idx = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Reshape to (N, 2)
            points = np.array(coords).reshape(-1, 2)
            
            # Denormalize
            points[:, 0] *= w
            points[:, 1] *= h
            points = points.astype(np.int32)
            
            # Draw polygon
            color = (0, 255, 0) # Green
            cv2.polylines(img, [points], True, color, 2)
            
            # Draw label
            label = class_names[cls_idx]
            pt = points[0]
            cv2.putText(img, label, (pt[0], pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
        # Save
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path.name}")

if __name__ == "__main__":
    visualize()
