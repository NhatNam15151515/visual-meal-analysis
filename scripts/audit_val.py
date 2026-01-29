
import cv2
import yaml
import numpy as np
import random
from pathlib import Path
import os
import shutil

# Config
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "uecfoodpix_yolo"
VAL_IMG_DIR = DATA_DIR / "val" / "images"
VAL_LABEL_DIR = DATA_DIR / "val" / "labels"
OUTPUT_DIR = ROOT_DIR / "outputs" / "audit_val"
NUM_SAMPLES = 20

# Colors for classes (random fixed)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8).tolist()

def audit_val():
    # Load config
    yaml_path = DATA_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found at {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', {})
    if isinstance(class_names, list): # Handle listing format if needed (though yaml usually parses dict for names)
        class_names = {i: n for i, n in enumerate(class_names)}

    # Get images
    all_imgs = list(VAL_IMG_DIR.glob("*.jpg")) + list(VAL_IMG_DIR.glob("*.png"))
    if not all_imgs:
        print(f"[ERROR] No images found in {VAL_IMG_DIR}")
        return
        
    print(f"[INFO] Found {len(all_imgs)} validation images.")
    
    # Analyze all label files for basic stats
    empty_labels = 0
    missing_labels = 0
    
    print("[INFO] Checking label integrity...")
    for img_path in all_imgs:
        label_path = VAL_LABEL_DIR / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        if label_path.stat().st_size == 0:
            empty_labels += 1
            
    print(f"[REPORT] Missing labels: {missing_labels}")
    print(f"[REPORT] Empty labels (background images): {empty_labels}")
    
    # Sample for visualization
    samples = random.sample(all_imgs, min(len(all_imgs), NUM_SAMPLES))
    
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Visualizing {len(samples)} samples to {OUTPUT_DIR}...")
    
    for img_path in samples:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        overlay = img.copy()
        
        # Load label
        label_path = VAL_LABEL_DIR / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Draw
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            cls_idx = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Reshape to (N, 2)
            points = np.array(coords).reshape(-1, 2)
            
            # Denormalize
            points[:, 0] *= w
            points[:, 1] *= h
            points = points.astype(np.int32)
            
            # Color
            color = [int(c) for c in COLORS[cls_idx % len(COLORS)]]
            
            # Draw polygon (Filled semi-transparent)
            cv2.fillPoly(overlay, [points], color)
            
            # Draw border
            cv2.polylines(img, [points], True, color, 2)
            
            # Label box
            pt = points[0]
            label_text = class_names.get(cls_idx, str(cls_idx))
            
            # Draw text w/ background
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (pt[0], pt[1] - text_h - 5), (pt[0] + text_w, pt[1]), color, -1)
            cv2.putText(img, label_text, (pt[0], pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Blend overlay
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Save
        out_path = OUTPUT_DIR / f"vis_{img_path.name}"
        cv2.imwrite(str(out_path), img)
        
    print(f"[SUCCESS] Visualization saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    audit_val()
