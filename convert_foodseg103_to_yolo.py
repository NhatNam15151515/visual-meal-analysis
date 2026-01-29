"""
Convert FoodSeg103 semantic masks to YOLO segmentation format.

Logic:
- Skip class 0 (background)
- Map class 1-103 to YOLO index 0-102
- Use cv2.RETR_EXTERNAL for contour extraction
- Filter small contours (area < 50 pixels)
- Output: polygon coordinates normalized to image size
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def mask_to_yolo_polygons(mask, min_area=50):
    """
    Convert semantic mask to YOLO segmentation polygons.
    
    Args:
        mask: 2D array with class IDs as pixel values
        min_area: Minimum contour area to include
        
    Returns:
        List of (yolo_class_id, normalized_polygon_coords)
    """
    h, w = mask.shape
    annotations = []
    
    # Get unique classes (skip background = 0)
    unique_classes = np.unique(mask)
    
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
        
        # YOLO class index = original class - 1 (shift down since no background)
        yolo_class_id = int(class_id) - 1
        
        # Create binary mask for this class
        binary_mask = (mask == class_id).astype(np.uint8) * 255
        
        # Find external contours only
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Simplify contour to reduce points (epsilon = 1% of perimeter)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 3:  # Need at least 3 points for a polygon
                continue
            
            # Normalize coordinates
            coords = []
            for point in approx:
                x, y = point[0]
                coords.append(x / w)
                coords.append(y / h)
            
            annotations.append((yolo_class_id, coords))
    
    return annotations

def convert_dataset(data_root, min_area=50):
    """Convert all masks in FoodSeg103 to YOLO format."""
    data_root = Path(data_root)
    
    splits = ['train', 'validation']
    stats = {'train': 0, 'validation': 0, 'empty': 0}
    
    for split in splits:
        masks_dir = data_root / 'masks' / split
        labels_dir = data_root / 'labels' / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        if not masks_dir.exists():
            print(f"Warning: {masks_dir} not found, skipping.")
            continue
        
        mask_files = list(masks_dir.glob('*.png'))
        print(f"\nConverting {split}: {len(mask_files)} masks")
        
        for mask_path in tqdm(mask_files, desc=f"Converting {split}"):
            # Read mask (grayscale = class IDs)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # Convert to YOLO polygons
            annotations = mask_to_yolo_polygons(mask, min_area=min_area)
            
            # Write label file
            label_path = labels_dir / (mask_path.stem + '.txt')
            
            if annotations:
                with open(label_path, 'w') as f:
                    for yolo_class_id, coords in annotations:
                        coords_str = ' '.join(f'{c:.6f}' for c in coords)
                        f.write(f'{yolo_class_id} {coords_str}\n')
                stats[split] += 1
            else:
                # Create empty file for images with only background
                label_path.touch()
                stats['empty'] += 1
    
    print("\n=== Conversion Complete ===")
    print(f"Train labels created: {stats['train']}")
    print(f"Validation labels created: {stats['validation']}")
    print(f"Empty labels (background only): {stats['empty']}")

if __name__ == "__main__":
    DATA_ROOT = r"c:\Nhat Nam\do an chuyen nganh\visual-meal-analysis\data\FoodSeg103"
    MIN_AREA = 50  # Minimum contour area in pixels
    
    convert_dataset(DATA_ROOT, min_area=MIN_AREA)
