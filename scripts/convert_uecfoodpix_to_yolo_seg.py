"""
Convert UECFOODPIXCOMPLETE to YOLOv8 Segmentation format (Phase 1: 35 Classes).

Dataset: UECFOODPIXCOMPLETE
Output: data/uecfoodpix_yolo

Rules:
1. Mask Encoding: Class ID = R channel. Fallback to G if R=0 and G>0.
2. Instance Separation: Use Connected Components on binary mask of each class.
3. Phase 1: Only convert 35 priority classes. Remap IDs to 0-34 for training.
4. Split: Use existing train9000.txt (train) and test1000.txt (val).
"""

import os
import cv2
import numpy as np
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROOT_DIR = Path(__file__).parent.parent
DATA_ROOT = ROOT_DIR / "data" / "UECFOODPIXCOMPLETE" / "data"
# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROOT_DIR = Path(__file__).parent.parent
DATA_ROOT = ROOT_DIR / "data" / "UECFOODPIXCOMPLETE" / "data"
OUTPUT_DIR = ROOT_DIR / "data" / "uecfoodpix_yolo_expanded"

# Original dataset paths
TRAIN_ROOT = DATA_ROOT / "UECFoodPIXCOMPLETE" / "train"
TEST_ROOT = DATA_ROOT / "UECFoodPIXCOMPLETE" / "test"
SPLIT_TRAIN_TXT = DATA_ROOT / "train9000.txt"
SPLIT_VAL_TXT = DATA_ROOT / "test1000.txt"
CATEGORY_TXT = DATA_ROOT / "category.txt"

# Phase 2: Expanded Classes (Original IDs)
PHASE1_IDS = [
    # Rice Group (Original + Confusers)
    1, 4, 5, 8, 9, 92, 99,       # Bases: Rice, Fried Rice, Beef Bowl, Mixed Rice
    2, 3, 10, 94, 76, 77,        # New: Eels, Pilaf, Tempura Bowl, Rice Ball, Sashimi/Sushi Bowl
    
    # Noodle Group
    20, 22, 23, 24, 26, 27, 84,  # Bases: Udon, Soba, Ramen, Pasta
    21, 96,                      # New: Tempura Udon, Dipping Noodles
    
    # Soup Group
    36, 39, 43, 91, 90,          # Bases: Miso, Potage (wait - 39 is sausage oops, 37 is potage), Stew
    37, 89,                      # New: Potage, Chowder (Fixing 39 sausage issue later? 39 is sausage)
    
    # Meats/Main
    55, 65, 80, 60, 61, 51, 73,  # Fried Chicken, Yakitori, Roast Chicken, Steak, etc.
    17, 40, 70,                  # New: Hamburger, Omelet, Cold Tofu
    
    # Fish/Seafood
    45, 46, 48, 85,              # Fried fish, Salmon, Sashimi, Shrimp
    
    # Sides/Salads
    31, 87, 86,                  # Bases: Veggies, Green Salad, Potato Salad
    34, 88, 100, 35,             # New: Spinach, Macaroni Salad, Youtube (Goya), Veggie Tempura
    
    # Bread/Others
    12, 98,                      # Bases: Toast, Fries
    13, 14, 19, 95, 97, 18       # New: Croissant, Roll, Sandwich, Pizza Toast, Hot Dog, Pizza
]

# Map Original ID -> New Index (0-34)
ID_MAPPING = {orig_id: idx for idx, orig_id in enumerate(PHASE1_IDS)}

MIN_AREA = 100  # Minimum pixel area to keep a polygon

# ==============================================================================
# UTILS
# ==============================================================================
def load_categories(cat_path):
    cats = {}
    with open(cat_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:] # Skip header
        for line in lines:
             parts = line.strip().split('\t')
             if len(parts) >= 2:
                 cats[int(parts[0])] = parts[1]
    return cats

def parse_mask(mask_path):
    """
    Read mask and return dict: {original_class_id: [contours_normalized]}
    """
    mask = cv2.imread(str(mask_path)) # BGR
    if mask is None:
        return {}

    # Fix: unpack correctly 2 values (h, w) from mask.shape[:2]
    h, w = mask.shape[:2]
    
    # Analyze classes present in mask
    # Encoding: Class = R (channel 2 in BGR). Fallback G (channel 1) if R=0 and G>0
    
    # Optimized numpy way to get class map
    # B, G, R channels
    B, G, R = mask[:,:,0], mask[:,:,1], mask[:,:,2]
    
    class_map = R.copy().astype(np.int32)
    
    # Fallback to G where R is 0
    fallback_mask = (R == 0) & (G > 0)
    class_map[fallback_mask] = G[fallback_mask]
    
    unique_classes = np.unique(class_map)
    
    mask_objects = {}
    
    for cls_id in unique_classes:
        if cls_id == 0: continue # Background
        if cls_id not in ID_MAPPING: continue # Skip classes not in Phase 1
        
        # Binary mask for this class
        binary_mask = (class_map == cls_id).astype(np.uint8) * 255
        
        # Find connected components to separate instances
        num_labels, labels_im = cv2.connectedComponents(binary_mask)
        
        contours_list = []
        
        for i in range(1, num_labels): # 0 is background
            component_mask = (labels_im == i).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_AREA:
                    continue
                
                # Simplify polygon
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) < 3: 
                    continue
                
                # Normalize coordinates (x, y) -> (0-1)
                normalized_poly = approx.flatten().astype(float)
                normalized_poly[0::2] /= w  # x
                normalized_poly[1::2] /= h  # y
                
                # Clip to [0, 1]
                normalized_poly = np.clip(normalized_poly, 0.0, 1.0)
                
                contours_list.append(normalized_poly)
        
        if contours_list:
            mask_objects[cls_id] = contours_list
            
    return mask_objects

def process_dataset():
    # Setup directories
    if OUTPUT_DIR.exists():
        try:
            shutil.rmtree(OUTPUT_DIR)
        except Exception as e:
            print(f"Warning cleaning dir: {e}")
            
    try:
        for split in ['train', 'val']:
            (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
        print(f"Output directory created: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error creating dirs: {e}")
        return
    
    # Load categories
    all_categories = load_categories(CATEGORY_TXT)
    phase1_names = [all_categories[uid] for uid in PHASE1_IDS]
    print(f"Loaded {len(phase1_names)} classes (Phase 2).")

    # Load file lists
    with open(SPLIT_TRAIN_TXT, 'r') as f:
        train_ids = [line.strip() for line in f.readlines() if line.strip()]
        
    with open(SPLIT_VAL_TXT, 'r') as f:
        val_ids = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Found {len(train_ids)} training images and {len(val_ids)} validation images.")
    
    # Process
    stats = {'train': 0, 'val': 0, 'skipped': 0}
    
    # Define source dirs for each split
    split_configs = [
        ('train', train_ids, TRAIN_ROOT),
        ('val', val_ids, TEST_ROOT)
    ]
    
    for split_name, id_list, source_root in split_configs:
        print(f"Processing {split_name} set from {source_root}...")
        
        src_img_dir = source_root / "img"
        src_mask_dir = source_root / "mask"
        
        # Limit for dry run testing if needed, but run full for real
        for img_id in tqdm(id_list):
            img_filename = f"{img_id}.jpg"
            mask_filename = f"{img_id}.png"
            
            src_img_path = src_img_dir / img_filename
            src_mask_path = src_mask_dir / mask_filename
            
            if not src_img_path.exists() or not src_mask_path.exists():
                # Debug missing files
                # print(f"Missing: {src_img_path} or {src_mask_path}") 
                continue
                
            # Parse Mask
            objects = parse_mask(src_mask_path)
            
            # Use 'skipped' only if image exists but no relevant objects
            if not objects:
                stats['skipped'] += 1
                continue
                
            # Copy Image
            dst_img_path = OUTPUT_DIR / split_name / 'images' / img_filename
            shutil.copy2(src_img_path, dst_img_path)
            
            # Write Label
            dst_label_path = OUTPUT_DIR / split_name / 'labels' / f"{img_id}.txt"
            with open(dst_label_path, 'w') as f:
                for cls_id, contours in objects.items():
                    new_idx = ID_MAPPING[cls_id]
                    for poly in contours:
                        # Format: class_index x1 y1 x2 y2 ...
                        poly_str = " ".join([f"{coord:.6f}" for coord in poly])
                        f.write(f"{new_idx} {poly_str}\n")
            
            stats[split_name] += 1

    # Create data.yaml
    yaml_content = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(phase1_names)}
    }
    
    with open(OUTPUT_DIR / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"\nProcessing Complete!")
    print(f"Train images: {stats['train']}")
    print(f"Val images: {stats['val']}")
    print(f"Skipped (no Phase 1 objects): {stats['skipped']}")
    print(f"Config saved to {OUTPUT_DIR / 'data.yaml'}")

if __name__ == '__main__':
    process_dataset()
