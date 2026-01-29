"""
Script to create a new dataset with merged and removed classes.

Modifications:
1. REMOVE classes:
   - Japanese tofu and vegetable chowder (ID 28)
   - pizza toast (ID 55)
   - pilaf (ID 8)
   - soba noodle (ID 14)

2. MERGE classes:
   - miso soup (22) + pork miso soup (26) -> miso soup
   - fried rice (4) + mixed rice (6) + chicken rice (3) -> fried rice
   - beef noodle (16) -> ramen noodle (15)
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

# Configuration
SRC_DATA = Path(r"c:\Nhat Nam\do an chuyen nganh\visual-meal-analysis\data\uecfoodpix_yolo_expanded_merged")
DST_DATA = Path(r"/data/uecfoodpix_yolo_expanded_merged")

# Classes to REMOVE (will delete images/labels containing ONLY these classes)
CLASSES_TO_REMOVE = {8, 14, 28, 55}  # pilaf, soba noodle, Japanese tofu..., pizza toast

# Class merging: source_id -> target_id
CLASS_MERGE_MAP = {
    26: 22,  # pork miso soup -> miso soup
    6: 4,    # mixed rice -> fried rice
    3: 4,    # chicken rice -> fried rice
    16: 15,  # beef noodle -> ramen noodle
}

# Original class names
ORIGINAL_NAMES = {
    0: "rice",
    1: "chicken-'n'-egg on rice",
    2: "pork cutlet on rice",
    3: "chicken rice",
    4: "fried rice",
    5: "beef bowl",
    6: "mixed rice",
    7: "eels on rice",
    8: "pilaf",
    9: "tempura bowl",
    10: "rice ball",
    11: "sashimi bowl",
    12: "sushi bowl",
    13: "udon noodle",
    14: "soba noodle",
    15: "ramen noodle",
    16: "beef noodle",
    17: "fried noodle",
    18: "spaghetti",
    19: "spaghetti meat sauce",
    20: "tempura udon",
    21: "dipping noodles",
    22: "miso soup",
    23: "oden",
    24: "stew",
    25: "chinese soup",
    26: "pork miso soup",
    27: "potage",
    28: "Japanese tofu and vegetable chowder",
    29: "fried chicken",
    30: "yakitori",
    31: "roast chicken",
    32: "hambarg steak",
    33: "beef steak",
    34: "sweet and sour pork",
    35: "stir-fried beef and peppers",
    36: "hamburger",
    37: "omelet",
    38: "cold tofu",
    39: "fried fish",
    40: "grilled salmon",
    41: "sashimi",
    42: "fried shrimp",
    43: "sauteed vegetables",
    44: "green salad",
    45: "potato salad",
    46: "sauteed spinach",
    47: "macaroni salad",
    48: "goya chanpuru",
    49: "vegetable tempura",
    50: "toast",
    51: "french fries",
    52: "croissant",
    53: "roll bread",
    54: "sandwiches",
    55: "pizza toast",
    56: "hot dog",
    57: "pizza",
}

def get_new_class_mapping():
    """
    Create new class ID mapping after removing and merging classes.
    Returns: 
        - old_to_new: dict mapping old class IDs to new class IDs
        - new_names: dict of new class ID to name
    """
    # First, apply merges to get effective IDs
    effective_ids = set()
    for old_id in ORIGINAL_NAMES.keys():
        if old_id in CLASSES_TO_REMOVE:
            continue
        if old_id in CLASS_MERGE_MAP:
            effective_ids.add(CLASS_MERGE_MAP[old_id])
        else:
            effective_ids.add(old_id)
    
    # Sort and create new sequential IDs
    sorted_ids = sorted(effective_ids)
    old_to_new = {}
    new_names = {}
    
    for new_id, old_id in enumerate(sorted_ids):
        old_to_new[old_id] = new_id
        new_names[new_id] = ORIGINAL_NAMES[old_id]
    
    # Add merge mappings
    for src_id, tgt_id in CLASS_MERGE_MAP.items():
        if tgt_id in old_to_new:
            old_to_new[src_id] = old_to_new[tgt_id]
    
    return old_to_new, new_names


def process_label_file(src_path, dst_path, old_to_new):
    """
    Process a label file: remap class IDs, remove lines with deleted classes.
    Returns True if file should be kept, False if it should be deleted.
    """
    new_lines = []
    
    with open(src_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            old_class_id = int(parts[0])
            
            # Skip if class is removed
            if old_class_id in CLASSES_TO_REMOVE:
                continue
            
            # Get new class ID
            if old_class_id in old_to_new:
                new_class_id = old_to_new[old_class_id]
            else:
                print(f"Warning: Unknown class ID {old_class_id} in {src_path}")
                continue
            
            # Rebuild line with new class ID
            parts[0] = str(new_class_id)
            new_lines.append(' '.join(parts))
    
    # If no valid lines remain, return False (delete file/image)
    if not new_lines:
        return False
    
    # Write new label file
    with open(dst_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
    
    return True


def main():
    print("Creating new merged dataset...")
    print(f"Source: {SRC_DATA}")
    print(f"Destination: {DST_DATA}")
    
    # Get new class mapping
    old_to_new, new_names = get_new_class_mapping()
    
    print(f"\nOriginal classes: {len(ORIGINAL_NAMES)}")
    print(f"New classes: {len(new_names)}")
    print(f"Classes removed: {len(CLASSES_TO_REMOVE)}")
    print(f"Classes merged: {len(CLASS_MERGE_MAP)}")
    
    # Create destination directory
    DST_DATA.mkdir(parents=True, exist_ok=True)
    
    stats = {"kept": 0, "removed": 0}
    
    for split in ['train', 'val']:
        src_images_dir = SRC_DATA / split / "images"
        src_labels_dir = SRC_DATA / split / "labels"
        dst_images_dir = DST_DATA / split / "images"
        dst_labels_dir = DST_DATA / split / "labels"
        
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)
        
        label_files = list(src_labels_dir.glob("*.txt"))
        
        for label_file in tqdm(label_files, desc=f"Processing {split}"):
            dst_label_path = dst_labels_dir / label_file.name
            
            # Process label file
            keep = process_label_file(label_file, dst_label_path, old_to_new)
            
            if keep:
                # Copy corresponding image
                img_stem = label_file.stem
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path = src_images_dir / (img_stem + ext)
                    if img_path.exists():
                        shutil.copy2(img_path, dst_images_dir / img_path.name)
                        break
                stats["kept"] += 1
            else:
                # Remove the label file we just created (it's empty)
                if dst_label_path.exists():
                    dst_label_path.unlink()
                stats["removed"] += 1
    
    # Create new data.yaml
    data_yaml = {
        "path": str(DST_DATA),
        "train": "train/images",
        "val": "val/images",
        "names": new_names
    }
    
    with open(DST_DATA / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\n=== Summary ===")
    print(f"Images kept: {stats['kept']}")
    print(f"Images removed: {stats['removed']}")
    print(f"New dataset saved to: {DST_DATA}")
    print(f"\nNew class list ({len(new_names)} classes):")
    for idx, name in new_names.items():
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
