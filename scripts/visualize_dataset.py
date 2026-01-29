"""
Visualize UECFOOD256_CROP dataset with bounding boxes.
Shows random samples from each class with bbox drawn.
"""

import random
from pathlib import Path
import cv2
import numpy as np


def load_bb_info(bb_file: Path) -> dict:
    """Load bb_info.txt -> dict of img_name -> (x1, y1, x2, y2)"""
    bboxes = {}
    with open(bb_file, "r") as f:
        for line in f.readlines()[1:]:  # Skip header
            parts = line.strip().split()
            if len(parts) >= 5:
                img_name = parts[0]
                x1, y1, x2, y2 = map(int, parts[1:5])
                bboxes[img_name] = (x1, y1, x2, y2)
    return bboxes


def load_categories(cat_file: Path) -> dict:
    """Load category.txt -> dict of id -> name"""
    cats = {}
    with open(cat_file, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cats[int(parts[0])] = parts[1]
    return cats


def draw_bbox(img, bbox, class_name, color=(0, 255, 0)):
    """Draw bounding box and class name on image"""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    label = class_name
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w + 5, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img


def create_grid(images, grid_size=(4, 4), cell_size=(300, 300)):
    """Create a grid of images"""
    rows, cols = grid_size
    cell_w, cell_h = cell_size
    
    grid = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255
    
    for i, img in enumerate(images[:rows * cols]):
        row = i // cols
        col = i % cols
        
        # Resize image to fit cell
        h, w = img.shape[:2]
        scale = min(cell_w / w, cell_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # Center in cell
        x_offset = col * cell_w + (cell_w - new_w) // 2
        y_offset = row * cell_h + (cell_h - new_h) // 2
        
        grid[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return grid


def main():
    base_dir = Path("data/UECFOOD256_CROP")
    output_dir = Path("data/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load categories
    cat_file = base_dir / "category.txt"
    categories = load_categories(cat_file)
    print(f"Loaded {len(categories)} categories.")
    
    # Collect samples from each class
    all_samples = []
    
    for class_id, class_name in categories.items():
        class_dir = base_dir / str(class_id)
        bb_file = class_dir / "bb_info.txt"
        
        if not bb_file.exists():
            continue
        
        bboxes = load_bb_info(bb_file)
        images = list(class_dir.glob("*.jpg"))
        
        if not images:
            continue
        
        # Pick random sample
        sample_img = random.choice(images)
        img_name = sample_img.stem
        
        if img_name not in bboxes:
            continue
        
        img = cv2.imread(str(sample_img))
        if img is None:
            continue
        
        bbox = bboxes[img_name]
        img_with_bbox = draw_bbox(img.copy(), bbox, class_name)
        
        all_samples.append((class_id, class_name, img_with_bbox))
    
    print(f"Collected {len(all_samples)} samples.")
    
    # Create visualization grids (16 images per grid)
    random.shuffle(all_samples)
    
    grid_size = 16
    num_grids = (len(all_samples) + grid_size - 1) // grid_size
    
    for i in range(min(num_grids, 5)):  # Max 5 grids
        start = i * grid_size
        end = min(start + grid_size, len(all_samples))
        
        grid_samples = [s[2] for s in all_samples[start:end]]
        grid = create_grid(grid_samples, grid_size=(4, 4), cell_size=(350, 280))
        
        output_path = output_dir / f"uecfood256_crop_grid_{i+1}.jpg"
        cv2.imwrite(str(output_path), grid)
        print(f"Saved: {output_path}")
    
    # Also save individual samples from first 10 classes
    print("\nSaving individual samples...")
    for class_id, class_name, img in all_samples[:10]:
        output_path = output_dir / f"sample_{class_id}_{class_name}.jpg"
        cv2.imwrite(str(output_path), img)
        print(f"  {output_path}")
    
    print(f"\nVisualization complete! Check: {output_dir}")


if __name__ == "__main__":
    main()
