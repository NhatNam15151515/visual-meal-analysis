"""
preprocess_data.py - Tiền Xử Lý Dữ Liệu UECFOODPIX

Thực hiện:
- Phase 1: Kiểm tra trùng lặp, file lỗi
- Phase 2: Augmentation offline cho class thiểu số (< 150 ảnh)

Theo Data Strategy Plan.
"""

import os
import hashlib
import shutil
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import random

# ===================== CONFIGURATION =====================
BASE_DIR = Path(r"c:\Nhat Nam\do an chuyen nganh\visual-meal-analysis\data\uecfoodpix_yolo_expanded")
TRAIN_IMAGES = BASE_DIR / "train" / "images"
TRAIN_LABELS = BASE_DIR / "train" / "labels"
DATA_YAML = BASE_DIR / "data.yaml"

# Soft target: Tăng class < 100 lên khoảng 150-200
MIN_SAMPLES_TARGET = 150
MAX_AUG_RATIO = 2  # Tối đa nhân đôi số ảnh gốc

# ===================== PHASE 1: MANDATORY =====================

def get_file_hash(filepath):
    """Tính MD5 hash của file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def check_duplicates(images_dir):
    """Kiểm tra ảnh trùng lặp dựa trên hash."""
    print("\n[PHASE 1.1] Kiểm tra ảnh trùng lặp...")
    hash_map = {}
    duplicates = []
    
    image_files = list(images_dir.glob("*.*"))
    for img_path in tqdm(image_files, desc="Hashing images"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        file_hash = get_file_hash(img_path)
        if file_hash in hash_map:
            duplicates.append((img_path, hash_map[file_hash]))
        else:
            hash_map[file_hash] = img_path
    
    if duplicates:
        print(f"  [WARNING] Tìm thấy {len(duplicates)} cặp ảnh trùng lặp!")
        for dup, orig in duplicates[:5]:
            print(f"    - {dup.name} trùng với {orig.name}")
        if len(duplicates) > 5:
            print(f"    ... và {len(duplicates) - 5} cặp khác.")
    else:
        print("  [OK] Không có ảnh trùng lặp.")
    
    return duplicates


def check_corrupt_files(images_dir, labels_dir):
    """Kiểm tra file ảnh và label bị lỗi."""
    print("\n[PHASE 1.2] Kiểm tra file lỗi...")
    corrupt_images = []
    corrupt_labels = []
    missing_labels = []
    
    image_files = list(images_dir.glob("*.*"))
    for img_path in tqdm(image_files, desc="Checking files"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        
        # Check image
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                corrupt_images.append(img_path)
                continue
        except Exception:
            corrupt_images.append(img_path)
            continue
        
        # Check label
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_labels.append(img_path)
            continue
        
        try:
            with open(label_path, 'r') as f:
                content = f.read()
                if not content.strip():
                    corrupt_labels.append(label_path)
        except Exception:
            corrupt_labels.append(label_path)
    
    print(f"  [INFO] Ảnh lỗi: {len(corrupt_images)}, Label lỗi: {len(corrupt_labels)}, Label thiếu: {len(missing_labels)}")
    
    if corrupt_images:
        print("  [WARNING] Ảnh lỗi:")
        for p in corrupt_images[:5]:
            print(f"    - {p.name}")
    
    return corrupt_images, corrupt_labels, missing_labels


def get_class_distribution(labels_dir, class_names):
    """Đếm số lượng ảnh cho mỗi class."""
    print("\n[PHASE 1.3] Phân tích phân bố class...")
    class_counts = Counter()
    class_to_images = {i: [] for i in class_names.keys()}
    
    label_files = list(labels_dir.glob("*.txt"))
    for label_path in tqdm(label_files, desc="Counting classes"):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            classes_in_file = set()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    class_id = int(line.split()[0])
                    classes_in_file.add(class_id)
                except (ValueError, IndexError):
                    pass
            
            for cls_id in classes_in_file:
                class_counts[cls_id] += 1
                if cls_id in class_to_images:
                    class_to_images[cls_id].append(label_path.stem)
    
    return class_counts, class_to_images


# ===================== PHASE 2: AUGMENTATION =====================

def apply_augmentation(img, mode):
    """Áp dụng một trong các augmentation an toàn."""
    h, w = img.shape[:2]
    
    if mode == 'hflip':
        return cv2.flip(img, 1)
    
    elif mode == 'rotate':
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif mode == 'hsv':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + random.uniform(-10, 10)) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.8, 1.2), 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(0.8, 1.2), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    elif mode == 'brightness':
        factor = random.uniform(0.8, 1.2)
        return np.clip(img * factor, 0, 255).astype(np.uint8)
    
    return img


def transform_polygon_hflip(polygon_str, img_width=1.0):
    """Lật ngang polygon (YOLO format đã normalize 0-1)."""
    parts = polygon_str.strip().split()
    class_id = parts[0]
    coords = list(map(float, parts[1:]))
    
    new_coords = []
    for i in range(0, len(coords), 2):
        x, y = coords[i], coords[i+1]
        new_coords.extend([1.0 - x, y])
    
    return class_id + " " + " ".join(f"{c:.6f}" for c in new_coords)


def transform_polygon_rotate(polygon_str, angle_deg, center=(0.5, 0.5)):
    """Xoay polygon quanh tâm (YOLO format đã normalize)."""
    parts = polygon_str.strip().split()
    class_id = parts[0]
    coords = list(map(float, parts[1:]))
    
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cx, cy = center
    
    new_coords = []
    for i in range(0, len(coords), 2):
        x, y = coords[i], coords[i+1]
        # Translate to origin
        x -= cx
        y -= cy
        # Rotate
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        # Translate back and clamp
        x_new = np.clip(x_new + cx, 0, 1)
        y_new = np.clip(y_new + cy, 0, 1)
        new_coords.extend([x_new, y_new])
    
    return class_id + " " + " ".join(f"{c:.6f}" for c in new_coords)


def augment_minority_classes(images_dir, labels_dir, class_counts, class_to_images, class_names):
    """Augment offline cho các class thiểu số."""
    print("\n[PHASE 2] Augmentation cho class thiểu số...")
    
    # Tìm class cần augment
    minority_classes = {cls_id: count for cls_id, count in class_counts.items() 
                       if count < MIN_SAMPLES_TARGET}
    
    if not minority_classes:
        print("  [OK] Không có class nào dưới ngưỡng 150. Bỏ qua augmentation.")
        return
    
    print(f"  [INFO] {len(minority_classes)} class cần augment (< {MIN_SAMPLES_TARGET} ảnh):")
    for cls_id, count in sorted(minority_classes.items(), key=lambda x: x[1])[:10]:
        name = class_names.get(cls_id, f"class_{cls_id}")
        target = min(MIN_SAMPLES_TARGET, count * MAX_AUG_RATIO)
        print(f"    - [{cls_id}] {name}: {count} → ~{target}")
    
    aug_modes = ['hflip', 'rotate', 'hsv', 'brightness']
    total_generated = 0
    
    for cls_id, current_count in tqdm(minority_classes.items(), desc="Augmenting classes"):
        target_count = min(MIN_SAMPLES_TARGET, current_count * MAX_AUG_RATIO)
        needed = target_count - current_count
        
        if needed <= 0:
            continue
        
        # Lấy danh sách ảnh chứa class này
        image_stems = class_to_images.get(cls_id, [])
        if not image_stems:
            continue
        
        generated = 0
        attempts = 0
        max_attempts = needed * 3
        
        while generated < needed and attempts < max_attempts:
            attempts += 1
            
            # Random chọn 1 ảnh gốc
            stem = random.choice(image_stems)
            
            # Tìm file ảnh (có thể là .jpg, .png, etc.)
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = images_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            
            if img_path is None:
                continue
            
            label_path = labels_dir / (stem + ".txt")
            if not label_path.exists():
                continue
            
            # Đọc ảnh và label
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            # Chọn augmentation
            mode = random.choice(aug_modes)
            
            # Áp dụng augmentation
            aug_img = apply_augmentation(img, mode)
            
            # Transform labels
            new_labels = []
            if mode == 'hflip':
                for line in labels:
                    if line.strip():
                        new_labels.append(transform_polygon_hflip(line))
            elif mode == 'rotate':
                angle = random.uniform(-15, 15)
                for line in labels:
                    if line.strip():
                        new_labels.append(transform_polygon_rotate(line, angle))
            else:
                # HSV và brightness không thay đổi polygon
                new_labels = [l.strip() for l in labels if l.strip()]
            
            # Tạo tên file mới
            aug_suffix = f"_aug_{mode}_{generated}"
            new_stem = stem + aug_suffix
            new_img_path = images_dir / (new_stem + img_path.suffix)
            new_label_path = labels_dir / (new_stem + ".txt")
            
            # Kiểm tra đã tồn tại chưa
            if new_img_path.exists():
                continue
            
            # Lưu
            cv2.imwrite(str(new_img_path), aug_img)
            with open(new_label_path, 'w') as f:
                f.write("\n".join(new_labels))
            
            generated += 1
            total_generated += 1
    
    print(f"  [DONE] Đã tạo {total_generated} ảnh augmented.")


# ===================== MAIN =====================

def main():
    print("=" * 60)
    print("PREPROCESSING DATA: UECFOODPIX YOLO EXPANDED")
    print("=" * 60)
    
    # Load class names
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data.get('names', {})
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    class_names = {int(k): v for k, v in class_names.items()}
    
    print(f"[INFO] Dataset: {BASE_DIR}")
    print(f"[INFO] Classes: {len(class_names)}")
    
    # Phase 1
    duplicates = check_duplicates(TRAIN_IMAGES)
    corrupt_imgs, corrupt_labels, missing_labels = check_corrupt_files(TRAIN_IMAGES, TRAIN_LABELS)
    
    # Get distribution before augmentation
    class_counts, class_to_images = get_class_distribution(TRAIN_LABELS, class_names)
    
    print("\n[INFO] Phân bố class hiện tại (Top 5 ít nhất):")
    for cls_id, count in sorted(class_counts.items(), key=lambda x: x[1])[:5]:
        print(f"  - [{cls_id}] {class_names.get(cls_id, '?')}: {count}")
    
    # Phase 2
    augment_minority_classes(TRAIN_IMAGES, TRAIN_LABELS, class_counts, class_to_images, class_names)
    
    # Final distribution
    print("\n[FINAL] Phân bố class sau augmentation:")
    final_counts, _ = get_class_distribution(TRAIN_LABELS, class_names)
    for cls_id, count in sorted(final_counts.items(), key=lambda x: x[1])[:10]:
        print(f"  - [{cls_id}] {class_names.get(cls_id, '?')}: {count}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
