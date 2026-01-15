# -*- coding: utf-8 -*-
"""
explore_dataset.py - Duyệt và kiểm tra tình trạng gán nhãn của NutritionVerse dataset

Author: Nam
Description:
    Script để khám phá dataset, kiểm tra annotations và visualize samples.
    
Usage:
    python explore_dataset.py
    python explore_dataset.py --visualize 10
    python explore_dataset.py --check-issues
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Fix encoding cho Windows console
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Paths
DATA_DIR = Path("data")
NUTRITIONVERSE_DIR = DATA_DIR / "nutritionverse-manual" / "nutritionverse-manual"
IMAGES_DIR = NUTRITIONVERSE_DIR / "images"
ANNOTATIONS_FILE = IMAGES_DIR / "_annotations.coco.json"
SPLITS_FILE = NUTRITIONVERSE_DIR / "updated-manual-dataset-splits.csv"
METADATA_FILE = DATA_DIR / "nutritionverse_dish_metadata3.csv"


def load_coco_annotations(annotation_path: Path) -> dict:
    """Load COCO format annotations."""
    print(f"\n[*] Loading annotations tu: {annotation_path}")
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def print_dataset_overview(coco_data: dict):
    """In tong quan ve dataset."""
    print("\n" + "=" * 70)
    print("[STATS] TONG QUAN DATASET")
    print("=" * 70)
    
    num_images = len(coco_data.get("images", []))
    num_annotations = len(coco_data.get("annotations", []))
    num_categories = len(coco_data.get("categories", []))
    
    print(f"  • Số lượng ảnh:        {num_images}")
    print(f"  • Số lượng annotations: {num_annotations}")
    print(f"  • Số lượng categories:  {num_categories}")
    print(f"  • Trung bình ann/ảnh:   {num_annotations/num_images:.2f}")


def analyze_categories(coco_data: dict):
    """Phan tich cac categories."""
    print("\n" + "=" * 70)
    print("[CATEGORIES] DANH SACH CATEGORIES")
    print("=" * 70)
    
    categories = coco_data.get("categories", [])
    
    # Đếm số annotations cho mỗi category
    ann_per_cat = Counter()
    for ann in coco_data.get("annotations", []):
        ann_per_cat[ann["category_id"]] += 1
    
    # Tạo mapping id -> name
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    
    print(f"\n{'ID':<5} {'Tên Category':<40} {'Số Annotations':<15} {'%':<10}")
    print("-" * 70)
    
    total_ann = sum(ann_per_cat.values())
    sorted_cats = sorted(categories, key=lambda x: ann_per_cat.get(x["id"], 0), reverse=True)
    
    for cat in sorted_cats:
        cat_id = cat["id"]
        cat_name = cat["name"]
        count = ann_per_cat.get(cat_id, 0)
        pct = (count / total_ann * 100) if total_ann > 0 else 0
        print(f"{cat_id:<5} {cat_name:<40} {count:<15} {pct:.1f}%")
    
    return cat_id_to_name, ann_per_cat


def analyze_images(coco_data: dict):
    """Phan tich cac images."""
    print("\n" + "=" * 70)
    print("[IMAGES] PHAN TICH IMAGES")
    print("=" * 70)
    
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    
    # Đếm annotations per image
    ann_per_image = Counter()
    for ann in annotations:
        ann_per_image[ann["image_id"]] += 1
    
    # Thống kê
    ann_counts = list(ann_per_image.values())
    images_with_ann = len(ann_per_image)
    images_without_ann = len(images) - images_with_ann
    
    print(f"\n  - Anh co annotation:     {images_with_ann}")
    print(f"  - Anh KHONG co annotation: {images_without_ann}")
    
    if ann_counts:
        print(f"\n  [DIST] Phan phoi so annotations/anh:")
        print(f"     - Min:    {min(ann_counts)}")
        print(f"     - Max:    {max(ann_counts)}")
        print(f"     - Mean:   {np.mean(ann_counts):.2f}")
        print(f"     - Median: {np.median(ann_counts):.1f}")
    
    # Histogram
    print(f"\n  [HIST] Histogram (so annotations/anh):")
    hist = Counter(ann_counts)
    for num_ann in sorted(hist.keys()):
        bar = "#" * min(hist[num_ann], 50)
        print(f"     {num_ann} ann: {bar} ({hist[num_ann]} anh)")
    
    # Liet ke anh khong co annotation
    if images_without_ann > 0:
        print(f"\n  [WARN] Anh KHONG co annotation ({images_without_ann}):")
        image_ids_with_ann = set(ann_per_image.keys())
        for img in images[:20]:  # Chi hien thi 20 anh dau
            if img["id"] not in image_ids_with_ann:
                print(f"     - {img['file_name']}")
        if images_without_ann > 20:
            print(f"     ... va {images_without_ann - 20} anh khac")
    
    return ann_per_image


def analyze_annotation_quality(coco_data: dict):
    """Kiem tra chat luong annotations."""
    print("\n" + "=" * 70)
    print("[QUALITY] KIEM TRA CHAT LUONG ANNOTATIONS")
    print("=" * 70)
    
    annotations = coco_data.get("annotations", [])
    issues = []
    
    for ann in annotations:
        ann_id = ann["id"]
        
        # Kiểm tra bbox
        if "bbox" in ann:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                issues.append(f"Annotation {ann_id}: bbox có width/height <= 0")
            if w < 10 or h < 10:
                issues.append(f"Annotation {ann_id}: bbox quá nhỏ ({w}x{h})")
        
        # Kiểm tra segmentation
        if "segmentation" in ann and ann["segmentation"]:
            for seg in ann["segmentation"]:
                if len(seg) < 6:  # Ít nhất 3 điểm (6 tọa độ)
                    issues.append(f"Annotation {ann_id}: segmentation có ít hơn 3 điểm")
        
        # Kiểm tra area
        if "area" in ann and ann["area"] <= 0:
            issues.append(f"Annotation {ann_id}: area <= 0")
    
    if issues:
        print(f"\n  [WARN] Phat hien {len(issues)} van de:")
        for issue in issues[:20]:
            print(f"     - {issue}")
        if len(issues) > 20:
            print(f"     ... va {len(issues) - 20} van de khac")
    else:
        print("\n  [OK] Khong phat hien van de voi annotations!")
    
    return issues


def analyze_splits(splits_file: Path):
    """Phan tich train/val splits."""
    if not splits_file.exists():
        print(f"\n  [WARN] File splits khong ton tai: {splits_file}")
        return None
    
    print("\n" + "=" * 70)
    print("[SPLITS] PHAN TICH TRAIN/VAL SPLITS")
    print("=" * 70)
    
    df = pd.read_csv(splits_file)
    
    split_counts = df["category"].value_counts()
    print(f"\n  • Total files: {len(df)}")
    for split, count in split_counts.items():
        pct = count / len(df) * 100
        print(f"  • {split}: {count} ({pct:.1f}%)")
    
    return df


def visualize_samples(coco_data: dict, num_samples: int = 5, output_dir: Path = None):
    """Visualize mot so samples voi annotations."""
    print("\n" + "=" * 70)
    print(f"[VIS] VISUALIZE {num_samples} SAMPLES")
    print("=" * 70)
    
    if output_dir is None:
        output_dir = Path("outputs") / "dataset_exploration"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    
    # Group annotations by image
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann["image_id"]].append(ann)
    
    # Chọn random samples
    sampled_images = random.sample(images, min(num_samples, len(images)))
    
    # Colors for visualization
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (255, 128, 0), (128, 0, 255), (0, 128, 255),
    ]
    
    for img_info in sampled_images:
        img_path = IMAGES_DIR / img_info["file_name"]
        
        if not img_path.exists():
            print(f"  [WARN] Khong tim thay: {img_path}")
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_anns = ann_by_image.get(img_info["id"], [])
        
        print(f"\n  [IMG] {img_info['file_name']}")
        print(f"     Kich thuoc: {img_info['width']}x{img_info['height']}")
        print(f"     So annotations: {len(img_anns)}")
        
        # Vẽ annotations
        for i, ann in enumerate(img_anns):
            color = colors[i % len(colors)]
            cat_name = categories.get(ann["category_id"], "unknown")
            
            print(f"     - [{ann['category_id']}] {cat_name}")
            
            # Vẽ segmentation polygon
            if "segmentation" in ann and ann["segmentation"]:
                for seg in ann["segmentation"]:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
                    # Fill với transparency
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
            
            # Vẽ bbox
            if "bbox" in ann:
                x, y, w, h = [int(v) for v in ann["bbox"]]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Label
                label = f"{cat_name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x, y-th-4), (x+tw, y), color, -1)
                cv2.putText(img, label, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Luu anh
        output_path = output_dir / f"sample_{img_info['file_name']}"
        cv2.imwrite(str(output_path), img)
        print(f"     [OK] Saved: {output_path}")


def generate_report(coco_data: dict, splits_df: pd.DataFrame = None):
    """Tao bao cao tong hop."""
    print("\n" + "=" * 70)
    print("[REPORT] BAO CAO TONG HOP")
    print("=" * 70)
    
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])
    
    # Đếm annotations per category
    ann_per_cat = Counter()
    for ann in annotations:
        ann_per_cat[ann["category_id"]] += 1
    
    # Tìm category ít/nhiều nhất
    if ann_per_cat:
        most_common = ann_per_cat.most_common(3)
        least_common = ann_per_cat.most_common()[-3:]
        
        cat_names = {cat["id"]: cat["name"] for cat in categories}
        
        print("\n  [TOP] Categories pho bien nhat:")
        for cat_id, count in most_common:
            print(f"     - {cat_names.get(cat_id, cat_id)}: {count} annotations")
        
        print("\n  [LOW] Categories it nhat:")
        for cat_id, count in reversed(least_common):
            print(f"     - {cat_names.get(cat_id, cat_id)}: {count} annotations")
    
    # De xuat
    print("\n  [SUGGEST] DE XUAT:")
    
    # Class imbalance
    if ann_per_cat:
        max_count = max(ann_per_cat.values())
        min_count = min(ann_per_cat.values())
        if max_count > 10 * min_count:
            print("     [WARN] Dataset co CLASS IMBALANCE nghiem trong!")
            print("        -> Can nhac su dung class weights hoac oversampling")
    
    # Kiem tra anh khong co annotation
    ann_per_image = Counter(ann["image_id"] for ann in annotations)
    images_without_ann = len(images) - len(ann_per_image)
    if images_without_ann > 0:
        print(f"     [WARN] Co {images_without_ann} anh KHONG co annotation!")
        print("        -> Kiem tra va gan nhan hoac loai bo khoi dataset")
    
    print("\n     [OK] Dataset san sang cho training!")


def main():
    parser = argparse.ArgumentParser(description="Explore NutritionVerse Dataset")
    parser.add_argument("--visualize", "-v", type=int, default=0,
                        help="Số lượng samples để visualize (default: 0)")
    parser.add_argument("--check-issues", "-c", action="store_true",
                        help="Kiểm tra chi tiết các vấn đề")
    args = parser.parse_args()
    
    print("=" * 70)
    print("NUTRITIONVERSE DATASET EXPLORER")
    print("=" * 70)
    
    # Load data
    coco_data = load_coco_annotations(ANNOTATIONS_FILE)
    
    # Tổng quan
    print_dataset_overview(coco_data)
    
    # Phân tích categories
    cat_names, ann_per_cat = analyze_categories(coco_data)
    
    # Phân tích images
    ann_per_image = analyze_images(coco_data)
    
    # Kiểm tra chất lượng
    if args.check_issues:
        analyze_annotation_quality(coco_data)
    
    # Phân tích splits
    splits_df = analyze_splits(SPLITS_FILE)
    
    # Visualize
    if args.visualize > 0:
        visualize_samples(coco_data, num_samples=args.visualize)
    
    # Báo cáo
    generate_report(coco_data, splits_df)
    
    print("\n" + "=" * 70)
    print("[DONE] HOAN THANH!")
    print("=" * 70)


if __name__ == "__main__":
    main()
