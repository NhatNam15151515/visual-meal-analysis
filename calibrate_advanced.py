# -*- coding: utf-8 -*-
"""
calibrate_advanced.py - Calibration nang cao cho cac thong so vat ly

Author: Nam
Description:
    Calibrate cac thong so:
    - focal_length_px: anh huong den area_cm2
    - reference_distance_cm: anh huong den pixel_to_cm  
    - depth_scale_factor: anh huong den height_cm
    
    Su dung optimization de tim bo thong so tot nhat.

Usage:
    python calibrate_advanced.py --samples 50
    python calibrate_advanced.py --samples 100 --update-config
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Fix encoding cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "nutritionverse-manual" / "nutritionverse-manual" / "images"
METADATA_FILE = DATA_DIR / "nutritionverse_dish_metadata3.csv"
CONFIG_FILE = PROJECT_ROOT / "src" / "config.py"


def load_ground_truth(metadata_file: Path) -> pd.DataFrame:
    """Load ground truth weights."""
    return pd.read_csv(metadata_file)


def extract_dish_id(filename: str) -> int:
    """Extract dish_id tu ten file (voi offset mapping)."""
    match = re.search(r'dish_(\d+)_', filename)
    if match:
        return int(match.group(1)) - 99  # Mapping: image_id - 99 = metadata_id
    return -1


def get_ground_truth_for_dish(df: pd.DataFrame, dish_id: int) -> dict:
    """Lay ground truth weight cho mot dish."""
    row = df[df['dish_id'] == dish_id]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        'dish_id': dish_id,
        'total_weight': row['total_food_weight'],
    }


def compute_volume_with_params(
    area_pixels: int,
    mean_depth: float,
    min_depth: float,
    max_depth: float,
    focal_length_px: float,
    reference_distance_cm: float,
    depth_scale_factor: float,
) -> Tuple[float, float, float]:
    """
    Tinh volume voi cac tham so cho truoc.
    
    Returns:
        (area_cm2, height_cm, volume_cm3)
    """
    # Pixel to cm
    pixel_to_cm = reference_distance_cm / focal_length_px
    area_cm2 = area_pixels * (pixel_to_cm ** 2)
    
    # Height estimation
    if min_depth > 0.01 and max_depth > min_depth:
        height_cm = depth_scale_factor * (1.0 / min_depth - 1.0 / max_depth)
        height_cm = max(0.5, min(height_cm, 20.0))
    else:
        height_cm = 2.5
    
    # Volume
    volume_cm3 = area_cm2 * height_cm
    
    return area_cm2, height_cm, volume_cm3


def collect_samples(num_samples: int = 50) -> List[Dict]:
    """
    Thu thap samples: chay segmentation va depth estimation,
    luu lai cac gia tri trung gian de calibration.
    """
    print("[1] Loading modules...")
    
    # Import here to avoid circular imports
    from src.tier1_segmentation import Tier1FoodSegmentation
    from src.tier2_depth_volume import Tier2DepthVolume
    from src.config import TIER1_CONFIG, TIER2_CONFIG
    
    segmenter = Tier1FoodSegmentation()
    depth_estimator = Tier2DepthVolume()
    
    # Load ground truth
    metadata_df = load_ground_truth(METADATA_FILE)
    
    # Collect valid images
    all_images = list(IMAGES_DIR.glob("*.jpg"))
    valid_images = []
    
    for img_path in all_images:
        dish_id = extract_dish_id(img_path.name)
        if dish_id > 0:
            gt = get_ground_truth_for_dish(metadata_df, dish_id)
            if gt and gt['total_weight'] > 0:
                valid_images.append((img_path, dish_id, gt))
    
    print(f"[2] Found {len(valid_images)} images with ground truth")
    
    # Random sample
    import random
    if len(valid_images) > num_samples:
        valid_images = random.sample(valid_images, num_samples)
    
    # Collect data
    samples = []
    
    print(f"[3] Collecting {len(valid_images)} samples...")
    
    for i, (img_path, dish_id, gt) in enumerate(valid_images):
        if (i + 1) % 10 == 0:
            print(f"    Processing {i+1}/{len(valid_images)}...")
        
        try:
            # Segmentation
            seg_results = segmenter.predict(str(img_path))
            if not seg_results:
                continue
            
            # Depth
            import cv2
            img = cv2.imread(str(img_path))
            depth_map = depth_estimator.estimate_depth(img)
            
            # Collect intermediate values for each detected item
            items = []
            for seg in seg_results:
                mask = seg.mask
                mask_bool = mask > 0
                
                if mask_bool.sum() == 0:
                    continue
                
                depth_values = depth_map[mask_bool]
                
                items.append({
                    'class_name': seg.class_name,
                    'area_pixels': int(mask_bool.sum()),
                    'mean_depth': float(np.mean(depth_values)),
                    'min_depth': float(np.min(depth_values)),
                    'max_depth': float(np.max(depth_values)),
                })
            
            if items:
                samples.append({
                    'dish_id': dish_id,
                    'image': img_path.name,
                    'gt_total_weight': gt['total_weight'],
                    'items': items,
                })
                
        except Exception as e:
            continue
    
    print(f"[4] Collected {len(samples)} valid samples")
    return samples


def load_density_map() -> Dict[str, float]:
    """Load density map tu file CSV."""
    density_file = PROJECT_ROOT / "src" / "food_density.csv"
    if not density_file.exists():
        return {}
    
    df = pd.read_csv(density_file)
    return {row['food_type'].lower(): row['density_g_per_cm3'] 
            for _, row in df.iterrows()}


def get_density(class_name: str, density_map: Dict[str, float], default: float = 0.9) -> float:
    """Lay density cho mot class."""
    class_lower = class_name.lower()
    if class_lower in density_map:
        return density_map[class_lower]
    # Partial match
    for key, val in density_map.items():
        if key in class_lower or class_lower in key:
            return val
    return default


def objective_function(
    params: np.ndarray,
    samples: List[Dict],
    density_map: Dict[str, float],
) -> float:
    """
    Ham muc tieu: toi thieu sai so giua predicted va ground truth.
    
    params: [focal_length_px, reference_distance_cm, depth_scale_factor]
    """
    focal = params[0]
    ref_dist = params[1]
    depth_scale = params[2]
    
    errors = []
    
    for sample in samples:
        gt_weight = sample['gt_total_weight']
        
        # Predict total weight
        pred_weight = 0
        for item in sample['items']:
            _, _, volume = compute_volume_with_params(
                item['area_pixels'],
                item['mean_depth'],
                item['min_depth'],
                item['max_depth'],
                focal,
                ref_dist,
                depth_scale,
            )
            density = get_density(item['class_name'], density_map)
            pred_weight += volume * density
        
        if pred_weight > 0 and gt_weight > 0:
            # Relative error
            error = abs(pred_weight - gt_weight) / gt_weight
            errors.append(error)
    
    if not errors:
        return 1e10
    
    # Return mean error
    return np.mean(errors)


def optimize_parameters(samples: List[Dict]) -> Dict:
    """
    Optimize cac tham so bang scipy.minimize.
    """
    density_map = load_density_map()
    
    print("\n[5] Optimizing parameters...")
    
    # Initial guess
    x0 = np.array([1000.0, 40.0, 1.0])  # [focal, ref_dist, depth_scale]
    
    # Bounds
    bounds = [
        (500, 3000),    # focal_length_px
        (20, 80),       # reference_distance_cm
        (0.1, 10.0),    # depth_scale_factor
    ]
    
    # Optimize
    result = minimize(
        objective_function,
        x0,
        args=(samples, density_map),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': True}
    )
    
    optimized = {
        'focal_length_px': result.x[0],
        'reference_distance_cm': result.x[1],
        'depth_scale_factor': result.x[2],
        'final_error': result.fun,
        'success': result.success,
    }
    
    return optimized


def evaluate_params(
    samples: List[Dict],
    focal: float,
    ref_dist: float,
    depth_scale: float,
) -> Dict:
    """Danh gia bo tham so tren samples."""
    density_map = load_density_map()
    
    errors = []
    ratios = []
    
    for sample in samples:
        gt_weight = sample['gt_total_weight']
        
        pred_weight = 0
        for item in sample['items']:
            _, _, volume = compute_volume_with_params(
                item['area_pixels'],
                item['mean_depth'],
                item['min_depth'],
                item['max_depth'],
                focal,
                ref_dist,
                depth_scale,
            )
            density = get_density(item['class_name'], density_map)
            pred_weight += volume * density
        
        if pred_weight > 0 and gt_weight > 0:
            error_pct = abs(pred_weight - gt_weight) / gt_weight * 100
            ratio = gt_weight / pred_weight
            errors.append(error_pct)
            ratios.append(ratio)
    
    return {
        'mean_error_pct': np.mean(errors),
        'median_error_pct': np.median(errors),
        'std_error_pct': np.std(errors),
        'mean_ratio': np.mean(ratios),
        'median_ratio': np.median(ratios),
    }


def update_config_file(params: Dict, config_path: Path = None):
    """Cap nhat cac tham so trong config file."""
    if config_path is None:
        config_path = CONFIG_FILE
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update focal_length_px
    content = re.sub(
        r'("focal_length_px":\s*)[\d.]+',
        f'"focal_length_px": {params["focal_length_px"]:.1f}',
        content
    )
    
    # Update reference_distance_cm
    content = re.sub(
        r'("reference_distance_cm":\s*)[\d.]+',
        f'"reference_distance_cm": {params["reference_distance_cm"]:.1f}',
        content
    )
    
    # Update depth_scale_factor
    content = re.sub(
        r'("depth_scale_factor":\s*)[\d.]+',
        f'"depth_scale_factor": {params["depth_scale_factor"]:.4f}',
        content
    )
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n[OK] Da cap nhat config file: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Advanced parameter calibration")
    parser.add_argument("--samples", "-n", type=int, default=50,
                        help="So luong samples (default: 50)")
    parser.add_argument("--update-config", "-u", action="store_true",
                        help="Tu dong cap nhat config file")
    parser.add_argument("--save-samples", "-s", type=str, default=None,
                        help="Luu samples ra file JSON")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ADVANCED PARAMETER CALIBRATION")
    print("=" * 70)
    
    # Collect samples
    samples = collect_samples(num_samples=args.samples)
    
    if not samples:
        print("[ERROR] Khong co samples!")
        return 1
    
    # Save samples if needed
    if args.save_samples:
        with open(args.save_samples, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"[OK] Samples saved to: {args.save_samples}")
    
    # Evaluate current params
    print("\n" + "=" * 70)
    print("DANH GIA THAM SO HIEN TAI")
    print("=" * 70)
    
    current_eval = evaluate_params(samples, 1000.0, 40.0, 1.0)
    print(f"  focal_length_px:      1000.0")
    print(f"  reference_distance_cm: 40.0")
    print(f"  depth_scale_factor:    1.0")
    print(f"\n  Mean error:   {current_eval['mean_error_pct']:.1f}%")
    print(f"  Median error: {current_eval['median_error_pct']:.1f}%")
    
    # Optimize
    optimized = optimize_parameters(samples)
    
    print("\n" + "=" * 70)
    print("KET QUA OPTIMIZATION")
    print("=" * 70)
    
    print(f"\n  [NEW] focal_length_px:      {optimized['focal_length_px']:.1f}")
    print(f"  [NEW] reference_distance_cm: {optimized['reference_distance_cm']:.1f}")
    print(f"  [NEW] depth_scale_factor:    {optimized['depth_scale_factor']:.4f}")
    
    # Evaluate optimized params
    opt_eval = evaluate_params(
        samples,
        optimized['focal_length_px'],
        optimized['reference_distance_cm'],
        optimized['depth_scale_factor'],
    )
    
    print(f"\n  Mean error:   {opt_eval['mean_error_pct']:.1f}% (truoc: {current_eval['mean_error_pct']:.1f}%)")
    print(f"  Median error: {opt_eval['median_error_pct']:.1f}% (truoc: {current_eval['median_error_pct']:.1f}%)")
    
    improvement = (current_eval['median_error_pct'] - opt_eval['median_error_pct']) / current_eval['median_error_pct'] * 100
    print(f"\n  [IMPROVEMENT] Giam {improvement:.1f}% median error!")
    
    # Update config
    if args.update_config:
        update_config_file(optimized)
    else:
        print("\n" + "-" * 70)
        print("[TIP] Chay voi --update-config de tu dong cap nhat config")
        print("      Hoac sua thu cong trong src/config.py:")
        print(f'      "focal_length_px": {optimized["focal_length_px"]:.1f},')
        print(f'      "reference_distance_cm": {optimized["reference_distance_cm"]:.1f},')
        print(f'      "depth_scale_factor": {optimized["depth_scale_factor"]:.4f},')
    
    print("\n" + "=" * 70)
    print("[DONE] Advanced calibration hoan thanh!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
