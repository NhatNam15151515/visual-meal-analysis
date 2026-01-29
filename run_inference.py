"""
run_inference.py - Chạy inference trên ảnh

Cách dùng:
    python run_inference.py                    # Demo mode (5 ảnh từ val set)
    python run_inference.py --image path.jpg  # Chạy trên 1 ảnh cụ thể
"""

import argparse
import sys
from pathlib import Path

from src.config import DATA_DIR, OUTPUT_DIR, ensure_directories
from src.pipeline import NutritionVersePipeline


# ============================================================
# CẤU HÌNH MẶC ĐỊNH
# ============================================================
CONF_THRESHOLD = 0.25  # Ngưỡng confidence
SAVE_RESULTS = True    # Lưu kết quả
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="NutritionVerse Food Analysis")
    parser.add_argument("--image", type=str, default=None,
                        help="Đường dẫn đến ảnh cần phân tích")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help=f"Ngưỡng confidence (default: {CONF_THRESHOLD})")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directories()
    output_dir = OUTPUT_DIR / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Xác định ảnh cần xử lý
    if args.image:
        images = [Path(args.image)]
        if not images[0].exists():
            print(f"[Error] Không tìm thấy ảnh: {args.image}")
            return 1
    else:
        # Demo mode: lấy 5 ảnh từ val set
        val_images = DATA_DIR / "val" / "images"
        images = list(val_images.glob("*.jpg"))[:5]
    
    if not images:
        print("[Error] Không tìm thấy ảnh nào!")
        return 1
    
    print("=" * 60)
    print("NUTRITIONVERSE FOOD ANALYSIS")
    print("=" * 60)
    print(f"Images: {len(images)}")
    print(f"Output: {output_dir}")
    
    # Load pipeline
    # Explicitly use the trained model defined in config, overriding any training defaults
    from src.config import INFERENCE_MODEL_PATH
    pipeline = NutritionVersePipeline(segmentation_model=str(INFERENCE_MODEL_PATH), verbose=True)
    
    # Xử lý
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_path.name}")
        
        try:
            result = pipeline.analyze(str(img_path), conf_threshold=args.conf)
            print(result.summary())
            
            if SAVE_RESULTS:
                pipeline.save_results(result, output_dir, save_masks=False)
                
        except Exception as e:
            print(f"[Error] {e}")
    
    print(f"\n[Done] Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
