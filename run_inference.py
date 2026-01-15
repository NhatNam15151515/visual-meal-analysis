"""
run_inference.py - Script chạy inference cho NutritionVerse Pipeline

Description:
    Script chính để chạy phân tích bữa ăn từ hình ảnh.
    
Usage:
    python run_inference.py --image path/to/image.jpg
    python run_inference.py --folder path/to/images/
    python run_inference.py --demo

Pipeline: ảnh → detect/segment → depth → volume → weight
"""

import argparse
import sys
from pathlib import Path

# Import từ src package (không cần thêm vào sys.path)
from src.config import IMAGES_DIR, OUTPUT_DIR, ensure_directories
from src.pipeline import NutritionVersePipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NutritionVerse Food Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Phân tích một ảnh
    python run_inference.py --image data/images/meal.jpg
    
    # Phân tích một folder ảnh
    python run_inference.py --folder data/images/ --limit 10
    
    # Chạy demo với dataset có sẵn
    python run_inference.py --demo
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Đường dẫn đến một ảnh cần phân tích"
    )
    
    parser.add_argument(
        "--folder", "-f",
        type=str,
        help="Đường dẫn đến folder chứa nhiều ảnh"
    )
    
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Chạy demo với ảnh từ NutritionVerse dataset"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=5,
        help="Giới hạn số ảnh xử lý (mặc định: 5)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Ngưỡng confidence cho detection (mặc định: 0.25)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Thư mục output (mặc định: outputs/)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Đường dẫn đến model YOLOv8-seg đã fine-tune"
    )
    
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Không lưu visualization"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Xác định output directory
    output_dir = Path(args.output) if args.output else OUTPUT_DIR / "inference_results"
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Thu thập danh sách ảnh
    images = []
    
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[Error] Không tìm thấy ảnh: {img_path}")
            return 1
        images.append(img_path)
        
    elif args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"[Error] Không tìm thấy folder: {folder}")
            return 1
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        images = images[:args.limit]
        
    elif args.demo:
        images = list(IMAGES_DIR.glob("*.jpg"))[:args.limit]
        
    else:
        print("[Error] Cần chỉ định --image, --folder, hoặc --demo")
        return 1
    
    if not images:
        print("[Error] Không tìm thấy ảnh nào để xử lý!")
        return 1
    
    print("=" * 70)
    print("NUTRITIONVERSE FOOD ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Images to process: {len(images)}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.conf}")
    print("=" * 70)
    
    # Khởi tạo pipeline
    print("\n[Init] Loading pipeline components...")
    pipeline = NutritionVersePipeline(
        segmentation_model=args.model,
        verbose=True,
    )
    
    # Xử lý từng ảnh
    all_results = []
    
    for i, img_path in enumerate(images):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(images)}] Processing: {img_path.name}")
        print("=" * 60)
        
        try:
            # Phân tích
            result = pipeline.analyze(
                str(img_path),
                conf_threshold=args.conf
            )
            
            # In summary
            print("\n" + result.summary())
            
            # Lưu kết quả
            if not args.no_vis:
                pipeline.save_results(result, output_dir, save_masks=False)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"[Error] Lỗi khi xử lý {img_path.name}: {e}")
            continue
    
    # Tổng kết
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {len(all_results)}/{len(images)} images")
    
    total_items = sum(r.num_items for r in all_results)
    total_weight = sum(r.total_weight_grams for r in all_results)
    
    print(f"Total food items detected: {total_items}")
    print(f"Total weight estimated: {total_weight:.1f}g")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
