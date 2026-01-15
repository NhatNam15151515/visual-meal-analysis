"""
train_segmentation.py - Script fine-tune YOLOv8-seg trên NutritionVerse
Description:
    Script để chuẩn bị dataset và fine-tune model YOLOv8-seg.
    
Usage:
    # Chuẩn bị dataset (COCO -> YOLO format)
    python train_segmentation.py --prepare
    
    # Fine-tune model
    python train_segmentation.py --train
    
    # Chuẩn bị và train
    python train_segmentation.py --prepare --train
"""

import argparse
import sys
from pathlib import Path

# Import từ src package
from src.config import MODELS_DIR, YOLO_DATASET_DIR, ensure_directories
from src.tier1_segmentation import Tier1Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg on NutritionVerse dataset"
    )
    
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Chuẩn bị dataset từ COCO sang YOLO format"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fine-tune model"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Số epochs (mặc định: 50)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (mặc định: 8)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục training từ checkpoint"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.prepare and not args.train:
        print("[Error] Cần chỉ định --prepare hoặc --train")
        return 1
    
    ensure_directories()
    
    print("=" * 60)
    print("YOLOV8-SEG TRAINING FOR NUTRITIONVERSE")
    print("=" * 60)
    
    # Override config nếu cần
    from src.config import TIER1_CONFIG
    TIER1_CONFIG["epochs"] = args.epochs
    TIER1_CONFIG["batch_size"] = args.batch
    
    trainer = Tier1Trainer()
    
    # Chuẩn bị dataset
    if args.prepare:
        print("\n[Step 1] Preparing dataset...")
        data_yaml = trainer.prepare_dataset()
        print(f"Dataset ready at: {YOLO_DATASET_DIR}")
        print(f"Data YAML: {data_yaml}")
    else:
        data_yaml = YOLO_DATASET_DIR / "data.yaml"
        if not data_yaml.exists():
            print("[Error] Dataset chưa được chuẩn bị. Chạy với --prepare trước.")
            return 1
    
    # Training
    if args.train:
        print(f"\n[Step 2] Training...")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Batch size: {args.batch}")
        
        best_model = trainer.train(data_yaml, resume=args.resume)
        
        print(f"\n[Done] Best model saved at: {best_model}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
