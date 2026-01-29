"""
train_103.py - Training YOLOv8-seg on FoodSeg103 (103 ingredient classes)

Purpose: Pretrain ingredient-aware segmentation backbone before fine-tuning 
on dish-level segmentation tasks.
"""

from pathlib import Path
from ultralytics import YOLO

# Import existing config and adjust for FoodSeg103
from src.config import TRAIN_CONFIG, MODELS_DIR, PROJECT_ROOT

# FoodSeg103 specific paths
FOODSEG103_DIR = PROJECT_ROOT / "data" / "FoodSeg103"
FOODSEG103_YAML = FOODSEG103_DIR / "data.yaml"

# FoodSeg103 has 103 classes (ingredients, no background)
NUM_CLASSES_103 = 103


def main():
    print("=" * 60)
    print("TRAINING YOLOv8s SEGMENTATION ON FOODSEG103 (103 CLASSES)")
    print("Purpose: Ingredient-aware backbone pretraining")
    print("=" * 60)
    
    # 1. Verify data.yaml exists
    if not FOODSEG103_YAML.exists():
        print(f"[ERROR] data.yaml not found: {FOODSEG103_YAML}")
        print("Run: python extract_foodseg103.py && python convert_foodseg103_to_yolo.py")
        return 1
    
    print(f"[INFO] Dataset: {FOODSEG103_YAML}")
    print(f"[INFO] Classes: {NUM_CLASSES_103}")
    
    # 2. Load pretrained YOLOv8s-seg
    pretrained_model = TRAIN_CONFIG["pretrained_weights"]
    print(f"[INFO] Loading model: {pretrained_model}")
    
    model = YOLO(str(pretrained_model))
    
    # 3. Training configuration (adjusted for FoodSeg103)
    # Use existing TRAIN_CONFIG but with some modifications
    print(f"[INFO] Config: Epochs={TRAIN_CONFIG['epochs']}, Batch={TRAIN_CONFIG['batch_size']}, ImgSz={TRAIN_CONFIG['imgsz']}")
    
    print("=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    
    try:
        results = model.train(
            data=str(FOODSEG103_YAML),
            project=str(MODELS_DIR),
            name="yolov8s_foodseg103_pretrain",
            exist_ok=True,
            
            # Training params from config
            epochs=TRAIN_CONFIG["epochs"],
            batch=TRAIN_CONFIG["batch_size"],
            imgsz=TRAIN_CONFIG["imgsz"],
            patience=TRAIN_CONFIG["patience"],
            device=TRAIN_CONFIG["device"],
            workers=TRAIN_CONFIG["workers"],
            
            # Optimizer from config
            optimizer=TRAIN_CONFIG["optimizer"],
            lr0=TRAIN_CONFIG["lr0"],
            lrf=TRAIN_CONFIG["lrf"],
            momentum=TRAIN_CONFIG["momentum"],
            weight_decay=TRAIN_CONFIG["weight_decay"],
            warmup_epochs=TRAIN_CONFIG["warmup_epochs"],
            
            # Augmentation from config
            mosaic=TRAIN_CONFIG["mosaic"],
            close_mosaic=TRAIN_CONFIG["close_mosaic"],
            mixup=TRAIN_CONFIG["mixup"],
            copy_paste=TRAIN_CONFIG["copy_paste"],
            scale=TRAIN_CONFIG["scale"],
            fliplr=TRAIN_CONFIG["fliplr"],
            hsv_h=TRAIN_CONFIG["hsv_h"],
            hsv_s=TRAIN_CONFIG["hsv_s"],
            hsv_v=TRAIN_CONFIG["hsv_v"],
            degrees=TRAIN_CONFIG["degrees"],
            translate=TRAIN_CONFIG["translate"],
            shear=TRAIN_CONFIG["shear"],
            perspective=TRAIN_CONFIG["perspective"],
            
            # Segmentation specific
            mask_ratio=4,
            overlap_mask=True,
            
            # System settings
            amp=True,
            cache=False,
            verbose=True,
            plots=True,
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        best_model_path = MODELS_DIR / "yolov8s_foodseg103_pretrain" / "weights" / "best.pt"
        print(f"Best model saved to: {best_model_path}")
        print("\nNext step: Use this model as pretrained weights for dish-level segmentation.")
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
