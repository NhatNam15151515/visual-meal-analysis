from pathlib import Path
from ultralytics import YOLO
from src.config import TRAIN_CONFIG, DATA_YAML, MODELS_DIR, CLASS_NAMES
from src.confusion_callback import create_confusion_callback


def main():
    print("=" * 60)
    print("TRAINING YOLOv8m SEGMENTATION ON UECFOODPIX (50 CLASSES)")
    print("=" * 60)
    
    # 1. Load Configuration
    data_yaml = DATA_YAML
    pretrained_model = TRAIN_CONFIG["pretrained_weights"]
    
    print(f"[INFO] Dataset path: {data_yaml}")
    print(f"[INFO] Model: {pretrained_model}")
    print(f"[INFO] Config: Epochs={TRAIN_CONFIG['epochs']}, Batch={TRAIN_CONFIG['batch_size']}, ImgSz={TRAIN_CONFIG['imgsz']}, Device={TRAIN_CONFIG['device']}")
    print(f"[INFO] Optimizer: {TRAIN_CONFIG['optimizer']}, LR0={TRAIN_CONFIG['lr0']}, LRF={TRAIN_CONFIG['lrf']}")
    
    # Check data.yaml
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        print("Run: python scripts/convert_uecfoodpix_to_yolo_seg.py first")
        return 1
    
    # 2. Load Model
    print(f"[INFO] Loading model...")
    model = YOLO(str(pretrained_model))
    
    # 3. Setup Confusion Analysis Callback
    log_dir = MODELS_DIR / "yolov8s_uecfood_seg_50_class" / "confusion_logs"
    callbacks = create_confusion_callback(
        class_names=CLASS_NAMES,
        log_dir=log_dir,
        log_every=10  # Phân tích mỗi 10 epochs
    )
    
    # Register callbacks
    for event, callback_fn in callbacks.items():
        model.add_callback(event, callback_fn)
    
    print(f"[INFO] Confusion analysis sẽ được log mỗi 10 epochs vào: {log_dir}")
    
    print("=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    
    # 4. Start Training using parameters from TRAIN_CONFIG
    try:
        results = model.train(
            data=str(data_yaml),
            project=str(MODELS_DIR),
            name="yolov8s_uecfood_seg_50_class",
            exist_ok=True,
            
            # Training params
            epochs=TRAIN_CONFIG["epochs"],
            batch=TRAIN_CONFIG["batch_size"],
            imgsz=TRAIN_CONFIG["imgsz"],
            patience=TRAIN_CONFIG["patience"],
            device=TRAIN_CONFIG["device"],
            workers=TRAIN_CONFIG["workers"],
            
            # Optimizer
            optimizer=TRAIN_CONFIG["optimizer"],
            lr0=TRAIN_CONFIG["lr0"],
            lrf=TRAIN_CONFIG["lrf"],
            momentum=TRAIN_CONFIG["momentum"],
            weight_decay=TRAIN_CONFIG["weight_decay"],
            warmup_epochs=TRAIN_CONFIG["warmup_epochs"],
            
            # Augmentation
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
        print(f"Best model saved to: {MODELS_DIR / 'yolov8s_uecfood_seg_50_class' / 'weights' / 'best.pt'}")
        print(f"Confusion analysis log: {log_dir / 'confusion_analysis.log'}")
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
