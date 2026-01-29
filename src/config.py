"""
config.py - Cấu hình cho Food Analysis Pipeline

Description: Định nghĩa các hằng số, đường dẫn và tham số cho pipeline phân tích bữa ăn.
"""

from pathlib import Path

# ĐƯỜNG DẪN DỮ LIỆU
PROJECT_ROOT = Path(__file__).parent.parent

# Dataset UECFOODPIX (Phase 2: 58 classes, YOLO seg format)
DATA_DIR = PROJECT_ROOT / "data" / "uecfoodpix_yolo_expanded_merged"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
DATA_YAML = DATA_DIR / "data.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

import yaml
def _load_class_names():
    if not DATA_YAML.exists():
        return {}
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', {})

CLASS_NAMES = _load_class_names()
NUM_CLASSES = len(CLASS_NAMES) if CLASS_NAMES else 58

# CẤU HÌNH TRAINING (YOLOv8-seg)
# Model dùng để bắt đầu training (Pretrained COCO hoặc checkpoint cũ)
PRETRAINING_MODEL_PATH = MODELS_DIR / "yolov8" / "yolov8s-seg.pt"

# Model dùng để chạy Inference (Predict)
INFERENCE_MODEL_PATH = MODELS_DIR / "yolov8s_uecfood_seg_58_class" / "weights" / "best.pt"

TRAIN_CONFIG = {
    # Model - YOLOv8m-seg for segmentation (58 classes)
    "model_name": "yolov8s-seg.pt",
    "pretrained_weights": PRETRAINING_MODEL_PATH,
    "trained_model": None,  # Force fresh train
    
    # Training params
    "epochs": 200,
    "batch_size": 8,
    "imgsz": 640,
    "patience": 50,         # Early stopping
    "workers": 4,
    
    # Optimizer
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,            # Cosine decay về cuối (final_lr = lr0 * lrf)
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    
    # Augmentation
    "mosaic": 1.0,
    "close_mosaic": 10,     # Đóng mosaic 10 epochs cuối
    "mixup": 0.05,
    "copy_paste": 0.0,      # Không dùng
    "scale": 0.6,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "shear": 0.0,
    "perspective": 0.0,
    
    # Loss (YOLOv8 default, không dùng focal loss hay class weights)
    "use_focal_loss": False,
    "use_class_weights": False,
    
    # Inference
    "conf_threshold": 0.25,
    "iou_threshold": 0.5,
    "max_det": 20,
    
    # Device
    "device": "cuda",
}

# Alias cho backward compatibility
TIER1_CONFIG = TRAIN_CONFIG

# CẤU HÌNH DEPTH ESTIMATION (MiDaS)
DEPTH_CONFIG = {
    "model_type": "DPT_Large",
    "device": "cuda",
    "focal_length_px": 2790.1,
    "depth_scale_factor": 0.1000,
    "reference_distance_cm": 20.0,
}
# Alias
TIER2_CONFIG = DEPTH_CONFIG

# CẤU HÌNH WEIGHT ESTIMATION
WEIGHT_CONFIG = {
    "density_file": PROJECT_ROOT / "src" / "food_density.csv",
    "default_density": 0.9,
    "global_scale_factor": 0.7709,
}
# Alias
TIER3_CONFIG = WEIGHT_CONFIG

# UTILITY FUNCTIONS

def ensure_directories():
    """Tạo các thư mục cần thiết."""
    for d in [OUTPUT_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    return True

def get_device():
    """Xác định device (CUDA hoặc CPU)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
