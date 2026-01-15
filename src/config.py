"""
config.py - Cấu hình toàn cục cho NutritionVerse Food Analysis Pipeline

Description: Định nghĩa các hằng số, đường dẫn và tham số cho pipeline phân tích bữa ăn.
"""

import os
from pathlib import Path

# ==============================================================================
# ĐƯỜNG DẪN DỮ LIỆU
# ==============================================================================
# Root directory của project
PROJECT_ROOT = Path(__file__).parent.parent

# Đường dẫn dataset
DATA_DIR = PROJECT_ROOT / "data"
NUTRITIONVERSE_DIR = DATA_DIR / "nutritionverse-manual" / "nutritionverse-manual"
IMAGES_DIR = NUTRITIONVERSE_DIR / "images"
ANNOTATIONS_FILE = IMAGES_DIR / "_annotations.coco.json"
SPLITS_FILE = NUTRITIONVERSE_DIR / "updated-manual-dataset-splits.csv"
METADATA_FILE = DATA_DIR / "nutritionverse_dish_metadata3.csv"

# Đường dẫn outputs
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_DATASET_DIR = PROJECT_ROOT / "yolo_dataset"

# ==============================================================================
# CẤU HÌNH TẦNG 1: SEGMENTATION (YOLOv8-seg)
# ==============================================================================
# Đường dẫn đến model đã fine-tune trên NutritionVerse
TRAINED_MODEL_PATH = MODELS_DIR / "yolov8_food_seg3" / "weights" / "best.pt"

TIER1_CONFIG = {
    # Model YOLO - sử dụng model đã train nếu có, nếu không dùng pretrained
    "model_name": "yolov8m-seg.pt",  # Medium model, cân bằng tốc độ và độ chính xác
    "pretrained_weights": "yolov8m-seg.pt",
    "trained_model": str(TRAINED_MODEL_PATH) if TRAINED_MODEL_PATH.exists() else None,
    
    # Fine-tuning parameters
    "epochs": 50,
    "batch_size": 8,
    "imgsz": 640,
    "learning_rate": 0.001,
    "patience": 10,  # Early stopping
    
    # Inference parameters
    "conf_threshold": 0.25,  # Ngưỡng confidence score
    "iou_threshold": 0.45,   # Ngưỡng IoU cho NMS
    "max_det": 20,           # Số detection tối đa mỗi ảnh
    
    # Device
    "device": "cuda",  # "cuda" hoặc "cpu"
}

# ==============================================================================
# CẤU HÌNH TẦNG 2: DEPTH ESTIMATION (MiDaS)
# ==============================================================================
TIER2_CONFIG = {
    # Model MiDaS
    # Các lựa chọn: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
    "model_type": "DPT_Large",  # Độ chính xác cao nhất
    
    # Device
    "device": "cuda",
    
    # Các tham số ước lượng thể tích
    # Giả định: Camera intrinsic parameters (cần calibrate cho độ chính xác cao)
    # Focal length ước lượng cho smartphone camera thông thường
    "focal_length_px": 2790.1,  # 1000, pixels (cần điều chỉnh theo camera)
    
    # Hệ số scale chuyển đổi depth tương đối sang thực tế
    # Giá trị này cần được calibrate với dữ liệu ground truth
    "depth_scale_factor": 0.1000,
    
    # Khoảng cách tham chiếu (cm) - khoảng cách trung bình từ camera đến bàn ăn
    "reference_distance_cm": 20.0, #40
}

# ==============================================================================
# CẤU HÌNH TẦNG 3: WEIGHT ESTIMATION
# ==============================================================================
TIER3_CONFIG = {
    # File chứa mật độ thực phẩm (g/cm³)
    "density_file": PROJECT_ROOT / "src" / "food_density.csv",
    
    # Mật độ mặc định (g/cm³) khi không tìm thấy trong database
    # Giá trị này dựa trên mật độ trung bình của thực phẩm hỗn hợp
    "default_density": 0.9,
    
    # Hệ số hiệu chỉnh toàn cục (calibration factor)
    # Dùng để điều chỉnh sai số hệ thống
    "global_scale_factor": 0.7709, #1.0
}

# ==============================================================================
# DANH SÁCH CÁC LOẠI THỰC PHẨM
# ==============================================================================
# Lấy từ NutritionVerse-Real dataset
FOOD_CATEGORIES = [
    "food-ingredients",  # Category cha
    "asian-pear",
    "captain-crunch-granola-bar", 
    "carrot",
    "chicken-breast",
    "chicken-leg",
    "chicken-sandwich",
    "chicken-wing",
    "chocolate-granola-bar",
    "corn",
    "costco-california-sushi-roll-1",
    "costco-egg",
    "costco-salad-sushi-roll-1",
    "costco-shrimp-sushi-roll-1",
    "crispy-pork-rib",
    "cucumber-piece",
    "french-fry",
    "half-bread-loaf",
    "hamburger",
    "lamb-shank",
    "lasagna",
    "lobster",
    "meatball",
    "nature-valley-granola-bar",
    "plain-toast",
    "pork-feet",
    "red-apple",
    "rib",
    "salad-chicken-strip",
    "salmon-nigiri",
    "shrimp-nigiri",
    "stack-of-tofu-4pc",
    "steak",
    "toast-with-strawberry-jam",
    "tuna-nigiri",
    "veal-kebab-piece",
]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def ensure_directories():
    """Tạo các thư mục cần thiết nếu chưa tồn tại."""
    directories = [OUTPUT_DIR, MODELS_DIR, YOLO_DATASET_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return True

def get_device():
    """Xác định device phù hợp (CUDA hoặc CPU)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
