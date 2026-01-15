"""
NutritionVerse Food Analysis Pipeline

Hệ thống phân tích bữa ăn từ hình ảnh sử dụng:
- Tầng 1: YOLOv8-seg cho food segmentation
- Tầng 2: MiDaS cho depth estimation và volume calculation
- Tầng 3: Physics-based weight estimation

"""

from .config import (
    TIER1_CONFIG,
    TIER2_CONFIG,
    TIER3_CONFIG,
    ensure_directories,
    get_device,
)
from .tier1_segmentation import (
    FoodSegmentationResult,
    Tier1FoodSegmentation,
    Tier1Trainer,
    COCOToYOLOConverter,
)
from .tier2_depth_volume import (
    DepthVolumeResult,
    Tier2DepthVolume,
)
from .tier3_weight_estimation import (
    WeightEstimationResult,
    FoodDensityDatabase,
    Tier3WeightEstimation,
)
from .pipeline import (
    FoodItemAnalysis,
    MealAnalysisResult,
    NutritionVersePipeline,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "TIER1_CONFIG",
    "TIER2_CONFIG", 
    "TIER3_CONFIG",
    "ensure_directories",
    "get_device",
    # Tier 1
    "FoodSegmentationResult",
    "Tier1FoodSegmentation",
    "Tier1Trainer",
    "COCOToYOLOConverter",
    # Tier 2
    "DepthVolumeResult",
    "Tier2DepthVolume",
    # Tier 3
    "WeightEstimationResult",
    "FoodDensityDatabase",
    "Tier3WeightEstimation",
    # Pipeline
    "FoodItemAnalysis",
    "MealAnalysisResult",
    "NutritionVersePipeline",
]
