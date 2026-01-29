"""
pipeline.py - Pipeline tổng hợp cho NutritionVerse Food Analysis
Description:
    Module tích hợp 3 tầng xử lý:
    - Tầng 1: Segmentation (YOLOv8-seg)
    - Tầng 2: Depth & Volume (MiDaS)
    - Tầng 3: Weight Estimation (Physics-based)

Pipeline: ảnh → detect/segment → depth → volume → weight
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    OUTPUT_DIR,
    ensure_directories,
)
from .tier1_segmentation import FoodSegmentationResult, Tier1FoodSegmentation
from .tier2_depth_volume import DepthVolumeResult, Tier2DepthVolume
from .tier3_weight_estimation import Tier3WeightEstimation, WeightEstimationResult


@dataclass
class FoodItemAnalysis:
    """Kết quả phân tích hoàn chỉnh cho một món ăn."""
    class_name: str
    confidence: float
    bbox: np.ndarray
    mask: np.ndarray
    area_pixels: int
    area_cm2: float
    height_cm: float
    volume_cm3: float
    density_g_per_cm3: float
    density_source: str
    weight_grams: float
    weight_confidence: float

    def to_dict(self) -> Dict:
        """Chuyển về dictionary để serialize."""
        return {
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox,
            "area_pixels": int(self.area_pixels),
            "area_cm2": float(self.area_cm2),
            "height_cm": float(self.height_cm),
            "volume_cm3": float(self.volume_cm3),
            "density_g_per_cm3": float(self.density_g_per_cm3),
            "density_source": self.density_source,
            "weight_grams": float(self.weight_grams),
            "weight_confidence": float(self.weight_confidence),
        }


@dataclass
class MealAnalysisResult:
    """Kết quả phân tích toàn bộ bữa ăn."""
    image_path: str
    food_items: List[FoodItemAnalysis]
    total_weight_grams: float
    depth_map: np.ndarray

    @property
    def num_items(self) -> int:
        return len(self.food_items)

    def to_dict(self) -> Dict:
        """Chuyển về dictionary để serialize."""
        return {
            "image_path": self.image_path,
            "num_items": self.num_items,
            "total_weight_grams": float(self.total_weight_grams),
            "food_items": [item.to_dict() for item in self.food_items],
        }

    def summary(self) -> str:
        """Tạo summary text."""
        lines = [
            f"Meal Analysis: {Path(self.image_path).name}",
            f"Total items: {self.num_items}",
            f"Total weight: {self.total_weight_grams:.1f}g",
            "-" * 40,
        ]
        for item in self.food_items:
            lines.append(
                f"  • {item.class_name}: {item.weight_grams:.1f}g "
                f"(vol={item.volume_cm3:.1f}cm³)"
            )
        return "\n".join(lines)


class NutritionVersePipeline:
    """
    Pipeline phân tích bữa ăn từ hình ảnh.
    
    Tích hợp 3 tầng xử lý:
    1. Tầng 1: Food Segmentation - detect và segment các món ăn
    2. Tầng 2: Depth & Volume - ước lượng độ sâu và thể tích
    3. Tầng 3: Weight Estimation - tính trọng lượng từ thể tích và mật độ
    
    Usage:
        pipeline = NutritionVersePipeline()
        result = pipeline.analyze("path/to/image.jpg")
        print(result.summary())
    """

    def __init__(
        self,
        segmentation_model: Optional[str] = None,
        tier1_config: Optional[Dict] = None,
        tier2_config: Optional[Dict] = None,
        tier3_config: Optional[Dict] = None,
        verbose: bool = True,
    ):
        """
        Khởi tạo pipeline.
        
        Args:
            segmentation_model: Đường dẫn đến model YOLOv8-seg đã fine-tune
            tier1_config: Config cho Tầng 1
            tier2_config: Config cho Tầng 2
            tier3_config: Config cho Tầng 3
            verbose: In thông tin debug
        """
        self.verbose = verbose
        ensure_directories()

        if verbose:
            print("[Pipeline] Initializing NutritionVerse Analysis Pipeline...")

        # Tầng 1: Segmentation
        if verbose:
            print("[Pipeline] Loading Tier 1: Food Segmentation...")
        self.tier1 = Tier1FoodSegmentation(
            model_path=segmentation_model,
            config=tier1_config,
        )

        # Tầng 2: Depth & Volume
        if verbose:
            print("[Pipeline] Loading Tier 2: Depth & Volume Estimation...")
        self.tier2 = Tier2DepthVolume(config=tier2_config)

        # Tầng 3: Weight Estimation
        if verbose:
            print("[Pipeline] Loading Tier 3: Weight Estimation...")
        self.tier3 = Tier3WeightEstimation(config=tier3_config)

        if verbose:
            print("[Pipeline] Initialization complete!")

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
    ) -> MealAnalysisResult:
        """
        Phân tích một ảnh bữa ăn.
        
        Args:
            image: Đường dẫn ảnh hoặc numpy array
            conf_threshold: Ngưỡng confidence cho segmentation
        
        Returns:
            MealAnalysisResult: Kết quả phân tích hoàn chỉnh
        """
        # Xác định image path
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_array = cv2.imread(image_path)
        else:
            image_path = "numpy_array"
            img_array = image

        if self.verbose:
            print(f"\n[Pipeline] Analyzing: {Path(image_path).name}")

        # ============================================================
        # TẦNG 1: SEGMENTATION
        # ============================================================
        if self.verbose:
            print("[Tier1] Running food segmentation...")

        seg_results: List[FoodSegmentationResult] = self.tier1.predict(
            img_array, conf_threshold=conf_threshold
        )

        if self.verbose:
            print(f"[Tier1] Detected {len(seg_results)} food items")

        if len(seg_results) == 0:
            return MealAnalysisResult(
                image_path=image_path,
                food_items=[],
                total_weight_grams=0.0,
                depth_map=np.zeros(img_array.shape[:2], dtype=np.float32),
            )

        # ============================================================
        # TẦNG 2: DEPTH & VOLUME
        # ============================================================
        if self.verbose:
            print("[Tier2] Estimating depth and volume...")

        # Sinh depth map
        depth_map = self.tier2.estimate_depth(img_array)

        # Chuẩn bị masks
        masks = [(r.class_name, r.mask) for r in seg_results]

        # Ước lượng volume
        vol_results: List[DepthVolumeResult] = self.tier2.estimate_volume(
            img_array, masks, depth_map
        )

        # ============================================================
        # TẦNG 3: WEIGHT ESTIMATION
        # ============================================================
        if self.verbose:
            print("[Tier3] Estimating weights...")

        items_for_weight = [(v.class_name, v.volume_cm3) for v in vol_results]
        weight_results: List[WeightEstimationResult] = self.tier3.estimate_weights_batch(
            items_for_weight
        )

        # ============================================================
        # TỔNG HỢP KẾT QUẢ
        # ============================================================
        food_items = []

        for seg, vol, weight in zip(seg_results, vol_results, weight_results):
            item = FoodItemAnalysis(
                class_name=seg.class_name,
                confidence=seg.confidence,
                bbox=seg.bbox,
                mask=seg.mask,
                area_pixels=seg.area_pixels,
                area_cm2=vol.area_cm2,
                height_cm=vol.height_cm,
                volume_cm3=vol.volume_cm3,
                density_g_per_cm3=weight.density_g_per_cm3,
                density_source=weight.density_source,
                weight_grams=weight.weight_grams,
                weight_confidence=weight.confidence,
            )
            food_items.append(item)

        total_weight = sum(item.weight_grams for item in food_items)

        result = MealAnalysisResult(
            image_path=image_path,
            food_items=food_items,
            total_weight_grams=total_weight,
            depth_map=depth_map,
        )

        if self.verbose:
            print(f"[Pipeline] Analysis complete: {len(food_items)} items, {total_weight:.1f}g total")

        return result

    def analyze_batch(
        self,
        images: List[Union[str, Path]],
        conf_threshold: Optional[float] = None,
    ) -> List[MealAnalysisResult]:
        """
        Phân tích nhiều ảnh.
        
        Args:
            images: Danh sách đường dẫn ảnh
            conf_threshold: Ngưỡng confidence
        
        Returns:
            List[MealAnalysisResult]
        """
        results = []
        for i, img_path in enumerate(images):
            if self.verbose:
                print(f"\n[Batch] Processing {i+1}/{len(images)}")
            result = self.analyze(img_path, conf_threshold)
            results.append(result)
        return results

    def visualize(
        self,
        result: MealAnalysisResult,
        output_path: Optional[Union[str, Path]] = None,
        show_depth: bool = True,
    ) -> np.ndarray:
        """
        Visualize kết quả phân tích.
        
        Args:
            result: MealAnalysisResult
            output_path: Đường dẫn lưu output
            show_depth: Hiển thị depth map
        
        Returns:
            np.ndarray: Visualization image
        """
        # Load ảnh gốc
        img = cv2.imread(result.image_path)
        h, w = img.shape[:2]

        # Vẽ masks và labels
        overlay = img.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

        for i, item in enumerate(result.food_items):
            color = colors[i % len(colors)]

            # Vẽ mask
            mask_bool = item.mask > 0
            overlay[mask_bool] = (
                0.4 * np.array(color) + 0.6 * overlay[mask_bool]
            ).astype(np.uint8)

            # Vẽ bbox
            x1, y1, x2, y2 = item.bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label với weight
            label = f"{item.class_name}: {item.weight_grams:.0f}g"
            cv2.putText(
                img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4
            )

        # Blend
        vis = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

        # Thêm depth map nếu cần
        if show_depth:
            depth_colored = self.tier2.visualize_depth(result.depth_map)
            if depth_colored.shape[:2] != (h, w):
                depth_colored = cv2.resize(depth_colored, (w, h))
            vis = np.hstack([vis, depth_colored])

        # Thêm tổng weight
        total_text = f"Total: {result.total_weight_grams:.1f}g"
        cv2.putText(
            vis, total_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        if output_path:
            cv2.imwrite(str(output_path), vis)

        return vis

    def save_results(
        self,
        result: MealAnalysisResult,
        output_dir: Union[str, Path],
        save_masks: bool = False,
    ):
        """
        Lưu kết quả phân tích.
        
        Args:
            result: MealAnalysisResult
            output_dir: Thư mục output
            save_masks: Lưu masks riêng lẻ
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        img_name = Path(result.image_path).stem

        # Lưu JSON
        json_path = output_dir / f"{img_name}_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        # Lưu visualization
        vis_path = output_dir / f"{img_name}_visualization.jpg"
        self.visualize(result, output_path=vis_path)

        # Lưu depth map
        depth_path = output_dir / f"{img_name}_depth.npy"
        np.save(depth_path, result.depth_map)

        # Lưu masks nếu cần
        if save_masks:
            for i, item in enumerate(result.food_items):
                mask_path = output_dir / f"{img_name}_mask_{i}_{item.class_name}.png"
                cv2.imwrite(str(mask_path), item.mask)

        print(f"[Pipeline] Results saved to: {output_dir}")


# ==============================================================================
# EXAMPLE: COMPLETE INFERENCE PIPELINE
# ==============================================================================
# Để chạy pipeline, sử dụng run_inference.py từ thư mục gốc project:
#   python run_inference.py --demo
#
# Không thể chạy trực tiếp file này do sử dụng relative imports.
