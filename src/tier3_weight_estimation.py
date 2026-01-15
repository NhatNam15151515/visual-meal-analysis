"""
tier3_weight_estimation.py - TẦNG 3: Ước lượng trọng lượng

Description:
    Tính trọng lượng thực phẩm dựa trên thể tích ước lượng và mật độ.
    
    KHÔNG sử dụng deep learning để dự đoán trọng lượng trực tiếp.
    Thay vào đó, sử dụng công thức vật lý:
    
        trọng_lượng (g) = thể_tích (cm³) × mật_độ (g/cm³)

Các giả định:
    1. Mật độ thực phẩm được tra từ bảng lookup (food_density.csv)
    2. Nếu không tìm thấy, sử dụng mật độ mặc định (~0.9 g/cm³)
    3. Thể tích đầu vào từ Tầng 2 đã được calibrate (hoặc chấp nhận sai số)
    4. Mật độ là giá trị trung bình, không tính đến:
       - Độ xốp/rỗng của thực phẩm
       - Trạng thái nấu chín (raw vs cooked)
       - Độ ẩm

Đầu ra:
    - Trọng lượng ước lượng (gram)
    - Độ tin cậy (confidence) dựa trên nguồn mật độ
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, TIER3_CONFIG, ensure_directories


class WeightEstimationResult:
    """
    Class lưu trữ kết quả ước lượng trọng lượng cho một món ăn.
    
    Attributes:
        class_name (str): Tên loại thực phẩm
        volume_cm3 (float): Thể tích ước lượng (cm³)
        density_g_per_cm3 (float): Mật độ (g/cm³)
        density_source (str): Nguồn mật độ ("database", "default", "manual")
        weight_grams (float): Trọng lượng ước lượng (g)
        confidence (float): Độ tin cậy (0-1)
    """
    
    def __init__(
        self,
        class_name: str,
        volume_cm3: float,
        density_g_per_cm3: float,
        density_source: str,
    ):
        self.class_name = class_name
        self.volume_cm3 = volume_cm3
        self.density_g_per_cm3 = density_g_per_cm3
        self.density_source = density_source
        
        # ============================================================
        # CÔNG THỨC TÍNH TRỌNG LƯỢNG
        # ============================================================
        # 
        # Định luật: m = ρ × V
        # Trong đó:
        #   m: khối lượng (mass) [grams]
        #   ρ (rho): mật độ (density) [g/cm³]
        #   V: thể tích (volume) [cm³]
        #
        # Công thức này dựa trên giả định:
        # - Thực phẩm có mật độ đồng nhất
        # - Thể tích đã được ước lượng chính xác
        # - Mật độ trong bảng là giá trị đại diện
        # ============================================================
        
        self.weight_grams = volume_cm3 * density_g_per_cm3
        
        # Tính confidence dựa trên nguồn mật độ
        if density_source == "database":
            self.confidence = 0.85  # Mật độ từ database đáng tin cậy
        elif density_source == "manual":
            self.confidence = 0.90  # Người dùng cung cấp
        else:
            self.confidence = 0.60  # Mật độ mặc định, ít tin cậy
    
    def __repr__(self):
        return (
            f"WeightResult("
            f"'{self.class_name}': "
            f"V={self.volume_cm3:.1f}cm³ × "
            f"ρ={self.density_g_per_cm3:.2f}g/cm³ = "
            f"{self.weight_grams:.1f}g "
            f"[{self.density_source}, conf={self.confidence:.0%}])"
        )


class FoodDensityDatabase:
    """
    Database quản lý mật độ các loại thực phẩm.
    
    Nguồn dữ liệu:
    - File CSV: food_density.csv
    - Các giá trị được thu thập từ:
      + USDA FoodData Central
      + Tài liệu khoa học dinh dưỡng
      + Đo đạc thực nghiệm
    
    Đơn vị: g/cm³ (grams per cubic centimeter)
    """
    
    def __init__(
        self,
        density_file: Optional[Union[str, Path]] = None,
        default_density: float = 0.9,
    ):
        """
        Args:
            density_file: Đường dẫn đến file CSV chứa mật độ
            default_density: Mật độ mặc định khi không tìm thấy
        """
        self.default_density = default_density
        self.density_map: Dict[str, float] = {}
        self.notes_map: Dict[str, str] = {}
        
        # Load từ file nếu có
        if density_file and Path(density_file).exists():
            self._load_from_csv(density_file)
        else:
            self._init_default_densities()
    
    def _load_from_csv(self, filepath: Union[str, Path]):
        """Load mật độ từ file CSV."""
        df = pd.read_csv(filepath)
        
        for _, row in df.iterrows():
            food_type = str(row["food_type"]).strip().lower()
            density = float(row["density_g_per_cm3"])
            notes = str(row.get("notes", ""))
            
            self.density_map[food_type] = density
            self.notes_map[food_type] = notes
        
        print(f"[DensityDB] Loaded {len(self.density_map)} food types from {filepath}")
    
    def _init_default_densities(self):
        """Khởi tạo bảng mật độ mặc định."""
        # Giá trị tham khảo từ USDA và tài liệu
        defaults = {
            # Trái cây
            "apple": 0.85,
            "red-apple": 0.85,
            "asian-pear": 0.58,
            "orange": 0.92,
            
            # Rau củ
            "carrot": 1.04,
            "cucumber": 0.96,
            "cucumber-piece": 0.96,
            "corn": 0.72,
            
            # Thịt
            "chicken-breast": 1.05,
            "chicken-leg": 1.02,
            "chicken-wing": 0.95,
            "steak": 1.05,
            "lamb-shank": 1.08,
            "pork-feet": 0.95,
            "rib": 0.90,
            "crispy-pork-rib": 0.85,
            "meatball": 1.10,
            "veal-kebab-piece": 1.02,
            
            # Hải sản
            "lobster": 1.05,
            "shrimp-nigiri": 1.00,
            "salmon-nigiri": 1.02,
            "tuna-nigiri": 1.08,
            
            # Bánh mì & tinh bột
            "bread": 0.30,
            "half-bread-loaf": 0.30,
            "plain-toast": 0.28,
            "toast-with-strawberry-jam": 0.35,
            "french-fry": 0.35,
            "lasagna": 0.95,
            
            # Fast food
            "hamburger": 0.65,
            "chicken-sandwich": 0.55,
            
            # Sushi
            "costco-california-sushi-roll-1": 1.05,
            "costco-salad-sushi-roll-1": 0.95,
            "costco-shrimp-sushi-roll-1": 1.02,
            
            # Khác
            "tofu": 0.80,
            "stack-of-tofu-4pc": 0.80,
            "costco-egg": 1.03,
            "salad-chicken-strip": 0.98,
            "nature-valley-granola-bar": 0.42,
            "captain-crunch-granola-bar": 0.45,
            "chocolate-granola-bar": 0.48,
        }
        
        self.density_map = defaults
        print(f"[DensityDB] Initialized with {len(defaults)} default densities")
    
    def get_density(
        self,
        food_type: str,
        manual_density: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        Lấy mật độ cho một loại thực phẩm.
        
        Args:
            food_type: Tên loại thực phẩm
            manual_density: Mật độ do người dùng cung cấp (override)
        
        Returns:
            Tuple[float, str]: (mật_độ, nguồn)
                nguồn: "manual", "database", hoặc "default"
        """
        # Ưu tiên manual
        if manual_density is not None:
            return (manual_density, "manual")
        
        # Tìm trong database
        food_type_lower = food_type.strip().lower()
        
        if food_type_lower in self.density_map:
            return (self.density_map[food_type_lower], "database")
        
        # Thử tìm partial match
        for key, density in self.density_map.items():
            if key in food_type_lower or food_type_lower in key:
                return (density, "database")
        
        # Fallback to default
        return (self.default_density, "default")
    
    def add_density(
        self,
        food_type: str,
        density: float,
        notes: str = "",
    ):
        """Thêm hoặc cập nhật mật độ cho một loại thực phẩm."""
        food_type_lower = food_type.strip().lower()
        self.density_map[food_type_lower] = density
        self.notes_map[food_type_lower] = notes
    
    def save_to_csv(self, filepath: Union[str, Path]):
        """Lưu database ra file CSV."""
        data = []
        for food_type, density in self.density_map.items():
            data.append({
                "food_type": food_type,
                "density_g_per_cm3": density,
                "notes": self.notes_map.get(food_type, ""),
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"[DensityDB] Saved {len(data)} entries to {filepath}")


class Tier3WeightEstimation:
    """
    TẦNG 3: Module ước lượng trọng lượng thực phẩm.
    
    Pipeline:
    1. Nhận thể tích từ Tầng 2
    2. Tra cứu mật độ từ database
    3. Tính trọng lượng = thể tích × mật độ
    4. Áp dụng hệ số hiệu chỉnh (nếu có)
    
    Lưu ý về độ chính xác:
    - Sai số tích lũy từ các tầng trước
    - Mật độ là giá trị trung bình, có biến động
    - Kết quả nên được xem như ƯỚC LƯỢNG, không phải đo lường chính xác
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Dictionary cấu hình. Nếu None, sử dụng TIER3_CONFIG.
        """
        self.config = config or TIER3_CONFIG
        
        # Load density database
        density_file = self.config.get("density_file")
        default_density = self.config.get("default_density", 0.9)
        
        self.density_db = FoodDensityDatabase(
            density_file=density_file,
            default_density=default_density,
        )
        
        # Global scale factor cho calibration
        self.scale_factor = self.config.get("global_scale_factor", 1.0)
    
    def estimate_weight(
        self,
        class_name: str,
        volume_cm3: float,
        manual_density: Optional[float] = None,
    ) -> WeightEstimationResult:
        """
        Ước lượng trọng lượng cho một món ăn.
        
        Args:
            class_name: Tên loại thực phẩm
            volume_cm3: Thể tích (cm³)
            manual_density: Mật độ override (g/cm³)
        
        Returns:
            WeightEstimationResult
        """
        # Lấy mật độ
        density, source = self.density_db.get_density(class_name, manual_density)
        
        # Tạo kết quả
        result = WeightEstimationResult(
            class_name=class_name,
            volume_cm3=volume_cm3 * self.scale_factor,  # Apply calibration
            density_g_per_cm3=density,
            density_source=source,
        )
        
        return result
    
    def estimate_weights_batch(
        self,
        items: List[Tuple[str, float]],
        manual_densities: Optional[Dict[str, float]] = None,
    ) -> List[WeightEstimationResult]:
        """
        Ước lượng trọng lượng cho nhiều món.
        
        Args:
            items: List of (class_name, volume_cm3)
            manual_densities: Dict mapping class_name -> density
        
        Returns:
            List[WeightEstimationResult]
        """
        manual_densities = manual_densities or {}
        results = []
        
        for class_name, volume in items:
            manual_d = manual_densities.get(class_name)
            result = self.estimate_weight(class_name, volume, manual_d)
            results.append(result)
        
        return results
    
    def calibrate_with_ground_truth(
        self,
        predictions: List[WeightEstimationResult],
        ground_truth_weights: List[float],
    ) -> float:
        """
        Calibrate scale factor dựa trên ground truth.
        
        Args:
            predictions: Danh sách predictions
            ground_truth_weights: Trọng lượng thực (grams)
        
        Returns:
            float: Scale factor mới
        """
        if len(predictions) != len(ground_truth_weights):
            raise ValueError("Số lượng predictions và ground truth không khớp")
        
        pred_weights = np.array([p.weight_grams for p in predictions])
        gt_weights = np.array(ground_truth_weights)
        
        # Least squares: gt = scale * pred
        # scale = sum(gt * pred) / sum(pred^2)
        new_scale = np.sum(gt_weights * pred_weights) / (np.sum(pred_weights ** 2) + 1e-8)
        
        print(f"[Tier3] Calibration: old_scale={self.scale_factor:.4f} -> new_scale={new_scale:.4f}")
        self.scale_factor = new_scale
        
        return new_scale
    
    def generate_report(
        self,
        results: List[WeightEstimationResult],
        image_name: str = "",
    ) -> str:
        """
        Tạo báo cáo text về kết quả ước lượng.
        
        Args:
            results: Danh sách kết quả
            image_name: Tên ảnh (optional)
        
        Returns:
            str: Báo cáo định dạng text
        """
        lines = ["=" * 60]
        if image_name:
            lines.append(f"BÁO CÁO ƯỚC LƯỢNG TRỌNG LƯỢNG: {image_name}")
        else:
            lines.append("BÁO CÁO ƯỚC LƯỢNG TRỌNG LƯỢNG")
        lines.append("=" * 60)
        lines.append("")
        
        total_weight = 0.0
        
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.class_name}")
            lines.append(f"    Thể tích:   {r.volume_cm3:.1f} cm³")
            lines.append(f"    Mật độ:     {r.density_g_per_cm3:.2f} g/cm³ ({r.density_source})")
            lines.append(f"    Trọng lượng: {r.weight_grams:.1f} g")
            lines.append(f"    Độ tin cậy: {r.confidence:.0%}")
            lines.append("")
            total_weight += r.weight_grams
        
        lines.append("-" * 40)
        lines.append(f"TỔNG TRỌNG LƯỢNG ƯỚC LƯỢNG: {total_weight:.1f} g")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ==============================================================================
# VALIDATION UTILITIES
# ==============================================================================

def validate_against_metadata(
    results: List[WeightEstimationResult],
    metadata_row: pd.Series,
) -> Dict:
    """
    So sánh kết quả ước lượng với ground truth từ metadata.
    
    Args:
        results: Danh sách WeightEstimationResult
        metadata_row: Row từ nutritionverse_dish_metadata3.csv
    
    Returns:
        Dict với thông tin validation
    """
    # Extract ground truth weights
    gt_weights = {}
    for i in range(1, 8):  # Tối đa 7 food items
        food_type_col = f"food_item_type_{i}"
        weight_col = f"food_weight_g_{i}"
        
        if food_type_col in metadata_row and pd.notna(metadata_row[food_type_col]):
            food_type = str(metadata_row[food_type_col])
            weight = float(metadata_row[weight_col])
            gt_weights[food_type] = weight
    
    # Match và tính sai số
    validation = {
        "total_gt_weight": metadata_row.get("total_food_weight", 0),
        "total_pred_weight": sum(r.weight_grams for r in results),
        "items": [],
    }
    
    for result in results:
        item_val = {
            "class": result.class_name,
            "predicted_weight": result.weight_grams,
            "ground_truth_weight": gt_weights.get(result.class_name, None),
        }
        
        if item_val["ground_truth_weight"] is not None:
            error = abs(item_val["predicted_weight"] - item_val["ground_truth_weight"])
            rel_error = error / (item_val["ground_truth_weight"] + 1e-8)
            item_val["absolute_error"] = error
            item_val["relative_error"] = rel_error
        
        validation["items"].append(item_val)
    
    # Overall error
    gt_total = validation["total_gt_weight"]
    pred_total = validation["total_pred_weight"]
    
    if gt_total > 0:
        validation["total_absolute_error"] = abs(pred_total - gt_total)
        validation["total_relative_error"] = abs(pred_total - gt_total) / gt_total
    
    return validation


# ==============================================================================
# DEMO / TEST
# ==============================================================================
# Để test module này, sử dụng run_inference.py từ thư mục gốc project
# Không thể chạy trực tiếp file này do sử dụng relative imports
