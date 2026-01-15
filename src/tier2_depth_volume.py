"""
tier2_depth_volume.py - TẦNG 2: Ước lượng độ sâu và thể tích

Description:
    Sử dụng MiDaS để sinh depth map từ ảnh RGB đơn,
    sau đó kết hợp với mask segmentation để ước lượng thể tích tương đối.

Phương pháp:
    1. Sinh depth map từ ảnh RGB sử dụng MiDaS (monocular depth estimation)
    2. Apply mask để lấy vùng depth của từng món ăn
    3. Ước lượng thể tích dựa trên:
       - Diện tích pixel (area)
       - Giá trị depth trung bình trong mask
       - Các giả định hình học

Các giả định quan trọng (cần chú ý khi sử dụng):
    1. MiDaS output là RELATIVE depth, không phải metric depth
    2. Giả định camera nhìn từ trên xuống (top-down view) hoặc góc nghiêng nhẹ
    3. Giả định mặt bàn là mặt phẳng tham chiếu
    4. Thể tích ước lượng mang tính TƯƠNG ĐỐI, cần calibrate với ground truth
    5. Công thức thể tích đơn giản hóa: V ∝ Area × Height
       (xem như hình trụ với đáy là mask và chiều cao từ depth)

Đầu ra:
    - Depth map toàn ảnh
    - Depth map cho từng mask
    - Thể tích ước lượng (đơn vị tương đối hoặc cm³ sau calibration)
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import OUTPUT_DIR, TIER2_CONFIG, ensure_directories, get_device


class DepthVolumeResult:
    """
    Class lưu trữ kết quả ước lượng depth và volume cho một món ăn.
    
    Attributes:
        class_name (str): Tên loại thực phẩm
        depth_map (np.ndarray): Depth map của vùng mask (H x W), giá trị float
        mean_depth (float): Giá trị depth trung bình trong mask
        std_depth (float): Độ lệch chuẩn depth trong mask
        min_depth (float): Giá trị depth nhỏ nhất (cao nhất trong không gian thực)
        max_depth (float): Giá trị depth lớn nhất (thấp nhất trong không gian thực)
        area_pixels (int): Diện tích mask tính theo pixels
        area_cm2 (float): Diện tích ước lượng (cm²) - cần calibration
        height_relative (float): Chiều cao tương đối (max_depth - min_depth trong mask)
        volume_relative (float): Thể tích tương đối (đơn vị tùy ý)
        volume_cm3 (float): Thể tích ước lượng (cm³) - cần calibration
    """
    
    def __init__(
        self,
        class_name: str,
        depth_map: np.ndarray,
        mask: np.ndarray,
        config: Dict,
    ):
        self.class_name = class_name
        self.mask = mask
        
        # Extract depth values trong mask
        mask_bool = mask > 0
        depth_values = depth_map[mask_bool]
        
        # Lưu depth map (masked)
        self.depth_map = np.where(mask_bool, depth_map, 0).astype(np.float32)
        
        # Tính toán statistics
        if len(depth_values) > 0:
            self.mean_depth = float(np.mean(depth_values))
            self.std_depth = float(np.std(depth_values))
            self.min_depth = float(np.min(depth_values))
            self.max_depth = float(np.max(depth_values))
        else:
            self.mean_depth = self.std_depth = 0.0
            self.min_depth = self.max_depth = 0.0
        
        # Diện tích
        self.area_pixels = int(np.sum(mask_bool))
        
        # Chiều cao tương đối từ depth variance trong mask
        # Giả định: các giá trị depth khác nhau trong mask phản ánh chiều cao món ăn
        self.height_relative = self.max_depth - self.min_depth
        
        # ==================================================================
        # ƯỚC LƯỢNG THỂ TÍCH
        # ==================================================================
        # 
        # Giả định hình học (Geometric Assumptions):
        # 1. Camera projection: pinhole model với focal length f
        # 2. Depth map từ MiDaS là INVERSE depth (gần -> giá trị lớn)
        # 3. Món ăn xem như hình trụ với:
        #    - Đáy có diện tích = area_pixels (trong image space)
        #    - Chiều cao tương ứng với độ biến thiên depth
        #
        # Công thức chuyển đổi:
        # - Diện tích thực (cm²) ≈ area_pixels × (Z / f)²
        #   với Z là khoảng cách thực và f là focal length
        # - Do MiDaS output relative depth, ta cần reference scale
        #
        # Công thức đơn giản hóa:
        # V_relative = area_pixels × mean_depth × height_factor
        # V_cm3 = V_relative × calibration_factor
        # ==================================================================
        
        # Tính diện tích ước lượng
        # Giả định: reference_distance_cm là khoảng cách trung bình camera-bàn ăn
        focal = config.get("focal_length_px", 1000.0)
        ref_dist = config.get("reference_distance_cm", 40.0)
        
        # Scale factor: pixels to cm tại khoảng cách tham chiếu
        # Với sensor size ~6mm và image width 640px:
        # pixel_size_at_ref = ref_dist * sensor_width / (focal * image_width)
        # Đơn giản hóa: 1 pixel ≈ ref_dist / focal (cm)
        pixel_to_cm = ref_dist / focal
        self.area_cm2 = self.area_pixels * (pixel_to_cm ** 2)
        
        # Thể tích tương đối
        # Công thức: V ∝ A × H
        # Với H là chiều cao ước lượng từ depth variation
        #
        # height_cm ước lượng từ depth range và scale factor
        # MiDaS inverse depth: depth_value ∝ 1/distance
        # Nên height ∝ (1/min_depth - 1/max_depth) * scale
        
        if self.min_depth > 0.01 and self.max_depth > self.min_depth:
            # Ước lượng chiều cao từ inverse depth
            # height_cm ≈ scale * (1/d_min - 1/d_max)
            depth_scale = config.get("depth_scale_factor", 1.0)
            self.height_cm = depth_scale * (1.0 / self.min_depth - 1.0 / self.max_depth)
            self.height_cm = max(0.5, min(self.height_cm, 20.0))  # Clamp hợp lý: 0.5-20cm
        else:
            # Fallback: ước lượng từ mean depth và area
            # Giả định chiều cao trung bình là 2-3cm cho món ăn
            self.height_cm = 2.5  # cm
        
        # Thể tích tương đối (không có đơn vị cụ thể)
        self.volume_relative = self.area_pixels * self.mean_depth * (1 + self.height_relative)
        
        # Thể tích ước lượng (cm³)
        # V = A × H (hình trụ đơn giản)
        self.volume_cm3 = self.area_cm2 * self.height_cm
    
    def __repr__(self):
        return (
            f"DepthVolumeResult("
            f"class='{self.class_name}', "
            f"area={self.area_cm2:.1f}cm², "
            f"height={self.height_cm:.1f}cm, "
            f"volume={self.volume_cm3:.1f}cm³)"
        )


class Tier2DepthVolume:
    """
    TẦNG 2: Module ước lượng độ sâu và thể tích sử dụng MiDaS.
    
    Pipeline:
    1. Load model MiDaS (pretrained)
    2. Sinh depth map từ ảnh RGB
    3. Kết hợp với masks từ Tầng 1
    4. Ước lượng thể tích cho từng món ăn
    
    MiDaS Output:
    - Depth map là INVERSE relative depth
    - Giá trị cao hơn = gần camera hơn
    - Giá trị thấp hơn = xa camera hơn
    - Không có đơn vị metric, cần calibration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Khởi tạo module depth estimation.
        
        Args:
            config: Dictionary cấu hình. Nếu None, sử dụng TIER2_CONFIG.
        """
        self.config = config or TIER2_CONFIG
        self.device = get_device()
        
        # Load MiDaS model
        print(f"[Tier2] Loading MiDaS model: {self.config['model_type']}")
        self._load_midas()
        print(f"[Tier2] MiDaS loaded trên device: {self.device}")
    
    def _load_midas(self):
        """Load MiDaS model và transform."""
        model_type = self.config["model_type"]
        
        # Load model từ torch hub
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.midas.to(self.device)
        self.midas.eval()
        
        # Load transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    @torch.no_grad()
    def estimate_depth(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> np.ndarray:
        """
        Sinh depth map từ ảnh RGB.
        
        Args:
            image: Ảnh RGB (HxWx3) hoặc đường dẫn file
        
        Returns:
            np.ndarray: Depth map (HxW), dtype=float32
                       Giá trị cao = gần, giá trị thấp = xa
        """
        # Load ảnh
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Không thể đọc ảnh: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
        
        orig_h, orig_w = img.shape[:2]
        
        # Transform
        input_batch = self.transform(img).to(self.device)
        
        # Inference
        prediction = self.midas(input_batch)
        
        # Resize về kích thước gốc
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map về range [0, 1] để dễ xử lý
        # Giữ nguyên tính chất: giá trị cao = gần
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-8:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map.astype(np.float32)
    
    def estimate_volume(
        self,
        image: Union[str, Path, np.ndarray],
        masks: List[Tuple[str, np.ndarray]],
        depth_map: Optional[np.ndarray] = None,
    ) -> List[DepthVolumeResult]:
        """
        Ước lượng thể tích cho từng món ăn dựa trên mask và depth.
        
        Args:
            image: Ảnh RGB
            masks: List of (class_name, binary_mask)
            depth_map: Depth map đã tính sẵn (optional)
        
        Returns:
            List[DepthVolumeResult]: Kết quả cho từng món
        """
        # Sinh depth map nếu chưa có
        if depth_map is None:
            depth_map = self.estimate_depth(image)
        
        results = []
        for class_name, mask in masks:
            result = DepthVolumeResult(
                class_name=class_name,
                depth_map=depth_map,
                mask=mask,
                config=self.config,
            )
            results.append(result)
        
        return results
    
    def visualize_depth(
        self,
        depth_map: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
        colormap: int = cv2.COLORMAP_MAGMA,
    ) -> np.ndarray:
        """
        Visualize depth map với colormap.
        
        Args:
            depth_map: Depth map (HxW)
            output_path: Đường dẫn lưu output
            colormap: OpenCV colormap
        
        Returns:
            np.ndarray: Depth visualization (HxWx3)
        """
        # Normalize về 0-255
        depth_normalized = (depth_map * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, colormap)
        
        if output_path:
            cv2.imwrite(str(output_path), depth_colored)
        
        return depth_colored
    
    def visualize_volume(
        self,
        image: Union[str, Path, np.ndarray],
        results: List[DepthVolumeResult],
        depth_map: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Visualize kết quả volume estimation.
        
        Args:
            image: Ảnh gốc
            results: Danh sách DepthVolumeResult
            depth_map: Depth map
            output_path: Đường dẫn lưu output
        
        Returns:
            np.ndarray: Visualization image
        """
        # Load ảnh
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()
        
        h, w = img.shape[:2]
        
        # Tạo canvas
        depth_vis = self.visualize_depth(depth_map)
        
        # Resize depth_vis nếu cần
        if depth_vis.shape[:2] != (h, w):
            depth_vis = cv2.resize(depth_vis, (w, h))
        
        # Ghép ảnh gốc và depth
        combined = np.hstack([img, depth_vis])
        
        # Thêm text info cho từng món
        y_offset = 30
        for result in results:
            text = f"{result.class_name}: V={result.volume_cm3:.1f}cm³"
            cv2.putText(
                combined, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y_offset += 25
        
        if output_path:
            cv2.imwrite(str(output_path), combined)
        
        return combined


# ==============================================================================
# CALIBRATION UTILITIES
# ==============================================================================

def calibrate_depth_scale(
    measured_heights: List[float],
    estimated_heights: List[float],
) -> float:
    """
    Tính hệ số scale từ ground truth measurements.
    
    Args:
        measured_heights: Chiều cao thực đo (cm)
        estimated_heights: Chiều cao ước lượng từ model
    
    Returns:
        float: Scale factor để nhân với estimated values
    """
    if len(measured_heights) != len(estimated_heights):
        raise ValueError("Số lượng measurements không khớp")
    
    measured = np.array(measured_heights)
    estimated = np.array(estimated_heights)
    
    # Least squares fit: measured = scale * estimated
    # scale = sum(measured * estimated) / sum(estimated^2)
    scale = np.sum(measured * estimated) / (np.sum(estimated ** 2) + 1e-8)
    
    return float(scale)


def calibrate_volume_scale(
    measured_volumes: List[float],
    estimated_volumes: List[float],
) -> float:
    """
    Tính hệ số scale cho volume từ ground truth.
    
    Args:
        measured_volumes: Thể tích thực (cm³)
        estimated_volumes: Thể tích ước lượng
    
    Returns:
        float: Scale factor
    """
    return calibrate_depth_scale(measured_volumes, estimated_volumes)


# ==============================================================================
# DEMO / TEST
# ==============================================================================
# Để test module này, sử dụng run_inference.py từ thư mục gốc project
# Không thể chạy trực tiếp file này do sử dụng relative imports
