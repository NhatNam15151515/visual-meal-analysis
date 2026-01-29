"""
tier1_segmentation.py - TẦNG 1: Phát hiện và phân đoạn thực phẩm

Description: 
    Sử dụng YOLOv8-seg để detect và segment các loại thực phẩm trong ảnh bữa ăn.
    Module này bao gồm:
    - Chuẩn bị dữ liệu từ COCO format sang YOLO format
    - Fine-tune model YOLOv8-seg trên NutritionVerse-Real
    - Inference để phát hiện và phân đoạn thực phẩm

Đầu ra:
    - Tên loại thực phẩm (class name)
    - Mask segmentation (binary mask cho từng món)
    - Bounding box [x1, y1, x2, y2]
    - Confidence score
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from .config import (
    DATA_DIR,
    DATA_YAML,
    MODELS_DIR,
    OUTPUT_DIR,
    TRAIN_CONFIG,
    ensure_directories,
    get_device,
)

# Alias cho backward compatibility
TIER1_CONFIG = TRAIN_CONFIG


class FoodSegmentationResult:
    """
    Class lưu trữ kết quả segmentation cho một món ăn.
    
    Attributes:
        class_id (int): ID của class thực phẩm
        class_name (str): Tên loại thực phẩm
        confidence (float): Độ tin cậy của prediction (0-1)
        bbox (np.ndarray): Bounding box [x1, y1, x2, y2]
        mask (np.ndarray): Binary mask (H x W), dtype=uint8, values 0 hoặc 255
        area_pixels (int): Diện tích mask tính theo pixels
    """
    
    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: np.ndarray,
        mask: np.ndarray,
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.mask = mask  # Binary mask
        self.area_pixels = int(np.sum(mask > 0))
    
    def __repr__(self):
        return (
            f"FoodSegmentationResult("
            f"class='{self.class_name}', "
            f"conf={self.confidence:.3f}, "
            f"area={self.area_pixels}px)"
        )


class Tier1FoodSegmentation:
    """
    TẦNG 1: Module phát hiện và phân đoạn thực phẩm sử dụng YOLOv8-seg.
    
    Pipeline:
    1. Load model YOLOv8-seg (pretrained hoặc fine-tuned)
    2. Inference trên ảnh đầu vào
    3. Trả về danh sách FoodSegmentationResult
    
    Các giả định:
    - Ảnh đầu vào là ảnh RGB (BGR từ OpenCV sẽ được convert)
    - Model đã được fine-tune trên NutritionVerse-Real
    - Mỗi instance được detect riêng biệt (instance segmentation)
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict] = None,
    ):
        """
        Khởi tạo module segmentation.
        
        Args:
            model_path: Đường dẫn đến model weights (.pt file).
                       Nếu None, sử dụng trained_model từ config hoặc pretrained.
            config: Dictionary cấu hình. Nếu None, sử dụng TIER1_CONFIG.
        """
        self.config = config or TIER1_CONFIG
        self.device = get_device()
        
        # Load model - ưu tiên: model_path > trained_model > pretrained
        if model_path and Path(model_path).exists():
            print(f"[Tier1] Loading model tu: {model_path}")
            self.model = YOLO(str(model_path))
        elif self.config.get("trained_model") and Path(self.config["trained_model"]).exists():
            trained_path = self.config["trained_model"]
            print(f"[Tier1] Loading trained model tu: {trained_path}")
            self.model = YOLO(str(trained_path))
        else:
            print(f"[Tier1] Loading pretrained model: {self.config['pretrained_weights']}")
            print("[Tier1] WARNING: Dang dung pretrained COCO, chua fine-tune!")
            self.model = YOLO(self.config["pretrained_weights"])
        
        # Lưu class names từ model
        self.class_names = self.model.names
        print(f"[Tier1] Model loaded với {len(self.class_names)} classes")
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[FoodSegmentationResult]:
        """
        Thực hiện inference trên một ảnh.
        
        Args:
            image: Đường dẫn ảnh hoặc numpy array (BGR hoặc RGB)
            conf_threshold: Ngưỡng confidence (override config)
            iou_threshold: Ngưỡng IoU cho NMS (override config)
        
        Returns:
            List[FoodSegmentationResult]: Danh sách kết quả segmentation
        """
        conf = conf_threshold or self.config["conf_threshold"]
        iou = iou_threshold or self.config["iou_threshold"]
        
        # Load ảnh nếu là đường dẫn
        if isinstance(image, (str, Path)):
            image_array = cv2.imread(str(image))
            if image_array is None:
                raise ValueError(f"Không thể đọc ảnh: {image}")
        else:
            image_array = image.copy()
        
        # Lưu kích thước ảnh gốc
        orig_h, orig_w = image_array.shape[:2]
        
        # Inference
        results = self.model.predict(
            source=image_array,
            conf=conf,
            iou=iou,
            max_det=self.config["max_det"],
            device=self.device,
            verbose=False,
        )
        
        # Parse kết quả
        segmentation_results = []
        
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            
            # Lấy masks và boxes
            masks = result.masks.data.cpu().numpy()  # (N, H, W)
            boxes = result.boxes.data.cpu().numpy()  # (N, 6): [x1, y1, x2, y2, conf, cls]
            
            for i in range(len(masks)):
                # Extract thông tin
                bbox = boxes[i, :4]  # [x1, y1, x2, y2]
                confidence = boxes[i, 4]
                class_id = int(boxes[i, 5])
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                
                # Resize mask về kích thước ảnh gốc
                mask = masks[i]  # (H_model, W_model)
                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR
                )
                # Chuyển về binary mask
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                
                seg_result = FoodSegmentationResult(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    mask=binary_mask,
                )
                segmentation_results.append(seg_result)
        
        return segmentation_results
    
    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        results: List[FoodSegmentationResult],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> np.ndarray:
        """
        Visualize kết quả segmentation lên ảnh.
        
        Args:
            image: Ảnh gốc
            results: Danh sách FoodSegmentationResult
            output_path: Đường dẫn lưu ảnh output
            show: Hiển thị ảnh (cv2.imshow)
        
        Returns:
            np.ndarray: Ảnh với visualization
        """
        # Load ảnh
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()
        
        # Tạo overlay cho masks
        overlay = img.copy()
        
        # Color palette (BGR)
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]
        
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            
            # Vẽ mask với transparency
            mask_bool = result.mask > 0
            overlay[mask_bool] = (
                0.5 * np.array(color) + 0.5 * overlay[mask_bool]
            ).astype(np.uint8)
            
            # Vẽ đường viền (contour) bao quanh đối tượng thay vì bbox
            contours, _ = cv2.findContours(result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 2)
            
            # Tính toán vị trí label (dùng đỉnh cao nhất của contour)
            if contours:
                # Tìm điểm cao nhất (y nhỏ nhất)
                c = max(contours, key=cv2.contourArea)
                top_point = tuple(c[c[:, :, 1].argmin()][0])
                label_x, label_y = top_point
                label_y = max(label_y - 10, 20) # Padding
            else:
                # Fallback về bbox nếu lỗi contour
                x1, y1, x2, y2 = result.bbox.astype(int)
                label_x, label_y = x1, y1 - 10

            # Vẽ label
            label = f"{result.class_name}: {result.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            # Vẽ background cho text để dễ đọc
            cv2.rectangle(img, (label_x, label_y - th - 5), (label_x + tw, label_y + 5), color, -1)
            cv2.putText(
                img, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
            )
        
        # Blend overlay
        output = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        if output_path:
            cv2.imwrite(str(output_path), output)
        
        if show:
            cv2.imshow("Segmentation Results", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return output


class COCOToYOLOConverter:
    """
    Chuyển đổi dataset từ COCO format sang YOLO format cho training.
    
    COCO format:
    - annotations.json: {"images": [...], "annotations": [...], "categories": [...]}
    - Segmentation: polygon points [[x1,y1,x2,y2,...], ...]
    
    YOLO format:
    - images/ và labels/ folders
    - Mỗi ảnh có file .txt tương ứng
    - Format: class_id x1 y1 x2 y2 ... (normalized polygon coordinates)
    """
    
    def __init__(self, coco_annotation_path: Union[str, Path]):
        """
        Args:
            coco_annotation_path: Đường dẫn đến file annotation COCO JSON
        """
        self.annotation_path = Path(coco_annotation_path)
        
        print(f"[Converter] Loading COCO annotations từ: {self.annotation_path}")
        with open(self.annotation_path, "r") as f:
            self.coco_data = json.load(f)
        
        # Build lookup tables
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}
        
        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        print(f"[Converter] Loaded {len(self.images)} images, "
              f"{len(self.coco_data['annotations'])} annotations, "
              f"{len(self.categories)} categories")
    
    def convert(
        self,
        output_dir: Union[str, Path],
        splits_csv: Optional[Union[str, Path]] = None,
        images_source_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Chuyển đổi và tổ chức dataset sang YOLO format.
        
        Args:
            output_dir: Thư mục đầu ra
            splits_csv: File CSV chứa train/val splits
            images_source_dir: Thư mục chứa ảnh gốc
        
        Returns:
            Path: Đường dẫn đến file data.yaml
        """
        output_dir = Path(output_dir)
        images_source = Path(images_source_dir) if images_source_dir else self.annotation_path.parent
        
        # Tạo cấu trúc thư mục YOLO
        for split in ["train", "val"]:
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Load splits nếu có
        if splits_csv and Path(splits_csv).exists():
            splits_df = pd.read_csv(splits_csv)
            file_to_split = dict(zip(splits_df["file_name"], splits_df["category"]))
        else:
            # Mặc định 80% train, 20% val
            file_to_split = None
        
        # Tạo mapping category_id -> index liên tục (bỏ qua category 0 và excluded categories)
        # Lọc bỏ các category không phổ biến ở châu Á
        excluded_cat_ids = set()
        for cat in self.coco_data["categories"]:
            if cat["name"] in EXCLUDED_CATEGORIES:
                excluded_cat_ids.add(cat["id"])
                print(f"[Converter] Loại bỏ category: {cat['name']} (id={cat['id']})")
        
        category_ids = sorted([
            c["id"] for c in self.coco_data["categories"] 
            if c["id"] != 0 and c["id"] not in excluded_cat_ids
        ])
        self.cat_id_to_yolo_id = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
        self.excluded_cat_ids = excluded_cat_ids
        
        # Danh sách tên class theo thứ tự YOLO
        class_names = [self.categories[cat_id]["name"] for cat_id in category_ids]
        print(f"[Converter] Số classes sau khi lọc: {len(class_names)}")
        
        # Chuyển đổi từng ảnh
        train_count, val_count = 0, 0
        
        for img_id, img_info in self.images.items():
            filename = img_info["file_name"]
            width = img_info["width"]
            height = img_info["height"]
            
            # Xác định split
            if file_to_split:
                split = file_to_split.get(filename, "Train")
                split = "train" if split.lower() == "train" else "val"
            else:
                split = "train" if np.random.rand() < 0.8 else "val"
            
            if split == "train":
                train_count += 1
            else:
                val_count += 1
            
            # Copy ảnh
            src_image = images_source / filename
            dst_image = output_dir / split / "images" / filename
            if src_image.exists() and not dst_image.exists():
                shutil.copy(src_image, dst_image)
            
            # Tạo label file
            label_filename = Path(filename).stem + ".txt"
            label_path = output_dir / split / "labels" / label_filename
            
            annotations = self.annotations_by_image.get(img_id, [])
            
            with open(label_path, "w") as f:
                for ann in annotations:
                    cat_id = ann["category_id"]
                    
                    # Bỏ qua category 0 (parent class) và excluded categories
                    if cat_id == 0 or cat_id in self.excluded_cat_ids or cat_id not in self.cat_id_to_yolo_id:
                        continue
                    
                    yolo_class_id = self.cat_id_to_yolo_id[cat_id]
                    
                    # Chuyển đổi segmentation polygon
                    # COCO: [[x1,y1,x2,y2,...]] -> YOLO: x1/w y1/h x2/w y2/h ...
                    if "segmentation" in ann and ann["segmentation"]:
                        for polygon in ann["segmentation"]:
                            # Normalize coordinates
                            normalized = []
                            for i in range(0, len(polygon), 2):
                                x = polygon[i] / width
                                y = polygon[i + 1] / height
                                normalized.extend([x, y])
                            
                            # Ghi ra file
                            coords_str = " ".join([f"{c:.6f}" for c in normalized])
                            f.write(f"{yolo_class_id} {coords_str}\n")
        
        print(f"[Converter] Đã chuyển đổi: {train_count} train, {val_count} val images")
        
        # Tạo file data.yaml
        yaml_content = f"""# NutritionVerse-Real dataset converted to YOLO format
# Generated automatically by COCOToYOLOConverter

path: {output_dir.absolute()}
train: train/images
val: val/images

# Number of classes
nc: {len(class_names)}

# Class names
names:
"""
        for i, name in enumerate(class_names):
            yaml_content += f"  {i}: {name}\n"
        
        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        
        print(f"[Converter] Dataset YAML saved to: {yaml_path}")
        return yaml_path


class Tier1Trainer:
    """
    Class để fine-tune YOLOv8-seg trên NutritionVerse-Real dataset.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or TIER1_CONFIG
        self.device = get_device()
        ensure_directories()
    
    def prepare_dataset(self) -> Path:
        """
        Chuẩn bị dataset từ COCO sang YOLO format.
        
        Returns:
            Path: Đường dẫn đến data.yaml
        """
        converter = COCOToYOLOConverter(ANNOTATIONS_FILE)
        yaml_path = converter.convert(
            output_dir=YOLO_DATASET_DIR,
            splits_csv=SPLITS_FILE,
            images_source_dir=IMAGES_DIR,
        )
        return yaml_path
    
    def train(
        self,
        data_yaml: Union[str, Path],
        resume: bool = False,
        name: str = "yolov8_food_seg",
    ) -> str:
        """
        Fine-tune model YOLOv8-seg.
        
        Args:
            data_yaml: Đường dẫn đến file data.yaml
            resume: Tiếp tục training từ checkpoint
            name: Tên project cho training
        
        Returns:
            str: Đường dẫn đến best model weights
        """
        # Load pretrained model
        model = YOLO(self.config["pretrained_weights"])
        
        # Training
        print(f"[Trainer] Bắt đầu training với config:")
        print(f"  - Data: {data_yaml}")
        print(f"  - Epochs: {self.config['epochs']}")
        print(f"  - Batch size: {self.config['batch_size']}")
        print(f"  - Image size: {self.config['imgsz']}")
        print(f"  - Device: {self.device}")
        print(f"  - Project name: {name}")
        
        results = model.train(
            data=str(data_yaml),
            epochs=self.config["epochs"],
            batch=self.config["batch_size"],
            imgsz=self.config["imgsz"],
            device=self.device,
            project=str(MODELS_DIR),
            name=name,
            patience=self.config["patience"],
            lr0=self.config["learning_rate"],
            resume=resume,
            plots=True,
            save=True,
            workers=self.config.get("workers", 8),  # Tận dụng CPU
            # Data Augmentation
            mosaic=self.config.get("mosaic", 1.0),
            mixup=self.config.get("mixup", 0.0),
            copy_paste=self.config.get("copy_paste", 0.0),
            scale=self.config.get("scale", 0.5),
            fliplr=self.config.get("fliplr", 0.5),
            hsv_h=self.config.get("hsv_h", 0.015),
            hsv_s=self.config.get("hsv_s", 0.7),
            hsv_v=self.config.get("hsv_v", 0.4),
            degrees=self.config.get("degrees", 0.0),
            translate=self.config.get("translate", 0.1),
            close_mosaic=self.config.get("close_mosaic", 10),
        )
        
        # Tìm best model
        best_model_path = MODELS_DIR / name / "weights" / "best.pt"
        print(f"[Trainer] Training hoàn thành!")
        print(f"[Trainer] Best model: {best_model_path}")
        
        return str(best_model_path)


# ==============================================================================
# DEMO / TEST
# ==============================================================================
# Để test module này, sử dụng run_inference.py hoặc train_segmentation.py 
# từ thư mục gốc project. Không thể chạy trực tiếp file này do sử dụng 
# relative imports.
