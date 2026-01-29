
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import shutil
from ultralytics import YOLO
from torchvision.ops import box_iou

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.config import TRAINED_MODEL_PATH, DATA_YAML, CLASS_NAMES

# If CLASSES_OF_INTEREST is not in config, define it here temporarily or import error
# We will define a local list based on previous analysis
TARGET_CLASSES = [31, 14, 0, 9, 33] # green salad, miso soup, rice, ramen noodle, toast

def visualize_errors():
    model_path = TRAINED_MODEL_PATH
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        return

    print(f"[INFO] Loading model from {model_path}...")
    model = YOLO(str(model_path))
    
    # Validation images dir
    val_img_dir = ROOT / "data" / "uecfoodpix_yolo" / "val" / "images"
    val_label_dir = ROOT / "data" / "uecfoodpix_yolo" / "val" / "labels"
    
    output_dir = ROOT / "outputs" / "error_vis"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] outputting to {output_dir}")

    # Get all images
    img_files = list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png"))
    
    print(f"[INFO] Processing {len(img_files)} validation images...")
    
    for img_path in img_files:
        # Load GT
        label_path = val_label_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        gt_classes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    cls = int(parts[0])
                    # YOLO format: cls x c y w h ... (we need boxes for IoU)
                    # Use ultralytics utils to convert if needed, or just manual
                    # But for simple visualization, we rely on model prediction vs GT matching
                    
                    # Store normalized center-wh
                    gt_classes.append(cls)
                    gt_boxes.append([float(x) for x in parts[1:5]])
                    
        gt_boxes = torch.tensor(gt_boxes) if gt_boxes else torch.zeros((0, 4))
        gt_classes = torch.tensor(gt_classes) if gt_classes else torch.zeros((0))
        
        # Run Inference
        results = model(str(img_path), verbose=False, iou=0.5, conf=0.25)[0]
        
        # Get Predictions
        pred_boxes = results.boxes.xywhn.cpu() # Normalized xywh
        pred_cls = results.boxes.cls.cpu().int()
        
        # Convert GT keys to xyxy for IoU check? No, box_iou expects xyxy usually.
        # Let's convert normalized xywh to xyxy for both
        if len(gt_boxes) > 0:
            gt_xyxy = xywhn2xyxy(gt_boxes)
        else:
            gt_xyxy = torch.zeros((0, 4))
            
        pred_xyxy = results.boxes.xyxyn.cpu()
        
        # Match Predictions to GT
        matches = match_predictions(pred_xyxy, pred_cls, gt_xyxy, gt_classes)
        
        # Check for FP and FN in TARGET_CLASSES
        has_issue = False
        
        # Draw Image
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Draw GT (Green)
        for i, box in enumerate(gt_xyxy):
            cls = int(gt_classes[i])
            if cls not in TARGET_CLASSES: continue
            
            # Check if this GT was matched
            is_matched = i in matches['gt_matches']
            
            if not is_matched:
                # FALSE NEGATIVE (Missed)
                has_issue = True
                draw_box(img, box, cls, (0, 255, 0), "FN", w, h)
                save_issue(img, output_dir, cls, "FN", img_path.name)

        # Draw Pred (Red if wrong)
        for i, box in enumerate(pred_xyxy):
            cls = int(pred_cls[i])
            if cls not in TARGET_CLASSES: continue
            
            # Check match
            is_matched = i in matches['pred_matches']
            
            if not is_matched:
                # FALSE POSITIVE (Wrong detection)
                has_issue = True
                draw_box(img, box, cls, (0, 0, 255), "FP", w, h)
                save_issue(img, output_dir, cls, "FP", img_path.name)
            else:
                 # True Positive (Optional: Draw Blue or ignore)
                 # draw_box(img, box, cls, (255, 0, 0), "TP", w, h)
                 pass

def xywhn2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thres=0.5):
    """
    Returns indices of matched predictions and matched GTs.
    """
    matches = {'pred_matches': [], 'gt_matches': []}
    
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matches
        
    # Calculate IoU matrix
    iou = box_iou(pred_boxes, gt_boxes) # (N_pred, N_gt)
    
    # Greedy matching
    # We want to match same-class only? Yes strictly.
    
    for i, p_cls in enumerate(pred_cls):
        best_iou = 0
        best_gt_idx = -1
        
        for j, g_cls in enumerate(gt_cls):
            if p_cls != g_cls: continue
            if j in matches['gt_matches']: continue
            
            curr_iou = iou[i, j]
            if curr_iou > iou_thres and curr_iou > best_iou:
                best_iou = curr_iou
                best_gt_idx = j
                
        if best_gt_idx != -1:
            matches['pred_matches'].append(i)
            matches['gt_matches'].append(best_gt_idx)
            
    return matches

def draw_box(img, box, cls, color, tag, w, h):
    x1, y1, x2, y2 = int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{tag}: {CLASS_NAMES.get(cls, cls)}"
    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def save_issue(img, root, cls, type, name):
    cls_name = CLASS_NAMES.get(cls, str(cls)).replace(" ", "_")
    d = root / cls_name
    d.mkdir(exist_ok=True)
    out_path = d / f"{type}_{name}"
    # Only save if not exists to avoid overwrite spam or just overwrite matches?
    # We might draw multiple boxes on one image. 
    # Current logic: 'save_issue' writes the CURRENT state of 'img' to disk.
    # If an image has multiple errors, it will be saved multiple times or overwrite.
    # Better: just write once at the loop?
    # Simplification: We write it every time an issue is found. 
    # Since we modify 'img' in place, the last save will have all annotations.
    cv2.imwrite(str(out_path), img)

if __name__ == "__main__":
    visualize_errors()
