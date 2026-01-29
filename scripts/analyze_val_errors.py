
import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.config import TRAINED_MODEL_PATH, DATA_YAML, CLASS_NAMES

def analyze_val_errors():
    model_path = TRAINED_MODEL_PATH
    if not model_path.exists():
        print(f"[ERROR] Trained model not found at {model_path}")
        print(f"[INFO] Please ensuring training has completed or update TRAINED_MODEL_PATH in src/config.py")
        return

    print(f"[INFO] Loading model from {model_path}...")
    model = YOLO(str(model_path))

    print("[INFO] Running validation...")
    # Run validation to get metrics
    metrics = model.val(data=str(DATA_YAML), split='val', verbose=False)
    
    # Access Confusion Matrix
    # Ultralytics stores the confusion matrix in the validator's confusion_matrix attribute
    # which is accessible via metrics.confusion_matrix
    
    try:
        # matrix is (n+1, n+1) where last index is background
        cm = metrics.confusion_matrix.matrix
    except AttributeError:
        print("[ERROR] Could not retrieve confusion matrix from metrics.")
        return

    print(f"[INFO] Analyzing Confusion Matrix (Shape: {cm.shape})...")
    
    results = []
    
    # Ensure we don't go out of bounds if CLASS_NAMES differs from model classes
    # usually cm is N x N or (N+1)x(N+1)
    
    num_classes = len(CLASS_NAMES)
    
    for i, class_name in enumerate(CLASS_NAMES):
        if i >= cm.shape[0]:
            break
            
        # True Positives: Diagonal
        tp = cm[i, i]
        
        # False Negatives: Row sum (Ground Truth) - TP
        # This includes misclassifications + missed detections (background)
        fn = np.sum(cm[i, :]) - tp
        
        # False Positives: Col sum (Predictions) - TP
        # This includes misclassifications + extra detections (background)
        fp = np.sum(cm[:, i]) - tp
        
        total_errors = fn + fp
        
        # Calculate metrics for reference
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        results.append({
            "class_name": class_name,
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "total_errors": int(total_errors),
            "precision": precision,
            "recall": recall
        })

    # Sort results by Total Errors (descending)
    results.sort(key=lambda x: x['total_errors'], reverse=True)

    # Print Summary Table
    print("\n" + "="*95)
    print(f"{'RANK':<5} | {'CLASS NAME':<30} | {'ERRORS':<8} | {'FP':<6} | {'FN':<6} | {'PREC':<6} | {'REC':<6}")
    print("-" * 95)
    
    for rank, r in enumerate(results, 1):
        if r['total_errors'] > 0:
            print(f"{rank:<5} | {r['class_name']:<30} | {r['total_errors']:<8} | {r['fp']:<6} | {r['fn']:<6} | {r['precision']:.2f}   | {r['recall']:.2f}")
    
    print("-" * 95)
    print(f"[INFO] Total Classes: {len(results)}")
    
    # Summary of most problematic
    if results:
        top_err_class = results[0]
        print(f"[SUMMARY] Most problematic class: '{top_err_class['class_name']}' with {top_err_class['total_errors']} errors.")

if __name__ == "__main__":
    analyze_val_errors()
