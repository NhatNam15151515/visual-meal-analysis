"""
confusion_callback.py - Custom Callback để phân tích confusion matrix

FIXED VERSION:
- Lấy confusion matrix đúng cách từ validator
- Xử lý cả detection và segmentation
- Log chi tiết hơn với percentages
- Debugging info để track issues
"""

from pathlib import Path
from datetime import datetime
import numpy as np


# Majority classes (class IDs có >500 samples)
MAJORITY_CLASS_IDS = [0, 22, 44]  # rice, miso soup, green salad


class ConfusionAnalysisCallback:
    """Callback để phân tích confusion matrix mỗi N epochs."""

    def __init__(self, class_names, log_dir, log_every=10, top_k=10):
        """
        Args:
            class_names: Dict mapping class_id -> class_name
            log_dir: Thư mục lưu log
            log_every: Phân tích mỗi N epochs
            top_k: Số lượng top confusions để log
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.top_k = top_k
        self.log_file = self.log_dir / "confusion_analysis.log"

        # Khởi tạo file log
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Confusion Analysis Log\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Number of classes: {self.num_classes}\n")
            f.write("=" * 80 + "\n\n")

    def _log(self, message, also_print=True):
        """Ghi log vào file và console."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
        if also_print:
            print(message)

    def _get_class_name(self, class_id):
        """Get class name with fallback"""
        if isinstance(class_id, (int, np.integer)):
            return self.class_names.get(int(class_id), f"class_{class_id}")
        return str(class_id)

    def analyze(self, confusion_matrix, epoch):
        """
        Phân tích confusion matrix.

        Args:
            confusion_matrix: np.array shape (num_classes+1, num_classes+1)
                             Row = True label, Col = Predicted label
                             Last row/col = background
            epoch: Epoch hiện tại
        """
        if confusion_matrix is None:
            self._log(f"[WARN] Epoch {epoch}: Confusion matrix is None")
            return

        try:
            cm = np.array(confusion_matrix, dtype=np.float64)

            # YOLOv8 confusion matrix có thêm background class (row/col cuối)
            # Shape thường là (num_classes+1, num_classes+1)
            actual_shape = cm.shape

            self._log(f"\n{'='*80}")
            self._log(f"📊 CONFUSION MATRIX ANALYSIS - Epoch {epoch}")
            self._log(f"{'='*80}")
            self._log(f"Matrix shape: {actual_shape}")
            self._log(f"Expected classes: {self.num_classes}")

            # Bỏ background class (row/col cuối) nếu có
            if cm.shape[0] > self.num_classes:
                cm = cm[:self.num_classes, :self.num_classes]
                self._log(f"Removed background class, new shape: {cm.shape}")

            num_classes = min(cm.shape[0], self.num_classes)
            total_predictions = cm.sum()

            if total_predictions == 0:
                self._log(f"[WARN] No predictions in confusion matrix")
                return

            self._log(f"Total predictions: {int(total_predictions)}")

            # Calculate per-class accuracy
            self._log(f"\n{'─'*80}")
            self._log(f"📈 Per-Class Metrics:")
            self._log(f"{'─'*80}")

            class_metrics = []
            for i in range(num_classes):
                tp = cm[i, i]
                total_true = cm[i, :].sum()  # Row sum = total true instances
                total_pred = cm[:, i].sum()  # Col sum = total predictions

                precision = tp / total_pred if total_pred > 0 else 0
                recall = tp / total_true if total_true > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                class_metrics.append({
                    'id': i,
                    'name': self._get_class_name(i),
                    'tp': int(tp),
                    'total_true': int(total_true),
                    'total_pred': int(total_pred),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

            # Sort by F1 (worst first)
            class_metrics.sort(key=lambda x: x['f1'])

            self._log(f"\n🔴 Worst 5 Classes (by F1-score):")
            for m in class_metrics[:5]:
                self._log(f"   {m['name']:30s} | "
                         f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} | "
                         f"TP={m['tp']:4d} True={m['total_true']:4d} Pred={m['total_pred']:4d}")

            self._log(f"\n🟢 Best 5 Classes (by F1-score):")
            for m in class_metrics[-5:][::-1]:
                self._log(f"   {m['name']:30s} | "
                         f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} | "
                         f"TP={m['tp']:4d} True={m['total_true']:4d} Pred={m['total_pred']:4d}")

            # 1. Top-K confusions (off-diagonal errors)
            self._log(f"\n{'─'*80}")
            self._log(f"❌ Top-{self.top_k} Confusions (Most Frequent Errors):")
            self._log(f"{'─'*80}")

            confusions = []
            for i in range(num_classes):
                for j in range(num_classes):
                    if i != j and cm[i, j] > 0:
                        count = int(cm[i, j])
                        percent = (cm[i, j] / cm[i, :].sum() * 100) if cm[i, :].sum() > 0 else 0
                        confusions.append((i, j, count, percent))

            if not confusions:
                self._log("   No confusions detected (perfect predictions!)")
            else:
                confusions.sort(key=lambda x: x[2], reverse=True)

                for true_id, pred_id, count, percent in confusions[:self.top_k]:
                    true_name = self._get_class_name(true_id)
                    pred_name = self._get_class_name(pred_id)
                    self._log(f"   {true_name:25s} → {pred_name:25s} | "
                             f"{count:4d} times ({percent:5.1f}% of true class)")

            # 2. Majority class swallowing minority
            self._log(f"\n{'─'*80}")
            self._log(f"🔵 Minority Classes Swallowed by Majority:")
            self._log(f"{'─'*80}")

            swallowed = []
            for i in range(num_classes):
                if i in MAJORITY_CLASS_IDS:
                    continue
                for maj_id in MAJORITY_CLASS_IDS:
                    if maj_id < num_classes and cm[i, maj_id] > 0:
                        count = int(cm[i, maj_id])
                        percent = (cm[i, maj_id] / cm[i, :].sum() * 100) if cm[i, :].sum() > 0 else 0
                        swallowed.append((i, maj_id, count, percent))

            if not swallowed:
                self._log("   No minority swallowing detected")
            else:
                swallowed.sort(key=lambda x: x[2], reverse=True)

                for true_id, maj_id, count, percent in swallowed[:self.top_k]:
                    true_name = self._get_class_name(true_id)
                    maj_name = self._get_class_name(maj_id)
                    self._log(f"   {true_name:25s} → {maj_name:25s} | "
                             f"{count:4d} times ({percent:5.1f}% of true class)")

            # 3. False Positives
            self._log(f"\n{'─'*80}")
            self._log(f"⚠️  Top-{self.top_k} Classes with Most False Positives:")
            self._log(f"{'─'*80}")

            fp_counts = []
            for j in range(num_classes):
                col_sum = cm[:, j].sum()
                tp = cm[j, j]
                fp = col_sum - tp
                if fp > 0:
                    fp_rate = (fp / col_sum * 100) if col_sum > 0 else 0
                    fp_counts.append((j, int(fp), int(tp), fp_rate))

            if not fp_counts:
                self._log("   No false positives detected")
            else:
                fp_counts.sort(key=lambda x: x[1], reverse=True)

                for class_id, fp, tp, fp_rate in fp_counts[:self.top_k]:
                    class_name = self._get_class_name(class_id)
                    self._log(f"   {class_name:30s} | FP={fp:4d} TP={tp:4d} | "
                             f"FP rate: {fp_rate:5.1f}%")

            # 4. Summary statistics
            self._log(f"\n{'─'*80}")
            self._log(f"📊 Summary Statistics:")
            self._log(f"{'─'*80}")

            # Overall accuracy
            total_correct = np.trace(cm)
            accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0

            # Macro average
            avg_precision = np.mean([m['precision'] for m in class_metrics])
            avg_recall = np.mean([m['recall'] for m in class_metrics])
            avg_f1 = np.mean([m['f1'] for m in class_metrics])

            self._log(f"   Overall Accuracy: {accuracy:.2f}%")
            self._log(f"   Macro Precision:  {avg_precision:.3f}")
            self._log(f"   Macro Recall:     {avg_recall:.3f}")
            self._log(f"   Macro F1-Score:   {avg_f1:.3f}")

            self._log(f"\n{'='*80}\n")

        except Exception as e:
            self._log(f"[ERROR] Failed to analyze confusion matrix: {e}")
            import traceback
            self._log(traceback.format_exc())


def create_confusion_callback(class_names, log_dir, log_every=10, top_k=10):
    """
    Tạo callback functions cho YOLO trainer.

    Args:
        class_names: Dict mapping class_id -> class_name
        log_dir: Directory to save logs
        log_every: Analyze every N epochs
        top_k: Number of top confusions to log

    Returns:
        Dict chứa callback functions
    """
    analyzer = ConfusionAnalysisCallback(class_names, log_dir, log_every, top_k)

    def on_fit_epoch_end(trainer):
        """Callback được gọi sau mỗi epoch."""
        epoch = trainer.epoch + 1

        # Only analyze on specified intervals
        if epoch % analyzer.log_every != 0:
            return

        # Try to get confusion matrix from validator
        try:
            # Check if validator exists and has run
            if not hasattr(trainer, 'validator') or trainer.validator is None:
                print(f"[DEBUG] Epoch {epoch}: No validator found")
                return

            validator = trainer.validator

            # Check for confusion_matrix object
            if not hasattr(validator, 'confusion_matrix'):
                print(f"[DEBUG] Epoch {epoch}: No confusion_matrix in validator")
                return

            cm_obj = validator.confusion_matrix

            # Get the actual matrix
            if cm_obj is None:
                print(f"[DEBUG] Epoch {epoch}: confusion_matrix object is None")
                return

            # YOLOv8's ConfusionMatrix has .matrix attribute
            if hasattr(cm_obj, 'matrix'):
                cm = cm_obj.matrix
                if cm is not None and cm.size > 0:
                    analyzer.analyze(cm, epoch)
                else:
                    print(f"[DEBUG] Epoch {epoch}: matrix is empty or None")
            else:
                print(f"[DEBUG] Epoch {epoch}: No .matrix attribute in confusion_matrix")

        except Exception as e:
            print(f"[ERROR] Epoch {epoch}: Failed to analyze confusion matrix: {e}")
            import traceback
            traceback.print_exc()

    # Return callback dict
    return {
        'on_fit_epoch_end': on_fit_epoch_end
    }