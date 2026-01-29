import yaml
from pathlib import Path
from collections import Counter
import glob

def analyze_class_balance(data_yaml_path, labels_dir):
    # Load class names
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    names = data.get('names', {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    
    # keys in yaml can be integers or strings, ensure they are ints for mapping
    names = {int(k): v for k, v in names.items()}

    # Get all label files
    label_files = glob.glob(str(Path(labels_dir) / "*.txt"))
    
    print(f"Found {len(label_files)} label files in {labels_dir}")
    
    class_counts = Counter()
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                except (ValueError, IndexError):
                    pass
    
    # Prepare report
    print("\n| Class ID | Class Name | Count |")
    print("|---|---|---|")
    
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    
    # Also include classes with 0 counts
    all_classes = set(names.keys())
    present_classes = set(class_counts.keys())
    missing_classes = all_classes - present_classes
    
    for cls_id in missing_classes:
        sorted_counts.append((cls_id, 0))
    
    # Re-sort to include 0 counts correctly if we want them at the top
    sorted_counts.sort(key=lambda x: x[1])

    for class_id, count in sorted_counts:
        class_name = names.get(class_id, f"Unknown-{class_id}")
        print(f"| {class_id} | {class_name} | {count} |")

if __name__ == "__main__":
    base_dir = Path(r"c:\Nhat Nam\do an chuyen nganh\visual-meal-analysis\data\uecfoodpix_yolo_expanded")
    data_yaml = base_dir / "data.yaml"
    train_labels = base_dir / "train/labels"
    
    analyze_class_balance(data_yaml, train_labels)
