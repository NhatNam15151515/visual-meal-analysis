# Meal Weight Estimation

Hệ thống ước lượng trọng lượng bữa ăn từ ảnh, sử dụng YOLOv8 segmentation và MiDaS depth estimation.

## Cấu trúc

```
├── src/                  # Core modules
│   ├── tier1_segmentation.py   # YOLOv8-seg
│   ├── tier2_depth_volume.py   # MiDaS depth + volume
│   ├── tier3_weight_estimation.py  # Weight calculation
│   └── pipeline.py       # Pipeline tổng hợp
├── api/                  # FastAPI server
├── run_inference.py      # Script inference
├── train_segmentation.py # Script training
└── data/                 # Dataset (không có trong repo)
```

## Cài đặt

```bash
pip install -r requirements.txt
```

GPU (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Sử dụng

### Inference

```bash
# Một ảnh
python run_inference.py --image path/to/image.jpg

# Folder
python run_inference.py --folder path/to/images/ --limit 10
```

### Training

```bash
# Chuẩn bị dataset + train
python train_segmentation.py --prepare --train --epochs 50
```

### API

```bash
python -m uvicorn api.main:app --port 8000
```

POST ảnh đến `http://localhost:8000/predict/`

## Dataset

Download NutritionVerse-Real dataset và đặt vào `data/nutritionverse-manual/`

## Model

Model YOLOv8 đã train: đặt vào `models/yolov8_food_seg3/weights/best.pt`
