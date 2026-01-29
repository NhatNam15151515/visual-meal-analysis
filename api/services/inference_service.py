import subprocess
import json
from pathlib import Path

# Sử dụng đường dẫn tuyệt đối
PROJECT_ROOT = Path(__file__).parent.parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
INFERENCE_SCRIPT = PROJECT_ROOT / "run_inference.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inference_results"
FILENAME = "meal"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def save_image(file) -> str:
    """Lưu ảnh upload."""
    ext = file.filename.split(".")[-1]
    path = IMAGE_DIR / f"{FILENAME}.{ext}"
    with open(path, "wb") as f:
        f.write(file.file.read())
    return str(path)

async def run_pipeline(image):
    # 1. Lưu ảnh
    image_path = save_image(image)
    
    # 2. Gọi pipeline
    subprocess.run(
        ["python", str(INFERENCE_SCRIPT), "--image", image_path],
        check=True,
        cwd=str(PROJECT_ROOT)  # Chạy từ thư mục project root
    )

    # 3. Đọc kết quả - file được lưu theo tên ảnh
    result_json = OUTPUT_DIR / f"{FILENAME}_analysis.json"
    with open(result_json, "r", encoding="utf-8") as f:
        return json.load(f)
