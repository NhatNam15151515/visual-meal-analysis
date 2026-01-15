import subprocess
import json
import os

IMAGE_DIR = "data/images"
INFERENCE_SCRIPT = "run_inference.py"
FILENAME = "meal"

os.makedirs(IMAGE_DIR, exist_ok=True)

def save_image(file) -> str:
    """Lưu ảnh upload."""
    ext = file.filename.split(".")[-1]
    path = os.path.join(IMAGE_DIR, f"{FILENAME}.{ext}")
    with open(path, "wb") as f:
        f.write(file.file.read())
    return path

async def run_pipeline(image):
    # 1. Lưu ảnh
    image_path = save_image(image)
    
    # 2. Gọi pipeline
    RESULT_JSON = f"outputs/inference_results/{FILENAME}_analysis.json"
    subprocess.run(["python", INFERENCE_SCRIPT, "--image", image_path], check=True)

    # 3. Đọc kết quả
    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)
