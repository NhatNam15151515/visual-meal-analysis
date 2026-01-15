from fastapi import APIRouter, UploadFile, File
from api.services.inference_service import run_pipeline

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/")
async def predict_meal(image: UploadFile = File(...)):
    result = await run_pipeline(image)
    return result
