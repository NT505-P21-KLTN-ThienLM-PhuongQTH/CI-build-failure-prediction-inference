# api/routes/predict.py
from fastapi import APIRouter, HTTPException
import pandas as pd
import logging
import time
from api.schemas.predict import PredictData
from src.pipeline.prediction import LSTMPredictionPipeline
logger = logging.getLogger("api.predict")
router = APIRouter()
@router.post("/predict")
async def predict_build_failure(data: PredictData):
    """
    Dự đoán khả năng thất bại của build dựa trên dữ liệu đầu vào.
    """
    start_time = time.perf_counter()
    try:
        pipeline = LSTMPredictionPipeline(
            predict_name=data.predict_name,
            predict_version=data.predict_version,
            padding_name=data.padding_name,
            padding_version=data.padding_version
        )
        predictions, threshold, predict_name, predict_version = pipeline.predict_build(data.ci_builds)

        # Xử lý kết quả dự đoán
        prediction_result = predictions[0]
        probability = float(prediction_result[0])
        build_failed = bool(probability > threshold)
        duration = time.perf_counter() - start_time
        return {
            "model_name": predict_name,
            "model_version": predict_version,
            "build_failed": build_failed,
            "probability": probability,
            "threshold": threshold,
            "timestamp": pd.Timestamp.now().isoformat(),
            "execution_time": duration
        }

    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")