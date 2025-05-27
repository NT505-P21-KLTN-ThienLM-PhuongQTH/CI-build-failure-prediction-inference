# api/schemas/predict.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PredictData(BaseModel):
    """
    Schema for input data to the prediction endpoint.
    Expects a dictionary of feature names and values.
    """
    predict_name: Optional[str] = "Stacked-LSTM"
    predict_version: Optional[int] = None
    padding_name: Optional[str] = None
    padding_version: Optional[int] = None
    ci_builds: List[Dict[str, Any]]