# api/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PredictData(BaseModel):
    """
    Schema for input data to the prediction endpoint.
    Expects a dictionary of feature names and values.
    """
    model: Optional[str] = "Stacked-LSTM"
    version: Optional[str] = "latest"
    ci_builds: List[Dict[str, Any]]