# api/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any

class InputData(BaseModel):
    """
    Schema for input data to the prediction endpoint.
    Expects a dictionary of feature names and values.
    """
    model: str
    status: str
    ci_builds: List[Dict[str, Any]]