# api/schemas/append.py
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Dict, Any, Optional

class AppendData(BaseModel):
    retrain: Optional[bool] = False
    model_name: Optional[str] = "Padding"
    ci_builds: List[Dict[str, Any]]