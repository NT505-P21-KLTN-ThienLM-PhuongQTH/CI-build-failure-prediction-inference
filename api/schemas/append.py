# api/schemas/append.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AppendData(BaseModel):
    retrain: Optional[bool] = False
    ci_builds: List[Dict[str, Any]]