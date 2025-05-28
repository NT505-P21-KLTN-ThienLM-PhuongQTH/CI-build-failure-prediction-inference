# api/schemas/retrain.py
from pydantic import BaseModel

class RetrainRequest(BaseModel):
    model_name: str