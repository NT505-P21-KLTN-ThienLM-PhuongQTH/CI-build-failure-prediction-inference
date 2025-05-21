# main.py
import os
import logging.config
import yaml
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, predict, append

# Load .env và config
load_dotenv()

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

with open("config/logging.yaml", "r") as f:
    logging_config = yaml.safe_load(f)
    logging.config.dictConfig(logging_config)

logger = logging.getLogger(__name__)


# Setup MLflow
import mlflow
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Create FastAPI app
app = FastAPI(title="ML Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routers
app.include_router(predict.router)
app.include_router(health.router)
app.include_router(append.router)