# api/main.py
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from src.api.schemas import PredictData
from src.data.main import preprocess_data, split_into_sequences
import mlflow
import os
import pandas as pd
import logging.config
from dotenv import load_dotenv
import yaml

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

load_dotenv()
logging.getLogger().setLevel(logging.WARNING)

log_dir = os.path.join(os.path.dirname(__file__), '../../logs')
os.makedirs(log_dir, exist_ok=True)

with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("config/logging.yaml", "r") as f:
    logging_config = yaml.safe_load(f)
    logging.config.dictConfig(logging_config)

logger = logging.getLogger(__name__)

app = FastAPI(title="ML Prediction API", version="1.0.0")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

@app.post("/predict")
async def predict(data: PredictData):
    """
    Predict using the loaded LSTM model.
    Expects input data as a dictionary of features.
    """
    try:
        client = MlflowClient()
        model_name = data.model
        model_version = data.version
        model_uri = f"models:/{model_name}/{model_version}"

        # Load the model from MLflow
        model = mlflow.keras.load_model(model_uri)
        logger.info(f"Loaded model from {model_uri}")

        # Load params from MLflow run
        model_version_info = client.get_model_version(name=model_name, version=model_version)
        run_id = model_version_info.run_id
        run = client.get_run(run_id)
        params = run.data.params
        time_step = int(params.get("best_time_step"))

        # Preprocess input data
        df = pd.DataFrame(data.ci_builds)
        logger.debug(f"Input DataFrame columns: {list(df.columns)}")
        X = preprocess_data(is_training=False, input=df)
        X_sequences = split_into_sequences(X, time_step)

        # Predict
        prediction = model.predict(X_sequences)
        logger.info(f"Prediction successful for {len(data.ci_builds)} input builds.")
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}