# api/models.py
import mlflow
import logging

logger = logging.getLogger(__name__)


def load_model(model_uri: str):
    """
    Load a Keras model from MLflow.

    Args:
        model_uri (str): MLflow model URI (e.g., 'runs:/<run-id>/model' or 'models:/<model-name>/latest')

    Returns:
        Model: Loaded Keras model
    """
    try:
        model = mlflow.keras.load_model(model_uri)
        logger.info(f"Successfully loaded model from {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_uri}: {str(e)}")
        raise