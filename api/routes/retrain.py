import logging
import os
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
import pika
import json
from api.schemas.retrain import RetrainRequest

logger = logging.getLogger("api.retrain")
load_dotenv()

router = APIRouter()
@router.post("/retrain")
async def retrain(request: RetrainRequest):
    """
    Trigger a retraining process by sending a message to RabbitMQ.
    Args:
        request (RetrainRequest): Request body containing the model_name ('Stacked-LSTM', 'Stacked-BiLSTM', or'Padding').
    Returns:
        dict: Confirmation message if successful.
    """
    model_name = request.model_name
    await trigger_training_message(model_name)
    return {"message": f"Training triggered successfully for model_name: {model_name}"}

async def trigger_training_message(model_name: str):
    """
    Send a training message to RabbitMQ with the specified model_name.
    Args:
        model_name (str): The type of model to train ('Stacked-LSTM', 'Stacked-BiLSTM', or 'Padding').
    """
    if model_name not in ["Stacked-LSTM", "Stacked-BiLSTM", "Padding", "ConvLSTM"]:
        raise HTTPException(status_code=400, detail="Invalid model_name. Must be 'Stacked-LSTM', 'Stacked-BiLSTM', or 'Padding'.")
    try:
        rabbitmq_user = os.getenv("RABBITMQ_USER")
        rabbitmq_password = os.getenv("RABBITMQ_PASSWORD")
        rabbitmq_host = os.getenv("RABBITMQ_HOST")
        rabbitmq_vhost = os.getenv("RABBITMQ_VHOST")

        # Kết nối tới RabbitMQ
        credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
        parameters = pika.ConnectionParameters(
            host=rabbitmq_host,
            port=5672,
            virtual_host=rabbitmq_vhost,
            credentials=credentials
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        queue_name = 'training_queue'
        channel.queue_declare(queue=queue_name, durable=True)

        # Message bao gồm model_name để consumer xử lý
        message = {"model_name": model_name}
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2)  # Message bền bỉ
        )
        logger.info(f"Training trigger sent to queue with model_name: {model_name}")
        connection.close()

    except Exception as e:
        logger.error(f"Error sending training trigger: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))