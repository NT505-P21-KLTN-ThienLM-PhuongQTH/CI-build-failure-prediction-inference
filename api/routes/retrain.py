import logging
import os
from fastapi import APIRouter
from dotenv import load_dotenv
import pika
import json

logger = logging.getLogger("api.retrain")
load_dotenv()

router = APIRouter()
@router.post("/retrain")
async def retrain():
    await trigger_training_message()

async def trigger_training_message():
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

    message = {"action": "trigger_training"}
    channel.basic_publish(
        exchange='',
        routing_key=queue_name,
        body=json.dumps(message),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    (print("Training trigger sent to queue"))
    connection.close()