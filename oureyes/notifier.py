import json
import os
from confluent_kafka import Producer
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Load Kafka server from .env
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")

_producer = None  # Singleton producer

def delivery_report(err, msg):
    """Delivery callback for Kafka messages."""
    if err is not None:
        print(f"❌ Message delivery failed: {err}")
    else:
        print(f"✅ Message delivered to {msg.topic()} [{msg.partition()}]")

def get_producer():
    """
    Return the singleton Kafka producer. Create if it doesn't exist.
    """
    global _producer
    if _producer is None:
        try:
            _producer = Producer({
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "acks": "all",
                "retries": 3,
                "enable.idempotence": True
            })
            print(f"Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS} ✅")
        except Exception as e:
            print(f"❌ Failed to connect to Kafka: {e}")
            raise
    return _producer

def notify_server(topic: str, data: dict):
    """
    Send a notification to Kafka using the singleton producer.

    Args:
        topic (str): Kafka topic name
        data (dict): Python dict with alert/message
    """
    producer = get_producer()
    try:
        producer.produce(
            topic,
            value=json.dumps(data).encode("utf-8"),
            callback=delivery_report
        )
        producer.poll(0)  # Serve delivery callbacks
    except Exception as e:
        print(f"❌ Error sending to Kafka: {e}")

def close_producer():
    """
    Flush remaining messages and close the producer.
    """
    global _producer
    if _producer is not None:
        _producer.flush()
        print("Kafka producer closed ✅")
        _producer = None
