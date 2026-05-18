import pika
import json
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    """Consumer for RabbitMQ that consumes messages from request_queue and jsonifies results."""

    def __init__(self, config: dict):
        """
        Initialize RabbitMQ consumer.
        
        Args:
            config (dict): Configuration dictionary containing RabbitMQ settings
        """
        self.config = config["RabbitMQ"]
        self.connection = None
        self.channel = None
        self.request_queue = self.config["request_queue"]

    def connect(self) -> None:
        """Establish connection to RabbitMQ server."""
        try:
            credentials = pika.PlainCredentials(
                self.config["user_name"],
                self.config["password"]
            )
            parameters = pika.ConnectionParameters(
                host=self.config["host_name"],
                port=self.config["port"],
                credentials=credentials,
                heartbeat=600
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare queue
            self.channel.queue_declare(
                queue=self.request_queue,
                durable=True
            )
            
            logger.info(f"Connected to RabbitMQ and declared queue: {self.request_queue}")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def consume(self, callback: Callable[[Any], Any]) -> None:
        """
        Start consuming messages from the request_queue.
        
        Args:
            callback (Callable): Callback function to process each message
        """
        if not self.channel:
            self.connect()

        def message_callback(ch, method, properties, body):
            """Internal callback to handle incoming messages."""
            try:
                # Decode message body
                message_str = body.decode('utf-8')
                
                # Jsonify the message
                try:
                    # Try to parse if it's already JSON
                    message_data = json.loads(message_str)
                except json.JSONDecodeError:
                    # If not JSON, wrap it as JSON object
                    message_data = {"message": message_str}
                
                logger.info(f"Consumed message: {json.dumps(message_data)}")
                
                # Execute the callback with the jsonified data
                callback(message_data)
                
                # Acknowledge the message
                ch.basic_ack(delivery_tag=method.delivery_tag)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # Negative acknowledge and requeue
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        # Set up consumer with prefetch
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.request_queue,
            on_message_callback=message_callback
        )

        logger.info(f"Starting to consume messages from queue: {self.request_queue}")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
            self.close()

    def close(self) -> None:
        """Close the RabbitMQ connection."""
        if self.connection:
            self.connection.close()
            logger.info("RabbitMQ connection closed")


def create_consumer(config: dict) -> RabbitMQConsumer:
    """
    Factory function to create a RabbitMQ consumer instance.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        RabbitMQConsumer: Consumer instance
    """
    return RabbitMQConsumer(config)
