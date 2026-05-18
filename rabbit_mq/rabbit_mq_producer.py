import pika
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MessageProducer:
    """Producer for RabbitMQ that sends response data to the response_queue."""

    def __init__(self, config: dict):
        """
        Initialize RabbitMQ message producer.

        Args:
            config (dict): Configuration dictionary containing RabbitMQ settings
        """
        self.config = config["RabbitMQ"]
        self.connection = None
        self.channel = None
        self.response_queue = self.config["response_queue"]
        self.exchange_name = self.config["exchange_name"]
        self.routing_key = self.config["response_routing_key"]

    def connect(self) -> None:
        """Establish connection to RabbitMQ server."""
        try:
            credentials = pika.PlainCredentials(
                self.config["user_name"], self.config["password"]
            )
            parameters = pika.ConnectionParameters(
                host=self.config["host_name"],
                port=self.config["port"],
                credentials=credentials,
                heartbeat=600,
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Declare exchange
            self.channel.exchange_declare(
                exchange=self.exchange_name, exchange_type="direct",
            )

            # Declare queue
            self.channel.queue_declare(queue=self.response_queue, durable=True)

            # Bind queue to exchange
            self.channel.queue_bind(
                exchange=self.exchange_name,
                queue=self.response_queue,
                routing_key=self.routing_key,
            )

            logger.info(
                f"Connected to RabbitMQ and declared queue: {self.response_queue}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def publish_message(
        self, response_data: Any, correlation_id: Optional[str] = None
    ) -> bool:
        """
        Publish response data to the response_queue.

        Args:
            response_data (Any): The response data to be published (dict, str, etc.)
            correlation_id (Optional[str]): Optional correlation ID for request-response tracking

        Returns:
            bool: True if message was published successfully, False otherwise
        """
        try:
            if not self.channel:
                self.connect()

            # Convert response_data to JSON if it's not already a string
            if isinstance(response_data, str):
                message_body = response_data
            else:
                message_body = json.dumps(response_data)

            # Prepare message properties
            properties = pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type="application/json",
            )

            # Add correlation ID if provided
            if correlation_id:
                properties.correlation_id = correlation_id

            # Publish message
            self.channel.basic_publish(
                exchange=self.exchange_name,
                routing_key=self.routing_key,
                body=message_body,
                properties=properties,
            )

            logger.info(
                f"Message published to {self.response_queue}: {message_body[:100]}..."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False

    def close(self) -> None:
        """Close the RabbitMQ connection."""
        if self.connection:
            self.connection.close()
            logger.info("RabbitMQ connection closed")


def create_producer(config: dict) -> MessageProducer:
    """
    Factory function to create a RabbitMQ message producer instance.

    Args:
        config (dict): Configuration dictionary

    Returns:
        MessageProducer: Producer instance
    """
    return MessageProducer(config)
