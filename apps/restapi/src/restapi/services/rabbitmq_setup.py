from interfaces.rabbitmq_client import setup_rabbitmq, AsyncRabbitMQClient
from restapi import config
import structlog

logger = structlog.get_logger()


def initialize_rabbitmq_connections() -> dict[str, AsyncRabbitMQClient]:
    """
    Initializes RabbitMQ connections for dispatch and status queues.

    Returns:
        dict[str, AsyncRabbitMQClient]: A dictionary with RabbitMQ clients.
    """
    connections = {}
    try:
        connections["rabbitmq_dispatch"] = setup_rabbitmq(
            config.RABBIT_USERNAME,
            config.RABBIT_PASSWORD,
            config.RABBIT_HOST,
            int(config.RABBIT_PORT),
            30,
            "dispatch",
            True,
        )
        logger.info("RabbitMQ dispatch connection initialized")

        connections["rabbitmq_status"] = setup_rabbitmq(
            config.RABBIT_USERNAME,
            config.RABBIT_PASSWORD,
            config.RABBIT_HOST,
            int(config.RABBIT_PORT),
            30,
            "status",
            True,
        )
        logger.info("RabbitMQ status connection initialized")

    except Exception as e:
        logger.error("Failed to initialize RabbitMQ connections", error=str(e))
        raise RuntimeError("RabbitMQ connection failed") from e

    return connections
