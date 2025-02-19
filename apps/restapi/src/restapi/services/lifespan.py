from contextlib import asynccontextmanager
from typing import AsyncGenerator
import structlog
from fastapi import FastAPI
from restapi.db import create_db_and_tables
from restapi.services.rabbitmq_setup import initialize_rabbitmq_connections
from interfaces.rabbitmq_client import AsyncRabbitMQClient
from restapi import config

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manages application startup and shutdown lifecycle."""
    logger.info("Starting FastAPI application...")

    try:
        await _setup_rabbitmq()
        await create_db_and_tables()

        yield

    finally:
        logger.info("Shutting down FastAPI application...")
        await _shutdown_rabbitmq_connections()


async def _setup_rabbitmq() -> None:
    """Initializes RabbitMQ connections and stores them in config."""
    rabbitmq_connections = initialize_rabbitmq_connections()
    config.obj.update(rabbitmq_connections)


async def _shutdown_rabbitmq_connections() -> None:
    """Closes all active RabbitMQ connections."""
    for key, rabbit_client in config.obj.items():
        if isinstance(rabbit_client, AsyncRabbitMQClient):
            await rabbit_client._close_connection()  # ✅ Use proper close method
            logger.info(f"Closed RabbitMQ connection: {key}")
