# rabbitmq_client.py
import asyncio
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
import aio_pika
import aio_pika.abc
import structlog

logger = structlog.get_logger()


@dataclass
class RabbitConfig:
    """Configuration settings for RabbitMQ."""

    user: str
    password: str
    host: str
    port: int
    heartbeat: int = 60
    max_retries: int = 5
    retry_delay_seconds: int = 2


@dataclass
class RabbitBinding:
    """Defines exchange, queue, and routing key bindings."""

    exchange: str
    queue: str
    routing_key: str


class AsyncRabbitMQClient:
    """Asynchronous RabbitMQ client for publishing and consuming messages."""

    def __init__(
        self, config: RabbitConfig, binding: RabbitBinding, declare: bool
    ) -> None:
        self._config = config
        self._binding = binding
        self._declare = declare
        self._connection: Optional[aio_pika.abc.AbstractRobustConnection] = None
        self._channel: Optional[aio_pika.abc.AbstractChannel] = None
        self._lock = asyncio.Lock()  # Prevents concurrent connection attempts

    async def _connect(self) -> None:
        """Establishes a connection to RabbitMQ with automatic retries."""
        async with self._lock:
            # If already connected and not closed, do nothing.
            if self._connection and not self._connection.is_closed:
                return

            retries = 0
            while retries < self._config.max_retries:
                try:
                    self._connection = await aio_pika.connect_robust(
                        f"amqp://{self._config.user}:{self._config.password}"
                        f"@{self._config.host}:{self._config.port}/",
                        heartbeat=self._config.heartbeat,
                    )
                    self._channel = await self._connection.channel()
                    await self._channel.set_qos(prefetch_count=1)

                    if self._declare:
                        await self._declare_bindings()

                    logger.info(
                        "Connected to RabbitMQ",
                        extra={"host": self._config.host, "port": self._config.port},
                    )
                    return
                except (
                    aio_pika.exceptions.AMQPConnectionError,
                    aio_pika.exceptions.AMQPChannelError,
                ) as e:
                    logger.warning(
                        "RabbitMQ connection error. Retrying...",
                        extra={"attempt": retries + 1, "error": str(e)},
                    )
                    await asyncio.sleep(self._config.retry_delay_seconds)
                    retries += 1

            logger.error("Failed to connect to RabbitMQ after multiple attempts")
            raise RuntimeError("RabbitMQ connection error")

    async def _declare_bindings(self) -> None:
        """Declares exchange and queue, then binds them together."""
        if self._channel is None:
            logger.error(
                "RabbitMQ channel is not initialized before declaring bindings"
            )
            raise RuntimeError("RabbitMQ channel is not initialized")

        try:
            exchange, queue = await asyncio.gather(
                self._channel.declare_exchange(self._binding.exchange, durable=True),
                self._channel.declare_queue(self._binding.queue, durable=True),
            )
            await queue.bind(exchange=exchange, routing_key=self._binding.routing_key)
            logger.info(
                "Exchange and Queue declared & bound",
                extra={
                    "exchange": self._binding.exchange,
                    "queue": self._binding.queue,
                },
            )
        except Exception as e:
            logger.error("Failed to declare exchange/queue", extra={"error": str(e)})
            raise

    async def _close_connection(self) -> None:
        """Closes RabbitMQ connection and channel safely."""
        try:
            if self._channel and not self._channel.is_closed:
                await self._channel.close()
                logger.info("RabbitMQ channel closed")
            if self._connection and not self._connection.is_closed:
                await self._connection.close()
                logger.info("RabbitMQ connection closed")
        except Exception as e:
            logger.error("Error closing RabbitMQ connection", extra={"error": str(e)})
        finally:
            self._channel = None
            self._connection = None

    async def publish_message(self, message: str) -> None:
        """
        Publishes a message to RabbitMQ. If a connection or channel error occurs,
        the client reconnects and retries sending the message.
        """
        while True:
            # Ensure we have an open channel.
            if not self._channel or self._channel.is_closed:
                await self._connect()

            try:
                exchange = await self._channel.get_exchange(self._binding.exchange)  # type: ignore
                await exchange.publish(
                    aio_pika.Message(body=message.encode()),
                    routing_key=self._binding.routing_key,
                )
                logger.info("Message published", extra={"message": message})
                return
            except (
                aio_pika.exceptions.AMQPConnectionError,
                aio_pika.exceptions.AMQPChannelError,
            ) as e:
                logger.warning(
                    "Publish failed due to connection issues. Reconnecting...",
                    extra={"error": str(e)},
                )
                await self._close_connection()
                await asyncio.sleep(self._config.retry_delay_seconds)
            except Exception as e:
                logger.error("Failed to publish message", extra={"error": str(e)})
                raise

    async def listen_forever(
        self, callback: Callable[[str], Awaitable[None]], retry_delay: int = 5
    ) -> None:
        """
        Listens indefinitely for messages on the queue, calling `callback(message_body)`
        for each message. On connection issues, the consumer re‑connects automatically.
        """
        while True:
            if not self._channel or self._channel.is_closed:
                await self._connect()

            try:
                queue = await self._channel.get_queue(self._binding.queue)  # type: ignore
                logger.info(
                    "Listening for messages", extra={"queue": self._binding.queue}
                )
                async for msg in queue.iterator():
                    async with msg.process():
                        decoded = msg.body.decode()
                        logger.info("Message received", extra={"message": decoded})
                        await callback(decoded)
            except asyncio.CancelledError:
                logger.info("RabbitMQ listener cancelled")
                break
            except (
                aio_pika.exceptions.AMQPConnectionError,
                aio_pika.exceptions.AMQPChannelError,
            ) as e:
                logger.warning(
                    "Connection lost during consumption. Reconnecting...",
                    extra={"error": str(e)},
                )
                await self._close_connection()
                await asyncio.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error in listen_forever: {e}")
                await asyncio.sleep(retry_delay)


def setup_rabbitmq(
    user: str,
    password: str,
    host: str,
    port: int,
    heartbeat: int,
    topic: str,
    declare: bool,
    exchange: str = "fl",
) -> AsyncRabbitMQClient:
    """
    Initializes and returns a RabbitMQ client.
    """
    config = RabbitConfig(user, password, host, port, heartbeat)
    binding = RabbitBinding(exchange=exchange, queue=topic, routing_key=topic)
    return AsyncRabbitMQClient(config, binding, declare)
