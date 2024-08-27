import atexit
import pika
from typing import Callable, cast

import pika.channel

from .utils import MutableBoolean, ensure_bool

# Global variables for connection and channel
_connection = cast(pika.BlockingConnection, None)
_channel = cast(pika.channel.Channel, None)
_queue_name = cast(str, None)
_is_configured = MutableBoolean(False)


def configure(
    user: str = "guest",
    password: str = "guest",
    host: str = "localhost",
    port: int = 5672,
    queue_name: str = "fml-engine",
) -> None:
    global _connection, _channel, _queue_name, _is_configured
    if _is_configured:
        raise RuntimeError("RabbitMQ client is already configured.")

    _queue_name = queue_name
    credentials = pika.PlainCredentials(user, password)
    parameters = pika.ConnectionParameters(
        host=host, port=port, credentials=credentials, heartbeat=10000
    )

    _connection = pika.BlockingConnection(parameters)
    _channel = _connection.channel()
    _channel.queue_declare(queue=_queue_name, durable=True)
    _is_configured.value = True


def _close_connection() -> None:
    global _connection, _channel
    if _channel and _channel.is_open:
        _channel.close()
    if _connection and _connection.is_open:
        _connection.close()


@ensure_bool(_is_configured)
def publish_message(message: str) -> None:
    _channel.basic_publish(exchange="", routing_key=_queue_name, body=message)


@ensure_bool(_is_configured)
def listen(callback: Callable) -> None:
    _channel.basic_consume(
        queue=_queue_name, on_message_callback=callback, auto_ack=True
    )
    try:
        _channel.start_consuming()
    except KeyboardInterrupt:
        _channel.stop_consuming()


atexit.register(_close_connection)
